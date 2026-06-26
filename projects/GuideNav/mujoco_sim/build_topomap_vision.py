#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


GUIDENAV_ROOT = Path(__file__).resolve().parents[1]
ODOM_FIELDS = [
    "timestamp",
    "pos_x",
    "pos_y",
    "pos_z",
    "ori_x",
    "ori_y",
    "ori_z",
    "ori_w",
    "lin_vel_x",
    "lin_vel_y",
    "lin_vel_z",
    "ang_vel_x",
    "ang_vel_y",
    "ang_vel_z",
]


@dataclass
class AlignedFrame:
    frame_index: int
    rgb_path: Path
    rgb_timestamp: float
    odom_timestamp: float
    odom_row: dict[str, str]
    depth_path: Path | None = None


@dataclass
class SelectedFrame:
    topo_index: int
    frame: AlignedFrame
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a GuideNav topomap from MuJoCo teaching data with visual "
            "embedding keyframe selection, matching the HRI'26 GuideNav pipeline "
            "more closely than odometry-only thresholding."
        )
    )
    parser.add_argument("input_dir", type=Path, help="raw teach run containing color/ and odom.csv")
    parser.add_argument("output_dir", type=Path, help="topomap output directory")
    parser.add_argument("--image-subdir", default="color")
    parser.add_argument("--depth-subdir", default="depth")
    parser.add_argument("--max-time-diff", type=float, default=0.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--torch-hub-dir", type=Path, default=GUIDENAV_ROOT / "model_weights" / "torch_hub_checkpoints")
    parser.add_argument("--feature-model", choices=("auto", "dinov3", "dinov2"), default="auto")
    parser.add_argument("--dinov3-repo", default=os.environ.get("DINOV3_REPO"))
    parser.add_argument("--dinov3-model", default="dinov3_vitl16")
    parser.add_argument("--dinov3-weights", type=Path, default=None)
    parser.add_argument("--dinov2-model", default="dinov2_vitl14")
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--force-frame-gap", type=int, default=50)
    parser.add_argument("--recent-keyframes", type=int, default=5)
    parser.add_argument("--history-size", type=int, default=20)
    parser.add_argument("--early-sim-threshold", type=float, default=0.95)
    parser.add_argument("--middle-sim-threshold", type=float, default=0.93)
    parser.add_argument("--late-sim-threshold", type=float, default=0.91)
    parser.add_argument("--early-diversity-threshold", type=float, default=0.92)
    parser.add_argument("--middle-diversity-threshold", type=float, default=0.90)
    parser.add_argument("--late-diversity-threshold", type=float, default=0.88)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def load_feature_model(args: argparse.Namespace, device: torch.device):
    args.torch_hub_dir.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(args.torch_hub_dir))

    if args.feature_model in ("auto", "dinov3") and args.dinov3_repo:
        repo = str(Path(args.dinov3_repo).expanduser())
        source = "local" if Path(repo).exists() else "github"
        kwargs = {"source": source}
        if args.dinov3_weights is not None:
            kwargs["weights"] = str(args.dinov3_weights.expanduser())
        print(f"Loading DINOv3 feature model: repo={repo} model={args.dinov3_model}")
        model = torch.hub.load(repo, args.dinov3_model, **kwargs)
        return model.to(device).eval(), "dinov3"

    if args.feature_model == "dinov3":
        raise RuntimeError(
            "DINOv3 was requested, but --dinov3-repo or DINOV3_REPO was not set. "
            "Pass a local DINOv3 checkout/torch.hub repo to match the paper exactly."
        )

    print(
        "DINOv3 repo was not provided; using DINOv2 fallback for the same "
        "embedding-based selection structure."
    )
    model = torch.hub.load("facebookresearch/dinov2", args.dinov2_model)
    return model.to(device).eval(), "dinov2"


def build_preprocess():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def output_to_descriptor(output) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("x_norm_clstoken", "global_descriptor", "features", "feat"):
            if key in output:
                output = output[key]
                break
        else:
            output = next(iter(output.values()))
    elif isinstance(output, (tuple, list)):
        output = output[0]

    if output.ndim == 3:
        output = output[:, 0, :]
    elif output.ndim > 3:
        output = output.flatten(1)
    return output


def extract_descriptor(path: Path, model, preprocess, device: torch.device) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    descriptor = output_to_descriptor(output).float()
    descriptor = descriptor / descriptor.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return descriptor.cpu().numpy().squeeze()


def numeric_pngs(path: Path) -> list[Path]:
    return sorted(path.glob("*.png"), key=lambda p: float(p.stem))


def load_odom(path: Path) -> dict[float, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {float(row["timestamp"]): row for row in reader}


def closest_key(target: float, candidates: list[float], max_diff: float) -> float | None:
    if not candidates:
        return None
    best = min(candidates, key=lambda value: abs(value - target))
    return best if abs(best - target) <= max_diff else None


def align_frames(args: argparse.Namespace) -> list[AlignedFrame]:
    color_dir = args.input_dir / args.image_subdir
    depth_dir = args.input_dir / args.depth_subdir
    odom_path = args.input_dir / "odom.csv"
    if not color_dir.exists():
        raise FileNotFoundError(f"Color image directory not found: {color_dir}")
    if not odom_path.exists():
        raise FileNotFoundError(f"Odometry file not found: {odom_path}")

    odom = load_odom(odom_path)
    odom_times = sorted(odom)
    depth_paths = {float(path.stem): path for path in numeric_pngs(depth_dir)} if depth_dir.exists() else {}
    depth_times = sorted(depth_paths)
    aligned: list[AlignedFrame] = []

    for frame_index, rgb_path in enumerate(numeric_pngs(color_dir)):
        rgb_time = float(rgb_path.stem)
        odom_time = closest_key(rgb_time, odom_times, args.max_time_diff)
        if odom_time is None:
            continue
        depth_time = closest_key(rgb_time, depth_times, args.max_time_diff)
        aligned.append(
            AlignedFrame(
                frame_index=frame_index,
                rgb_path=rgb_path,
                rgb_timestamp=rgb_time,
                odom_timestamp=odom_time,
                odom_row=odom[odom_time],
                depth_path=depth_paths[depth_time] if depth_time is not None else None,
            )
        )

    if len(aligned) < 2:
        raise RuntimeError(f"Need at least two RGB/odom-aligned frames; got {len(aligned)}")
    return aligned


class AdaptiveVisionSelector:
    def __init__(self, args: argparse.Namespace, total_frames: int) -> None:
        self.args = args
        self.total_frames = max(total_frames, 1)
        self.keyframe_features: deque[np.ndarray] = deque(maxlen=args.history_size)
        self.recent_similarities: deque[float] = deque(maxlen=50)
        self.last_selected_frame_index = -1
        self.selected_count = 0

    def thresholds(self, progress: float) -> tuple[float, float]:
        if progress < 0.33:
            sim = self.args.early_sim_threshold
            diversity = self.args.early_diversity_threshold
        elif progress < 0.66:
            sim = self.args.middle_sim_threshold
            diversity = self.args.middle_diversity_threshold
        else:
            sim = self.args.late_sim_threshold
            diversity = self.args.late_diversity_threshold

        if len(self.recent_similarities) >= 10:
            sim_std = float(np.std(self.recent_similarities))
            sim = max(0.85, min(0.98, sim - 2.0 * sim_std))
        return sim, diversity

    def should_select(self, feature: np.ndarray, frame: AlignedFrame) -> tuple[bool, str]:
        if not self.keyframe_features:
            return True, "first"

        gap = frame.frame_index - self.last_selected_frame_index
        if gap < self.args.min_frame_gap:
            return False, "min_frame_gap"
        if gap >= self.args.force_frame_gap:
            return True, "force_frame_gap"

        last_similarity = float(np.dot(self.keyframe_features[-1], feature))
        self.recent_similarities.append(last_similarity)
        recent = list(self.keyframe_features)[-self.args.recent_keyframes :]
        min_recent_similarity = min(float(np.dot(previous, feature)) for previous in recent)
        progress = frame.frame_index / max(self.total_frames - 1, 1)
        sim_threshold, diversity_threshold = self.thresholds(progress)

        if last_similarity < sim_threshold:
            return True, f"appearance_change:{last_similarity:.3f}"
        if min_recent_similarity < diversity_threshold:
            return True, f"recent_diversity:{min_recent_similarity:.3f}"
        return False, f"similar:{last_similarity:.3f}"

    def mark_selected(self, feature: np.ndarray, frame: AlignedFrame) -> None:
        self.keyframe_features.append(feature)
        self.last_selected_frame_index = frame.frame_index
        self.selected_count += 1


def select_keyframes(args: argparse.Namespace, frames: list[AlignedFrame]) -> tuple[list[SelectedFrame], str]:
    device = choose_device(args.device)
    print(f"Using device: {device}")
    model, model_name = load_feature_model(args, device)
    preprocess = build_preprocess()
    selector = AdaptiveVisionSelector(args, len(frames))
    selected: list[SelectedFrame] = []
    last_feature: np.ndarray | None = None

    for i, frame in enumerate(frames):
        feature = extract_descriptor(frame.rgb_path, model, preprocess, device)
        last_feature = feature
        should_select, reason = selector.should_select(feature, frame)
        if should_select:
            selector.mark_selected(feature, frame)
            selected.append(SelectedFrame(len(selected), frame, reason))
            print(
                f"Keyframe {len(selected)-1}: frame={frame.frame_index} "
                f"time={frame.rgb_timestamp:.3f} reason={reason}"
            )

        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(frames)} frames; selected={len(selected)}")

    if selected[-1].frame != frames[-1] and last_feature is not None:
        selector.mark_selected(last_feature, frames[-1])
        selected.append(SelectedFrame(len(selected), frames[-1], "last"))
        print(f"Keyframe {len(selected)-1}: frame={frames[-1].frame_index} reason=last")

    return selected, model_name


def reset_outputs(output_dir: Path) -> tuple[Path, Path, Path]:
    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    topo_dir = output_dir / "topo"
    for path in (color_dir, depth_dir, topo_dir):
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    for path in (output_dir / "odom.csv", output_dir / "selection_log.csv"):
        if path.exists():
            path.unlink()
    return color_dir, depth_dir, topo_dir


def write_outputs(args: argparse.Namespace, selected: list[SelectedFrame], model_name: str) -> None:
    color_dir, depth_dir, topo_dir = reset_outputs(args.output_dir)
    odom_path = args.output_dir / "odom.csv"
    selection_log_path = args.output_dir / "selection_log.csv"

    with odom_path.open("w", newline="", encoding="utf-8") as odom_file, selection_log_path.open(
        "w", newline="", encoding="utf-8"
    ) as log_file:
        odom_writer = csv.DictWriter(odom_file, fieldnames=ODOM_FIELDS)
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "topo_index",
                "frame_index",
                "rgb_timestamp",
                "odom_timestamp",
                "reason",
                "feature_model",
            ],
        )
        odom_writer.writeheader()
        log_writer.writeheader()

        for item in selected:
            frame = item.frame
            rgb_dst = color_dir / frame.rgb_path.name
            topo_dst = topo_dir / f"{item.topo_index}.png"
            shutil.copy2(frame.rgb_path, rgb_dst)
            shutil.copy2(frame.rgb_path, topo_dst)
            if frame.depth_path is not None:
                shutil.copy2(frame.depth_path, depth_dir / frame.depth_path.name)

            odom_writer.writerow({field: frame.odom_row[field] for field in ODOM_FIELDS})
            log_writer.writerow(
                {
                    "topo_index": item.topo_index,
                    "frame_index": frame.frame_index,
                    "rgb_timestamp": f"{frame.rgb_timestamp:.9f}",
                    "odom_timestamp": f"{frame.odom_timestamp:.9f}",
                    "reason": item.reason,
                    "feature_model": model_name,
                }
            )

    print("\nVision topomap complete")
    print(f"  selected keyframes: {len(selected)}")
    print(f"  output topomap    : {args.output_dir}")
    print(f"  keyframe images   : {topo_dir}")
    print(f"  odometry          : {odom_path}")
    print(f"  selection log     : {selection_log_path}")
    if not any(depth_dir.iterdir()):
        print("  depth             : none copied; this topomap is RGB/odom only")


def main() -> None:
    args = parse_args()
    frames = align_frames(args)
    print(f"Aligned RGB/odom frames: {len(frames)}")
    selected, model_name = select_keyframes(args, frames)
    write_outputs(args, selected, model_name)


if __name__ == "__main__":
    main()
