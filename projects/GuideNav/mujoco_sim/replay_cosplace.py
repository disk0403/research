#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from common import (
    APPEARANCE_PRESETS,
    DEFAULT_SCENE_CONFIG,
    GUIDENAV_ROOT,
    MODEL_DIR,
    POLICY_DIR,
    MujocoGo2Runtime,
    Pose2D,
    contact_with_named_obstacle,
    final_uprightness,
    setup_display,
    sleep_for_realtime,
)
import mujoco
from guidenav.match_to_control import control, feature_match
from guidenav.place_recognition.bayesian_querier import PlaceRecognitionTopologicalFilter
from guidenav.place_recognition.feature_extractor import FeatureExtractor
from guidenav.place_recognition.sliding_window_querier import PlaceRecognitionSlidingWindowFilter
from guidenav.utils import get_image_transform, read_image


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-guidenav")
os.environ.setdefault("HF_HOME", str(GUIDENAV_ROOT / "model_weights" / "huggingface"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GUI autonomous repeat using official GuideNav place-recognition "
            "classes with a CosPlace topomap database."
        )
    )
    parser.add_argument("--topomap-dir", type=Path, default=GUIDENAV_ROOT / "data" / "mujoco_teach" / "topomap")
    parser.add_argument("--topomap-image-subdir", default="topo")
    parser.add_argument("--model-config-path", type=Path, default=GUIDENAV_ROOT / "config" / "models.yaml")
    parser.add_argument("--model-weight-dir", type=Path, default=GUIDENAV_ROOT / "model_weights")
    parser.add_argument("--pr-model", default="cosplace_hub")
    parser.add_argument("--img-size", type=int, nargs=2, default=(85, 64), metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--filter-mode", choices=("bayesian", "sliding_window"), default="bayesian")
    parser.add_argument("--filter-delta", type=int, default=10)
    parser.add_argument("--transition-model-window-lower", type=int, default=-1)
    parser.add_argument("--transition-model-window-upper", type=int, default=3)
    parser.add_argument("--window-radius", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--vpr-fps", type=float, default=2.0)
    parser.add_argument("--feature-matching", choices=("reloc3r",), default="reloc3r")
    parser.add_argument("--display", default=":1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=180)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--scene-preset", choices=sorted(APPEARANCE_PRESETS), default="sunny_morning")
    parser.add_argument("--cycle-appearance", action="store_true")
    parser.add_argument("--cycle-period", type=float, default=18.0)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--reset-base-height", type=float, default=0.25)
    parser.add_argument("--stance-crouch", type=float, default=0.08)
    parser.add_argument("--normal-speed", type=float, default=0.32)
    parser.add_argument("--yaw-speed", type=float, default=0.48)
    parser.add_argument("--yaw-safety-limit", type=float, default=1.0)
    parser.add_argument("--command-smoothing", type=float, default=10.0)
    parser.add_argument("--max-vx", type=float, default=0.32)
    parser.add_argument("--max-vy", type=float, default=0.16)
    parser.add_argument("--max-yaw", type=float, default=0.55)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--waypoint-radius", type=float, default=0.35)
    parser.add_argument("--goal-radius", type=float, default=0.40)
    parser.add_argument("--fall-height", type=float, default=0.16)
    parser.add_argument("--fall-uprightness", type=float, default=0.55)
    parser.add_argument("--stop-on-obstacle-contact", action="store_true")
    parser.add_argument("--print-interval", type=float, default=1.0)
    return parser.parse_args()


def quat_xyzw_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def load_topomap_poses(topomap_dir: Path) -> list[Pose2D]:
    odom_path = topomap_dir / "odom.csv"
    if not odom_path.exists():
        raise FileNotFoundError(f"Topomap odometry file not found: {odom_path}")

    poses: list[Pose2D] = []
    with odom_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yaw = quat_xyzw_to_yaw(
                float(row["ori_x"]),
                float(row["ori_y"]),
                float(row["ori_z"]),
                float(row["ori_w"]),
            )
            poses.append(Pose2D(float(row["pos_x"]), float(row["pos_y"]), yaw))

    if len(poses) < 2:
        raise ValueError(f"Need at least 2 topomap poses in {odom_path}; got {len(poses)}")
    return poses


def set_robot_planar_pose(runtime: MujocoGo2Runtime, pose: Pose2D, args: argparse.Namespace) -> None:
    runtime.policy.initialize_pose(
        runtime.data,
        base_height=args.reset_base_height,
        stance_crouch=args.stance_crouch,
    )
    runtime.data.qpos[0] = pose.x
    runtime.data.qpos[1] = pose.y
    runtime.data.qpos[2] = args.reset_base_height
    runtime.data.qpos[3:7] = np.array(
        [math.cos(0.5 * pose.yaw), 0.0, 0.0, math.sin(0.5 * pose.yaw)],
        dtype=np.float64,
    )
    runtime.data.qvel[:] = 0.0
    runtime.dynamic_actors.apply(runtime.data, float(runtime.data.time))
    runtime.command[:] = 0.0
    runtime.next_policy_time = float(runtime.data.time)
    mujoco.mj_forward(runtime.model, runtime.data)


def load_model_conf(args: argparse.Namespace) -> dict:
    with args.model_config_path.open("r", encoding="utf-8") as f:
        confs = yaml.safe_load(f)
    if args.pr_model not in confs:
        raise KeyError(f"Unknown place-recognition model '{args.pr_model}' in {args.model_config_path}")

    conf = copy.deepcopy(confs[args.pr_model])
    model_conf = conf.setdefault("model", {})

    checkpoint_path = model_conf.get("checkpoint_path")
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = args.model_weight_dir / checkpoint_path
        model_conf["checkpoint_path"] = checkpoint_path

    hub_dir = model_conf.get("torch_hub_dir")
    if hub_dir is not None:
        hub_dir = Path(hub_dir).expanduser()
        if not hub_dir.is_absolute():
            hub_dir = GUIDENAV_ROOT / hub_dir
        model_conf["torch_hub_dir"] = str(hub_dir)

    return conf


def prepare_query_tensor(rgb: np.ndarray, transform, device: torch.device) -> torch.Tensor:
    # GuideNav's DB builder reads topomap images through OpenCV as BGR.
    # MuJoCo renders RGB, so convert to BGR before applying the same transform.
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    return transform(bgr.astype(np.uint8)).unsqueeze(0).to(device)


class VprNavigator:
    def __init__(self, args: argparse.Namespace, poses: list[Pose2D]) -> None:
        self.args = args
        self.poses = poses
        self.device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        self.topomap_img_dir = args.topomap_dir / args.topomap_image_subdir
        self.conf = load_model_conf(args)
        self.feature_path = self.topomap_img_dir / f"{self.conf['output']}.h5"
        if not self.feature_path.exists():
            raise FileNotFoundError(
                f"VPR database not found: {self.feature_path}\n"
                "Run mujoco_sim/build_vpr_db.py first."
            )

        self.transform = get_image_transform(list(args.img_size))
        self.topomap_images = self._load_topomap_images()
        self._validate_feature_db_size()
        self.extractor = FeatureExtractor(self.conf, self.device)
        if args.filter_mode == "bayesian":
            self.querier = PlaceRecognitionTopologicalFilter(
                self.extractor,
                self.feature_path,
                self.topomap_img_dir,
                delta=args.filter_delta,
                window_lower=args.transition_model_window_lower,
                window_upper=args.transition_model_window_upper,
            )
        else:
            self.querier = PlaceRecognitionSlidingWindowFilter(
                self.extractor,
                self.feature_path,
                self.topomap_img_dir,
            )
        self.initialized = False
        self.closest_index = 0
        self.score = 0.0
        self.target_index = min(len(poses) - 1, args.lookahead)

    def _load_topomap_images(self) -> list[np.ndarray]:
        images: list[np.ndarray] = []
        for path in sorted(self.topomap_img_dir.glob("*.png"), key=lambda p: int(p.stem)):
            images.append(read_image(path))
        if len(images) < 2:
            raise ValueError(f"Need at least 2 topomap images in {self.topomap_img_dir}; got {len(images)}")
        return images

    def _validate_feature_db_size(self) -> None:
        with h5py.File(self.feature_path, "r") as f:
            db_count = len(f.keys())
        if db_count != len(self.topomap_images):
            raise ValueError(
                "VPR DB and topomap images are out of sync:\n"
                f"  VPR DB entries : {db_count}\n"
                f"  topomap images : {len(self.topomap_images)}\n"
                "Rebuild with:\n"
                "  python3 mujoco_sim/run.py build"
            )

    def update(self, rgb: np.ndarray) -> tuple[int, int, float]:
        query = prepare_query_tensor(rgb, self.transform, self.device)
        if self.args.filter_mode == "bayesian":
            if not self.initialized:
                self.querier.initialize_model(query)
                self.initialized = True
            self.closest_index, self.score = self.querier.match(query)
        else:
            start = max(self.closest_index - self.args.window_radius, 0)
            end = min(self.closest_index + self.args.window_radius + 1, len(self.poses))
            self.closest_index = self.querier.match(query, start, end)
            self.score = 0.0

        self.target_index = min(len(self.poses) - 1, self.closest_index + max(self.args.lookahead, 0))
        return self.closest_index, self.target_index, self.score


def bgr_from_mujoco_rgb(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(rgb[:, :, ::-1])


def rgb_from_bgr(bgr: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(bgr[:, :, ::-1])


def command_from_relative_pose(
    current_rgb: np.ndarray,
    topomap_images: list[np.ndarray],
    target_index: int,
    relpose_model,
    relpose_img_reso: str,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, tuple[float, float, float, float, float]]:
    current_bgr = bgr_from_mujoco_rgb(current_rgb)
    target_index = min(max(target_index, 0), len(topomap_images) - 1)

    while True:
        x_rel, y_rel, yaw_rel = feature_match.matching_features_reloc3r_inv(
            current_bgr,
            topomap_images[target_index],
            relpose_model,
            relpose_img_reso,
        )
        x_rel = float(x_rel)
        y_rel = float(y_rel)
        yaw_rel = float(yaw_rel)
        if x_rel >= 0.0 or target_index >= len(topomap_images) - 1:
            break
        target_index += 1

    v, w = control.vtr_controller(
        x_rel,
        y_rel,
        yaw_rel,
        v_max=args.max_vx,
        w_max=args.max_yaw,
    )
    command = np.array([float(v), 0.0, float(w)], dtype=np.float32)
    return command, target_index, (x_rel, y_rel, yaw_rel, float(v), float(w))


def main() -> None:
    args = parse_args()
    setup_display(args.display, args.headless)

    poses = load_topomap_poses(args.topomap_dir)
    runtime = MujocoGo2Runtime(args, "GuideNav MuJoCo CosPlace VPR repeat", not args.headless)
    set_robot_planar_pose(runtime, poses[0], args)
    if runtime.viewer is not None:
        runtime.viewer.render_status(
            runtime.data,
            np.zeros(3, dtype=np.float32),
            "loading",
            "loading CosPlace and ReLoc3R models...",
            "GUI is ready; autonomous repeat starts after model loading.",
        )

    print("GUI/runtime ready. Loading CosPlace VPR model...")
    vpr = VprNavigator(args, poses)
    print(f"Loading feature-matching model: {args.feature_matching}")
    relpose_model, relpose_img_reso = feature_match.init_reloc3r()

    next_step_wall = time.perf_counter()
    next_render_wall = next_step_wall
    next_vpr_time = 0.0
    next_print_time = 0.0
    render_interval = 1.0 / args.render_fps
    vpr_interval = 1.0 / max(args.vpr_fps, 1e-6)
    closest_index = 0
    target_index = min(len(poses) - 1, args.lookahead)
    score = 0.0
    relpose = (0.0, 0.0, 0.0, 0.0, 0.0)
    target_command = np.zeros(3, dtype=np.float32)
    applied_command = np.zeros(3, dtype=np.float32)
    current_panel: np.ndarray | None = None
    matched_panel: np.ndarray | None = None
    subgoal_panel: np.ndarray | None = None
    stop_reason = "viewer_closed"

    print(f"Loaded {len(poses)} topomap poses from: {args.topomap_dir}")
    print(f"Using VPR database: {vpr.feature_path}")
    print("CosPlace VPR autonomous repeat started. GUI controls: Esc closes, mouse drags camera.")

    try:
        while True:
            now = time.perf_counter()
            if args.duration > 0.0 and runtime.data.time >= args.duration:
                stop_reason = "duration"
                break
            if runtime.viewer is not None and not runtime.viewer.is_running():
                stop_reason = "viewer_closed"
                break
            if args.real_time:
                sleep_for_realtime(next_step_wall, runtime.model.opt.timestep)

            if runtime.data.time >= next_vpr_time:
                rgb, _depth = runtime.camera.render_rgb_depth(runtime.data)
                closest_index, target_index, score = vpr.update(rgb)
                target_command, target_index, relpose = command_from_relative_pose(
                    rgb,
                    vpr.topomap_images,
                    target_index,
                    relpose_model,
                    relpose_img_reso,
                    args,
                )
                current_panel = rgb
                matched_panel = rgb_from_bgr(vpr.topomap_images[closest_index])
                subgoal_panel = rgb_from_bgr(vpr.topomap_images[target_index])
                next_vpr_time += vpr_interval

            if closest_index >= len(poses) - 1:
                target_command = np.zeros(3, dtype=np.float32)
                runtime.step_policy(target_command, False, args)
                stop_reason = "goal_reached"
                break

            active_motion = bool(np.linalg.norm(target_command) > 1e-6)
            applied_command = runtime.step_policy(target_command, active_motion, args)

            if runtime.data.time >= next_print_time:
                x_rel, y_rel, yaw_rel, v, w = relpose
                print(
                    f"vpr t={runtime.data.time:.2f}s closest={closest_index}/{len(poses)-1} "
                    f"target={target_index}/{len(poses)-1} score={score:.3f} "
                    f"rel=({x_rel:+.2f},{y_rel:+.2f},{yaw_rel:+.1f}deg) "
                    f"twist=({v:+.2f},{w:+.2f}) "
                    f"cmd=[{applied_command[0]:+.2f},{applied_command[1]:+.2f},{applied_command[2]:+.2f}]"
                )
                next_print_time += args.print_interval

            if runtime.viewer is not None and now >= next_render_wall:
                image_panels = []
                if current_panel is not None:
                    image_panels.append(("current camera", current_panel))
                if matched_panel is not None:
                    image_panels.append((f"VPR match #{closest_index}", matched_panel))
                if subgoal_panel is not None:
                    image_panels.append((f"subgoal #{target_index}", subgoal_panel))
                runtime.viewer.render_status(
                    runtime.data,
                    applied_command,
                    "cosplace-vpr-reloc3r-repeat",
                    f"closest={closest_index}/{len(poses)-1} target={target_index}/{len(poses)-1}",
                    f"score={score:.3f} rel=({relpose[0]:+.2f},{relpose[1]:+.2f},{relpose[2]:+.1f}deg)",
                    image_panels=image_panels,
                )
                next_render_wall += render_interval
                if next_render_wall < now - render_interval:
                    next_render_wall = now

            if runtime.data.qpos[2] < args.fall_height:
                stop_reason = "fall_height"
                break
            if final_uprightness(runtime.data) < args.fall_uprightness:
                stop_reason = "fall_uprightness"
                break
            if args.stop_on_obstacle_contact:
                obstacle = contact_with_named_obstacle(runtime.model, runtime.data)
                if obstacle:
                    stop_reason = f"obstacle_contact:{obstacle}"
                    break

            next_step_wall += runtime.model.opt.timestep
            if next_step_wall < now - 0.1:
                next_step_wall = now
    finally:
        runtime.close()

    print(f"CosPlace VPR autonomous repeat stopped: {stop_reason}")


if __name__ == "__main__":
    main()
