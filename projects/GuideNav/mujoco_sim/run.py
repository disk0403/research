#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import DEFAULT_SCENE_CONFIG, GUIDENAV_ROOT


DATA_DIR = GUIDENAV_ROOT / "data" / "mujoco_teach"
RAW_DIR = DATA_DIR / "raw"
DEFAULT_TOPOMAP_DIR = DATA_DIR / "topomap"
ARCHIVE_DIR = DATA_DIR / "archive"
LATEST_RUN_FILE = DATA_DIR / "latest_raw.txt"
DEFAULT_REVIEW_DIR = GUIDENAV_ROOT.parents[1] / "local" / "guidenav_mujoco" / "latest"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Simple GuideNav MuJoCo workflow: teach once, build a topomap, "
            "then repeat autonomously."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    teach = subparsers.add_parser("teach", help="teleop once and record RGB-D/odometry")
    add_scene_args(teach)
    teach.add_argument("--output-dir", type=Path, default=None)
    teach.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means until Esc/window close")
    teach.add_argument("--camera-fps", type=float, default=2.0)
    teach.add_argument("--render-fps", type=float, default=30.0)
    teach.add_argument("--image-width", type=int, default=320)
    teach.add_argument("--image-height", type=int, default=180)
    teach.add_argument("--headless", action="store_true")
    teach.add_argument("--scripted-teacher", action="store_true")

    build = subparsers.add_parser("build", help="build topomap and CosPlace VPR DB from the latest teach run")
    build.add_argument("--run", type=Path, default=None, help="raw teach run directory; default is latest")
    build.add_argument("--topomap-dir", type=Path, default=DEFAULT_TOPOMAP_DIR)
    build.add_argument("--distance", type=float, default=0.35)
    build.add_argument("--yaw", type=float, default=14.0)
    build.add_argument("--pr-model", default="cosplace_hub")
    build.add_argument("--skip-vpr", action="store_true")
    build.add_argument(
        "--keep-existing",
        action="store_true",
        help="do not archive an existing topomap directory before rebuilding",
    )

    repeat = subparsers.add_parser("repeat", help="run autonomous repeat using CosPlace + ReLoc3R")
    add_scene_args(repeat)
    repeat.add_argument("--topomap-dir", type=Path, default=DEFAULT_TOPOMAP_DIR)
    repeat.add_argument("--pr-model", default="cosplace_hub")
    repeat.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    repeat.add_argument("--vpr-fps", type=float, default=2.0)
    repeat.add_argument("--duration", type=float, default=0.0, help="seconds; 0 means until Esc/window close")
    repeat.add_argument("--render-fps", type=float, default=30.0)
    repeat.add_argument("--image-width", type=int, default=320)
    repeat.add_argument("--image-height", type=int, default=180)
    repeat.add_argument("--headless", action="store_true")
    repeat.add_argument("--no-real-time", action="store_true")

    visuals = subparsers.add_parser("visuals", help="export recorded images and keyframes to local/ for review")
    visuals.add_argument("--run", type=Path, default=None, help="raw teach run directory; default is latest")
    visuals.add_argument("--topomap-dir", type=Path, default=DEFAULT_TOPOMAP_DIR)
    visuals.add_argument("--output-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    visuals.add_argument("--max-raw-thumbs", type=int, default=80)

    check = subparsers.add_parser("check", help="check installed dependencies and expected files")
    check.set_defaults(func=cmd_check)

    return parser.parse_args()


def add_scene_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument(
        "--scene-preset",
        choices=("cloudy_noon", "night", "rainy_evening", "sunny_morning"),
        default="sunny_morning",
    )
    parser.add_argument("--cycle-appearance", action="store_true")
    parser.add_argument("--cycle-period", type=float, default=18.0)


def run_command(args: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-guidenav")
    env.setdefault("HF_HOME", str(GUIDENAV_ROOT / "model_weights" / "huggingface"))
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=GUIDENAV_ROOT, env=env, check=True)


def print_next(title: str, command: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(command)


def latest_raw_run() -> Path:
    if LATEST_RUN_FILE.exists():
        candidate = Path(LATEST_RUN_FILE.read_text(encoding="utf-8").strip())
        if candidate.exists():
            return candidate

    runs = [path for path in RAW_DIR.glob("run_*") if path.is_dir()]
    if not runs:
        raise FileNotFoundError(
            "No teaching run found. First run:\n"
            "  python3 mujoco_sim/run.py teach"
        )
    return max(runs, key=lambda path: path.stat().st_mtime)


def archive_existing_topomap(topomap_dir: Path) -> None:
    if not topomap_dir.exists() or not any(topomap_dir.iterdir()):
        return
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = ARCHIVE_DIR / f"{topomap_dir.name}_{stamp}"
    shutil.move(str(topomap_dir), str(archived))
    print(f"Existing topomap moved to: {archived}")


def auto_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def cmd_teach(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or RAW_DIR / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    command = [
        sys.executable,
        "mujoco_sim/record_teach_run.py",
        "--real-time",
        "--scene-config",
        str(args.scene_config),
        "--scene-preset",
        args.scene_preset,
        "--image-width",
        str(args.image_width),
        "--image-height",
        str(args.image_height),
        "--camera-fps",
        str(args.camera_fps),
        "--render-fps",
        str(args.render_fps),
        "--cycle-period",
        str(args.cycle_period),
        "--output-dir",
        str(output_dir),
    ]
    if args.cycle_appearance:
        command.append("--cycle-appearance")
    if args.duration > 0.0:
        command += ["--duration", str(args.duration)]
    if args.headless:
        command.append("--headless")
    if args.scripted_teacher:
        command.append("--scripted-teacher")

    run_command(command)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.write_text(str(output_dir.resolve()) + "\n", encoding="utf-8")
    print_next(
        "次に実行: topomap と VPR DB を作る",
        "python3 mujoco_sim/run.py build",
    )


def cmd_build(args: argparse.Namespace) -> None:
    run_dir = args.run or latest_raw_run()
    if not (run_dir / "odom.csv").exists():
        raise FileNotFoundError(f"Teaching run does not contain odom.csv: {run_dir}")
    if not (run_dir / "color").exists():
        raise FileNotFoundError(f"Teaching run does not contain color/: {run_dir}")

    if not args.keep_existing:
        archive_existing_topomap(args.topomap_dir)

    run_command(
        [
            sys.executable,
            "sensor/build_topomap.py",
            str(run_dir),
            str(args.topomap_dir),
            "--distance",
            str(args.distance),
            "--yaw",
            str(args.yaw),
        ]
    )

    if not args.skip_vpr:
        run_command(
            [
                sys.executable,
                "mujoco_sim/build_vpr_db.py",
                "--topomap-dir",
                str(args.topomap_dir),
                "--pr-model",
                args.pr_model,
                "--overwrite",
            ]
        )

    export_review_folder(run_dir, args.topomap_dir, DEFAULT_REVIEW_DIR)
    print_next(
        "次に実行: 自律 repeat",
        "python3 mujoco_sim/run.py repeat",
    )


def cmd_repeat(args: argparse.Namespace) -> None:
    feature_path = args.topomap_dir / "topo" / "global-feats-cosplace_hub.h5"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"VPR DB not found: {feature_path}\n"
            "First run:\n"
            "  python3 mujoco_sim/run.py build"
        )

    device = auto_device(args.device)
    print(f"Using device: {device}")
    command = [
        sys.executable,
        "mujoco_sim/replay_cosplace.py",
        "--topomap-dir",
        str(args.topomap_dir),
        "--pr-model",
        args.pr_model,
        "--feature-matching",
        "reloc3r",
        "--device",
        device,
        "--scene-config",
        str(args.scene_config),
        "--scene-preset",
        args.scene_preset,
        "--image-width",
        str(args.image_width),
        "--image-height",
        str(args.image_height),
        "--vpr-fps",
        str(args.vpr_fps),
        "--render-fps",
        str(args.render_fps),
        "--cycle-period",
        str(args.cycle_period),
    ]
    command.append("--no-real-time" if args.no_real_time else "--real-time")
    if args.cycle_appearance:
        command.append("--cycle-appearance")
    if args.duration > 0.0:
        command += ["--duration", str(args.duration)]
    if args.headless:
        command.append("--headless")

    run_command(command)


def cmd_check(_args: argparse.Namespace) -> None:
    run_command([sys.executable, "mujoco_sim/check_env.py"])


def export_review_folder(raw_dir: Path, topomap_dir: Path, output_dir: Path, max_raw_thumbs: int = 80) -> None:
    run_command(
        [
            sys.executable,
            "mujoco_sim/export_visuals.py",
            "--raw-dir",
            str(raw_dir),
            "--topomap-dir",
            str(topomap_dir),
            "--output-dir",
            str(output_dir),
            "--max-raw-thumbs",
            str(max_raw_thumbs),
        ]
    )
    print_next(
        "画像確認フォルダ",
        str(output_dir / "index.html"),
    )


def cmd_visuals(args: argparse.Namespace) -> None:
    export_review_folder(
        args.run or latest_raw_run(),
        args.topomap_dir,
        args.output_dir,
        args.max_raw_thumbs,
    )


def main() -> None:
    args = parse_args()
    command_map = {
        "teach": cmd_teach,
        "build": cmd_build,
        "repeat": cmd_repeat,
        "visuals": cmd_visuals,
        "check": cmd_check,
    }
    command_map[args.command](args)


if __name__ == "__main__":
    main()
