#!/usr/bin/env python3
from __future__ import annotations

import argparse
import traceback
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from common import (
    APPEARANCE_PRESETS,
    DEFAULT_SCENE_CONFIG,
    GUIDENAV_ROOT,
    MODEL_DIR,
    POLICY_DIR,
    MujocoGo2Runtime,
    contact_with_named_obstacle,
    final_uprightness,
    setup_display,
    sleep_for_realtime,
    teleop_or_scripted_command,
    write_odom_header,
    write_odom_row,
    write_png_depth16,
    write_png_rgb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record MuJoCo RGB-D and odometry into the exact directory format "
            "consumed by the official GuideNav sensor/build_topomap.py script."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--display", default=":1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--real-time", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--camera-fps", type=float, default=4.0)
    parser.add_argument("--render-fps", type=float, default=30.0)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=360)
    parser.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    parser.add_argument("--scene-preset", choices=sorted(APPEARANCE_PRESETS), default="sunny_morning")
    parser.add_argument("--cycle-appearance", action="store_true")
    parser.add_argument("--cycle-period", type=float, default=18.0)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    parser.add_argument("--scripted-teacher", action="store_true")
    parser.add_argument("--normal-speed", type=float, default=0.32)
    parser.add_argument("--dash-forward-speed", type=float, default=0.55)
    parser.add_argument("--dash-backward-speed", type=float, default=0.35)
    parser.add_argument("--dash-lateral-speed", type=float, default=0.35)
    parser.add_argument("--yaw-speed", type=float, default=0.40)
    parser.add_argument("--yaw-safety-limit", type=float, default=1.0)
    parser.add_argument("--command-smoothing", type=float, default=10.0)
    parser.add_argument("--reset-base-height", type=float, default=0.25)
    parser.add_argument("--stance-crouch", type=float, default=0.08)
    parser.add_argument("--fall-height", type=float, default=0.16)
    parser.add_argument("--fall-uprightness", type=float, default=0.55)
    parser.add_argument("--stop-on-obstacle-contact", action="store_true")
    parser.add_argument("--print-interval", type=float, default=1.0)
    return parser.parse_args()


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return GUIDENAV_ROOT / "data" / "mujoco_teach" / "raw" / stamp


def main() -> None:
    args = parse_args()
    setup_display(args.display, args.headless)

    output_dir = args.output_dir or default_output_dir()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already has files: {output_dir}\n"
            "Make a new RUN value before recording again:\n"
            "  RUN=\"$PWD/data/mujoco_teach/raw/run_$(date +%Y%m%d_%H%M%S)\""
        )

    color_dir = output_dir / "color"
    depth_dir = output_dir / "depth"
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    odom_file, odom_writer = write_odom_header(output_dir / "odom.csv")
    log_path = output_dir / "run.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(message + "\n")

    log(f"Starting GuideNav MuJoCo recorder: {output_dir}")
    try:
        runtime = MujocoGo2Runtime(args, "GuideNav MuJoCo teaching recorder", not args.headless)
    except BaseException:
        log("Startup failed before the first frame. Traceback:")
        with log_path.open("a", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        odom_file.close()
        raise
    next_step_wall = time.perf_counter()
    next_camera_time = float(runtime.data.time)
    next_print_time = 0.0
    render_interval = 1.0 / args.render_fps
    next_render_wall = next_step_wall
    camera_interval = 1.0 / args.camera_fps
    scripted_state: dict = {}
    frames = 0
    stop_reason = "viewer_closed"

    log(f"Saving GuideNav teaching data to: {output_dir}")
    log("Controls: W/S forward/back, A/D strafe, Q/E turn, Shift faster, R reset, Esc finish.")
    log("After recording, build the official topomap with:")
    log(f"  python sensor/build_topomap.py {output_dir} data/mujoco_teach/topomap --distance 0.35 --yaw 14")

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

            target_command, active_motion, scripted_stop = teleop_or_scripted_command(runtime, args, scripted_state)
            command = runtime.step_policy(target_command, active_motion, args)

            if runtime.data.time >= next_camera_time:
                rgb, depth = runtime.camera.render_rgb_depth(runtime.data)
                timestamp = f"{float(runtime.data.time):.9f}"
                write_png_rgb(color_dir / f"{timestamp}.png", rgb)
                write_png_depth16(depth_dir / f"{timestamp}.png", depth)
                write_odom_row(odom_writer, runtime.data)
                odom_file.flush()
                frames += 1
                next_camera_time += camera_interval

            if runtime.data.time >= next_print_time:
                log(
                    f"record t={runtime.data.time:.2f}s frames={frames} "
                    f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}]"
                )
                next_print_time += args.print_interval

            if runtime.viewer is not None and now >= next_render_wall:
                runtime.viewer.render_status(
                    runtime.data,
                    command,
                    "record",
                    f"frames={frames}  output={output_dir}",
                    f"preset={args.scene_preset} camera_fps={args.camera_fps:.1f}",
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
            if scripted_stop:
                stop_reason = scripted_stop
                break

            next_step_wall += runtime.model.opt.timestep
            if next_step_wall < now - 0.1:
                next_step_wall = now
    finally:
        odom_file.close()
        runtime.close()

    log(f"Recorder stopped: {stop_reason}, frames={frames}, output={output_dir}")


if __name__ == "__main__":
    main()
