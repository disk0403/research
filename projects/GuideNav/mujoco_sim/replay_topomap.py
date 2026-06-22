#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

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
    pose_distance,
    pose_from_data,
    setup_display,
    sleep_for_realtime,
    waypoint_command,
)

import mujoco


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "GUI autonomous repeat preview for MuJoCo. It follows the official "
            "GuideNav topomap/odom.csv keyframes with the Go2 velocity policy."
        )
    )
    parser.add_argument("--topomap-dir", type=Path, default=GUIDENAV_ROOT / "data" / "mujoco_teach" / "topomap")
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


def choose_target_index(
    current: Pose2D,
    poses: list[Pose2D],
    active_index: int,
    waypoint_radius: float,
    lookahead: int,
) -> tuple[int, int]:
    index = active_index
    while index < len(poses) - 1 and pose_distance(current, poses[index]) < waypoint_radius:
        index += 1
    target_index = min(len(poses) - 1, index + max(lookahead, 0))
    return index, target_index


def main() -> None:
    args = parse_args()
    setup_display(args.display, args.headless)

    poses = load_topomap_poses(args.topomap_dir)
    runtime = MujocoGo2Runtime(args, "GuideNav MuJoCo autonomous repeat", not args.headless)
    set_robot_planar_pose(runtime, poses[0], args)

    next_step_wall = time.perf_counter()
    next_render_wall = next_step_wall
    render_interval = 1.0 / args.render_fps
    next_print_time = 0.0
    active_index = 1
    stop_reason = "viewer_closed"

    print(f"Loaded {len(poses)} topomap poses from: {args.topomap_dir}")
    print("Autonomous repeat started. GUI controls: Esc closes, mouse drags camera.")

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

            current = pose_from_data(runtime.data)
            if pose_distance(current, poses[-1]) < args.goal_radius:
                command = np.zeros(3, dtype=np.float32)
                runtime.step_policy(command, False, args)
                stop_reason = "goal_reached"
                break

            active_index, target_index = choose_target_index(
                current,
                poses,
                active_index,
                args.waypoint_radius,
                args.lookahead,
            )
            command = waypoint_command(
                current,
                poses[target_index],
                max_vx=args.max_vx,
                max_vy=args.max_vy,
                max_yaw=args.max_yaw,
            )
            active_motion = bool(np.linalg.norm(command) > 1e-6)
            command = runtime.step_policy(command, active_motion, args)

            if runtime.data.time >= next_print_time:
                print(
                    f"auto t={runtime.data.time:.2f}s target={target_index}/{len(poses)-1} "
                    f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}]"
                )
                next_print_time += args.print_interval

            if runtime.viewer is not None and now >= next_render_wall:
                runtime.viewer.render_status(
                    runtime.data,
                    command,
                    "autonomous-repeat",
                    f"target={target_index}/{len(poses)-1}  topomap={args.topomap_dir}",
                    f"preset={args.scene_preset}",
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

    print(f"Autonomous repeat stopped: {stop_reason}")


if __name__ == "__main__":
    main()
