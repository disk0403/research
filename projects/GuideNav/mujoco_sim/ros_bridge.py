#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import (
    APPEARANCE_PRESETS,
    DEFAULT_SCENE_CONFIG,
    MODEL_DIR,
    POLICY_DIR,
    MujocoGo2Runtime,
    contact_with_named_obstacle,
    final_uprightness,
    setup_display,
    sleep_for_realtime,
    teleop_or_scripted_command,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish MuJoCo camera/depth/odom topics to the official GuideNav "
            "navigate.py and apply its /cmd_vel output back to the Go2 policy."
        )
    )
    parser.add_argument("mode", choices=["teach", "replay"])
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
    parser.add_argument("--max-cmd-vx", type=float, default=0.35)
    parser.add_argument("--max-cmd-vy", type=float, default=0.20)
    parser.add_argument("--max-cmd-yaw", type=float, default=0.8)
    parser.add_argument("--print-interval", type=float, default=1.0)
    return parser.parse_args()


class GuideNavMujocoBridge:
    def __init__(self, rclpy, args: argparse.Namespace):
        from geometry_msgs.msg import Twist
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        self.rclpy = rclpy
        self.Image = Image
        self.Odometry = Odometry
        self.node = rclpy.create_node("guidenav_mujoco_bridge")
        self.color_pub = self.node.create_publisher(Image, "/d435/color/image_raw", 10)
        self.d435i_color_pub = self.node.create_publisher(Image, "/d435i/color/image_raw", 10)
        self.depth_pub = self.node.create_publisher(Image, "/d435i/aligned_depth_to_color/image_raw", 10)
        self.odom_pub = self.node.create_publisher(Odometry, "/visual_slam/tracking/odometry", 10)
        self.cmd_sub = self.node.create_subscription(Twist, "/cmd_vel", self.cmd_vel_callback, 10)
        self.command = np.zeros(3, dtype=np.float32)
        self.command_count = 0
        self.args = args

    def cmd_vel_callback(self, msg) -> None:
        self.command = np.array(
            [
                np.clip(float(msg.linear.x), -self.args.max_cmd_vx, self.args.max_cmd_vx),
                np.clip(float(msg.linear.y), -self.args.max_cmd_vy, self.args.max_cmd_vy),
                np.clip(float(msg.angular.z), -self.args.max_cmd_yaw, self.args.max_cmd_yaw),
            ],
            dtype=np.float32,
        )
        self.command_count += 1

    def publish_sensors(self, runtime: MujocoGo2Runtime) -> None:
        rgb, depth = runtime.camera.render_rgb_depth(runtime.data)
        stamp = self.node.get_clock().now().to_msg()
        self.color_pub.publish(self._rgb_msg(rgb, stamp))
        self.d435i_color_pub.publish(self._rgb_msg(rgb, stamp))
        self.depth_pub.publish(self._depth_msg(depth, stamp))
        self.odom_pub.publish(self._odom_msg(runtime.data, stamp))

    def _rgb_msg(self, image: np.ndarray, stamp):
        msg = self.Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "guidenav_front_camera"
        msg.height = int(image.shape[0])
        msg.width = int(image.shape[1])
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(image.shape[1] * image.shape[2])
        msg.data = np.ascontiguousarray(image, dtype=np.uint8).tobytes()
        return msg

    def _depth_msg(self, depth: np.ndarray, stamp):
        depth = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        msg = self.Image()
        msg.header.stamp = stamp
        msg.header.frame_id = "guidenav_front_camera"
        msg.height = int(depth.shape[0])
        msg.width = int(depth.shape[1])
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = int(depth.shape[1] * 4)
        msg.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        return msg

    def _odom_msg(self, data, stamp):
        msg = self.Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = float(data.qpos[0])
        msg.pose.pose.position.y = float(data.qpos[1])
        msg.pose.pose.position.z = float(data.qpos[2])
        msg.pose.pose.orientation.w = float(data.qpos[3])
        msg.pose.pose.orientation.x = float(data.qpos[4])
        msg.pose.pose.orientation.y = float(data.qpos[5])
        msg.pose.pose.orientation.z = float(data.qpos[6])
        msg.twist.twist.linear.x = float(data.qvel[0])
        msg.twist.twist.linear.y = float(data.qvel[1])
        msg.twist.twist.linear.z = float(data.qvel[2])
        msg.twist.twist.angular.x = float(data.qvel[3])
        msg.twist.twist.angular.y = float(data.qvel[4])
        msg.twist.twist.angular.z = float(data.qvel[5])
        return msg

    def close(self) -> None:
        self.node.destroy_node()


def main() -> None:
    args = parse_args()
    setup_display(args.display, args.headless)
    try:
        import rclpy
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "rclpy is not available. Install/source ROS2 Humble, then run this bridge again."
        ) from exc

    rclpy.init()
    bridge = None
    runtime = None
    try:
        bridge = GuideNavMujocoBridge(rclpy, args)
        runtime = MujocoGo2Runtime(args, f"GuideNav MuJoCo bridge {args.mode}", not args.headless)
        next_step_wall = time.perf_counter()
        next_camera_time = float(runtime.data.time)
        next_print_time = 0.0
        render_interval = 1.0 / args.render_fps
        next_render_wall = next_step_wall
        camera_interval = 1.0 / args.camera_fps
        scripted_state: dict = {}
        stop_reason = "viewer_closed"

        print("Bridge topics:")
        print("  publishes /d435/color/image_raw")
        print("  publishes /d435i/color/image_raw")
        print("  publishes /d435i/aligned_depth_to_color/image_raw")
        print("  publishes /visual_slam/tracking/odometry")
        print("  subscribes /cmd_vel")

        while rclpy.ok():
            now = time.perf_counter()
            if args.duration > 0.0 and runtime.data.time >= args.duration:
                stop_reason = "duration"
                break
            if runtime.viewer is not None and not runtime.viewer.is_running():
                stop_reason = "viewer_closed"
                break
            if args.real_time:
                sleep_for_realtime(next_step_wall, runtime.model.opt.timestep)

            rclpy.spin_once(bridge.node, timeout_sec=0.0)
            if args.mode == "teach":
                target_command, active_motion, scripted_stop = teleop_or_scripted_command(runtime, args, scripted_state)
            else:
                target_command = bridge.command
                active_motion = bool(np.linalg.norm(target_command) > 1e-6)
                scripted_stop = ""

            command = runtime.step_policy(target_command, active_motion, args)

            if runtime.data.time >= next_camera_time:
                bridge.publish_sensors(runtime)
                next_camera_time += camera_interval

            if runtime.data.time >= next_print_time:
                print(
                    f"{args.mode} t={runtime.data.time:.2f}s "
                    f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
                    f"cmd_vel_msgs={bridge.command_count}"
                )
                next_print_time += args.print_interval

            if runtime.viewer is not None and now >= next_render_wall:
                runtime.viewer.render_status(
                    runtime.data,
                    command,
                    f"bridge-{args.mode}",
                    f"cmd_vel_msgs={bridge.command_count}",
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
            if scripted_stop:
                stop_reason = scripted_stop
                break

            next_step_wall += runtime.model.opt.timestep
            if next_step_wall < now - 0.1:
                next_step_wall = now

        print(f"Bridge stopped: {stop_reason}")
    finally:
        if runtime is not None:
            runtime.close()
        if bridge is not None:
            bridge.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
