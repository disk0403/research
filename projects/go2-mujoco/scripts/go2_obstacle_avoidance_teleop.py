from __future__ import annotations

import argparse
import math
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from go2_teleop import (
    DEFAULT_DISPLAY,
    DEFAULT_FALL_HEIGHT,
    DEFAULT_FALL_UPRIGHTNESS,
    DEFAULT_FALL_WARMUP,
    DEFAULT_IDLE_BASE_DAMPING,
    DEFAULT_IDLE_DAMPING_SCALE,
    DEFAULT_IDLE_SPEED_DEADBAND,
    DEFAULT_MAX_STANCE_CROUCH,
    DEFAULT_MIN_STANCE_CROUCH,
    DEFAULT_RENDER_FPS,
    DEFAULT_RESET_BASE_HEIGHT,
    DEFAULT_STANCE_ADJUST_STEP,
    DEFAULT_STANCE_CROUCH,
    DEFAULT_YAW_SAFETY_LIMIT,
    DEFAULT_YAW_SPEED,
    FlatWasdDashViewer,
    JointSpaceJumpController,
    SimToRealPolicyController,
    apply_idle_base_damping,
    clamp_value,
    fall_reason,
    final_uprightness,
    limit_yaw_command,
    reset_robot_pose,
    root_rotation_matrix,
    should_apply_idle_stabilization,
    update_command_with_release_cutoff,
    validate_args as validate_teleop_args,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "go2"
POLICY_DIR = ROOT / "external" / "policies" / "unitree-go2-velocity-flat"


@dataclass(frozen=True)
class BoxObstacle:
    name: str
    pos: tuple[float, float, float]
    size: tuple[float, float, float]
    rgba: tuple[float, float, float, float]


@dataclass
class RuntimeScene:
    temp_dir: tempfile.TemporaryDirectory
    scene_path: Path

    def cleanup(self) -> None:
        self.temp_dir.cleanup()


@dataclass(frozen=True)
class ObstacleScan:
    distances: np.ndarray
    angles: np.ndarray
    path_distance: float
    min_distance: float
    left_clearance: float
    right_clearance: float
    steer_sign: float
    hit_count: int


DEFAULT_OBSTACLES = (
    BoxObstacle(
        "box_center",
        (1.85, 0.00, 0.24),
        (0.20, 0.34, 0.24),
        (0.85, 0.25, 0.18, 1.0),
    ),
    BoxObstacle(
        "box_left",
        (3.10, 0.70, 0.20),
        (0.26, 0.22, 0.20),
        (0.22, 0.55, 0.95, 1.0),
    ),
    BoxObstacle(
        "box_right",
        (4.20, -0.55, 0.28),
        (0.22, 0.30, 0.28),
        (0.65, 0.50, 0.12, 1.0),
    ),
)


def parse_obstacle(value: str) -> tuple[float, float, float, float, float, float]:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "Obstacle must be x,y,z,half_x,half_y,half_z."
        )

    try:
        numbers = tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Obstacle values must be numeric: x,y,z,half_x,half_y,half_z."
        ) from exc

    if any(size <= 0.0 for size in numbers[3:]):
        raise argparse.ArgumentTypeError("Obstacle half sizes must be positive.")
    return numbers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WASD/QE Go2 teleoperation with simple raycast obstacle avoidance."
        )
    )
    parser.add_argument("--display", default=DEFAULT_DISPLAY)
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=POLICY_DIR,
        help="Directory containing policy.onnx, policy.onnx.data, and params/deploy.yaml.",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 means run until the viewer closes.",
    )
    parser.add_argument("--render-fps", type=float, default=DEFAULT_RENDER_FPS)
    parser.add_argument("--normal-speed", type=float, default=0.45)
    parser.add_argument("--dash-forward-speed", type=float, default=1.0)
    parser.add_argument("--dash-backward-speed", type=float, default=0.7)
    parser.add_argument("--dash-lateral-speed", type=float, default=0.7)
    parser.add_argument("--yaw-speed", type=float, default=DEFAULT_YAW_SPEED)
    parser.add_argument(
        "--yaw-safety-limit",
        type=float,
        default=DEFAULT_YAW_SAFETY_LIMIT,
        help=(
            "Clamp commanded yaw rate before policy inference. Use 0 to disable."
        ),
    )
    parser.add_argument("--command-smoothing", type=float, default=12.0)
    parser.add_argument("--reset-base-height", type=float, default=DEFAULT_RESET_BASE_HEIGHT)
    parser.add_argument("--stance-crouch", type=float, default=DEFAULT_STANCE_CROUCH)
    parser.add_argument("--min-stance-crouch", type=float, default=DEFAULT_MIN_STANCE_CROUCH)
    parser.add_argument("--max-stance-crouch", type=float, default=DEFAULT_MAX_STANCE_CROUCH)
    parser.add_argument("--stance-adjust-step", type=float, default=DEFAULT_STANCE_ADJUST_STEP)
    parser.add_argument("--fall-height", type=float, default=DEFAULT_FALL_HEIGHT)
    parser.add_argument("--fall-uprightness", type=float, default=DEFAULT_FALL_UPRIGHTNESS)
    parser.add_argument("--fall-warmup", type=float, default=DEFAULT_FALL_WARMUP)
    parser.add_argument("--no-auto-reset-on-fall", action="store_true")
    parser.add_argument("--idle-damping-scale", type=float, default=DEFAULT_IDLE_DAMPING_SCALE)
    parser.add_argument("--idle-base-damping", type=float, default=DEFAULT_IDLE_BASE_DAMPING)
    parser.add_argument("--idle-speed-deadband", type=float, default=DEFAULT_IDLE_SPEED_DEADBAND)
    parser.add_argument("--jump-duration", type=float, default=0.78)
    parser.add_argument("--jump-blend", type=float, default=0.95)
    parser.add_argument("--test-command-vx", type=float, default=0.0)
    parser.add_argument("--test-command-vy", type=float, default=0.0)
    parser.add_argument("--test-command-yaw", type=float, default=0.0)
    parser.add_argument("--test-jump-time", type=float, default=-1.0)
    parser.add_argument(
        "--test-jump-hold",
        type=float,
        default=0.35,
        help="Deprecated compatibility option. Jump is tap-triggered.",
    )
    parser.add_argument(
        "--no-default-obstacles",
        action="store_true",
        help="Do not add the default cuboid obstacles.",
    )
    parser.add_argument(
        "--obstacle",
        type=parse_obstacle,
        action="append",
        default=[],
        metavar="X,Y,Z,HALF_X,HALF_Y,HALF_Z",
        help=(
            "Add a cuboid obstacle. MuJoCo box sizes are half extents in meters. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--sensor-range",
        type=float,
        default=1.60,
        help="Raycast distance treated as clear when no obstacle is hit.",
    )
    parser.add_argument(
        "--avoid-distance",
        type=float,
        default=1.15,
        help="Start reducing forward command below this path distance.",
    )
    parser.add_argument(
        "--stop-distance",
        type=float,
        default=0.42,
        help="Stop forward command below this path distance.",
    )
    parser.add_argument(
        "--ray-count",
        type=int,
        default=9,
        help="Number of horizontal raycast beams in the front fan.",
    )
    parser.add_argument(
        "--ray-angle-span-deg",
        type=float,
        default=80.0,
        help="Total horizontal fan angle in degrees.",
    )
    parser.add_argument(
        "--path-angle-deg",
        type=float,
        default=22.0,
        help="Central fan region used for forward-path distance.",
    )
    parser.add_argument(
        "--ray-origin-forward",
        type=float,
        default=0.38,
        help="Ray origin offset forward from the base frame in meters.",
    )
    parser.add_argument(
        "--ray-height-offset",
        type=float,
        default=0.00,
        help="Ray origin z offset from the base/root position in meters.",
    )
    parser.add_argument(
        "--avoid-forward-deadband",
        type=float,
        default=0.05,
        help="Only apply avoidance when requested forward speed exceeds this.",
    )
    parser.add_argument(
        "--min-forward-scale",
        type=float,
        default=0.18,
        help="Lowest forward-speed scale before the hard stop distance.",
    )
    parser.add_argument(
        "--max-avoid-yaw-rate",
        type=float,
        default=DEFAULT_YAW_SPEED,
        help="Maximum yaw rate injected by obstacle avoidance.",
    )
    parser.add_argument(
        "--avoid-lateral-speed",
        type=float,
        default=0.15,
        help="Optional lateral speed injected toward the clearer side.",
    )
    parser.add_argument(
        "--prefer-side",
        choices=("left", "right"),
        default="left",
        help="Tie-break side when both sides look equally clear.",
    )
    parser.add_argument(
        "--disable-avoidance",
        action="store_true",
        help="Keep obstacles in the scene but do not alter teleop commands.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="Status print interval in seconds. Use 0 to disable.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    validate_teleop_args(args)
    if args.sensor_range <= 0.0:
        raise ValueError("Sensor range must be positive.")
    if args.avoid_distance <= 0.0:
        raise ValueError("Avoid distance must be positive.")
    if args.stop_distance <= 0.0:
        raise ValueError("Stop distance must be positive.")
    if args.stop_distance >= args.avoid_distance:
        raise ValueError("Stop distance must be smaller than avoid distance.")
    if args.sensor_range < args.avoid_distance:
        raise ValueError("Sensor range must be >= avoid distance.")
    if args.ray_count < 3:
        raise ValueError("Ray count must be at least 3.")
    if args.ray_angle_span_deg <= 0.0 or args.ray_angle_span_deg >= 180.0:
        raise ValueError("Ray angle span must be in (0, 180) degrees.")
    if args.path_angle_deg <= 0.0:
        raise ValueError("Path angle must be positive.")
    if args.path_angle_deg > 0.5 * args.ray_angle_span_deg:
        raise ValueError("Path angle must be within half of ray angle span.")
    if args.min_forward_scale < 0.0 or args.min_forward_scale > 1.0:
        raise ValueError("Minimum forward scale must be in [0, 1].")
    if args.max_avoid_yaw_rate < 0.0:
        raise ValueError("Maximum avoidance yaw rate must be non-negative.")
    if args.avoid_lateral_speed < 0.0:
        raise ValueError("Avoid lateral speed must be non-negative.")
    if args.print_interval < 0.0:
        raise ValueError("Print interval must be non-negative.")


def build_obstacles(args: argparse.Namespace) -> list[BoxObstacle]:
    obstacles: list[BoxObstacle] = []
    if not args.no_default_obstacles:
        obstacles.extend(DEFAULT_OBSTACLES)

    for index, values in enumerate(args.obstacle):
        x, y, z, sx, sy, sz = values
        obstacles.append(
            BoxObstacle(
                f"custom_box_{index}",
                (x, y, z),
                (sx, sy, sz),
                (0.70, 0.70, 0.72, 1.0),
            )
        )
    return obstacles


def create_runtime_scene(obstacles: list[BoxObstacle]) -> RuntimeScene:
    temp_dir = tempfile.TemporaryDirectory(prefix="go2_obstacle_avoidance_")
    temp_path = Path(temp_dir.name)
    runtime_go2_xml = temp_path / "go2.xml"
    runtime_scene_xml = temp_path / "obstacle_avoidance_scene.xml"

    shutil.copy2(MODEL_DIR / "go2.xml", runtime_go2_xml)
    (temp_path / "assets").symlink_to(MODEL_DIR / "assets", target_is_directory=True)
    runtime_scene_xml.write_text(build_scene_xml(obstacles), encoding="utf-8")
    return RuntimeScene(temp_dir, runtime_scene_xml)


def build_scene_xml(obstacles: list[BoxObstacle]) -> str:
    materials = []
    geoms = []
    for obstacle in obstacles:
        rgba = " ".join(f"{value:.3f}" for value in obstacle.rgba)
        pos = " ".join(f"{value:.6f}" for value in obstacle.pos)
        size = " ".join(f"{value:.6f}" for value in obstacle.size)
        materials.append(
            f'    <material name="{obstacle.name}_mat" rgba="{rgba}" />'
        )
        geoms.append(
            "\n".join(
                [
                    "    <geom",
                    f'      name="{obstacle.name}"',
                    '      type="box"',
                    f'      pos="{pos}"',
                    f'      size="{size}"',
                    f'      material="{obstacle.name}_mat"',
                    '      friction="0.8 0.1 0.1"',
                    "    />",
                ]
            )
        )

    material_xml = "\n".join(materials)
    obstacle_xml = "\n".join(geoms)
    return f"""<mujoco model="go2 obstacle avoidance teleop">
  <include file="go2.xml" />

  <statistic center="0 0 0.1" extent="0.8" />

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-130" elevation="-20" />
  </visual>

  <asset>
    <texture
      type="skybox"
      builtin="gradient"
      rgb1="0.3 0.5 0.7"
      rgb2="0 0 0"
      width="512"
      height="3072"
    />
    <texture
      type="2d"
      name="groundplane"
      builtin="checker"
      mark="edge"
      rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8"
      width="300"
      height="300"
    />
    <material
      name="groundplane"
      texture="groundplane"
      texuniform="true"
      texrepeat="8 8"
      reflectance="0.2"
    />
{material_xml}
  </asset>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane" />
{obstacle_xml}
  </worldbody>
</mujoco>
"""


class ObstacleAvoidanceViewer(FlatWasdDashViewer):
    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        if self._window is not None:
            self._glfw.set_window_title(
                self._window,
                "Go2 obstacle avoidance teleop - WASD/QE",
            )


def ray_angles(args: argparse.Namespace) -> np.ndarray:
    span = math.radians(args.ray_angle_span_deg)
    return np.linspace(-0.5 * span, 0.5 * span, args.ray_count, dtype=np.float64)


def scan_obstacles(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    angles: np.ndarray,
    args: argparse.Namespace,
) -> ObstacleScan:
    rotation = root_rotation_matrix(data)
    origin = np.asarray(data.qpos[:3], dtype=np.float64) + rotation @ np.array(
        [args.ray_origin_forward, 0.0, args.ray_height_offset],
        dtype=np.float64,
    )
    distances = np.empty(len(angles), dtype=np.float64)
    geomid = np.array([-1], dtype=np.int32)

    for index, angle in enumerate(angles):
        body_direction = np.array(
            [math.cos(float(angle)), math.sin(float(angle)), 0.0],
            dtype=np.float64,
        )
        world_direction = rotation @ body_direction
        world_direction[2] = 0.0
        norm = float(np.linalg.norm(world_direction))
        if norm < 1e-9:
            distances[index] = args.sensor_range
            continue

        world_direction /= norm
        geomid[0] = -1
        distance = mujoco.mj_ray(
            model,
            data,
            origin,
            world_direction,
            None,
            1,
            -1,
            geomid,
        )
        if distance < 0.0 or distance > args.sensor_range:
            distances[index] = args.sensor_range
        else:
            distances[index] = float(distance)

    path_mask = np.abs(angles) <= math.radians(args.path_angle_deg)
    if not np.any(path_mask):
        path_mask[len(angles) // 2] = True

    left_mask = angles > math.radians(args.path_angle_deg)
    right_mask = angles < -math.radians(args.path_angle_deg)
    path_distance = float(np.min(distances[path_mask]))
    min_distance = float(np.min(distances))
    left_clearance = side_clearance(distances[left_mask], args.sensor_range)
    right_clearance = side_clearance(distances[right_mask], args.sensor_range)
    steer_sign = choose_steer_sign(distances, angles, left_clearance, right_clearance, args)
    hit_count = int(np.count_nonzero(distances < args.sensor_range))
    return ObstacleScan(
        distances=distances,
        angles=angles,
        path_distance=path_distance,
        min_distance=min_distance,
        left_clearance=left_clearance,
        right_clearance=right_clearance,
        steer_sign=steer_sign,
        hit_count=hit_count,
    )


def side_clearance(distances: np.ndarray, fallback: float) -> float:
    if distances.size == 0:
        return fallback
    return float(np.mean(distances))


def choose_steer_sign(
    distances: np.ndarray,
    angles: np.ndarray,
    left_clearance: float,
    right_clearance: float,
    args: argparse.Namespace,
) -> float:
    danger = np.clip(
        (args.avoid_distance - distances) / (args.avoid_distance - args.stop_distance),
        0.0,
        1.0,
    )
    side_pressure = -float(np.sum(np.sign(angles) * danger * np.abs(angles)))
    if abs(side_pressure) > 1e-4:
        return 1.0 if side_pressure > 0.0 else -1.0

    if abs(left_clearance - right_clearance) < 1e-3:
        return 1.0 if args.prefer_side == "left" else -1.0
    return 1.0 if left_clearance > right_clearance else -1.0


def apply_obstacle_avoidance(
    command: np.ndarray,
    scan: ObstacleScan,
    args: argparse.Namespace,
) -> tuple[np.ndarray, bool, str]:
    if args.disable_avoidance or command[0] <= args.avoid_forward_deadband:
        note = (
            f"avoid=off path={scan.path_distance:.2f}m"
            if args.disable_avoidance
            else f"scan path={scan.path_distance:.2f}m"
        )
        return command.astype(np.float32, copy=True), False, note

    adjusted = command.astype(np.float32, copy=True)
    if scan.path_distance >= args.avoid_distance:
        return (
            adjusted,
            False,
            (
                f"clear path={scan.path_distance:.2f}m "
                f"L={scan.left_clearance:.2f} R={scan.right_clearance:.2f}"
            ),
        )

    strength = clamp_value(
        (args.avoid_distance - scan.path_distance)
        / (args.avoid_distance - args.stop_distance),
        0.0,
        1.0,
    )
    if scan.path_distance <= args.stop_distance:
        forward_scale = 0.0
    else:
        forward_scale = max(args.min_forward_scale, 1.0 - strength)

    adjusted[0] *= forward_scale
    if args.avoid_lateral_speed > 0.0:
        adjusted[1] += np.float32(scan.steer_sign * args.avoid_lateral_speed * strength)

    avoid_yaw = scan.steer_sign * args.max_avoid_yaw_rate * max(strength, 0.25)
    if scan.steer_sign > 0.0:
        adjusted[2] = max(float(adjusted[2]), avoid_yaw)
    else:
        adjusted[2] = min(float(adjusted[2]), avoid_yaw)

    max_yaw = max(args.yaw_speed, args.max_avoid_yaw_rate)
    adjusted[2] = np.float32(clamp_value(float(adjusted[2]), -max_yaw, max_yaw))
    side = "left" if scan.steer_sign > 0.0 else "right"
    mode = "stop_turn" if scan.path_distance <= args.stop_distance else "avoid"
    return (
        adjusted,
        True,
        (
            f"{mode}={side} path={scan.path_distance:.2f}m "
            f"scale={forward_scale:.2f} L={scan.left_clearance:.2f} "
            f"R={scan.right_clearance:.2f}"
        ),
    )


def print_status(
    data: mujoco.MjData,
    raw_command: np.ndarray,
    command: np.ndarray,
    scan: ObstacleScan,
    note: str,
) -> None:
    ray_summary = " ".join(f"{distance:.2f}" for distance in scan.distances)
    print(
        f"t={data.time:6.2f} "
        f"raw=[{raw_command[0]:+.2f},{raw_command[1]:+.2f},{raw_command[2]:+.2f}] "
        f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
        f"hits={scan.hit_count:d} rays=[{ray_summary}] {note}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.headless:
        os.environ["DISPLAY"] = args.display

    obstacles = build_obstacles(args)
    runtime_scene = create_runtime_scene(obstacles)
    try:
        model = mujoco.MjModel.from_xml_path(str(runtime_scene.scene_path))
        data = mujoco.MjData(model)
        policy = SimToRealPolicyController(model, args.policy_dir)
        jump = JointSpaceJumpController(policy, args.jump_duration, args.jump_blend)
        stance_crouch = clamp_value(
            args.stance_crouch,
            args.min_stance_crouch,
            args.max_stance_crouch,
        )
        policy.initialize_pose(
            data,
            base_height=args.reset_base_height,
            stance_crouch=stance_crouch,
        )

        print(f"Scene: {runtime_scene.scene_path}")
        print(f"Policy: {args.policy_dir}")
        print(f"Obstacles: {len(obstacles)} cuboid(s)")
        for obstacle in obstacles:
            print(
                f"  {obstacle.name}: pos={obstacle.pos}, half_size={obstacle.size}"
            )
        print(
            "Avoidance: "
            f"{'off' if args.disable_avoidance else 'on'}, "
            f"range={args.sensor_range:.2f}m, "
            f"avoid<{args.avoid_distance:.2f}m, "
            f"stop<{args.stop_distance:.2f}m, "
            f"rays={args.ray_count:d}/{args.ray_angle_span_deg:.0f}deg, "
            f"yaw_safety={args.yaw_safety_limit:.2f}rad/s"
        )
        print(
            "Controls: W/A/S/D move, Q/E turn, Shift+WASD dashes, "
            "Space taps jump, R resets."
        )

        viewer = None
        if not args.headless:
            viewer = ObstacleAvoidanceViewer(model)

        angles = ray_angles(args)
        command = np.zeros(3, dtype=np.float32)
        policy_target_reference = policy.target_joint_positions()
        next_policy_time = data.time
        next_step_wall = time.perf_counter()
        render_interval = 1.0 / args.render_fps
        next_render_wall = next_step_wall
        next_print_time = data.time
        headless_jump_requested = False
        reset_count = 0
        status_note = ""
        fall_warmup_until = data.time + args.fall_warmup
        scan = scan_obstacles(model, data, angles, args)

        try:
            while True:
                now = time.perf_counter()
                if args.duration > 0.0 and data.time >= args.duration:
                    break
                if viewer is not None and not viewer.is_running():
                    break
                if now < next_step_wall:
                    time.sleep(min(next_step_wall - now, model.opt.timestep))
                    continue

                data.xfrc_applied[:] = 0.0

                if viewer is None:
                    raw_target_command = np.array(
                        [
                            args.test_command_vx,
                            args.test_command_vy,
                            args.test_command_yaw,
                        ],
                        dtype=np.float32,
                    )
                    active_motion = bool(np.linalg.norm(raw_target_command) > 1e-6)
                    jump_requested = (
                        args.test_jump_time >= 0.0
                        and not headless_jump_requested
                        and data.time >= args.test_jump_time
                    )
                    if jump_requested:
                        headless_jump_requested = True
                    dash_held = False
                else:
                    state = viewer.read_command_state(
                        args.normal_speed,
                        args.dash_forward_speed,
                        args.dash_backward_speed,
                        args.dash_lateral_speed,
                        args.yaw_speed,
                    )
                    raw_target_command = state.command
                    active_motion = state.active_motion
                    jump_requested = state.jump_requested
                    dash_held = state.dash_held
                    if state.stance_adjust:
                        stance_crouch = clamp_value(
                            stance_crouch
                            + state.stance_adjust * args.stance_adjust_step,
                            args.min_stance_crouch,
                            args.max_stance_crouch,
                        )
                        policy.set_stance_crouch(stance_crouch)
                        status_note = f"stance adjusted to {stance_crouch:.3f}"

                    if state.reset_requested:
                        reset_robot_pose(
                            policy,
                            jump,
                            data,
                            args.reset_base_height,
                            stance_crouch,
                        )
                        command[:] = 0.0
                        policy_target_reference = policy.target_joint_positions()
                        next_policy_time = data.time
                        next_step_wall = time.perf_counter()
                        fall_warmup_until = data.time + args.fall_warmup
                        reset_count += 1
                        status_note = "manual reset"
                        continue

                scan = scan_obstacles(model, data, angles, args)
                target_command, avoidance_active, avoidance_note = apply_obstacle_avoidance(
                    raw_target_command,
                    scan,
                    args,
                )
                target_command = limit_yaw_command(
                    target_command,
                    args.yaw_safety_limit,
                )
                active_motion = active_motion or avoidance_active
                overlay_note = avoidance_note if avoidance_active else status_note or avoidance_note

                command = update_command_with_release_cutoff(
                    command,
                    target_command,
                    model.opt.timestep,
                    args.command_smoothing,
                    active_motion,
                )

                if data.time >= next_policy_time:
                    policy.update_policy(data, command)
                    policy_target_reference = policy.target_joint_positions()
                    next_policy_time += policy.step_dt

                if not jump.apply_if_requested(
                    data,
                    jump_requested,
                    policy_target_reference,
                ):
                    idle_stabilizing = should_apply_idle_stabilization(
                        command,
                        active_motion,
                        jump,
                    )
                    policy.apply_pd(
                        data,
                        damping_scale=(
                            args.idle_damping_scale if idle_stabilizing else 1.0
                        ),
                    )
                    if idle_stabilizing:
                        apply_idle_base_damping(
                            data,
                            jump,
                            args.idle_base_damping,
                            args.idle_speed_deadband,
                            model.opt.timestep,
                        )

                mujoco.mj_step(model, data)

                if (
                    not args.no_auto_reset_on_fall
                    and data.time >= fall_warmup_until
                    and (reason := fall_reason(
                        data,
                        args.fall_height,
                        args.fall_uprightness,
                    ))
                ):
                    reset_robot_pose(
                        policy,
                        jump,
                        data,
                        args.reset_base_height,
                        stance_crouch,
                    )
                    command[:] = 0.0
                    policy_target_reference = policy.target_joint_positions()
                    next_policy_time = data.time
                    next_step_wall = time.perf_counter()
                    fall_warmup_until = data.time + args.fall_warmup
                    reset_count += 1
                    status_note = f"auto reset: {reason}"
                    continue

                next_step_wall += model.opt.timestep
                if next_step_wall < now - 0.1:
                    next_step_wall = now

                if args.print_interval > 0.0 and data.time >= next_print_time:
                    print_status(data, raw_target_command, command, scan, overlay_note)
                    next_print_time += args.print_interval

                if viewer is not None and now >= next_render_wall:
                    viewer.render_status(
                        data,
                        command,
                        dash_held,
                        jump.active,
                        stance_crouch,
                        reset_count,
                        overlay_note,
                    )
                    next_render_wall += render_interval
                    if next_render_wall < now - render_interval:
                        next_render_wall = now
        finally:
            data.xfrc_applied[:] = 0.0
            if viewer is not None:
                viewer.close()

        print(
            "Final base pose: "
            f"x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}, "
            f"uprightness={final_uprightness(data):+.3f}"
        )
    finally:
        runtime_scene.cleanup()


if __name__ == "__main__":
    main()
