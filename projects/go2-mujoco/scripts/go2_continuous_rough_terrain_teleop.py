from __future__ import annotations

import argparse
import os
import secrets
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
    MODEL_DIR,
    POLICY_DIR,
    FlatWasdDashViewer,
    JointSpaceJumpController,
    SimToRealPolicyController,
    apply_idle_base_damping,
    clamp_value,
    fall_reason,
    final_uprightness,
    limit_yaw_command,
    reset_robot_pose,
    should_apply_idle_stabilization,
    update_command_with_release_cutoff,
)


DEFAULT_TERRAIN_SEED = "random"
MAX_TERRAIN_SEED = 2**31 - 1
DEFAULT_TERRAIN_AMPLITUDE = 0.065
DEFAULT_TERRAIN_SMOOTHING_PASSES = 2
DEFAULT_TERRAIN_RESOLUTION = 0.25
DEFAULT_TERRAIN_X_HALF_SIZE = 18.0
DEFAULT_TERRAIN_Y_HALF_SIZE = 5.0
DEFAULT_START_FLAT_RADIUS = 0.65
TERRAIN_NAME = "continuous_rough_ground"


def terrain_grid_count(
    half_size: float,
    resolution: float = DEFAULT_TERRAIN_RESOLUTION,
) -> int:
    intervals = max(2, int(np.ceil((2.0 * half_size) / resolution)))
    return intervals + 1


DEFAULT_TERRAIN_ROWS = terrain_grid_count(DEFAULT_TERRAIN_Y_HALF_SIZE)
DEFAULT_TERRAIN_COLS = terrain_grid_count(DEFAULT_TERRAIN_X_HALF_SIZE)


def resolve_terrain_seed(raw_seed: object) -> tuple[int, bool]:
    if isinstance(raw_seed, str) and raw_seed.strip().lower() == "random":
        return secrets.randbelow(MAX_TERRAIN_SEED), True

    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError("Terrain seed must be an integer or 'random'.") from exc

    if seed < 0:
        raise ValueError("Terrain seed must be non-negative.")
    return seed, False


@dataclass
class RuntimeTerrainScene:
    temp_dir: tempfile.TemporaryDirectory
    scene_path: Path

    def cleanup(self) -> None:
        self.temp_dir.cleanup()


@dataclass(frozen=True)
class TerrainStats:
    x_half_size: float
    y_half_size: float
    rows: int
    cols: int
    min_height: float
    max_height: float
    max_neighbor_height_delta: float
    max_neighbor_slope_degrees: float


class ContinuousRoughTerrainViewer(FlatWasdDashViewer):
    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        if self._window is not None:
            self._glfw.set_window_title(
                self._window,
                "Go2 continuous rough-terrain teleop - WASD, QE, shift dash",
            )
        self._camera.distance = 2.7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "WASD/QE teleoperation for Go2 on a continuous rough heightfield. "
            "Neighboring surface patches share vertices, so the floor has "
            "changing local angles without disconnected tile steps."
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
    parser.add_argument(
        "--render-fps",
        type=float,
        default=DEFAULT_RENDER_FPS,
        help="Viewer refresh rate. Physics still runs at the MuJoCo timestep.",
    )
    parser.add_argument(
        "--terrain-seed",
        default=DEFAULT_TERRAIN_SEED,
        help="Terrain random seed, or 'random' for a fresh terrain each run.",
    )
    parser.add_argument(
        "--terrain-amplitude",
        type=float,
        default=DEFAULT_TERRAIN_AMPLITUDE,
        help="Maximum absolute surface height around z=0 in meters.",
    )
    parser.add_argument(
        "--terrain-smoothing-passes",
        type=int,
        default=DEFAULT_TERRAIN_SMOOTHING_PASSES,
        help="Local averaging passes. Lower values produce sharper angle changes.",
    )
    parser.add_argument(
        "--start-flat-radius",
        type=float,
        default=DEFAULT_START_FLAT_RADIUS,
        help="Flat radius around the initial pose before roughness ramps in.",
    )
    parser.add_argument(
        "--normal-speed",
        type=float,
        default=0.45,
        help="Normal WASD planar speed in m/s.",
    )
    parser.add_argument(
        "--dash-forward-speed",
        type=float,
        default=1.0,
        help="Shift+W dash speed limit in m/s.",
    )
    parser.add_argument(
        "--dash-backward-speed",
        type=float,
        default=0.65,
        help="Shift+S dash speed limit in m/s.",
    )
    parser.add_argument(
        "--dash-lateral-speed",
        type=float,
        default=0.65,
        help="Shift+A/D dash speed limit in m/s.",
    )
    parser.add_argument(
        "--yaw-speed",
        type=float,
        default=DEFAULT_YAW_SPEED,
        help="Q/E yaw rate command in rad/s.",
    )
    parser.add_argument(
        "--yaw-safety-limit",
        type=float,
        default=DEFAULT_YAW_SAFETY_LIMIT,
        help=(
            "Clamp commanded yaw rate before policy inference. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--command-smoothing",
        type=float,
        default=12.0,
        help="First-order smoothing rate while movement keys are held.",
    )
    parser.add_argument(
        "--reset-base-height",
        type=float,
        default=DEFAULT_RESET_BASE_HEIGHT,
        help="Base/root z height used when initializing or resetting the robot.",
    )
    parser.add_argument(
        "--stance-crouch",
        type=float,
        default=DEFAULT_STANCE_CROUCH,
        help=(
            "Low-stance joint bias in radians. Positive values crouch the legs "
            "by increasing thigh targets and decreasing calf targets."
        ),
    )
    parser.add_argument(
        "--min-stance-crouch",
        type=float,
        default=DEFAULT_MIN_STANCE_CROUCH,
        help="Lower bound for runtime Z/X stance adjustment.",
    )
    parser.add_argument(
        "--max-stance-crouch",
        type=float,
        default=DEFAULT_MAX_STANCE_CROUCH,
        help="Upper bound for runtime Z/X stance adjustment.",
    )
    parser.add_argument(
        "--stance-adjust-step",
        type=float,
        default=DEFAULT_STANCE_ADJUST_STEP,
        help="Runtime stance crouch change per Z/X key press.",
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=DEFAULT_FALL_HEIGHT,
        help="Auto-reset when base/root z drops below this height.",
    )
    parser.add_argument(
        "--fall-uprightness",
        type=float,
        default=DEFAULT_FALL_UPRIGHTNESS,
        help="Auto-reset when root z-axis uprightness drops below this value.",
    )
    parser.add_argument(
        "--fall-warmup",
        type=float,
        default=DEFAULT_FALL_WARMUP,
        help="Seconds after each reset before fall auto-reset checks start.",
    )
    parser.add_argument(
        "--no-auto-reset-on-fall",
        action="store_true",
        help="Disable automatic reset on fall. Manual R reset still works.",
    )
    parser.add_argument(
        "--idle-damping-scale",
        type=float,
        default=DEFAULT_IDLE_DAMPING_SCALE,
        help=(
            "Joint damping multiplier used only while command input is zero "
            "and the jump assist is inactive."
        ),
    )
    parser.add_argument(
        "--idle-base-damping",
        type=float,
        default=DEFAULT_IDLE_BASE_DAMPING,
        help=(
            "Light root velocity damping while stopped and feet are in contact. "
            "Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--idle-speed-deadband",
        type=float,
        default=DEFAULT_IDLE_SPEED_DEADBAND,
        help="Maximum planar speed where idle root damping is allowed.",
    )
    parser.add_argument(
        "--jump-duration",
        type=float,
        default=0.78,
        help="Seconds for one tap-triggered jump assist.",
    )
    parser.add_argument(
        "--jump-blend",
        type=float,
        default=0.95,
        help="Maximum blend from policy targets to the joint-space jump posture.",
    )
    parser.add_argument(
        "--test-command-vx",
        type=float,
        default=0.0,
        help="Headless/debug forward command.",
    )
    parser.add_argument(
        "--test-command-vy",
        type=float,
        default=0.0,
        help="Headless/debug lateral command.",
    )
    parser.add_argument(
        "--test-command-yaw",
        type=float,
        default=0.0,
        help="Headless/debug yaw command.",
    )
    parser.add_argument(
        "--test-jump-time",
        type=float,
        default=-1.0,
        help="Headless/debug option: trigger one jump tap at this sim time.",
    )
    parser.add_argument(
        "--test-jump-hold",
        type=float,
        default=0.35,
        help="Deprecated compatibility option. Jump is now tap-triggered.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration < 0.0:
        raise ValueError("Duration must be non-negative.")
    if args.render_fps <= 0.0:
        raise ValueError("Render FPS must be positive.")
    if args.terrain_amplitude <= 0.0:
        raise ValueError("Terrain amplitude must be positive.")
    if args.terrain_smoothing_passes < 0:
        raise ValueError("Terrain smoothing passes must be non-negative.")
    if args.start_flat_radius < 0.0:
        raise ValueError("Start flat radius must be non-negative.")
    if args.normal_speed < 0.0:
        raise ValueError("Normal speed must be non-negative.")
    if args.dash_forward_speed < 0.0:
        raise ValueError("Dash forward speed must be non-negative.")
    if args.dash_backward_speed < 0.0:
        raise ValueError("Dash backward speed must be non-negative.")
    if args.dash_lateral_speed < 0.0:
        raise ValueError("Dash lateral speed must be non-negative.")
    if args.yaw_speed < 0.0:
        raise ValueError("Yaw speed must be non-negative.")
    if args.yaw_safety_limit < 0.0:
        raise ValueError("Yaw safety limit must be non-negative.")
    if args.command_smoothing < 0.0:
        raise ValueError("Command smoothing must be non-negative.")
    if args.reset_base_height <= 0.0:
        raise ValueError("Reset base height must be positive.")
    if args.min_stance_crouch < 0.0:
        raise ValueError("Minimum stance crouch must be non-negative.")
    if args.max_stance_crouch < args.min_stance_crouch:
        raise ValueError("Maximum stance crouch must be >= minimum stance crouch.")
    if args.stance_adjust_step < 0.0:
        raise ValueError("Stance adjust step must be non-negative.")
    if args.fall_height <= 0.0:
        raise ValueError("Fall height must be positive.")
    if not -1.0 <= args.fall_uprightness <= 1.0:
        raise ValueError("Fall uprightness must be in [-1, 1].")
    if args.fall_warmup < 0.0:
        raise ValueError("Fall warmup must be non-negative.")
    if args.idle_damping_scale < 1.0:
        raise ValueError("Idle damping scale must be >= 1.")
    if args.idle_base_damping < 0.0:
        raise ValueError("Idle base damping must be non-negative.")
    if args.idle_speed_deadband < 0.0:
        raise ValueError("Idle speed deadband must be non-negative.")
    if args.jump_duration <= 0.0:
        raise ValueError("Jump duration must be positive.")
    if not 0.0 <= args.jump_blend <= 1.0:
        raise ValueError("Jump blend must be in [0, 1].")
    if args.test_jump_hold < 0.0:
        raise ValueError("Test jump hold must be non-negative.")


def create_runtime_scene(
    model_dir: Path,
    terrain_amplitude: float,
    terrain_x_half_size: float = DEFAULT_TERRAIN_X_HALF_SIZE,
    terrain_y_half_size: float = DEFAULT_TERRAIN_Y_HALF_SIZE,
    terrain_rows: int | None = None,
    terrain_cols: int | None = None,
) -> RuntimeTerrainScene:
    terrain_rows = terrain_rows or terrain_grid_count(terrain_y_half_size)
    terrain_cols = terrain_cols or terrain_grid_count(terrain_x_half_size)
    temp_dir = tempfile.TemporaryDirectory(prefix="go2_continuous_rough_terrain_")
    temp_path = Path(temp_dir.name)
    (temp_path / "assets").symlink_to(
        model_dir / "assets",
        target_is_directory=True,
    )
    (temp_path / "go2.xml").write_text(
        (model_dir / "go2.xml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    scene_path = temp_path / "continuous_rough_terrain_scene.xml"
    scene_path.write_text(
        build_scene_xml(
            terrain_amplitude,
            terrain_x_half_size,
            terrain_y_half_size,
            terrain_rows,
            terrain_cols,
        ),
        encoding="utf-8",
    )
    return RuntimeTerrainScene(temp_dir, scene_path)


def build_scene_xml(
    terrain_amplitude: float,
    terrain_x_half_size: float,
    terrain_y_half_size: float,
    terrain_rows: int,
    terrain_cols: int,
) -> str:
    terrain_base_depth = 0.22
    terrain_z_scale = 2.0 * terrain_amplitude
    terrain_z_offset = -terrain_amplitude
    return f"""<mujoco model="go2 continuous rough-terrain teleop">
  <include file="go2.xml" />

  <statistic center="0 0 0.1" extent="3.0" />

  <visual>
    <headlight diffuse="0.75 0.75 0.75" ambient="0.45 0.45 0.45" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="-130" elevation="-20" />
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7"
      rgb2="0 0 0" width="512" height="3072" />
    <texture type="2d" name="rough_ground_texture" builtin="checker"
      mark="edge" rgb1="0.23 0.31 0.38" rgb2="0.12 0.20 0.27"
      markrgb="0.75 0.78 0.82" width="300" height="300" />
    <material name="rough_ground_material" texture="rough_ground_texture"
      texuniform="true" texrepeat="16 10" reflectance="0.1" />
    <hfield name="{TERRAIN_NAME}" nrow="{terrain_rows}"
      ncol="{terrain_cols}"
      size="{terrain_x_half_size} {terrain_y_half_size}
      {terrain_z_scale:.6f} {terrain_base_depth:.6f}" />
  </asset>

  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" directional="true" />
    <geom name="continuous_rough_floor" type="hfield" hfield="{TERRAIN_NAME}"
      pos="0 0 {terrain_z_offset:.6f}" material="rough_ground_material"
      friction="0.8 0.02 0.01" />
  </worldbody>
</mujoco>
"""


def generate_continuous_heights(
    seed: int,
    smoothing_passes: int,
    start_flat_radius: float,
    x_half_size: float,
    y_half_size: float,
    rows: int,
    cols: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    heights = rng.uniform(-1.0, 1.0, size=(rows, cols))
    for _ in range(smoothing_passes):
        padded = np.pad(heights, 1, mode="edge")
        heights = (
            4.0 * padded[1:-1, 1:-1]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 8.0

    heights -= float(np.mean(heights))
    heights /= max(float(np.max(np.abs(heights))), 1e-9)

    x = np.linspace(
        -x_half_size,
        x_half_size,
        cols,
    )
    y = np.linspace(
        -y_half_size,
        y_half_size,
        rows,
    )
    xx, yy = np.meshgrid(x, y)
    distance = np.sqrt(xx**2 + yy**2)
    ramp_width = 0.8
    roughness_ramp = np.clip(
        (distance - start_flat_radius) / ramp_width,
        0.0,
        1.0,
    )
    return heights * roughness_ramp


def configure_heightfield(
    model: mujoco.MjModel,
    seed: int,
    smoothing_passes: int,
    start_flat_radius: float,
    terrain_amplitude: float,
) -> TerrainStats:
    hfield_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_HFIELD,
        TERRAIN_NAME,
    )
    if hfield_id < 0:
        raise RuntimeError(f"Heightfield was not found: {TERRAIN_NAME}")

    rows = int(model.hfield_nrow[hfield_id])
    cols = int(model.hfield_ncol[hfield_id])
    x_half_size = float(model.hfield_size[hfield_id, 0])
    y_half_size = float(model.hfield_size[hfield_id, 1])

    normalized_heights = generate_continuous_heights(
        seed,
        smoothing_passes,
        start_flat_radius,
        x_half_size,
        y_half_size,
        rows,
        cols,
    )
    data_adr = int(model.hfield_adr[hfield_id])
    data_size = rows * cols
    model.hfield_data[data_adr : data_adr + data_size] = (
        0.5 + 0.5 * normalized_heights.reshape(-1)
    )

    physical_heights = terrain_amplitude * normalized_heights
    x_resolution = 2.0 * x_half_size / (cols - 1)
    y_resolution = 2.0 * y_half_size / (rows - 1)
    x_steps = np.abs(np.diff(physical_heights, axis=1))
    y_steps = np.abs(np.diff(physical_heights, axis=0))
    max_neighbor_height_delta = max(float(np.max(x_steps)), float(np.max(y_steps)))
    max_neighbor_slope = max(
        float(np.max(np.arctan2(x_steps, x_resolution))),
        float(np.max(np.arctan2(y_steps, y_resolution))),
    )
    return TerrainStats(
        x_half_size=x_half_size,
        y_half_size=y_half_size,
        rows=rows,
        cols=cols,
        min_height=float(np.min(physical_heights)),
        max_height=float(np.max(physical_heights)),
        max_neighbor_height_delta=max_neighbor_height_delta,
        max_neighbor_slope_degrees=float(np.degrees(max_neighbor_slope)),
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    terrain_seed, terrain_seed_random = resolve_terrain_seed(args.terrain_seed)
    args.terrain_seed = terrain_seed
    if not args.headless:
        os.environ["DISPLAY"] = args.display

    runtime_scene = create_runtime_scene(MODEL_DIR, args.terrain_amplitude)
    viewer = None
    try:
        model = mujoco.MjModel.from_xml_path(str(runtime_scene.scene_path))
        terrain_stats = configure_heightfield(
            model,
            args.terrain_seed,
            args.terrain_smoothing_passes,
            args.start_flat_radius,
            args.terrain_amplitude,
        )
        data = mujoco.MjData(model)
        policy = SimToRealPolicyController(model, args.policy_dir)
        jump = JointSpaceJumpController(
            policy,
            args.jump_duration,
            args.jump_blend,
        )
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
        print(
            "Continuous rough terrain: "
            f"seed={args.terrain_seed}"
            f"{' (random)' if terrain_seed_random else ''}, "
            f"height=[{terrain_stats.min_height:+.3f}, "
            f"{terrain_stats.max_height:+.3f}]m, "
            f"max_neighbor_height_delta="
            f"{terrain_stats.max_neighbor_height_delta:.3f}m, "
            f"max_neighbor_slope={terrain_stats.max_neighbor_slope_degrees:.1f}deg"
        )
        print(
            "Controls: W/A/S/D move, Q/E turn, Shift+WASD dashes, "
            "Space taps jump."
        )
        print(
            "Command limits: "
            f"forward={args.dash_forward_speed:.2f} m/s, "
            f"backward={args.dash_backward_speed:.2f} m/s, "
            f"lateral={args.dash_lateral_speed:.2f} m/s, "
            f"yaw={args.yaw_speed:.2f} rad/s, "
            f"yaw_safety={args.yaw_safety_limit:.2f} rad/s"
        )
        print(
            "Posture: "
            f"reset_base_height={args.reset_base_height:.3f}m, "
            f"stance_crouch={stance_crouch:.3f}rad "
            f"(Z/X adjust by {args.stance_adjust_step:.3f})"
        )
        print(
            "Reset: "
            f"R manual reset, "
            f"auto_reset={'off' if args.no_auto_reset_on_fall else 'on'}, "
            f"fall_height<{args.fall_height:.2f}m, "
            f"uprightness<{args.fall_uprightness:.2f}"
        )

        if not args.headless:
            viewer = ContinuousRoughTerrainViewer(model)

        command = np.zeros(3, dtype=np.float32)
        policy_target_reference = policy.target_joint_positions()
        next_policy_time = data.time
        next_step_wall = time.perf_counter()
        render_interval = 1.0 / args.render_fps
        next_render_wall = next_step_wall
        headless_jump_requested = False
        reset_count = 0
        status_note = ""
        fall_warmup_until = data.time + args.fall_warmup

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
                target_command = np.array(
                    [
                        args.test_command_vx,
                        args.test_command_vy,
                        args.test_command_yaw,
                    ],
                    dtype=np.float32,
                )
                active_motion = bool(np.linalg.norm(target_command) > 1e-6)
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
                target_command = state.command
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

            target_command = limit_yaw_command(target_command, args.yaw_safety_limit)
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
            if viewer is not None and now >= next_render_wall:
                viewer.render_status(
                    data,
                    command,
                    dash_held,
                    jump.active,
                    stance_crouch,
                    reset_count,
                    status_note,
                )
                next_render_wall += render_interval
                if next_render_wall < now - render_interval:
                    next_render_wall = now

        print(
            "Final base pose: "
            f"x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, "
            f"z={data.qpos[2]:.3f}, uprightness={final_uprightness(data):+.3f}"
        )
    finally:
        if viewer is not None:
            viewer.close()
        runtime_scene.cleanup()


if __name__ == "__main__":
    main()
