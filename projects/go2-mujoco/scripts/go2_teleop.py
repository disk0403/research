import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "go2"
FLAT_ONLY_SCENE = MODEL_DIR / "scene_flat.xml"
POLICY_DIR = ROOT / "external" / "policies" / "unitree-go2-velocity-flat"
DEFAULT_DISPLAY = ":1"
DEFAULT_RENDER_FPS = 60.0
DEFAULT_YAW_SPEED = 0.5
DEFAULT_YAW_SAFETY_LIMIT = 1.0
DEFAULT_RESET_BASE_HEIGHT = 0.25
DEFAULT_STANCE_CROUCH = 0.08
DEFAULT_MIN_STANCE_CROUCH = 0.0
DEFAULT_MAX_STANCE_CROUCH = 0.20
DEFAULT_STANCE_ADJUST_STEP = 0.02
DEFAULT_FALL_HEIGHT = 0.16
DEFAULT_FALL_UPRIGHTNESS = 0.55
DEFAULT_FALL_WARMUP = 0.5
DEFAULT_IDLE_DAMPING_SCALE = 1.8
DEFAULT_IDLE_BASE_DAMPING = 2.0
DEFAULT_IDLE_SPEED_DEADBAND = 0.12

OFFICIAL_ABDUCTION_LIMIT = math.radians(48.0)
WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)
LEG_NAMES = ("FR", "FL", "RR", "RL")
FRONT_LEGS = frozenset(("FR", "FL"))
REAR_LEGS = frozenset(("RR", "RL"))


@dataclass(frozen=True)
class ActuatorBinding:
    actuator_id: int
    qpos_adr: int
    dof_adr: int
    ctrl_min: float
    ctrl_max: float


@dataclass
class CommandState:
    command: np.ndarray
    jump_requested: bool
    jump_held: bool
    dash_held: bool
    active_motion: bool
    reset_requested: bool = False
    stance_adjust: int = 0


def vector_from_config(
    value: object,
    length: int,
    name: str,
    default: np.ndarray | None = None,
) -> np.ndarray:
    if value is None:
        if default is None:
            raise ValueError(f"Missing required config value: {name}")
        return default.astype(np.float64, copy=True)

    if isinstance(value, (float, int)):
        return np.full(length, float(value), dtype=np.float64)

    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain {length} values, got {array.shape}")
    return array


def root_rotation_matrix(data: mujoco.MjData) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, data.qpos[3:7])
    return matrix.reshape(3, 3)


def update_smoothed_command(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
    rate: float,
) -> np.ndarray:
    if rate <= 0.0:
        return target.astype(np.float32, copy=True)

    alpha = 1.0 - math.exp(-rate * dt)
    return (current + alpha * (target - current)).astype(np.float32)


class Go2PolicyController:
    def __init__(self, model: mujoco.MjModel, policy_dir: Path) -> None:
        self._model = model
        self._policy_dir = Path(policy_dir)

        config_path = self._policy_dir / "params" / "deploy.yaml"
        model_path = self._policy_dir / "policy.onnx"
        if not config_path.exists():
            raise FileNotFoundError(f"Policy config not found: {config_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX policy not found: {model_path}")

        with config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        joint_ids_map = np.asarray(config["joint_ids_map"], dtype=np.int64)
        if joint_ids_map.shape != (12,):
            raise ValueError("joint_ids_map must contain 12 actuator ids.")
        self._bindings = [self._bind_actuator(int(i)) for i in joint_ids_map]

        self.step_dt = float(config.get("step_dt", 0.02))
        self._stiffness = vector_from_config(config.get("stiffness"), 12, "stiffness")
        self._damping = vector_from_config(config.get("damping"), 12, "damping")
        self._default_joint_pos = vector_from_config(
            config.get("default_joint_pos"),
            12,
            "default_joint_pos",
        )

        action_config = config.get("actions", {}).get("JointPositionAction", {})
        self._action_scale = vector_from_config(
            action_config.get("scale"),
            12,
            "actions.JointPositionAction.scale",
            default=np.ones(12, dtype=np.float64),
        )
        self._action_offset = vector_from_config(
            action_config.get("offset"),
            12,
            "actions.JointPositionAction.offset",
            default=self._default_joint_pos,
        )
        self._target_joint_pos = self._default_joint_pos.copy()
        self._last_action = np.zeros(12, dtype=np.float32)
        self._stance_crouch = 0.0

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _bind_actuator(self, actuator_id: int) -> ActuatorBinding:
        if actuator_id < 0 or actuator_id >= self._model.nu:
            raise ValueError(f"Invalid actuator id in joint_ids_map: {actuator_id}")

        joint_id = int(self._model.actuator_trnid[actuator_id, 0])
        return ActuatorBinding(
            actuator_id=actuator_id,
            qpos_adr=int(self._model.jnt_qposadr[joint_id]),
            dof_adr=int(self._model.jnt_dofadr[joint_id]),
            ctrl_min=float(self._model.actuator_ctrlrange[actuator_id, 0]),
            ctrl_max=float(self._model.actuator_ctrlrange[actuator_id, 1]),
        )

    def initialize_pose(
        self,
        data: mujoco.MjData,
        base_height: float = 0.27,
        stance_crouch: float | None = None,
    ) -> None:
        mujoco.mj_resetData(self._model, data)
        if stance_crouch is not None:
            self.set_stance_crouch(stance_crouch)

        data.qpos[0:3] = np.array([0.0, 0.0, base_height], dtype=np.float64)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qvel[:] = 0.0

        self._target_joint_pos[:] = self._default_joint_pos
        initial_joint_pos = self.target_joint_positions()
        for i, binding in enumerate(self._bindings):
            data.qpos[binding.qpos_adr] = initial_joint_pos[i]

        self._last_action[:] = 0.0
        mujoco.mj_forward(self._model, data)

    def update_policy(self, data: mujoco.MjData, command: np.ndarray) -> None:
        obs = self._build_observation(data, command).reshape(1, -1)
        raw_action = self._session.run(
            [self._output_name],
            {self._input_name: obs.astype(np.float32)},
        )[0]
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.shape != (12,):
            raise RuntimeError(f"Policy returned {action.shape}, expected 12 values.")

        self._last_action[:] = action
        self._target_joint_pos[:] = self._action_offset + self._action_scale * action

    def clamp_target_positions(self, target: np.ndarray) -> np.ndarray:
        return target

    def set_stance_crouch(self, stance_crouch: float) -> None:
        self._stance_crouch = max(0.0, float(stance_crouch))

    @property
    def stance_crouch(self) -> float:
        return self._stance_crouch

    def stance_offset(self) -> np.ndarray:
        offset = np.zeros(12, dtype=np.float64)
        for base in range(0, 12, 3):
            offset[base + 1] = self._stance_crouch
            offset[base + 2] = -2.0 * self._stance_crouch
        return offset

    def target_joint_positions(self) -> np.ndarray:
        return self.clamp_target_positions(self._target_joint_pos + self.stance_offset())

    def apply_pd(self, data: mujoco.MjData, damping_scale: float = 1.0) -> None:
        target_joint_pos = self.target_joint_positions()
        for i, binding in enumerate(self._bindings):
            q = data.qpos[binding.qpos_adr]
            dq = data.qvel[binding.dof_adr]
            tau = self._stiffness[i] * (target_joint_pos[i] - q)
            tau -= damping_scale * self._damping[i] * dq
            data.ctrl[binding.actuator_id] = np.clip(
                tau,
                binding.ctrl_min,
                binding.ctrl_max,
            )

    def _build_observation(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
    ) -> np.ndarray:
        rotation = root_rotation_matrix(data)
        # The exported policy expects MuJoCo's free-joint angular velocity
        # directly. Rotating it again makes combined walking/turning unstable.
        base_ang_vel = data.qvel[3:6]
        projected_gravity = rotation.T @ WORLD_GRAVITY
        joint_pos_rel = self._joint_positions(data) - self._default_joint_pos
        joint_vel_rel = self._joint_velocities(data)

        return np.concatenate(
            [
                base_ang_vel,
                projected_gravity,
                np.asarray(command, dtype=np.float64),
                joint_pos_rel,
                joint_vel_rel,
                self._last_action.astype(np.float64),
            ]
        ).astype(np.float32)

    def _joint_positions(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(
            [data.qpos[binding.qpos_adr] for binding in self._bindings],
            dtype=np.float64,
        )

    def _joint_velocities(self, data: mujoco.MjData) -> np.ndarray:
        return np.asarray(
            [data.qvel[binding.dof_adr] for binding in self._bindings],
            dtype=np.float64,
        )


class MouseKeyboardViewer:
    def __init__(self, model: mujoco.MjModel) -> None:
        import glfw

        self._glfw = glfw
        self._model = model
        self._window = None
        self._last_cursor: tuple[float, float] | None = None
        self._space_was_down = False
        self._reset_was_down = False
        self._lower_was_down = False
        self._raise_was_down = False

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        self._window = glfw.create_window(
            1280,
            720,
            "Go2 teleop",
            None,
            None,
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        glfw.make_context_current(self._window)
        glfw.swap_interval(0)
        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_scroll_callback(self._window, self._scroll_callback)

        self._camera = mujoco.MjvCamera()
        self._camera.azimuth = -130.0
        self._camera.elevation = -20.0
        self._camera.distance = 2.0
        self._camera.lookat[:] = np.array([0.0, 0.0, 0.25])
        self._option = mujoco.MjvOption()
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._context = mujoco.MjrContext(
            model,
            mujoco.mjtFontScale.mjFONTSCALE_150.value,
        )

    def is_running(self) -> bool:
        return self._window is not None and not self._glfw.window_should_close(
            self._window
        )

    def is_key_down(self, key: int) -> bool:
        if self._window is None:
            return False

        state = self._glfw.get_key(self._window, key)
        return state in (self._glfw.PRESS, self._glfw.REPEAT)

    def close(self) -> None:
        if getattr(self, "_context", None) is not None:
            self._context.free()
        if self._window is not None:
            self._glfw.destroy_window(self._window)
            self._window = None
        self._glfw.terminate()

    def _mouse_button_callback(self, window, button, action, mods) -> None:
        del button, action, mods
        self._last_cursor = self._glfw.get_cursor_pos(window)

    def _cursor_pos_callback(self, window, xpos: float, ypos: float) -> None:
        if self._last_cursor is None:
            self._last_cursor = (xpos, ypos)
            return

        last_x, last_y = self._last_cursor
        self._last_cursor = (xpos, ypos)
        dx = xpos - last_x
        dy = ypos - last_y

        left = self._glfw.get_mouse_button(window, self._glfw.MOUSE_BUTTON_LEFT)
        right = self._glfw.get_mouse_button(window, self._glfw.MOUSE_BUTTON_RIGHT)
        middle = self._glfw.get_mouse_button(window, self._glfw.MOUSE_BUTTON_MIDDLE)
        if left == self._glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        elif right == self._glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        elif middle == self._glfw.PRESS:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
        else:
            return

        _, height = self._glfw.get_window_size(window)
        height = max(height, 1)
        mujoco.mjv_moveCamera(
            self._model,
            action,
            dx / height,
            dy / height,
            self._scene,
            self._camera,
        )

    def _scroll_callback(self, window, xoffset: float, yoffset: float) -> None:
        del window, xoffset
        mujoco.mjv_moveCamera(
            self._model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self._scene,
            self._camera,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Flat-ground WASD/QE teleoperation for Go2 with shift dash and a "
            "joint-space jump assist."
        )
    )
    parser.add_argument("--display", default=DEFAULT_DISPLAY)
    parser.add_argument(
        "--scene",
        type=Path,
        default=FLAT_ONLY_SCENE,
        help="Flat MuJoCo scene to load. Defaults to the flat-only Go2 scene.",
    )
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
        "--normal-speed",
        type=float,
        default=0.55,
        help="Normal WASD planar speed in m/s.",
    )
    parser.add_argument(
        "--dash-forward-speed",
        type=float,
        default=2.0,
        help="Shift+W dash speed limit in m/s. Default matches the policy range.",
    )
    parser.add_argument(
        "--dash-backward-speed",
        type=float,
        default=1.0,
        help="Shift+S dash speed limit in m/s. Default matches the policy range.",
    )
    parser.add_argument(
        "--dash-lateral-speed",
        type=float,
        default=1.0,
        help="Shift+A/D dash speed limit in m/s. Default matches the policy range.",
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
            "Clamp commanded yaw rate before policy inference. Use 0 to disable "
            "this guard. The default matches the policy command range."
        ),
    )
    parser.add_argument(
        "--command-smoothing",
        type=float,
        default=12.0,
        help=(
            "First-order smoothing rate while keys are held. Release still cuts "
            "the command to zero immediately. Use 0 for fully direct input."
        ),
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


class SimToRealPolicyController(Go2PolicyController):
    def __init__(self, model: mujoco.MjModel, policy_dir: Path) -> None:
        super().__init__(model, policy_dir)
        self._target_min, self._target_max = self._build_target_limits()

    def clamp_target_positions(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self._target_min, self._target_max)

    def apply_pd(self, data: mujoco.MjData, damping_scale: float = 1.0) -> None:
        super().apply_pd(data, damping_scale=damping_scale)

    def _build_target_limits(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []

        for binding in self._bindings:
            joint_id = int(self._model.actuator_trnid[binding.actuator_id, 0])
            if self._model.jnt_limited[joint_id]:
                low = float(self._model.jnt_range[joint_id, 0])
                high = float(self._model.jnt_range[joint_id, 1])
            else:
                low = -np.inf
                high = np.inf

            joint_name = mujoco.mj_id2name(
                self._model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_id,
            )
            if joint_name is not None and joint_name.endswith("_hip_joint"):
                low = max(low, -OFFICIAL_ABDUCTION_LIMIT)
                high = min(high, OFFICIAL_ABDUCTION_LIMIT)

            lower.append(low)
            upper.append(high)

        return (
            np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64),
        )


class JointSpaceJumpController:
    def __init__(
        self,
        policy: SimToRealPolicyController,
        jump_duration: float,
        jump_blend: float,
    ) -> None:
        if jump_duration <= 0.0:
            raise ValueError("Jump duration must be positive.")

        self._policy = policy
        self._bindings = policy._bindings
        self._default = policy._default_joint_pos.astype(np.float64)
        self._leg_bases = self._resolve_leg_bases()
        self._foot_geom_ids = self._resolve_named_ids(mujoco.mjtObj.mjOBJ_GEOM)
        self._foot_body_ids = self._resolve_named_ids(
            mujoco.mjtObj.mjOBJ_BODY,
            suffix="_foot",
        )
        self._jump_duration = jump_duration
        self._jump_blend = max(0.0, min(jump_blend, 1.0))
        self._active = False
        self._recovering = False
        self._start_time = 0.0
        self._start_targets = self._default.copy()
        self._active_duration = jump_duration
        self._active_blend = self._jump_blend
        self._motion_factor = 0.0
        self._last_motion_time: float | None = None
        self._last_planar_velocity = np.zeros(2, dtype=np.float64)
        self._last_jump_target = self._default.copy()
        self._recovery_start_time = 0.0
        self._recovery_duration = 0.24
        self._recovery_start_targets = self._default.copy()

        min_stiffness = np.tile([32.0, 56.0, 90.0], 4)
        min_damping = np.tile([2.6, 4.4, 6.4], 4)
        self._base_stiffness = np.maximum(
            policy._stiffness.astype(np.float64) * 2.2,
            min_stiffness,
        )
        self._base_damping = np.maximum(
            policy._damping.astype(np.float64) * 2.6,
            min_damping,
        )
        self._stiffness = self._base_stiffness.copy()
        self._damping = self._base_damping.copy()

        self._static_compress = self._posture(thigh=1.12, calf=-2.22)
        self._static_extend = self._posture(thigh=0.42, calf=-0.94)
        self._static_air = self._posture(thigh=0.64, calf=-1.28)
        self._static_landing = self._posture(thigh=1.00, calf=-2.00)
        (
            self._moving_compress,
            self._moving_extend,
            self._moving_air,
            self._moving_landing,
        ) = self._build_moving_targets(True, {leg: 1.0 for leg in LEG_NAMES})
        self._compress = self._static_compress.copy()
        self._extend = self._static_extend.copy()
        self._air = self._static_air.copy()
        self._landing = self._static_landing.copy()

    @property
    def active(self) -> bool:
        return self._active or self._recovering

    def cancel(self) -> None:
        self._active = False
        self._recovering = False
        self._last_motion_time = None
        self._last_planar_velocity[:] = 0.0

    def foot_contact_count(self, data: mujoco.MjData) -> int:
        return sum(score >= 0.5 for score in self._foot_contact_scores(data).values())

    def apply_if_requested(
        self,
        data: mujoco.MjData,
        jump_requested: bool,
        policy_target: np.ndarray,
    ) -> bool:
        planar_speed, planar_accel = self._update_motion_estimate(data)

        if jump_requested and not self.active:
            self._active = True
            self._recovering = False
            self._start_time = data.time
            self._start_targets = np.array(
                [data.qpos[b.qpos_adr] for b in self._bindings],
                dtype=np.float64,
            )
            self._configure_profile(data, planar_speed, planar_accel)

        if not self._active:
            return self._apply_recovery(data, policy_target)

        elapsed = data.time - self._start_time
        if elapsed >= self._active_duration:
            self._begin_recovery(data)
            return self._apply_recovery(data, policy_target)

        jump_target, blend, stiffness_scale, damping_scale = self._target_for_time(
            data,
            elapsed
        )
        target = (1.0 - blend) * policy_target + blend * jump_target
        self._last_jump_target = target.copy()
        self._apply_joint_targets(
            data,
            self._policy.clamp_target_positions(target),
            stiffness_scale,
            damping_scale,
        )
        return True

    def apply_if_held(
        self,
        data: mujoco.MjData,
        jump_held: bool,
        policy_target: np.ndarray,
    ) -> bool:
        return self.apply_if_requested(data, jump_held, policy_target)

    def _resolve_leg_bases(self) -> dict[str, int]:
        bases: dict[str, int] = {}
        for index, binding in enumerate(self._bindings):
            actuator_name = mujoco.mj_id2name(
                self._policy._model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                binding.actuator_id,
            )
            if actuator_name is None:
                continue
            for leg in LEG_NAMES:
                if actuator_name == f"{leg}_hip":
                    bases[leg] = index

        if set(bases) == set(LEG_NAMES):
            return bases

        return {"FR": 0, "FL": 3, "RR": 6, "RL": 9}

    def _resolve_named_ids(
        self,
        obj_type: mujoco.mjtObj,
        suffix: str = "",
    ) -> dict[str, int]:
        ids: dict[str, int] = {}
        for leg in LEG_NAMES:
            name = f"{leg}{suffix}"
            obj_id = mujoco.mj_name2id(self._policy._model, obj_type, name)
            if obj_id >= 0:
                ids[leg] = int(obj_id)
        return ids

    def _posture(self, thigh: float, calf: float) -> np.ndarray:
        target = self._default.copy()
        for leg in LEG_NAMES:
            self._set_leg_posture(target, leg, thigh, calf)
        return self._policy.clamp_target_positions(target)

    def _set_leg_posture(
        self,
        target: np.ndarray,
        leg: str,
        thigh: float,
        calf: float,
    ) -> None:
        base = self._leg_bases[leg]
        target[base + 1] = thigh
        target[base + 2] = calf

    def _per_leg_posture(self, postures: dict[str, tuple[float, float]]) -> np.ndarray:
        target = self._default.copy()
        for leg, (thigh, calf) in postures.items():
            self._set_leg_posture(target, leg, thigh, calf)
        return self._policy.clamp_target_positions(target)

    def _configure_profile(
        self,
        data: mujoco.MjData,
        planar_speed: float,
        planar_accel: float,
    ) -> None:
        speed_factor = min(planar_speed / 1.1, 1.0)
        accel_factor = min(planar_accel / 3.0, 1.0)
        self._motion_factor = min(0.75 * speed_factor + 0.25 * accel_factor, 1.0)
        moving_forward = float(data.qvel[0]) >= -0.05
        contact_scores = self._foot_contact_scores(data)
        (
            self._moving_compress,
            self._moving_extend,
            self._moving_air,
            self._moving_landing,
        ) = self._build_moving_targets(moving_forward, contact_scores)

        alpha = self._motion_factor
        self._compress = self._blend_posture(
            self._static_compress,
            self._moving_compress,
            alpha,
        )
        self._extend = self._blend_posture(
            self._static_extend,
            self._moving_extend,
            alpha,
        )
        self._air = self._blend_posture(self._static_air, self._moving_air, alpha)
        self._landing = self._blend_posture(
            self._static_landing,
            self._moving_landing,
            alpha,
        )
        self._active_duration = self._jump_duration * (1.0 + 0.08 * alpha)
        self._active_blend = min(self._jump_blend, 0.93) * (1.0 - 0.10 * alpha)

    def _build_moving_targets(
        self,
        moving_forward: bool,
        contact_scores: dict[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        push_legs = REAR_LEGS if moving_forward else FRONT_LEGS
        landing_legs = FRONT_LEGS if moving_forward else REAR_LEGS

        compress: dict[str, tuple[float, float]] = {}
        extend: dict[str, tuple[float, float]] = {}
        air: dict[str, tuple[float, float]] = {}
        landing: dict[str, tuple[float, float]] = {}

        for leg in LEG_NAMES:
            contact = contact_scores.get(leg, 0.0)
            support = 0.45 + 0.55 * contact
            if leg in push_legs:
                compress[leg] = (
                    0.88 + 0.23 * support,
                    -1.72 - 0.47 * support,
                )
                extend[leg] = (
                    0.76 - 0.34 * support,
                    -1.48 + 0.55 * support,
                )
                air[leg] = (0.86, -1.66)
                landing[leg] = (0.92, -1.82)
            else:
                compress[leg] = (0.82, -1.58)
                extend[leg] = (0.78, -1.48)
                air[leg] = (0.62, -1.24) if leg in landing_legs else (0.82, -1.60)
                landing[leg] = (
                    (1.02, -2.02) if leg in landing_legs else (0.90, -1.78)
                )

        return (
            self._per_leg_posture(compress),
            self._per_leg_posture(extend),
            self._per_leg_posture(air),
            self._per_leg_posture(landing),
        )

    def _blend_posture(
        self,
        static_target: np.ndarray,
        moving_target: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        target = (1.0 - alpha) * static_target + alpha * moving_target
        return self._policy.clamp_target_positions(target)

    def _update_motion_estimate(self, data: mujoco.MjData) -> tuple[float, float]:
        planar_velocity = np.asarray(data.qvel[0:2], dtype=np.float64)
        speed = float(np.linalg.norm(planar_velocity))
        if self._last_motion_time is None:
            accel = 0.0
        else:
            dt = max(float(data.time - self._last_motion_time), 1e-6)
            accel = float(
                np.linalg.norm(planar_velocity - self._last_planar_velocity) / dt
            )

        self._last_motion_time = float(data.time)
        self._last_planar_velocity[:] = planar_velocity
        return speed, accel

    def _foot_contact_scores(self, data: mujoco.MjData) -> dict[str, float]:
        scores = {leg: 0.0 for leg in LEG_NAMES}
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            for leg, geom_id in self._foot_geom_ids.items():
                if contact.geom1 == geom_id or contact.geom2 == geom_id:
                    if contact.dist < 0.025:
                        scores[leg] = 1.0

        missing_contact = all(score == 0.0 for score in scores.values())
        if missing_contact and self._foot_body_ids:
            heights = {
                leg: float(data.xpos[body_id, 2])
                for leg, body_id in self._foot_body_ids.items()
            }
            if heights:
                lowest = min(heights.values())
                for leg, height in heights.items():
                    if height <= lowest + 0.035:
                        scores[leg] = max(scores[leg], 0.65)

        return scores

    def _target_for_time(
        self,
        data: mujoco.MjData,
        elapsed: float,
    ) -> tuple[np.ndarray, float, float, float]:
        compress_end = 0.16 * self._active_duration
        push_end = 0.34 * self._active_duration
        air_end = 0.68 * self._active_duration

        if elapsed < compress_end:
            phase = smootherstep(elapsed / max(compress_end, 1e-6))
            target = (1.0 - phase) * self._start_targets + phase * self._compress
            blend = self._active_blend * 0.72 * phase
            target = self._apply_pitch_stabilization(data, target, 0.35 * phase)
            return target, blend, 0.98, 1.25

        if elapsed < push_end:
            phase = smootherstep(
                (elapsed - compress_end) / max(push_end - compress_end, 1e-6)
            )
            target = (1.0 - phase) * self._compress + phase * self._extend
            blend = self._active_blend * (0.72 + 0.28 * phase)
            target = self._apply_pitch_stabilization(data, target, 0.85)
            return target, blend, 1.08 - 0.08 * self._motion_factor, 1.02

        if elapsed < air_end:
            phase = smootherstep(
                (elapsed - push_end) / max(air_end - push_end, 1e-6)
            )
            target = (1.0 - phase) * self._extend + phase * self._air
            target = self._apply_pitch_stabilization(data, target, 1.0)
            return target, self._active_blend, 0.86, 1.20

        phase = smootherstep(
            (elapsed - air_end) / max(self._active_duration - air_end, 1e-6)
        )
        target = (1.0 - phase) * self._air + phase * self._landing
        blend = self._active_blend * (
            1.0 - (0.56 - 0.10 * self._motion_factor) * phase
        )
        target = self._apply_pitch_stabilization(data, target, 0.70 * (1.0 - phase))
        return target, blend, 0.82, 1.55 + 0.15 * self._motion_factor

    def _apply_pitch_stabilization(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        if strength <= 0.0:
            return target

        pitch_degrees, pitch_rate_degrees = self._base_pitch_and_rate(data)
        correction = -0.0045 * pitch_degrees - 0.0012 * pitch_rate_degrees
        correction = clamp_value(correction * strength, -0.10, 0.10)
        if abs(correction) < 1e-5:
            return target

        stabilized = target.copy()
        for leg in FRONT_LEGS:
            base = self._leg_bases[leg]
            stabilized[base + 1] -= correction
            stabilized[base + 2] += 1.55 * correction
        for leg in REAR_LEGS:
            base = self._leg_bases[leg]
            stabilized[base + 1] += 0.75 * correction
            stabilized[base + 2] -= 1.20 * correction
        return self._policy.clamp_target_positions(stabilized)

    def _base_pitch_and_rate(self, data: mujoco.MjData) -> tuple[float, float]:
        rotation = root_rotation_matrix(data)
        pitch = math.atan2(
            -rotation[2, 0],
            math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2),
        )
        body_ang_vel = rotation.T @ data.qvel[3:6]
        return math.degrees(pitch), math.degrees(float(body_ang_vel[1]))

    def _begin_recovery(self, data: mujoco.MjData) -> None:
        self._active = False
        self._recovering = True
        self._recovery_start_time = float(data.time)
        self._recovery_start_targets = self._last_jump_target.copy()

    def _apply_recovery(
        self,
        data: mujoco.MjData,
        policy_target: np.ndarray,
    ) -> bool:
        if not self._recovering:
            return False

        elapsed = float(data.time - self._recovery_start_time)
        if elapsed >= self._recovery_duration:
            self._recovering = False
            return False

        phase = smootherstep(elapsed / max(self._recovery_duration, 1e-6))
        target = (1.0 - phase) * self._recovery_start_targets + phase * policy_target
        target = self._apply_pitch_stabilization(data, target, 0.45 * (1.0 - phase))
        self._apply_joint_targets(
            data,
            self._policy.clamp_target_positions(target),
            0.78,
            1.65,
        )
        return True

    def _apply_joint_targets(
        self,
        data: mujoco.MjData,
        target: np.ndarray,
        stiffness_scale: float,
        damping_scale: float,
    ) -> None:
        for i, binding in enumerate(self._bindings):
            q = data.qpos[binding.qpos_adr]
            dq = data.qvel[binding.dof_adr]
            tau = stiffness_scale * self._stiffness[i] * (target[i] - q)
            tau -= damping_scale * self._damping[i] * dq
            data.ctrl[binding.actuator_id] = np.clip(
                tau,
                binding.ctrl_min,
                binding.ctrl_max,
            )


class FlatWasdDashViewer(MouseKeyboardViewer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self._window is not None:
            self._glfw.set_window_title(
                self._window,
                "Go2 flat teleop - WASD, QE, shift dash",
            )

    def read_command_state(
        self,
        normal_speed: float,
        dash_forward_speed: float,
        dash_backward_speed: float,
        dash_lateral_speed: float,
        yaw_speed: float,
    ) -> CommandState:
        if self._window is None:
            return CommandState(
                np.zeros(3, dtype=np.float32),
                False,
                False,
                False,
                False,
            )

        glfw = self._glfw
        if self.is_key_down(glfw.KEY_ESCAPE):
            glfw.set_window_should_close(self._window, True)

        reset_held = self.is_key_down(glfw.KEY_R)
        reset_requested = reset_held and not self._reset_was_down
        self._reset_was_down = reset_held

        lower_held = self.is_key_down(glfw.KEY_Z)
        raise_held = self.is_key_down(glfw.KEY_X)
        lower_requested = lower_held and not self._lower_was_down
        raise_requested = raise_held and not self._raise_was_down
        self._lower_was_down = lower_held
        self._raise_was_down = raise_held
        stance_adjust = int(lower_requested) - int(raise_requested)

        vx_key = float(self.is_key_down(glfw.KEY_W)) - float(
            self.is_key_down(glfw.KEY_S)
        )
        vy_key = float(self.is_key_down(glfw.KEY_A)) - float(
            self.is_key_down(glfw.KEY_D)
        )
        yaw_key = float(self.is_key_down(glfw.KEY_Q)) - float(
            self.is_key_down(glfw.KEY_E)
        )
        dash_held = self.is_key_down(glfw.KEY_LEFT_SHIFT) or self.is_key_down(
            glfw.KEY_RIGHT_SHIFT
        )
        jump_held = self.is_key_down(glfw.KEY_SPACE)
        jump_requested = jump_held and not self._space_was_down
        self._space_was_down = jump_held

        planar = np.array([vx_key, vy_key], dtype=np.float64)
        norm = float(np.linalg.norm(planar))
        if norm > 1.0:
            planar /= norm
            norm = 1.0

        if norm > 1e-6 and dash_held:
            speed = directional_dash_speed(
                planar,
                dash_forward_speed,
                dash_backward_speed,
                dash_lateral_speed,
            )
        else:
            speed = normal_speed

        command = np.array(
            [
                speed * planar[0],
                speed * planar[1],
                yaw_speed * yaw_key,
            ],
            dtype=np.float32,
        )
        active_motion = norm > 1e-6 or abs(yaw_key) > 0.0
        return CommandState(
            command,
            jump_requested,
            jump_held,
            dash_held and norm > 1e-6,
            active_motion,
            reset_requested,
            stance_adjust,
        )

    def render_status(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        dash_held: bool,
        jump_active: bool,
        stance_crouch: float = 0.0,
        reset_count: int = 0,
        status_note: str = "",
    ) -> None:
        if self._window is None:
            return

        self._glfw.make_context_current(self._window)
        width, height = self._glfw.get_framebuffer_size(self._window)
        if width <= 0 or height <= 0:
            self._glfw.poll_events()
            return

        self._camera.lookat[:] = data.qpos[:3] + np.array([0.0, 0.0, 0.05])
        viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjv_updateScene(
            self._model,
            data,
            self._option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            (
                "W/A/S/D: move   Q/E: turn   Shift+WASD: dash   Space: jump\n"
                "R: reset   Z/X: lower/raise stance\n"
                "Left drag: rotate camera   Wheel: zoom   Esc: quit"
            ),
            (
                f"vx={command[0]:+.2f}  vy={command[1]:+.2f}  "
                f"yaw={command[2]:+.2f}  "
                f"dash={'on' if dash_held else 'off'}  "
                f"jump={'on' if jump_active else 'off'}\n"
                f"stance_crouch={stance_crouch:.3f}  "
                f"resets={reset_count}  "
                f"{status_note}"
            ),
            self._context,
        )

        self._glfw.swap_buffers(self._window)
        self._glfw.poll_events()


def directional_dash_speed(
    direction: np.ndarray,
    forward_limit: float,
    backward_limit: float,
    lateral_limit: float,
) -> float:
    constraints = []
    if direction[0] > 1e-6:
        constraints.append(max(0.0, forward_limit) / direction[0])
    elif direction[0] < -1e-6:
        constraints.append(max(0.0, backward_limit) / -direction[0])

    if abs(direction[1]) > 1e-6:
        constraints.append(max(0.0, lateral_limit) / abs(direction[1]))

    if not constraints:
        return 0.0
    return min(constraints)


def update_command_with_release_cutoff(
    current: np.ndarray,
    target: np.ndarray,
    dt: float,
    rate: float,
    active_motion: bool,
) -> np.ndarray:
    if not active_motion:
        return np.zeros(3, dtype=np.float32)
    return update_smoothed_command(current, target, dt, rate)


def limit_yaw_command(command: np.ndarray, yaw_safety_limit: float) -> np.ndarray:
    limited = command.astype(np.float32, copy=True)
    if yaw_safety_limit > 0.0:
        limited[2] = np.float32(
            clamp_value(float(limited[2]), -yaw_safety_limit, yaw_safety_limit)
        )
    return limited


def should_apply_idle_stabilization(
    command: np.ndarray,
    active_motion: bool,
    jump: JointSpaceJumpController,
) -> bool:
    return (
        not active_motion
        and not jump.active
        and float(np.linalg.norm(command)) < 1e-4
    )


def apply_idle_base_damping(
    data: mujoco.MjData,
    jump: JointSpaceJumpController,
    damping: float,
    speed_deadband: float,
    timestep: float,
) -> None:
    if damping <= 0.0 or jump.foot_contact_count(data) < 3:
        return

    planar_speed = float(np.linalg.norm(data.qvel[0:2]))
    yaw_speed = abs(float(data.qvel[5]))
    factor = math.exp(-damping * timestep)
    if planar_speed <= speed_deadband:
        data.qvel[0:2] *= factor
    if yaw_speed <= 2.0 * speed_deadband:
        data.qvel[5] *= factor


def smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def final_uprightness(data: mujoco.MjData) -> float:
    return float(root_rotation_matrix(data)[2, 2])


def clamp_value(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def reset_robot_pose(
    policy: SimToRealPolicyController,
    jump: JointSpaceJumpController,
    data: mujoco.MjData,
    base_height: float,
    stance_crouch: float,
) -> None:
    sim_time = float(data.time)
    policy.initialize_pose(
        data,
        base_height=base_height,
        stance_crouch=stance_crouch,
    )
    data.time = sim_time
    jump.cancel()


def fall_reason(
    data: mujoco.MjData,
    fall_height: float,
    fall_uprightness: float,
) -> str:
    if float(data.qpos[2]) < fall_height:
        return "fall_height"
    if final_uprightness(data) < fall_uprightness:
        return "fall_uprightness"
    return ""


def validate_args(args: argparse.Namespace) -> None:
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
    if getattr(args, "yaw_safety_limit", 0.0) < 0.0:
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
    if args.render_fps <= 0.0:
        raise ValueError("Render FPS must be positive.")
    if args.test_jump_hold < 0.0:
        raise ValueError("Test jump hold must be non-negative.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.headless:
        os.environ["DISPLAY"] = args.display

    model = mujoco.MjModel.from_xml_path(str(args.scene))
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

    print(f"Scene: {args.scene}")
    print(f"Policy: {args.policy_dir}")
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

    viewer = None
    if not args.headless:
        viewer = FlatWasdDashViewer(model)

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
    finally:
        data.xfrc_applied[:] = 0.0
        if viewer is not None:
            viewer.close()

    print(
        "Final base pose: "
        f"x={data.qpos[0]:.3f}, y={data.qpos[1]:.3f}, z={data.qpos[2]:.3f}, "
        f"uprightness={final_uprightness(data):+.3f}"
    )


if __name__ == "__main__":
    main()
