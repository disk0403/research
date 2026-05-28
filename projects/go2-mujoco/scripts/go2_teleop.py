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

OFFICIAL_ABDUCTION_LIMIT = math.radians(48.0)
WORLD_GRAVITY = np.array([0.0, 0.0, -1.0], dtype=np.float64)


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
    jump_held: bool
    dash_held: bool
    active_motion: bool


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

    def initialize_pose(self, data: mujoco.MjData) -> None:
        mujoco.mj_resetData(self._model, data)
        data.qpos[0:3] = np.array([0.0, 0.0, 0.27], dtype=np.float64)
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        data.qvel[:] = 0.0

        for i, binding in enumerate(self._bindings):
            data.qpos[binding.qpos_adr] = self._default_joint_pos[i]

        self._target_joint_pos[:] = self._default_joint_pos
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

    def apply_pd(self, data: mujoco.MjData) -> None:
        for i, binding in enumerate(self._bindings):
            q = data.qpos[binding.qpos_adr]
            dq = data.qvel[binding.dof_adr]
            tau = self._stiffness[i] * (self._target_joint_pos[i] - q)
            tau -= self._damping[i] * dq
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
        base_ang_vel = rotation.T @ data.qvel[3:6]
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
        default=0.5,
        help="Q/E yaw rate command in rad/s.",
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
        "--jump-duration",
        type=float,
        default=0.72,
        help="Maximum seconds for one held space-key jump assist.",
    )
    parser.add_argument(
        "--jump-blend",
        type=float,
        default=0.88,
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
        help="Headless/debug option: hold the jump key from this sim time.",
    )
    parser.add_argument(
        "--test-jump-hold",
        type=float,
        default=0.35,
        help="Headless/debug seconds to hold the jump key.",
    )
    return parser.parse_args()


class SimToRealPolicyController(Go2PolicyController):
    def __init__(self, model: mujoco.MjModel, policy_dir: Path) -> None:
        super().__init__(model, policy_dir)
        self._target_min, self._target_max = self._build_target_limits()

    def clamp_target_positions(self, target: np.ndarray) -> np.ndarray:
        return np.clip(target, self._target_min, self._target_max)

    def apply_pd(self, data: mujoco.MjData) -> None:
        self._target_joint_pos[:] = self.clamp_target_positions(
            self._target_joint_pos
        )
        super().apply_pd(data)

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
        self._jump_duration = jump_duration
        self._jump_blend = max(0.0, min(jump_blend, 1.0))
        self._active = False
        self._release_required = False
        self._start_time = 0.0
        self._start_targets = self._default.copy()

        min_stiffness = np.tile([26.0, 42.0, 68.0], 4)
        min_damping = np.tile([1.8, 2.8, 4.0], 4)
        self._stiffness = np.maximum(
            policy._stiffness.astype(np.float64) * 2.1,
            min_stiffness,
        )
        self._damping = np.maximum(
            policy._damping.astype(np.float64) * 2.0,
            min_damping,
        )

        self._compress = self._default.copy()
        self._extend = self._default.copy()
        self._air = self._default.copy()
        self._landing = self._default.copy()
        for base in range(0, 12, 3):
            self._compress[base + 1] = 1.03
            self._compress[base + 2] = -2.04
            self._extend[base + 1] = 0.52
            self._extend[base + 2] = -1.04
            self._air[base + 1] = 0.76
            self._air[base + 2] = -1.44
            self._landing[base + 1] = 0.94
            self._landing[base + 2] = -1.92

        self._compress = policy.clamp_target_positions(self._compress)
        self._extend = policy.clamp_target_positions(self._extend)
        self._air = policy.clamp_target_positions(self._air)
        self._landing = policy.clamp_target_positions(self._landing)

    @property
    def active(self) -> bool:
        return self._active

    def apply_if_held(
        self,
        data: mujoco.MjData,
        jump_held: bool,
        policy_target: np.ndarray,
    ) -> bool:
        if not jump_held:
            self._active = False
            self._release_required = False
            return False

        if self._release_required:
            return False

        if not self._active:
            self._active = True
            self._start_time = data.time
            self._start_targets = np.array(
                [data.qpos[b.qpos_adr] for b in self._bindings],
                dtype=np.float64,
            )

        elapsed = data.time - self._start_time
        if elapsed >= self._jump_duration:
            self._active = False
            self._release_required = True
            return False

        jump_target, blend = self._target_for_time(elapsed)
        target = (1.0 - blend) * policy_target + blend * jump_target
        self._apply_joint_targets(data, self._policy.clamp_target_positions(target))
        return True

    def _target_for_time(self, elapsed: float) -> tuple[np.ndarray, float]:
        if elapsed < 0.12:
            phase = smootherstep(elapsed / 0.12)
            target = (1.0 - phase) * self._start_targets + phase * self._compress
            return target, self._jump_blend * 0.70 * phase

        if elapsed < 0.24:
            phase = smootherstep((elapsed - 0.12) / 0.12)
            target = (1.0 - phase) * self._compress + phase * self._extend
            blend = self._jump_blend * (0.70 + 0.30 * phase)
            return target, blend

        if elapsed < 0.48:
            phase = smootherstep((elapsed - 0.24) / 0.24)
            target = (1.0 - phase) * self._extend + phase * self._air
            return target, self._jump_blend

        phase = smootherstep((elapsed - 0.48) / max(self._jump_duration - 0.48, 1e-6))
        target = (1.0 - phase) * self._air + phase * self._landing
        blend = self._jump_blend * (1.0 - 0.60 * phase)
        return target, blend

    def _apply_joint_targets(self, data: mujoco.MjData, target: np.ndarray) -> None:
        for i, binding in enumerate(self._bindings):
            q = data.qpos[binding.qpos_adr]
            dq = data.qvel[binding.dof_adr]
            tau = self._stiffness[i] * (target[i] - q) - self._damping[i] * dq
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
            return CommandState(np.zeros(3, dtype=np.float32), False, False, False)

        glfw = self._glfw
        if self.is_key_down(glfw.KEY_ESCAPE):
            glfw.set_window_should_close(self._window, True)

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
            jump_held,
            dash_held and norm > 1e-6,
            active_motion,
        )

    def render_status(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        dash_held: bool,
        jump_active: bool,
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
                "Left drag: rotate camera   Wheel: zoom   Esc: quit"
            ),
            (
                f"vx={command[0]:+.2f}  vy={command[1]:+.2f}  "
                f"yaw={command[2]:+.2f}  "
                f"dash={'on' if dash_held else 'off'}  "
                f"jump={'on' if jump_active else 'off'}"
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


def smootherstep(x: float) -> float:
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def final_uprightness(data: mujoco.MjData) -> float:
    return float(root_rotation_matrix(data)[2, 2])


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
    if args.command_smoothing < 0.0:
        raise ValueError("Command smoothing must be non-negative.")
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
    policy.initialize_pose(data)
    jump = JointSpaceJumpController(policy, args.jump_duration, args.jump_blend)

    print(f"Scene: {args.scene}")
    print(f"Policy: {args.policy_dir}")
    print(
        "Controls: W/A/S/D move, Q/E turn, Shift+WASD dashes, "
        "Space jumps while held."
    )
    print(
        "Command limits: "
        f"forward={args.dash_forward_speed:.2f} m/s, "
        f"backward={args.dash_backward_speed:.2f} m/s, "
        f"lateral={args.dash_lateral_speed:.2f} m/s, "
        f"yaw={args.yaw_speed:.2f} rad/s"
    )

    viewer = None
    if not args.headless:
        viewer = FlatWasdDashViewer(model)

    command = np.zeros(3, dtype=np.float32)
    policy_target_reference = policy._target_joint_pos.copy()
    next_policy_time = data.time
    next_step_wall = time.perf_counter()
    render_interval = 1.0 / args.render_fps
    next_render_wall = next_step_wall

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
                jump_held = (
                    args.test_jump_time >= 0.0
                    and args.test_jump_time
                    <= data.time
                    < args.test_jump_time + args.test_jump_hold
                )
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
                jump_held = state.jump_held
                dash_held = state.dash_held

            command = update_command_with_release_cutoff(
                command,
                target_command,
                model.opt.timestep,
                args.command_smoothing,
                active_motion,
            )

            if data.time >= next_policy_time:
                policy.update_policy(data, command)
                policy_target_reference = policy._target_joint_pos.copy()
                next_policy_time += policy.step_dt

            if not jump.apply_if_held(data, jump_held, policy_target_reference):
                policy.apply_pd(data)

            mujoco.mj_step(model, data)
            next_step_wall += model.opt.timestep
            if next_step_wall < now - 0.1:
                next_step_wall = now

            if viewer is not None and now >= next_render_wall:
                viewer.render_status(data, command, dash_held, jump.active)
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
