from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "go2"
POLICY_DIR = ROOT / "external" / "policies" / "unitree-go2-velocity-flat"
DEFAULT_DISPLAY = ":1"
DEFAULT_RENDER_FPS = 60.0
DEFAULT_IMAGE_WIDTH = 320
DEFAULT_IMAGE_HEIGHT = 240
DEFAULT_CAMERA_NAME = "front_camera"
DEFAULT_TARGET_POS = (2.0, 0.35, 0.24)
DEFAULT_TARGET_RADIUS = 0.18


def configure_mujoco_backend(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--display", default=DEFAULT_DISPLAY)
    args, _ = parser.parse_known_args(argv)
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("DISPLAY", args.display)
        if os.environ.get("MUJOCO_GL") in {"egl", "osmesa"}:
            os.environ.pop("MUJOCO_GL")


configure_mujoco_backend(sys.argv[1:])

import mujoco

from go2_teleop import (
    MouseKeyboardViewer,
    SimToRealPolicyController,
    final_uprightness,
    update_smoothed_command,
)


@dataclass(frozen=True)
class TargetDetection:
    found: bool
    pixel_count: int
    area_fraction: float
    centroid_x: float
    centroid_y: float
    error_x: float


@dataclass
class VisionRuntimeScene:
    temp_dir: tempfile.TemporaryDirectory
    scene_path: Path

    def cleanup(self) -> None:
        self.temp_dir.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track a simple magenta target sphere from a simulated front camera "
            "and command Go2 to follow it."
        )
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=POLICY_DIR,
        help="Directory containing policy.onnx and params/deploy.yaml.",
    )
    parser.add_argument(
        "--display",
        default=DEFAULT_DISPLAY,
        help="X display used for the GUI viewer.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without a GUI window and use offscreen rendering for vision.",
    )
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--real-time", action="store_true")
    parser.add_argument(
        "--render-fps",
        type=float,
        default=DEFAULT_RENDER_FPS,
        help="GUI refresh rate. Physics still runs at the MuJoCo timestep.",
    )
    parser.add_argument("--vision-fps", type=float, default=15.0)
    parser.add_argument("--image-width", type=int, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT)
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET_POS[0])
    parser.add_argument("--target-y", type=float, default=DEFAULT_TARGET_POS[1])
    parser.add_argument("--target-z", type=float, default=DEFAULT_TARGET_POS[2])
    parser.add_argument("--target-radius", type=float, default=DEFAULT_TARGET_RADIUS)
    parser.add_argument(
        "--goal-distance",
        type=float,
        default=0.75,
        help="Stop when the base is within this planar distance from the target.",
    )
    parser.add_argument(
        "--max-forward-speed",
        type=float,
        default=0.35,
        help="Upper bound for visual following forward command in m/s.",
    )
    parser.add_argument(
        "--turning-forward-speed",
        type=float,
        default=0.12,
        help="Forward speed cap while the target is far from image center.",
    )
    parser.add_argument("--max-yaw-rate", type=float, default=0.55)
    parser.add_argument("--yaw-gain", type=float, default=0.85)
    parser.add_argument("--turn-only-error", type=float, default=0.35)
    parser.add_argument("--search-yaw-rate", type=float, default=0.25)
    parser.add_argument("--stop-area-fraction", type=float, default=0.18)
    parser.add_argument("--min-target-pixels", type=int, default=18)
    parser.add_argument("--min-red", type=int, default=80)
    parser.add_argument("--min-blue", type=int, default=60)
    parser.add_argument("--color-margin", type=int, default=35)
    parser.add_argument("--command-smoothing", type=float, default=10.0)
    parser.add_argument("--min-base-height", type=float, default=0.18)
    parser.add_argument("--min-uprightness", type=float, default=0.75)
    parser.add_argument("--print-interval", type=float, default=0.5)
    parser.add_argument(
        "--debug-frames-dir",
        type=Path,
        default=None,
        help="Optional directory for PPM camera frames with detected target boxes.",
    )
    parser.add_argument("--debug-frame-interval", type=float, default=0.5)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration < 0.0:
        raise ValueError("Duration must be non-negative. Use 0 to run until closed.")
    if args.render_fps <= 0.0:
        raise ValueError("Render FPS must be positive.")
    if args.vision_fps <= 0.0:
        raise ValueError("Vision FPS must be positive.")
    if args.image_width <= 0 or args.image_height <= 0:
        raise ValueError("Image size must be positive.")
    if args.target_radius <= 0.0:
        raise ValueError("Target radius must be positive.")
    if args.goal_distance <= 0.0:
        raise ValueError("Goal distance must be positive.")
    if args.max_forward_speed < 0.0:
        raise ValueError("Max forward speed must be non-negative.")
    if args.turning_forward_speed < 0.0:
        raise ValueError("Turning forward speed must be non-negative.")
    if args.max_yaw_rate < 0.0:
        raise ValueError("Max yaw rate must be non-negative.")
    if args.yaw_gain < 0.0:
        raise ValueError("Yaw gain must be non-negative.")
    if args.stop_area_fraction <= 0.0:
        raise ValueError("Stop area fraction must be positive.")
    if args.min_target_pixels < 1:
        raise ValueError("Minimum target pixels must be at least 1.")
    if args.command_smoothing < 0.0:
        raise ValueError("Command smoothing must be non-negative.")
    if args.print_interval <= 0.0:
        raise ValueError("Print interval must be positive.")
    if args.debug_frame_interval <= 0.0:
        raise ValueError("Debug frame interval must be positive.")


def create_runtime_scene(
    model_dir: Path,
    camera_name: str,
    target_pos: tuple[float, float, float],
    target_radius: float,
    image_width: int,
    image_height: int,
) -> VisionRuntimeScene:
    temp_dir = tempfile.TemporaryDirectory(prefix="go2_vision_target_")
    temp_path = Path(temp_dir.name)

    source_go2_xml = model_dir / "go2.xml"
    source_assets = model_dir / "assets"
    runtime_go2_xml = temp_path / "go2.xml"
    runtime_scene_xml = temp_path / "vision_target_scene.xml"

    tree = ET.parse(source_go2_xml)
    root = tree.getroot()
    base_link = root.find(".//body[@name='base_link']")
    if base_link is None:
        temp_dir.cleanup()
        raise RuntimeError("base_link body was not found in the Go2 model.")

    camera = ET.Element(
        "camera",
        {
            "name": camera_name,
            "mode": "fixed",
            "pos": "0.32 0 0.08",
            "xyaxes": "0 -1 0 0 0 1",
            "fovy": "75",
        },
    )
    base_link.insert(0, camera)
    tree.write(runtime_go2_xml, encoding="unicode")

    (temp_path / "assets").symlink_to(source_assets, target_is_directory=True)
    runtime_scene_xml.write_text(
        build_scene_xml(target_pos, target_radius, image_width, image_height),
        encoding="utf-8",
    )
    return VisionRuntimeScene(temp_dir, runtime_scene_xml)


def build_scene_xml(
    target_pos: tuple[float, float, float],
    target_radius: float,
    image_width: int,
    image_height: int,
) -> str:
    tx, ty, tz = target_pos
    return f"""<mujoco model="go2 vision target follow">
  <include file="go2.xml" />

  <statistic center="0 0 0.1" extent="0.8" />

  <visual>
    <headlight diffuse="0.9 0.9 0.9" ambient="0.8 0.8 0.8" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global
      azimuth="-130"
      elevation="-20"
      offwidth="{image_width:d}"
      offheight="{image_height:d}"
    />
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
    <material name="target_magenta" rgba="1 0 1 1" emission="0.8" />
  </asset>

  <worldbody>
    <light pos="0 0 2.0" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.05" type="plane" rgba="0.2 0.3 0.4 1" />
    <geom
      name="vision_target"
      type="sphere"
      pos="{tx:.6f} {ty:.6f} {tz:.6f}"
      size="{target_radius:.6f}"
      material="target_magenta"
      contype="0"
      conaffinity="0"
    />
  </worldbody>
</mujoco>
"""


def detect_magenta_target(frame: np.ndarray, args: argparse.Namespace) -> TargetDetection:
    red = frame[:, :, 0].astype(np.int16)
    green = frame[:, :, 1].astype(np.int16)
    blue = frame[:, :, 2].astype(np.int16)
    mask = (
        (red >= args.min_red)
        & (blue >= args.min_blue)
        & (red - green >= args.color_margin)
        & (blue - green >= args.color_margin)
    )
    pixel_count = int(np.count_nonzero(mask))
    area_fraction = pixel_count / float(frame.shape[0] * frame.shape[1])
    if pixel_count < args.min_target_pixels:
        return TargetDetection(False, pixel_count, area_fraction, math.nan, math.nan, 0.0)

    ys, xs = np.nonzero(mask)
    centroid_x = float(xs.mean())
    centroid_y = float(ys.mean())
    image_center_x = 0.5 * (frame.shape[1] - 1)
    error_x = (centroid_x - image_center_x) / max(image_center_x, 1.0)
    return TargetDetection(
        True,
        pixel_count,
        area_fraction,
        centroid_x,
        centroid_y,
        float(error_x),
    )


def command_from_detection(
    detection: TargetDetection,
    args: argparse.Namespace,
    last_error_x: float,
) -> tuple[np.ndarray, float]:
    if not detection.found:
        search_sign = 1.0 if last_error_x <= 0.0 else -1.0
        command = np.array(
            [0.0, 0.0, search_sign * args.search_yaw_rate],
            dtype=np.float32,
        )
        return command, last_error_x

    yaw_rate = np.clip(
        -args.yaw_gain * detection.error_x,
        -args.max_yaw_rate,
        args.max_yaw_rate,
    )
    area_ratio = min(detection.area_fraction / args.stop_area_fraction, 1.0)
    alignment_scale = max(0.0, 1.0 - abs(detection.error_x))
    forward_speed = args.max_forward_speed * (1.0 - area_ratio) * alignment_scale
    if abs(detection.error_x) >= args.turn_only_error:
        forward_speed = min(forward_speed, args.turning_forward_speed)

    command = np.array([forward_speed, 0.0, yaw_rate], dtype=np.float32)
    return command, detection.error_x


def planar_distance_to_target(
    data: mujoco.MjData,
    target_pos: np.ndarray,
) -> float:
    delta = data.qpos[:2] - target_pos[:2]
    return float(np.linalg.norm(delta))


def save_debug_frame(
    frame: np.ndarray,
    detection: TargetDetection,
    output_dir: Path,
    index: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = frame.copy()
    if detection.found:
        cx = int(round(detection.centroid_x))
        cy = int(round(detection.centroid_y))
        x0 = max(0, cx - 8)
        x1 = min(annotated.shape[1], cx + 9)
        y0 = max(0, cy - 8)
        y1 = min(annotated.shape[0], cy + 9)
        annotated[y0:y1, max(0, cx - 1) : min(annotated.shape[1], cx + 2)] = (
            0,
            255,
            0,
        )
        annotated[max(0, cy - 1) : min(annotated.shape[0], cy + 2), x0:x1] = (
            0,
            255,
            0,
        )

    path = output_dir / f"frame_{index:04d}.ppm"
    with path.open("wb") as f:
        f.write(f"P6\n{annotated.shape[1]} {annotated.shape[0]}\n255\n".encode("ascii"))
        f.write(annotated.astype(np.uint8).tobytes())


class VisionTargetViewer(MouseKeyboardViewer):
    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        if self._window is not None:
            self._glfw.set_window_title(
                self._window,
                "Go2 vision target follow - Esc to quit",
            )
        self._camera.azimuth = -145.0
        self._camera.elevation = -18.0
        self._camera.distance = 3.0

    def render_status(
        self,
        data: mujoco.MjData,
        detection: TargetDetection,
        command: np.ndarray,
        distance: float,
        target_pos: np.ndarray,
    ) -> None:
        if self._window is None:
            return

        if self.is_key_down(self._glfw.KEY_ESCAPE):
            self._glfw.set_window_should_close(self._window, True)

        self._glfw.make_context_current(self._window)
        width, height = self._glfw.get_framebuffer_size(self._window)
        if width <= 0 or height <= 0:
            self._glfw.poll_events()
            return

        midpoint = 0.5 * (data.qpos[:3] + target_pos)
        self._camera.lookat[:] = np.array(
            [midpoint[0], midpoint[1], max(0.35, data.qpos[2])],
            dtype=np.float64,
        )
        self._camera.distance = max(1.6, distance + 1.0)
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
            "Vision target follow   Esc: quit",
            (
                f"detected={'yes' if detection.found else 'no'}  "
                f"pixels={detection.pixel_count}  "
                f"err_x={detection.error_x:+.2f}  "
                f"cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f})  "
                f"distance={distance:.2f} m"
            ),
            self._context,
        )

        self._glfw.swap_buffers(self._window)
        self._glfw.poll_events()


class HeadlessCameraFrameSource:
    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self._renderer.update_scene(data, camera=DEFAULT_CAMERA_NAME)
        return self._renderer.render()

    def close(self) -> None:
        self._renderer.close()


class GuiCameraFrameSource:
    def __init__(
        self,
        model: mujoco.MjModel,
        viewer: VisionTargetViewer,
        width: int,
        height: int,
    ) -> None:
        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            DEFAULT_CAMERA_NAME,
        )
        if camera_id < 0:
            raise RuntimeError(f"Camera not found: {DEFAULT_CAMERA_NAME}")

        self._model = model
        self._viewer = viewer
        self._option = mujoco.MjvOption()
        self._scene = mujoco.MjvScene(model, maxgeom=10000)
        self._camera = mujoco.MjvCamera()
        self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._camera.fixedcamid = camera_id
        self._viewport = mujoco.MjrRect(0, 0, width, height)
        self._rgb = np.empty((height, width, 3), dtype=np.uint8)

    def render(self, data: mujoco.MjData) -> np.ndarray:
        self._viewer._glfw.make_context_current(self._viewer._window)
        mujoco.mjv_updateScene(
            self._model,
            data,
            self._option,
            None,
            self._camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_OFFSCREEN,
            self._viewer._context,
        )
        mujoco.mjr_render(self._viewport, self._scene, self._viewer._context)
        mujoco.mjr_readPixels(self._rgb, None, self._viewport, self._viewer._context)
        mujoco.mjr_setBuffer(
            mujoco.mjtFramebuffer.mjFB_WINDOW,
            self._viewer._context,
        )
        return np.flipud(self._rgb).copy()

    def close(self) -> None:
        self._scene.free()


def main() -> None:
    args = parse_args()
    validate_args(args)

    target_pos = np.array(
        [args.target_x, args.target_y, args.target_z],
        dtype=np.float64,
    )
    runtime_scene = create_runtime_scene(
        MODEL_DIR,
        DEFAULT_CAMERA_NAME,
        tuple(target_pos),
        args.target_radius,
        args.image_width,
        args.image_height,
    )

    camera_frames = None
    viewer = None
    try:
        model = mujoco.MjModel.from_xml_path(str(runtime_scene.scene_path))
        data = mujoco.MjData(model)
        policy = SimToRealPolicyController(model, args.policy_dir)
        policy.initialize_pose(data)
        if not args.headless:
            try:
                viewer = VisionTargetViewer(model)
                camera_frames = GuiCameraFrameSource(
                    model,
                    viewer,
                    args.image_width,
                    args.image_height,
                )
            except Exception as exc:
                if viewer is not None:
                    viewer.close()
                raise RuntimeError(
                    "GUI viewer could not be started. Check DISPLAY/OpenGL, "
                    "pass --display if needed, or use --headless for offscreen runs."
                ) from exc
        else:
            camera_frames = HeadlessCameraFrameSource(
                model,
                args.image_width,
                args.image_height,
            )

        print(f"Scene: {runtime_scene.scene_path}")
        print(f"Policy: {args.policy_dir}")
        print(
            "Target: "
            f"x={target_pos[0]:.2f}, y={target_pos[1]:.2f}, "
            f"z={target_pos[2]:.2f}, radius={args.target_radius:.2f}"
        )
        print(
            "Vision follow: "
            f"{args.image_width}x{args.image_height} @ {args.vision_fps:.1f} FPS, "
            f"GUI={'off' if args.headless else 'on'}, "
            f"MUJOCO_GL={os.environ.get('MUJOCO_GL', '<unset>')}, "
            f"DISPLAY={os.environ.get('DISPLAY', '<unset>')}"
        )

        command = np.zeros(3, dtype=np.float32)
        target_command = np.zeros(3, dtype=np.float32)
        detection = TargetDetection(False, 0, 0.0, math.nan, math.nan, 0.0)
        last_error_x = 0.0
        next_policy_time = data.time
        next_vision_time = data.time
        next_print_time = data.time
        next_debug_frame_time = data.time
        debug_frame_index = 0
        next_step_wall = time.perf_counter()
        next_render_wall = next_step_wall
        vision_interval = 1.0 / args.vision_fps
        render_interval = 1.0 / args.render_fps
        run_real_time = args.real_time or viewer is not None
        stop_reason = "duration"

        while args.duration <= 0.0 or data.time < args.duration:
            now = time.perf_counter()
            if viewer is not None and not viewer.is_running():
                stop_reason = "viewer_closed"
                break
            if run_real_time:
                if now < next_step_wall:
                    time.sleep(min(next_step_wall - now, model.opt.timestep))
                    continue

            data.xfrc_applied[:] = 0.0
            distance = planar_distance_to_target(data, target_pos)
            uprightness = final_uprightness(data)
            base_height = float(data.qpos[2])
            if distance <= args.goal_distance:
                stop_reason = "target_reached"
                target_command[:] = 0.0
                break
            if base_height < args.min_base_height:
                stop_reason = "base_height_low"
                target_command[:] = 0.0
                break
            if uprightness < args.min_uprightness:
                stop_reason = "uprightness_low"
                target_command[:] = 0.0
                break

            if data.time >= next_vision_time:
                frame = camera_frames.render(data)
                detection = detect_magenta_target(frame, args)
                target_command, last_error_x = command_from_detection(
                    detection,
                    args,
                    last_error_x,
                )
                if (
                    args.debug_frames_dir is not None
                    and data.time >= next_debug_frame_time
                ):
                    save_debug_frame(
                        frame,
                        detection,
                        args.debug_frames_dir,
                        debug_frame_index,
                    )
                    debug_frame_index += 1
                    next_debug_frame_time += args.debug_frame_interval
                next_vision_time += vision_interval

            command = update_smoothed_command(
                command,
                target_command,
                model.opt.timestep,
                args.command_smoothing,
            )

            if data.time >= next_policy_time:
                policy.update_policy(data, command)
                next_policy_time += policy.step_dt

            policy.apply_pd(data)
            mujoco.mj_step(model, data)

            if viewer is not None and now >= next_render_wall:
                viewer.render_status(data, detection, command, distance, target_pos)
                next_render_wall += render_interval
                if next_render_wall < now - render_interval:
                    next_render_wall = now

            if data.time >= next_print_time:
                print(
                    f"t={data.time:5.2f}s "
                    f"detected={str(detection.found).lower()} "
                    f"pixels={detection.pixel_count:5d} "
                    f"area={detection.area_fraction:.4f} "
                    f"err_x={detection.error_x:+.3f} "
                    f"cmd=[{command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f}] "
                    f"dist={distance:.2f} "
                    f"upright={uprightness:+.2f}"
                )
                next_print_time += args.print_interval

            if run_real_time:
                next_step_wall += model.opt.timestep
                if next_step_wall < now - 0.1:
                    next_step_wall = now

        print(
            "Result: "
            f"reason={stop_reason}, sim_time={data.time:.2f}s, "
            f"distance={planar_distance_to_target(data, target_pos):.2f}m, "
            f"base_height={data.qpos[2]:.3f}m, "
            f"uprightness={final_uprightness(data):+.3f}"
        )
        if args.debug_frames_dir is not None:
            print(f"Debug frames: {args.debug_frames_dir}")
    finally:
        if camera_frames is not None:
            camera_frames.close()
        if viewer is not None:
            viewer.close()
        runtime_scene.cleanup()


if __name__ == "__main__":
    main()
