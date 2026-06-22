#!/usr/bin/env python3
from __future__ import annotations

import csv
import html
import json
import math
import os
import struct
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass
from pathlib import Path

if "DISPLAY" not in os.environ and "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import mujoco
import numpy as np


GUIDENAV_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = GUIDENAV_ROOT.parent
GO2_MUJOCO_ROOT = PROJECTS_ROOT / "go2-mujoco"
GO2_SCRIPTS = GO2_MUJOCO_ROOT / "scripts"
MODEL_DIR = GO2_MUJOCO_ROOT / "external" / "unitree_mujoco" / "unitree_robots" / "go2"
POLICY_DIR = GO2_MUJOCO_ROOT / "external" / "policies" / "unitree-go2-velocity-flat"
DEFAULT_SCENE_CONFIG = GUIDENAV_ROOT / "mujoco_sim" / "scenes" / "outdoor_city_route.json"
DEFAULT_CAMERA_NAME = "guidenav_front_camera"

if str(GUIDENAV_ROOT) not in sys.path:
    sys.path.insert(0, str(GUIDENAV_ROOT))
if str(GO2_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(GO2_SCRIPTS))

from go2_teleop import (  # noqa: E402
    MouseKeyboardViewer,
    SimToRealPolicyController,
    final_uprightness,
    limit_yaw_command,
    update_smoothed_command,
)


APPEARANCE_PRESETS = {
    "sunny_morning": {
        "sky1": "0.55 0.72 0.92",
        "sky2": "0.90 0.96 1.00",
        "haze": "0.58 0.70 0.82 1",
        "headlight_ambient": [0.55, 0.55, 0.52],
        "headlight_diffuse": [0.95, 0.91, 0.82],
        "light_diffuse": [1.00, 0.88, 0.68],
        "light_ambient": [0.40, 0.42, 0.45],
        "sun_pos": "-2.0 -3.5 5.2",
    },
    "cloudy_noon": {
        "sky1": "0.58 0.64 0.70",
        "sky2": "0.80 0.84 0.86",
        "haze": "0.62 0.66 0.70 1",
        "headlight_ambient": [0.68, 0.68, 0.66],
        "headlight_diffuse": [0.72, 0.72, 0.70],
        "light_diffuse": [0.78, 0.78, 0.75],
        "light_ambient": [0.55, 0.55, 0.55],
        "sun_pos": "0.0 -2.0 5.5",
    },
    "rainy_evening": {
        "sky1": "0.18 0.22 0.30",
        "sky2": "0.40 0.43 0.48",
        "haze": "0.22 0.26 0.32 1",
        "headlight_ambient": [0.35, 0.36, 0.42],
        "headlight_diffuse": [0.45, 0.47, 0.55],
        "light_diffuse": [0.55, 0.53, 0.48],
        "light_ambient": [0.24, 0.25, 0.30],
        "sun_pos": "3.5 -4.0 2.5",
    },
    "night": {
        "sky1": "0.02 0.03 0.06",
        "sky2": "0.08 0.10 0.16",
        "haze": "0.04 0.05 0.08 1",
        "headlight_ambient": [0.13, 0.14, 0.18],
        "headlight_diffuse": [0.25, 0.27, 0.32],
        "light_diffuse": [0.25, 0.28, 0.35],
        "light_ambient": [0.08, 0.09, 0.12],
        "sun_pos": "0.0 -2.0 3.0",
    },
}


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


@dataclass
class RuntimeScene:
    temp_dir: tempfile.TemporaryDirectory
    scene_path: Path

    def cleanup(self) -> None:
        self.temp_dir.cleanup()


@dataclass
class ActorTrack:
    name: str
    start: np.ndarray
    end: np.ndarray
    period: float
    phase: float
    mocap_id: int = -1


def setup_display(display: str | None, headless: bool) -> None:
    if headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    elif display:
        os.environ.setdefault("DISPLAY", display)


def load_scene_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def create_runtime_scene(
    model_dir: Path,
    scene_config: dict,
    image_width: int,
    image_height: int,
    scene_preset: str,
) -> RuntimeScene:
    temp_dir = tempfile.TemporaryDirectory(prefix="guidenav_mujoco_")
    temp_path = Path(temp_dir.name)
    source_go2_xml = Path(model_dir) / "go2.xml"
    source_assets = Path(model_dir) / "assets"
    runtime_go2_xml = temp_path / "go2.xml"
    runtime_scene_xml = temp_path / "guidenav_mujoco_scene.xml"

    tree = ET.parse(source_go2_xml)
    root = tree.getroot()
    base_link = root.find(".//body[@name='base_link']")
    if base_link is None:
        temp_dir.cleanup()
        raise RuntimeError("base_link body was not found in the Go2 model.")

    camera = ET.Element(
        "camera",
        {
            "name": DEFAULT_CAMERA_NAME,
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
        build_scene_xml(scene_config, image_width, image_height, scene_preset),
        encoding="utf-8",
    )
    return RuntimeScene(temp_dir, runtime_scene_xml)


def fmt_vec(values: list[float] | tuple[float, ...] | np.ndarray) -> str:
    return " ".join(f"{float(v):.4f}" for v in values)


def build_scene_xml(
    scene_config: dict,
    image_width: int,
    image_height: int,
    scene_preset: str,
) -> str:
    preset = APPEARANCE_PRESETS[scene_preset]
    route_xml = build_route_markers(scene_config)
    street_xml = build_city_static_xml(scene_preset)
    actor_xml = build_dynamic_actor_xml(scene_config)

    return f"""<mujoco model="guidenav official mujoco outdoor">
  <include file="go2.xml" />

  <statistic center="3.5 2.2 0.4" extent="7.5" />

  <visual>
    <headlight diffuse="{fmt_vec(preset['headlight_diffuse'])}" ambient="{fmt_vec(preset['headlight_ambient'])}" specular="0.05 0.05 0.05" />
    <rgba haze="{preset['haze']}" />
    <global azimuth="-130" elevation="-22" offwidth="{image_width:d}" offheight="{image_height:d}" />
  </visual>

  <asset>
    <texture name="sky_tex" type="skybox" builtin="gradient" rgb1="{preset['sky1']}" rgb2="{preset['sky2']}" width="512" height="3072" />
    <texture name="asphalt_tex" type="2d" builtin="checker" rgb1="0.075 0.080 0.083" rgb2="0.105 0.110 0.115" width="128" height="128" />
    <texture name="paving_tex" type="2d" builtin="checker" rgb1="0.42 0.40 0.36" rgb2="0.50 0.48 0.43" width="128" height="128" />
    <material name="ground_mat" rgba="0.19 0.25 0.19 1" />
    <material name="asphalt_mat" texture="asphalt_tex" texrepeat="10 10" rgba="0.95 0.95 0.95 1" />
    <material name="sidewalk_mat" texture="paving_tex" texrepeat="8 8" rgba="0.95 0.95 0.95 1" />
    <material name="lane_mat" rgba="0.96 0.90 0.48 1" emission="0.05" />
    <material name="crosswalk_mat" rgba="0.95 0.95 0.90 1" />
    <material name="route_marker_mat" rgba="0.1 0.75 0.95 0.75" emission="0.25" />
    <material name="building_mat" rgba="0.42 0.45 0.48 1" />
    <material name="glass_mat" rgba="0.15 0.25 0.34 0.65" />
    <material name="tree_trunk_mat" rgba="0.30 0.18 0.09 1" />
    <material name="tree_leaf_mat" rgba="0.12 0.36 0.16 1" />
    <material name="car_red_mat" rgba="0.70 0.12 0.09 1" />
    <material name="car_blue_mat" rgba="0.08 0.16 0.55 1" />
    <material name="pedestrian_mat" rgba="0.12 0.18 0.24 1" />
    <material name="cone_mat" rgba="1.00 0.38 0.04 1" />
    <material name="barrier_mat" rgba="0.95 0.72 0.12 1" />
    <material name="rain_mat" rgba="0.62 0.76 0.95 0.32" />
    <material name="lamp_mat" rgba="1.00 0.87 0.48 1" emission="0.35" />
  </asset>

  <worldbody>
    <light name="sun" pos="{preset['sun_pos']}" dir="-0.35 0.15 -1" directional="true" diffuse="{fmt_vec(preset['light_diffuse'])}" ambient="{fmt_vec(preset['light_ambient'])}" />
    <light name="street_lamp_01" pos="4.4 1.2 3.0" dir="0 0 -1" diffuse="0.85 0.72 0.48" ambient="0.05 0.04 0.03" />
    <light name="street_lamp_02" pos="2.0 5.2 3.0" dir="0 0 -1" diffuse="0.70 0.72 0.88" ambient="0.05 0.05 0.06" />
    <geom name="ground" type="plane" size="0 0 0.05" material="ground_mat" friction="1.0 0.1 0.1" />
{street_xml}
{route_xml}
{actor_xml}
  </worldbody>
</mujoco>
"""


def build_route_markers(scene_config: dict) -> str:
    lines = []
    for i, waypoint in enumerate(scene_config["route_waypoints"]):
        x, y = waypoint
        lines.append(
            f'    <geom name="route_marker_{i:02d}" type="sphere" '
            f'pos="{float(x):.4f} {float(y):.4f} 0.0350" size="0.055" '
            'material="route_marker_mat" contype="0" conaffinity="0" />'
        )
    return "\n".join(lines)


def box_geom(name: str, pos, size, material: str, collide: bool = False) -> str:
    contact = "" if collide else ' contype="0" conaffinity="0"'
    return (
        f'    <geom name="{name}" type="box" pos="{fmt_vec(pos)}" '
        f'size="{fmt_vec(size)}" material="{material}"{contact} />'
    )


def build_city_static_xml(scene_preset: str) -> str:
    lines: list[str] = []
    lines.extend(
        [
            box_geom("road_east_west", (2.8, 0.0, 0.006), (3.4, 0.95, 0.006), "asphalt_mat"),
            box_geom("road_north_south", (5.0, 2.2, 0.007), (0.95, 2.8, 0.006), "asphalt_mat"),
            box_geom("road_westbound", (3.4, 4.4, 0.008), (2.5, 0.95, 0.006), "asphalt_mat"),
            box_geom("sidewalk_lower", (2.8, -1.28, 0.018), (3.7, 0.28, 0.018), "sidewalk_mat"),
            box_geom("sidewalk_upper_a", (2.5, 1.28, 0.018), (2.9, 0.28, 0.018), "sidewalk_mat"),
            box_geom("sidewalk_right", (6.28, 2.2, 0.018), (0.28, 3.1, 0.018), "sidewalk_mat"),
            box_geom("sidewalk_left", (3.72, 2.2, 0.018), (0.28, 3.1, 0.018), "sidewalk_mat"),
            box_geom("sidewalk_top", (3.4, 5.68, 0.018), (2.7, 0.28, 0.018), "sidewalk_mat"),
        ]
    )

    for i, x in enumerate(np.linspace(0.4, 5.2, 7)):
        lines.append(box_geom(f"lane_dash_ew_{i:02d}", (x, 0.0, 0.020), (0.18, 0.018, 0.006), "lane_mat"))
    for i, y in enumerate(np.linspace(0.6, 3.8, 6)):
        lines.append(box_geom(f"lane_dash_ns_{i:02d}", (5.0, y, 0.021), (0.018, 0.16, 0.006), "lane_mat"))
    for i, y in enumerate(np.linspace(-0.55, 0.55, 6)):
        lines.append(box_geom(f"crosswalk_01_{i:02d}", (4.15, y, 0.026), (0.09, 0.055, 0.006), "crosswalk_mat"))
    for i, x in enumerate(np.linspace(4.35, 5.65, 6)):
        lines.append(box_geom(f"crosswalk_02_{i:02d}", (x, 3.65, 0.026), (0.055, 0.09, 0.006), "crosswalk_mat"))

    buildings = [
        ("building_01", (-0.9, 1.8, 0.9), (0.55, 1.2, 0.9)),
        ("building_02", (1.4, 2.55, 1.1), (0.75, 0.55, 1.1)),
        ("building_03", (6.95, 2.4, 1.2), (0.55, 1.4, 1.2)),
        ("building_04", (3.2, 6.55, 1.0), (1.2, 0.55, 1.0)),
    ]
    for name, pos, size in buildings:
        lines.append(box_geom(name, pos, size, "building_mat"))
        lines.append(box_geom(f"{name}_windows", (pos[0], pos[1] - size[1] - 0.01, pos[2] + 0.15), (size[0] * 0.72, 0.012, size[2] * 0.42), "glass_mat"))

    parked_cars = [
        ("parked_car_01", (1.25, -0.72, 0.16), (0.52, 0.22, 0.16), "car_blue_mat"),
        ("parked_car_02", (3.65, -0.72, 0.16), (0.52, 0.22, 0.16), "car_red_mat"),
        ("parked_car_03", (5.72, 2.35, 0.16), (0.22, 0.52, 0.16), "car_blue_mat"),
    ]
    for name, pos, size, mat in parked_cars:
        lines.append(box_geom(name, pos, size, mat, collide=True))
        lines.append(box_geom(f"{name}_cabin", (pos[0], pos[1], pos[2] + 0.18), (size[0] * 0.55, size[1] * 0.65, 0.10), "glass_mat"))

    cones = [(4.25, 0.55), (4.55, 0.75), (4.85, 0.95), (5.32, 3.28), (5.62, 3.18)]
    for i, (x, y) in enumerate(cones):
        lines.append(
            f'    <geom name="traffic_cone_{i:02d}" type="cylinder" pos="{x:.4f} {y:.4f} 0.1200" '
            'size="0.065 0.180" material="cone_mat" />'
        )
    lines.append(box_geom("work_barrier_01", (4.65, 1.08, 0.22), (0.45, 0.035, 0.16), "barrier_mat", collide=True))
    lines.append(box_geom("work_barrier_02", (5.70, 3.16, 0.22), (0.035, 0.45, 0.16), "barrier_mat", collide=True))

    trees = [(0.2, -1.55), (2.6, -1.55), (6.55, 0.65), (6.55, 4.25), (1.35, 5.9)]
    for i, (x, y) in enumerate(trees):
        lines.append(
            f'    <geom name="tree_trunk_{i:02d}" type="cylinder" pos="{x:.4f} {y:.4f} 0.4200" '
            'size="0.055 0.420" material="tree_trunk_mat" contype="0" conaffinity="0" />'
        )
        lines.append(
            f'    <geom name="tree_leaf_{i:02d}" type="sphere" pos="{x:.4f} {y:.4f} 1.0500" '
            'size="0.36" material="tree_leaf_mat" contype="0" conaffinity="0" />'
        )

    lamps = [(4.4, 1.2), (2.0, 5.2)]
    for i, (x, y) in enumerate(lamps):
        lines.append(
            f'    <geom name="lamp_post_{i:02d}" type="cylinder" pos="{x:.4f} {y:.4f} 1.1500" '
            'size="0.028 1.150" material="building_mat" contype="0" conaffinity="0" />'
        )
        lines.append(
            f'    <geom name="lamp_head_{i:02d}" type="sphere" pos="{x:.4f} {y:.4f} 2.3400" '
            'size="0.085" material="lamp_mat" contype="0" conaffinity="0" />'
        )

    if scene_preset == "rainy_evening":
        for i in range(42):
            x = -0.5 + (i % 14) * 0.55
            y = -1.4 + (i // 14) * 2.3
            lines.append(
                f'    <geom name="rain_streak_{i:02d}" type="capsule" '
                f'fromto="{x:.3f} {y:.3f} 2.30 {x + 0.10:.3f} {y - 0.03:.3f} 1.35" '
                'size="0.006" material="rain_mat" contype="0" conaffinity="0" />'
            )

    return "\n".join(lines)


def build_dynamic_actor_xml(scene_config: dict) -> str:
    lines: list[str] = []
    for actor in scene_config.get("dynamic_actors", []):
        name = actor["name"]
        start = actor["start"]
        kind = actor.get("kind", "pedestrian")
        if kind == "car":
            lines.append(
                f'    <body name="{name}" mocap="true" pos="{fmt_vec(start)}">\n'
                f'      <geom name="{name}_body" type="box" pos="0 0 0.1900" size="0.5600 0.2600 0.1700" material="car_red_mat" />\n'
                f'      <geom name="{name}_cabin" type="box" pos="-0.0500 0 0.3900" size="0.3000 0.2200 0.1200" material="glass_mat" contype="0" conaffinity="0" />\n'
                f'    </body>'
            )
        else:
            lines.append(
                f'    <body name="{name}" mocap="true" pos="{fmt_vec(start)}">\n'
                f'      <geom name="{name}_body" type="capsule" fromto="0 0 0.1200 0 0 1.0500" size="0.1050" material="pedestrian_mat" />\n'
                f'      <geom name="{name}_head" type="sphere" pos="0 0 1.2500" size="0.1300" material="pedestrian_mat" />\n'
                f'    </body>'
            )
    return "\n".join(lines)


class DynamicActors:
    def __init__(self, model: mujoco.MjModel, scene_config: dict) -> None:
        self._tracks: list[ActorTrack] = []
        for actor in scene_config.get("dynamic_actors", []):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, actor["name"])
            if body_id < 0:
                continue
            mocap_id = int(model.body_mocapid[body_id])
            if mocap_id < 0:
                continue
            self._tracks.append(
                ActorTrack(
                    name=actor["name"],
                    start=np.asarray(actor["start"], dtype=np.float64),
                    end=np.asarray(actor["end"], dtype=np.float64),
                    period=max(float(actor.get("period", 10.0)), 1e-6),
                    phase=float(actor.get("phase", 0.0)),
                    mocap_id=mocap_id,
                )
            )

    def apply(self, data: mujoco.MjData, sim_time: float) -> None:
        for track in self._tracks:
            u = (sim_time / track.period + track.phase) % 1.0
            if u < 0.5:
                s = 2.0 * u
                direction = track.end - track.start
            else:
                s = 2.0 * (1.0 - u)
                direction = track.start - track.end
            pos = track.start + s * (track.end - track.start)
            yaw = math.atan2(float(direction[1]), float(direction[0]))
            data.mocap_pos[track.mocap_id, :] = pos
            data.mocap_quat[track.mocap_id, :] = np.array(
                [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)],
                dtype=np.float64,
            )


def apply_appearance(
    model: mujoco.MjModel,
    scene_preset: str,
    sim_time: float,
    cycle_appearance: bool,
    cycle_period: float,
) -> None:
    preset_name = scene_preset
    if cycle_appearance:
        names = list(APPEARANCE_PRESETS)
        preset_name = names[int(sim_time / max(cycle_period, 1e-6)) % len(names)]
    preset = APPEARANCE_PRESETS[preset_name]
    try:
        model.vis.headlight.ambient[:] = preset["headlight_ambient"]
        model.vis.headlight.diffuse[:] = preset["headlight_diffuse"]
        haze = [float(v) for v in str(preset["haze"]).split()]
        model.vis.rgba.haze[:] = haze
        if model.nlight > 0:
            model.light_diffuse[:, :] = np.asarray(preset["light_diffuse"], dtype=np.float32)
            model.light_ambient[:, :] = np.asarray(preset["light_ambient"], dtype=np.float32)
    except AttributeError:
        return


class MuJoCoGuideNavViewer(MouseKeyboardViewer):
    def __init__(self, model: mujoco.MjModel, title: str) -> None:
        super().__init__(model)
        self._glfw.set_window_title(self._window, title)
        self._camera.azimuth = -135.0
        self._camera.elevation = -24.0
        self._camera.distance = 5.2
        self._image_panel_error_reported = False

    def read_command_state(
        self,
        normal_speed: float,
        dash_forward_speed: float,
        dash_backward_speed: float,
        dash_lateral_speed: float,
        yaw_speed: float,
    ):
        glfw = self._glfw
        if self.is_key_down(glfw.KEY_ESCAPE):
            glfw.set_window_should_close(self._window, True)

        reset_held = self.is_key_down(glfw.KEY_R)
        reset_requested = reset_held and not self._reset_was_down
        self._reset_was_down = reset_held

        vx_key = float(self.is_key_down(glfw.KEY_W)) - float(self.is_key_down(glfw.KEY_S))
        vy_key = float(self.is_key_down(glfw.KEY_A)) - float(self.is_key_down(glfw.KEY_D))
        yaw_key = float(self.is_key_down(glfw.KEY_Q)) - float(self.is_key_down(glfw.KEY_E))
        dash_held = self.is_key_down(glfw.KEY_LEFT_SHIFT) or self.is_key_down(glfw.KEY_RIGHT_SHIFT)

        planar = np.array([vx_key, vy_key], dtype=np.float64)
        norm = float(np.linalg.norm(planar))
        if norm > 1.0:
            planar /= norm
            norm = 1.0

        if norm > 1e-6 and dash_held:
            if planar[0] >= 0.5:
                speed = dash_forward_speed
            elif planar[0] <= -0.5:
                speed = dash_backward_speed
            else:
                speed = dash_lateral_speed
        else:
            speed = normal_speed

        command = np.array([speed * planar[0], speed * planar[1], yaw_speed * yaw_key], dtype=np.float32)
        active_motion = norm > 1e-6 or abs(yaw_key) > 0.0
        return command, active_motion, reset_requested

    def render_status(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        mode: str,
        status: str,
        info: str,
        image_panels: list[tuple[str, np.ndarray]] | None = None,
    ) -> None:
        if self._window is None:
            return

        self._glfw.make_context_current(self._window)
        width, height = self._glfw.get_framebuffer_size(self._window)
        if width <= 0 or height <= 0:
            self._glfw.poll_events()
            return

        self._camera.lookat[:] = data.qpos[:3] + np.array([0.6, 0.0, 0.18])
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
        if image_panels:
            try:
                self._draw_image_panels(viewport, image_panels)
            except Exception as exc:
                if not self._image_panel_error_reported:
                    print(f"Image panel rendering disabled after error: {exc}", flush=True)
                    self._image_panel_error_reported = True
        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            "GuideNav MuJoCo   W/S move   A/D strafe   Q/E turn   Shift faster   R reset   Esc quit",
            (
                f"{mode}  t={data.time:.2f}s  pose=({data.qpos[0]:+.2f}, {data.qpos[1]:+.2f})\n"
                f"cmd=({command[0]:+.2f}, {command[1]:+.2f}, {command[2]:+.2f})  {info}\n"
                f"{status}"
            ),
            self._context,
        )
        self._glfw.swap_buffers(self._window)
        self._glfw.poll_events()

    def _draw_image_panels(
        self,
        viewport: mujoco.MjrRect,
        image_panels: list[tuple[str, np.ndarray]],
    ) -> None:
        panel_width = min(320, max(140, viewport.width // 4))
        gap = 10
        title_height = 24
        x = max(8, viewport.width - panel_width - gap)
        y_top = max(8, viewport.height - gap)

        for title, image in image_panels[:3]:
            image_rgb = resize_rgb_for_panel(image, panel_width)
            height, width = image_rgb.shape[:2]
            y_top -= title_height + height
            if y_top < 8:
                break
            rect = mujoco.MjrRect(x, y_top, width, height)
            mujoco.mjr_rectangle(rect, 0.02, 0.02, 0.02, 0.84)
            # MuJoCo's Python binding expects a flat client RGB buffer here.
            rgb_buffer = np.ascontiguousarray(np.flipud(image_rgb).reshape(-1))
            mujoco.mjr_drawPixels(rgb_buffer, None, rect, self._context)

            title_rect = mujoco.MjrRect(x, y_top + height, width, title_height)
            mujoco.mjr_rectangle(title_rect, 0.02, 0.02, 0.02, 0.90)
            safe_title = html.escape(title, quote=False)
            mujoco.mjr_overlay(
                mujoco.mjtFontScale.mjFONTSCALE_100,
                mujoco.mjtGridPos.mjGRID_TOPRIGHT,
                title_rect,
                "",
                safe_title,
                self._context,
            )
            y_top -= gap
            y_top -= gap + 24

    def render_camera_rgb_depth(
        self,
        data: mujoco.MjData,
        camera_name: str,
        width: int,
        height: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        self._glfw.make_context_current(self._window)

        camera_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise RuntimeError(f"Camera not found in MuJoCo model: {camera_name}")

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        viewport = mujoco.MjrRect(0, 0, width, height)
        rgb = np.empty((height, width, 3), dtype=np.uint8)
        depth = np.empty((height, width), dtype=np.float32)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._context)
        mujoco.mjv_updateScene(
            self._model,
            data,
            self._option,
            None,
            camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._scene,
        )
        mujoco.mjr_render(viewport, self._scene, self._context)
        mujoco.mjr_readPixels(rgb, depth, viewport, self._context)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self._context)

        return np.flipud(rgb).copy(), np.flipud(depth).copy()


class HeadlessCameraCapture:
    def __init__(self, model: mujoco.MjModel, width: int, height: int) -> None:
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def render_rgb_depth(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=DEFAULT_CAMERA_NAME)
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=DEFAULT_CAMERA_NAME)
        depth = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return rgb, depth

    def close(self) -> None:
        self._renderer.close()


class GuiCameraCapture:
    def __init__(self, viewer: MuJoCoGuideNavViewer, width: int, height: int) -> None:
        self._viewer = viewer
        self._width = width
        self._height = height

    def render_rgb_depth(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return self._viewer.render_camera_rgb_depth(
            data,
            DEFAULT_CAMERA_NAME,
            self._width,
            self._height,
        )

    def close(self) -> None:
        return


class MujocoGo2Runtime:
    def __init__(self, args, title: str, with_viewer: bool) -> None:
        self.scene_config = load_scene_config(args.scene_config)
        self.runtime_scene = create_runtime_scene(
            args.model_dir,
            self.scene_config,
            args.image_width,
            args.image_height,
            args.scene_preset,
        )
        self.model = mujoco.MjModel.from_xml_path(str(self.runtime_scene.scene_path))
        self.data = mujoco.MjData(self.model)
        self.dynamic_actors = DynamicActors(self.model, self.scene_config)
        self.dynamic_actors.apply(self.data, 0.0)
        self.policy = SimToRealPolicyController(self.model, args.policy_dir)
        self.policy.initialize_pose(
            self.data,
            base_height=args.reset_base_height,
            stance_crouch=args.stance_crouch,
        )
        self.dynamic_actors.apply(self.data, 0.0)
        mujoco.mj_forward(self.model, self.data)
        self.viewer = MuJoCoGuideNavViewer(self.model, title) if with_viewer else None
        self.camera = (
            GuiCameraCapture(self.viewer, args.image_width, args.image_height)
            if self.viewer is not None
            else HeadlessCameraCapture(self.model, args.image_width, args.image_height)
        )
        self.next_policy_time = float(self.data.time)
        self.command = np.zeros(3, dtype=np.float32)
        self.scene_preset = args.scene_preset
        self.cycle_appearance = bool(args.cycle_appearance)
        self.cycle_period = float(args.cycle_period)

    def reset_pose(self, args) -> None:
        self.policy.initialize_pose(
            self.data,
            base_height=args.reset_base_height,
            stance_crouch=args.stance_crouch,
        )
        self.dynamic_actors.apply(self.data, float(self.data.time))
        self.command[:] = 0.0
        self.next_policy_time = float(self.data.time)
        mujoco.mj_forward(self.model, self.data)

    def step_policy(self, target_command: np.ndarray, active_motion: bool, args) -> np.ndarray:
        apply_appearance(
            self.model,
            self.scene_preset,
            float(self.data.time),
            self.cycle_appearance,
            self.cycle_period,
        )
        self.dynamic_actors.apply(self.data, float(self.data.time))
        target_command = limit_yaw_command(target_command, args.yaw_safety_limit)
        self.command = (
            update_smoothed_command(
                self.command,
                target_command,
                self.model.opt.timestep,
                args.command_smoothing,
            )
            if active_motion
            else np.zeros(3, dtype=np.float32)
        )
        if self.data.time >= self.next_policy_time:
            self.policy.update_policy(self.data, self.command)
            self.next_policy_time += self.policy.step_dt
        self.policy.apply_pd(self.data)
        mujoco.mj_step(self.model, self.data)
        return self.command

    def close(self) -> None:
        self.camera.close()
        if self.viewer is not None:
            self.viewer.close()
        self.runtime_scene.cleanup()


def write_png_rgb(path: Path, image: np.ndarray) -> None:
    image = np.asarray(image, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 uint8 RGB image, got {image.shape}")
    _write_png(path, image.shape[1], image.shape[0], 8, 2, image)


def ensure_rgb_uint8(image: np.ndarray, bgr: bool = False) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    if bgr:
        array = array[:, :, ::-1]
    return np.ascontiguousarray(array)


def resize_rgb_for_panel(image: np.ndarray, panel_width: int) -> np.ndarray:
    image = ensure_rgb_uint8(image)
    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image shape: {image.shape}")
    scale = panel_width / float(width)
    panel_height = max(1, int(round(height * scale)))
    y_idx = np.minimum((np.arange(panel_height) / scale).astype(np.int64), height - 1)
    x_idx = np.minimum((np.arange(panel_width) / scale).astype(np.int64), width - 1)
    return np.ascontiguousarray(image[y_idx][:, x_idx])


def write_png_depth16(path: Path, depth_m: np.ndarray) -> None:
    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(depth * 1000.0, 0, 65535).astype(">u2", copy=False)
    _write_png(path, depth_mm.shape[1], depth_mm.shape[0], 16, 0, depth_mm)


def _write_png(path: Path, width: int, height: int, bit_depth: int, color_type: int, pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = b"".join(b"\x00" + np.ascontiguousarray(row).tobytes() for row in pixels)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc)


def write_odom_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    file_obj = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file_obj)
    writer.writerow(
        [
            "timestamp",
            "pos_x",
            "pos_y",
            "pos_z",
            "ori_x",
            "ori_y",
            "ori_z",
            "ori_w",
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel_x",
            "ang_vel_y",
            "ang_vel_z",
        ]
    )
    return file_obj, writer


def write_odom_row(writer, data: mujoco.MjData) -> None:
    quat_wxyz = data.qpos[3:7]
    writer.writerow(
        [
            f"{float(data.time):.9f}",
            f"{float(data.qpos[0]):.9f}",
            f"{float(data.qpos[1]):.9f}",
            f"{float(data.qpos[2]):.9f}",
            f"{float(quat_wxyz[1]):.9f}",
            f"{float(quat_wxyz[2]):.9f}",
            f"{float(quat_wxyz[3]):.9f}",
            f"{float(quat_wxyz[0]):.9f}",
            f"{float(data.qvel[0]):.9f}",
            f"{float(data.qvel[1]):.9f}",
            f"{float(data.qvel[2]):.9f}",
            f"{float(data.qvel[3]):.9f}",
            f"{float(data.qvel[4]):.9f}",
            f"{float(data.qvel[5]):.9f}",
        ]
    )


def pose_from_data(data: mujoco.MjData) -> Pose2D:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, data.qpos[3:7])
    rotation = matrix.reshape(3, 3)
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return Pose2D(float(data.qpos[0]), float(data.qpos[1]), yaw)


def pose_distance(a: Pose2D, b: Pose2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def normalize_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def route_waypoint_poses(scene_config: dict) -> list[Pose2D]:
    points = [(float(x), float(y)) for x, y in scene_config["route_waypoints"]]
    poses: list[Pose2D] = []
    for i, (x, y) in enumerate(points):
        if i + 1 < len(points):
            nx, ny = points[i + 1]
            yaw = math.atan2(ny - y, nx - x)
        else:
            px, py = points[i - 1]
            yaw = math.atan2(y - py, x - px)
        poses.append(Pose2D(x, y, yaw))
    return poses


def waypoint_command(
    current: Pose2D,
    target: Pose2D,
    max_vx: float,
    max_vy: float,
    max_yaw: float,
) -> np.ndarray:
    dx = target.x - current.x
    dy = target.y - current.y
    c = math.cos(current.yaw)
    s = math.sin(current.yaw)
    forward = c * dx + s * dy
    left = -s * dx + c * dy
    distance = math.hypot(forward, left)
    if distance < 1e-6:
        return np.zeros(3, dtype=np.float32)
    bearing = math.atan2(left, forward)
    heading_error = normalize_angle(target.yaw - current.yaw)
    vx = min(max_vx, 0.72 * distance) * max(0.0, math.cos(bearing))
    if abs(bearing) > math.radians(45.0):
        vx *= 0.35
    vy = max(-max_vy, min(max_vy, 0.45 * left))
    yaw = 1.10 * bearing + 0.20 * heading_error
    yaw = max(-max_yaw, min(max_yaw, yaw))
    return np.array([vx, vy, yaw], dtype=np.float32)


def scripted_teacher_command(
    current: Pose2D,
    waypoints: list[Pose2D],
    active_index: int,
    normal_speed: float,
    yaw_speed: float,
) -> tuple[np.ndarray, int, bool]:
    index = active_index
    while index < len(waypoints) - 1 and pose_distance(current, waypoints[index]) < 0.35:
        index += 1
    done = index >= len(waypoints) - 1 and pose_distance(current, waypoints[-1]) < 0.35
    if done:
        return np.zeros(3, dtype=np.float32), index, True
    command = waypoint_command(
        current,
        waypoints[index],
        max_vx=min(normal_speed, 0.34),
        max_vy=0.16,
        max_yaw=yaw_speed,
    )
    return command, index, False


def teleop_or_scripted_command(runtime: MujocoGo2Runtime, args, scripted_state: dict) -> tuple[np.ndarray, bool, str]:
    if args.scripted_teacher or runtime.viewer is None:
        waypoints = scripted_state.setdefault("waypoints", route_waypoint_poses(runtime.scene_config))
        index = scripted_state.setdefault("index", 1)
        target_command, index, done = scripted_teacher_command(
            pose_from_data(runtime.data),
            waypoints,
            index,
            args.normal_speed,
            args.yaw_speed,
        )
        scripted_state["index"] = index
        return target_command, bool(np.linalg.norm(target_command) > 1e-6), "scripted_goal_reached" if done else ""

    command, active_motion, reset_requested = runtime.viewer.read_command_state(
        args.normal_speed,
        args.dash_forward_speed,
        args.dash_backward_speed,
        args.dash_lateral_speed,
        args.yaw_speed,
    )
    if reset_requested:
        runtime.reset_pose(args)
    return command, active_motion, ""


def contact_with_named_obstacle(model: mujoco.MjModel, data: mujoco.MjData) -> str:
    prefixes = ("moving_car_", "pedestrian_", "traffic_cone_", "work_barrier_", "parked_car_")
    for i in range(data.ncon):
        contact = data.contact[i]
        for geom_id in (int(contact.geom1), int(contact.geom2)):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            if name and name.startswith(prefixes):
                return name
    return ""


def sleep_for_realtime(next_step_wall: float, timestep: float) -> None:
    now = time.perf_counter()
    if now < next_step_wall:
        time.sleep(min(next_step_wall - now, timestep))
