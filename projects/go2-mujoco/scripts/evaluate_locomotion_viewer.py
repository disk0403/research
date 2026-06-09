from __future__ import annotations

import argparse
import csv
import json
import os
import secrets
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np

from go2_continuous_rough_terrain_teleop import (
    DEFAULT_START_FLAT_RADIUS,
    DEFAULT_TERRAIN_AMPLITUDE,
    DEFAULT_TERRAIN_SEED,
    DEFAULT_TERRAIN_SMOOTHING_PASSES,
    DEFAULT_TERRAIN_X_HALF_SIZE,
    DEFAULT_TERRAIN_Y_HALF_SIZE,
    TERRAIN_NAME,
    ContinuousRoughTerrainViewer,
    configure_heightfield,
    create_runtime_scene,
    resolve_terrain_seed,
)
from go2_teleop import (
    DEFAULT_DISPLAY,
    DEFAULT_RENDER_FPS,
    MODEL_DIR,
    POLICY_DIR,
    ROOT,
    SimToRealPolicyController,
    final_uprightness,
    update_command_with_release_cutoff,
)


DEFAULT_DURATION = 5.0
DEFAULT_EPISODES = 5
DEFAULT_FORWARD_COMMAND = 0.3
DEFAULT_GOAL_DISTANCE = "auto"
DEFAULT_GOAL_PROGRESS_RATIO = 0.75
DEFAULT_FALL_HEIGHT = 0.16
DEFAULT_FALL_UPRIGHTNESS = 0.55
DEFAULT_FALL_WARMUP = 0.5
DEFAULT_EPISODE_PAUSE = 0.8
DEFAULT_LOG_DIR = ROOT / "logs" / "locomotion_eval"
DEFAULT_RESULTS_CSV = DEFAULT_LOG_DIR / "locomotion_results.csv"
DEFAULT_RESULTS_MD = DEFAULT_LOG_DIR / "locomotion_results.md"
TERRAIN_GOAL_MARGIN = 2.0
CLI_SPACE_SEPARATORS = ("\u3000", "\u00a0")
GOAL_MARKER_RGBA = np.array([0.0, 0.95, 1.0, 1.0], dtype=np.float32)
GOAL_MARKER_MAT = np.eye(3, dtype=np.float64).reshape(-1)
GOAL_MARKER_ZERO = np.zeros(3, dtype=np.float64)
RESULT_FIELDNAMES = [
    "row_id",
    "run_id",
    "run_started_at",
    "result_jsonl",
    "episode",
    "episodes_planned",
    "policy_dir",
    "headless",
    "success",
    "status_jp",
    "termination_reason",
    "fell",
    "reached_goal",
    "terrain_seed_random",
    "terrain_seed",
    "terrain_seed_stride",
    "terrain_amplitude",
    "terrain_smoothing_passes",
    "start_flat_radius",
    "terrain_x_half_size",
    "terrain_y_half_size",
    "terrain_rows",
    "terrain_cols",
    "terrain_min_height",
    "terrain_max_height",
    "terrain_max_neighbor_height_delta",
    "terrain_max_neighbor_slope_degrees",
    "command_vx",
    "command_vy",
    "command_yaw",
    "command_smoothing",
    "duration_limit",
    "sim_time",
    "goal_auto",
    "goal_progress_ratio",
    "goal_distance",
    "goal_x",
    "progress_x",
    "progress_goal_ratio",
    "planar_displacement",
    "path_length",
    "path_efficiency",
    "average_forward_speed",
    "final_x",
    "final_y",
    "final_z",
    "final_terrain_height",
    "final_base_height_above_terrain",
    "final_uprightness",
    "min_base_height",
    "min_base_height_above_terrain",
    "min_uprightness",
    "fall_height",
    "fall_uprightness",
    "fall_warmup",
]


@dataclass
class EpisodeResult:
    episode: int
    terrain_seed: int
    terrain_amplitude: float
    terrain_smoothing_passes: int
    command_vx: float
    command_vy: float
    command_yaw: float
    duration_limit: float
    goal_auto: bool
    goal_progress_ratio: float
    goal_distance: float
    goal_x: float
    terrain_x_half_size: float
    terrain_y_half_size: float
    terrain_rows: int
    terrain_cols: int
    sim_time: float
    termination_reason: str
    fell: bool
    reached_goal: bool
    final_x: float
    final_y: float
    final_z: float
    final_terrain_height: float
    final_base_height_above_terrain: float
    final_uprightness: float
    progress_x: float
    planar_displacement: float
    path_length: float
    min_base_height: float
    min_base_height_above_terrain: float
    min_uprightness: float
    terrain_min_height: float
    terrain_max_height: float
    terrain_max_neighbor_height_delta: float
    terrain_max_neighbor_slope_degrees: float


@dataclass
class TerrainHeightSampler:
    model: mujoco.MjModel
    hfield_id: int
    geom_id: int
    rows: int
    cols: int
    data_adr: int
    data_size: int
    x_half_size: float
    y_half_size: float
    z_scale: float

    @classmethod
    def from_model(cls, model: mujoco.MjModel) -> TerrainHeightSampler:
        hfield_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_HFIELD,
            TERRAIN_NAME,
        )
        if hfield_id < 0:
            raise RuntimeError(f"Heightfield was not found: {TERRAIN_NAME}")

        geom_id = -1
        for candidate in range(model.ngeom):
            if (
                int(model.geom_type[candidate])
                == int(mujoco.mjtGeom.mjGEOM_HFIELD)
                and int(model.geom_dataid[candidate]) == hfield_id
            ):
                geom_id = candidate
                break
        if geom_id < 0:
            raise RuntimeError(f"Heightfield geom was not found: {TERRAIN_NAME}")

        rows = int(model.hfield_nrow[hfield_id])
        cols = int(model.hfield_ncol[hfield_id])
        return cls(
            model=model,
            hfield_id=hfield_id,
            geom_id=geom_id,
            rows=rows,
            cols=cols,
            data_adr=int(model.hfield_adr[hfield_id]),
            data_size=rows * cols,
            x_half_size=float(model.hfield_size[hfield_id, 0]),
            y_half_size=float(model.hfield_size[hfield_id, 1]),
            z_scale=float(model.hfield_size[hfield_id, 2]),
        )

    def height_at(self, x: float, y: float) -> float:
        geom_pos = self.model.geom_pos[self.geom_id]
        local_x = float(x) - float(geom_pos[0])
        local_y = float(y) - float(geom_pos[1])
        x_grid = np.clip(
            (local_x + self.x_half_size)
            / (2.0 * self.x_half_size)
            * (self.cols - 1),
            0.0,
            self.cols - 1,
        )
        y_grid = np.clip(
            (local_y + self.y_half_size)
            / (2.0 * self.y_half_size)
            * (self.rows - 1),
            0.0,
            self.rows - 1,
        )

        x0 = int(np.floor(x_grid))
        y0 = int(np.floor(y_grid))
        x1 = min(x0 + 1, self.cols - 1)
        y1 = min(y0 + 1, self.rows - 1)
        tx = float(x_grid - x0)
        ty = float(y_grid - y0)

        heights = self.model.hfield_data[
            self.data_adr : self.data_adr + self.data_size
        ].reshape(self.rows, self.cols)
        h00 = float(heights[y0, x0])
        h10 = float(heights[y0, x1])
        h01 = float(heights[y1, x0])
        h11 = float(heights[y1, x1])
        h0 = (1.0 - tx) * h00 + tx * h10
        h1 = (1.0 - tx) * h01 + tx * h11
        normalized_height = (1.0 - ty) * h0 + ty * h1
        return float(geom_pos[2]) + normalized_height * self.z_scale

    def base_height_above_terrain(self, data: mujoco.MjData) -> float:
        terrain_height = self.height_at(float(data.qpos[0]), float(data.qpos[1]))
        return float(data.qpos[2]) - terrain_height


@dataclass
class EpisodeTracker:
    start_xy: np.ndarray
    previous_xy: np.ndarray
    min_base_height: float
    min_base_height_above_terrain: float
    min_uprightness: float
    path_length: float = 0.0

    @classmethod
    def from_data(
        cls,
        data: mujoco.MjData,
        terrain_sampler: TerrainHeightSampler,
    ) -> EpisodeTracker:
        xy = data.qpos[0:2].astype(np.float64, copy=True)
        return cls(
            start_xy=xy.copy(),
            previous_xy=xy.copy(),
            min_base_height=float(data.qpos[2]),
            min_base_height_above_terrain=(
                terrain_sampler.base_height_above_terrain(data)
            ),
            min_uprightness=final_uprightness(data),
        )

    def update(
        self,
        data: mujoco.MjData,
        terrain_sampler: TerrainHeightSampler,
    ) -> None:
        xy = data.qpos[0:2].astype(np.float64, copy=True)
        self.path_length += float(np.linalg.norm(xy - self.previous_xy))
        self.previous_xy[:] = xy
        self.min_base_height = min(self.min_base_height, float(data.qpos[2]))
        self.min_base_height_above_terrain = min(
            self.min_base_height_above_terrain,
            terrain_sampler.base_height_above_terrain(data),
        )
        self.min_uprightness = min(self.min_uprightness, final_uprightness(data))


class LocomotionEvaluationViewer(ContinuousRoughTerrainViewer):
    def __init__(self, model: mujoco.MjModel) -> None:
        super().__init__(model)
        if self._window is not None:
            self._glfw.set_window_title(
                self._window,
                "Go2 rough-terrain locomotion evaluation",
            )
            self._glfw.maximize_window(self._window)
        self._camera.distance = 3.0

    def upload_terrain_visual(self) -> None:
        if self._window is None:
            return

        hfield_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_HFIELD,
            TERRAIN_NAME,
        )
        if hfield_id < 0:
            raise RuntimeError(f"Heightfield was not found: {TERRAIN_NAME}")

        self._glfw.make_context_current(self._window)
        mujoco.mjr_uploadHField(self._model, self._context, hfield_id)

    def _add_goal_connector(
        self,
        start: np.ndarray,
        end: np.ndarray,
        width: float,
    ) -> None:
        if self._scene.ngeom >= self._scene.maxgeom:
            return

        geom = self._scene.geoms[self._scene.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            GOAL_MARKER_ZERO,
            GOAL_MARKER_ZERO,
            GOAL_MARKER_MAT,
            GOAL_MARKER_RGBA,
        )
        mujoco.mjv_connector(
            geom,
            mujoco.mjtGeom.mjGEOM_LINE,
            width,
            start,
            end,
        )
        self._scene.ngeom += 1

    def _add_goal_marker(
        self,
        tracker: EpisodeTracker,
        goal_distance: float,
        terrain_amplitude: float,
    ) -> None:
        goal_x = float(tracker.start_xy[0] + goal_distance)
        hfield_id = mujoco.mj_name2id(
            self._model,
            mujoco.mjtObj.mjOBJ_HFIELD,
            TERRAIN_NAME,
        )
        if hfield_id >= 0:
            terrain_y_half_size = float(self._model.hfield_size[hfield_id, 1])
        else:
            terrain_y_half_size = DEFAULT_TERRAIN_Y_HALF_SIZE
        y_limit = max(0.2, terrain_y_half_size - 0.15)
        line_z = float(terrain_amplitude + 0.05)
        post_bottom_z = float(-terrain_amplitude)
        post_top_z = float(terrain_amplitude + 0.45)

        self._add_goal_connector(
            np.array([goal_x, -y_limit, line_z], dtype=np.float64),
            np.array([goal_x, y_limit, line_z], dtype=np.float64),
            6.0,
        )
        for y in (-y_limit, y_limit):
            self._add_goal_connector(
                np.array([goal_x, y, post_bottom_z], dtype=np.float64),
                np.array([goal_x, y, post_top_z], dtype=np.float64),
                4.0,
            )

    def render_evaluation(
        self,
        data: mujoco.MjData,
        command: np.ndarray,
        episode: int,
        episodes: int,
        terrain_seed: int,
        terrain_amplitude: float,
        tracker: EpisodeTracker,
        elapsed: float,
        duration: float,
        goal_distance: float,
        termination_reason: str,
        fell: bool,
        completed: int,
        falls: int,
        last_result: EpisodeResult | None,
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
        self._add_goal_marker(tracker, goal_distance, terrain_amplitude)
        mujoco.mjr_render(viewport, self._scene, self._context)

        displacement = data.qpos[0:2] - tracker.start_xy
        status = "fall" if fell else termination_reason
        left = (
            "Go2 rough-terrain auto evaluation\n"
            "Esc: quit   Left drag: rotate camera   Wheel: zoom\n"
            f"episode={episode}/{episodes}  seed={terrain_seed}  "
            f"amp={terrain_amplitude:.3f}m\n"
            f"time={elapsed:.2f}/{duration:.2f}s  status={status}\n"
            f"cmd vx={command[0]:+.2f}  vy={command[1]:+.2f}  "
            f"yaw={command[2]:+.2f}"
        )
        right = (
            f"successes={completed}  falls={falls}\n"
            f"x={data.qpos[0]:+.2f}  y={data.qpos[1]:+.2f}  "
            f"z={data.qpos[2]:+.2f}\n"
            f"progress_x={displacement[0]:+.2f}/{goal_distance:+.2f}m  "
            f"path={tracker.path_length:.2f}m\n"
            f"upright={final_uprightness(data):+.2f}  "
            f"min_rel_z={tracker.min_base_height_above_terrain:.2f}  "
            f"min_up={tracker.min_uprightness:+.2f}"
        )
        if last_result is not None:
            right += (
                "\nlast: "
                f"ep={last_result.episode} "
                f"{last_result.termination_reason} "
                f"dx={last_result.progress_x:+.2f}m "
                f"min_up={last_result.min_uprightness:+.2f}"
            )

        mujoco.mjr_overlay(
            mujoco.mjtFontScale.mjFONTSCALE_150,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            left,
            right,
            self._context,
        )

        self._glfw.swap_buffers(self._window)
        self._glfw.poll_events()


def normalize_cli_args(argv: list[str]) -> list[str]:
    normalized_args: list[str] = []
    changed = False
    for arg in argv:
        parts = [arg]
        for separator in CLI_SPACE_SEPARATORS:
            split_parts: list[str] = []
            for part in parts:
                split_parts.extend(part.split(separator))
            parts = split_parts
        cleaned_parts = [part for part in parts if part]
        if cleaned_parts != [arg]:
            changed = True
        normalized_args.extend(cleaned_parts)

    if changed:
        print(
            "Note: normalized full-width/non-breaking spaces in command-line arguments.",
            file=sys.stderr,
        )
    return normalized_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visual multi-episode evaluation for the Go2 flat policy on a "
            "continuous rough MuJoCo heightfield."
        )
    )
    parser.add_argument("--display", default=DEFAULT_DISPLAY)
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=POLICY_DIR,
        help="Directory containing policy.onnx, policy.onnx.data, and params/deploy.yaml.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening a viewer. Useful for smoke tests.",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--render-fps", type=float, default=DEFAULT_RENDER_FPS)
    parser.add_argument(
        "--terrain-seed",
        default=DEFAULT_TERRAIN_SEED,
        help=(
            "Terrain random seed, or 'random' for fresh random terrain. "
            "With evaluation episodes, 'random' samples a new seed per episode."
        ),
    )
    parser.add_argument("--terrain-seed-stride", type=int, default=1)
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
    )
    parser.add_argument(
        "--start-flat-radius",
        type=float,
        default=DEFAULT_START_FLAT_RADIUS,
    )
    parser.add_argument("--test-command-vx", type=float, default=DEFAULT_FORWARD_COMMAND)
    parser.add_argument("--test-command-vy", type=float, default=0.0)
    parser.add_argument("--test-command-yaw", type=float, default=0.0)
    parser.add_argument(
        "--goal-distance",
        default=DEFAULT_GOAL_DISTANCE,
        help=(
            "Signed x displacement from the episode start required for success, "
            "or 'auto'. Auto uses test-command-vx * duration * "
            "--goal-progress-ratio."
        ),
    )
    parser.add_argument(
        "--goal-progress-ratio",
        type=float,
        default=DEFAULT_GOAL_PROGRESS_RATIO,
        help="Auto goal distance ratio against ideal commanded x displacement.",
    )
    parser.add_argument(
        "--command-smoothing",
        type=float,
        default=12.0,
        help="First-order smoothing rate for the automatic command.",
    )
    parser.add_argument(
        "--fall-height",
        type=float,
        default=DEFAULT_FALL_HEIGHT,
        help=(
            "Terminate an episode when base height above the local terrain "
            "surface drops below this value."
        ),
    )
    parser.add_argument(
        "--fall-uprightness",
        type=float,
        default=DEFAULT_FALL_UPRIGHTNESS,
        help="Terminate an episode when root z-axis uprightness drops below this value.",
    )
    parser.add_argument(
        "--fall-warmup",
        type=float,
        default=DEFAULT_FALL_WARMUP,
        help="Seconds before fall checks become active.",
    )
    parser.add_argument(
        "--episode-pause",
        type=float,
        default=DEFAULT_EPISODE_PAUSE,
        help="Seconds to keep showing a finished episode before resetting.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for JSONL evaluation logs.",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help="Persistent CSV file that accumulates one row per episode.",
    )
    parser.add_argument(
        "--results-md",
        type=Path,
        default=DEFAULT_RESULTS_MD,
        help="Persistent Japanese Markdown summary regenerated from the CSV.",
    )
    parser.add_argument(
        "--results-reset",
        action="store_true",
        help="Clear persistent CSV/Markdown before running this evaluation.",
    )
    parser.add_argument(
        "--results-reset-only",
        action="store_true",
        help="Clear persistent CSV/Markdown and exit without simulation.",
    )
    parser.add_argument(
        "--results-delete-run",
        action="append",
        default=[],
        metavar="RUN_ID",
        help="Delete all persisted rows for a run_id and exit. Repeatable.",
    )
    parser.add_argument(
        "--results-delete-row",
        action="append",
        default=[],
        metavar="ROW_ID",
        help="Delete one persisted episode row_id and exit. Repeatable.",
    )
    parser.add_argument(
        "--results-summary-only",
        action="store_true",
        help="Regenerate the Markdown summary from the persistent CSV and exit.",
    )
    return parser.parse_args(normalize_cli_args(sys.argv[1:]))


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("Episodes must be positive.")
    if args.duration <= 0.0:
        raise ValueError("Duration must be positive.")
    if args.render_fps <= 0.0:
        raise ValueError("Render FPS must be positive.")
    if args.terrain_amplitude <= 0.0:
        raise ValueError("Terrain amplitude must be positive.")
    if args.terrain_smoothing_passes < 0:
        raise ValueError("Terrain smoothing passes must be non-negative.")
    if args.start_flat_radius < 0.0:
        raise ValueError("Start flat radius must be non-negative.")
    if not 0.0 < args.goal_progress_ratio <= 1.0:
        raise ValueError("Goal progress ratio must be in (0, 1].")
    if args.command_smoothing < 0.0:
        raise ValueError("Command smoothing must be non-negative.")
    if args.fall_height <= 0.0:
        raise ValueError("Fall height must be positive.")
    if not -1.0 <= args.fall_uprightness <= 1.0:
        raise ValueError("Fall uprightness must be in [-1, 1].")
    if args.fall_warmup < 0.0:
        raise ValueError("Fall warmup must be non-negative.")
    if args.episode_pause < 0.0:
        raise ValueError("Episode pause must be non-negative.")


def resolve_goal_distance(args: argparse.Namespace) -> None:
    raw_goal_distance = args.goal_distance
    if (
        isinstance(raw_goal_distance, str)
        and raw_goal_distance.strip().lower() == "auto"
    ):
        if abs(args.test_command_vx) <= 1e-9:
            raise ValueError(
                "Automatic goal distance requires non-zero --test-command-vx. "
                "Set --goal-distance manually for this command."
            )
        args.goal_distance_auto = True
        args.goal_distance = (
            float(args.test_command_vx)
            * float(args.duration)
            * float(args.goal_progress_ratio)
        )
    else:
        try:
            args.goal_distance = float(raw_goal_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Goal distance must be a number or 'auto'."
            ) from exc
        args.goal_distance_auto = False

    if abs(args.goal_distance) <= 1e-9:
        raise ValueError("Goal distance must be non-zero.")


def terrain_x_half_size_for_goal(goal_distance: float) -> float:
    return max(DEFAULT_TERRAIN_X_HALF_SIZE, abs(goal_distance) + TERRAIN_GOAL_MARGIN)


def terrain_y_half_size_for_command(args: argparse.Namespace) -> float:
    lateral_travel = abs(float(args.test_command_vy)) * float(args.duration)
    return max(DEFAULT_TERRAIN_Y_HALF_SIZE, lateral_travel + TERRAIN_GOAL_MARGIN)


def make_log_path(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return log_dir / f"locomotion_view_{timestamp}.jsonl"


def make_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{timestamp}_{secrets.token_hex(3)}"


def status_label(result: EpisodeResult) -> str:
    if result.reached_goal:
        return "成功"
    if result.fell:
        return "転倒"
    if result.termination_reason == "timeout":
        return "時間切れ"
    if result.termination_reason == "viewer_closed":
        return "中断"
    return "失敗"


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1e-12:
        return 0.0
    return numerator / denominator


def history_row(
    result: EpisodeResult,
    args: argparse.Namespace,
    run_id: str,
    run_started_at: str,
    log_path: Path,
    episodes_planned: int,
    terrain_seed_random: bool,
) -> dict[str, object]:
    progress_goal_ratio = safe_ratio(result.progress_x, result.goal_distance)
    path_efficiency = safe_ratio(abs(result.progress_x), result.path_length)
    average_forward_speed = safe_ratio(result.progress_x, result.sim_time)
    row_id = f"{run_id}-ep{result.episode:04d}"
    return {
        "row_id": row_id,
        "run_id": run_id,
        "run_started_at": run_started_at,
        "result_jsonl": str(log_path),
        "episode": result.episode,
        "episodes_planned": episodes_planned,
        "policy_dir": str(args.policy_dir),
        "headless": bool(args.headless),
        "success": bool(result.reached_goal),
        "status_jp": status_label(result),
        "termination_reason": result.termination_reason,
        "fell": bool(result.fell),
        "reached_goal": bool(result.reached_goal),
        "terrain_seed_random": terrain_seed_random,
        "terrain_seed": result.terrain_seed,
        "terrain_seed_stride": args.terrain_seed_stride,
        "terrain_amplitude": result.terrain_amplitude,
        "terrain_smoothing_passes": result.terrain_smoothing_passes,
        "start_flat_radius": args.start_flat_radius,
        "terrain_x_half_size": result.terrain_x_half_size,
        "terrain_y_half_size": result.terrain_y_half_size,
        "terrain_rows": result.terrain_rows,
        "terrain_cols": result.terrain_cols,
        "terrain_min_height": result.terrain_min_height,
        "terrain_max_height": result.terrain_max_height,
        "terrain_max_neighbor_height_delta": result.terrain_max_neighbor_height_delta,
        "terrain_max_neighbor_slope_degrees": (
            result.terrain_max_neighbor_slope_degrees
        ),
        "command_vx": result.command_vx,
        "command_vy": result.command_vy,
        "command_yaw": result.command_yaw,
        "command_smoothing": args.command_smoothing,
        "duration_limit": result.duration_limit,
        "sim_time": result.sim_time,
        "goal_auto": bool(result.goal_auto),
        "goal_progress_ratio": result.goal_progress_ratio,
        "goal_distance": result.goal_distance,
        "goal_x": result.goal_x,
        "progress_x": result.progress_x,
        "progress_goal_ratio": progress_goal_ratio,
        "planar_displacement": result.planar_displacement,
        "path_length": result.path_length,
        "path_efficiency": path_efficiency,
        "average_forward_speed": average_forward_speed,
        "final_x": result.final_x,
        "final_y": result.final_y,
        "final_z": result.final_z,
        "final_terrain_height": result.final_terrain_height,
        "final_base_height_above_terrain": result.final_base_height_above_terrain,
        "final_uprightness": result.final_uprightness,
        "min_base_height": result.min_base_height,
        "min_base_height_above_terrain": result.min_base_height_above_terrain,
        "min_uprightness": result.min_uprightness,
        "fall_height": args.fall_height,
        "fall_uprightness": args.fall_uprightness,
        "fall_warmup": args.fall_warmup,
    }


def read_history_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_history_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULT_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_history_rows(csv_path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not csv_path.exists() or csv_path.stat().st_size == 0
    if not needs_header:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.reader(csv_file)
            existing_header = next(reader, [])
        if existing_header != RESULT_FIELDNAMES:
            existing_rows = read_history_rows(csv_path)
            write_history_rows(csv_path, existing_rows + rows)
            return
    with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=RESULT_FIELDNAMES,
            extrasaction="ignore",
        )
        if needs_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def numeric(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "")
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(rows: list[dict[str, str]], key: str) -> float:
    values = [value for row in rows if (value := numeric(row, key)) is not None]
    if not values:
        return 0.0
    return sum(values) / len(values)


def percent(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return 100.0 * part / total


def success_count(rows: list[dict[str, str]]) -> int:
    return sum(1 for row in rows if is_true(row.get("success")))


def run_summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("run_id", ""), []).append(row)

    summaries = []
    for run_id, run_rows in grouped.items():
        summaries.append(
            {
                "run_id": run_id,
                "started_at": run_rows[0].get("run_started_at", ""),
                "rows": run_rows,
            }
        )
    summaries.sort(key=lambda item: str(item["started_at"]), reverse=True)
    return summaries


def condition_key(row: dict[str, str]) -> tuple[str, ...]:
    goal = (
        f"auto:{row.get('goal_progress_ratio', '')}"
        if is_true(row.get("goal_auto"))
        else f"fixed:{row.get('goal_distance', '')}"
    )
    return (
        row.get("command_vx", ""),
        row.get("command_vy", ""),
        row.get("command_yaw", ""),
        row.get("terrain_amplitude", ""),
        row.get("terrain_smoothing_passes", ""),
        goal,
    )


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return ["該当データはまだありません。"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def write_markdown_summary(csv_path: Path, md_path: Path) -> None:
    rows = read_history_rows(csv_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Go2 歩行評価 結果サマリ",
        "",
        f"生成時刻: `{generated_at}`",
        f"CSV: `{csv_path}`",
        "",
    ]

    if not rows:
        lines.extend(
            [
                "まだ蓄積された評価結果はありません。",
                "",
                "リセット直後、または評価スクリプトをまだ実行していない状態です。",
            ]
        )
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    total = len(rows)
    successes = success_count(rows)
    falls = sum(1 for row in rows if is_true(row.get("fell")))
    timeouts = sum(1 for row in rows if row.get("termination_reason") == "timeout")
    run_count = len({row.get("run_id", "") for row in rows})
    has_relative_height = any(
        row.get("min_base_height_above_terrain") for row in rows
    )
    min_height_key = (
        "min_base_height_above_terrain" if has_relative_height else "min_base_height"
    )
    min_height_label = (
        "平均最小相対ベース高さ"
        if has_relative_height
        else "平均最小ベース絶対高さ"
    )
    lines.extend(
        [
            "## 全体サマリ",
            "",
            f"- 総エピソード数: **{total}**",
            f"- 実行回数: **{run_count}**",
            f"- 成功率: **{percent(successes, total):.1f}%** ({successes}/{total})",
            f"- 転倒: **{falls}**",
            f"- 時間切れ: **{timeouts}**",
            f"- 平均進捗 x: **{fmt(mean(rows, 'progress_x'))} m**",
            f"- 平均ゴール距離: **{fmt(mean(rows, 'goal_distance'))} m**",
            f"- 平均経路長: **{fmt(mean(rows, 'path_length'))} m**",
            f"- 平均前進速度: **{fmt(mean(rows, 'average_forward_speed'))} m/s**",
            f"- 平均最小直立度: **{fmt(mean(rows, 'min_uprightness'))}**",
            f"- {min_height_label}: **{fmt(mean(rows, min_height_key))} m**",
            "",
            "## 条件別サマリ",
            "",
        ]
    )

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(condition_key(row), []).append(row)
    condition_rows = []
    for key, group_rows in sorted(
        grouped.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )[:30]:
        vx, vy, yaw, amp, smoothing, goal = key
        condition_rows.append(
            [
                f"vx={vx}, vy={vy}, yaw={yaw}",
                f"amp={amp}, smooth={smoothing}",
                goal,
                str(len(group_rows)),
                f"{percent(success_count(group_rows), len(group_rows)):.1f}%",
                fmt(mean(group_rows, "progress_x")),
                fmt(mean(group_rows, "min_uprightness")),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "指令",
                "地形",
                "ゴール",
                "n",
                "成功率",
                "平均進捗x[m]",
                "平均min_up",
            ],
            condition_rows,
        )
    )
    lines.extend(["", "## 最近の実行", ""])

    recent_rows = []
    for summary in run_summaries(rows)[:12]:
        run_rows = summary["rows"]
        assert isinstance(run_rows, list)
        recent_rows.append(
            [
                str(summary["run_id"]),
                str(summary["started_at"]),
                str(len(run_rows)),
                f"{percent(success_count(run_rows), len(run_rows)):.1f}%",
                fmt(mean(run_rows, "command_vx"), 2),
                fmt(mean(run_rows, "goal_distance"), 2),
                fmt(mean(run_rows, "terrain_amplitude"), 3),
            ]
        )
    lines.extend(
        markdown_table(
            [
                "run_id",
                "開始",
                "n",
                "成功率",
                "平均vx",
                "平均ゴール[m]",
                "地形amp",
            ],
            recent_rows,
        )
    )

    lines.extend(
        [
            "",
            "## メンテナンス",
            "",
            "```bash",
            "python3 scripts/evaluate_locomotion_viewer.py --results-summary-only",
            "python3 scripts/evaluate_locomotion_viewer.py --results-reset-only",
            "python3 scripts/evaluate_locomotion_viewer.py --results-delete-run RUN_ID",
            "python3 scripts/evaluate_locomotion_viewer.py --results-delete-row ROW_ID",
            "```",
            "",
            "CSV には 1 エピソード 1 行で詳細値を保存しています。Markdown は CSV から再生成されます。",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset_results(csv_path: Path, md_path: Path) -> None:
    write_history_rows(csv_path, [])
    write_markdown_summary(csv_path, md_path)


def delete_result_rows(
    csv_path: Path,
    md_path: Path,
    run_ids: list[str],
    row_ids: list[str],
) -> int:
    rows = read_history_rows(csv_path)
    run_set = set(run_ids)
    row_set = set(row_ids)
    kept = [
        row
        for row in rows
        if row.get("run_id") not in run_set and row.get("row_id") not in row_set
    ]
    removed = len(rows) - len(kept)
    write_history_rows(csv_path, kept)
    write_markdown_summary(csv_path, md_path)
    return removed


def process_results_maintenance(args: argparse.Namespace) -> bool:
    if args.results_reset or args.results_reset_only:
        reset_results(args.results_csv, args.results_md)
        print(f"Results reset: {args.results_csv}")
        print(f"Markdown summary: {args.results_md}")
        if args.results_reset_only:
            return True

    if args.results_delete_run or args.results_delete_row:
        removed = delete_result_rows(
            args.results_csv,
            args.results_md,
            args.results_delete_run,
            args.results_delete_row,
        )
        print(f"Deleted persisted result rows: {removed}")
        print(f"CSV: {args.results_csv}")
        print(f"Markdown summary: {args.results_md}")
        return True

    if args.results_summary_only:
        write_markdown_summary(args.results_csv, args.results_md)
        print(f"Markdown summary regenerated: {args.results_md}")
        return True

    return False


def episode_result(
    episode: int,
    terrain_seed: int,
    args: argparse.Namespace,
    terrain_sampler: TerrainHeightSampler,
    data: mujoco.MjData,
    tracker: EpisodeTracker,
    terrain_stats,
    termination_reason: str,
    fell: bool,
) -> EpisodeResult:
    displacement = data.qpos[0:2] - tracker.start_xy
    goal_distance = float(args.goal_distance)
    final_terrain_height = terrain_sampler.height_at(
        float(data.qpos[0]),
        float(data.qpos[1]),
    )
    return EpisodeResult(
        episode=episode,
        terrain_seed=terrain_seed,
        terrain_amplitude=float(args.terrain_amplitude),
        terrain_smoothing_passes=int(args.terrain_smoothing_passes),
        command_vx=float(args.test_command_vx),
        command_vy=float(args.test_command_vy),
        command_yaw=float(args.test_command_yaw),
        duration_limit=float(args.duration),
        goal_auto=bool(args.goal_distance_auto),
        goal_progress_ratio=float(args.goal_progress_ratio),
        goal_distance=goal_distance,
        goal_x=float(tracker.start_xy[0] + goal_distance),
        terrain_x_half_size=float(terrain_stats.x_half_size),
        terrain_y_half_size=float(terrain_stats.y_half_size),
        terrain_rows=int(terrain_stats.rows),
        terrain_cols=int(terrain_stats.cols),
        sim_time=float(data.time),
        termination_reason=termination_reason,
        fell=fell,
        reached_goal=termination_reason == "goal_reached",
        final_x=float(data.qpos[0]),
        final_y=float(data.qpos[1]),
        final_z=float(data.qpos[2]),
        final_terrain_height=final_terrain_height,
        final_base_height_above_terrain=(
            float(data.qpos[2]) - final_terrain_height
        ),
        final_uprightness=final_uprightness(data),
        progress_x=float(displacement[0]),
        planar_displacement=float(np.linalg.norm(displacement)),
        path_length=float(tracker.path_length),
        min_base_height=float(tracker.min_base_height),
        min_base_height_above_terrain=float(tracker.min_base_height_above_terrain),
        min_uprightness=float(tracker.min_uprightness),
        terrain_min_height=float(terrain_stats.min_height),
        terrain_max_height=float(terrain_stats.max_height),
        terrain_max_neighbor_height_delta=float(
            terrain_stats.max_neighbor_height_delta
        ),
        terrain_max_neighbor_slope_degrees=float(
            terrain_stats.max_neighbor_slope_degrees
        ),
    )


def write_result(log_file, result: EpisodeResult) -> None:
    log_file.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    log_file.flush()


def should_stop_for_fall(
    data: mujoco.MjData,
    args: argparse.Namespace,
    terrain_sampler: TerrainHeightSampler,
) -> tuple[bool, str]:
    if data.time < args.fall_warmup:
        return False, ""
    if terrain_sampler.base_height_above_terrain(data) < args.fall_height:
        return True, "fall_height"
    if final_uprightness(data) < args.fall_uprightness:
        return True, "fall_uprightness"
    return False, ""


def has_reached_goal(data: mujoco.MjData, tracker: EpisodeTracker, args) -> bool:
    progress_x = float(data.qpos[0] - tracker.start_xy[0])
    if args.goal_distance > 0.0:
        return progress_x >= args.goal_distance
    return progress_x <= args.goal_distance


def pause_after_episode(
    viewer: LocomotionEvaluationViewer | None,
    data: mujoco.MjData,
    command: np.ndarray,
    episode: int,
    args: argparse.Namespace,
    terrain_seed: int,
    tracker: EpisodeTracker,
    result: EpisodeResult,
    completed: int,
    falls: int,
    last_result: EpisodeResult | None,
) -> bool:
    if viewer is None or args.episode_pause <= 0.0:
        return True

    render_interval = 1.0 / args.render_fps
    pause_start = time.perf_counter()
    next_render = pause_start
    while time.perf_counter() - pause_start < args.episode_pause:
        if not viewer.is_running():
            return False
        now = time.perf_counter()
        if now >= next_render:
            viewer.render_evaluation(
                data,
                command,
                episode,
                args.episodes,
                terrain_seed,
                args.terrain_amplitude,
                tracker,
                data.time,
                args.duration,
                args.goal_distance,
                result.termination_reason,
                result.fell,
                completed,
                falls,
                last_result,
            )
            next_render += render_interval
        time.sleep(min(render_interval, 0.01))
    return viewer.is_running()


def run_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    policy: SimToRealPolicyController,
    viewer: LocomotionEvaluationViewer | None,
    args: argparse.Namespace,
    episode: int,
    terrain_seed: int,
    terrain_stats,
    terrain_sampler: TerrainHeightSampler,
    completed: int,
    falls: int,
    last_result: EpisodeResult | None,
) -> tuple[EpisodeResult, EpisodeTracker]:
    policy.initialize_pose(data)
    tracker = EpisodeTracker.from_data(data, terrain_sampler)

    command = np.zeros(3, dtype=np.float32)
    target_command = np.array(
        [args.test_command_vx, args.test_command_vy, args.test_command_yaw],
        dtype=np.float32,
    )
    active_motion = bool(np.linalg.norm(target_command) > 1e-6)
    next_policy_time = data.time
    next_step_wall = time.perf_counter()
    render_interval = 1.0 / args.render_fps
    next_render_wall = next_step_wall
    termination_reason = "timeout"
    fell = False

    while data.time < args.duration:
        now = time.perf_counter()
        if viewer is not None and not viewer.is_running():
            termination_reason = "viewer_closed"
            break
        if not args.headless and now < next_step_wall:
            time.sleep(min(next_step_wall - now, model.opt.timestep))
            continue

        data.xfrc_applied[:] = 0.0
        command = update_command_with_release_cutoff(
            command,
            target_command,
            model.opt.timestep,
            args.command_smoothing,
            active_motion,
        )

        if data.time >= next_policy_time:
            policy.update_policy(data, command)
            next_policy_time += policy.step_dt

        policy.apply_pd(data)
        mujoco.mj_step(model, data)
        tracker.update(data, terrain_sampler)

        fall_detected, fall_reason = should_stop_for_fall(
            data,
            args,
            terrain_sampler,
        )
        if fall_detected:
            fell = True
            termination_reason = fall_reason

        if not fell and has_reached_goal(data, tracker, args):
            termination_reason = "goal_reached"

        if viewer is not None and now >= next_render_wall:
            viewer.render_evaluation(
                data,
                command,
                episode,
                args.episodes,
                terrain_seed,
                args.terrain_amplitude,
                tracker,
                data.time,
                args.duration,
                args.goal_distance,
                termination_reason,
                fell,
                completed,
                falls,
                last_result,
            )
            next_render_wall += render_interval
            if next_render_wall < now - render_interval:
                next_render_wall = now

        if fell or termination_reason == "goal_reached":
            break

        next_step_wall += model.opt.timestep
        if next_step_wall < now - 0.1:
            next_step_wall = now

    return (
        episode_result(
            episode,
            terrain_seed,
            args,
            terrain_sampler,
            data,
            tracker,
            terrain_stats,
            termination_reason,
            fell,
        ),
        tracker,
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    if process_results_maintenance(args):
        return

    resolve_goal_distance(args)
    terrain_seed_random = (
        isinstance(args.terrain_seed, str)
        and args.terrain_seed.strip().lower() == "random"
    )
    if not terrain_seed_random:
        args.terrain_seed, _ = resolve_terrain_seed(args.terrain_seed)
    if not args.headless:
        os.environ["DISPLAY"] = args.display

    terrain_x_half_size = terrain_x_half_size_for_goal(args.goal_distance)
    terrain_y_half_size = terrain_y_half_size_for_command(args)
    runtime_scene = create_runtime_scene(
        MODEL_DIR,
        args.terrain_amplitude,
        terrain_x_half_size=terrain_x_half_size,
        terrain_y_half_size=terrain_y_half_size,
    )
    viewer = None
    try:
        model = mujoco.MjModel.from_xml_path(str(runtime_scene.scene_path))
        data = mujoco.MjData(model)
        policy = SimToRealPolicyController(model, args.policy_dir)
        terrain_sampler = TerrainHeightSampler.from_model(model)
        if not args.headless:
            viewer = LocomotionEvaluationViewer(model)

        run_id = make_run_id()
        run_started_at = datetime.now().isoformat(timespec="seconds")
        log_path = make_log_path(args.log_dir)
        print(f"Scene: {runtime_scene.scene_path}")
        print(f"Policy: {args.policy_dir}")
        print(f"Log: {log_path}")
        print(f"Run ID: {run_id}")
        print(f"Results CSV: {args.results_csv}")
        print(f"Results Markdown: {args.results_md}")
        print(
            "Evaluation command: "
            f"vx={args.test_command_vx:+.2f}, "
            f"vy={args.test_command_vy:+.2f}, "
            f"yaw={args.test_command_yaw:+.2f}"
        )
        goal_mode = (
            f"auto {args.goal_progress_ratio:.0%} of vx*duration"
            if args.goal_distance_auto
            else "manual"
        )
        print(f"Goal line: progress_x={args.goal_distance:+.2f}m ({goal_mode})")
        if terrain_seed_random:
            print("Terrain seed: random per episode")
        else:
            print(
                "Terrain seed: "
                f"base={args.terrain_seed}, stride={args.terrain_seed_stride}"
            )
        print(
            "Terrain field: "
            f"x=[{-terrain_x_half_size:.1f}, {terrain_x_half_size:.1f}]m "
            f"y=[{-terrain_y_half_size:.1f}, {terrain_y_half_size:.1f}]m "
            f"with {TERRAIN_GOAL_MARGIN:.1f}m margin"
        )
        print(
            "Fall thresholds: "
            f"base-terrain height<{args.fall_height:.2f}m, "
            f"uprightness<{args.fall_uprightness:.2f}"
        )

        completed = 0
        falls = 0
        episodes_run = 0
        last_result = None
        history_rows = []
        with log_path.open("w", encoding="utf-8") as log_file:
            for episode in range(1, args.episodes + 1):
                if terrain_seed_random:
                    terrain_seed, _ = resolve_terrain_seed("random")
                else:
                    terrain_seed = (
                        args.terrain_seed + (episode - 1) * args.terrain_seed_stride
                    )
                terrain_stats = configure_heightfield(
                    model,
                    terrain_seed,
                    args.terrain_smoothing_passes,
                    args.start_flat_radius,
                    args.terrain_amplitude,
                )
                if viewer is not None:
                    viewer.upload_terrain_visual()
                result, tracker = run_episode(
                    model,
                    data,
                    policy,
                    viewer,
                    args,
                    episode,
                    terrain_seed,
                    terrain_stats,
                    terrain_sampler,
                    completed,
                    falls,
                    last_result,
                )
                write_result(log_file, result)
                history_rows.append(
                    history_row(
                        result,
                        args,
                        run_id,
                        run_started_at,
                        log_path,
                        args.episodes,
                        terrain_seed_random,
                    )
                )
                episodes_run += 1
                if result.fell:
                    falls += 1
                elif result.reached_goal:
                    completed += 1

                print(
                    f"episode={result.episode} seed={result.terrain_seed} "
                    f"reason={result.termination_reason} "
                    f"goal={result.goal_distance:+.3f}m "
                    f"reached_goal={result.reached_goal} "
                    f"progress_x={result.progress_x:+.3f}m "
                    f"path={result.path_length:.3f}m "
                    f"min_rel_z={result.min_base_height_above_terrain:.3f} "
                    f"min_up={result.min_uprightness:+.3f}"
                )

                should_continue = pause_after_episode(
                    viewer,
                    data,
                    np.array(
                        [
                            args.test_command_vx,
                            args.test_command_vy,
                            args.test_command_yaw,
                        ],
                        dtype=np.float32,
                    ),
                    episode,
                    args,
                    terrain_seed,
                    tracker,
                    result,
                    completed,
                    falls,
                    last_result,
                )
                last_result = result
                if not should_continue or result.termination_reason == "viewer_closed":
                    break

        print(
            "Summary: "
            f"episodes_run={episodes_run} "
            f"successes={completed} falls={falls} log={log_path}"
        )
        append_history_rows(args.results_csv, history_rows)
        write_markdown_summary(args.results_csv, args.results_md)
        print(f"Persistent results appended: {args.results_csv}")
        print(f"Markdown summary updated: {args.results_md}")
    finally:
        if viewer is not None:
            viewer.close()
        runtime_scene.cleanup()


if __name__ == "__main__":
    main()
