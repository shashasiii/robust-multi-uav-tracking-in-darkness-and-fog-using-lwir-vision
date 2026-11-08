# -*- coding: UTF-8 -*-
"""
One-pass online tracking postprocess runner for image folders.

This script keeps the stable YOLO + BoT-SORT baseline, then applies causal
postprocessing that is compatible with future real-time deployment:
confidence-aware pending confirmation and display-layer ID handling.
"""
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import argparse
import csv
import math
import re
import sys

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


# =========================================================
# 1) User configuration
# =========================================================

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_IMG_ROOT = REPO_ROOT / "outputs" / "tracking" / "images"
OUTPUT_VIDEO_ROOT = REPO_ROOT / "outputs" / "tracking" / "artifacts"

FPS = 25

DEVICE = "0"
CONF = 0.30
IMGSZ = None
VERBOSE = False
CLASSES = [0, 1]
CLASS_NAMES = {0: "drone", 1: "person"}
DISPLAY_PREFIXES = {0: "D", 1: "P"}
BOX_COLORS = {
    0: (0, 220, 255),
    1: (80, 220, 80),
}

TRACKER = str(Path(__file__).with_name("botsort_uav_stable.yaml"))
IOU = 0.7

SHOW_PREVIEW = False
SAVE_WITH_ORIGINAL_NAME = True


# =========================================================
# 2) Online postprocess parameters
# =========================================================

FAST_CONFIRM_FRAMES = 2
FAST_CONFIRM_MEAN_CONF = 0.85
MID_CONFIRM_FRAMES = 4
MID_CONFIRM_MEAN_CONF = 0.78
MID_CONFIRM_MIN_CONF = 0.75
MID_CONFIRM_MIN_HITS = 3

PENDING_WINDOW = 8
STRONG_CONF = 0.80
STRONG_HIT_COUNT = 4

FALLBACK_HITS = 8
FALLBACK_MEAN_CONF = 0.72

MAX_LOST_FRAMES_TO_KEEP_STATE = 30

REENTRY_MAX_GAP_FRAMES = 23
REENTRY_EDGE_MARGIN_PX = 40.0
REENTRY_CENTER_DISTANCE_PX = 75.0
REENTRY_AREA_RATIO_MIN = 0.50
REENTRY_AREA_RATIO_MAX = 2.00
REENTRY_AMBIGUOUS_SCORE_MARGIN = 0.20

MOTION_MIN_CONF = 0.50
MOTION_MIN_GAP_FRAMES = 2
MOTION_MAHALANOBIS_THRESHOLD = 9.21
MOTION_DISTANCE_BASE_PX = 15.0
MOTION_DISTANCE_PER_GAP_PX = 4.0
MOTION_FAST_REJOIN_MAX_GAP_FRAMES = 5
MOTION_FAST_REJOIN_MIN_CONF = 0.85
MOTION_FAST_REJOIN_DISTANCE_PX = 80.0
MOTION_ABSOLUTE_DIRECT_DISTANCE_PX = 80.0
MOTION_INITIAL_CANDIDATE_MAX_GAP_FRAMES = 15
MOTION_FAST_DIRECT_MAX_GAP_FRAMES = 6
MOTION_FAST_DIRECT_DISTANCE_PX = 30.0
MOTION_FAST_DIRECT_MAHALANOBIS_THRESHOLD = 4.61
MOTION_EDGE_FAST_DIRECT_DISTANCE_PX = 45.0
MOTION_EDGE_FAST_DIRECT_MAHALANOBIS_THRESHOLD = 9.21
MOTION_SLOW_WINDOW_FRAMES = 8
MOTION_SLOW_BASE_WINDOW_FRAMES = 6
MOTION_SLOW_MISS_TOLERANCE_FRAMES = 2
MOTION_SLOW_MIN_OBSERVATIONS = 3
MOTION_SLOW_COST_MARGIN = 0.50
MOTION_EDGE_MIN_INWARD_PX = 3.0
MOTION_EDGE_AREA_RATIO_MIN = 0.20
MOTION_EDGE_AREA_RATIO_MAX = 5.00
MOTION_ACCEL_STD_PX = 3.0
MOTION_MEASUREMENT_STD_PX = 4.0
MOTION_INITIAL_POSITION_STD_PX = 8.0
MOTION_INITIAL_VELOCITY_STD_PX = 8.0


# =========================================================
# 3) Image sorting rules
# =========================================================

TIMESTAMP_PATTERN = re.compile(
    r"thermal_inferno_image_(\d+)_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$",
    re.IGNORECASE,
)

VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


RAW_CSV_FIELDS = [
    "frame_idx",
    "image_name",
    "track_id",
    "class_id",
    "class_name",
    "conf",
    "x1",
    "y1",
    "x2",
    "y2",
    "frame_width",
    "frame_height",
]

ONLINE_CSV_FIELDS = [
    "frame_idx",
    "image_name",
    "track_id",
    "display_id",
    "display_label",
    "class_id",
    "class_name",
    "conf",
    "raw_x1",
    "raw_y1",
    "raw_x2",
    "raw_y2",
    "total_seen",
    "pending_seen",
    "mean_conf",
    "recent_strong_hits",
    "status",
    "status_reason",
    "visibility_approved",
    "visibility_confirmed_frame",
    "visibility_confirm_reason",
    "identity_status",
    "reentry_source_raw_track_id",
    "reentry_gap_frames",
    "reentry_match_score",
    "motion_model",
    "motion_center_distance",
    "motion_direct_distance",
    "motion_mahalanobis_sq",
    "motion_area_ratio",
    "frame_width",
    "frame_height",
]

INHERITANCE_AUDIT_FIELDS = [
    "frame_idx", "event_type", "reason", "class_id", "display_id",
    "source_raw_track_id", "candidate_raw_track_id", "source_last_seen_frame",
    "gap_frames", "candidate_conf", "candidate_status", "visibility_approved",
    "visibility_level", "coarse_eligible", "strict_eligible", "direct_distance",
    "area_ratio", "score", "observation_count", "mean_score", "missed_frames",
    "required_window_frames",
]


@dataclass
class OnlineTrackState:
    track_id: int
    class_id: int
    status: str = "pending"
    first_seen_frame: int = -1
    last_seen_frame: int = -1
    total_seen: int = 0
    pending_seen: int = 0
    confirmed_frame: int | None = None
    lost_frames: int = 0
    conf_history: list[float] = field(default_factory=list)
    strong_conf_history: deque = field(default_factory=lambda: deque(maxlen=PENDING_WINDOW))
    last_raw_box: list[float] | None = None
    last_status_reason: str = "pending_waiting_confirmation"
    visibility_approved: bool = False
    visibility_confirmed_frame: int | None = None
    visibility_confirm_reason: str = ""
    visibility_level: str = "NONE"
    identity_status: str = "unassigned"
    display_id: int | None = None
    first_shown_frame: int | None = None
    last_shown_frame: int | None = None
    shown_frames: int = 0
    max_conf: float = 0.0
    confirm_reason: str = ""
    reentry_source_raw_track_id: int | None = None
    reentry_gap_frames: int | None = None
    reentry_match_score: float | None = None
    reentry_center_distance: float | None = None
    reentry_area_ratio: float | None = None
    reentry_source_edge_side: str = ""
    reentry_new_edge_side: str = ""
    motion_model: str = ""
    motion_direct_distance: float | None = None
    motion_mahalanobis_sq: float | None = None

    @property
    def mean_conf(self) -> float:
        if not self.conf_history:
            return 0.0
        return sum(self.conf_history) / len(self.conf_history)

    @property
    def recent_strong_hits(self) -> int:
        return sum(1 for hit in self.strong_conf_history if hit)


@dataclass
class LostDisplayRecord:
    source_raw_track_id: int
    class_id: int
    display_id: int
    last_seen_frame: int
    last_shown_frame: int
    last_box: list[float]
    last_center: tuple[float, float]
    last_area: float
    last_edge_side: str


@dataclass
class ReentryCandidate:
    record: LostDisplayRecord
    gap_frames: int
    center_distance: float
    area_ratio: float
    new_edge_side: str
    score: float


def get_class_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


def get_display_label(class_id: int, display_id: int | None) -> str:
    if display_id is None:
        return ""
    prefix = DISPLAY_PREFIXES.get(class_id, f"C{class_id}")
    return f"D-{prefix}{display_id}"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run stable YOLO + BoT-SORT tracking with one-pass online "
            "postprocessing for image folders."
        ),
    )
    parser.add_argument(
        "--output-name",
        required=True,
        help=(
            "Unique run name. Used as the output image subdirectory and video "
            "file stem. Allowed characters: letters, numbers, underscore, dash, dot."
        ),
    )
    parser.add_argument(
        "--tracker-path",
        default=None,
        help="Optional tracker YAML path. Defaults to the retained stable BoT-SORT config.",
    )
    parser.add_argument(
        "--source-dir",
        required=True,
        help="Input image directory.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="YOLO .pt weights path.",
    )
    parser.add_argument(
        "--output-img-root",
        default=str(OUTPUT_IMG_ROOT),
        help="Root directory for annotated frame folders.",
    )
    parser.add_argument(
        "--output-video-root",
        default=str(OUTPUT_VIDEO_ROOT),
        help="Root directory for video, CSV, report, and audit outputs.",
    )
    parser.add_argument(
        "--inheritance-audit-csv",
        action="store_true",
        help=(
            "Write a frame-level inheritance candidate audit CSV. This is diagnostic "
            "only and does not change V19 decisions."
        ),
    )
    inheritance_group = parser.add_mutually_exclusive_group()
    inheritance_group.add_argument(
        "--enable-reentry-display-id",
        action="store_true",
        help="Enable causal motion-based display-ID inheritance for short raw-ID breaks.",
    )
    inheritance_group.add_argument(
        "--disable-id-inheritance",
        action="store_false",
        dest="enable_reentry_display_id",
        help="Disable display-ID inheritance while keeping online visibility filtering.",
    )
    parser.set_defaults(enable_reentry_display_id=True)
    return parser.parse_args()


def sort_key(p: Path):
    """Sort by embedded thermal timestamp, then fall back to filename order."""
    m = TIMESTAMP_PATTERN.match(p.name)
    if m:
        sec = int(m.group(1))
        nsec = int(m.group(2))
        return (0, sec, nsec, p.name.lower())
    return (1, p.name.lower())


def validate_output_name(output_name: str):
    if not OUTPUT_NAME_PATTERN.match(output_name):
        raise ValueError(
            "Invalid --output-name. Use only letters, numbers, underscore, dash, and dot; "
            "do not include path separators."
        )


def check_paths(
    model_path: Path,
    output_img_dir: Path,
    output_video_dir: Path,
    input_dir: Path,
):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    existing_paths = [output_img_dir, output_video_dir]
    for path in existing_paths:
        if path.exists():
            raise FileExistsError(f"Output path already exists: {path}. Choose a new --output-name.")

    OUTPUT_IMG_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_VIDEO_ROOT.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=False, exist_ok=False)
    output_video_dir.mkdir(parents=False, exist_ok=False)


def collect_images(input_dir: Path):
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]
    files.sort(key=sort_key)
    return files


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def edge_side(center: tuple[float, float], width: int, height: int) -> str:
    cx, cy = center
    distances = {
        "left": cx,
        "right": width - cx,
        "top": cy,
        "bottom": height - cy,
    }
    side, distance = min(distances.items(), key=lambda item: item[1])
    if distance <= REENTRY_EDGE_MARGIN_PX:
        return side
    return "none"


def edges_compatible(source_side: str, new_side: str) -> tuple[bool, float]:
    if source_side == "none" or new_side == "none":
        return False, 0.0
    if source_side == new_side:
        return True, 0.0
    opposite = {
        "left": "right",
        "right": "left",
        "top": "bottom",
        "bottom": "top",
    }
    if opposite[source_side] == new_side:
        return False, 0.0
    return True, 40.0


def center_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_lost_display_record(state: OnlineTrackState, width: int, height: int) -> LostDisplayRecord | None:
    if state.display_id is None or state.last_raw_box is None or state.first_shown_frame is None:
        return None
    last_center = box_center(state.last_raw_box)
    last_side = edge_side(last_center, width, height)
    if last_side == "none":
        return None
    return LostDisplayRecord(
        source_raw_track_id=state.track_id,
        class_id=state.class_id,
        display_id=state.display_id,
        last_seen_frame=state.last_seen_frame,
        last_shown_frame=state.last_shown_frame if state.last_shown_frame is not None else state.last_seen_frame,
        last_box=list(state.last_raw_box),
        last_center=last_center,
        last_area=box_area(state.last_raw_box),
        last_edge_side=last_side,
    )


def expire_lost_display_pool(
    lost_display_pool: dict[int, LostDisplayRecord],
    frame_idx: int,
) -> int:
    expired_ids = [
        raw_id for raw_id, record in lost_display_pool.items()
        if frame_idx - record.last_seen_frame > REENTRY_MAX_GAP_FRAMES
    ]
    for raw_id in expired_ids:
        del lost_display_pool[raw_id]
    return len(expired_ids)


def add_disappeared_tracks_to_lost_pool(
    states: dict[int, OnlineTrackState],
    seen_track_ids: set[int],
    lost_display_pool: dict[int, LostDisplayRecord],
    consumed_lost_source_ids: set[int],
    width: int,
    height: int,
):
    for track_id, state in states.items():
        if track_id in seen_track_ids:
            lost_display_pool.pop(track_id, None)
            continue
        if state.display_id is None or state.shown_frames <= 0:
            continue
        if track_id in consumed_lost_source_ids:
            continue
        if track_id in lost_display_pool:
            continue
        record = build_lost_display_record(state, width, height)
        if record is not None:
            lost_display_pool[track_id] = record


def find_reentry_candidate(
    state: OnlineTrackState,
    frame_idx: int,
    width: int,
    height: int,
    lost_display_pool: dict[int, LostDisplayRecord],
) -> tuple[ReentryCandidate | None, bool]:
    if state.last_raw_box is None:
        return None, False

    new_center = box_center(state.last_raw_box)
    new_side = edge_side(new_center, width, height)
    new_area = box_area(state.last_raw_box)
    candidates: list[ReentryCandidate] = []

    for record in lost_display_pool.values():
        if record.source_raw_track_id == state.track_id:
            continue
        if record.class_id != state.class_id:
            continue
        gap_frames = frame_idx - record.last_seen_frame
        if gap_frames <= 0 or gap_frames > REENTRY_MAX_GAP_FRAMES:
            continue

        compatible, edge_penalty = edges_compatible(record.last_edge_side, new_side)
        if not compatible:
            continue

        dist = center_distance(record.last_center, new_center)
        if dist > REENTRY_CENTER_DISTANCE_PX:
            continue

        if record.last_area <= 0.0:
            continue
        area_ratio = new_area / record.last_area
        if area_ratio < REENTRY_AREA_RATIO_MIN or area_ratio > REENTRY_AREA_RATIO_MAX:
            continue

        score = dist + 2.0 * gap_frames + 50.0 * abs(math.log(area_ratio)) + edge_penalty
        candidates.append(
            ReentryCandidate(
                record=record,
                gap_frames=gap_frames,
                center_distance=dist,
                area_ratio=area_ratio,
                new_edge_side=new_side,
                score=score,
            )
        )

    if not candidates:
        return None, False

    candidates.sort(key=lambda c: c.score)
    best = candidates[0]
    if len(candidates) > 1:
        second_best = candidates[1]
        if second_best.score - best.score < REENTRY_AMBIGUOUS_SCORE_MARGIN:
            return None, True
    return best, False


def assign_display_id(
    state: OnlineTrackState,
    frame_idx: int,
    width: int,
    height: int,
    next_display_ids: dict[int, int],
    enable_reentry_display_id: bool,
    lost_display_pool: dict[int, LostDisplayRecord],
    consumed_lost_source_ids: set[int],
    reentry_events: list[dict],
    stats: dict[str, int],
) -> str:
    if state.display_id is not None:
        return "shown"

    if enable_reentry_display_id:
        candidate, ambiguous = find_reentry_candidate(
            state,
            frame_idx,
            width,
            height,
            lost_display_pool,
        )
        if ambiguous:
            stats["reentry_rejected_ambiguous_count"] += 1
        if candidate is not None:
            record = candidate.record
            state.display_id = record.display_id
            state.identity_status = "confirmed_inherited"
            state.reentry_source_raw_track_id = record.source_raw_track_id
            state.reentry_gap_frames = candidate.gap_frames
            state.reentry_match_score = candidate.score
            state.reentry_center_distance = candidate.center_distance
            state.reentry_area_ratio = candidate.area_ratio
            state.reentry_source_edge_side = record.last_edge_side
            state.reentry_new_edge_side = candidate.new_edge_side
            stats["reentry_inheritance_count"] += 1
            reentry_events.append(
                {
                    "display_id": record.display_id,
                    "display_label": get_display_label(state.class_id, record.display_id),
                    "class_id": state.class_id,
                    "class_name": get_class_name(state.class_id),
                    "source_raw_track_id": record.source_raw_track_id,
                    "new_raw_track_id": state.track_id,
                    "gap_frames": candidate.gap_frames,
                    "center_distance": candidate.center_distance,
                    "area_ratio": candidate.area_ratio,
                    "edge_side_source": record.last_edge_side,
                    "edge_side_new": candidate.new_edge_side,
                    "score": candidate.score,
                }
            )
            del lost_display_pool[record.source_raw_track_id]
            consumed_lost_source_ids.add(record.source_raw_track_id)
            return "inherited_reentry_display_id"

    next_display_id = next_display_ids.get(state.class_id, 1)
    state.display_id = next_display_id
    state.identity_status = "confirmed_new"
    next_display_ids[state.class_id] = next_display_id + 1
    return "shown"


def _motion_transition(dt: int) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt],
         [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=float,
    )


def _motion_process_noise(dt: int) -> np.ndarray:
    dt2, dt3, dt4 = dt * dt, dt * dt * dt, dt * dt * dt * dt
    return (MOTION_ACCEL_STD_PX ** 2) * np.array(
        [[dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
         [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
         [dt3 / 2.0, 0.0, dt2, 0.0],
         [0.0, dt3 / 2.0, 0.0, dt2]],
        dtype=float,
    )


def _build_motion_lost_display_record(
    state: OnlineTrackState, width: int, height: int
) -> LostDisplayRecord | None:
    if state.display_id is None or state.last_raw_box is None or state.first_shown_frame is None:
        return None
    center = box_center(state.last_raw_box)
    previous_box = getattr(state, "previous_raw_box", None)
    previous_frame = getattr(state, "previous_seen_frame", state.last_seen_frame)
    vx = vy = 0.0
    if previous_box is not None and state.last_seen_frame > previous_frame:
        previous_center = box_center(previous_box)
        dt = state.last_seen_frame - previous_frame
        vx = (center[0] - previous_center[0]) / dt
        vy = (center[1] - previous_center[1]) / dt
    record = LostDisplayRecord(
        source_raw_track_id=state.track_id,
        class_id=state.class_id,
        display_id=state.display_id,
        last_seen_frame=state.last_seen_frame,
        last_shown_frame=state.last_shown_frame if state.last_shown_frame is not None else state.last_seen_frame,
        last_box=list(state.last_raw_box),
        last_center=center,
        last_area=box_area(state.last_raw_box),
        last_edge_side=edge_side(center, width, height),
    )
    record.kf_x = np.array([center[0], center[1], vx, vy], dtype=float)
    record.kf_p = np.diag(
        [MOTION_INITIAL_POSITION_STD_PX ** 2] * 2
        + [MOTION_INITIAL_VELOCITY_STD_PX ** 2] * 2
    )
    return record


# Runtime replacement for the old edge-only pool builder. The stable V9 copy
# retains the original implementation for reproducibility.
build_lost_display_record = _build_motion_lost_display_record


@dataclass
class MotionMatchCandidate:
    state: OnlineTrackState
    record: LostDisplayRecord
    gap_frames: int
    center_distance: float
    direct_distance: float
    mahalanobis_sq: float
    area_ratio: float
    new_edge_side: str
    score: float
    motion_model: str


def build_motion_candidate(
    state: OnlineTrackState,
    record: LostDisplayRecord,
    frame_idx: int,
    width: int,
    height: int,
) -> MotionMatchCandidate | None:
    if state.last_raw_box is None or getattr(state, "last_conf", 0.0) < MOTION_MIN_CONF:
        return None
    if record.class_id != state.class_id:
        return None
    gap = frame_idx - record.last_seen_frame
    if gap < MOTION_MIN_GAP_FRAMES or gap > REENTRY_MAX_GAP_FRAMES:
        return None

    f = _motion_transition(gap)
    x_pred = f @ record.kf_x
    p_pred = f @ record.kf_p @ f.T + _motion_process_noise(gap)
    h_matrix = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    r_matrix = np.eye(2) * (MOTION_MEASUREMENT_STD_PX ** 2)
    measurement = np.asarray(box_center(state.last_raw_box), dtype=float)
    innovation = measurement - h_matrix @ x_pred
    innovation_cov = h_matrix @ p_pred @ h_matrix.T + r_matrix
    cv_mahalanobis_sq = float(innovation.T @ np.linalg.solve(innovation_cov, innovation))
    cv_distance = float(np.linalg.norm(innovation))
    direct_innovation = measurement - np.asarray(record.last_center, dtype=float)
    direct_distance = float(np.linalg.norm(direct_innovation))
    if direct_distance > MOTION_ABSOLUTE_DIRECT_DISTANCE_PX:
        return None
    stationary_variance = (
        MOTION_INITIAL_POSITION_STD_PX ** 2
        + (MOTION_ACCEL_STD_PX ** 2) * (gap ** 4) / 4.0
        + MOTION_MEASUREMENT_STD_PX ** 2
    )
    stationary_mahalanobis_sq = direct_distance ** 2 / stationary_variance
    if cv_distance <= direct_distance:
        center_distance = cv_distance
        mahalanobis_sq = cv_mahalanobis_sq
        motion_model = "constant_velocity"
    else:
        center_distance = direct_distance
        mahalanobis_sq = stationary_mahalanobis_sq
        motion_model = "near_stationary"
    distance_limit = MOTION_DISTANCE_BASE_PX + MOTION_DISTANCE_PER_GAP_PX * gap
    if (
        gap <= MOTION_FAST_REJOIN_MAX_GAP_FRAMES
        and getattr(state, "last_conf", 0.0) >= MOTION_FAST_REJOIN_MIN_CONF
    ):
        distance_limit = max(distance_limit, MOTION_FAST_REJOIN_DISTANCE_PX)
    if mahalanobis_sq > MOTION_MAHALANOBIS_THRESHOLD or center_distance > distance_limit:
        return None

    new_area = box_area(state.last_raw_box)
    if record.last_area <= 0.0 or new_area <= 0.0:
        return None
    area_ratio = new_area / record.last_area
    new_side = edge_side(tuple(measurement), width, height)
    edge_clipped = record.last_edge_side != "none" or new_side != "none"
    area_min = MOTION_EDGE_AREA_RATIO_MIN if edge_clipped else REENTRY_AREA_RATIO_MIN
    area_max = MOTION_EDGE_AREA_RATIO_MAX if edge_clipped else REENTRY_AREA_RATIO_MAX
    if area_ratio < area_min or area_ratio > area_max:
        return None

    score = (
        mahalanobis_sq / MOTION_MAHALANOBIS_THRESHOLD
        + 0.35 * center_distance / distance_limit
        + 0.30 * abs(math.log(area_ratio))
        + 0.10 * gap / REENTRY_MAX_GAP_FRAMES
    )
    return MotionMatchCandidate(
        state=state,
        record=record,
        gap_frames=gap,
        center_distance=center_distance,
        direct_distance=direct_distance,
        mahalanobis_sq=mahalanobis_sq,
        area_ratio=area_ratio,
        new_edge_side=new_side,
        score=score,
        motion_model=motion_model,
    )


def _accept_motion_inheritance(
    candidate: MotionMatchCandidate,
    frame_idx: int,
    lost_display_pool: dict[int, LostDisplayRecord],
    consumed_lost_source_ids: set[int],
    reentry_events: list[dict],
    stats: dict[str, int],
):
    state, record = candidate.state, candidate.record
    state.provisional_box_visible = False
    state.identity_status = "confirmed_inherited"
    state.display_id = record.display_id
    state.reentry_source_raw_track_id = record.source_raw_track_id
    state.reentry_gap_frames = candidate.gap_frames
    state.reentry_match_score = candidate.score
    state.reentry_center_distance = candidate.center_distance
    state.reentry_area_ratio = candidate.area_ratio
    state.reentry_source_edge_side = record.last_edge_side
    state.reentry_new_edge_side = candidate.new_edge_side
    state.motion_model = candidate.motion_model
    state.motion_direct_distance = candidate.direct_distance
    state.motion_mahalanobis_sq = candidate.mahalanobis_sq
    if state.status != "confirmed":
        state.status = "confirmed"
        state.confirmed_frame = frame_idx
        state.confirm_reason = "inherited_motion_display_id"
    state.last_status_reason = "inherited_motion_display_id"
    stats["reentry_inheritance_count"] += 1
    reentry_events.append(
        {
            "display_id": record.display_id,
            "display_label": get_display_label(state.class_id, record.display_id),
            "class_id": state.class_id,
            "class_name": get_class_name(state.class_id),
            "source_raw_track_id": record.source_raw_track_id,
            "new_raw_track_id": state.track_id,
            "gap_frames": candidate.gap_frames,
            "center_distance": candidate.center_distance,
            "direct_distance": candidate.direct_distance,
            "mahalanobis_sq": candidate.mahalanobis_sq,
            "area_ratio": candidate.area_ratio,
            "edge_side_source": record.last_edge_side,
            "edge_side_new": candidate.new_edge_side,
            "score": candidate.score,
            "motion_model": candidate.motion_model,
        }
    )
    lost_display_pool.pop(record.source_raw_track_id, None)
    consumed_lost_source_ids.add(record.source_raw_track_id)


# Deprecated experiment implementation. Retained only for historical comparison;
# the runtime entry point is apply_motion_inheritance_for_frame below.
def _legacy_apply_motion_inheritance_for_frame_v12(
    frame_items: list[tuple],
    frame_idx: int,
    width: int,
    height: int,
    lost_display_pool: dict[int, LostDisplayRecord],
    consumed_lost_source_ids: set[int],
    reentry_events: list[dict],
    stats: dict[str, int],
    aggregate_tracklets: dict[tuple[int, int], "AggregateInheritanceTracklet"] | None = None,
):
    if aggregate_tracklets is None:
        aggregate_tracklets = AGGREGATE_INHERITANCE_TRACKLETS
    current_raw_ids = {item[0].track_id for item in frame_items}
    for raw_id in current_raw_ids:
        lost_display_pool.pop(raw_id, None)
    active_display_keys = {
        (item[0].class_id, item[0].display_id)
        for item in frame_items
        if item[0].display_id is not None
    }
    available_records = [
        record for record in lost_display_pool.values()
        if (record.class_id, record.display_id) not in active_display_keys
    ]
    candidate_states = [item[0] for item in frame_items if item[0].display_id is None]
    for class_id in CLASSES:
        states_for_class = [state for state in candidate_states if state.class_id == class_id]
        records_for_class = [record for record in available_records if record.class_id == class_id]
        if not states_for_class or not records_for_class:
            continue
        matrix = np.full((len(states_for_class), len(records_for_class)), 1e6, dtype=float)
        candidates: dict[tuple[int, int], MotionMatchCandidate] = {}
        for row, state in enumerate(states_for_class):
            for col, record in enumerate(records_for_class):
                candidate = build_motion_candidate(state, record, frame_idx, width, height)
                if candidate is not None:
                    matrix[row, col] = candidate.score
                    candidates[(row, col)] = candidate
        if not candidates:
            continue
        rows, cols = linear_sum_assignment(matrix)
        for row, col in zip(rows, cols):
            candidate = candidates.get((int(row), int(col)))
            if candidate is None:
                continue
            row_alternatives = [matrix[row, j] for j in range(matrix.shape[1]) if j != col and matrix[row, j] < 1e6]
            col_alternatives = [matrix[i, col] for i in range(matrix.shape[0]) if i != row and matrix[i, col] < 1e6]
            closest_alternative = min(row_alternatives + col_alternatives, default=float("inf"))
            if closest_alternative - candidate.score < REENTRY_AMBIGUOUS_SCORE_MARGIN:
                stats["reentry_rejected_ambiguous_count"] += 1
                continue
            _accept_motion_inheritance(
                candidate, frame_idx, lost_display_pool, consumed_lost_source_ids,
                reentry_events, stats,
            )


def _clear_inherited_display_identity(state: OnlineTrackState):
    state.display_id = None
    state.reentry_source_raw_track_id = None
    state.reentry_gap_frames = None
    state.reentry_match_score = None
    state.reentry_center_distance = None
    state.reentry_area_ratio = None
    state.reentry_source_edge_side = ""
    state.reentry_new_edge_side = ""
    state.motion_model = ""
    state.motion_direct_distance = None
    state.motion_mahalanobis_sq = None
    state.status = "pending"
    state.confirmed_frame = None
    state.confirm_reason = ""
    state.last_status_reason = "pending_after_identity_conflict"


def resolve_display_identity_conflicts(
    frame_items: list[tuple], frame_idx: int, stats: dict[str, int]
):
    groups: dict[tuple[int, int], list[OnlineTrackState]] = {}
    for state, *_ in frame_items:
        if state.display_id is not None:
            groups.setdefault((state.class_id, state.display_id), []).append(state)
    for states_with_same_display in groups.values():
        if len(states_with_same_display) <= 1:
            continue
        raw_ids = {state.track_id for state in states_with_same_display}
        source_raw_ids = {
            state.reentry_source_raw_track_id
            for state in states_with_same_display
            if state.reentry_source_raw_track_id is not None
        }
        winner = min(
            states_with_same_display,
            key=lambda state: (
                0 if state.track_id in source_raw_ids else 1,
                state.first_seen_frame,
                state.track_id,
            ),
        )
        for state in states_with_same_display:
            if state is winner:
                continue
            _clear_inherited_display_identity(state)
            confirm_pending_track(state, frame_idx)
            stats["motion_conflict_rollback_count"] += 1


@dataclass
class PendingInheritanceObservation:
    source_raw_track_id: int
    candidate_raw_track_id: int
    first_selected_frame: int
    last_selected_frame: int
    selected_count: int
    required_window: int


PENDING_INHERITANCE_OBSERVATIONS: dict[tuple[int, int], PendingInheritanceObservation] = {}
SOURCE_ASSIGNMENT_STABILITY: dict[int, tuple[int, int, int]] = {}


def _hold_inheritance_candidate(state: OnlineTrackState):
    if state.display_id is not None:
        return
    if state.status == "confirmed":
        state.visibility_approved = True
        if state.visibility_confirmed_frame is None:
            state.visibility_confirmed_frame = state.confirmed_frame
        if not state.visibility_confirm_reason:
            state.visibility_confirm_reason = state.confirm_reason
    state.provisional_box_visible = state.visibility_approved
    state.identity_status = "pending_inheritance"
    state.status = "pending"
    state.last_status_reason = "pending_inheritance_candidate"


def _release_inheritance_candidate(state: OnlineTrackState, frame_idx: int):
    if state.display_id is not None:
        return
    state.provisional_box_visible = False
    state.identity_status = "unassigned"
    state.status = "pending"
    state.last_status_reason = "pending_waiting_confirmation"
    if state.visibility_approved:
        state.status = "confirmed"
        state.confirmed_frame = state.visibility_confirmed_frame
        state.confirm_reason = state.visibility_confirm_reason
        state.last_status_reason = state.visibility_confirm_reason
    else:
        state.confirmed_frame = None
        state.confirm_reason = ""
        confirm_pending_track(state, frame_idx)


# Final simplified policy: context size controls observation latency, while the
# valid motion graph controls assignment. This later definition intentionally
# replaces the immediate-commit implementation above.
# Deprecated experiment implementation. Retained only for historical comparison;
# the runtime entry point is apply_motion_inheritance_for_frame below.
def _legacy_apply_motion_inheritance_for_frame_v13(
    frame_items: list[tuple],
    frame_idx: int,
    width: int,
    height: int,
    lost_display_pool: dict[int, LostDisplayRecord],
    consumed_lost_source_ids: set[int],
    reentry_events: list[dict],
    stats: dict[str, int],
):
    stats.setdefault("inheritance_candidate_started_count", 0)
    stats.setdefault("inheritance_candidate_committed_count", 0)
    stats.setdefault("inheritance_candidate_source_return_count", 0)
    stats.setdefault("inheritance_candidate_rejected_count", 0)
    stats.setdefault("inheritance_context_upgrade_count", 0)

    current_by_raw = {item[0].track_id: item[0] for item in frame_items}
    current_raw_ids = set(current_by_raw)

    returned_sources = {
        source_raw_id
        for source_raw_id, _candidate_raw_id in PENDING_INHERITANCE_OBSERVATIONS
        if source_raw_id in current_raw_ids
    }
    for source_raw_id in returned_sources:
        keys = [key for key in PENDING_INHERITANCE_OBSERVATIONS if key[0] == source_raw_id]
        for key in keys:
            candidate_state = current_by_raw.get(key[1])
            if candidate_state is not None:
                _release_inheritance_candidate(candidate_state, frame_idx)
            del PENDING_INHERITANCE_OBSERVATIONS[key]
            stats["inheritance_candidate_source_return_count"] += 1
        SOURCE_ASSIGNMENT_STABILITY.pop(source_raw_id, None)

    for raw_id in current_raw_ids:
        lost_display_pool.pop(raw_id, None)

    available_source_ids = set(lost_display_pool)
    stale_keys = [
        key for key in PENDING_INHERITANCE_OBSERVATIONS
        if key[0] not in available_source_ids
    ]
    for key in stale_keys:
        candidate_state = current_by_raw.get(key[1])
        if candidate_state is not None:
            _release_inheritance_candidate(candidate_state, frame_idx)
        del PENDING_INHERITANCE_OBSERVATIONS[key]
        SOURCE_ASSIGNMENT_STABILITY.pop(key[0], None)
        stats["inheritance_candidate_rejected_count"] += 1

    active_display_keys = {
        (item[0].class_id, item[0].display_id)
        for item in frame_items
        if item[0].display_id is not None
    }
    available_records = [
        record for record in lost_display_pool.values()
        if (record.class_id, record.display_id) not in active_display_keys
    ]
    candidate_states = [item[0] for item in frame_items if item[0].display_id is None]
    held_raw_ids: set[int] = set()

    for class_id in CLASSES:
        states_for_class = [state for state in candidate_states if state.class_id == class_id]
        records_for_class = [record for record in available_records if record.class_id == class_id]
        if not states_for_class or not records_for_class:
            continue

        context_lost_count = len(records_for_class)
        context_candidate_count = len(states_for_class)
        matrix = np.full((len(states_for_class), len(records_for_class)), 1e6, dtype=float)
        candidates: dict[tuple[int, int], MotionMatchCandidate] = {}
        for row, state in enumerate(states_for_class):
            for col, record in enumerate(records_for_class):
                candidate = build_motion_candidate(state, record, frame_idx, width, height)
                if candidate is None:
                    continue
                matrix[row, col] = candidate.score
                candidates[(row, col)] = candidate
                held_raw_ids.add(state.track_id)
                _hold_inheritance_candidate(state)

        if not candidates:
            continue

        rows, cols = linear_sum_assignment(matrix)
        for row, col in zip(rows, cols):
            candidate = candidates.get((int(row), int(col)))
            if candidate is None:
                continue

            row_alternatives = [
                matrix[row, j] for j in range(matrix.shape[1])
                if j != col and matrix[row, j] < 1e6
            ]
            col_alternatives = [
                matrix[i, col] for i in range(matrix.shape[0])
                if i != row and matrix[i, col] < 1e6
            ]
            closest_alternative = min(row_alternatives + col_alternatives, default=float("inf"))
            if closest_alternative - candidate.score < REENTRY_AMBIGUOUS_SCORE_MARGIN:
                stats["reentry_rejected_ambiguous_count"] += 1
                continue

            is_edge_reentry = (
                candidate.record.last_edge_side != "none"
                or candidate.new_edge_side != "none"
            )
            context_is_ambiguous = context_lost_count > 1 or context_candidate_count > 1
            required_window = 8 if context_is_ambiguous or is_edge_reentry else 3
            key = (candidate.record.source_raw_track_id, candidate.state.track_id)
            observation = PENDING_INHERITANCE_OBSERVATIONS.get(key)
            if observation is None:
                observation = PendingInheritanceObservation(
                    source_raw_track_id=key[0],
                    candidate_raw_track_id=key[1],
                    first_selected_frame=frame_idx,
                    last_selected_frame=frame_idx,
                    selected_count=1,
                    required_window=required_window,
                )
                PENDING_INHERITANCE_OBSERVATIONS[key] = observation
                stats["inheritance_candidate_started_count"] += 1
            else:
                observation.last_selected_frame = frame_idx
                observation.selected_count += 1
                if required_window > observation.required_window:
                    observation.required_window = required_window
                    stats["inheritance_context_upgrade_count"] += 1

            previous_stability = SOURCE_ASSIGNMENT_STABILITY.get(key[0])
            if (
                previous_stability is not None
                and previous_stability[0] == key[1]
                and previous_stability[2] == frame_idx - 1
            ):
                stable_frames = previous_stability[1] + 1
            else:
                stable_frames = 1
            SOURCE_ASSIGNMENT_STABILITY[key[0]] = (key[1], stable_frames, frame_idx)

            elapsed_frames = frame_idx - observation.first_selected_frame + 1
            minimum_stable_frames = 3 if observation.required_window == 8 else 2
            minimum_selected_count = 3 if observation.required_window == 8 else 2
            if (
                elapsed_frames < observation.required_window
                or stable_frames < minimum_stable_frames
                or observation.selected_count < minimum_selected_count
            ):
                continue

            _accept_motion_inheritance(
                candidate,
                frame_idx,
                lost_display_pool,
                consumed_lost_source_ids,
                reentry_events,
                stats,
            )
            stats["inheritance_candidate_committed_count"] += 1

            related_keys = [
                pending_key for pending_key in PENDING_INHERITANCE_OBSERVATIONS
                if pending_key[0] == key[0] or pending_key[1] == key[1]
            ]
            for pending_key in related_keys:
                del PENDING_INHERITANCE_OBSERVATIONS[pending_key]
            SOURCE_ASSIGNMENT_STABILITY.pop(key[0], None)

            for other_state in states_for_class:
                if other_state.track_id != candidate.state.track_id and other_state.track_id in held_raw_ids:
                    _release_inheritance_candidate(other_state, frame_idx)

    for state in candidate_states:
        if (
            state.track_id not in held_raw_ids
            and state.last_status_reason == "pending_inheritance_candidate"
        ):
            _release_inheritance_candidate(state, frame_idx)


@dataclass
class AggregateInheritanceTracklet:
    source_record: LostDisplayRecord
    candidate_raw_track_id: int
    first_frame: int
    last_observed_frame: int
    latest_state: OnlineTrackState
    latest_candidate: MotionMatchCandidate
    scores: list[float] = field(default_factory=list)
    centers: list[tuple[float, float]] = field(default_factory=list)
    first_edge_side: str = "none"
    missed_frames: int = 0
    required_window_frames: int = MOTION_SLOW_BASE_WINDOW_FRAMES

    @property
    def mean_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else float("inf")

AGGREGATE_INHERITANCE_TRACKLETS: dict[tuple[int, int], AggregateInheritanceTracklet] = {}


def _coarse_tracklet_candidate(
    state: OnlineTrackState,
    record: LostDisplayRecord,
    frame_idx: int,
    width: int,
    height: int,
) -> MotionMatchCandidate | None:
    if state.last_raw_box is None or record.class_id != state.class_id:
        return None
    confidence = getattr(state, "last_conf", 0.0)
    if confidence < MOTION_MIN_CONF:
        return None
    gap = frame_idx - record.last_seen_frame
    if gap < MOTION_MIN_GAP_FRAMES or gap > MOTION_INITIAL_CANDIDATE_MAX_GAP_FRAMES:
        return None
    center = box_center(state.last_raw_box)
    direct_distance = math.dist(center, record.last_center)
    if direct_distance > MOTION_ABSOLUTE_DIRECT_DISTANCE_PX:
        return None
    new_area = box_area(state.last_raw_box)
    if record.last_area <= 0.0 or new_area <= 0.0:
        return None
    area_ratio = new_area / record.last_area
    new_side = edge_side(center, width, height)
    edge_context = record.last_edge_side != "none" or new_side != "none"
    area_min = MOTION_EDGE_AREA_RATIO_MIN if edge_context else REENTRY_AREA_RATIO_MIN
    area_max = MOTION_EDGE_AREA_RATIO_MAX if edge_context else REENTRY_AREA_RATIO_MAX
    if area_ratio < area_min or area_ratio > area_max:
        return None
    score = (
        direct_distance / MOTION_ABSOLUTE_DIRECT_DISTANCE_PX
        + 0.30 * abs(math.log(area_ratio))
        + 0.10 * gap / MOTION_INITIAL_CANDIDATE_MAX_GAP_FRAMES
        + 0.20 * (1.0 - confidence)
    )
    return MotionMatchCandidate(
        state=state,
        record=record,
        gap_frames=gap,
        center_distance=direct_distance,
        direct_distance=direct_distance,
        mahalanobis_sq=0.0,
        area_ratio=area_ratio,
        new_edge_side=new_side,
        score=score,
        motion_model="aggregate_tracklet",
    )


def _coarse_candidate_rejection_reason(
    state: OnlineTrackState,
    record: LostDisplayRecord,
    frame_idx: int,
    width: int,
    height: int,
) -> str:
    """Explain the first failed V19 coarse gate without changing the gate itself."""
    if state.last_raw_box is None:
        return "candidate_missing_box"
    if record.class_id != state.class_id:
        return "class_mismatch"
    confidence = getattr(state, "last_conf", 0.0)
    if confidence < MOTION_MIN_CONF:
        return "confidence_below_min"
    gap = frame_idx - record.last_seen_frame
    if gap < MOTION_MIN_GAP_FRAMES:
        return "gap_below_min"
    if gap > MOTION_INITIAL_CANDIDATE_MAX_GAP_FRAMES:
        return "gap_above_initial_candidate_max"
    center = box_center(state.last_raw_box)
    if math.dist(center, record.last_center) > MOTION_ABSOLUTE_DIRECT_DISTANCE_PX:
        return "direct_distance_above_max"
    new_area = box_area(state.last_raw_box)
    if record.last_area <= 0.0 or new_area <= 0.0:
        return "invalid_area"
    area_ratio = new_area / record.last_area
    new_side = edge_side(center, width, height)
    edge_context = record.last_edge_side != "none" or new_side != "none"
    area_min = MOTION_EDGE_AREA_RATIO_MIN if edge_context else REENTRY_AREA_RATIO_MIN
    area_max = MOTION_EDGE_AREA_RATIO_MAX if edge_context else REENTRY_AREA_RATIO_MAX
    if area_ratio < area_min:
        return "area_ratio_below_min"
    if area_ratio > area_max:
        return "area_ratio_above_max"
    return "coarse_eligible"


def _append_inheritance_audit(
    audit_events: list[dict] | None,
    frame_idx: int,
    event_type: str,
    reason: str,
    state: OnlineTrackState | None = None,
    record: LostDisplayRecord | None = None,
    candidate: MotionMatchCandidate | None = None,
    tracklet: AggregateInheritanceTracklet | None = None,
    coarse_eligible: bool | None = None,
    strict_eligible: bool | None = None,
):
    if audit_events is None:
        return
    if tracklet is not None:
        state = tracklet.latest_state
        record = tracklet.source_record
        candidate = tracklet.latest_candidate
    audit_events.append(
        {
            "frame_idx": frame_idx,
            "event_type": event_type,
            "reason": reason,
            "class_id": "" if state is None else state.class_id,
            "display_id": "" if record is None else record.display_id,
            "source_raw_track_id": "" if record is None else record.source_raw_track_id,
            "candidate_raw_track_id": "" if state is None else state.track_id,
            "source_last_seen_frame": "" if record is None else record.last_seen_frame,
            "gap_frames": "" if candidate is None else candidate.gap_frames,
            "candidate_conf": "" if state is None else getattr(state, "last_conf", 0.0),
            "candidate_status": "" if state is None else state.status,
            "visibility_approved": "" if state is None else state.visibility_approved,
            "visibility_level": "" if state is None else state.visibility_level,
            "coarse_eligible": "" if coarse_eligible is None else coarse_eligible,
            "strict_eligible": "" if strict_eligible is None else strict_eligible,
            "direct_distance": "" if candidate is None else candidate.direct_distance,
            "area_ratio": "" if candidate is None else candidate.area_ratio,
            "score": "" if candidate is None else candidate.score,
            "observation_count": "" if tracklet is None else len(tracklet.scores),
            "mean_score": "" if tracklet is None else tracklet.mean_score,
            "missed_frames": "" if tracklet is None else tracklet.missed_frames,
            "required_window_frames": "" if tracklet is None else tracklet.required_window_frames,
        }
    )


def _edge_inward_depth(center: tuple[float, float], side: str, width: int, height: int) -> float:
    if side == "left":
        return center[0]
    if side == "right":
        return width - center[0]
    if side == "top":
        return center[1]
    if side == "bottom":
        return height - center[1]
    return 0.0


def _aggregate_tracklet_is_committable(
    tracklet: AggregateInheritanceTracklet, frame_idx: int, width: int, height: int
) -> bool:
    if frame_idx - tracklet.first_frame + 1 < tracklet.required_window_frames:
        return False
    if len(tracklet.scores) < MOTION_SLOW_MIN_OBSERVATIONS:
        return False
    if not tracklet.latest_state.visibility_approved:
        return False
    source_side = tracklet.source_record.last_edge_side
    if source_side != "none":
        if tracklet.first_edge_side != source_side or len(tracklet.centers) < 3:
            return False
        first_depth = _edge_inward_depth(tracklet.centers[0], tracklet.first_edge_side, width, height)
        last_depth = _edge_inward_depth(tracklet.centers[-1], tracklet.first_edge_side, width, height)
        if last_depth - first_depth < MOTION_EDGE_MIN_INWARD_PX:
            return False
    return True


def _sample_existing_aggregate_tracklet(
    tracklet: AggregateInheritanceTracklet,
    state: OnlineTrackState,
    frame_idx: int,
    width: int,
    height: int,
) -> bool:
    if state.class_id != tracklet.source_record.class_id or state.last_raw_box is None:
        return False
    confidence = getattr(state, "last_conf", 0.0)
    if confidence < MOTION_MIN_CONF:
        return False
    center = box_center(state.last_raw_box)
    current_area = box_area(state.last_raw_box)
    if current_area <= 0.0 or tracklet.source_record.last_area <= 0.0:
        return False
    area_ratio = current_area / tracklet.source_record.last_area
    direct_distance = math.dist(center, tracklet.source_record.last_center)
    gap = frame_idx - tracklet.source_record.last_seen_frame
    score = (
        min(direct_distance / MOTION_ABSOLUTE_DIRECT_DISTANCE_PX, 1.0)
        + 0.30 * abs(math.log(area_ratio))
        + 0.10 * min(gap, REENTRY_MAX_GAP_FRAMES) / REENTRY_MAX_GAP_FRAMES
        + 0.20 * (1.0 - confidence)
    )
    candidate = MotionMatchCandidate(
        state=state,
        record=tracklet.source_record,
        gap_frames=gap,
        center_distance=direct_distance,
        direct_distance=direct_distance,
        mahalanobis_sq=0.0,
        area_ratio=area_ratio,
        new_edge_side=edge_side(center, width, height),
        score=score,
        motion_model="aggregate_tracklet_follow",
    )
    tracklet.last_observed_frame = frame_idx
    tracklet.latest_state = state
    tracklet.latest_candidate = candidate
    tracklet.scores.append(score)
    tracklet.centers.append(center)
    tracklet.missed_frames = 0
    return True


# V14 policy: fast direct commit for a strictly safe 1x1 relation; all other
# relations are retained as aggregate tracklets and judged once per 8-frame
# context window. This later definition replaces the V13 instantaneous-edge
# observation implementation above.
def apply_motion_inheritance_for_frame(
    frame_items: list[tuple],
    frame_idx: int,
    width: int,
    height: int,
    lost_display_pool: dict[int, LostDisplayRecord],
    consumed_lost_source_ids: set[int],
    reentry_events: list[dict],
    stats: dict[str, int],
    aggregate_tracklets: dict[tuple[int, int], AggregateInheritanceTracklet] | None = None,
    audit_events: list[dict] | None = None,
):
    if aggregate_tracklets is None:
        aggregate_tracklets = AGGREGATE_INHERITANCE_TRACKLETS
    stats.setdefault("inheritance_fast_commit_count", 0)
    stats.setdefault("inheritance_slow_started_count", 0)
    stats.setdefault("inheritance_slow_commit_count", 0)
    stats.setdefault("inheritance_slow_source_return_count", 0)
    stats.setdefault("inheritance_slow_rejected_count", 0)

    current_by_raw = {item[0].track_id: item[0] for item in frame_items}
    current_raw_ids = set(current_by_raw)

    returned_sources = {
        source_raw_id
        for source_raw_id, _candidate_raw_id in aggregate_tracklets
        if source_raw_id in current_raw_ids
    }
    for source_raw_id in returned_sources:
        keys = [key for key in aggregate_tracklets if key[0] == source_raw_id]
        for key in keys:
            tracklet = aggregate_tracklets[key]
            _append_inheritance_audit(
                audit_events, frame_idx, "tracklet_rejected", "source_returned",
                tracklet=tracklet,
            )
            candidate_state = current_by_raw.get(key[1])
            if candidate_state is not None:
                _release_inheritance_candidate(candidate_state, frame_idx)
            del aggregate_tracklets[key]
            stats["inheritance_slow_source_return_count"] += 1

    for raw_id in current_raw_ids:
        lost_display_pool.pop(raw_id, None)

    active_display_keys = {
        (item[0].class_id, item[0].display_id)
        for item in frame_items
        if item[0].display_id is not None
    }
    available_records = [
        record for record in lost_display_pool.values()
        if (record.class_id, record.display_id) not in active_display_keys
    ]
    candidate_states = [item[0] for item in frame_items if item[0].display_id is None]
    held_raw_ids: set[int] = set()

    for class_id in CLASSES:
        states_for_class = [state for state in candidate_states if state.class_id == class_id]
        records_for_class = [record for record in available_records if record.class_id == class_id]
        if not states_for_class or not records_for_class:
            continue
        context_is_one_to_one = len(states_for_class) == 1 and len(records_for_class) == 1

        coarse_candidates: dict[tuple[int, int], MotionMatchCandidate] = {}
        strict_candidates: dict[tuple[int, int], MotionMatchCandidate] = {}
        for row, state in enumerate(states_for_class):
            for col, record in enumerate(records_for_class):
                coarse = _coarse_tracklet_candidate(state, record, frame_idx, width, height)
                if coarse is None:
                    _append_inheritance_audit(
                        audit_events, frame_idx, "pair_evaluated",
                        _coarse_candidate_rejection_reason(state, record, frame_idx, width, height),
                        state=state, record=record, coarse_eligible=False, strict_eligible=False,
                    )
                    continue
                coarse_candidates[(row, col)] = coarse
                strict = build_motion_candidate(state, record, frame_idx, width, height)
                if strict is not None:
                    strict_candidates[(row, col)] = strict
                _append_inheritance_audit(
                    audit_events, frame_idx, "pair_evaluated",
                    "strict_eligible" if strict is not None else "strict_gate_rejected",
                    state=state, record=record, candidate=strict if strict is not None else coarse,
                    coarse_eligible=True, strict_eligible=strict is not None,
                )

        if context_is_one_to_one and (0, 0) in strict_candidates:
            candidate = strict_candidates[(0, 0)]
            safe_non_edge_fast_path = (
                candidate.gap_frames <= MOTION_FAST_DIRECT_MAX_GAP_FRAMES
                and candidate.direct_distance <= MOTION_FAST_DIRECT_DISTANCE_PX
                and candidate.mahalanobis_sq <= MOTION_FAST_DIRECT_MAHALANOBIS_THRESHOLD
                and candidate.record.last_edge_side == "none"
                and candidate.new_edge_side == "none"
                and 0.60 <= candidate.area_ratio <= 1.67
            )
            source_side = candidate.record.last_edge_side
            edge_compatible = (
                source_side != "none" and source_side == candidate.new_edge_side
            )
            edge_inward = False
            if (
                source_side != "none"
                and candidate.new_edge_side != "none"
                and candidate.state.previous_raw_box is not None
            ):
                previous_center = box_center(candidate.state.previous_raw_box)
                current_center = box_center(candidate.state.last_raw_box)
                edge_inward = (
                    _edge_inward_depth(current_center, candidate.new_edge_side, width, height)
                    - _edge_inward_depth(previous_center, candidate.new_edge_side, width, height)
                    >= MOTION_EDGE_MIN_INWARD_PX
                )
            safe_edge_fast_path = (
                candidate.gap_frames <= MOTION_FAST_DIRECT_MAX_GAP_FRAMES
                and candidate.direct_distance <= MOTION_EDGE_FAST_DIRECT_DISTANCE_PX
                and candidate.mahalanobis_sq <= MOTION_EDGE_FAST_DIRECT_MAHALANOBIS_THRESHOLD
                and candidate.state.visibility_level in {"FAST", "MID", "MATURE"}
                and edge_compatible
                and edge_inward
                and MOTION_EDGE_AREA_RATIO_MIN <= candidate.area_ratio <= MOTION_EDGE_AREA_RATIO_MAX
            )
            if safe_non_edge_fast_path or safe_edge_fast_path:
                _append_inheritance_audit(
                    audit_events, frame_idx, "inheritance_accepted", "fast_path",
                    candidate=candidate, state=candidate.state, record=candidate.record,
                    coarse_eligible=True, strict_eligible=True,
                )
                _accept_motion_inheritance(
                    candidate, frame_idx, lost_display_pool, consumed_lost_source_ids,
                    reentry_events, stats,
                )
                stats["inheritance_fast_commit_count"] += 1
                continue

        for (row, col), candidate in coarse_candidates.items():
            state = states_for_class[row]
            record = records_for_class[col]
            held_raw_ids.add(state.track_id)
            _hold_inheritance_candidate(state)
            key = (record.source_raw_track_id, state.track_id)
            tracklet = aggregate_tracklets.get(key)
            if tracklet is None:
                tracklet = AggregateInheritanceTracklet(
                    source_record=record,
                    candidate_raw_track_id=state.track_id,
                    first_frame=frame_idx,
                    last_observed_frame=frame_idx,
                    latest_state=state,
                    latest_candidate=candidate,
                    first_edge_side=candidate.new_edge_side,
                    required_window_frames=(
                        MOTION_SLOW_BASE_WINDOW_FRAMES
                        if context_is_one_to_one
                        else MOTION_SLOW_WINDOW_FRAMES
                    ),
                )
                aggregate_tracklets[key] = tracklet
                stats["inheritance_slow_started_count"] += 1
                _append_inheritance_audit(
                    audit_events, frame_idx, "tracklet_started", "coarse_eligible",
                    tracklet=tracklet, coarse_eligible=True,
                    strict_eligible=(row, col) in strict_candidates,
                )
            elif not context_is_one_to_one:
                tracklet.required_window_frames = MOTION_SLOW_WINDOW_FRAMES
            tracklet.last_observed_frame = frame_idx
            tracklet.latest_state = state
            tracklet.latest_candidate = candidate
            tracklet.scores.append(candidate.score)
            tracklet.centers.append(box_center(state.last_raw_box))
            tracklet.missed_frames = 0

    for key, tracklet in list(aggregate_tracklets.items()):
        source_raw_id, candidate_raw_id = key
        if source_raw_id not in lost_display_pool:
            _append_inheritance_audit(
                audit_events, frame_idx, "tracklet_rejected", "source_unavailable",
                tracklet=tracklet,
            )
            candidate_state = current_by_raw.get(candidate_raw_id)
            if candidate_state is not None:
                _release_inheritance_candidate(candidate_state, frame_idx)
            del aggregate_tracklets[key]
            stats["inheritance_slow_rejected_count"] += 1
            continue
        candidate_state = current_by_raw.get(candidate_raw_id)
        if candidate_state is not None:
            held_raw_ids.add(candidate_raw_id)
            _hold_inheritance_candidate(candidate_state)
        if tracklet.last_observed_frame < frame_idx:
            sampled = (
                candidate_state is not None
                and _sample_existing_aggregate_tracklet(
                    tracklet, candidate_state, frame_idx, width, height
                )
            )
            if not sampled:
                tracklet.missed_frames += 1
        if tracklet.missed_frames > MOTION_SLOW_MISS_TOLERANCE_FRAMES:
            _append_inheritance_audit(
                audit_events, frame_idx, "tracklet_rejected", "miss_tolerance_exceeded",
                tracklet=tracklet,
            )
            if candidate_state is not None:
                _release_inheritance_candidate(candidate_state, frame_idx)
            del aggregate_tracklets[key]
            stats["inheritance_slow_rejected_count"] += 1

    committable_by_source: dict[int, list[AggregateInheritanceTracklet]] = {}
    for tracklet in aggregate_tracklets.values():
        if _aggregate_tracklet_is_committable(tracklet, frame_idx, width, height):
            committable_by_source.setdefault(tracklet.source_record.source_raw_track_id, []).append(tracklet)

    for source_raw_id, options in committable_by_source.items():
        options.sort(key=lambda item: item.mean_score)
        best = options[0]
        if len(options) > 1 and options[1].mean_score - best.mean_score < MOTION_SLOW_COST_MARGIN:
            for option in options:
                _append_inheritance_audit(
                    audit_events, frame_idx, "commit_deferred", "ambiguous_cost_margin",
                    tracklet=option,
                )
            continue
        if best.candidate_raw_track_id not in current_by_raw:
            _append_inheritance_audit(
                audit_events, frame_idx, "commit_deferred", "candidate_not_current",
                tracklet=best,
            )
            continue
        _append_inheritance_audit(
            audit_events, frame_idx, "inheritance_accepted", "slow_path",
            tracklet=best, coarse_eligible=True,
        )
        _accept_motion_inheritance(
            best.latest_candidate,
            frame_idx,
            lost_display_pool,
            consumed_lost_source_ids,
            reentry_events,
            stats,
        )
        stats["inheritance_slow_commit_count"] += 1
        related_keys = [
            key for key in aggregate_tracklets
            if key[0] == source_raw_id or key[1] == best.candidate_raw_track_id
        ]
        for key in related_keys:
            del aggregate_tracklets[key]

    for state in candidate_states:
        if (
            state.track_id not in held_raw_ids
            and state.last_status_reason == "pending_inheritance_candidate"
        ):
            _release_inheritance_candidate(state, frame_idx)


def get_or_create_state(
    states: dict[int, OnlineTrackState],
    track_id: int,
    class_id: int,
    frame_idx: int,
) -> OnlineTrackState:
    state = states.get(track_id)
    if state is None:
        state = OnlineTrackState(
            track_id=track_id,
            class_id=class_id,
            first_seen_frame=frame_idx,
            last_seen_frame=frame_idx,
        )
        states[track_id] = state
    return state


def confirm_pending_track(state: OnlineTrackState, frame_idx: int):
    if state.status == "confirmed":
        return

    first_conf_values = state.conf_history[:FAST_CONFIRM_FRAMES]
    if len(first_conf_values) >= FAST_CONFIRM_FRAMES:
        first_mean = sum(first_conf_values) / len(first_conf_values)
        if first_mean >= FAST_CONFIRM_MEAN_CONF:
            state.status = "confirmed"
            state.confirmed_frame = frame_idx
            state.confirm_reason = "confirmed_fast_conf"
            state.last_status_reason = "confirmed_fast_conf"
            state.visibility_approved = True
            state.visibility_confirmed_frame = frame_idx
            state.visibility_confirm_reason = "confirmed_fast_conf"
            state.visibility_level = "FAST"
            return

    if len(state.conf_history) >= MID_CONFIRM_FRAMES:
        recent_mid = state.conf_history[-MID_CONFIRM_FRAMES:]
        mid_mean = sum(recent_mid) / len(recent_mid)
        mid_hits = sum(conf >= MID_CONFIRM_MIN_CONF for conf in recent_mid)
        if mid_mean >= MID_CONFIRM_MEAN_CONF and mid_hits >= MID_CONFIRM_MIN_HITS:
            state.status = "confirmed"
            state.confirmed_frame = frame_idx
            state.confirm_reason = "confirmed_mid_conf"
            state.last_status_reason = "confirmed_mid_conf"
            state.visibility_approved = True
            state.visibility_confirmed_frame = frame_idx
            state.visibility_confirm_reason = "confirmed_mid_conf"
            state.visibility_level = "MID"
            return

    if state.recent_strong_hits >= STRONG_HIT_COUNT:
        state.status = "confirmed"
        state.confirmed_frame = frame_idx
        state.confirm_reason = "confirmed_window_strong_hits"
        state.last_status_reason = "confirmed_window_strong_hits"
        state.visibility_approved = True
        state.visibility_confirmed_frame = frame_idx
        state.visibility_confirm_reason = "confirmed_window_strong_hits"
        state.visibility_level = "MATURE"
        return

    if state.total_seen >= FALLBACK_HITS and state.mean_conf >= FALLBACK_MEAN_CONF:
        state.status = "confirmed"
        state.confirmed_frame = frame_idx
        state.confirm_reason = "confirmed_fallback_mean_conf"
        state.last_status_reason = "confirmed_fallback_mean_conf"
        state.visibility_approved = True
        state.visibility_confirmed_frame = frame_idx
        state.visibility_confirm_reason = "confirmed_fallback_mean_conf"
        state.visibility_level = "MATURE"
        return

    state.last_status_reason = "pending_waiting_confirmation"


def update_track_state(
    state: OnlineTrackState,
    frame_idx: int,
    class_id: int,
    conf: float,
    raw_box: list[float],
) -> tuple[float, bool]:
    state.previous_raw_box = None if state.last_raw_box is None else list(state.last_raw_box)
    state.previous_seen_frame = state.last_seen_frame
    state.class_id = class_id
    state.last_seen_frame = frame_idx
    state.lost_frames = 0
    state.total_seen += 1
    if state.status != "confirmed":
        state.pending_seen += 1

    state.conf_history.append(conf)
    state.strong_conf_history.append(conf >= STRONG_CONF)
    state.max_conf = max(state.max_conf, conf)
    state.last_conf = conf
    state.last_raw_box = raw_box

    confirm_pending_track(state, frame_idx)


def draw_online_box(frame, state: OnlineTrackState, conf: float, provisional: bool = False):
    if state.last_raw_box is None:
        return

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = state.last_raw_box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w - 1, int(round(x2))))
    y2 = max(0, min(h - 1, int(round(y2))))

    color = BOX_COLORS.get(state.class_id, (255, 255, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if provisional:
        provisional_label = f"D-{DISPLAY_PREFIXES.get(state.class_id, '?')}-"
        label = f"{provisional_label} conf {conf:.2f}"
    elif state.display_id is None:
        label = f"{get_class_name(state.class_id)} R{state.track_id} conf {conf:.2f}"
    else:
        label = f"{get_display_label(state.class_id, state.display_id)} R{state.track_id} conf {conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y1 = max(0, y1 - text_h - baseline - 4)
    label_y2 = label_y1 + text_h + baseline + 4
    label_x2 = min(w - 1, x1 + text_w + 6)
    cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, label_y2 - baseline - 2),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def write_raw_row(writer, frame_idx, image_name, track_id, class_id, conf, box, width, height):
    writer.writerow(
        {
            "frame_idx": frame_idx,
            "image_name": image_name,
            "track_id": int(track_id),
            "class_id": int(class_id),
            "class_name": get_class_name(int(class_id)),
            "conf": f"{float(conf):.6f}",
            "x1": f"{float(box[0]):.3f}",
            "y1": f"{float(box[1]):.3f}",
            "x2": f"{float(box[2]):.3f}",
            "y2": f"{float(box[3]):.3f}",
            "frame_width": width,
            "frame_height": height,
        }
    )


def write_online_row(
    writer,
    frame_idx,
    image_name,
    state: OnlineTrackState,
    conf: float,
    raw_box: list[float],
    status: str,
    status_reason: str,
    width: int,
    height: int,
):
    writer.writerow(
        {
            "frame_idx": frame_idx,
            "image_name": image_name,
            "track_id": int(state.track_id),
            "display_id": "" if state.display_id is None else int(state.display_id),
            "display_label": get_display_label(state.class_id, state.display_id),
            "class_id": int(state.class_id),
            "class_name": get_class_name(state.class_id),
            "conf": f"{float(conf):.6f}",
            "raw_x1": f"{float(raw_box[0]):.3f}",
            "raw_y1": f"{float(raw_box[1]):.3f}",
            "raw_x2": f"{float(raw_box[2]):.3f}",
            "raw_y2": f"{float(raw_box[3]):.3f}",
            "total_seen": state.total_seen,
            "pending_seen": state.pending_seen,
            "mean_conf": f"{float(state.mean_conf):.6f}",
            "recent_strong_hits": state.recent_strong_hits,
            "status": status,
            "status_reason": status_reason,
            "visibility_approved": state.visibility_approved,
            "visibility_confirmed_frame": (
                "" if state.visibility_confirmed_frame is None else int(state.visibility_confirmed_frame)
            ),
            "visibility_confirm_reason": state.visibility_confirm_reason,
            "identity_status": state.identity_status,
            "reentry_source_raw_track_id": (
                "" if state.reentry_source_raw_track_id is None else int(state.reentry_source_raw_track_id)
            ),
            "reentry_gap_frames": "" if state.reentry_gap_frames is None else int(state.reentry_gap_frames),
            "reentry_match_score": (
                "" if state.reentry_match_score is None else f"{float(state.reentry_match_score):.3f}"
            ),
            "motion_model": state.motion_model,
            "motion_center_distance": (
                "" if state.reentry_center_distance is None else f"{float(state.reentry_center_distance):.3f}"
            ),
            "motion_direct_distance": (
                "" if state.motion_direct_distance is None else f"{float(state.motion_direct_distance):.3f}"
            ),
            "motion_mahalanobis_sq": (
                "" if state.motion_mahalanobis_sq is None else f"{float(state.motion_mahalanobis_sq):.3f}"
            ),
            "motion_area_ratio": (
                "" if state.reentry_area_ratio is None else f"{float(state.reentry_area_ratio):.3f}"
            ),
            "frame_width": width,
            "frame_height": height,
        }
    )


def expire_lost_states(states: dict[int, OnlineTrackState], seen_track_ids: set[int]) -> int:
    expired = 0
    to_delete = []
    for track_id, state in states.items():
        if track_id in seen_track_ids:
            continue
        state.lost_frames += 1
        if state.lost_frames > MAX_LOST_FRAMES_TO_KEEP_STATE:
            state.status = "expired"
            to_delete.append(track_id)
    for track_id in to_delete:
        del states[track_id]
        expired += 1
    return expired


def write_report(
    report_path: Path,
    output_name: str,
    processed: int,
    skipped: int,
    stats: dict[str, int],
    all_track_summaries: dict[int, OnlineTrackState],
    enable_reentry_display_id: bool,
    reentry_events: list[dict],
):
    confirmed_tracks = [
        state for state in all_track_summaries.values()
        if state.confirmed_frame is not None
    ]
    shown_tracks = [
        state for state in all_track_summaries.values()
        if state.shown_frames > 0
    ]

    lines = [
        "# Online Tracking Postprocess Report",
        "",
        "## Summary",
        "",
        f"- output_name: `{output_name}`",
        f"- processed_frames: {processed}",
        f"- skipped_frames: {skipped}",
        f"- raw_detection_rows: {stats['raw_detection_rows']}",
        f"- online_rows: {stats['online_rows']}",
        f"- shown_rows: {stats['shown_rows']}",
        f"- pending_rows: {stats['pending_rows']}",
        f"- provisional_shown_rows: {stats.get('provisional_shown_rows', 0)}",
        f"- confirmed_track_count: {len(confirmed_tracks)}",
        f"- expired_track_count: {stats['expired_track_count']}",
        f"- unique_raw_track_count: {len(all_track_summaries)}",
        f"- unique_shown_track_count: {len(shown_tracks)}",
        f"- reentry_enabled: {enable_reentry_display_id}",
        f"- reentry_inheritance_count: {stats['reentry_inheritance_count']}",
        f"- reentry_rejected_ambiguous_count: {stats['reentry_rejected_ambiguous_count']}",
        f"- motion_conflict_rollback_count: {stats['motion_conflict_rollback_count']}",
        f"- inheritance_candidate_started_count: {stats.get('inheritance_candidate_started_count', 0)}",
        f"- inheritance_candidate_committed_count: {stats.get('inheritance_candidate_committed_count', 0)}",
        f"- inheritance_candidate_source_return_count: {stats.get('inheritance_candidate_source_return_count', 0)}",
        f"- inheritance_candidate_rejected_count: {stats.get('inheritance_candidate_rejected_count', 0)}",
        f"- inheritance_context_upgrade_count: {stats.get('inheritance_context_upgrade_count', 0)}",
        f"- inheritance_fast_commit_count: {stats.get('inheritance_fast_commit_count', 0)}",
        f"- inheritance_slow_started_count: {stats.get('inheritance_slow_started_count', 0)}",
        f"- inheritance_slow_commit_count: {stats.get('inheritance_slow_commit_count', 0)}",
        f"- inheritance_slow_source_return_count: {stats.get('inheritance_slow_source_return_count', 0)}",
        f"- inheritance_slow_rejected_count: {stats.get('inheritance_slow_rejected_count', 0)}",
        f"- lost_pool_expired_count: {stats['lost_pool_expired_count']}",
        "",
        "## Class Summary",
        "",
        "| class_id | class_name | raw_rows | shown_rows | pending_rows | raw_tracks | confirmed_tracks | shown_tracks |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for class_id in CLASSES:
        class_tracks = [state for state in all_track_summaries.values() if state.class_id == class_id]
        class_confirmed = [state for state in class_tracks if state.confirmed_frame is not None]
        class_shown = [state for state in class_tracks if state.shown_frames > 0]
        lines.append(
            f"| {class_id} | {get_class_name(class_id)} | "
            f"{stats.get(f'raw_rows_class_{class_id}', 0)} | "
            f"{stats.get(f'shown_rows_class_{class_id}', 0)} | "
            f"{stats.get(f'pending_rows_class_{class_id}', 0)} | "
            f"{len(class_tracks)} | {len(class_confirmed)} | {len(class_shown)} |"
        )

    lines.extend([
        "",
        "## Parameters",
        "",
        "```text",
        f"CONF = {CONF}",
        f"CLASSES = {CLASSES}",
        f"IOU = {IOU}",
        f"FAST_CONFIRM_FRAMES = {FAST_CONFIRM_FRAMES}",
        f"FAST_CONFIRM_MEAN_CONF = {FAST_CONFIRM_MEAN_CONF}",
        f"MID_CONFIRM_FRAMES = {MID_CONFIRM_FRAMES}",
        f"MID_CONFIRM_MEAN_CONF = {MID_CONFIRM_MEAN_CONF}",
        f"MID_CONFIRM_MIN_CONF = {MID_CONFIRM_MIN_CONF}",
        f"MID_CONFIRM_MIN_HITS = {MID_CONFIRM_MIN_HITS}",
        f"PENDING_WINDOW = {PENDING_WINDOW}",
        f"STRONG_CONF = {STRONG_CONF}",
        f"STRONG_HIT_COUNT = {STRONG_HIT_COUNT}",
        f"FALLBACK_HITS = {FALLBACK_HITS}",
        f"FALLBACK_MEAN_CONF = {FALLBACK_MEAN_CONF}",
        f"MAX_LOST_FRAMES_TO_KEEP_STATE = {MAX_LOST_FRAMES_TO_KEEP_STATE}",
        f"ENABLE_REENTRY_DISPLAY_ID = {enable_reentry_display_id}",
        f"REENTRY_MAX_GAP_FRAMES = {REENTRY_MAX_GAP_FRAMES}",
        f"REENTRY_EDGE_MARGIN_PX = {REENTRY_EDGE_MARGIN_PX}",
        f"REENTRY_CENTER_DISTANCE_PX = {REENTRY_CENTER_DISTANCE_PX}",
        f"REENTRY_AREA_RATIO_MIN = {REENTRY_AREA_RATIO_MIN}",
        f"REENTRY_AREA_RATIO_MAX = {REENTRY_AREA_RATIO_MAX}",
        f"REENTRY_AMBIGUOUS_SCORE_MARGIN = {REENTRY_AMBIGUOUS_SCORE_MARGIN}",
        f"MOTION_MIN_GAP_FRAMES = {MOTION_MIN_GAP_FRAMES}",
        f"MOTION_MAHALANOBIS_THRESHOLD = {MOTION_MAHALANOBIS_THRESHOLD}",
        f"MOTION_FAST_REJOIN_MAX_GAP_FRAMES = {MOTION_FAST_REJOIN_MAX_GAP_FRAMES}",
        f"MOTION_FAST_REJOIN_MIN_CONF = {MOTION_FAST_REJOIN_MIN_CONF}",
        f"MOTION_FAST_REJOIN_DISTANCE_PX = {MOTION_FAST_REJOIN_DISTANCE_PX}",
        f"MOTION_ABSOLUTE_DIRECT_DISTANCE_PX = {MOTION_ABSOLUTE_DIRECT_DISTANCE_PX}",
        "```",
        "",
        "## Per-track Summary",
        "",
        "| raw_track_id | class_id | class_name | display_id | display_label | first_seen | last_seen | total_seen | shown_frames | mean_conf | max_conf | confirmed_frame | first_shown_frame | confirm_reason | reentry_source_raw_track_id | reentry_gap_frames | reentry_match_score |",
        "|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ])

    for track_id in sorted(all_track_summaries):
        state = all_track_summaries[track_id]
        confirmed_frame = "" if state.confirmed_frame is None else str(state.confirmed_frame)
        display_id = "" if state.display_id is None else str(state.display_id)
        first_shown_frame = "" if state.first_shown_frame is None else str(state.first_shown_frame)
        reentry_source = "" if state.reentry_source_raw_track_id is None else str(state.reentry_source_raw_track_id)
        reentry_gap = "" if state.reentry_gap_frames is None else str(state.reentry_gap_frames)
        reentry_score = "" if state.reentry_match_score is None else f"{state.reentry_match_score:.3f}"
        lines.append(
            "| "
            f"{state.track_id} | "
            f"{state.class_id} | "
            f"{get_class_name(state.class_id)} | "
            f"{display_id} | "
            f"{get_display_label(state.class_id, state.display_id)} | "
            f"{state.first_seen_frame} | "
            f"{state.last_seen_frame} | "
            f"{state.total_seen} | "
            f"{state.shown_frames} | "
            f"{state.mean_conf:.6f} | "
            f"{state.max_conf:.6f} | "
            f"{confirmed_frame} | "
            f"{first_shown_frame} | "
            f"{state.confirm_reason} | "
            f"{reentry_source} | "
            f"{reentry_gap} | "
            f"{reentry_score} |"
        )

    lines.extend(
        [
            "",
            "## Display ID Mapping",
            "",
            "| class_id | class_name | display_id | display_label | raw_track_id | first_shown_frame | shown_frames | confirm_reason |",
            "|---:|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for state in sorted(shown_tracks, key=lambda s: (s.class_id, s.display_id or 0, s.track_id)):
        first_shown_frame = "" if state.first_shown_frame is None else str(state.first_shown_frame)
        lines.append(
            "| "
            f"{state.class_id} | "
            f"{get_class_name(state.class_id)} | "
            f"{state.display_id} | "
            f"{get_display_label(state.class_id, state.display_id)} | "
            f"{state.track_id} | "
            f"{first_shown_frame} | "
            f"{state.shown_frames} | "
            f"{state.confirm_reason} |"
        )

    lines.extend(
        [
            "",
            "## Re-entry Display ID Inheritance",
            "",
            "| class_id | class_name | display_id | display_label | source_raw_track_id | new_raw_track_id | gap_frames | center_distance | area_ratio | edge_side_source | edge_side_new | score |",
            "|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|---:|",
        ]
    )
    for event in reentry_events:
        lines.append(
            "| "
            f"{event['class_id']} | "
            f"{event['class_name']} | "
            f"{event['display_id']} | "
            f"{event['display_label']} | "
            f"{event['source_raw_track_id']} | "
            f"{event['new_raw_track_id']} | "
            f"{event['gap_frames']} | "
            f"{event['center_distance']:.3f} | "
            f"{event['area_ratio']:.3f} | "
            f"{event['edge_side_source']} | "
            f"{event['edge_side_new']} | "
            f"{event['score']:.3f} |"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class FrameTrackObservation:
    raw_track_id: int
    class_id: int
    confidence: float
    raw_box: list[float]
    image_name: str = ""


@dataclass
class FrameRuleDecision:
    state: OnlineTrackState
    confidence: float
    raw_box: list[float]
    class_id: int
    image_name: str
    status: str
    status_reason: str
    draw_mode: str


@dataclass
class OnlineFrameRuleState:
    tracks: dict[int, OnlineTrackState]
    summaries: dict[int, OnlineTrackState]
    lost_display_pool: dict[int, LostDisplayRecord]
    consumed_lost_source_ids: set[int]
    reentry_events: list[dict]
    next_display_ids: dict[int, int]
    aggregate_tracklets: dict[tuple[int, int], AggregateInheritanceTracklet]
    stats: dict[str, int]
    inheritance_audit_events: list[dict] | None = None


@dataclass
class OnlineFrameRuleOutput:
    next_state: OnlineFrameRuleState
    decisions: list[FrameRuleDecision]
    seen_raw_track_ids: set[int]


def process_tracking_frame(
    observations: list[FrameTrackObservation],
    frame_idx: int,
    width: int,
    height: int,
    previous_state: OnlineFrameRuleState,
    enable_id_inheritance: bool = True,
) -> OnlineFrameRuleOutput:
    """Apply causal visibility and identity rules for exactly one camera frame."""
    frame_items: list[tuple] = []
    seen_track_ids: set[int] = set()

    for observation in observations:
        if observation.raw_track_id in seen_track_ids:
            raise ValueError(
                f"Duplicate raw_track_id {observation.raw_track_id} in frame {frame_idx}."
            )
        seen_track_ids.add(observation.raw_track_id)
        state = get_or_create_state(
            previous_state.tracks,
            observation.raw_track_id,
            observation.class_id,
            frame_idx,
        )
        previous_state.summaries[observation.raw_track_id] = state
        update_track_state(
            state,
            frame_idx,
            observation.class_id,
            observation.confidence,
            observation.raw_box,
        )
        frame_items.append(
            (
                state,
                observation.confidence,
                observation.raw_box,
                observation.class_id,
                observation.image_name,
            )
        )

    if enable_id_inheritance:
        apply_motion_inheritance_for_frame(
            frame_items,
            frame_idx,
            width,
            height,
            previous_state.lost_display_pool,
            previous_state.consumed_lost_source_ids,
            previous_state.reentry_events,
            previous_state.stats,
            previous_state.aggregate_tracklets,
            previous_state.inheritance_audit_events,
        )
        resolve_display_identity_conflicts(frame_items, frame_idx, previous_state.stats)

    decisions: list[FrameRuleDecision] = []
    for state, conf, raw_box, class_id, image_name in frame_items:
        if state.status == "confirmed":
            status = "confirmed"
            if state.display_id is None:
                status_reason = assign_display_id(
                    state,
                    frame_idx,
                    width,
                    height,
                    previous_state.next_display_ids,
                    False,
                    previous_state.lost_display_pool,
                    previous_state.consumed_lost_source_ids,
                    previous_state.reentry_events,
                    previous_state.stats,
                )
            elif state.last_status_reason == "inherited_motion_display_id":
                status_reason = "inherited_motion_display_id"
            else:
                status_reason = "shown"
            if state.first_shown_frame is None:
                state.first_shown_frame = frame_idx
            state.last_shown_frame = frame_idx
            state.shown_frames += 1
            draw_mode = "formal"
            previous_state.stats["shown_rows"] += 1
            previous_state.stats[f"shown_rows_class_{class_id}"] += 1
        else:
            status = "pending"
            status_reason = state.last_status_reason
            draw_mode = "hidden"
            if getattr(state, "provisional_box_visible", False):
                status_reason = "shown_provisional_identity_pending"
                draw_mode = "provisional"
                previous_state.stats["provisional_shown_rows"] = (
                    previous_state.stats.get("provisional_shown_rows", 0) + 1
                )
                class_key = f"provisional_shown_rows_class_{class_id}"
                previous_state.stats[class_key] = previous_state.stats.get(class_key, 0) + 1
            previous_state.stats["pending_rows"] += 1
            previous_state.stats[f"pending_rows_class_{class_id}"] += 1

        decisions.append(
            FrameRuleDecision(
                state=state,
                confidence=conf,
                raw_box=raw_box,
                class_id=class_id,
                image_name=image_name,
                status=status,
                status_reason=status_reason,
                draw_mode=draw_mode,
            )
        )
        previous_state.stats["online_rows"] += 1

    return OnlineFrameRuleOutput(
        next_state=previous_state,
        decisions=decisions,
        seen_raw_track_ids=seen_track_ids,
    )


def main():
    args = parse_args()
    output_name = args.output_name
    validate_output_name(output_name)
    tracker_path = args.tracker_path or TRACKER
    enable_reentry_display_id = args.enable_reentry_display_id
    enable_inheritance_audit = args.inheritance_audit_csv
    input_dir = Path(args.source_dir)
    model_path = Path(args.model).expanduser()

    output_img_dir = Path(args.output_img_root).expanduser() / output_name
    output_video_dir = Path(args.output_video_root).expanduser() / output_name
    output_video_path = output_video_dir / f"{output_name}.mp4"
    raw_tracks_path = output_video_dir / f"{output_name}_tracks_raw.csv"
    online_tracks_path = output_video_dir / f"{output_name}_tracks_online.csv"
    report_path = output_video_dir / f"{output_name}_online_report.md"
    inheritance_audit_path = output_video_dir / f"{output_name}_inheritance_audit.csv"

    check_paths(
        model_path,
        output_img_dir,
        output_video_dir,
        input_dir,
    )

    img_files = collect_images(input_dir)
    if not img_files:
        raise RuntimeError(f"No usable images found in input directory: {input_dir}")

    print("=" * 70)
    print("Online tracking postprocess job started")
    print(f"OUTPUT_NAME       : {output_name}")
    print(f"MODEL_PATH        : {model_path}")
    print(f"INPUT_DIR         : {input_dir}")
    print(f"OUTPUT_IMG_DIR    : {output_img_dir}")
    print(f"OUTPUT_VIDEO_DIR  : {output_video_dir}")
    print(f"OUTPUT_VIDEO_PATH : {output_video_path}")
    print(f"RAW_TRACKS_CSV    : {raw_tracks_path}")
    print(f"ONLINE_TRACKS_CSV : {online_tracks_path}")
    print(f"ONLINE_REPORT     : {report_path}")
    if enable_inheritance_audit:
        print(f"INHERITANCE_AUDIT : {inheritance_audit_path}")
    print(f"FPS               : {FPS}")
    print(f"DEVICE            : {DEVICE}")
    print(f"CONF              : {CONF}")
    print(f"IMGSZ             : {IMGSZ}")
    print(f"CLASSES           : {CLASSES}")
    print(f"TRACKER           : {tracker_path}")
    print(f"IOU               : {IOU}")
    print(f"REENTRY_DISPLAY_ID: {enable_reentry_display_id}")
    print(f"TOTAL_IMAGES      : {len(img_files)}")
    print("=" * 70)

    print("First 5 sorted files:")
    for p in img_files[:5]:
        print("  ", p.name)

    from ultralytics import YOLO

    model = YOLO(str(model_path))

    writer = None
    raw_csv_file = None
    online_csv_file = None
    raw_csv_writer = None
    online_csv_writer = None
    processed = 0
    skipped = 0
    states: dict[int, OnlineTrackState] = {}
    all_track_summaries: dict[int, OnlineTrackState] = {}
    lost_display_pool: dict[int, LostDisplayRecord] = {}
    consumed_lost_source_ids: set[int] = set()
    reentry_events: list[dict] = []
    inheritance_audit_events: list[dict] | None = [] if enable_inheritance_audit else None
    next_display_ids = {class_id: 1 for class_id in CLASSES}
    stats = {
        "raw_detection_rows": 0,
        "online_rows": 0,
        "shown_rows": 0,
        "pending_rows": 0,
        "expired_track_count": 0,
        "reentry_inheritance_count": 0,
        "reentry_rejected_ambiguous_count": 0,
        "motion_conflict_rollback_count": 0,
        "lost_pool_expired_count": 0,
    }
    for class_id in CLASSES:
        stats[f"raw_rows_class_{class_id}"] = 0
        stats[f"shown_rows_class_{class_id}"] = 0
        stats[f"pending_rows_class_{class_id}"] = 0
    rule_state = OnlineFrameRuleState(
        tracks=states,
        summaries=all_track_summaries,
        lost_display_pool=lost_display_pool,
        consumed_lost_source_ids=consumed_lost_source_ids,
        reentry_events=reentry_events,
        next_display_ids=next_display_ids,
        aggregate_tracklets={},
        stats=stats,
        inheritance_audit_events=inheritance_audit_events,
    )

    try:
        raw_csv_file = raw_tracks_path.open("w", newline="", encoding="utf-8")
        online_csv_file = online_tracks_path.open("w", newline="", encoding="utf-8")

        raw_csv_writer = csv.DictWriter(raw_csv_file, fieldnames=RAW_CSV_FIELDS)
        online_csv_writer = csv.DictWriter(online_csv_file, fieldnames=ONLINE_CSV_FIELDS)
        raw_csv_writer.writeheader()
        online_csv_writer.writeheader()

        for idx, img_path in enumerate(img_files):
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[WARN] Failed to read image, skipped: {img_path}")
                skipped += 1
                continue

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            h, w = frame.shape[:2]

            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_video_path), fourcc, FPS, (w, h))
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to create video writer: {output_video_path}")

            track_kwargs = {
                "source": frame,
                "persist": True,
                "tracker": tracker_path,
                "conf": CONF,
                "iou": IOU,
                "classes": CLASSES,
                "device": DEVICE,
                "verbose": VERBOSE,
            }
            if IMGSZ is not None:
                track_kwargs["imgsz"] = IMGSZ

            results = model.track(**track_kwargs)
            result = results[0]
            raw_track_annotated = result.plot()
            online_annotated = frame.copy()
            observations: list[FrameTrackObservation] = []

            boxes = result.boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.detach().cpu().numpy().astype(int)
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy()
                xyxy = boxes.xyxy.detach().cpu().numpy()

                for track_id, class_id, conf, box in zip(ids, classes, confs, xyxy):
                    track_id = int(track_id)
                    class_id = int(class_id)
                    conf = float(conf)
                    raw_box = [float(v) for v in box]
                    write_raw_row(
                        raw_csv_writer,
                        idx,
                        img_path.name,
                        track_id,
                        class_id,
                        conf,
                        raw_box,
                        w,
                        h,
                    )
                    stats["raw_detection_rows"] += 1
                    stats[f"raw_rows_class_{class_id}"] += 1

                    observations.append(
                        FrameTrackObservation(
                            raw_track_id=track_id,
                            class_id=class_id,
                            confidence=conf,
                            raw_box=raw_box,
                            image_name=img_path.name,
                        )
                    )

            frame_rule_output = process_tracking_frame(
                observations,
                idx,
                w,
                h,
                rule_state,
                enable_id_inheritance=enable_reentry_display_id,
            )
            rule_state = frame_rule_output.next_state
            frame_decisions = frame_rule_output.decisions
            seen_track_ids = frame_rule_output.seen_raw_track_ids

            for decision in frame_decisions:
                if decision.draw_mode == "formal":
                    draw_online_box(online_annotated, decision.state, decision.confidence)
                elif decision.draw_mode == "provisional":
                    draw_online_box(
                        online_annotated,
                        decision.state,
                        decision.confidence,
                        provisional=True,
                    )

                write_online_row(
                    online_csv_writer,
                    idx,
                    decision.image_name,
                    decision.state,
                    decision.confidence,
                    decision.raw_box,
                    decision.status,
                    decision.status_reason,
                    w,
                    h,
                )

            if enable_reentry_display_id:
                add_disappeared_tracks_to_lost_pool(
                    states,
                    seen_track_ids,
                    lost_display_pool,
                    consumed_lost_source_ids,
                    w,
                    h,
                )
                stats["lost_pool_expired_count"] += expire_lost_display_pool(
                    lost_display_pool,
                    idx,
                )
            stats["expired_track_count"] += expire_lost_states(states, seen_track_ids)

            if SAVE_WITH_ORIGINAL_NAME:
                out_img_name = f"{img_path.stem}_tracked.jpg"
            else:
                out_img_name = f"{idx:06d}.jpg"

            out_img_path = output_img_dir / out_img_name
            ok = cv2.imwrite(str(out_img_path), raw_track_annotated)
            if not ok:
                print(f"[WARN] Failed to save result image: {out_img_path}")

            writer.write(online_annotated)
            processed += 1

            if SHOW_PREVIEW:
                cv2.imshow("Online Tracking Preview", online_annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    print("[INFO] ESC detected, stopping early.")
                    break

            if (idx + 1) % 100 == 0 or (idx + 1) == len(img_files):
                print(f"[INFO] Processed {idx + 1}/{len(img_files)}")

    finally:
        if raw_csv_file is not None:
            raw_csv_file.close()
        if online_csv_file is not None:
            online_csv_file.close()
        if writer is not None:
            writer.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    write_report(
        report_path,
        output_name,
        processed,
        skipped,
        stats,
        all_track_summaries,
        enable_reentry_display_id,
        reentry_events,
    )

    if inheritance_audit_events is not None:
        for tracklet in rule_state.aggregate_tracklets.values():
            _append_inheritance_audit(
                inheritance_audit_events, processed - 1, "tracklet_unresolved", "end_of_run",
                tracklet=tracklet,
            )
        with inheritance_audit_path.open("w", newline="", encoding="utf-8") as audit_file:
            audit_writer = csv.DictWriter(audit_file, fieldnames=INHERITANCE_AUDIT_FIELDS)
            audit_writer.writeheader()
            audit_writer.writerows(inheritance_audit_events)

    print("=" * 70)
    print("Online tracking postprocess finished")
    print(f"Processed frames  : {processed}")
    print(f"Skipped frames    : {skipped}")
    print(f"Saved images to   : {output_img_dir}")
    print(f"Saved video to    : {output_video_path}")
    print(f"Saved raw tracks  : {raw_tracks_path}")
    print(f"Saved online CSV  : {online_tracks_path}")
    print(f"Saved report      : {report_path}")
    if enable_inheritance_audit:
        print(f"Saved audit CSV   : {inheritance_audit_path}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
