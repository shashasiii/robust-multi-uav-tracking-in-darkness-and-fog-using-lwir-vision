# -*- coding: UTF-8 -*-
"""
Stereo tracking + distance estimation pipeline for left/right thermal folders.

This script keeps the existing online rule state for each eye independently
(`track_folder_to_video_online_postprocess_stable_v19.py`), then performs
per-frame Hungarian matching on confirmed detections grouped by class.

Outputs:
- Annotated stitched side-by-side images
- Side-by-side video
- Stereo matched distance CSV
- A compact report with matching and distance statistics
- A per-frame distance scatter plot (unless disabled)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import csv
import json
import math
import re
import sys
import time
from typing import Any

import cv2
import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO

# Ensure root repo path is in module search path when running this file as a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracking.track_folder_to_video_online_postprocess_stable_v19 import (
    FrameTrackObservation,
    OnlineFrameRuleState,
    process_tracking_frame,
)
from tracking import track_folder_to_video_online_postprocess_stable_v19 as tracking_rules


# =========================================================
# 1) User configuration
# =========================================================

STEREO_PARAMS_PATH = REPO_ROOT / "stereo" / "thermal_stereo_params.yaml"
OUTPUT_IMG_ROOT = REPO_ROOT / "outputs" / "stereo" / "images"
OUTPUT_VIDEO_ROOT = REPO_ROOT / "outputs" / "stereo" / "artifacts"

FPS = 25

DEVICE = "0"
CUDA_DEVICE_INDEX = 0
CONF = 0.30
IMGSZ = None
VERBOSE = False
IOU = 0.7
TRACKER = str(Path(__file__).resolve().parent.parent / "tracking" / "botsort_uav_stable.yaml")

CLASSES = [0, 1]
CLASS_NAMES = {0: "drone", 1: "person"}
CLASS_PREFIX = {0: "D", 1: "P"}

TIMESTAMP_PATTERN = re.compile(
    r"thermal_inferno_image_(\d+)_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$",
    re.IGNORECASE,
)
ALT_TIMESTAMP_PATTERN = re.compile(r"thermal_(?:inferno|grayscale)_image_(\d+)_(\d+)\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)
OUTPUT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
VALID_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

MATCH_Y_GAP_PX = 12.0
MATCH_AREA_RATIO_WEIGHT = 12.0
MATCH_CENTER_COST_LIMIT = 160.0
MATCH_MAX_DISPARITY_PX = 260.0
PAIR_STALE_FRAMES = 20
STEREO_SYNC_TOLERANCE_MS = 20.0
STEREO_SYNC_PHASE_TOLERANCE_MS = 5.0
STEREO_SYNC_PHASE_WINDOW = 31
STEREO_SYNC_PHASE_MIN_SAMPLES = 5
FONT_SCALE = 0.40
FONT_THICKNESS = 1


CSV_FIELDS = [
    "frame_idx",
    "image_name_left",
    "image_name_right",
    "class_id",
    "class_name",
    "stereo_pair_id",
    "pair_label",
    "left_raw_track_id",
    "right_raw_track_id",
    "left_display_id",
    "right_display_id",
    "match_cost",
    "x_left",
    "y_left",
    "x_right",
    "y_right",
    "disparity_px",
    "depth_m",
    "filtered_depth_m",
    "distance_valid",
    "filter_reason",
    "distance_filter_mode",
    "filtered_disparity_px",
    "reacquire_candidate_count",
    "reacquire_event",
    "raw_id_changed",
    "y_diff_px",
    "center_distance_px",
    "left_conf",
    "right_conf",
    "left_status",
    "right_status",
    "left_status_reason",
    "right_status_reason",
    "left_draw_mode",
    "right_draw_mode",
    "left_timestamp_ns",
    "right_timestamp_ns",
    "stereo_delta_ns",
    "stereo_delta_ms",
    "temporal_common_timestamp_ns",
    "temporal_compensation_enabled",
    "temporal_compensation_valid",
    "temporal_compensation_applied",
    "temporal_compensation_reason",
    "temporal_invalid_policy",
    "left_motion_samples",
    "right_motion_samples",
    "left_history_span_ms",
    "right_history_span_ms",
    "left_vx_rect_px_s",
    "left_vy_rect_px_s",
    "right_vx_rect_px_s",
    "right_vy_rect_px_s",
    "left_shift_px",
    "right_shift_px",
    "left_shift_x_px",
    "left_shift_y_px",
    "right_shift_x_px",
    "right_shift_y_px",
    "left_rect_x_raw",
    "left_rect_y_raw",
    "right_rect_x_raw",
    "right_rect_y_raw",
    "left_rect_x_compensated",
    "left_rect_y_compensated",
    "right_rect_x_compensated",
    "right_rect_y_compensated",
    "disparity_raw_px",
    "disparity_compensated_px",
    "y_diff_raw_px",
    "y_diff_compensated_px",
]


@dataclass
class StereoCameraGeometry:
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    K_left: np.ndarray
    K_right: np.ndarray
    dist_left: np.ndarray
    dist_right: np.ndarray
    T_m: np.ndarray
    size: tuple[int, int]
    baseline_m: float


@dataclass(frozen=True)
class StereoFramePair:
    left_path: Path
    right_path: Path
    left_timestamp_ns: int
    right_timestamp_ns: int

    @property
    def stereo_delta_ns(self) -> int:
        """Signed recorded-frame offset: right host timestamp minus left."""
        return self.right_timestamp_ns - self.left_timestamp_ns

    @property
    def common_timestamp_ns(self) -> int:
        return self.left_timestamp_ns + self.stereo_delta_ns // 2


@dataclass(frozen=True)
class RectifiedMotionSample:
    timestamp_ns: int
    x: float
    y: float


@dataclass
class RectifiedMotionState:
    samples: list[RectifiedMotionSample] = field(default_factory=list)
    last_seen_frame: int = -1


@dataclass(frozen=True)
class MotionEstimate:
    vx_px_s: float
    vy_px_s: float
    sample_count: int
    span_ms: float


@dataclass(frozen=True)
class TemporalCompensationResult:
    enabled: bool
    valid: bool
    applied: bool
    reason: str
    common_timestamp_ns: int
    left_raw: tuple[float, float]
    right_raw: tuple[float, float]
    left_compensated: tuple[float, float] | None
    right_compensated: tuple[float, float] | None
    left_motion: MotionEstimate | None
    right_motion: MotionEstimate | None
    left_shift_px: float | None
    right_shift_px: float | None
    disparity_raw_px: float
    disparity_compensated_px: float | None
    y_diff_raw_px: float
    y_diff_compensated_px: float | None


@dataclass
class TrackedPair:
    stereo_pair_id: int
    class_id: int
    label: str
    left_raw_id: int
    right_raw_id: int
    left_state_display_id: int | None
    right_state_display_id: int | None
    left_decision: Any
    right_decision: Any
    match_cost: float
    left_center_rect: tuple[float, float]
    right_center_rect: tuple[float, float]
    depth_m: float | None
    distance_valid: bool
    filtered_depth_m: float | None
    disparity_px: float | None
    ema_invalid: bool
    filter_reason: str
    distance_filter_mode: str
    filtered_disparity_px: float | None
    reacquire_candidate_count: int
    reacquire_event: bool
    raw_id_changed: bool
    temporal: TemporalCompensationResult


@dataclass
class StereoPairState:
    pair_id: int
    class_id: int
    left_raw_id: int
    right_raw_id: int
    last_seen_frame: int
    ema_distance: float | None = None
    ema_count: int = 0
    ema_streak_invalid: int = 0
    ema_recoveries: int = 0
    disparity_ema: float | None = None
    disparity_candidates: list[float] = field(default_factory=list)
    disparity_candidate_last_frame: int | None = None
    disparity_forced_reacquire: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Track left/right thermal folders with per-eye online rules, "
            "match same-frame decisions by class using Hungarian assignment, "
            "and estimate stereo distance."
        ),
    )
    parser.add_argument("--output-name", required=True, help="Unique run name.")
    parser.add_argument(
        "--output-img-root",
        default=str(OUTPUT_IMG_ROOT),
        help="Directory for annotated stitched images.",
    )
    parser.add_argument(
        "--output-video-root",
        default=str(OUTPUT_VIDEO_ROOT),
        help="Directory for stitched videos, CSV, report, and distance plot outputs.",
    )
    parser.add_argument(
        "--left-dir",
        required=True,
        help="Left-camera image folder.",
    )
    parser.add_argument(
        "--right-dir",
        required=True,
        help="Right-camera image folder.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="YOLO .pt weights path.",
    )
    parser.add_argument(
        "--params-path",
        default=str(STEREO_PARAMS_PATH),
        help="YAML with thermal stereo intrinsics/extrinsics.",
    )
    parser.add_argument(
        "--tracker-path",
        default=TRACKER,
        help="BoT-SORT tracker YAML path.",
    )
    parser.add_argument("--fps", type=float, default=FPS, help="Output video fps.")
    parser.add_argument(
        "--latency-report-path",
        default="",
        help=(
            "Optional JSON path for core-pipeline latency. Timing includes dual-eye "
            "YOLO/BoT-SORT, online rules, stereo matching, compensation/depth, and distance filtering; "
            "it excludes file decoding, drawing, CSV/image/video writes, and plotting."
        ),
    )
    parser.add_argument(
        "--latency-warmup-frames",
        type=int,
        default=25,
        help="Frames excluded from latency statistics while CUDA and trackers warm up.",
    )
    parser.add_argument("--match-y-gap", type=float, default=MATCH_Y_GAP_PX)
    parser.add_argument("--match-cost-limit", type=float, default=MATCH_CENTER_COST_LIMIT)
    parser.add_argument("--match-max-disparity", type=float, default=MATCH_MAX_DISPARITY_PX)
    parser.add_argument(
        "--pair-ttl",
        type=int,
        default=PAIR_STALE_FRAMES,
        help="How many frames to keep unmatched pair IDs for continuity.",
    )
    parser.add_argument(
        "--sync-tolerance-ms",
        type=float,
        default=STEREO_SYNC_TOLERANCE_MS,
        help="Max timestamp difference (ms) when pairing left/right filenames.",
    )
    parser.add_argument(
        "--sync-phase-tolerance-ms",
        type=float,
        default=STEREO_SYNC_PHASE_TOLERANCE_MS,
        help="Max deviation from the rolling signed timestamp phase; <=0 disables the guard.",
    )
    parser.add_argument(
        "--sync-phase-window",
        type=int,
        default=STEREO_SYNC_PHASE_WINDOW,
        help="Accepted-pair window used for the rolling signed timestamp median.",
    )
    parser.add_argument(
        "--sync-phase-min-samples",
        type=int,
        default=STEREO_SYNC_PHASE_MIN_SAMPLES,
        help="Accepted pairs required before enforcing the phase guard.",
    )
    parser.add_argument(
        "--enable-causal-time-compensation",
        action="store_true",
        help=(
            "Predict both rectified centers to the midpoint of their recorded host-side "
            "frame timestamps before disparity estimation."
        ),
    )
    parser.add_argument("--temporal-history-size", type=int, default=5)
    parser.add_argument("--temporal-min-history-samples", "--temporal-min-samples", dest="temporal_min_samples", type=int, default=2)
    parser.add_argument("--temporal-velocity-decay", "--temporal-decay", dest="temporal_decay", type=float, default=0.70)
    parser.add_argument("--temporal-max-history-gap-ms", "--temporal-max-gap-ms", dest="temporal_max_gap_ms", type=float, default=120.0)
    parser.add_argument("--temporal-max-shift-px", type=float, default=12.0)
    parser.add_argument("--temporal-prediction-margin-px", "--temporal-bound-margin-px", dest="temporal_bound_margin_px", type=float, default=16.0)
    parser.add_argument(
        "--temporal-invalid-policy",
        choices=("invalid", "raw"),
        default="invalid",
        help="On compensation failure, invalidate distance (default) or use raw disparity for diagnostics.",
    )
    parser.add_argument(
        "--use-provisional",
        action="store_true",
        help="Also allow provisional status rows in stereo matching.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process first N frames only (0 means full sequence).",
    )
    parser.add_argument(
        "--distance-ema-alpha",
        type=float,
        default=0.35,
        help="EMA smoothing factor for valid distance history (0~1).",
    )
    parser.add_argument(
        "--distance-ema-max-change",
        type=float,
        default=0.6,
        help="Mark distance invalid when jump exceeds prev*threshold.",
    )
    parser.add_argument(
        "--distance-ema-min-abs-change",
        type=float,
        default=0.5,
        help="Additional absolute change floor in meters (helps small-distance robustness).",
    )
    parser.add_argument(
        "--distance-min-disparity",
        type=float,
        default=2.0,
        help="Reject distance if stereo disparity is below this value (px).",
    )
    parser.add_argument(
        "--distance-ema-reset-after-invalid",
        type=int,
        default=3,
        help="Reset/reacquire rule after this many consecutive invalid distance frames.",
    )
    parser.add_argument(
        "--distance-filter-mode",
        choices=("distance_ema", "disparity_ema_reacquire"),
        default="distance_ema",
        help="Distance-domain legacy EMA or disparity-domain EMA with causal reacquisition.",
    )
    parser.add_argument("--disparity-ema-alpha", type=float, default=0.35)
    parser.add_argument("--disparity-ema-jump-ratio", type=float, default=0.50)
    parser.add_argument("--disparity-ema-jump-abs-px", type=float, default=2.0)
    parser.add_argument("--disparity-reacquire-samples", type=int, default=3)
    parser.add_argument("--disparity-reacquire-stability-ratio", type=float, default=0.10)
    parser.add_argument("--disparity-reacquire-stability-abs-px", type=float, default=0.50)
    parser.add_argument("--disparity-reacquire-max-frame-gap", type=int, default=2)
    parser.add_argument(
        "--skip-distance-plot",
        action="store_true",
        help="Do not generate the distance scatter plot after the CSV is written.",
    )
    inheritance_group = parser.add_mutually_exclusive_group()
    inheritance_group.add_argument(
        "--enable-id-inheritance",
        action="store_true",
        default=True,
        help="Enable online display-ID inheritance (default).",
    )
    inheritance_group.add_argument(
        "--disable-id-inheritance",
        action="store_false",
        dest="enable_id_inheritance",
        help="Disable display-ID inheritance but keep visibility gating.",
    )
    return parser.parse_args()


def validate_output_name(output_name: str):
    if not OUTPUT_NAME_PATTERN.match(output_name):
        raise ValueError(
            "Invalid --output-name. Use only letters, numbers, underscore, dash, and dot; "
            "do not include path separators."
        )


def validate_temporal_args(args):
    if args.sync_tolerance_ms <= 0.0:
        raise ValueError("--sync-tolerance-ms must be > 0")
    if args.sync_phase_window < 1:
        raise ValueError("--sync-phase-window must be >= 1")
    if not 1 <= args.sync_phase_min_samples <= args.sync_phase_window:
        raise ValueError("--sync-phase-min-samples must be in [1, sync-phase-window]")
    if args.temporal_history_size < 2:
        raise ValueError("--temporal-history-size must be >= 2")
    if not 2 <= args.temporal_min_samples <= args.temporal_history_size:
        raise ValueError("--temporal-min-samples must be in [2, temporal-history-size]")
    if not 0.0 < args.temporal_decay <= 1.0:
        raise ValueError("--temporal-decay must be in (0, 1]")
    if args.temporal_max_gap_ms <= 0.0:
        raise ValueError("--temporal-max-gap-ms must be > 0")
    if args.temporal_max_shift_px <= 0.0:
        raise ValueError("--temporal-max-shift-px must be > 0")
    if args.temporal_bound_margin_px < 0.0:
        raise ValueError("--temporal-bound-margin-px must be >= 0")
    if args.latency_warmup_frames < 0:
        raise ValueError("--latency-warmup-frames must be >= 0")
    if not 0.0 < args.disparity_ema_alpha <= 1.0:
        raise ValueError("--disparity-ema-alpha must be in (0, 1]")
    if min(
        args.disparity_ema_jump_ratio,
        args.disparity_ema_jump_abs_px,
        args.disparity_reacquire_stability_ratio,
        args.disparity_reacquire_stability_abs_px,
    ) < 0.0:
        raise ValueError("Disparity EMA thresholds must be non-negative")
    if args.disparity_reacquire_samples < 2:
        raise ValueError("--disparity-reacquire-samples must be >= 2")
    if args.disparity_reacquire_max_frame_gap < 1:
        raise ValueError("--disparity-reacquire-max-frame-gap must be >= 1")


def sort_key(p: Path):
    m = TIMESTAMP_PATTERN.match(p.name)
    if m:
        return (0, int(m.group(1)), int(m.group(2)), p.name.lower())
    m_alt = ALT_TIMESTAMP_PATTERN.match(p.name)
    if m_alt:
        return (0, int(m_alt.group(1)), int(m_alt.group(2)), p.name.lower())
    return (1, p.name.lower())


def _frame_signature(p: Path) -> str:
    m = TIMESTAMP_PATTERN.match(p.name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m_alt = ALT_TIMESTAMP_PATTERN.match(p.name)
    if m_alt:
        return f"{m_alt.group(1)}_{m_alt.group(2)}"
    nums = re.findall(r"(\\d+)", p.stem)
    if len(nums) >= 2:
        return f"{nums[-2]}_{nums[-1]}"
    return p.stem


def collect_images(input_dir: Path):
    files = [
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES
    ]
    files.sort(key=sort_key)
    return files


def _parse_frame_timestamp(p: Path) -> tuple[int, int] | None:
    m = TIMESTAMP_PATTERN.match(p.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    m_alt = ALT_TIMESTAMP_PATTERN.match(p.name)
    if m_alt:
        return int(m_alt.group(1)), int(m_alt.group(2))
    nums = re.findall(r"(\\d+)", p.stem)
    if len(nums) >= 2:
        return int(nums[-2]), int(nums[-1])
    return None


def _to_ns(ts: tuple[int, int]) -> int:
    sec, nsec = ts
    if nsec >= 1_000_000_000:
        sec += nsec // 1_000_000_000
        nsec = nsec % 1_000_000_000
    return sec * 1_000_000_000 + nsec


def collect_stereo_pairs(
    left_dir: Path,
    right_dir: Path,
    max_frames: int,
    sync_tolerance_ms: float,
    sync_phase_tolerance_ms: float = STEREO_SYNC_PHASE_TOLERANCE_MS,
    sync_phase_window: int = STEREO_SYNC_PHASE_WINDOW,
    sync_phase_min_samples: int = STEREO_SYNC_PHASE_MIN_SAMPLES,
    audit: dict[str, Any] | None = None,
):
    left_files = collect_images(left_dir)
    right_files = collect_images(right_dir)
    if not left_files:
        raise RuntimeError(f"No usable images in left dir: {left_dir}")
    if not right_files:
        raise RuntimeError(f"No usable images in right dir: {right_dir}")

    left_candidates = [(idx, p, _parse_frame_timestamp(p)) for idx, p in enumerate(left_files)]
    right_candidates = [(idx, p, _parse_frame_timestamp(p)) for idx, p in enumerate(right_files)]

    left_with_ts = [(i, p, ts) for i, p, ts in left_candidates if ts is not None]
    right_with_ts = [(i, p, ts) for i, p, ts in right_candidates if ts is not None]

    if not left_with_ts or not right_with_ts:
        raise RuntimeError("No timestamped image filenames found for stereo sync.")

    left_with_ts.sort(key=lambda item: (_to_ns(item[2]), item[0]))
    right_with_ts.sort(key=lambda item: (_to_ns(item[2]), item[0]))

    right_ns = np.array([_to_ns(ts) for _, _, ts in right_with_ts], dtype=np.int64)
    right_used = np.zeros(len(right_with_ts), dtype=bool)
    pairs: list[StereoFramePair] = []
    tol_ns = int(max(0.0, sync_tolerance_ms) * 1_000_000)
    phase_tol_ns = int(max(0.0, sync_phase_tolerance_ms) * 1_000_000)
    phase_history_ns: list[int] = []
    phase_rejected_left = 0
    no_candidate_left = 0

    for _, left_path, left_ts in left_with_ts:
        left_ns = _to_ns(left_ts)
        window_start = int(np.searchsorted(right_ns, left_ns - tol_ns, side="left"))
        window_end = int(np.searchsorted(right_ns, left_ns + tol_ns, side="right"))
        best_idx = -1
        best_diff = None
        time_candidates = 0
        phase_center_ns = None
        if phase_tol_ns > 0 and len(phase_history_ns) >= sync_phase_min_samples:
            phase_center_ns = int(np.median(np.asarray(phase_history_ns, dtype=np.int64)))

        for cand in range(window_start, window_end):
            if right_used[cand]:
                continue
            signed_diff = int(right_ns[cand] - left_ns)
            diff = abs(signed_diff)
            if diff > tol_ns:
                continue
            time_candidates += 1
            if phase_center_ns is not None and abs(signed_diff - phase_center_ns) > phase_tol_ns:
                continue
            if best_diff is None or diff < best_diff:
                best_idx = cand
                best_diff = diff

        if best_idx >= 0:
            right_used[best_idx] = True
            signed_delta_ns = int(right_ns[best_idx] - left_ns)
            phase_history_ns.append(signed_delta_ns)
            if len(phase_history_ns) > sync_phase_window:
                del phase_history_ns[:-sync_phase_window]
            pairs.append(
                StereoFramePair(
                    left_path=left_path,
                    right_path=right_with_ts[best_idx][1],
                    left_timestamp_ns=left_ns,
                    right_timestamp_ns=int(right_ns[best_idx]),
                )
            )
        elif time_candidates > 0 and phase_center_ns is not None:
            phase_rejected_left += 1
        else:
            no_candidate_left += 1

    if not pairs:
        raise RuntimeError(
            "No synchronized left/right pairs matched by timestamp. "
            "Check filename timestamp format and --sync-tolerance-ms."
        )

    if max_frames > 0:
        pairs = pairs[:max_frames]

    if not pairs:
        raise RuntimeError("Synchronized frame pairing is empty after max-frame truncation.")

    if audit is not None:
        paired_deltas_ms = np.asarray(
            [pair.stereo_delta_ns / 1_000_000.0 for pair in pairs], dtype=np.float64
        )
        audit.update({
            "left_timestamped": len(left_with_ts),
            "right_timestamped": len(right_with_ts),
            "paired": len(pairs),
            "phase_rejected_left": phase_rejected_left,
            "no_candidate_left": no_candidate_left,
            "sync_tolerance_ms": float(sync_tolerance_ms),
            "phase_tolerance_ms": float(sync_phase_tolerance_ms),
            "phase_window": int(sync_phase_window),
            "phase_min_samples": int(sync_phase_min_samples),
            "delta_min_ms": float(np.min(paired_deltas_ms)),
            "delta_median_ms": float(np.median(paired_deltas_ms)),
            "delta_p95_ms": float(np.percentile(paired_deltas_ms, 95)),
            "delta_max_ms": float(np.max(paired_deltas_ms)),
        })

    return pairs


def color_for_pair(index: int) -> tuple[int, int, int]:
    hue = (37 * index + 30) % 180
    hsv = np.uint8([[[hue, 240, 240]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def stereo_pair_label(class_id: int, pair_id: int) -> str:
    # Keep pair labels human-readable and stable while staying numeric.
    return f"D-{CLASS_PREFIX.get(class_id, f'C{class_id}')}-{int(pair_id) + 1}"


def make_rule_state(classes: list[int]) -> OnlineFrameRuleState:
    states: dict[int, Any] = {}
    summaries: dict[int, Any] = {}
    lost_display_pool: dict[int, Any] = {}
    consumed_lost_source_ids: set[int] = set()
    reentry_events: list[dict] = []
    next_display_ids = {class_id: 1 for class_id in classes}
    aggregate_tracklets: dict[tuple[int, int], Any] = {}
    stats = {
        "raw_rows": 0,
        "raw_detection_rows": 0,
        "online_rows": 0,
        "shown_rows": 0,
        "pending_rows": 0,
        "provisional_shown_rows": 0,
        "expired_track_count": 0,
        "reentry_inheritance_count": 0,
        "reentry_rejected_ambiguous_count": 0,
        "motion_conflict_rollback_count": 0,
        "lost_pool_expired_count": 0,
    }
    for class_id in classes:
        stats[f"raw_rows_class_{class_id}"] = 0
        stats[f"shown_rows_class_{class_id}"] = 0
        stats[f"pending_rows_class_{class_id}"] = 0
        stats[f"provisional_shown_rows_class_{class_id}"] = 0

    return OnlineFrameRuleState(
        tracks=states,
        summaries=summaries,
        lost_display_pool=lost_display_pool,
        consumed_lost_source_ids=consumed_lost_source_ids,
        reentry_events=reentry_events,
        next_display_ids=next_display_ids,
        aggregate_tracklets=aggregate_tracklets,
        stats=stats,
    )


def _load_float_list(value: list[float] | dict[str, float], default: float = 0.0) -> np.ndarray:
    if isinstance(value, dict):
        keys = ["k1", "k2", "p1", "p2", "k3"]
        return np.array([float(value.get("k1", default)),
                         float(value.get("k2", default)),
                         float(value.get("p1", default)),
                         float(value.get("p2", default)),
                         float(value.get("k3", default))], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64)
    if arr.size >= 4:
        arr = arr.astype(np.float64).reshape(-1)
        return np.array([arr[0], arr[1], arr[2], arr[3], arr[4] if arr.size > 4 else 0.0], dtype=np.float64)
    raise ValueError("distortion coeff list must have at least 4 elements")


def _load_matrix_3x3(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {arr.shape}")
    return arr


def _read_stereo_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Stereo param file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    if not isinstance(params, dict):
        raise ValueError(f"Invalid params yaml: {path}")
    if "calibration" in params and isinstance(params["calibration"], dict):
        params = params["calibration"]
    return params


def build_stereo_geometry(
    params: dict[str, Any],
    frame_size: tuple[int, int],
) -> StereoCameraGeometry:
    intr = params.get("intrinsic", params)
    left = intr.get("left_thermal", params.get("left_thermal"))
    right = intr.get("right_thermal", params.get("right_thermal"))
    if left is None or right is None:
        raise KeyError("Missing left_thermal/right_thermal intrinsic blocks.")

    K_left = np.array([
        [float(left["fx"]), 0.0, float(left["cx"])],
        [0.0, float(left["fy"]), float(left["cy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    K_right = np.array([
        [float(right["fx"]), 0.0, float(right["cx"])],
        [0.0, float(right["fy"]), float(right["cy"])],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    dist_left = _load_float_list(left.get("dist", left))
    dist_right = _load_float_list(right.get("dist", right))

    extr = params.get("extrinsic_left_to_right", params.get("extrinsic", params))
    T_candidates = None
    R_candidates = None
    if "R_left_to_right" in extr:
        R_candidates = extr["R_left_to_right"]
    elif "R" in extr:
        R_candidates = extr["R"]

    if "T_left_to_right_cm" in extr:
        T_candidates = extr["T_left_to_right_cm"]
    elif "T_left_to_right_m" in extr:
        T_candidates = [v * 100.0 for v in extr["T_left_to_right_m"]]
        T_candidates = [float(v) for v in T_candidates]
    elif "T_m" in extr:
        T_candidates = [v * 100.0 for v in extr["T_m"]]
        T_candidates = [float(v) for v in T_candidates]
    elif "T" in extr:
        T_candidates = extr["T"]

    if R_candidates is None:
        raise KeyError("Missing stereo rotation block (R or R_left_to_right).")
    if T_candidates is None:
        raise KeyError("Missing stereo translation block (T/T_left_to_right_*).")

    R = _load_matrix_3x3(R_candidates)
    T_cm = np.array(T_candidates, dtype=np.float64).reshape(3)
    if np.linalg.norm(T_cm) < 1e-9:
        raise ValueError("Stereo translation vector is zero.")
    T_m = T_cm / 100.0

    w, h = frame_size
    image_size = (int(w), int(h))
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K_left,
        dist_left,
        K_right,
        dist_right,
        image_size,
        R,
        T_m.reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map_left_x, map_left_y = cv2.initUndistortRectifyMap(
        K_left, dist_left, R1, P1, image_size, cv2.CV_16SC2
    )
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(
        K_right, dist_right, R2, P2, image_size, cv2.CV_16SC2
    )

    baseline_m = float(np.linalg.norm(T_m))
    return StereoCameraGeometry(
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
        K_left=K_left,
        K_right=K_right,
        dist_left=dist_left,
        dist_right=dist_right,
        T_m=T_m,
        size=image_size,
        baseline_m=baseline_m,
    )


def rectify_point(
    point: tuple[float, float],
    K: np.ndarray,
    D: np.ndarray,
    R: np.ndarray,
    P: np.ndarray,
) -> tuple[float, float]:
    arr = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    out = cv2.undistortPoints(arr, K, D, R=R, P=P)
    return float(out[0, 0, 0]), float(out[0, 0, 1])


def estimate_depth_from_disparity(
    left_rect: tuple[float, float],
    right_rect: tuple[float, float],
    geometry: StereoCameraGeometry,
    max_disparity_px: float,
    min_disparity_px: float,
) -> tuple[float | None, bool, float, float, str]:
    xL, yL = left_rect
    xR, yR = right_rect
    disparity = xL - xR
    y_diff = abs(yL - yR)
    if not math.isfinite(disparity) or not math.isfinite(y_diff):
        return None, False, float("nan"), float("inf"), "geom_disparity_nonfinite"
    if disparity <= 0.0 or disparity < min_disparity_px or disparity > max_disparity_px:
        if disparity <= 0.0:
            return None, False, y_diff, float(disparity), "geom_disparity_nonpositive"
        if disparity < min_disparity_px:
            return None, False, y_diff, float(disparity), "geom_disparity_below_min"
        return None, False, y_diff, float(disparity), "geom_disparity_above_max"
    fx = float(geometry.P1[0, 0])
    z = (fx * float(geometry.baseline_m)) / disparity
    if not math.isfinite(z) or z <= 0:
        return None, False, y_diff, float(disparity), "geom_depth_invalid"
    return z, True, y_diff, float(disparity), "ok"


def smooth_distance_with_ema(
    raw_depth_m: float | None,
    pair_state: StereoPairState,
    ema_alpha: float,
    max_change_ratio: float,
    min_abs_change: float,
    reset_after_invalid: int,
 ) -> tuple[float | None, bool, bool, str]:
    reset_after = max(1, reset_after_invalid)
    if raw_depth_m is None or not math.isfinite(raw_depth_m) or raw_depth_m <= 0:
        pair_state.ema_streak_invalid += 1
        return None, False, False, "ema_input_invalid"

    raw = float(raw_depth_m)
    if pair_state.ema_distance is None or pair_state.ema_count <= 0:
        pair_state.ema_distance = raw
        pair_state.ema_count = 1
        pair_state.ema_streak_invalid = 0
        return raw, True, False, "ok"

    prev = float(pair_state.ema_distance)
    threshold = max(min_abs_change, abs(prev) * max_change_ratio)
    if abs(raw - prev) > threshold:
        pair_state.ema_streak_invalid += 1
        return None, False, True, "ema_jump"

    ema_distance = (1.0 - ema_alpha) * prev + ema_alpha * raw
    if pair_state.ema_streak_invalid >= reset_after:
        pair_state.ema_recoveries += 1
    pair_state.ema_distance = ema_distance
    pair_state.ema_count += 1
    pair_state.ema_streak_invalid = 0
    return ema_distance, True, False, "ok"


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _stable_disparity_candidates(
    values: list[float], stability_ratio: float, stability_abs_px: float
) -> bool:
    center = _median(values)
    limit = max(stability_abs_px, abs(center) * stability_ratio)
    return max(abs(value - center) for value in values) <= limit


def smooth_disparity_with_reacquire(
    raw_depth_m: float | None,
    raw_disparity_px: float | None,
    pair_state: StereoPairState,
    frame_idx: int,
    raw_id_changed: bool,
    *,
    alpha: float,
    jump_ratio: float,
    jump_abs_px: float,
    reacquire_samples: int,
    stability_ratio: float,
    stability_abs_px: float,
    max_frame_gap: int,
) -> tuple[float | None, bool, bool, str, float | None, int, bool]:
    if (
        raw_depth_m is None
        or raw_disparity_px is None
        or not math.isfinite(raw_depth_m)
        or not math.isfinite(raw_disparity_px)
        or raw_depth_m <= 0.0
        or raw_disparity_px <= 0.0
    ):
        pair_state.disparity_candidates.clear()
        pair_state.disparity_candidate_last_frame = None
        return None, False, False, "disparity_input_invalid", None, 0, False

    disparity = float(raw_disparity_px)
    depth_scale = float(raw_depth_m) * disparity
    if pair_state.disparity_ema is None:
        pair_state.disparity_ema = disparity
        return raw_depth_m, True, False, "ok", disparity, 0, False

    jump_limit = max(jump_abs_px, abs(pair_state.disparity_ema) * jump_ratio)
    within_old_level = abs(disparity - pair_state.disparity_ema) <= jump_limit
    if raw_id_changed and not within_old_level:
        pair_state.disparity_candidates.clear()
        pair_state.disparity_candidate_last_frame = None
        pair_state.disparity_forced_reacquire = True

    if (
        pair_state.disparity_candidates
        and not pair_state.disparity_forced_reacquire
        and within_old_level
    ):
        pair_state.disparity_candidates.clear()
        pair_state.disparity_candidate_last_frame = None

    if pair_state.disparity_forced_reacquire or not within_old_level:
        if (
            pair_state.disparity_candidate_last_frame is None
            or frame_idx - pair_state.disparity_candidate_last_frame > max_frame_gap
        ):
            pair_state.disparity_candidates = [disparity]
        else:
            proposed = pair_state.disparity_candidates + [disparity]
            if _stable_disparity_candidates(proposed, stability_ratio, stability_abs_px):
                pair_state.disparity_candidates = proposed
            else:
                pair_state.disparity_candidates = [disparity]
        pair_state.disparity_candidate_last_frame = frame_idx
        candidate_count = len(pair_state.disparity_candidates)
        if candidate_count >= reacquire_samples:
            pair_state.disparity_ema = _median(
                pair_state.disparity_candidates[-reacquire_samples:]
            )
            pair_state.disparity_candidates.clear()
            pair_state.disparity_candidate_last_frame = None
            pair_state.disparity_forced_reacquire = False
            pair_state.ema_recoveries += 1
            distance = depth_scale / pair_state.disparity_ema
            return (
                distance,
                True,
                False,
                "disparity_ema_reacquired",
                pair_state.disparity_ema,
                reacquire_samples,
                True,
            )
        reason = (
            "raw_id_reacquire_pending"
            if pair_state.disparity_forced_reacquire
            else "disparity_jump_pending"
        )
        return None, False, True, reason, None, candidate_count, False

    pair_state.disparity_ema = (
        (1.0 - alpha) * pair_state.disparity_ema + alpha * disparity
    )
    distance = depth_scale / pair_state.disparity_ema
    return distance, True, False, "ok", pair_state.disparity_ema, 0, False


def candidate_is_available(decision, use_provisional: bool) -> bool:
    if decision.draw_mode == "formal":
        return True
    return use_provisional and decision.draw_mode == "provisional"


def dedupe_decisions_by_track_id(decisions: list[Any]) -> list[Any]:
    selected: dict[tuple[int, int], Any] = {}
    for d in decisions:
        key = (int(d.class_id), int(d.state.track_id))
        conf = float(d.confidence)
        prev = selected.get(key)
        if prev is None or conf > float(prev.confidence):
            selected[key] = d
    return list(selected.values())


def decision_center(decision) -> tuple[float, float]:
    x1, y1, x2, y2 = decision.raw_box
    return (float(x1 + x2) * 0.5, float(y1 + y2) * 0.5)


def box_area(decision) -> float:
    x1, y1, x2, y2 = decision.raw_box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def solve_lexicographic_assignment(
    feasible_cost: np.ndarray,
    cost_limit: float,
) -> list[tuple[int, int, float]]:
    """Maximise feasible cardinality, then minimise its non-negative cost."""
    if feasible_cost.ndim != 2 or 0 in feasible_cost.shape:
        return []

    cardinality_bound = min(feasible_cost.shape)
    infeasible_cost = (cardinality_bound + 1) * max(0.0, cost_limit) + 1.0
    assignment_cost = np.where(
        np.isfinite(feasible_cost), feasible_cost, infeasible_cost
    )
    rows, cols = linear_sum_assignment(assignment_cost)
    pairs: list[tuple[int, int, float]] = []
    for row, col in zip(rows, cols):
        value = float(feasible_cost[row, col])
        if math.isfinite(value):
            pairs.append((int(row), int(col), value))
    return pairs


def match_frame_candidates(
    left_candidates: list[Any],
    right_candidates: list[Any],
    y_gap_px: float,
    cost_limit: float,
) -> list[tuple[int, int, float]]:
    if not left_candidates or not right_candidates:
        return []

    n_left = len(left_candidates)
    n_right = len(right_candidates)
    feasible_cost = np.full((n_left, n_right), np.inf, dtype=np.float64)

    for i, left in enumerate(left_candidates):
        lx, ly = decision_center(left)
        l_area = box_area(left)
        for j, right in enumerate(right_candidates):
            rx, ry = decision_center(right)
            y_diff = abs(ly - ry)
            if y_diff > y_gap_px:
                continue
            center_dist = math.hypot(lx - rx, ly - ry)
            r_area = box_area(right)
            area_ratio = (l_area + 1e-6) / (r_area + 1e-6)
            area_cost = MATCH_AREA_RATIO_WEIGHT * abs(math.log(area_ratio))
            total = center_dist + area_cost
            if total <= cost_limit:
                feasible_cost[i, j] = total

    return solve_lexicographic_assignment(feasible_cost, cost_limit)


def rectified_from_decision(
    decision,
    geometry: StereoCameraGeometry,
    side: str,
) -> tuple[float, float]:
    cx, cy = decision_center(decision)
    if side == "left":
        return rectify_point(
            (cx, cy),
            geometry.K_left,
            geometry.dist_left,
            geometry.R1,
            geometry.P1,
        )
    return rectify_point(
        (cx, cy),
        geometry.K_right,
        geometry.dist_right,
        geometry.R2,
        geometry.P2,
    )


def rectified_from_observation(
    observation: FrameTrackObservation,
    geometry: StereoCameraGeometry,
    side: str,
) -> tuple[float, float]:
    x1, y1, x2, y2 = observation.raw_box
    point = ((float(x1) + float(x2)) * 0.5, (float(y1) + float(y2)) * 0.5)
    if side == "left":
        return rectify_point(point, geometry.K_left, geometry.dist_left, geometry.R1, geometry.P1)
    return rectify_point(point, geometry.K_right, geometry.dist_right, geometry.R2, geometry.P2)


def update_rectified_motion_history(
    histories: dict[tuple[int, int], RectifiedMotionState],
    observations: list[FrameTrackObservation],
    timestamp_ns: int,
    frame_idx: int,
    geometry: StereoCameraGeometry,
    side: str,
    history_size: int,
) -> None:
    for observation in observations:
        key = (int(observation.class_id), int(observation.raw_track_id))
        state = histories.setdefault(key, RectifiedMotionState())
        x, y = rectified_from_observation(observation, geometry, side)
        sample = RectifiedMotionSample(int(timestamp_ns), float(x), float(y))
        if state.samples and timestamp_ns < state.samples[-1].timestamp_ns:
            # A non-monotonic stream cannot be causal relative to the stored state.
            state.samples = [sample]
        elif state.samples and timestamp_ns == state.samples[-1].timestamp_ns:
            state.samples[-1] = sample
        else:
            state.samples.append(sample)
        if len(state.samples) > history_size:
            del state.samples[:-history_size]
        state.last_seen_frame = frame_idx


def fit_causal_rectified_velocity(
    state: RectifiedMotionState | None,
    current_timestamp_ns: int,
    min_samples: int,
    decay: float,
    max_gap_ms: float,
) -> tuple[MotionEstimate | None, str]:
    if state is None or len(state.samples) < min_samples:
        return None, "temporal_insufficient_history"
    samples = state.samples
    if samples[-1].timestamp_ns != current_timestamp_ns:
        return None, "temporal_current_sample_missing"
    times_ns = np.asarray([sample.timestamp_ns for sample in samples], dtype=np.int64)
    gaps_ms = np.diff(times_ns.astype(np.float64)) / 1_000_000.0
    if gaps_ms.size == 0 or np.any(gaps_ms <= 0.0):
        return None, "temporal_nonmonotonic_history"
    if np.any(gaps_ms > max_gap_ms):
        return None, "temporal_stale_history"

    times_s = (times_ns.astype(np.float64) - float(times_ns[-1])) / 1_000_000_000.0
    span_ms = float(-times_s[0] * 1000.0)
    if span_ms <= 0.0:
        return None, "temporal_zero_history_span"
    ages = np.arange(len(samples) - 1, -1, -1, dtype=np.float64)
    weights = np.power(float(decay), ages)
    design = np.column_stack([np.ones(len(samples), dtype=np.float64), times_s])
    weighted_design = design * np.sqrt(weights)[:, None]
    xs = np.asarray([sample.x for sample in samples], dtype=np.float64) * np.sqrt(weights)
    ys = np.asarray([sample.y for sample in samples], dtype=np.float64) * np.sqrt(weights)
    beta_x, _, rank_x, _ = np.linalg.lstsq(weighted_design, xs, rcond=None)
    beta_y, _, rank_y, _ = np.linalg.lstsq(weighted_design, ys, rcond=None)
    if rank_x < 2 or rank_y < 2:
        return None, "temporal_velocity_fit_singular"
    vx, vy = float(beta_x[1]), float(beta_y[1])
    if not math.isfinite(vx) or not math.isfinite(vy):
        return None, "temporal_velocity_nonfinite"
    return MotionEstimate(vx, vy, len(samples), span_ms), "ok"


def compensate_rectified_centers(
    frame_pair: StereoFramePair,
    left_raw: tuple[float, float],
    right_raw: tuple[float, float],
    left_state: RectifiedMotionState | None,
    right_state: RectifiedMotionState | None,
    enabled: bool,
    min_samples: int,
    decay: float,
    max_gap_ms: float,
    max_shift_px: float,
    bound_margin_px: float,
    frame_size: tuple[int, int],
    min_disparity_px: float,
    max_disparity_px: float,
) -> TemporalCompensationResult:
    raw_disparity = float(left_raw[0] - right_raw[0])
    raw_y_diff = float(abs(left_raw[1] - right_raw[1]))
    common_ns = frame_pair.common_timestamp_ns

    def result(
        valid: bool,
        applied: bool,
        reason: str,
        left_comp: tuple[float, float] | None = None,
        right_comp: tuple[float, float] | None = None,
        left_motion: MotionEstimate | None = None,
        right_motion: MotionEstimate | None = None,
    ) -> TemporalCompensationResult:
        left_shift = None if left_comp is None else math.hypot(left_comp[0] - left_raw[0], left_comp[1] - left_raw[1])
        right_shift = None if right_comp is None else math.hypot(right_comp[0] - right_raw[0], right_comp[1] - right_raw[1])
        disparity_comp = None if left_comp is None or right_comp is None else float(left_comp[0] - right_comp[0])
        y_diff_comp = None if left_comp is None or right_comp is None else float(abs(left_comp[1] - right_comp[1]))
        return TemporalCompensationResult(
            enabled=enabled,
            valid=valid,
            applied=applied,
            reason=reason,
            common_timestamp_ns=common_ns,
            left_raw=left_raw,
            right_raw=right_raw,
            left_compensated=left_comp,
            right_compensated=right_comp,
            left_motion=left_motion,
            right_motion=right_motion,
            left_shift_px=left_shift,
            right_shift_px=right_shift,
            disparity_raw_px=raw_disparity,
            disparity_compensated_px=disparity_comp,
            y_diff_raw_px=raw_y_diff,
            y_diff_compensated_px=y_diff_comp,
        )

    if not enabled:
        return result(True, False, "disabled")

    left_motion, left_reason = fit_causal_rectified_velocity(
        left_state, frame_pair.left_timestamp_ns, min_samples, decay, max_gap_ms
    )
    if left_motion is None:
        return result(False, False, f"left_{left_reason}")
    right_motion, right_reason = fit_causal_rectified_velocity(
        right_state, frame_pair.right_timestamp_ns, min_samples, decay, max_gap_ms
    )
    if right_motion is None:
        return result(False, False, f"right_{right_reason}", left_motion=left_motion)

    left_dt_s = (common_ns - frame_pair.left_timestamp_ns) / 1_000_000_000.0
    right_dt_s = (common_ns - frame_pair.right_timestamp_ns) / 1_000_000_000.0
    left_comp = (
        left_raw[0] + left_motion.vx_px_s * left_dt_s,
        left_raw[1] + left_motion.vy_px_s * left_dt_s,
    )
    right_comp = (
        right_raw[0] + right_motion.vx_px_s * right_dt_s,
        right_raw[1] + right_motion.vy_px_s * right_dt_s,
    )
    values = (*left_comp, *right_comp)
    if not all(math.isfinite(value) for value in values):
        return result(False, False, "temporal_prediction_nonfinite", left_comp, right_comp, left_motion, right_motion)
    left_shift = math.hypot(left_comp[0] - left_raw[0], left_comp[1] - left_raw[1])
    right_shift = math.hypot(right_comp[0] - right_raw[0], right_comp[1] - right_raw[1])
    if left_shift > max_shift_px:
        return result(False, False, "temporal_left_shift_exceeded", left_comp, right_comp, left_motion, right_motion)
    if right_shift > max_shift_px:
        return result(False, False, "temporal_right_shift_exceeded", left_comp, right_comp, left_motion, right_motion)
    width, height = frame_size
    for side, point in (("left", left_comp), ("right", right_comp)):
        if not (-bound_margin_px <= point[0] <= width - 1 + bound_margin_px and -bound_margin_px <= point[1] <= height - 1 + bound_margin_px):
            return result(False, False, f"temporal_{side}_out_of_bounds", left_comp, right_comp, left_motion, right_motion)
    compensated_disparity = left_comp[0] - right_comp[0]
    if compensated_disparity <= 0.0:
        return result(False, False, "temporal_disparity_nonpositive", left_comp, right_comp, left_motion, right_motion)
    if compensated_disparity < min_disparity_px:
        return result(False, False, "temporal_disparity_below_min", left_comp, right_comp, left_motion, right_motion)
    if compensated_disparity > max_disparity_px:
        return result(False, False, "temporal_disparity_above_max", left_comp, right_comp, left_motion, right_motion)
    return result(True, True, "ok", left_comp, right_comp, left_motion, right_motion)


def write_report(
    path: Path,
    output_name: str,
    processed: int,
    skipped: int,
    pair_stats: dict[str, Any],
    match_y_gap: float,
    match_cost_limit: float,
    match_max_disparity: float,
    match_min_disparity: float,
    pair_ttl: int,
    sync_config: dict[str, Any],
    temporal_config: dict[str, Any],
    distance_filter_config: dict[str, Any],
):
    total_pairs = pair_stats["total_pairs"]
    valid_pairs = pair_stats["valid_pairs"]
    invalid_pairs = pair_stats["invalid_pairs"]
    ema_invalid_pairs = pair_stats.get("ema_invalid_pairs", 0)
    ema_recovery_count = pair_stats.get("ema_recovery_count", 0)
    filter_reason_counts = pair_stats.get("filter_reason_counts", {})
    temporal_reason_counts = pair_stats.get("temporal_reason_counts", {})
    temporal_attempts = pair_stats.get("temporal_attempts", 0)
    temporal_applied = pair_stats.get("temporal_applied", 0)
    temporal_deltas_ms = pair_stats.get("temporal_deltas_ms", [])
    temporal_left_shifts = pair_stats.get("temporal_left_shifts", [])
    temporal_right_shifts = pair_stats.get("temporal_right_shifts", [])
    temporal_disparity_changes = pair_stats.get("temporal_disparity_changes", [])
    temporal_raw_disparities = pair_stats.get("temporal_raw_disparities", [])
    temporal_comp_disparities = pair_stats.get("temporal_comp_disparities", [])
    raw_geometry_valid = pair_stats.get("raw_geometry_valid", 0)
    compensated_geometry_valid = pair_stats.get("compensated_geometry_valid", 0)

    def distribution(values: list[float]) -> str:
        if not values:
            return "n/a"
        arr = np.asarray(values, dtype=np.float64)
        return (
            f"min={np.min(arr):.3f}, p50={np.percentile(arr, 50):.3f}, "
            f"p90={np.percentile(arr, 90):.3f}, p95={np.percentile(arr, 95):.3f}, max={np.max(arr):.3f}"
        )
    avg_filtered_depth = pair_stats["avg_filtered_depth_m"]
    min_filtered_depth = pair_stats["min_filtered_depth_m"]
    max_filtered_depth = pair_stats["max_filtered_depth_m"]
    if distance_filter_config["mode"] == "disparity_ema_reacquire":
        filter_config_lines = [
            f"- disparity_ema_alpha: {distance_filter_config['alpha']}",
            f"- disparity_jump_gate: max({distance_filter_config['jump_abs_px']} px, previous * {distance_filter_config['jump_ratio']})",
            f"- disparity_reacquire_samples: {distance_filter_config['reacquire_samples']}",
            f"- disparity_stability_gate: max({distance_filter_config['stability_abs_px']} px, median * {distance_filter_config['stability_ratio']})",
            f"- disparity_reacquire_max_frame_gap: {distance_filter_config['max_frame_gap']}",
        ]
    else:
        filter_config_lines = [
            f"- distance_ema_alpha: {distance_filter_config['legacy_alpha']}",
            f"- distance_ema_jump_gate: max({distance_filter_config['legacy_min_abs_change']} m, previous * {distance_filter_config['legacy_max_change']})",
            f"- distance_ema_reset_after_invalid: {distance_filter_config['legacy_reset_after_invalid']}",
            "- legacy_reset_semantics: recovery counter only; the metre-domain EMA baseline is not rebuilt",
        ]

    lines = [
        "# Stereo Tracking Distance Report",
        "",
        "## Summary",
        f"- output_name: `{output_name}`",
        f"- processed_frames: {processed}",
        f"- skipped_frames: {skipped}",
        f"- total_matched_pairs: {total_pairs}",
        f"- valid_distance_pairs: {valid_pairs}",
        f"- invalid_distance_pairs: {invalid_pairs}",
        f"- distance_filter_invalid_pairs: {ema_invalid_pairs}",
        f"- filter_recovery_or_reacquisition_events: {ema_recovery_count}",
        f"- min_filtered_depth_m: {min_filtered_depth:.3f}",
        f"- max_filtered_depth_m: {max_filtered_depth:.3f}",
        f"- avg_filtered_depth_m: {avg_filtered_depth:.3f}",
        "",
        "## Matching/Distance Config",
        f"- match_y_gap_px: {match_y_gap}",
        f"- match_cost_limit: {match_cost_limit}",
        f"- max_disparity_px: {match_max_disparity}",
        f"- min_disparity_px: {match_min_disparity}",
        f"- pair_ttl_frames: {pair_ttl}",
        f"- distance_filter_mode: `{distance_filter_config['mode']}`",
        *filter_config_lines,
        "",
        "## Frame Synchronization",
        f"- tolerance_ms: {sync_config['sync_tolerance_ms']}",
        f"- phase_tolerance_ms: {sync_config['phase_tolerance_ms']}",
        f"- phase_window: {sync_config['phase_window']}",
        f"- phase_min_samples: {sync_config['phase_min_samples']}",
        f"- timestamped_left/right: {sync_config['left_timestamped']} / {sync_config['right_timestamped']}",
        f"- synchronized_pairs: {sync_config['paired']}",
        f"- phase_rejected_left: {sync_config['phase_rejected_left']}",
        f"- no_candidate_left: {sync_config['no_candidate_left']}",
        f"- signed_delta_ms_min/median/p95/max: {sync_config['delta_min_ms']:.3f} / {sync_config['delta_median_ms']:.3f} / {sync_config['delta_p95_ms']:.3f} / {sync_config['delta_max_ms']:.3f}",
        "",
        "## Causal Time Compensation",
        f"- enabled: {temporal_config['enabled']}",
        f"- invalid_policy: `{temporal_config['invalid_policy']}`",
        f"- history_size: {temporal_config['history_size']}",
        f"- min_samples: {temporal_config['min_samples']}",
        f"- decay: {temporal_config['decay']}",
        f"- max_gap_ms: {temporal_config['max_gap_ms']}",
        f"- max_shift_px: {temporal_config['max_shift_px']}",
        f"- bound_margin_px: {temporal_config['bound_margin_px']}",
        f"- attempts: {temporal_attempts}",
        f"- applied: {temporal_applied}",
        f"- apply_rate: {(temporal_applied / temporal_attempts * 100.0) if temporal_attempts else 0.0:.2f}%",
        f"- signed_stereo_delta_ms: {distribution(temporal_deltas_ms)}",
        f"- left_shift_px: {distribution(temporal_left_shifts)}",
        f"- right_shift_px: {distribution(temporal_right_shifts)}",
        f"- raw_disparity_px: {distribution(temporal_raw_disparities)}",
        f"- compensated_disparity_px: {distribution(temporal_comp_disparities)}",
        f"- compensated_minus_raw_disparity_px: {distribution(temporal_disparity_changes)}",
        f"- raw_geometry_valid_rate: {(raw_geometry_valid / total_pairs * 100.0) if total_pairs else 0.0:.2f}%",
        f"- compensated_geometry_valid_rate: {(compensated_geometry_valid / temporal_attempts * 100.0) if temporal_attempts else 0.0:.2f}%",
        f"- final_distance_filter_pass_rate: {(valid_pairs / total_pairs * 100.0) if total_pairs else 0.0:.2f}%",
        "",
        "### Temporal Reason Breakdown",
    ]
    if not temporal_reason_counts:
        lines.append("- no_temporal_rows: 0")
    else:
        for reason in sorted(temporal_reason_counts):
            lines.append(f"- {reason}: {temporal_reason_counts[reason]}")
    lines.extend([
        "",
        "## Distance Filter Reason Breakdown",
    ])
    if total_pairs <= 0:
        lines.append("- no_pairs: 0")
    else:
        for reason in sorted(filter_reason_counts):
            reason_count = filter_reason_counts[reason]
            lines.append(
                f"- {reason}: {reason_count} ({reason_count / total_pairs * 100:.2f}%)"
            )
    lines.extend([
        "## Notes",
        "- Left/right cameras maintain independent rule states and do not share `OnlineFrameRuleState`.",
        "- Per-frame matching groups are done by class using Hungarian assignment on formal (and optional provisional) detections.",
        "- Final stereo IDs are pair-level `D-{D/P}-{N}` tags mapped to matched pairs.",
        "- `depth_m` is the selected raw/compensated geometric forward depth; `filtered_depth_m` is the output of the configured temporal filter.",
        "- In `distance_ema` mode, the event counter means legacy recovery after invalid samples and is not a baseline reset. In `disparity_ema_reacquire` mode, it counts completed three-sample controlled reacquisitions.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check_paths(
    model_path: Path,
    left_input: Path,
    right_input: Path,
    output_img_dir: Path,
    output_video_path: Path,
    output_csv: Path,
    output_report: Path,
    output_plot: Path | None,
):
    for p in (left_input, right_input):
        if not p.exists():
            raise FileNotFoundError(f"Input directory not found: {p}")
        if not p.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {p}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    output_paths = [output_img_dir, output_video_path, output_csv, output_report]
    if output_plot is not None:
        output_paths.append(output_plot)
    for p in output_paths:
        if p.exists():
            raise FileExistsError(f"Output path already exists: {p}")

    output_img_root = output_img_dir.parent
    output_video_dir = output_video_path.parent
    output_img_root.mkdir(parents=True, exist_ok=True)
    output_video_dir.mkdir(parents=True, exist_ok=True)
    output_img_dir.mkdir(parents=False, exist_ok=False)


def draw_bbox(
    canvas: np.ndarray,
    decision,
    color: tuple[int, int, int],
    line_label: str,
):
    x1, y1, x2, y2 = [int(round(v)) for v in decision.raw_box]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = max(0, x2)
    y2 = max(0, y2)
    h, w = canvas.shape[:2]
    x1 = min(x1, w - 1)
    x2 = min(max(x1 + 1, x2), w - 1)
    y1 = min(y1, h - 1)
    y2 = min(max(y1 + 1, y2), h - 1)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    label = line_label
    tw, th = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
    text_x = x1
    text_y = max(18, y1 - 4)
    cv2.rectangle(
        canvas,
        (text_x, text_y - th - 2),
        (text_x + tw + 6, text_y + 2),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        label,
        (text_x + 3, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_distance_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
):
    text = str(text)
    if not text:
        text = "invalid"
    tw, th = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
    x, y = origin
    h, w = canvas.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(th + 4, min(y, h - 1))
    bg_x1 = max(0, x)
    bg_y1 = y - th - 4
    bg_x2 = min(w - 1, x + tw + 6)
    bg_y2 = y + 2
    cv2.rectangle(
        canvas,
        (bg_x1, bg_y1),
        (bg_x2, bg_y2),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        text,
        (x + 3, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        (255, 255, 255),
        FONT_THICKNESS,
        cv2.LINE_AA,
    )


def run():
    args = parse_args()
    validate_output_name(args.output_name)
    validate_temporal_args(args)

    left_dir = Path(args.left_dir)
    right_dir = Path(args.right_dir)
    params_path = Path(args.params_path)
    model_path = Path(args.model).expanduser()
    output_img_root = Path(args.output_img_root)
    output_video_root = Path(args.output_video_root)
    output_name = args.output_name
    enable_reentry = args.enable_id_inheritance
    use_provisional = args.use_provisional

    output_img_dir = output_img_root / output_name
    output_video_dir = output_video_root / output_name
    output_video_path = output_video_dir / f"{output_name}.mp4"
    output_csv_path = output_video_dir / f"{output_name}_tracks_distance.csv"
    output_report_path = output_video_dir / f"{output_name}_distance_report.md"
    output_plot_path = output_video_dir / f"{output_name}_distance_scatter.png"

    sync_audit: dict[str, Any] = {}
    paired_frames = collect_stereo_pairs(
        left_dir,
        right_dir,
        args.max_frames,
        args.sync_tolerance_ms,
        args.sync_phase_tolerance_ms,
        args.sync_phase_window,
        args.sync_phase_min_samples,
        sync_audit,
    )
    if not paired_frames:
        raise RuntimeError("No synchronized frame pairs to process.")
    pairs_count = len(paired_frames)
    first_left = paired_frames[0].left_path

    first_frame = cv2.imread(str(first_left), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first left frame: {first_left}")
    if len(first_frame.shape) == 2:
        first_frame = cv2.cvtColor(first_frame, cv2.COLOR_GRAY2BGR)
    h, w = first_frame.shape[:2]

    params = _read_stereo_params(params_path)
    geometry = build_stereo_geometry(params, (w, h))

    tracker_path = args.tracker_path
    model_left = YOLO(str(model_path))
    model_right = YOLO(str(model_path))
    latency_enabled = bool(args.latency_report_path)
    core_latencies_ms: list[float] = []
    latency_frame_indices: list[int] = []
    stage_latencies_ms: dict[str, list[float]] = {
        "left_yolo_botsort": [],
        "right_yolo_botsort": [],
        "left_preprocess": [],
        "left_yolo_inference": [],
        "left_yolo_postprocess": [],
        "left_botsort_update": [],
        "left_botsort_gmc": [],
        "left_botsort_association_and_state": [],
        "left_ultralytics_tracking_overhead": [],
        "right_preprocess": [],
        "right_yolo_inference": [],
        "right_yolo_postprocess": [],
        "right_botsort_update": [],
        "right_botsort_gmc": [],
        "right_botsort_association_and_state": [],
        "right_ultralytics_tracking_overhead": [],
        "observation_extraction": [],
        "temporal_history": [],
        "online_v19_rules": [],
        "state_lifecycle": [],
        "stereo_match_depth_ema": [],
    }
    if latency_enabled and torch.cuda.is_available():
        torch.cuda.set_device(CUDA_DEVICE_INDEX)
        torch.cuda.reset_peak_memory_stats()

    check_paths(
        model_path,
        left_dir,
        right_dir,
        output_img_dir,
        output_video_path,
        output_csv_path,
        output_report_path,
        None if args.skip_distance_plot else output_plot_path,
    )

    print("=" * 70)
    print("Stereo tracking distance job started")
    print(f"OUTPUT_NAME      : {output_name}")
    print(f"LEFT_DIR         : {left_dir}")
    print(f"RIGHT_DIR        : {right_dir}")
    print(f"OUTPUT_IMG_DIR   : {output_img_dir}")
    print(f"OUTPUT_VIDEO_PATH: {output_video_path}")
    print(f"OUTPUT_CSV       : {output_csv_path}")
    print(f"OUTPUT_PLOT      : {'disabled' if args.skip_distance_plot else output_plot_path}")
    print(f"PARAMS_PATH      : {params_path}")
    print(f"TRACKER_PATH     : {tracker_path}")
    print(f"FPS              : {args.fps}")
    print(f"ENABLE_ID_INHERIT: {enable_reentry}")
    print(f"USE_PROVISIONAL  : {use_provisional}")
    print(f"MATCH_Y_GAP_PX   : {args.match_y_gap}")
    print(f"MATCH_COST_LIMIT : {args.match_cost_limit}")
    print(f"MATCH_MAX_DSP    : {args.match_max_disparity}")
    print(f"EMA_ALPHA       : {args.distance_ema_alpha}")
    print(f"EMA_MAX_CHANGE  : {args.distance_ema_max_change}")
    print(f"EMA_MIN_ABS_CHG : {args.distance_ema_min_abs_change}")
    print(f"MIN_DISPARITY   : {args.distance_min_disparity}")
    print(f"EMA_RESET_AFTER : {args.distance_ema_reset_after_invalid}")
    print(f"DIST_FILTER_MODE : {args.distance_filter_mode}")
    print(f"DSP_EMA_ALPHA    : {args.disparity_ema_alpha}")
    print(f"DSP_JUMP_GATE    : max({args.disparity_ema_jump_abs_px}px, {args.disparity_ema_jump_ratio:.2f}x)")
    print(f"DSP_REACQUIRE    : {args.disparity_reacquire_samples} samples")
    print(f"DSP_STABILITY    : max({args.disparity_reacquire_stability_abs_px}px, {args.disparity_reacquire_stability_ratio:.2f}x)")
    print(f"TEMPORAL_COMP    : {args.enable_causal_time_compensation}")
    print(f"TEMPORAL_HISTORY : {args.temporal_history_size}/{args.temporal_min_samples}")
    print(f"TEMPORAL_DECAY   : {args.temporal_decay}")
    print(f"TEMPORAL_MAX_GAP : {args.temporal_max_gap_ms} ms")
    print(f"TEMPORAL_MAX_MOVE: {args.temporal_max_shift_px} px")
    print(f"TEMPORAL_POLICY  : {args.temporal_invalid_policy}")
    print(f"MAX_FRAMES       : {args.max_frames if args.max_frames > 0 else 'full'}")
    print(f"TOTAL_FRAMES     : {pairs_count}")
    print("=" * 70)

    left_state = make_rule_state(CLASSES)
    right_state = make_rule_state(CLASSES)

    stereo_pair_states: dict[int, StereoPairState] = {}
    left_raw_to_pair: dict[tuple[int, int], int] = {}
    right_raw_to_pair: dict[tuple[int, int], int] = {}
    next_pair_id = 0
    left_motion_histories: dict[tuple[int, int], RectifiedMotionState] = {}
    right_motion_histories: dict[tuple[int, int], RectifiedMotionState] = {}

    pair_stats = {
        "total_pairs": 0,
        "valid_pairs": 0,
        "invalid_pairs": 0,
        "ema_invalid_pairs": 0,
        "ema_recovery_count": 0,
        "filtered_depth_values": [],
        "filter_reason_counts": {},
        "temporal_attempts": 0,
        "temporal_applied": 0,
        "temporal_reason_counts": {},
        "temporal_deltas_ms": [],
        "temporal_left_shifts": [],
        "temporal_right_shifts": [],
        "temporal_disparity_changes": [],
        "temporal_raw_disparities": [],
        "temporal_comp_disparities": [],
        "raw_geometry_valid": 0,
        "compensated_geometry_valid": 0,
    }

    writer = None
    csv_fh = output_csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDS)
    csv_writer.writeheader()

    processed = 0
    skipped = 0

    try:
        for frame_idx in range(pairs_count):
            frame_pair = paired_frames[frame_idx]
            left_path = frame_pair.left_path
            right_path = frame_pair.right_path

            left_raw = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
            right_raw = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
            if left_raw is None or right_raw is None:
                print(f"[WARN] Failed to read frame pair: {left_path.name} / {right_path.name}")
                skipped += 1
                continue
            if len(left_raw.shape) == 2:
                left_raw = cv2.cvtColor(left_raw, cv2.COLOR_GRAY2BGR)
            if len(right_raw.shape) == 2:
                right_raw = cv2.cvtColor(right_raw, cv2.COLOR_GRAY2BGR)

            lh, lw = left_raw.shape[:2]
            rh, rw = right_raw.shape[:2]
            if (lw, lh) != (rw, rh):
                right_raw = cv2.resize(right_raw, (lw, lh), interpolation=cv2.INTER_AREA)

            if writer is None:
                video_size = (lw * 2, lh)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_video_path), fourcc, args.fps, video_size)
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to create video writer: {output_video_path}")

            # Core latency starts after file decode/color conversion/resize and excludes
            # drawing, CSV/image/video writes, report generation, and plotting.
            if latency_enabled and torch.cuda.is_available():
                torch.cuda.synchronize()
            core_started_at = time.perf_counter() if latency_enabled else 0.0

            def stage_start() -> float:
                if not latency_enabled:
                    return 0.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                return time.perf_counter()

            def stage_finish(name: str, started_at: float) -> float:
                if not latency_enabled:
                    return 0.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed_ms = (time.perf_counter() - started_at) * 1000.0
                if frame_idx >= args.latency_warmup_frames:
                    stage_latencies_ms[name].append(elapsed_ms)
                return elapsed_ms

            def install_tracker_update_timer(model) -> None:
                """Time only BoT-SORT.update(), leaving Ultralytics callback I/O in residual."""
                predictor = getattr(model, "predictor", None)
                for tracker in getattr(predictor, "trackers", []):
                    if getattr(tracker, "_ceg_update_timer_installed", False):
                        continue
                    original_update = tracker.update

                    def timed_update(*update_args, _original=original_update, _tracker=tracker, **update_kwargs):
                        update_started_at = time.perf_counter()
                        try:
                            return _original(*update_args, **update_kwargs)
                        finally:
                            _tracker._ceg_last_update_ms = (
                                time.perf_counter() - update_started_at
                            ) * 1000.0

                    tracker.update = timed_update
                    tracker._ceg_last_update_ms = 0.0
                    tracker._ceg_update_timer_installed = True
                    gmc = getattr(tracker, "gmc", None)
                    if gmc is not None and not getattr(gmc, "_ceg_apply_timer_installed", False):
                        original_apply = gmc.apply

                        def timed_apply(*apply_args, _original=original_apply, _gmc=gmc, **apply_kwargs):
                            apply_started_at = time.perf_counter()
                            try:
                                return _original(*apply_args, **apply_kwargs)
                            finally:
                                _gmc._ceg_last_apply_ms = (
                                    time.perf_counter() - apply_started_at
                                ) * 1000.0

                        gmc.apply = timed_apply
                        gmc._ceg_last_apply_ms = 0.0
                        gmc._ceg_apply_timer_installed = True

            def reset_tracker_update_timer(model) -> None:
                predictor = getattr(model, "predictor", None)
                for tracker in getattr(predictor, "trackers", []):
                    tracker._ceg_last_update_ms = 0.0
                    gmc = getattr(tracker, "gmc", None)
                    if gmc is not None:
                        gmc._ceg_last_apply_ms = 0.0

            def tracker_update_ms(model) -> float:
                predictor = getattr(model, "predictor", None)
                return sum(
                    float(getattr(tracker, "_ceg_last_update_ms", 0.0))
                    for tracker in getattr(predictor, "trackers", [])
                )

            def tracker_gmc_ms(model) -> float:
                predictor = getattr(model, "predictor", None)
                return sum(
                    float(getattr(getattr(tracker, "gmc", None), "_ceg_last_apply_ms", 0.0))
                    for tracker in getattr(predictor, "trackers", [])
                )

            def record_track_breakdown(
                side: str, result, wall_ms: float, botsort_ms: float, gmc_ms: float
            ) -> None:
                if not latency_enabled or frame_idx < args.latency_warmup_frames:
                    return
                speed = getattr(result, "speed", {}) or {}
                preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
                inference_ms = float(speed.get("inference", 0.0) or 0.0)
                postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)
                accounted_ms = preprocess_ms + inference_ms + postprocess_ms + botsort_ms
                # Includes per-call predictor/dataloader setup, detection tensor CPU transfer,
                # result reindex/update, callbacks, and Python dispatch around model.track().
                overhead_ms = max(0.0, wall_ms - accounted_ms)
                stage_latencies_ms[f"{side}_preprocess"].append(preprocess_ms)
                stage_latencies_ms[f"{side}_yolo_inference"].append(inference_ms)
                stage_latencies_ms[f"{side}_yolo_postprocess"].append(postprocess_ms)
                stage_latencies_ms[f"{side}_botsort_update"].append(botsort_ms)
                stage_latencies_ms[f"{side}_botsort_gmc"].append(gmc_ms)
                stage_latencies_ms[f"{side}_botsort_association_and_state"].append(
                    max(0.0, botsort_ms - gmc_ms)
                )
                stage_latencies_ms[f"{side}_ultralytics_tracking_overhead"].append(overhead_ms)

            # Run tracking on each eye (independent state + model).
            left_kwargs = {
                "source": left_raw,
                "persist": True,
                "tracker": tracker_path,
                "conf": CONF,
                "iou": IOU,
                "classes": CLASSES,
                "device": DEVICE,
                "verbose": VERBOSE,
            }
            right_kwargs = dict(left_kwargs)
            right_kwargs["source"] = right_raw
            if IMGSZ is not None:
                left_kwargs["imgsz"] = IMGSZ
                right_kwargs["imgsz"] = IMGSZ

            stage_started_at = stage_start()
            reset_tracker_update_timer(model_left)
            left_result = model_left.track(**left_kwargs)[0]
            left_wall_ms = stage_finish("left_yolo_botsort", stage_started_at)
            install_tracker_update_timer(model_left)
            record_track_breakdown(
                "left", left_result, left_wall_ms,
                tracker_update_ms(model_left), tracker_gmc_ms(model_left),
            )

            stage_started_at = stage_start()
            reset_tracker_update_timer(model_right)
            right_result = model_right.track(**right_kwargs)[0]
            right_wall_ms = stage_finish("right_yolo_botsort", stage_started_at)
            install_tracker_update_timer(model_right)
            record_track_breakdown(
                "right", right_result, right_wall_ms,
                tracker_update_ms(model_right), tracker_gmc_ms(model_right),
            )

            left_obs: list[FrameTrackObservation] = []
            right_obs: list[FrameTrackObservation] = []

            def append_observations(result, image_name: str, out_obs: list[FrameTrackObservation]):
                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    return
                ids = boxes.id.detach().cpu().numpy().astype(int)
                classes = boxes.cls.detach().cpu().numpy().astype(int)
                confs = boxes.conf.detach().cpu().numpy()
                xyxy = boxes.xyxy.detach().cpu().numpy()
                for track_id, class_id, conf, box in zip(ids, classes, confs, xyxy):
                    track_id = int(track_id)
                    class_id = int(class_id)
                    conf = float(conf)
                    raw_box = [float(v) for v in box]
                    out_obs.append(
                        FrameTrackObservation(
                            raw_track_id=track_id,
                            class_id=class_id,
                            confidence=conf,
                            raw_box=raw_box,
                            image_name=image_name,
                        )
                    )

            stage_started_at = stage_start()
            append_observations(left_result, left_path.name, left_obs)
            append_observations(right_result, right_path.name, right_obs)
            stage_finish("observation_extraction", stage_started_at)

            # Histories use only observations available at each eye's current exposure.
            # Keep the disabled baseline free of the extra rectification/fitting work.
            stage_started_at = stage_start()
            if args.enable_causal_time_compensation:
                update_rectified_motion_history(
                    left_motion_histories, left_obs, frame_pair.left_timestamp_ns, frame_idx,
                    geometry, "left", args.temporal_history_size,
                )
                update_rectified_motion_history(
                    right_motion_histories, right_obs, frame_pair.right_timestamp_ns, frame_idx,
                    geometry, "right", args.temporal_history_size,
                )
            stage_finish("temporal_history", stage_started_at)

            stage_started_at = stage_start()
            output_left = process_tracking_frame(
                observations=left_obs,
                frame_idx=frame_idx,
                width=lw,
                height=lh,
                previous_state=left_state,
                enable_id_inheritance=enable_reentry,
            )
            left_state = output_left.next_state

            output_right = process_tracking_frame(
                observations=right_obs,
                frame_idx=frame_idx,
                width=lw,
                height=lh,
                previous_state=right_state,
                enable_id_inheritance=enable_reentry,
            )
            right_state = output_right.next_state
            stage_finish("online_v19_rules", stage_started_at)

            stage_started_at = stage_start()
            if enable_reentry:
                tracking_rules.add_disappeared_tracks_to_lost_pool(
                    left_state.tracks,
                    output_left.seen_raw_track_ids,
                    left_state.lost_display_pool,
                    left_state.consumed_lost_source_ids,
                    lw,
                    lh,
                )
                tracking_rules.add_disappeared_tracks_to_lost_pool(
                    right_state.tracks,
                    output_right.seen_raw_track_ids,
                    right_state.lost_display_pool,
                    right_state.consumed_lost_source_ids,
                    lw,
                    lh,
                )
                left_state.stats["lost_pool_expired_count"] += tracking_rules.expire_lost_display_pool(
                    left_state.lost_display_pool,
                    frame_idx,
                )
                right_state.stats["lost_pool_expired_count"] += tracking_rules.expire_lost_display_pool(
                    right_state.lost_display_pool,
                    frame_idx,
                )
            left_state.stats["expired_track_count"] += tracking_rules.expire_lost_states(
                left_state.tracks,
                output_left.seen_raw_track_ids,
            )
            right_state.stats["expired_track_count"] += tracking_rules.expire_lost_states(
                right_state.tracks,
                output_right.seen_raw_track_ids,
            )
            stage_finish("state_lifecycle", stage_started_at)

            stage_started_at = stage_start()
            left_candidates = dedupe_decisions_by_track_id(
                [d for d in output_left.decisions if candidate_is_available(d, use_provisional)]
            )
            right_candidates = dedupe_decisions_by_track_id(
                [d for d in output_right.decisions if candidate_is_available(d, use_provisional)]
            )

            matches: list[TrackedPair] = []
            used_pair_ids: set[int] = set()
            for class_id in CLASSES:
                left_cls = [d for d in left_candidates if d.class_id == class_id]
                right_cls = [d for d in right_candidates if d.class_id == class_id]
                for li, rj, cost in match_frame_candidates(
                    left_cls,
                    right_cls,
                    args.match_y_gap,
                    args.match_cost_limit,
                ):
                    ld = left_cls[li]
                    rd = right_cls[rj]
                    left_rect_center = rectified_from_decision(ld, geometry, "left")
                    right_rect_center = rectified_from_decision(rd, geometry, "right")
                    motion_key_left = (class_id, int(ld.state.track_id))
                    motion_key_right = (class_id, int(rd.state.track_id))
                    temporal = compensate_rectified_centers(
                        frame_pair=frame_pair,
                        left_raw=left_rect_center,
                        right_raw=right_rect_center,
                        left_state=left_motion_histories.get(motion_key_left),
                        right_state=right_motion_histories.get(motion_key_right),
                        enabled=args.enable_causal_time_compensation,
                        min_samples=args.temporal_min_samples,
                        decay=args.temporal_decay,
                        max_gap_ms=args.temporal_max_gap_ms,
                        max_shift_px=args.temporal_max_shift_px,
                        bound_margin_px=args.temporal_bound_margin_px,
                        frame_size=(lw, lh),
                        min_disparity_px=args.distance_min_disparity,
                        max_disparity_px=args.match_max_disparity,
                    )
                    if temporal.applied:
                        effective_left = temporal.left_compensated
                        effective_right = temporal.right_compensated
                    elif not temporal.enabled or args.temporal_invalid_policy == "raw":
                        effective_left = left_rect_center
                        effective_right = right_rect_center
                    else:
                        effective_left = None
                        effective_right = None
                    if effective_left is None or effective_right is None:
                        depth, valid, disparity = None, False, None
                        depth_filter_reason = temporal.reason
                    else:
                        depth, valid, _y_diff, disparity, depth_filter_reason = estimate_depth_from_disparity(
                            effective_left,
                            effective_right,
                            geometry,
                            args.match_max_disparity,
                            args.distance_min_disparity,
                        )
                    pair_key_left = (class_id, ld.state.track_id)
                    pair_key_right = (class_id, rd.state.track_id)

                    pair_id = None
                    if pair_id is None:
                        left_prev = left_raw_to_pair.get(pair_key_left)
                        if left_prev is not None:
                            pair_id = left_prev
                    if pair_id is None:
                        right_prev = right_raw_to_pair.get(pair_key_right)
                        if right_prev is not None:
                            pair_id = right_prev
                    if pair_id in used_pair_ids:
                        pair_id = None
                    if pair_id is None:
                        pair_id = next_pair_id
                        next_pair_id += 1
                    if pair_id not in stereo_pair_states:
                        stereo_pair_states[pair_id] = StereoPairState(
                            pair_id=pair_id,
                            class_id=class_id,
                            left_raw_id=ld.state.track_id,
                            right_raw_id=rd.state.track_id,
                            last_seen_frame=frame_idx,
                        )
                    pair_state = stereo_pair_states[pair_id]
                    raw_id_changed = (
                        pair_state.left_raw_id != ld.state.track_id
                        or pair_state.right_raw_id != rd.state.track_id
                    )
                    if pair_state.class_id != class_id:
                        pair_state.class_id = class_id
                    pair_state.left_raw_id = ld.state.track_id
                    pair_state.right_raw_id = rd.state.track_id
                    pair_state.last_seen_frame = frame_idx
                    used_pair_ids.add(pair_id)

                    label = stereo_pair_label(class_id, pair_id)
                    raw_depth_m = float(depth) if valid and depth is not None else None
                    prev_recoveries = pair_state.ema_recoveries
                    filtered_disparity_px = None
                    reacquire_candidate_count = 0
                    reacquire_event = False
                    if args.distance_filter_mode == "disparity_ema_reacquire":
                        (
                            filtered_depth_m,
                            distance_valid,
                            ema_invalid,
                            ema_filter_reason,
                            filtered_disparity_px,
                            reacquire_candidate_count,
                            reacquire_event,
                        ) = smooth_disparity_with_reacquire(
                            raw_depth_m,
                            disparity,
                            pair_state,
                            frame_idx,
                            raw_id_changed,
                            alpha=args.disparity_ema_alpha,
                            jump_ratio=args.disparity_ema_jump_ratio,
                            jump_abs_px=args.disparity_ema_jump_abs_px,
                            reacquire_samples=args.disparity_reacquire_samples,
                            stability_ratio=args.disparity_reacquire_stability_ratio,
                            stability_abs_px=args.disparity_reacquire_stability_abs_px,
                            max_frame_gap=args.disparity_reacquire_max_frame_gap,
                        )
                    else:
                        filtered_depth_m, distance_valid, ema_invalid, ema_filter_reason = smooth_distance_with_ema(
                            raw_depth_m,
                            pair_state,
                            args.distance_ema_alpha,
                            args.distance_ema_max_change,
                            args.distance_ema_min_abs_change,
                            args.distance_ema_reset_after_invalid,
                        )
                    if pair_state.ema_recoveries > prev_recoveries and distance_valid:
                        pair_stats["ema_recovery_count"] += 1
                    if raw_depth_m is None:
                        filter_reason = depth_filter_reason
                    elif distance_valid:
                        filter_reason = "ok"
                    else:
                        filter_reason = ema_filter_reason

                    matches.append(
                        TrackedPair(
                            stereo_pair_id=pair_id,
                            class_id=class_id,
                            label=label,
                            left_raw_id=ld.state.track_id,
                            right_raw_id=rd.state.track_id,
                            left_state_display_id=ld.state.display_id,
                            right_state_display_id=rd.state.display_id,
                            left_decision=ld,
                            right_decision=rd,
                            match_cost=cost,
                            left_center_rect=left_rect_center,
                            right_center_rect=right_rect_center,
                            depth_m=depth,
                            distance_valid=distance_valid,
                            filtered_depth_m=filtered_depth_m,
                            ema_invalid=ema_invalid,
                            filter_reason=filter_reason,
                            distance_filter_mode=args.distance_filter_mode,
                            filtered_disparity_px=filtered_disparity_px,
                            reacquire_candidate_count=reacquire_candidate_count,
                            reacquire_event=reacquire_event,
                            raw_id_changed=raw_id_changed,
                            disparity_px=disparity,
                            temporal=temporal,
                        )
                    )
                    left_raw_to_pair[pair_key_left] = pair_id
                    right_raw_to_pair[pair_key_right] = pair_id
            stage_finish("stereo_match_depth_ema", stage_started_at)

            if latency_enabled:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                core_elapsed_ms = (time.perf_counter() - core_started_at) * 1000.0
                if frame_idx >= args.latency_warmup_frames:
                    core_latencies_ms.append(core_elapsed_ms)
                    latency_frame_indices.append(frame_idx)

            # draw side-by-side output
            left_canvas = left_raw.copy()
            right_canvas = right_raw.copy()
            composite = np.hstack([left_canvas, right_canvas])

            for match in matches:
                color = color_for_pair(match.stereo_pair_id)
                draw_bbox(
                    left_canvas,
                    match.left_decision,
                    color,
                    f"{match.label} " + (f"{match.filtered_depth_m:.2f}m" if match.distance_valid and match.filtered_depth_m is not None else "invalid"),
                )
                draw_bbox(
                    right_canvas,
                    match.right_decision,
                    color,
                    f"{match.label} " + (f"{match.filtered_depth_m:.2f}m" if match.distance_valid and match.filtered_depth_m is not None else "invalid"),
                )

                lx1, ly1, lx2, ly2 = [int(round(v)) for v in match.left_decision.raw_box]
                rx1, ry1, rx2, ry2 = [int(round(v)) for v in match.right_decision.raw_box]
                lc = ((lx1 + lx2) // 2, (ly1 + ly2) // 2)
                rc = ((rx1 + rx2) // 2 + lw, (ry1 + ry2) // 2)
                pt1 = (
                    max(0, min(composite.shape[1] - 1, lc[0])),
                    max(0, min(composite.shape[0] - 1, lc[1])),
                )
                pt2 = (
                    max(0, min(composite.shape[1] - 1, rc[0])),
                    max(0, min(composite.shape[0] - 1, rc[1])),
                )
                cv2.line(composite, pt1, pt2, color, 1)

            # Remove stale pair IDs to keep continuity and avoid collisions.
            stale = [
                pair_id for pair_id, state in stereo_pair_states.items()
                if frame_idx - state.last_seen_frame > args.pair_ttl
            ]
            for pair_id in stale:
                stereo_pair_states.pop(pair_id, None)
                left_raw_to_pair = {
                    k: v for k, v in left_raw_to_pair.items() if v != pair_id
                }
                right_raw_to_pair = {
                    k: v for k, v in right_raw_to_pair.items() if v != pair_id
                }
            # Motion history is keyed independently from pair IDs; bound its memory.
            motion_state_ttl = max(120, args.pair_ttl)
            left_motion_histories = {
                key: state for key, state in left_motion_histories.items()
                if frame_idx - state.last_seen_frame <= motion_state_ttl
            }
            right_motion_histories = {
                key: state for key, state in right_motion_histories.items()
                if frame_idx - state.last_seen_frame <= motion_state_ttl
            }

            for match in matches:
                pair_stats["total_pairs"] += 1
                if match.distance_valid:
                    pair_stats["valid_pairs"] += 1
                    pair_stats["filtered_depth_values"].append(float(match.filtered_depth_m or 0.0))
                else:
                    pair_stats["invalid_pairs"] += 1
                    if match.ema_invalid:
                        pair_stats["ema_invalid_pairs"] += 1
                pair_stats["filter_reason_counts"][match.filter_reason] = (
                    pair_stats["filter_reason_counts"].get(match.filter_reason, 0) + 1
                )
                temporal = match.temporal
                pair_stats["temporal_reason_counts"][temporal.reason] = (
                    pair_stats["temporal_reason_counts"].get(temporal.reason, 0) + 1
                )
                pair_stats["temporal_deltas_ms"].append(frame_pair.stereo_delta_ns / 1_000_000.0)
                pair_stats["temporal_raw_disparities"].append(temporal.disparity_raw_px)
                if (
                    math.isfinite(temporal.disparity_raw_px)
                    and args.distance_min_disparity <= temporal.disparity_raw_px <= args.match_max_disparity
                ):
                    pair_stats["raw_geometry_valid"] += 1
                if temporal.enabled:
                    pair_stats["temporal_attempts"] += 1
                if temporal.applied:
                    pair_stats["temporal_applied"] += 1
                    pair_stats["compensated_geometry_valid"] += 1
                if temporal.left_shift_px is not None:
                    pair_stats["temporal_left_shifts"].append(temporal.left_shift_px)
                if temporal.right_shift_px is not None:
                    pair_stats["temporal_right_shifts"].append(temporal.right_shift_px)
                if temporal.disparity_compensated_px is not None:
                    pair_stats["temporal_comp_disparities"].append(temporal.disparity_compensated_px)
                    pair_stats["temporal_disparity_changes"].append(
                        temporal.disparity_compensated_px - temporal.disparity_raw_px
                    )
                lm = temporal.left_motion
                rm = temporal.right_motion
                lc = temporal.left_compensated
                rc = temporal.right_compensated
                csv_writer.writerow({
                    "frame_idx": frame_idx,
                    "image_name_left": left_path.name,
                    "image_name_right": right_path.name,
                    "class_id": match.class_id,
                    "class_name": CLASS_NAMES.get(match.class_id, f"class_{match.class_id}"),
                    "stereo_pair_id": match.stereo_pair_id,
                    "pair_label": match.label,
                    "left_raw_track_id": match.left_raw_id,
                    "right_raw_track_id": match.right_raw_id,
                    "left_display_id": "" if match.left_state_display_id is None else match.left_state_display_id,
                    "right_display_id": "" if match.right_state_display_id is None else match.right_state_display_id,
                    "match_cost": f"{match.match_cost:.3f}",
                    "x_left": f"{match.left_decision.raw_box[0]:.3f}",
                    "y_left": f"{match.left_decision.raw_box[1]:.3f}",
                    "x_right": f"{match.right_decision.raw_box[0]:.3f}",
                    "y_right": f"{match.right_decision.raw_box[1]:.3f}",
                    "disparity_px": "" if match.disparity_px is None else f"{match.disparity_px:.3f}",
                    "depth_m": "" if match.depth_m is None else f"{match.depth_m:.4f}",
                    "filtered_depth_m": "" if match.filtered_depth_m is None else f"{match.filtered_depth_m:.4f}",
                    "distance_valid": int(bool(match.distance_valid)),
                    "filter_reason": match.filter_reason,
                    "distance_filter_mode": match.distance_filter_mode,
                    "filtered_disparity_px": "" if match.filtered_disparity_px is None else f"{match.filtered_disparity_px:.6f}",
                    "reacquire_candidate_count": match.reacquire_candidate_count,
                    "reacquire_event": int(match.reacquire_event),
                    "raw_id_changed": int(match.raw_id_changed),
                    "y_diff_px": f"{abs(match.left_center_rect[1] - match.right_center_rect[1]):.3f}",
                    "center_distance_px": f"{math.hypot(match.left_center_rect[0] - match.right_center_rect[0], match.left_center_rect[1] - match.right_center_rect[1]):.3f}",
                    "left_conf": f"{float(match.left_decision.confidence):.3f}",
                    "right_conf": f"{float(match.right_decision.confidence):.3f}",
                    "left_status": match.left_decision.status,
                    "right_status": match.right_decision.status,
                    "left_status_reason": match.left_decision.status_reason,
                    "right_status_reason": match.right_decision.status_reason,
                    "left_draw_mode": match.left_decision.draw_mode,
                    "right_draw_mode": match.right_decision.draw_mode,
                    "left_timestamp_ns": frame_pair.left_timestamp_ns,
                    "right_timestamp_ns": frame_pair.right_timestamp_ns,
                    "stereo_delta_ns": frame_pair.stereo_delta_ns,
                    "stereo_delta_ms": f"{frame_pair.stereo_delta_ns / 1_000_000.0:.6f}",
                    "temporal_common_timestamp_ns": temporal.common_timestamp_ns,
                    "temporal_compensation_enabled": int(temporal.enabled),
                    "temporal_compensation_valid": int(temporal.valid),
                    "temporal_compensation_applied": int(temporal.applied),
                    "temporal_compensation_reason": temporal.reason,
                    "temporal_invalid_policy": args.temporal_invalid_policy,
                    "left_motion_samples": "" if lm is None else lm.sample_count,
                    "right_motion_samples": "" if rm is None else rm.sample_count,
                    "left_history_span_ms": "" if lm is None else f"{lm.span_ms:.3f}",
                    "right_history_span_ms": "" if rm is None else f"{rm.span_ms:.3f}",
                    "left_vx_rect_px_s": "" if lm is None else f"{lm.vx_px_s:.6f}",
                    "left_vy_rect_px_s": "" if lm is None else f"{lm.vy_px_s:.6f}",
                    "right_vx_rect_px_s": "" if rm is None else f"{rm.vx_px_s:.6f}",
                    "right_vy_rect_px_s": "" if rm is None else f"{rm.vy_px_s:.6f}",
                    "left_shift_px": "" if temporal.left_shift_px is None else f"{temporal.left_shift_px:.6f}",
                    "right_shift_px": "" if temporal.right_shift_px is None else f"{temporal.right_shift_px:.6f}",
                    "left_shift_x_px": "" if lc is None else f"{lc[0] - temporal.left_raw[0]:.6f}",
                    "left_shift_y_px": "" if lc is None else f"{lc[1] - temporal.left_raw[1]:.6f}",
                    "right_shift_x_px": "" if rc is None else f"{rc[0] - temporal.right_raw[0]:.6f}",
                    "right_shift_y_px": "" if rc is None else f"{rc[1] - temporal.right_raw[1]:.6f}",
                    "left_rect_x_raw": f"{temporal.left_raw[0]:.6f}",
                    "left_rect_y_raw": f"{temporal.left_raw[1]:.6f}",
                    "right_rect_x_raw": f"{temporal.right_raw[0]:.6f}",
                    "right_rect_y_raw": f"{temporal.right_raw[1]:.6f}",
                    "left_rect_x_compensated": "" if lc is None else f"{lc[0]:.6f}",
                    "left_rect_y_compensated": "" if lc is None else f"{lc[1]:.6f}",
                    "right_rect_x_compensated": "" if rc is None else f"{rc[0]:.6f}",
                    "right_rect_y_compensated": "" if rc is None else f"{rc[1]:.6f}",
                    "disparity_raw_px": f"{temporal.disparity_raw_px:.6f}",
                    "disparity_compensated_px": "" if temporal.disparity_compensated_px is None else f"{temporal.disparity_compensated_px:.6f}",
                    "y_diff_raw_px": f"{temporal.y_diff_raw_px:.6f}",
                    "y_diff_compensated_px": "" if temporal.y_diff_compensated_px is None else f"{temporal.y_diff_compensated_px:.6f}",
                })

            out_img = np.hstack([left_canvas, right_canvas])
            out_path = output_img_dir / f"{left_path.stem}_stereo.jpg"
            if not cv2.imwrite(str(out_path), out_img):
                print(f"[WARN] Failed to save image: {out_path}")
            writer.write(out_img)
            processed += 1

            if (frame_idx + 1) % 100 == 0 or (frame_idx + 1) == pairs_count:
                print(f"[INFO] Processed {frame_idx + 1}/{pairs_count}")

    finally:
        csv_fh.close()
        if writer is not None:
            writer.release()

    avg_filtered_depth = float("nan")
    min_filtered_depth = float("nan")
    max_filtered_depth = float("nan")
    if pair_stats["filtered_depth_values"]:
        filtered_depths = np.array(pair_stats["filtered_depth_values"], dtype=float)
        avg_filtered_depth = float(np.mean(filtered_depths))
        min_filtered_depth = float(np.min(filtered_depths))
        max_filtered_depth = float(np.max(filtered_depths))
    final_stats = {
        "total_pairs": pair_stats["total_pairs"],
        "valid_pairs": pair_stats["valid_pairs"],
        "invalid_pairs": pair_stats["invalid_pairs"],
        "ema_invalid_pairs": pair_stats["ema_invalid_pairs"],
        "ema_recovery_count": pair_stats["ema_recovery_count"],
        "filter_reason_counts": pair_stats["filter_reason_counts"],
        "temporal_attempts": pair_stats["temporal_attempts"],
        "temporal_applied": pair_stats["temporal_applied"],
        "temporal_reason_counts": pair_stats["temporal_reason_counts"],
        "temporal_deltas_ms": pair_stats["temporal_deltas_ms"],
        "temporal_left_shifts": pair_stats["temporal_left_shifts"],
        "temporal_right_shifts": pair_stats["temporal_right_shifts"],
        "temporal_disparity_changes": pair_stats["temporal_disparity_changes"],
        "temporal_raw_disparities": pair_stats["temporal_raw_disparities"],
        "temporal_comp_disparities": pair_stats["temporal_comp_disparities"],
        "raw_geometry_valid": pair_stats["raw_geometry_valid"],
        "compensated_geometry_valid": pair_stats["compensated_geometry_valid"],
        "avg_filtered_depth_m": avg_filtered_depth,
        "min_filtered_depth_m": min_filtered_depth,
        "max_filtered_depth_m": max_filtered_depth,
    }
    write_report(
        output_report_path,
        output_name,
        processed,
        skipped,
        final_stats,
        args.match_y_gap,
        args.match_cost_limit,
        args.match_max_disparity,
        args.distance_min_disparity,
        args.pair_ttl,
        sync_audit,
        {
            "enabled": args.enable_causal_time_compensation,
            "invalid_policy": args.temporal_invalid_policy,
            "history_size": args.temporal_history_size,
            "min_samples": args.temporal_min_samples,
            "decay": args.temporal_decay,
            "max_gap_ms": args.temporal_max_gap_ms,
            "max_shift_px": args.temporal_max_shift_px,
            "bound_margin_px": args.temporal_bound_margin_px,
        },
        {
            "mode": args.distance_filter_mode,
            "legacy_alpha": args.distance_ema_alpha,
            "legacy_max_change": args.distance_ema_max_change,
            "legacy_min_abs_change": args.distance_ema_min_abs_change,
            "legacy_reset_after_invalid": args.distance_ema_reset_after_invalid,
            "alpha": args.disparity_ema_alpha,
            "jump_ratio": args.disparity_ema_jump_ratio,
            "jump_abs_px": args.disparity_ema_jump_abs_px,
            "reacquire_samples": args.disparity_reacquire_samples,
            "stability_ratio": args.disparity_reacquire_stability_ratio,
            "stability_abs_px": args.disparity_reacquire_stability_abs_px,
            "max_frame_gap": args.disparity_reacquire_max_frame_gap,
        },
    )

    plot_created = False
    if not args.skip_distance_plot:
        try:
            from stereo.plot_stereo_distance_scatter import load_class_distance_data, plot_scatter

            plot_data = load_class_distance_data(output_csv_path)
            plot_scatter(plot_data, output_name, output_plot_path)
            plot_created = True
        except (ImportError, FileNotFoundError, OSError, ValueError) as exc:
            print(f"[WARN] Distance scatter plot was not generated: {exc}")

    if latency_enabled:
        if not core_latencies_ms:
            raise RuntimeError(
                "No measured latency frames remain after warm-up; increase --max-frames "
                "or reduce --latency-warmup-frames."
            )
        values = np.asarray(core_latencies_ms, dtype=float)
        mean_ms = float(np.mean(values))
        module_latency: dict[str, dict[str, float | int]] = {}
        for name, stage_values in stage_latencies_ms.items():
            stage_array = np.asarray(stage_values, dtype=float)
            if stage_array.size == 0:
                continue
            stage_mean = float(np.mean(stage_array))
            module_latency[name] = {
                "frames": int(stage_array.size),
                "mean_ms": stage_mean,
                "median_ms": float(np.percentile(stage_array, 50)),
                "p95_ms": float(np.percentile(stage_array, 95)),
                "share_of_core_mean": stage_mean / mean_ms if mean_ms > 0.0 else 0.0,
            }
        latency_report = {
            "schema_version": 1,
            "scope": "dual-eye core online pipeline",
            "included": [
                "left/right YOLO26n + BoT-SORT",
                "tensor-to-CPU observation extraction",
                "left/right V19 online causal rules",
                "same-class Hungarian matching",
                "optional causal time compensation",
                "rectified disparity/depth and EMA",
            ],
            "excluded": [
                "input file discovery and stereo filename synchronization",
                "image file read/decode and size normalization",
                "drawing and compositing",
                "CSV/image/video/report writes",
                "plot generation",
                "model and stereo-geometry initialization",
            ],
            "input_resolution": [int(w), int(h)],
            "nominal_input_fps": float(args.fps),
            "nominal_frame_budget_ms": 1000.0 / float(args.fps),
            "warmup_frames": int(args.latency_warmup_frames),
            "measured_frames": int(values.size),
            "first_measured_frame": int(latency_frame_indices[0]),
            "last_measured_frame": int(latency_frame_indices[-1]),
            "causal_time_compensation": bool(args.enable_causal_time_compensation),
            "mean_ms": mean_ms,
            "median_ms": float(np.percentile(values, 50)),
            "p90_ms": float(np.percentile(values, 90)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "throughput_fps_from_mean": 1000.0 / mean_ms if mean_ms > 0.0 else None,
            "within_25fps_budget_fraction": float(np.mean(values <= 40.0)),
            "cuda_available": bool(torch.cuda.is_available()),
            "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "gpu_peak_allocated_mib": (
                float(torch.cuda.max_memory_allocated() / (1024.0 ** 2))
                if torch.cuda.is_available() else None
            ),
            "gpu_peak_reserved_mib": (
                float(torch.cuda.max_memory_reserved() / (1024.0 ** 2))
                if torch.cuda.is_available() else None
            ),
            "module_latency": module_latency,
            "module_timing_note": (
                "CUDA is synchronized at every module boundary for attribution. "
                "This can add small boundary overhead relative to end-to-end timing."
            ),
            "per_frame_ms": core_latencies_ms,
        }
        latency_report_path = Path(args.latency_report_path)
        latency_report_path.parent.mkdir(parents=True, exist_ok=True)
        latency_report_path.write_text(
            json.dumps(latency_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Latency report  : {latency_report_path}")

    print("=" * 70)
    print("Stereo distance job finished")
    print(f"Processed frames: {processed}")
    print(f"Skipped frames  : {skipped}")
    print(f"Output images   : {output_img_dir}")
    print(f"Output video    : {output_video_path}")
    print(f"Output csv      : {output_csv_path}")
    print(f"Output report   : {output_report_path}")
    if plot_created:
        print(f"Output plot     : {output_plot_path}")
    elif args.skip_distance_plot:
        print("Output plot     : skipped by --skip-distance-plot")
    else:
        print("Output plot     : not generated (see warning above)")
    print(f"Matched pairs   : {pair_stats['total_pairs']}")
    print(f"Valid distances : {pair_stats['valid_pairs']}")
    print(f"Invalid dist    : {pair_stats['invalid_pairs']}")
    print(f"Filter invalid  : {pair_stats['ema_invalid_pairs']}")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
