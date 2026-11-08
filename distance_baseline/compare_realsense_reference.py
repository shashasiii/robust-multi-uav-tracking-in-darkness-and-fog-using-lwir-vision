"""Build per-frame RealSense reference-depth comparisons for thermal stereo CSVs.

The historical stereo CSV stores only each thermal detection's top-left point.
This command reruns the detector on the left thermal image to recover the full
box, matches it back to the CSV row, projects the box into the RGB/Depth view,
and samples a robust central depth ROI.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "weights" / "v8_yolo26n_e300_b128" / "weights" / "best.pt"
DEFAULT_CALIBRATION = Path(__file__).with_name("realsense_thermal_calibration.yaml")


OUTPUT_FIELDS = [
    "frame_idx",
    "image_name_left",
    "class_id",
    "class_name",
    "pair_label",
    "thermal_timestamp_ns",
    "realsense_timestamp_ns",
    "realsense_image_name",
    "timestamp_match_source",
    "timestamp_delta_ms",
    "bbox_match_distance_px",
    "thermal_x1",
    "thermal_y1",
    "thermal_x2",
    "thermal_y2",
    "rgb_roi_x1",
    "rgb_roi_y1",
    "rgb_roi_x2",
    "rgb_roi_y2",
    "rgb_center_roi_x1",
    "rgb_center_roi_y1",
    "rgb_center_roi_x2",
    "rgb_center_roi_y2",
    "rgb_cue_found",
    "rgb_cue_x",
    "rgb_cue_y",
    "rgb_cue_pixels",
    "reference_method",
    "rs_valid_pixels",
    "rs_total_pixels",
    "rs_valid_fraction",
    "rs_depth_bbox_m",
    "rs_depth_m",
    "rs_depth_p25_m",
    "rs_depth_p75_m",
    "rs_depth_iqr_m",
    "reference_valid",
    "reference_reason",
    "stereo_disparity_px",
    "stereo_depth_m",
    "stereo_distance_m",
    "stereo_distance_valid",
    "stereo_filter_reason",
    "raw_error_m",
    "raw_abs_error_m",
    "raw_relative_error_pct",
    "ema_error_m",
    "ema_abs_error_m",
    "ema_relative_error_pct",
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
    "reference_quality_valid",
    "reference_quality_reason",
    "reference_temporal_checked",
    "reference_temporal_warning",
    "reference_local_median_m",
    "reference_temporal_residual_m",
    "reference_temporal_threshold_m",
]

TEMPORAL_PASSTHROUGH_FIELDS = [
    "left_timestamp_ns", "right_timestamp_ns", "stereo_delta_ns", "stereo_delta_ms",
    "temporal_common_timestamp_ns", "temporal_compensation_enabled",
    "temporal_compensation_valid", "temporal_compensation_applied",
    "temporal_compensation_reason", "temporal_invalid_policy",
    "left_motion_samples", "right_motion_samples", "left_history_span_ms",
    "right_history_span_ms", "left_vx_rect_px_s", "left_vy_rect_px_s",
    "right_vx_rect_px_s", "right_vy_rect_px_s", "left_shift_px", "right_shift_px",
    "left_shift_x_px", "left_shift_y_px", "right_shift_x_px", "right_shift_y_px",
    "left_rect_x_raw", "left_rect_y_raw", "right_rect_x_raw", "right_rect_y_raw",
    "left_rect_x_compensated", "left_rect_y_compensated",
    "right_rect_x_compensated", "right_rect_y_compensated",
    "disparity_raw_px", "disparity_compensated_px", "y_diff_raw_px",
    "y_diff_compensated_px",
]

REFERENCE_REUSE_FIELDS = [
    "thermal_timestamp_ns",
    "realsense_timestamp_ns",
    "realsense_image_name",
    "timestamp_match_source",
    "timestamp_delta_ms",
    "bbox_match_distance_px",
    "thermal_x1",
    "thermal_y1",
    "thermal_x2",
    "thermal_y2",
    "rgb_roi_x1",
    "rgb_roi_y1",
    "rgb_roi_x2",
    "rgb_roi_y2",
    "rgb_center_roi_x1",
    "rgb_center_roi_y1",
    "rgb_center_roi_x2",
    "rgb_center_roi_y2",
    "rgb_cue_found",
    "rgb_cue_x",
    "rgb_cue_y",
    "rgb_cue_pixels",
    "reference_method",
    "rs_valid_pixels",
    "rs_total_pixels",
    "rs_valid_fraction",
    "rs_depth_bbox_m",
    "rs_depth_m",
    "rs_depth_p25_m",
    "rs_depth_p75_m",
    "rs_depth_iqr_m",
    "reference_valid",
    "reference_reason",
]


@dataclass(frozen=True)
class Calibration:
    thermal_k: np.ndarray
    thermal_dist: np.ndarray
    rgb_k: np.ndarray
    rgb_dist: np.ndarray
    rotation: np.ndarray
    translation_m: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stereo-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument(
        "--reuse-boxes-from-csv",
        type=Path,
        help="Reuse recovered thermal boxes from an earlier comparison CSV.",
    )
    parser.add_argument(
        "--reuse-reference-from-csv",
        type=Path,
        help=(
            "Freeze RealSense synchronization, ROI, depth, validity, and reason from an "
            "earlier comparison CSV; only candidate stereo errors are recomputed."
        ),
    )
    parser.add_argument("--device", default="cpu", help="Ultralytics device, for example cpu or 0.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.30)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--max-timestamp-delta-ms", type=float, default=20.0)
    parser.add_argument("--max-bbox-match-distance-px", type=float, default=8.0)
    parser.add_argument("--center-fraction", type=float, default=0.35)
    parser.add_argument("--rgb-search-margin-fraction", type=float, default=1.0)
    parser.add_argument("--blue-min", type=int, default=80)
    parser.add_argument("--blue-minus-green-min", type=int, default=20)
    parser.add_argument("--blue-minus-red-min", type=int, default=40)
    parser.add_argument("--blue-component-max-pixels", type=int, default=200)
    parser.add_argument(
        "--allow-projection-only-reference",
        action="store_true",
        help="Accept projection-center depth without an RGB blue target cue.",
    )
    parser.add_argument("--projection-iterations", type=int, default=3)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=9)
    parser.add_argument("--max-depth-iqr-m", type=float, default=0.50)
    parser.add_argument("--quality-max-depth-iqr-m", type=float, default=0.20)
    parser.add_argument("--quality-min-valid-depth-pixels", type=int, default=25)
    parser.add_argument("--min-depth-m", type=float, default=0.20)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--skip-summary", action="store_true")
    return parser.parse_args()


def parse_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return ""
    return f"{value:.{digits}f}"


def timestamp_ns_from_name(name: str) -> int:
    stem = Path(name).stem
    parts = stem.rsplit("_", 2)
    if len(parts) < 2 or not parts[-2].isdigit() or not parts[-1].isdigit():
        raise ValueError(f"Cannot parse sec/nsec timestamp from {name!r}")
    return int(parts[-2]) * 1_000_000_000 + int(parts[-1])


def load_calibration(path: Path) -> Calibration:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    thermal = data["left_thermal"]
    rgb = data["rgb"]
    extrinsic = data["left_thermal_to_rgb"]
    return Calibration(
        thermal_k=np.asarray(thermal["K"], dtype=np.float64),
        thermal_dist=np.asarray(thermal["dist"], dtype=np.float64),
        rgb_k=np.asarray(rgb["K"], dtype=np.float64),
        rgb_dist=np.asarray(rgb["dist"], dtype=np.float64),
        rotation=np.asarray(extrinsic["R"], dtype=np.float64),
        translation_m=np.asarray(extrinsic["T_m"], dtype=np.float64),
    )


def load_pair_map(dataset_root: Path) -> dict[int, int]:
    pair_path = dataset_root / "paired_rgb_left" / "pairs_rgb_left.csv"
    if not pair_path.exists():
        return {}
    result: dict[int, int] = {}
    with pair_path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                left_ns = timestamp_ns_from_name(row["left_thermal_frame_orig"])
                rs_ns = timestamp_ns_from_name(row["realsense_frame_orig"])
            except (KeyError, ValueError):
                continue
            result[left_ns] = rs_ns
    return result


def nearest_timestamp(target: int, sorted_values: list[int]) -> int:
    index = bisect.bisect_left(sorted_values, target)
    candidates = []
    if index:
        candidates.append(sorted_values[index - 1])
    if index < len(sorted_values):
        candidates.append(sorted_values[index])
    if not candidates:
        raise ValueError("No RealSense depth timestamps are available")
    return min(candidates, key=lambda value: abs(value - target))


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def comparison_key(row: dict[str, str]) -> str:
    """Return a stable per-target key that also supports multi-target frames."""
    return "|".join(
        [
            row.get("image_name_left", ""),
            row.get("class_id", ""),
            row.get("pair_label", ""),
        ]
    )


def recover_boxes(
    image_paths: list[Path],
    rows_by_name: dict[str, list[dict[str, str]]],
    model_path: Path,
    device: str,
    batch_size: int,
    conf: float,
    iou: float,
) -> dict[str, tuple[np.ndarray, float] | None]:
    model = YOLO(str(model_path))
    recovered: dict[str, tuple[np.ndarray, float] | None] = {}
    sources = [str(path) for path in image_paths]
    processed = 0
    for batch in chunks(sources, batch_size):
        results = model.predict(
            source=batch,
            conf=conf,
            iou=iou,
            device=device,
            batch=len(batch),
            verbose=False,
        )
        for source, result in zip(batch, results):
            name = Path(source).name
            boxes = result.boxes
            xyxy = (
                boxes.xyxy.detach().cpu().numpy()
                if boxes is not None and len(boxes) > 0
                else np.empty((0, 4), dtype=np.float64)
            )
            classes = (
                boxes.cls.detach().cpu().numpy().astype(int)
                if boxes is not None and len(boxes) > 0
                else np.empty(0, dtype=int)
            )
            for row in rows_by_name[name]:
                key = comparison_key(row)
                wanted_class = int(row["class_id"])
                target_xy = np.asarray([float(row["x_left"]), float(row["y_left"])])
                candidates = np.flatnonzero(classes == wanted_class)
                if candidates.size == 0:
                    recovered[key] = None
                    continue
                distances = np.linalg.norm(xyxy[candidates, :2] - target_xy, axis=1)
                best_local = int(np.argmin(distances))
                best = int(candidates[best_local])
                recovered[key] = (
                    xyxy[best].astype(np.float64),
                    float(distances[best_local]),
                )
        processed += len(batch)
        print(f"Recovered thermal boxes: {processed}/{len(sources)}", flush=True)
    return recovered


def load_recovered_boxes(path: Path) -> dict[str, tuple[np.ndarray, float] | None]:
    recovered: dict[str, tuple[np.ndarray, float] | None] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            key = comparison_key(row)
            values = [parse_float(row.get(key)) for key in ("thermal_x1", "thermal_y1", "thermal_x2", "thermal_y2")]
            distance = parse_float(row.get("bbox_match_distance_px"))
            if row.get("image_name_left") and all(value is not None for value in values) and distance is not None:
                recovered[key] = (np.asarray(values, dtype=np.float64), distance)
            elif row.get("image_name_left"):
                recovered[key] = None
    return recovered


def load_reference_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    result = {comparison_key(row): row for row in rows if row.get("image_name_left")}
    if len(result) != len(rows):
        raise ValueError(f"Reference reuse CSV has empty or duplicate target keys: {path}")
    return result


def project_box(box: np.ndarray, depth_m: float, calibration: Calibration) -> np.ndarray:
    x1, y1, x2, y2 = box.tolist()
    corners = np.asarray(
        [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float64
    )
    normalized = cv2.undistortPoints(
        corners, calibration.thermal_k, calibration.thermal_dist
    ).reshape(-1, 2)
    points_left = np.column_stack(
        [normalized * depth_m, np.full(normalized.shape[0], depth_m)]
    )
    # MATLAB calibration matrices are recorded in row-vector convention.
    points_rgb = points_left @ calibration.rotation + calibration.translation_m
    projected, _ = cv2.projectPoints(
        points_rgb,
        np.zeros(3),
        np.zeros(3),
        calibration.rgb_k,
        calibration.rgb_dist,
    )
    return projected.reshape(-1, 2)


def bounded_roi(projected: np.ndarray, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not np.isfinite(projected).all():
        return None
    low = np.floor(projected.min(axis=0)).astype(int)
    high = np.ceil(projected.max(axis=0)).astype(int)
    x1 = max(0, min(width - 1, int(low[0])))
    y1 = max(0, min(height - 1, int(low[1])))
    x2 = max(0, min(width - 1, int(high[0])))
    y2 = max(0, min(height - 1, int(high[1])))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def central_roi(roi: tuple[int, int, int, int], fraction: float) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    width = max(1, int(round((x2 - x1 + 1) * fraction)))
    height = max(1, int(round((y2 - y1 + 1) * fraction)))
    sx1 = max(x1, int(round(cx - (width - 1) / 2.0)))
    sy1 = max(y1, int(round(cy - (height - 1) / 2.0)))
    sx2 = min(x2, sx1 + width - 1)
    sy2 = min(y2, sy1 + height - 1)
    return sx1, sy1, sx2, sy2


def expanded_roi(
    roi: tuple[int, int, int, int],
    margin_fraction: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = roi
    margin_x = int(round((x2 - x1 + 1) * margin_fraction))
    margin_y = int(round((y2 - y1 + 1) * margin_fraction))
    return (
        max(0, x1 - margin_x),
        max(0, y1 - margin_y),
        min(width - 1, x2 + margin_x),
        min(height - 1, y2 + margin_y),
    )


def recenter_roi(
    template: tuple[int, int, int, int],
    center_x: float,
    center_y: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    tw = template[2] - template[0] + 1
    th = template[3] - template[1] + 1
    x1 = int(round(center_x - (tw - 1) / 2.0))
    y1 = int(round(center_y - (th - 1) / 2.0))
    x1 = max(0, min(width - tw, x1))
    y1 = max(0, min(height - th, y1))
    return x1, y1, x1 + tw - 1, y1 + th - 1


def find_blue_cue(
    rgb_image: np.ndarray,
    search_roi: tuple[int, int, int, int],
    target_center: tuple[float, float],
    blue_min: int,
    blue_minus_green_min: int,
    blue_minus_red_min: int,
    component_max_pixels: int,
) -> tuple[float, float, int] | None:
    x1, y1, x2, y2 = search_roi
    crop = rgb_image[y1 : y2 + 1, x1 : x2 + 1]
    blue, green, red = cv2.split(crop)
    blue_i = blue.astype(np.int16)
    mask = (
        (blue >= blue_min)
        & ((blue_i - green.astype(np.int16)) >= blue_minus_green_min)
        & ((blue_i - red.astype(np.int16)) >= blue_minus_red_min)
    ).astype(np.uint8)
    count, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[float, float, int, float]] = []
    tx, ty = target_center
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if not 1 <= area <= component_max_pixels:
            continue
        cx = float(centroids[label, 0] + x1)
        cy = float(centroids[label, 1] + y1)
        score = (cx - tx) ** 2 + (cy - ty) ** 2
        candidates.append((cx, cy, area, score))
    if not candidates:
        return None
    cx, cy, area, _ = min(candidates, key=lambda item: item[3])
    return cx, cy, area


def depth_values(
    depth_raw: np.ndarray,
    roi: tuple[int, int, int, int],
    scale: float,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, int]:
    x1, y1, x2, y2 = roi
    patch = depth_raw[y1 : y2 + 1, x1 : x2 + 1]
    values = patch.astype(np.float64).reshape(-1) * scale
    valid = values[(values >= min_depth_m) & (values <= max_depth_m)]
    return valid, int(values.size)


def error_fields(estimate: float | None, reference: float | None) -> tuple[str, str, str]:
    if estimate is None or reference is None or reference <= 0:
        return "", "", ""
    error = estimate - reference
    return (
        format_number(error),
        format_number(abs(error)),
        format_number(100.0 * error / reference, 3),
    )


def apply_reference_quality(
    path: Path,
    max_iqr_m: float,
    min_valid_pixels: int,
) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))

    valid_by_target_frame: dict[tuple[str, str, int], float] = {}
    for row in rows:
        reference = parse_float(row.get("rs_depth_m"))
        if row.get("reference_valid") == "1" and reference is not None:
            target = (row.get("class_id", ""), row.get("pair_label", ""))
            valid_by_target_frame[(*target, int(row["frame_idx"]))] = reference

    for row in rows:
        reasons: list[str] = []
        if row.get("reference_valid") != "1":
            reasons.append(f"reference_invalid:{row.get('reference_reason', '') or 'unknown'}")
        else:
            iqr = parse_float(row.get("rs_depth_iqr_m"))
            pixels = parse_float(row.get("rs_valid_pixels"))
            if iqr is None or pixels is None:
                reasons.append("quality_fields_missing")
            else:
                if iqr > max_iqr_m:
                    reasons.append("quality_depth_iqr_above_limit")
                if pixels < min_valid_pixels:
                    reasons.append("quality_insufficient_valid_depth_pixels")
        row["reference_quality_valid"] = "0" if reasons else "1"
        row["reference_quality_reason"] = ";".join(reasons) if reasons else "ok"

        frame_idx = int(row["frame_idx"])
        target = (row.get("class_id", ""), row.get("pair_label", ""))
        reference = parse_float(row.get("rs_depth_m"))
        window = [
            valid_by_target_frame[(*target, index)]
            for index in range(frame_idx - 2, frame_idx + 3)
            if (*target, index) in valid_by_target_frame
        ]
        if reference is None or len(window) < 3:
            row["reference_temporal_checked"] = "0"
            row["reference_temporal_warning"] = "0"
            continue
        values = np.asarray(window, dtype=np.float64)
        local_median = float(np.median(values))
        mad = float(np.median(np.abs(values - local_median)))
        threshold = max(1.0, 3.0 * 1.4826 * mad)
        residual = float(reference - local_median)
        row["reference_temporal_checked"] = "1"
        row["reference_temporal_warning"] = "1" if abs(residual) > threshold else "0"
        row["reference_local_median_m"] = format_number(local_median)
        row["reference_temporal_residual_m"] = format_number(residual)
        row["reference_temporal_threshold_m"] = format_number(threshold)

    temp_path = path.with_suffix(path.suffix + ".quality.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, path)


def main() -> int:
    args = parse_args()
    for path, label in [
        (args.dataset_root, "dataset root"),
        (args.stereo_csv, "stereo CSV"),
        (args.model, "model"),
        (args.calibration, "calibration"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if not 0 < args.center_fraction <= 1:
        raise ValueError("--center-fraction must be in (0, 1]")
    if args.quality_max_depth_iqr_m <= 0:
        raise ValueError("--quality-max-depth-iqr-m must be > 0")
    if args.quality_min_valid_depth_pixels < 1:
        raise ValueError("--quality-min-valid-depth-pixels must be >= 1")

    with args.stereo_csv.open("r", encoding="utf-8-sig", newline="") as fh:
        stereo_rows = list(csv.DictReader(fh))
    if not stereo_rows:
        raise ValueError(f"Stereo CSV has no rows: {args.stereo_csv}")
    names = [row["image_name_left"] for row in stereo_rows]
    keys = [comparison_key(row) for row in stereo_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Stereo CSV has duplicate image/class/pair target keys")

    thermal_dir = args.dataset_root / "left_thermal"
    depth_dir = args.dataset_root / "realsense" / "depth_raw"
    depth_paths = sorted(depth_dir.glob("realsense_depth_raw_*.png"))
    if not depth_paths:
        raise FileNotFoundError(f"No RealSense raw depth PNGs found in {depth_dir}")
    depth_by_timestamp = {timestamp_ns_from_name(path.name): path for path in depth_paths}
    depth_timestamps = sorted(depth_by_timestamp)
    pair_map = load_pair_map(args.dataset_root)
    rows_by_name: dict[str, list[dict[str, str]]] = {}
    for row in stereo_rows:
        rows_by_name.setdefault(row["image_name_left"], []).append(row)
    unique_names = list(dict.fromkeys(names))
    thermal_paths = [thermal_dir / name for name in unique_names]
    missing_thermal = [path for path in thermal_paths if not path.exists()]
    if missing_thermal:
        raise FileNotFoundError(f"Missing thermal image: {missing_thermal[0]}")

    if args.reuse_boxes_from_csv:
        if not args.reuse_boxes_from_csv.exists():
            raise FileNotFoundError(args.reuse_boxes_from_csv)
        boxes = load_recovered_boxes(args.reuse_boxes_from_csv)
        print(f"Reused thermal boxes from: {args.reuse_boxes_from_csv}", flush=True)
    else:
        boxes = recover_boxes(
            thermal_paths,
            rows_by_name,
            args.model,
            args.device,
            args.batch_size,
            args.conf,
            args.iou,
        )
    frozen_references: dict[str, dict[str, str]] = {}
    if args.reuse_reference_from_csv:
        if not args.reuse_reference_from_csv.exists():
            raise FileNotFoundError(args.reuse_reference_from_csv)
        frozen_references = load_reference_rows(args.reuse_reference_from_csv)
        missing_references = sorted(set(keys) - set(frozen_references))
        if missing_references:
            raise ValueError(
                f"Frozen reference CSV is missing {len(missing_references)} candidate targets; "
                f"first missing: {missing_references[0]}"
            )
        print(f"Reused frozen RealSense references from: {args.reuse_reference_from_csv}", flush=True)
    calibration = load_calibration(args.calibration)
    scale = float((args.dataset_root / "depth_scale.txt").read_text(encoding="utf-8").strip())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    depth_cache: dict[Path, np.ndarray] = {}
    rgb_cache: dict[Path, np.ndarray] = {}
    with args.output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for index, row in enumerate(stereo_rows, start=1):
            image_name = row["image_name_left"]
            key = comparison_key(row)
            thermal_ns = timestamp_ns_from_name(image_name)
            paired_ns = pair_map.get(thermal_ns)
            if paired_ns in depth_by_timestamp:
                rs_ns = int(paired_ns)
                match_source = "paired_rgb_left_csv"
            else:
                rs_ns = nearest_timestamp(thermal_ns, depth_timestamps)
                match_source = "nearest_timestamp"
            delta_ms = abs(rs_ns - thermal_ns) / 1_000_000.0
            depth_path = depth_by_timestamp[rs_ns]
            output = {field: "" for field in OUTPUT_FIELDS}
            output.update(
                {
                    "frame_idx": row["frame_idx"],
                    "image_name_left": image_name,
                    "class_id": row["class_id"],
                    "class_name": row["class_name"],
                    "pair_label": row["pair_label"],
                    "thermal_timestamp_ns": str(thermal_ns),
                    "realsense_timestamp_ns": str(rs_ns),
                    "realsense_image_name": depth_path.name,
                    "timestamp_match_source": match_source,
                    "timestamp_delta_ms": format_number(delta_ms, 3),
                    "stereo_disparity_px": row["disparity_px"],
                    "stereo_depth_m": row["depth_m"],
                    "stereo_distance_m": row["distance_m"],
                    "stereo_distance_valid": row["distance_valid"],
                    "stereo_filter_reason": row["filter_reason"],
                }
            )
            for field in TEMPORAL_PASSTHROUGH_FIELDS:
                output[field] = row.get(field, "")
            frozen = frozen_references.get(key)
            if frozen is not None:
                for field in REFERENCE_REUSE_FIELDS:
                    output[field] = frozen.get(field, "")
                reference = (
                    parse_float(frozen.get("rs_depth_m"))
                    if frozen.get("reference_valid") == "1"
                    else None
                )
                raw_error = error_fields(parse_float(row.get("depth_m")), reference)
                ema_error = error_fields(parse_float(row.get("distance_m")), reference)
                output["raw_error_m"], output["raw_abs_error_m"], output["raw_relative_error_pct"] = raw_error
                output["ema_error_m"], output["ema_abs_error_m"], output["ema_relative_error_pct"] = ema_error
                writer.writerow(output)
                if index % 100 == 0 or index == len(stereo_rows):
                    print(f"Wrote comparisons: {index}/{len(stereo_rows)}", flush=True)
                continue
            recovered = boxes.get(key)
            reason = "ok"
            reference: float | None = None
            if delta_ms > args.max_timestamp_delta_ms:
                reason = "timestamp_delta_above_limit"
            elif recovered is None:
                reason = "thermal_detection_not_recovered"
            else:
                box, box_distance = recovered
                output["bbox_match_distance_px"] = format_number(box_distance, 3)
                for key, value in zip(
                    ["thermal_x1", "thermal_y1", "thermal_x2", "thermal_y2"], box
                ):
                    output[key] = format_number(float(value), 3)
                if box_distance > args.max_bbox_match_distance_px:
                    reason = "thermal_bbox_match_above_limit"
                else:
                    if depth_path not in depth_cache:
                        image = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
                        if image is not None and image.ndim == 3 and image.shape[2] == 1:
                            image = image[:, :, 0]
                        if image is None:
                            reason = "depth_image_read_failed"
                        elif image.ndim != 2 or image.dtype != np.uint16:
                            reason = "depth_image_not_uint16_single_channel"
                        else:
                            depth_cache.clear()
                            depth_cache[depth_path] = image
                    depth_raw = depth_cache.get(depth_path)
                    if depth_raw is not None:
                        initial_guess = parse_float(row.get("depth_m")) or 4.0
                        guess = initial_guess
                        final_roi = None
                        final_center = None
                        center_values = np.asarray([], dtype=np.float64)
                        center_total = 0
                        bbox_values = np.asarray([], dtype=np.float64)
                        initial_projected = project_box(box, initial_guess, calibration)
                        initial_roi = bounded_roi(
                            initial_projected, depth_raw.shape[1], depth_raw.shape[0]
                        )
                        cue = None
                        if initial_roi is not None:
                            search_roi = expanded_roi(
                                initial_roi,
                                args.rgb_search_margin_fraction,
                                depth_raw.shape[1],
                                depth_raw.shape[0],
                            )
                            rgb_name = depth_path.name.replace(
                                "realsense_depth_raw_", "realsense_rgb_image_"
                            )
                            rgb_path = args.dataset_root / "realsense" / "rgb" / rgb_name
                            if rgb_path.exists():
                                if rgb_path not in rgb_cache:
                                    rgb_image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                                    if rgb_image is not None:
                                        rgb_cache.clear()
                                        rgb_cache[rgb_path] = rgb_image
                                rgb_image = rgb_cache.get(rgb_path)
                                if rgb_image is not None:
                                    target_center = (
                                        (initial_roi[0] + initial_roi[2]) / 2.0,
                                        (initial_roi[1] + initial_roi[3]) / 2.0,
                                    )
                                    cue = find_blue_cue(
                                        rgb_image,
                                        search_roi,
                                        target_center,
                                        args.blue_min,
                                        args.blue_minus_green_min,
                                        args.blue_minus_red_min,
                                        args.blue_component_max_pixels,
                                    )
                        output["rgb_cue_found"] = "1" if cue is not None else "0"
                        if cue is not None:
                            output["rgb_cue_x"] = format_number(cue[0], 3)
                            output["rgb_cue_y"] = format_number(cue[1], 3)
                            output["rgb_cue_pixels"] = str(cue[2])
                        for _ in range(max(1, args.projection_iterations)):
                            projected = project_box(box, guess, calibration)
                            final_roi = bounded_roi(projected, depth_raw.shape[1], depth_raw.shape[0])
                            if final_roi is None:
                                break
                            final_center = central_roi(final_roi, args.center_fraction)
                            center_values, center_total = depth_values(
                                depth_raw,
                                final_center,
                                scale,
                                args.min_depth_m,
                                args.max_depth_m,
                            )
                            if center_values.size:
                                guess = float(np.median(center_values))
                        if final_roi is None or final_center is None:
                            reason = "projected_roi_outside_depth_image"
                        else:
                            if cue is not None:
                                template = central_roi(final_roi, args.center_fraction)
                                final_center = recenter_roi(
                                    template,
                                    cue[0],
                                    cue[1],
                                    depth_raw.shape[1],
                                    depth_raw.shape[0],
                                )
                                output["reference_method"] = "rgb_blue_cue"
                            else:
                                output["reference_method"] = "projection_center"
                            center_values, center_total = depth_values(
                                depth_raw,
                                final_center,
                                scale,
                                args.min_depth_m,
                                args.max_depth_m,
                            )
                            bbox_values, _ = depth_values(
                                depth_raw,
                                final_roi,
                                scale,
                                args.min_depth_m,
                                args.max_depth_m,
                            )
                            for prefix, values in [("rgb_roi", final_roi), ("rgb_center_roi", final_center)]:
                                for suffix, value in zip(("x1", "y1", "x2", "y2"), values):
                                    output[f"{prefix}_{suffix}"] = str(value)
                            output["rs_valid_pixels"] = str(center_values.size)
                            output["rs_total_pixels"] = str(center_total)
                            output["rs_valid_fraction"] = format_number(
                                center_values.size / center_total if center_total else 0.0, 4
                            )
                            if bbox_values.size:
                                output["rs_depth_bbox_m"] = format_number(
                                    float(np.median(bbox_values))
                                )
                            if cue is None and not args.allow_projection_only_reference:
                                reason = "rgb_target_cue_not_found"
                            elif center_values.size < args.min_valid_depth_pixels:
                                reason = "insufficient_valid_depth_pixels"
                            else:
                                p25, median, p75 = np.percentile(center_values, [25, 50, 75])
                                iqr = float(p75 - p25)
                                output["rs_depth_m"] = format_number(float(median))
                                output["rs_depth_p25_m"] = format_number(float(p25))
                                output["rs_depth_p75_m"] = format_number(float(p75))
                                output["rs_depth_iqr_m"] = format_number(iqr)
                                if iqr > args.max_depth_iqr_m:
                                    reason = "depth_iqr_above_limit"
                                else:
                                    reference = float(median)

            output["reference_valid"] = "1" if reason == "ok" else "0"
            output["reference_reason"] = reason
            raw_error = error_fields(parse_float(row.get("depth_m")), reference)
            ema_error = error_fields(parse_float(row.get("distance_m")), reference)
            output["raw_error_m"], output["raw_abs_error_m"], output["raw_relative_error_pct"] = raw_error
            output["ema_error_m"], output["ema_abs_error_m"], output["ema_relative_error_pct"] = ema_error
            writer.writerow(output)
            if index % 100 == 0 or index == len(stereo_rows):
                print(f"Wrote comparisons: {index}/{len(stereo_rows)}", flush=True)

    apply_reference_quality(
        args.output_csv,
        args.quality_max_depth_iqr_m,
        args.quality_min_valid_depth_pixels,
    )
    print(f"Comparison CSV: {args.output_csv}")
    if not args.skip_summary:
        summary_script = Path(__file__).with_name("summarize_distance_comparison.py")
        subprocess.run(
            [sys.executable, str(summary_script), "--comparison-csv", str(args.output_csv)],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
