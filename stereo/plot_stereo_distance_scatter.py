# -*- coding: UTF-8 -*-
"""Plot per-frame stereo distance scatter charts by class.

Usage:
  python stereo\\plot_stereo_distance_scatter.py --output-name <实验名> --csv-path <csv-path>
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "distances"
DEFAULT_CSV_ROOT = REPO_ROOT / "outputs" / "stereo" / "artifacts"
CLASS_NAMES = {0: "drone", 1: "person"}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot stereo distance per-frame scatter by class.")
    parser.add_argument("--output-name", required=True, help="Experiment name.")
    parser.add_argument(
        "--csv-path",
        default=str(DEFAULT_CSV_ROOT / "<output-name>" / f"<output-name>_tracks_distance.csv"),
        help=(
            "Input stereo distance CSV path. "
            "Default resolves from --output-name under outputs/stereo/artifacts."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to store the output distance plot image.",
    )
    parser.add_argument(
        "--image-name",
        default=None,
        help="Optional output image filename; defaults to <output-name>_distance_scatter.png.",
    )
    return parser.parse_args()


def resolve_csv_path(output_name: str, csv_path_template: str) -> Path:
    if "<output-name>" in csv_path_template:
        csv_path = Path(
            csv_path_template.replace("<output-name>", output_name)
        )
    else:
        csv_path = Path(csv_path_template)
    return csv_path


def load_class_distance_data(csv_path: Path) -> dict[int, list[tuple[int, float]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data: dict[int, list[tuple[int, float]]] = {0: [], 1: []}
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                class_id = int(row.get("class_id", "").strip())
                if class_id not in data:
                    continue
                if row.get("distance_valid", "").strip() not in {"1", "true", "True"}:
                    continue
                distance_raw = row.get("distance_m", "").strip()
                if not distance_raw:
                    continue
                distance = float(distance_raw)
                if not math.isfinite(distance):
                    continue
                frame_idx = int(row.get("frame_idx", "").strip())
            except (TypeError, ValueError):
                continue

            data[class_id].append((frame_idx, distance))
    return data


def plot_scatter(data: dict[int, list[tuple[int, float]]], output_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(14, 8))

    colors = {0: "#1f77b4", 1: "#ff7f0e"}
    markers = {0: "o", 1: "s"}

    has_any = False
    for class_id in sorted(data):
        points = data[class_id]
        if not points:
            continue
        has_any = True
        frames, dists = zip(*points)
        frames = list(frames)
        dists = list(dists)

        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")
        color = colors.get(class_id, "#666666")
        marker = markers.get(class_id, "o")

        ax.scatter(
            frames,
            dists,
            s=10,
            alpha=0.6,
            marker=marker,
            color=color,
            label=f"{class_name} points",
            edgecolor="none",
        )

        if dists:
            avg_d = sum(dists) / len(dists)
            ax.axhline(
                avg_d,
                color=color,
                linestyle="--",
                linewidth=1.4,
                alpha=0.9,
                label=f"{class_name} mean = {avg_d:.3f} m",
            )

            min_idx = min(range(len(dists)), key=dists.__getitem__)
            max_idx = max(range(len(dists)), key=dists.__getitem__)
            min_frame, min_dist = frames[min_idx], dists[min_idx]
            max_frame, max_dist = frames[max_idx], dists[max_idx]

            ax.scatter([min_frame], [min_dist], s=55, marker="^", color=color, edgecolors="black", zorder=4)
            ax.scatter([max_frame], [max_dist], s=55, marker="v", color=color, edgecolors="black", zorder=4)
            ax.annotate(
                f"min {min_dist:.2f}m",
                (min_frame, min_dist),
                textcoords="offset points",
                xytext=(6, -12),
                fontsize=9,
                color=color,
            )
            ax.annotate(
                f"max {max_dist:.2f}m",
                (max_frame, max_dist),
                textcoords="offset points",
                xytext=(6, 10),
                fontsize=9,
                color=color,
            )

    if not has_any:
        raise ValueError("No valid distance points found in CSV.")

    ax.set_title(f"Stereo Distance by Frame - {output_name}")
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Distance (m)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run():
    args = parse_args()
    csv_path = resolve_csv_path(args.output_name, args.csv_path)
    data = load_class_distance_data(csv_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_name = args.image_name or f"{args.output_name}_distance_scatter.png"
    output_path = output_dir / image_name
    plot_scatter(data, args.output_name, output_path)

    print("Distance scatter plot saved:")
    print(f"  CSV: {csv_path}")
    print(f"  IMG: {output_path}")


if __name__ == "__main__":
    run()
