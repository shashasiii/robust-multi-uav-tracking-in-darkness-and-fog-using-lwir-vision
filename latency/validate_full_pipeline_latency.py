"""Validate core stereo online-pipeline latency without timing data I/O/rendering.

The production stereo entry point performs the actual processing. Its latency
window starts after each stereo pair has been decoded/resized and ends after
dual-eye detection/tracking, V19 rules, stereo matching, optional temporal
compensation, depth, and the configured temporal distance filter. Output
rendering and writes still occur to keep the production path unchanged, but
they are outside the measured window.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARAMS = REPO_ROOT / "stereo" / "thermal_stereo_params.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "latency"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline and/or causal core-pipeline latency validation.",
    )
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--params-path", type=Path, default=DEFAULT_PARAMS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-name", default="rtx3070laptop_v34_core_latency")
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--warmup-frames", type=int, default=25)
    parser.add_argument(
        "--distance-filter-mode",
        choices=("distance_ema", "disparity_ema_reacquire"),
        default="distance_ema",
        help="Temporal distance-filter implementation included in the timed core.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("baseline", "causal"),
        default=("baseline", "causal"),
    )
    parser.add_argument(
        "--expected-gpu",
        default="RTX 3070 Laptop",
        help="Case-insensitive substring expected in the CUDA device name.",
    )
    parser.add_argument(
        "--allow-gpu-mismatch",
        action="store_true",
        help="Run even when the active CUDA device does not match --expected-gpu.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    if args.max_frames <= args.warmup_frames:
        raise ValueError("--max-frames must be greater than --warmup-frames")
    for path in (args.left_dir, args.right_dir, args.params_path):
        if not path.exists():
            raise FileNotFoundError(path)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the deployment-oriented latency validation")
    gpu_name = torch.cuda.get_device_name(0)
    if args.expected_gpu.lower() not in gpu_name.lower() and not args.allow_gpu_mismatch:
        raise RuntimeError(
            f"Active GPU is {gpu_name!r}, expected a name containing {args.expected_gpu!r}. "
            "Use --allow-gpu-mismatch only for an explicitly labeled alternative run."
        )
    return gpu_name


def run_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    run_name = f"{args.run_name}_{mode}"
    run_root = args.output_root / run_name
    latency_json = run_root / f"{run_name}_latency.json"
    image_root = run_root / "images"
    video_root = run_root / "artifacts"

    command = [
        sys.executable,
        str(REPO_ROOT / "stereo" / "tracking_stereo_distance_online.py"),
        "--output-name",
        run_name,
        "--left-dir",
        str(args.left_dir),
        "--right-dir",
        str(args.right_dir),
        "--model",
        str(args.model),
        "--params-path",
        str(args.params_path),
        "--output-img-root",
        str(image_root),
        "--output-video-root",
        str(video_root),
        "--fps",
        "25",
        "--max-frames",
        str(args.max_frames),
        "--latency-warmup-frames",
        str(args.warmup_frames),
        "--latency-report-path",
        str(latency_json),
        "--skip-distance-plot",
        "--distance-filter-mode",
        args.distance_filter_mode,
    ]
    if mode == "causal":
        command.append("--enable-causal-time-compensation")

    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "command.txt").write_text(
        subprocess.list2cmdline(command) + "\n",
        encoding="utf-8",
    )
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return json.loads(latency_json.read_text(encoding="utf-8"))


def write_summary(
    args: argparse.Namespace,
    gpu_name: str,
    reports: dict[str, dict[str, Any]],
) -> Path:
    output_path = args.output_root / f"{args.run_name}_summary.md"
    lines = [
        f"# Core Pipeline Latency: {args.run_name}",
        "",
        f"- GPU: `{gpu_name}`",
        "- Nominal thermal input: `25 FPS` (`40 ms` frame-pair budget)",
        f"- Frames requested per mode: `{args.max_frames}`",
        f"- Warm-up frames excluded: `{args.warmup_frames}`",
        f"- Distance-filter mode: `{args.distance_filter_mode}`",
        "- Timing scope: dual-eye YOLO/BoT-SORT, observation extraction, V19 rules, stereo matching, optional causal compensation, depth, and the configured temporal distance filter.",
        "- Excluded: file discovery/synchronization, image decode/resize, initialization, drawing, CSV/image/video/report writes, and plots.",
        "",
        "## Local RTX 3070 Laptop result",
        "",
        "| Mode | Frames | Mean ms | Median ms | P95 ms | P99 ms | Mean FPS | <=40 ms | Peak allocated MiB | Peak reserved MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, report in reports.items():
        lines.append(
            "| {mode} | {n} | {mean:.2f} | {median:.2f} | {p95:.2f} | {p99:.2f} | "
            "{fps:.2f} | {within:.1%} | {alloc:.1f} | {reserved:.1f} |".format(
                mode=mode,
                n=report["measured_frames"],
                mean=report["mean_ms"],
                median=report["median_ms"],
                p95=report["p95_ms"],
                p99=report["p99_ms"],
                fps=report["throughput_fps_from_mean"],
                within=report["within_25fps_budget_fraction"],
                alloc=report["gpu_peak_allocated_mib"],
                reserved=report["gpu_peak_reserved_mib"],
            )
        )

    lines.extend([
        "",
        "## Module latency attribution",
        "",
        "CUDA is synchronized at each module boundary for attribution. Shares use each module mean divided by the measured core mean; small boundary/timer overhead means shares may not sum to exactly 100%.",
    ])
    for mode, report in reports.items():
        lines.extend([
            "",
            f"### {mode}",
            "",
            "| Module | Mean ms | P95 ms | Share of core mean |",
            "|---|---:|---:|---:|",
        ])
        modules = sorted(
            report.get("module_latency", {}).items(),
            key=lambda item: item[1]["mean_ms"],
            reverse=True,
        )
        for name, values in modules:
            lines.append(
                f"| {name} | {values['mean_ms']:.2f} | {values['p95_ms']:.2f} | "
                f"{values['share_of_core_mean']:.1%} |"
            )

    lines.extend([
        "",
        "## Jetson interpretation boundary",
        "",
        "No whole-pipeline Jetson latency is extrapolated from the laptop result. Ultralytics Jetson tables measure standard-model inference only and exclude preprocessing, postprocessing, BoT-SORT, stereo logic, and output work. They may be used as an inference-layer reference, but deployment acceptance requires the custom checkpoint and complete pipeline to be benchmarked on the selected Orin model, power mode, JetPack/TensorRT version, precision, clocks, and thermal condition.",
        "",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    gpu_name = validate_args(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    reports = {mode: run_mode(args, mode) for mode in args.modes}
    summary_path = write_summary(args, gpu_name, reports)
    print(f"Latency summary: {summary_path}")


if __name__ == "__main__":
    main()
