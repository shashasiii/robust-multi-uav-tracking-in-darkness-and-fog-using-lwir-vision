"""Profile pure YOLO backend forward latency without tracking or postprocess."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch
from ultralytics import __version__ as ultralytics_version
from ultralytics.nn.autobackend import AutoBackend


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = REPO_ROOT / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def measure(model_path: Path, shape: tuple[int, int, int, int], half: bool, warmup: int, iterations: int) -> dict:
    device = torch.device("cuda:0")
    backend = AutoBackend(model_path, device=device, fp16=half, fuse=True, verbose=False)
    tensor = torch.rand(shape, device=device, dtype=torch.float16 if half else torch.float32)
    backend.warmup(imgsz=shape)
    with torch.inference_mode():
        for _ in range(warmup):
            backend(tensor)
        torch.cuda.synchronize()
        values: list[float] = []
        for _ in range(iterations):
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            started.record()
            backend(tensor)
            finished.record()
            finished.synchronize()
            values.append(float(started.elapsed_time(finished)))
    return {
        "precision": "FP16" if half else "FP32",
        "iterations": iterations,
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    shape = (1, 3, args.height, args.width)
    report = {
        "model": str(args.model.resolve()),
        "backend": "PyTorch AutoBackend",
        "input_shape": list(shape),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "ultralytics": ultralytics_version,
        "results": [
            measure(args.model, shape, False, args.warmup, args.iterations),
            measure(args.model, shape, True, args.warmup, args.iterations),
        ],
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
