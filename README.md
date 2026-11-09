# Robust Multi-UAV Tracking in Darkness and Fog Using Long-Wave Infrared (LWIR) Vision

This repository is the minimal public implementation accompanying the CEG5003
project of the same name. It provides an online-causal dual-LWIR pipeline for
YOLO26n detection, independent BoT-SORT tracking, precision-first Display-ID
inheritance, timestamp-aware stereo synchronization, same-class Hungarian
assignment, rectification, forward-depth estimation, optional causal
recorded-timestamp-midpoint compensation, and auditable machine-readable
outputs. Stereo synchronisation searches the complete causal tolerance window,
and same-class assignment maximises feasible match cardinality before minimising
cost.

Release v3.0 adds disparity-domain EMA with causal three-sample controlled
reacquisition and replaces the ambiguous output name `distance_m` with
`filtered_depth_m`. The legacy metre-domain EMA remains available through the
explicit filter mode for compatibility with historical experiments.

The public repository is deliberately smaller than the research workspace. It
contains the stable runtime code, configurations, latency tools, and synthetic
unit tests. It does **not** contain the original RGB-D-T recordings, Roboflow
exports, trained weights, experiment runs, report sources, private evidence
snapshots, or internal project-management documents.

## Included files

- `detection/test_yolo_pt.py`: detector validation and inference entry point.
- `tracking/track_folder_to_video_online_postprocess_stable_v19.py`: frozen
  implementation of precision-first online Display-ID inheritance. `V19` is a
  version identifier, not the method name.
- `tracking/botsort_uav_stable.yaml`: retained BoT-SORT configuration.
- `stereo/tracking_stereo_distance_online.py`: end-to-end dual-LWIR pipeline.
- `stereo/thermal_stereo_params.yaml`: calibration schema and retained project
  calibration values; users should replace these values for another rig.
- `distance_baseline/compare_realsense_reference.py`: independent-reference
  alignment and quality-control functions.
- `latency/`: detector and end-to-end latency measurement tools.
- `tests/`: synthetic timestamp, lexicographic-assignment, compensation, and
  reference-QC tests, including disparity-EMA reacquisition state transitions.

## Environment

The tested Windows/CUDA environment is defined in `environment.yml`:

```powershell
conda env create -f environment.yml
conda activate ceg-lwir
python -m unittest discover -s tests -v
```

The retained tests construct their own small synthetic inputs and do not need
the restricted project dataset or model weights. The verified reference system
used Python 3.10, PyTorch 2.9.1 with CUDA 12.6, Ultralytics 8.4.2, and an NVIDIA
RTX 3070 Laptop GPU.

## Input and model preparation

Thermal frames used in the project are three-channel PNG files whose channel
values are identical. This preserves compatibility with the standard
three-channel YOLO input and COCO-pretrained weights without changing the input
layer. A single-channel training route is also supported by Ultralytics dataset
configuration, but it was not evaluated in this project.

Place an authorised checkpoint anywhere outside Git history and pass its path
with `--model`. Do not commit weights unless their redistribution rights are
known. Stereo filenames must contain timestamps in the retained form, for
example:

```text
thermal_inferno_image_<seconds>_<nanoseconds>.png
thermal_grayscale_image_<seconds>_<nanoseconds>.png
```

Left and right folders are supplied explicitly; no personal absolute paths are
embedded in the public commands.

## Single-eye tracking and Display-ID inheritance

```powershell
python tracking/track_folder_to_video_online_postprocess_stable_v19.py `
  --output-name demo_tracking `
  --source-dir <left-thermal-folder> `
  --model <checkpoint.pt>
```

Outputs default to `outputs/tracking/`. Raw Track IDs remain immutable;
Display-IDs are a separate presentation-layer namespace. Ambiguous inheritance
candidates are rejected rather than merged.

## End-to-end stereo depth

```powershell
python stereo/tracking_stereo_distance_online.py `
  --output-name demo_stereo `
  --left-dir <left-thermal-folder> `
  --right-dir <right-thermal-folder> `
  --model <checkpoint.pt> `
  --params-path stereo/thermal_stereo_params.yaml `
  --enable-causal-time-compensation `
  --distance-filter-mode disparity_ema_reacquire
```

Outputs default to `outputs/stereo/`. The command emits annotated frames, a
video, a tracking/depth CSV, a compact run report, and—unless disabled—a depth
plot. The CSV keeps raw geometric `depth_m` separate from temporally filtered
`filtered_depth_m`; it also records the filter mode, filtered disparity,
candidate count, Raw-ID-change flag, and completed-reacquisition flag.
Calibration values are rig-specific and must not be reused blindly.

## Detector and latency checks

Detector inference:

```powershell
python detection/test_yolo_pt.py --model <checkpoint.pt> `
  --run_mode predict_dir --source_dir <image-folder>
```

Pure detector latency:

```powershell
python latency/profile_yolo_forward.py --model <checkpoint.pt>
```

End-to-end latency:

```powershell
python latency/validate_full_pipeline_latency.py `
  --left-dir <left-thermal-folder> `
  --right-dir <right-thermal-folder> `
  --model <checkpoint.pt>
```

Latency results depend on hardware, software versions, warm-up policy, and
timing scope and should not be compared without those conditions.

## Data, weights, and evidence boundaries

The original RGB-D-T data were provided by Dr. Huai Jianzhu and are not
redistributed here. The project checkpoint is likewise excluded until its
training-data and redistribution status has been confirmed. Users can run the
synthetic tests immediately, but full inference requires independently
authorised timestamped thermal frames, compatible model weights, and valid
camera calibration.

RealSense measurements in the project are an independent reference rather than
surveyed physical ground truth. The retained identity audit is a bounded
two-identity event reconstruction, not an open-scene MOT benchmark.

## Citation and attribution

If this code supports academic work, cite the associated report:

> Deng, H. (2026). *Robust Multi-UAV Tracking in Darkness and Fog Using
> Long-Wave Infrared (LWIR) Vision*. CEG5003 Project Report, National University
> of Singapore.

Project integration, experiments, analysis, and report preparation were
performed by Deng Haonan. The project uses Ultralytics, PyTorch, OpenCV, SciPy,
NumPy, and other open-source tools. Roboflow was used for dataset annotation and
validation support.

## Licence

This project is released under `AGPL-3.0-only`; see `LICENSE`. This choice is
consistent with the open-source licensing requirements of the Ultralytics YOLO
dependency used by the pipeline. Commercial or closed-source integration may
require a separate Ultralytics Enterprise License. The licence applies to this
repository's source code and does not grant rights to the excluded provider
data, third-party datasets, or model weights.
