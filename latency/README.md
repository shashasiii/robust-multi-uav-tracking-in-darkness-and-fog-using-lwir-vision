# Full-Pipeline Core Latency Validation

`validate_full_pipeline_latency.py` measures the production stereo pipeline on the local CUDA GPU while excluding unrelated data-processing time from the metric.

Included in each timed frame pair:

- left/right YOLO26n and BoT-SORT;
- tensor-to-CPU observation extraction;
- left/right V19 online causal rules;
- same-class Hungarian stereo matching;
- optional causal exposure-time compensation;
- rectified disparity, depth, and the explicitly selected temporal distance filter.

Excluded from the timer:

- file discovery and filename synchronization;
- image read/decode and resize;
- model/geometry initialization and warm-up;
- drawing and compositing;
- CSV, image, video, report, and plot writes.

The production script still creates normal artifacts outside the timed window so the benchmark uses the same processing path. The default validation runs both the uncompensated baseline and causal-compensation mode on 250 V34 frame pairs, excludes the first 25 pairs, and checks for an RTX 3070 Laptop GPU.

The JSON and Markdown summary record `distance_filter_mode` and break the core window into left/right preprocessing, YOLO forward, YOLO postprocess, BoT-SORT update, tracking-call overhead, observation extraction, temporal history, V19 rules, state lifecycle, and `stereo_match_depth_filter`. BoT-SORT is further split into sparse-optical-flow GMC and association/state management. CUDA is synchronized at each GPU boundary for attribution, so the breakdown can have small synchronization overhead relative to the single end-to-end core timer.

```powershell
Set-Location D:\PycharmProjects\CEG_Project_2026
D:\Python\dl_proj_env\python.exe latency\validate_full_pipeline_latency.py `
  --left-dir E:\path\to\v34\left_thermal `
  --right-dir E:\path\to\v34\right_thermal `
  --model weights\v8_yolo26n_e300_b128\weights\best.pt `
  --run-name v7_candidate_dispema `
  --distance-filter-mode disparity_ema_reacquire
```

For a same-session compatibility control, repeat with a distinct `--run-name`
and `--distance-filter-mode distance_ema`. Do not attribute the difference
between two sequential whole-core means to the filter; compare the directly
timed `stereo_match_depth_filter` stage and report detector/GPU run variation.

Outputs are written under `outputs/latency/`. The summary compares local mean/P95 latency against the `40 ms` budget implied by the thermal camera's nominal `25 FPS`.

The validator no longer extrapolates whole-pipeline Orin latency from an assumed fraction of laptop throughput. Official Jetson figures may only be compared with the isolated YOLO inference component. Before a deployment claim, rerun the custom checkpoint and full pipeline on the selected Orin model and record its power mode, JetPack/TensorRT version, precision, clocks, and thermal condition.

`profile_yolo_forward.py` separately measures the fused PyTorch backend at the actual tensor shape in FP32 and FP16, without preprocessing, NMS, tracking, or stereo processing.
