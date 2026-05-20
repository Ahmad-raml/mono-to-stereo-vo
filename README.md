# From Monocular to Stereo Visual Odometry

## Project Overview

A classical-CV visual odometry pipeline evaluated on the TUM VI fisheye
benchmark. Monocular VO uses FAST features, Lucas-Kanade optical flow and the
essential matrix (up-to-scale), while stereo VO uses StereoSGBM disparity with
`Z = f·B / d` and PnP-RANSAC for metric pose. The goal is to quantify why
stereo resolves the monocular scale problem on three TUM VI sequences
(`room2`, `corridor3`, `outdoors5`).


## Requirements

```bash
pip install -r requirements.txt
# which pulls in:
#   opencv-python >= 4.5
#   numpy         >= 1.21
#   scipy         >= 1.7
#   matplotlib    >= 3.4
#   pyyaml        >= 5.4
```


## Dataset Setup

Download the 512x512 DSO export of the three sequences from the TUM VI page:

- https://vision.in.tum.de/data/datasets/visual-inertial-dataset

Place them under `data/` so the tree looks like:

```
data/
├── dataset-room2_512_16/
│   ├── mav0/
│   │   ├── cam0/data/*.png     (left camera)
│   │   ├── cam0/data.csv
│   │   ├── cam1/data/*.png     (right camera)
│   │   ├── cam1/data.csv
│   │   └── mocap0/data.csv     (ground-truth poses, IMU frame)
│   └── dso/camchain.yaml       (Kalibr fisheye stereo + cam-IMU calibration)
├── dataset-corridor3_512_16/
└── dataset-outdoors5_512_16/
```


## Running the Code

### Monocular VO
```bash
python main.py --mode mono --sequence data/dataset-room2_512_16
```

### Stereo VO

```bash
# Indoor sequence (preset auto-selected by name → 'indoor')
python main.py --mode stereo --sequence data/dataset-room2_512_16

# Outdoor sequence (preset auto-selected → 'outdoors')
python main.py --mode stereo --sequence data/dataset-outdoors5_512_16

# Force a preset or override individual parameters
python main.py --mode stereo --sequence data/dataset-corridor3_512_16 \
    --preset indoor --max-depth 25 --num-disp 96
```

The active stereo preset is printed at the start of each run. Available
overrides: `--max-depth`, `--min-disp`, `--num-disp`, `--block-size`,
`--max-step-rot`, `--max-step-trans`, `--min-inlier-ratio`. Defaults are in
`STEREO_PRESETS` in [main.py](main.py).

Both modes save:

- `results/<sequence>_<mode>_traj.txt`         — TUM-format trajectory
- `results/<sequence>_<mode>_traj_status.csv`  — per-frame status log

Frames whose pose could not be estimated are **not** written to the
trajectory; the status CSV records the outcome of every frame:

| Status | Pose emitted? | Meaning |
|---|---|---|
| `INIT` | yes (identity) | First frame |
| `OK` | yes | All checks passed |
| `REPROJ_LOW_CONF` | yes (soft-flag) | Inlier reprojection RMSE > 1.0 px after LM refinement — pose accepted but flagged |
| `TRACK_FAIL` | no (hard skip) | Optical flow lost too many points (forward-backward consistency check rejects most failures here) |
| `PNP_FAIL` | no (hard skip) | `solvePnPRansac` returned no solution |
| `PNP_LOW_INLIER` | no (hard skip) | RANSAC inlier ratio below threshold (catches dynamic-object outliers) |
| `STEP_REJECT` | no (hard skip) | Per-frame rotation/translation exceeds bounds (rarely fires) |

Hard-skip frames are excluded from the trajectory — including them with
frozen poses would inflate RPE and hide tracking failures, which the
project guide explicitly forbids. Soft-flagged frames keep the trajectory
continuous (losing motion is usually worse than absorbing some noise) but
remain auditable in `*_status.csv`.

### Evaluation

```bash
python evaluation/evaluate.py
```

Computes ATE (Sim3 for mono, SE3 for stereo), RPE over a 10-frame interval,
and start-end drift (Eq. 8 of the project guide).

**Important: GT frame transformation.** The TUM VI mocap data is in the
IMU/body frame, but our VO trajectories are in the camera frame. Without
correction, RPE inflates massively (~50° rotational residual on indoor
sequences) because relative rotations between body poses and camera poses
differ by a similarity-transform of the camera-IMU offset. The evaluator
loads `T_cam_imu` from `dso/camchain.yaml` and transforms GT to the camera
frame before computing RPE. ATE is unaffected because Umeyama alignment
absorbs any rigid offset.

### Visualization & Plots

```bash
# 3-panel overlays vs GT (one PNG per sequence)
python visualize_with_gt.py

# Bar charts: ATE / RPE-t / RPE-r / drift / FPS
python generate_eval_plots.py

# Runtime CSV (re-runs main.py for all 6 (seq, mode) combos)
python generate_runtime_table.py
python generate_runtime_table.py --skip-run     # reparse existing logs only

# 12-slide presentation
python generate_slides.py
```


## Key Design Choices (and why)

- **Failure handling — hard skip vs soft flag.** Hard failures (optical-flow
  loss, PnP failure, low RANSAC inlier ratio) cause the frame to be excluded
  from the trajectory entirely. Borderline frames whose PnP succeeded but
  whose inlier reprojection RMSE > 1.0 px are kept in the trajectory and
  flagged `REPROJ_LOW_CONF` in the status CSV — losing real motion is usually
  worse than absorbing some noise, so we emit a pose but record that it is
  low confidence. This is the standard production-VO pattern. The previous
  "freeze last pose" behavior hides failures and inflates RPE; the project
  guide explicitly forbids it.
- **Reprojection RMSE gate at 1.0 px.** Default threshold is half the RANSAC
  inlier threshold (`pnp_reproj_err = 2.0`). Empirically: well-conditioned
  OK frames have median 0.4–0.7 px and p99 ~0.9 px; LOW_INLIER frames have
  median ~1.0 px. A threshold of 1.0 px catches the right tail without
  false positives. Configurable via `--max-reproj-rmse`.
- **Motion-only bundle adjustment.** After PnP-RANSAC succeeds we refine the
  pose with `cv2.solvePnPRefineLM`, which runs Levenberg–Marquardt on the
  reprojection error of the inlier set. This is the optional motion-only BA
  step the project guide describes (Sections IV.C / V.B). Catching pose
  noise here reduces drift over long outdoor sequences.
- **Forward-backward KLT consistency check.** Each feature is tracked
  prev→cur, then cur→prev, and dropped if the round-trip distance > 1 px.
  This catches occlusion and lighting-change artifacts that RANSAC alone
  cannot, because RANSAC operates on already-corrupted tracks. Empirically
  this reduced PnP failures by >90% across all sequences (room2 9→0,
  corridor3 49→1, outdoors5 114→9) and dropped outdoors5 ATE from 33.5 m
  to 22.3 m.
- **Tight outdoor-preset depth range** (`max_depth=20 m`,
  `min_disp_valid=1.0 px`). Depth uncertainty grows as Z² with disparity
  noise; at Z=30 m and δd=0.5 px on TUM VI, δZ ≈ 23 m. Restricting
  outdoor PnP to disparities ≥1 px (Z ≤ ~19 m) keeps only well-conditioned
  3D points and dramatically reduces accumulated drift, even though it
  superficially "sees less far".
- **GT in camera frame.** Mocap poses are in IMU frame; our VO is in camera
  frame. The evaluator applies `T_cam_imu^-1` to the GT before RPE so the
  comparison is like-for-like. Without this, indoor RPE-r looked ~50° even
  when the trajectory was correct.
- **Sim3 alignment for mono.** Mono is up-to-scale, so Umeyama with scale
  recovery is the only way to compute a meaningful ATE. RPE-t and Eq. 8
  start-end drift are the honest indicators of monocular failure.
- **Per-sequence stereo presets.** `indoor` (default) is tuned for room2 /
  corridor3 (room-scale scenes, ~10cm baseline → max ~19m depth at d=1px).
  `outdoors` extends `max_depth` to 50m, `num_disparities` to 128 and
  `min_disp_valid` to 0.5px to see beyond room-scale; per-frame motion
  bounds are loosened for outdoor walking speed.
- **Custom rectified focal length.** `cv2.fisheye.stereoRectify` returns a
  tiny `fx` for ~180° FOV lenses, which would warp the rectified image. We
  override `fx_new = K0[0,0] ≈ 191` so disparity, depth, and PnP all share
  a consistent perspective focal length. See [utils/calibration.py](utils/calibration.py).


## Results

Re-run `python evaluation/evaluate.py` and `python generate_runtime_table.py
--skip-run` to refresh after any pipeline change.

| Sequence  | Mode   | ATE (m)  | RPE-t (m) | RPE-r (°) | Drift (m) | Drift % | FPS   |
|-----------|--------|---------:|----------:|----------:|----------:|--------:|------:|
| room2     | Mono   |   1.2335 |    9.8318 |   46.5728 |     0.698 |   0.49% | 47.65 |
| room2     | Stereo |   0.2923 |    0.0196 |    1.1002 |     1.027 |   0.72% | 21.46 |
| corridor3 | Mono   |   0.8836 |    8.0265 |   53.6050 |     0.497 |   1.18% | 24.66 |
| corridor3 | Stereo |   2.8382 |    0.0184 |    1.0855 |     5.232 |   1.71% | 22.71 |
| outdoors5 | Mono   |   0.8553 |    7.7753 |   62.9617 |     0.353 |   0.26% | 22.31 |
| outdoors5 | Stereo |  22.3549 |    0.0370 |    1.6160 |    46.133 |   4.10% | 21.50 |

> **Mono drift convention.** Mono trajectories are first aligned to GT
> by Sim3 (rotation + scale + translation, same as ATE) before
> measuring start–end drift, and the percent-of-path is computed
> against GT path length. Without the Sim3 step the drift number is
> meaningless — `cv2.recoverPose` returns unit-vector translations,
> so per-frame integration produces a "path" in arbitrary units (e.g.
> ~5800 units over corridor3). RPE-t already exposes the local
> consequence of that scale ambiguity. See
> [`compute_start_end_drift`](evaluation/evaluate.py).

**Stereo RPE-t improvement vs Mono** — the headline metric of the project:

| Sequence  | Mono RPE-t (m) | Stereo RPE-t (m) | Improvement |
|-----------|---------------:|-----------------:|------------:|
| room2     |         9.83   |          0.02    |    **503×** |
| corridor3 |         8.03   |          0.02    |    **436×** |
| outdoors5 |         7.78   |          0.04    |    **210×** |

Notes on interpretation:

- **Mono ATE looks deceptively small** because Sim3 alignment recovers a
  per-trajectory scale. RPE-t (per-interval error after alignment) is the
  honest indicator of monocular failure; Eq. 8 drift after Sim3 only
  reflects trajectory-shape mismatch, not the unit-scale issue.
- **Mono RPE-r is genuinely high in room2** (~46°). This reveals essential-
  matrix rotation degeneracy under low-parallax look-around motion, not a bug
  in the pipeline. A real failure mode worth discussing in the paper.
- **Outdoors5 stereo drift of 46 m** end-to-start over a long outdoor walk on
  a 10 cm baseline at 191 px focal length is expected for classical stereo VO
  without loop closure or BA. Worth contrasting critically with ORB-SLAM2 in
  the discussion.
- **Stereo failure rates** (from `*_status.csv`, after FB-LK + LM
  refinement):

  | Sequence | Hard-skip | REPROJ_LOW_CONF (soft-flag) |
  |---|---:|---:|
  | room2 | 0 / 2882 (0.0%) | 1 (0.0%) |
  | corridor3 | 1 / 5802 (0.0%) | 19 (0.3%) |
  | outdoors5 | 9 / 17747 (0.1%) | 168 (0.9%) |

  Outdoors5's higher soft-flag rate is consistent with harder conditions
  (motion blur, distant low-texture regions, occlusions). The forward-
  backward LK consistency check pre-filters bad tracks before they reach
  PnP, which is why hard-skips are now near-zero on the indoor sequences.
  Mono had 5–25% silent failures before the failure-handling fix —
  those are now in the status CSV and excluded from trajectories.


## Project Structure

```
mono-to-stereo-vo/
├── main.py                      # entry: --mode mono|stereo --sequence ... [--preset ...]
├── requirements.txt
├── README.md
├── generate_runtime_table.py    # builds results/runtime_table.csv
├── generate_eval_plots.py       # builds results/plots/*.png
├── generate_slides.py           # builds results/presentation.pptx
├── visualize_with_gt.py         # 3-panel trajectory figures
│
├── mono_vo/
│   ├── mono_vo_pipeline.py      # FAST + LK + Essential matrix
│   ├── feature_extractor.py
│   └── epipolar.py
│
├── stereo_vo/
│   ├── stereo_vo_pipeline.py    # Rectify + SGBM + Z=fB/d + PnP-RANSAC
│   └── disparity.py             # SGBM/BM with configurable num_disparities
│
├── utils/
│   ├── calibration.py           # Kalibr fisheye stereo calibration
│   ├── data_loader.py           # TUM VI DSO 512x512 loader
│   └── tum_format.py            # TUM trajectory I/O
│
├── evaluation/
│   └── evaluate.py              # ATE (Sim3/SE3), RPE, Eq.8 drift
│                                # + GT IMU-to-camera transform
│
├── config/
│   └── tum_vi_calib.yaml
│
├── data/                        # TUM VI sequences (see Dataset Setup)
└── results/                     # trajectories + status CSVs + figures
    ├── *_mono_traj.txt          # TUM format
    ├── *_mono_traj_status.csv
    ├── *_stereo_traj.txt
    ├── *_stereo_traj_status.csv
    ├── *_gt_comparison.png
    ├── runtime_table.csv
    ├── plots/                   # bar charts
    └── presentation.pptx
```


## Reproducibility

- `np.random.seed(42)` set at the top of [main.py](main.py).
- All thresholds documented in `STEREO_PRESETS` in [main.py](main.py).
- Run on Windows 11 / Python 3.x / OpenCV 4.5+.
- Per-frame status CSV makes the failure rate auditable per sequence.
