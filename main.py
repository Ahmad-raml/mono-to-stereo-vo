"""
main.py — Entry point for monocular VO and stereo VO.

Usage:
    python main.py --mode mono   --sequence data/dataset-room2_512_16
    python main.py --mode stereo --sequence data/dataset-room2_512_16
    python main.py --mode stereo --sequence data/dataset-outdoors5_512_16
        (auto-applies the 'outdoors' preset: deeper Z range, more disparities,
         looser per-frame motion bounds)

Stereo parameters can be overridden from the CLI; any preset value is
overridden if its corresponding flag is passed.
"""
import argparse
import numpy as np
import os
import time

np.random.seed(42)

from utils.data_loader import TUMVILoader
from utils.calibration import Calibration
from mono_vo.mono_vo_pipeline import MonoVO
from stereo_vo.stereo_vo_pipeline import StereoVO


# ─────────────────────────────────────────────────────────────────────
# Per-sequence presets for stereo VO.
#
# 'indoor' is the default — tuned for room2/corridor3 (room-scale scenes,
# slow handheld motion, baseline ~10cm, fx ~191 px ⇒ Z = fx*B/d ≈ 19m at d=1).
#
# 'outdoors' loosens depth/disparity validity and motion bounds because
# outdoors5 has scene depth well beyond 20m and faster locomotion.
# ─────────────────────────────────────────────────────────────────────
STEREO_PRESETS = {
    'indoor': dict(
        max_depth=20.0,
        min_disp_valid=1.0,
        num_disparities=64,
        block_size=7,
        max_step_rot_deg=25.0,
        max_step_trans=0.50,
        min_inlier_ratio=0.25,
        # Keyframe spacing for indoor handheld at 20 Hz: room-scale motion
        # rarely exceeds 30 cm or 12 deg between keyframes before tracking
        # quality drops. Empirically gives ~1 KF per 5-10 frames.
        min_kf_features=200,
        max_kf_trans=0.30,
        max_kf_rot_deg=12.0,
    ),
    'outdoors': dict(
        # Tightened depth/disparity gating: although outdoor scenes contain
        # very far points, those have huge depth uncertainty
        # (delta_Z = Z^2 * delta_d / (fx*B); at Z=30m, 0.5px disparity error
        # gives ~23m depth error). Including them in PnP biases the pose
        # estimate even though the inliers fit well. Restricting to Z<=20m
        # drops the noisy far tail and dramatically reduces drift.
        max_depth=20.0,
        min_disp_valid=1.0,
        num_disparities=128,    # search range still wide for SGBM matching
        block_size=9,
        max_step_rot_deg=30.0,
        max_step_trans=1.50,
        # Outdoor scenes have moving objects (pedestrians, cars). A coherent
        # ~25% inlier set can come from a parallel-rigid-motion outlier, so
        # require >=30% to accept. The OK distribution sits at median 0.93,
        # so this catches degenerate cases without false positives.
        min_inlier_ratio=0.30,
        # Outdoor walking covers ~0.5-1 m/s, so keyframe budgets are larger
        # than indoor; rotation budget slightly higher to absorb head turns.
        min_kf_features=200,
        max_kf_trans=0.60,
        max_kf_rot_deg=15.0,
    ),
}


def pick_preset(sequence_path):
    """Detect a preset from the sequence folder name."""
    name = os.path.basename(sequence_path).lower()
    if 'outdoors' in name:
        return 'outdoors'
    return 'indoor'


def merge_overrides(preset_dict, args):
    """CLI flags override the preset value for any flag the user passed."""
    cfg = dict(preset_dict)
    overrides = {
        'max_depth':        args.max_depth,
        'min_disp_valid':   args.min_disp,
        'num_disparities':  args.num_disp,
        'block_size':       args.block_size,
        'max_step_rot_deg': args.max_step_rot,
        'max_step_trans':   args.max_step_trans,
        'min_inlier_ratio': args.min_inlier_ratio,
        'max_reproj_rmse':  args.max_reproj_rmse,
        'min_kf_features':  args.min_kf_features,
        'max_kf_trans':     args.max_kf_trans,
        'max_kf_rot_deg':   args.max_kf_rot,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Mono/Stereo VO on TUM VI")
    parser.add_argument('--mode',     choices=['mono', 'stereo'], default='mono')
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output',   default='results/')
    parser.add_argument('--frames',   type=int, default=None)
    parser.add_argument('--method',   choices=['ORB', 'SIFT'], default='ORB')
    parser.add_argument('--disp',     choices=['SGBM', 'BM'], default='SGBM')
    parser.add_argument('--preset',   choices=list(STEREO_PRESETS.keys()), default=None,
                        help="Stereo preset (default: auto-detected from sequence name)")
    # Stereo overrides — all optional; preset is used otherwise
    parser.add_argument('--max-depth',         type=float, default=None)
    parser.add_argument('--min-disp',          type=float, default=None,
                        help='Min disparity to consider valid (px)')
    parser.add_argument('--num-disp',          type=int,   default=None,
                        help='SGBM/BM numDisparities (rounded to multiple of 16)')
    parser.add_argument('--block-size',        type=int,   default=None)
    parser.add_argument('--max-step-rot',      type=float, default=None,
                        help='Max per-frame rotation step (deg) before reject')
    parser.add_argument('--max-step-trans',    type=float, default=None,
                        help='Max per-frame translation step (m) before reject')
    parser.add_argument('--min-inlier-ratio',  type=float, default=None,
                        help='Min PnP-inlier / matches ratio to accept pose')
    parser.add_argument('--max-reproj-rmse',   type=float, default=None,
                        help='Max inlier reprojection RMSE (px) to accept pose '
                             '(default 1.0 = half of RANSAC threshold)')
    parser.add_argument('--min-kf-features',   type=int,   default=None,
                        help='Re-keyframe if alive features drop below this')
    parser.add_argument('--max-kf-trans',      type=float, default=None,
                        help='Re-keyframe if translation from KF exceeds this (m)')
    parser.add_argument('--max-kf-rot',        type=float, default=None,
                        help='Re-keyframe if rotation from KF exceeds this (deg)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    loader   = TUMVILoader(args.sequence)
    camchain = os.path.join(args.sequence, "dso", "camchain.yaml")
    calib    = Calibration(camchain)
    seq_name = os.path.basename(args.sequence)

    if args.mode == 'mono':
        output_path = os.path.join(args.output, f"{seq_name}_mono_traj.txt")
        vo = MonoVO(calib, method=args.method)

    elif args.mode == 'stereo':
        output_path = os.path.join(args.output, f"{seq_name}_stereo_traj.txt")
        preset_name = args.preset or pick_preset(args.sequence)
        cfg = merge_overrides(STEREO_PRESETS[preset_name], args)
        print(f"[main] Stereo preset: {preset_name}")
        print(f"[main] Stereo config: {cfg}")
        vo = StereoVO(
            calib,
            method=args.method,
            disp_method=args.disp,
            **cfg,
        )

    n = min(len(loader), args.frames) if args.frames else len(loader)
    start = time.time()
    vo.run(loader, output_path, max_frames=args.frames)
    elapsed = time.time() - start

    ms_per_frame = (elapsed / n * 1000.0) if n > 0 else 0.0
    fps          = (n / elapsed) if elapsed > 0 else 0.0

    print(
        f"\n[Runtime] "
        f"Sequence: {seq_name} | "
        f"Mode: {args.mode} | "
        f"Frames: {n} | "
        f"Total: {elapsed:.2f}s | "
        f"Per-frame: {ms_per_frame:.1f}ms | "
        f"FPS: {fps:.2f}"
    )


if __name__ == "__main__":
    main()
