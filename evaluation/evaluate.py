"""
evaluation/evaluate.py
Computes ATE (Absolute Trajectory Error) and RPE (Relative Pose Error)
for monocular and stereo VO trajectories against ground truth.

Metrics:
    ATE : global consistency — RMSE of position errors after alignment
    RPE : local consistency — RMSE of relative pose errors over fixed intervals
"""
import numpy as np
import os
import sys
import time
import yaml
from scipy.spatial.transform import Rotation
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# ─────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────

def load_T_cam_imu(camchain_path):
    """
    Load T_cam_imu (cam0) from a Kalibr camchain.yaml.

    Returns the 4x4 transform that maps IMU-frame coordinates to camera
    frame: x_cam = T_cam_imu @ x_imu. Used to convert TUM VI mocap GT
    (which lives in the IMU/body frame) into the camera frame so that RPE
    can compare like-for-like with our VO output.
    """
    with open(camchain_path) as f:
        data = yaml.safe_load(f)
    return np.array(data['cam0']['T_cam_imu'], dtype=np.float64)


def transform_gt_to_camera_frame(positions, quaternions, T_cam_imu):
    """
    Convert mocap GT poses from imu-in-world to cam-in-world.
        T_world_imu = [R_world_imu | p_world_imu]
        T_world_cam = T_world_imu @ inv(T_cam_imu)
    Without this step, RPE inflates by a similarity-transform residual
    proportional to the body-camera rotation offset (which is ~180° on
    TUM VI), even when the trajectory is otherwise correct.
    """
    T_imu_cam = np.linalg.inv(T_cam_imu)
    new_positions  = np.zeros_like(positions)
    new_quaternions = np.zeros_like(quaternions)

    for i, (p, q) in enumerate(zip(positions, quaternions)):
        T_world_imu = np.eye(4)
        T_world_imu[:3, :3] = Rotation.from_quat(q).as_matrix()
        T_world_imu[:3,  3] = p
        T_world_cam = T_world_imu @ T_imu_cam
        new_positions[i]   = T_world_cam[:3, 3]
        new_quaternions[i] = Rotation.from_matrix(T_world_cam[:3, :3]).as_quat()

    return new_positions, new_quaternions


def load_tum_trajectory(path):
    """
    Load TUM format trajectory.
    Returns:
        timestamps : np.array (N,)
        positions  : np.array (N, 3)
        quaternions: np.array (N, 4)  [qx qy qz qw]
    """
    timestamps, positions, quaternions = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            timestamps.append(float(p[0]))
            positions.append([float(p[1]), float(p[2]), float(p[3])])
            quaternions.append([float(p[4]), float(p[5]),
                                float(p[6]), float(p[7])])
    return (np.array(timestamps),
            np.array(positions),
            np.array(quaternions))


def load_ground_truth(mocap_csv):
    """
    Load TUM VI ground truth from mocap0/data.csv.
    CSV format: timestamp [ns], tx, ty, tz, qw, qx, qy, qz
    Returns timestamps in seconds, positions, quaternions [qx qy qz qw].
    """
    timestamps, positions, quaternions = [], [], []
    with open(mocap_csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            ts = float(p[0]) * 1e-9          # ns → seconds
            tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
            qw, qx, qy, qz = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            timestamps.append(ts)
            positions.append([tx, ty, tz])
            quaternions.append([qx, qy, qz, qw])  # reorder to [qx qy qz qw]
    return (np.array(timestamps),
            np.array(positions),
            np.array(quaternions))


def associate_trajectories(ts_est, ts_gt, max_diff=0.05):
    """
    Associate estimated and ground truth trajectories by timestamp.
    Returns indices into est and gt arrays for matched pairs.
    max_diff: maximum time difference in seconds to consider a match.
    """
    matches = []
    for i, t in enumerate(ts_est):
        diffs = np.abs(ts_gt - t)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            matches.append((i, j))
    if not matches:
        return np.array([]), np.array([])
    matches = np.array(matches)
    return matches[:, 0], matches[:, 1]


# ─────────────────────────────────────────────
# Umeyama alignment (rotation + translation + scale)
# ─────────────────────────────────────────────

def umeyama_alignment(src, dst, with_scale=True):
    """
    Align src trajectory to dst using Umeyama method.
    Finds s, R, t such that dst ≈ s * R * src + t

    Args:
        src       : (N, 3) estimated positions
        dst       : (N, 3) ground truth positions
        with_scale: True for mono VO (Sim3), False for stereo VO (SE3)

    Returns:
        src_aligned : (N, 3) aligned estimated positions
        s, R, t     : scale, rotation, translation
    """
    n = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    var_src = np.mean(np.sum(src_c ** 2, axis=1))
    if var_src < 1e-10:
        return src, 1.0, np.eye(3), mu_dst - mu_src

    W = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(W)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    R = U @ S @ Vt
    s = (np.sum(D * np.diag(S)) / var_src) if with_scale else 1.0
    t = mu_dst - s * R @ mu_src

    src_aligned = s * (R @ src_c.T).T + mu_dst
    return src_aligned, s, R, t


# ─────────────────────────────────────────────
# ATE — Absolute Trajectory Error
# ─────────────────────────────────────────────

def compute_ate(pos_est, pos_gt, with_scale=True):
    """
    Compute ATE after Umeyama alignment.

    Returns:
        rmse   : ATE RMSE in meters
        errors : per-pose position errors
        scale  : estimated scale factor
    """
    pos_aligned, s, R, t = umeyama_alignment(pos_est, pos_gt,
                                              with_scale=with_scale)
    errors = np.linalg.norm(pos_aligned - pos_gt, axis=1)
    rmse   = np.sqrt(np.mean(errors ** 2))
    return rmse, errors, s


# ─────────────────────────────────────────────
# RPE — Relative Pose Error
# ─────────────────────────────────────────────

def poses_to_matrices(positions, quaternions):
    """Convert position + quaternion arrays to list of 4x4 SE(3) matrices."""
    matrices = []
    for p, q in zip(positions, quaternions):
        T = np.eye(4)
        T[:3, :3] = Rotation.from_quat(q).as_matrix()
        T[:3,  3] = p
        matrices.append(T)
    return matrices


def compute_rpe(pos_est, quat_est, pos_gt, quat_gt, delta=10,
                ts=None, max_dt=None):
    """
    Compute RPE over fixed frame intervals.
    delta: frame interval for relative motion computation.

    If ts and max_dt are provided, pairs (i, i+delta) whose real-time
    separation exceeds max_dt seconds are skipped. This prevents a small
    number of pairs that span a GT blackout (e.g. the 12.5-min mocap gap
    on TUM VI outdoors5) from contributing relative-pose errors equal to
    the entire outdoor walk distance.

    Returns:
        trans_rmse, rot_rmse : RMSE values
        trans_errors, rot_errors : per-pair error arrays
    """
    T_est = poses_to_matrices(pos_est, quat_est)
    T_gt  = poses_to_matrices(pos_gt,  quat_gt)

    n = min(len(T_est), len(T_gt))
    trans_errors = []
    rot_errors   = []

    for i in range(0, n - delta, delta):
        if ts is not None and max_dt is not None:
            if ts[i + delta] - ts[i] > max_dt:
                continue
        T_est_rel = np.linalg.inv(T_est[i]) @ T_est[i + delta]
        T_gt_rel  = np.linalg.inv(T_gt[i])  @ T_gt[i + delta]
        T_err     = np.linalg.inv(T_gt_rel)  @ T_est_rel

        trans_err = np.linalg.norm(T_err[:3, 3])
        trans_errors.append(trans_err)

        R_err   = T_err[:3, :3]
        cos_val = np.clip((np.trace(R_err) - 1) / 2, -1, 1)
        rot_err = np.degrees(np.arccos(cos_val))
        rot_errors.append(rot_err)

    trans_errors = np.array(trans_errors)
    rot_errors   = np.array(rot_errors)

    if len(trans_errors) == 0:
        return float('nan'), float('nan'), trans_errors, rot_errors

    trans_rmse = np.sqrt(np.mean(trans_errors ** 2))
    rot_rmse   = np.sqrt(np.mean(rot_errors   ** 2))

    return trans_rmse, rot_rmse, trans_errors, rot_errors


def split_on_gap(ts_matched, gap_threshold_s=5.0):
    """
    Split matched-pair indices into contiguous segments wherever
    consecutive matched timestamps jump by more than gap_threshold_s.

    Returns a list of (a, b) inclusive index ranges into the matched
    arrays. Single-segment trajectories (e.g. room2) return one range.

    Used to evaluate ATE/RPE per-segment on sequences where the mocap
    GT has time gaps (corridor3, outdoors5 on TUM VI). Computing
    Umeyama over a gap conflates real VO error with drift accumulated
    in a blackout interval where GT does not exist.
    """
    n = len(ts_matched)
    if n == 0:
        return []
    diffs = np.diff(ts_matched)
    break_idx = np.where(diffs > gap_threshold_s)[0]
    if len(break_idx) == 0:
        return [(0, n - 1)]
    segments = []
    start = 0
    for b in break_idx:
        segments.append((start, int(b)))
        start = int(b) + 1
    segments.append((start, n - 1))
    return segments


def compute_path_length(positions):
    """Sum of consecutive Euclidean distances along the trajectory."""
    if len(positions) < 2:
        return 0.0
    diffs = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


# ─────────────────────────────────────────────
# Start-End Drift  (Eq. 8 from project guide)
# ─────────────────────────────────────────────

def compute_start_end_drift(pos_est, ts_est, pos_gt, ts_gt):
    """
    Eq. (8): align start pose, measure end-pose error.
    Used for corridor3 and outdoors5 (start+end GT only).

    Returns (drift_m, path_len_m, drift_pct) where path_len_m is the
    total traversed distance of the *estimated* trajectory and
    drift_pct = 100 * drift_m / path_len_m. The percent-of-path
    figure calibrates drift against well-known VO baselines
    (ORB-SLAM2 stereo ~1% on KITTI; classical VO without loop
    closure typically 3-5% on long outdoor runs).
    """
    ie, ig = associate_trajectories(ts_est, ts_gt)
    if len(ie) < 2:
        return None, None, None

    pos_est_m = pos_est[ie]
    pos_gt_m  = pos_gt[ig]

    # Align by translating start to GT start
    t_offset        = pos_gt_m[0] - pos_est_m[0]
    pos_est_shifted = pos_est_m + t_offset

    drift = float(np.linalg.norm(pos_est_shifted[-1] - pos_gt_m[-1]))

    # Path length over the full estimated trajectory (not just matched
    # pairs) -- this is what the VO actually integrated, and what we
    # divide the drift by to get a percent-of-path number.
    path_len = compute_path_length(pos_est)
    drift_pct = (100.0 * drift / path_len) if path_len > 0 else float('nan')

    return drift, path_len, drift_pct


# ─────────────────────────────────────────────
# Per-sequence evaluation
# ─────────────────────────────────────────────

def evaluate_sequence(seq_path, seq_name, results_dir="results"):
    """Evaluate mono and stereo VO for one sequence."""
    print(f"\n{'='*60}")
    print(f"Sequence: {seq_name}")
    print(f"{'='*60}")

    gt_csv = os.path.join(seq_path, "mav0", "mocap0", "data.csv")
    if not os.path.exists(gt_csv):
        print(f"  [WARN] No ground truth: {gt_csv}")
        return None

    ts_gt, pos_gt, quat_gt = load_ground_truth(gt_csv)
    print(f"  GT poses: {len(ts_gt)}")

    # Mocap is in IMU/body frame; our VO trajectories are in camera frame.
    # Transform GT to camera frame so RPE compares like-for-like.
    camchain_path = os.path.join(seq_path, "dso", "camchain.yaml")
    if os.path.exists(camchain_path):
        T_cam_imu = load_T_cam_imu(camchain_path)
        pos_gt, quat_gt = transform_gt_to_camera_frame(pos_gt, quat_gt, T_cam_imu)
        print(f"  GT transformed: imu-frame -> cam0 frame via T_cam_imu")
    else:
        print(f"  [WARN] No camchain at {camchain_path} — RPE-r may be inflated")

    results = {}

    for mode in ["mono", "stereo"]:
        traj_file = os.path.join(results_dir, f"{seq_name}_{mode}_traj.txt")
        if not os.path.exists(traj_file):
            print(f"  [{mode.upper()}] Not found: {traj_file}")
            continue

        ts_est, pos_est, quat_est = load_tum_trajectory(traj_file)
        n_frames = len(ts_est)
        print(f"\n  [{mode.upper()}] Poses: {n_frames}")

        # Timestamp association
        idx_est, idx_gt = associate_trajectories(ts_est, ts_gt)
        if len(idx_est) < 10:
            print(f"  [{mode.upper()}] Too few matches ({len(idx_est)}), skipping")
            continue

        pos_est_m  = pos_est[idx_est]
        pos_gt_m   = pos_gt[idx_gt]
        quat_est_m = quat_est[idx_est]
        quat_gt_m  = quat_gt[idx_gt]
        ts_est_m   = ts_est[idx_est]
        n_matched  = len(idx_est)
        print(f"  [{mode.upper()}] Matched pairs: {n_matched}")

        # Detect GT-coverage gaps. On TUM VI outdoors5/corridor3 the
        # mocap drops out for minutes while the user is outside the
        # tracking volume, so matched-pair timestamps jump by hundreds
        # of seconds between the "start" and "end" indoor clusters.
        # Computing a single Umeyama / RPE over both clusters lets
        # drift accumulated in the no-GT interval dominate every
        # metric. We split on a 5 s gap and report per-segment.
        segments = split_on_gap(ts_est_m, gap_threshold_s=5.0)
        has_gap = len(segments) > 1
        if has_gap:
            print(f"  [{mode.upper()}] GT gap detected -> {len(segments)} "
                  f"segments: {[(a, b, b - a + 1) for (a, b) in segments]}")

        # ATE — Sim3 for mono, SE3 for stereo
        with_scale = (mode == "mono")

        # Whole-trajectory ATE/RPE (kept for backward compat / room2)
        ate_rmse, _, scale = compute_ate(
            pos_est_m, pos_gt_m, with_scale=with_scale)
        rpe_trans, rpe_rot, _, _ = compute_rpe(
            pos_est_m, quat_est_m,
            pos_gt_m,  quat_gt_m,
            delta=10,
            ts=ts_est_m, max_dt=1.0)  # skip pairs spanning a GT gap

        # Per-segment ATE/RPE. For gapped sequences this is the
        # meaningful number; for ungapped sequences it duplicates the
        # whole-trajectory metric.
        seg_results = []
        for seg_id, (a, b) in enumerate(segments):
            seg_n = b - a + 1
            if seg_n < 10:
                continue
            seg_ate, _, seg_scale = compute_ate(
                pos_est_m[a:b + 1], pos_gt_m[a:b + 1],
                with_scale=with_scale)
            seg_rpe_t, seg_rpe_r, _, _ = compute_rpe(
                pos_est_m[a:b + 1],  quat_est_m[a:b + 1],
                pos_gt_m[a:b + 1],   quat_gt_m[a:b + 1],
                delta=10,
                ts=ts_est_m[a:b + 1], max_dt=1.0)
            seg_results.append({
                "seg_id"   : seg_id,
                "n"        : seg_n,
                "ate_rmse" : seg_ate,
                "rpe_trans": seg_rpe_t,
                "rpe_rot"  : seg_rpe_r,
                "scale"    : seg_scale,
            })

        # Start-end drift (Eq. 8) with path length + drift%
        drift, path_len, drift_pct = compute_start_end_drift(
            pos_est, ts_est, pos_gt, ts_gt)

        # Displacement comparison
        gt_disp  = np.linalg.norm(pos_gt_m[-1]  - pos_gt_m[0])
        est_disp = np.linalg.norm(pos_est_m[-1] - pos_est_m[0])

        print(f"  [{mode.upper()}] ATE RMSE          : {ate_rmse:.4f} m  "
              f"(whole-trajectory{' — UNRELIABLE: spans GT gap' if has_gap else ''})")
        if with_scale:
            print(f"  [{mode.upper()}] Scale factor      : {scale:.6f}")
        print(f"  [{mode.upper()}] RPE trans         : {rpe_trans:.4f} m  "
              f"(gap-aware)")
        print(f"  [{mode.upper()}] RPE rot           : {rpe_rot:.4f} deg "
              f"(gap-aware)")
        for s in seg_results:
            print(f"  [{mode.upper()}] seg{s['seg_id']} (n={s['n']}): "
                  f"ATE={s['ate_rmse']:.4f}  RPE-t={s['rpe_trans']:.4f}  "
                  f"RPE-r={s['rpe_rot']:.4f}")
        if drift is not None:
            print(f"  [{mode.upper()}] Start-End drift   : {drift:.4f} m  (Eq.8)")
            print(f"  [{mode.upper()}] Path length (est) : {path_len:.2f} m")
            print(f"  [{mode.upper()}] Drift %% of path   : {drift_pct:.2f}%")
        print(f"  [{mode.upper()}] GT displacement   : {gt_disp:.3f} m")
        print(f"  [{mode.upper()}] Est displacement  : {est_disp:.3f} m")

        results[mode] = {
            "ate_rmse"  : ate_rmse,
            "scale"     : scale,
            "rpe_trans" : rpe_trans,
            "rpe_rot"   : rpe_rot,
            "drift"     : drift if drift is not None else float('nan'),
            "path_len"  : path_len if path_len is not None else float('nan'),
            "drift_pct" : drift_pct if drift_pct is not None else float('nan'),
            "has_gap"   : has_gap,
            "segments"  : seg_results,
            "n_frames"  : n_frames,
            "n_matched" : n_matched,
            "gt_disp"   : gt_disp,
            "est_disp"  : est_disp,
        }

    return results


# ─────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────

def print_summary(all_results):
    """Print clean summary tables."""

    print(f"\n{'='*92}")
    print("FULL EVALUATION SUMMARY")
    print(f"{'='*92}")
    print(f"{'Sequence':<20} {'Mode':<8} {'ATE(m)':<9} {'RPE-t(m)':<10} "
          f"{'RPE-r(°)':<10} {'Drift(m)':<10} {'Drift%':<8} {'Scale'}")
    print(f"{'-'*92}")

    for seq_name, res in all_results.items():
        if res is None:
            continue
        short = seq_name.replace("dataset-", "").replace("_512_16", "")
        for mode in ["mono", "stereo"]:
            if mode not in res:
                continue
            r = res[mode]
            scale_str = f"{r['scale']:.5f}" if mode == "mono" else "metric"
            drift_str = f"{r['drift']:.3f}" if not np.isnan(r['drift']) else "N/A"
            dpct_str  = (f"{r['drift_pct']:.2f}%"
                         if not np.isnan(r['drift_pct']) else "N/A")
            print(f"{short:<20} {mode:<8} {r['ate_rmse']:<9.4f} "
                  f"{r['rpe_trans']:<10.4f} {r['rpe_rot']:<10.4f} "
                  f"{drift_str:<10} {dpct_str:<8} {scale_str}")

    print(f"\n{'='*92}")
    print("PER-SEGMENT METRICS  (sequences with a GT-coverage gap)")
    print(f"{'='*92}")
    print(f"{'Sequence':<20} {'Mode':<8} {'Seg':<5} {'n':<7} "
          f"{'ATE(m)':<9} {'RPE-t(m)':<10} {'RPE-r(°)':<10}")
    print(f"{'-'*92}")
    for seq_name, res in all_results.items():
        if res is None:
            continue
        short = seq_name.replace("dataset-", "").replace("_512_16", "")
        for mode in ["mono", "stereo"]:
            if mode not in res or not res[mode].get("has_gap"):
                continue
            for s in res[mode]["segments"]:
                print(f"{short:<20} {mode:<8} {s['seg_id']:<5} {s['n']:<7} "
                      f"{s['ate_rmse']:<9.4f} {s['rpe_trans']:<10.4f} "
                      f"{s['rpe_rot']:<10.4f}")

    print(f"\n{'='*80}")
    print("RPE IMPROVEMENT: Stereo vs Mono (translational)")
    print(f"{'='*80}")
    for seq_name, res in all_results.items():
        if res is None or "mono" not in res or "stereo" not in res:
            continue
        short = seq_name.replace("dataset-", "").replace("_512_16", "")
        imp = res["mono"]["rpe_trans"] / res["stereo"]["rpe_trans"]
        print(f"  {short:<20} Mono={res['mono']['rpe_trans']:.2f}m  "
              f"Stereo={res['stereo']['rpe_trans']:.2f}m  "
              f"Improvement={imp:.1f}x")
    print(f"{'='*80}")


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    sequences = {
        "dataset-room2_512_16"    : "data/dataset-room2_512_16",
        "dataset-corridor3_512_16": "data/dataset-corridor3_512_16",
        "dataset-outdoors5_512_16": "data/dataset-outdoors5_512_16",
    }

    all_results = {}
    for seq_name, seq_path in sequences.items():
        all_results[seq_name] = evaluate_sequence(
            seq_path, seq_name, results_dir="results"
        )

    print_summary(all_results)