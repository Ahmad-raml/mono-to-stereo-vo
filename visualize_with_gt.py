"""
visualize_with_gt.py - Fixed version
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def load_tum(path):
    if not os.path.exists(path):
        return None, None
    traj, ts = [], []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            ts.append(float(p[0]))
            traj.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(traj), np.array(ts)


def load_gt(mocap_csv):
    if not os.path.exists(mocap_csv):
        return None, None
    traj, ts = [], []
    with open(mocap_csv) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            ts.append(float(p[0]) * 1e-9)
            traj.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(traj), np.array(ts)


def associate(ts_est, ts_gt, max_diff=0.05):
    idx_est, idx_gt = [], []
    for i, t in enumerate(ts_est):
        diffs = np.abs(ts_gt - t)
        j = np.argmin(diffs)
        if diffs[j] < max_diff:
            idx_est.append(i)
            idx_gt.append(j)
    return np.array(idx_est), np.array(idx_gt)


def umeyama(src, dst, with_scale=True):
    n = src.shape[0]
    mu_s = src.mean(0); mu_d = dst.mean(0)
    sc = src - mu_s;    dc = dst - mu_d
    var_s = np.mean(np.sum(sc**2, axis=1))
    if var_s < 1e-10:
        return src, 1.0, np.eye(3), mu_d - mu_s
    W = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(W)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2,2] = -1
    R = U @ S @ Vt
    s = (np.sum(D * np.diag(S)) / var_s) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    aligned = s * (R @ (src - mu_s).T).T + mu_d
    return aligned, s, R, t


def plot_sequence(seq_name, seq_path, results_dir="results", out_dir="results"):
    print(f"\nPlotting: {seq_name}")
    short = seq_name.replace("dataset-","").replace("_512_16","")

    mono,   ts_mono   = load_tum(f"{results_dir}/{seq_name}_mono_traj.txt")
    stereo, ts_stereo = load_tum(f"{results_dir}/{seq_name}_stereo_traj.txt")
    gt,     ts_gt     = load_gt(f"{seq_path}/mav0/mocap0/data.csv")

    has_gt     = gt     is not None and len(gt) > 2
    has_mono   = mono   is not None
    has_stereo = stereo is not None

    # Associate trajectories with GT
    mono_aligned = stereo_aligned = gt_mono = gt_stereo = None
    s_mono = 1.0

    if has_gt and has_mono:
        ie, ig = associate(ts_mono, ts_gt)
        if len(ie) > 10:
            mono_aligned, s_mono, _, _ = umeyama(mono[ie], gt[ig], with_scale=True)
            gt_mono = gt[ig]
            print(f"  Mono matched: {len(ie)} | scale={s_mono:.6f}")

    if has_gt and has_stereo:
        ie, ig = associate(ts_stereo, ts_gt)
        if len(ie) > 10:
            stereo_aligned, _, _, _ = umeyama(stereo[ie], gt[ig], with_scale=False)
            gt_stereo = gt[ig]
            print(f"  Stereo matched: {len(ie)}")

    # ── Figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(f"Trajectory Comparison — {short}", fontsize=15, fontweight='bold')

    # ── Panel 1: Raw mono trajectory ────────────────────────
    ax = axes[0]
    ax.set_title("Monocular VO — raw (up-to-scale)", fontsize=11)
    if has_mono:
        ax.plot(mono[:,0], mono[:,2], 'b-', linewidth=0.8,
                alpha=0.8, label='Mono VO (raw)')
        ax.scatter(mono[0,0],  mono[0,2],  c='green', s=80, zorder=5, label='Start')
        ax.scatter(mono[-1,0], mono[-1,2], c='black', s=80, zorder=5, label='End')
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9); ax.set_aspect('equal')

    # ── Panel 2: Raw stereo trajectory ──────────────────────
    ax = axes[1]
    ax.set_title("Stereo VO — raw (metric scale)", fontsize=11)
    if has_stereo:
        ax.plot(stereo[:,0], stereo[:,2], 'r-', linewidth=0.8,
                alpha=0.8, label='Stereo VO (metric)')
        ax.scatter(stereo[0,0],  stereo[0,2],  c='green', s=80, zorder=5, label='Start')
        ax.scatter(stereo[-1,0], stereo[-1,2], c='black', s=80, zorder=5, label='End')
    ax.set_xlabel("X"); ax.set_ylabel("Z (m)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9); ax.set_aspect('equal')

    # ── Panel 3: GT-aligned overlay ─────────────────────────
    ax = axes[2]
    ax.set_title("Aligned Comparison vs Ground Truth", fontsize=11)

    if has_gt:
        # Break GT polyline on time gaps so a 12.5-min mocap blackout
        # (outdoors5) does not get drawn as a fake straight line across
        # ground the camera never crossed.
        if ts_gt is not None and len(ts_gt) > 1:
            gaps = np.where(np.diff(ts_gt) > 5.0)[0]
        else:
            gaps = np.array([], dtype=int)
        segs = np.split(np.arange(len(gt)), gaps + 1)
        for k, s in enumerate(segs):
            if len(s) < 2:
                continue
            ax.plot(gt[s, 0], gt[s, 2], 'g-', linewidth=2.5,
                    alpha=0.9,
                    label='Ground Truth' if k == 0 else None,
                    zorder=4)

    if mono_aligned is not None:
        ax.plot(mono_aligned[:,0], mono_aligned[:,2], 'b-',
                linewidth=1.0, alpha=0.7,
                label=f'Mono VO (Sim3 aligned, s={s_mono:.4f})', zorder=3)

    if stereo_aligned is not None:
        ax.plot(stereo_aligned[:,0], stereo_aligned[:,2], 'r-',
                linewidth=1.0, alpha=0.7,
                label='Stereo VO (SE3 aligned)', zorder=2)

    # Mark start/end on GT
    if has_gt:
        ax.scatter(gt[0,0],  gt[0,2],  c='lime',  s=120,
                   zorder=6, edgecolors='black', linewidths=1, label='GT Start')
        ax.scatter(gt[-1,0], gt[-1,2], c='black', s=120,
                   zorder=6, label='GT End')

    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_aspect('equal')

    plt.tight_layout()
    out = f"{out_dir}/{seq_name}_gt_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"  Saved: {out}")
    plt.close()


if __name__ == "__main__":
    sequences = {
        "dataset-room2_512_16"    : "data/dataset-room2_512_16",
        "dataset-corridor3_512_16": "data/dataset-corridor3_512_16",
        "dataset-outdoors5_512_16": "data/dataset-outdoors5_512_16",
    }
    for seq_name, seq_path in sequences.items():
        plot_sequence(seq_name, seq_path)