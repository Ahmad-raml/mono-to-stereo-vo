"""
generate_eval_plots.py
Grouped bar charts comparing monocular vs stereo VO across 3 TUM VI sequences.

Outputs (PNG, 200 dpi):
    results/plots/rpe_translational.png
    results/plots/rpe_translational_log.png
    results/plots/rpe_rotational.png
    results/plots/ate_comparison.png
    results/plots/drift_comparison.png
    results/plots/drift_comparison_log.png
    results/plots/runtime_fps.png

If results/runtime_table.csv exists (see generate_runtime_table.py), FPS is
read from it. Otherwise placeholder values (mono=15, stereo=8) are used.
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Pre-computed evaluation numbers
# Gap-aware protocol (see evaluation/evaluate.py and PAPRT.TXT § VI.B):
#   - ATE is the worst per-segment ATE on sequences with a GT-coverage
#     gap (corridor3, outdoors5); on room2 it is the whole-trajectory
#     ATE since there is only one segment.
#   - RPE is gap-aware: pairs whose real-time separation exceeds the
#     gap threshold are skipped.
#   - Drift is start-end (Eq. 8); the % column lives in the paper, not
#     in the plot.
# ─────────────────────────────────────────────
SEQUENCES = ["room2", "corridor3", "outdoors5"]

ATE_MONO    = [1.2335,  0.7737,  0.9062]
ATE_STEREO  = [0.2657,  0.0460,  0.1700]

RPE_T_MONO   = [9.8318,  8.0265,  7.7753]
RPE_T_STEREO = [0.0181,  0.0193,  0.0370]

RPE_R_MONO   = [46.5728, 53.6050, 62.9617]
RPE_R_STEREO = [ 1.0500,  1.0655,  1.6160]

DRIFT_MONO   = [1879.484,  707.405, 4564.984]
DRIFT_STEREO = [   1.116,    4.333,   46.133]

# Colours per spec
COL_MONO   = "#1A6BBF"
COL_STEREO = "#D9502A"

OUT_DIR = "results/plots"


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#222222",
        "axes.labelcolor":   "#222222",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.color":        "#DDDDDD",
        "grid.linewidth":    0.7,
        "font.size":         11,
    })


def _annotate(ax, bars, fmt="{:.2f}", log=False):
    for b in bars:
        h = b.get_height()
        y = h
        if log:
            # Place slightly above bar in log-space
            y = h * 1.08 if h > 0 else 1e-3
        else:
            y = h + (ax.get_ylim()[1] * 0.01)
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            y,
            fmt.format(h),
            ha="center", va="bottom",
            fontsize=9, color="#222222",
        )


def _grouped_bars(ax, mono, stereo, title, ylabel,
                  fmt="{:.2f}", log=False):
    x        = np.arange(len(SEQUENCES))
    width    = 0.38
    b_mono   = ax.bar(x - width / 2, mono,   width,
                      label="Monocular VO", color=COL_MONO,
                      edgecolor="#0D1B2A", linewidth=0.6)
    b_stereo = ax.bar(x + width / 2, stereo, width,
                      label="Stereo VO",    color=COL_STEREO,
                      edgecolor="#0D1B2A", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(SEQUENCES)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=13, fontweight="bold", color="#0D1B2A")
    ax.legend(frameon=False, loc="upper left")

    if log:
        ax.set_yscale("log")
        ymin = max(1e-2, min(min(mono), min(stereo)) * 0.5)
        ymax = max(max(mono), max(stereo)) * 3.0
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_ylim(0, max(max(mono), max(stereo)) * 1.18)

    _annotate(ax, b_mono,   fmt=fmt, log=log)
    _annotate(ax, b_stereo, fmt=fmt, log=log)


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved: {path}")


# ─────────────────────────────────────────────
# Runtime lookup
# ─────────────────────────────────────────────
def load_fps(csv_path="results/runtime_table.csv"):
    """
    Returns (fps_mono, fps_stereo) lists in SEQUENCES order.
    Falls back to placeholder 15/8 if the CSV is missing or incomplete.
    """
    placeholder_mono   = [15.0, 15.0, 15.0]
    placeholder_stereo = [ 8.0,  8.0,  8.0]

    if not os.path.exists(csv_path):
        print(f"  [fps] {csv_path} not found; using placeholders 15/8")
        return placeholder_mono, placeholder_stereo

    table = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            short = (row["sequence"]
                     .replace("dataset-", "")
                     .replace("_512_16", ""))
            table[(short, row["mode"])] = float(row["fps"])

    fps_mono, fps_stereo = [], []
    for s in SEQUENCES:
        fps_mono.append  (table.get((s, "mono"),   placeholder_mono[0]))
        fps_stereo.append(table.get((s, "stereo"), placeholder_stereo[0]))
    return fps_mono, fps_stereo


# ─────────────────────────────────────────────
# Plot generators
# ─────────────────────────────────────────────
def plot_rpe_t():
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, RPE_T_MONO, RPE_T_STEREO,
                  title="Relative Pose Error — Translational",
                  ylabel="RPE-t (m)", fmt="{:.2f}")
    _save(fig, f"{OUT_DIR}/rpe_translational.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, RPE_T_MONO, RPE_T_STEREO,
                  title="Relative Pose Error — Translational (log scale)",
                  ylabel="RPE-t (m, log)", fmt="{:.2f}", log=True)
    _save(fig, f"{OUT_DIR}/rpe_translational_log.png")


def plot_rpe_r():
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, RPE_R_MONO, RPE_R_STEREO,
                  title="Relative Pose Error — Rotational",
                  ylabel="RPE-r (deg)", fmt="{:.2f}")
    _save(fig, f"{OUT_DIR}/rpe_rotational.png")


def plot_ate():
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, ATE_MONO, ATE_STEREO,
                  title="Absolute Trajectory Error  "
                        "(mono: Sim3, stereo: SE3)",
                  ylabel="ATE RMSE (m)", fmt="{:.2f}")
    _save(fig, f"{OUT_DIR}/ate_comparison.png")


def plot_drift():
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, DRIFT_MONO, DRIFT_STEREO,
                  title="Start-End Drift (Eq. 8)",
                  ylabel="Drift (m)", fmt="{:.2f}")
    _save(fig, f"{OUT_DIR}/drift_comparison.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, DRIFT_MONO, DRIFT_STEREO,
                  title="Start-End Drift — log scale",
                  ylabel="Drift (m, log)", fmt="{:.2f}", log=True)
    _save(fig, f"{OUT_DIR}/drift_comparison_log.png")


def plot_fps():
    fps_mono, fps_stereo = load_fps()
    fig, ax = plt.subplots(figsize=(8, 5))
    _grouped_bars(ax, fps_mono, fps_stereo,
                  title="Runtime Performance (FPS)",
                  ylabel="Frames per second", fmt="{:.1f}")
    _save(fig, f"{OUT_DIR}/runtime_fps.png")


def main():
    _style()
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Writing plots to {OUT_DIR}/ ...")
    plot_rpe_t()
    plot_rpe_r()
    plot_ate()
    plot_drift()
    plot_fps()
    print("Done.")


if __name__ == "__main__":
    main()
