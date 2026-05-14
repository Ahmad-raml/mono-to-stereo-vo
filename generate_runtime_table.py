"""
generate_runtime_table.py
Runs main.py across the 3 TUM VI sequences in both mono and stereo modes,
parses the [Runtime] line printed by main.py, and writes a clean CSV to
results/runtime_table.csv. Also prints a readable summary to stdout.

Usage:
    python generate_runtime_table.py
    python generate_runtime_table.py --frames 500     # limit frames (debug)
    python generate_runtime_table.py --skip-run       # just reparse existing logs
"""
import argparse
import csv
import os
import re
import subprocess
import sys


SEQUENCES = [
    ("dataset-room2_512_16",     "data/dataset-room2_512_16"),
    ("dataset-corridor3_512_16", "data/dataset-corridor3_512_16"),
    ("dataset-outdoors5_512_16", "data/dataset-outdoors5_512_16"),
]
MODES = ["mono", "stereo"]

RUNTIME_RE = re.compile(
    r"\[Runtime\]\s+"
    r"Sequence:\s*(?P<seq>\S+)\s*\|\s*"
    r"Mode:\s*(?P<mode>\w+)\s*\|\s*"
    r"Frames:\s*(?P<frames>\d+)\s*\|\s*"
    r"Total:\s*(?P<total>[\d.]+)s\s*\|\s*"
    r"Per-frame:\s*(?P<ms>[\d.]+)ms\s*\|\s*"
    r"FPS:\s*(?P<fps>[\d.]+)"
)


def run_one(seq_path, mode, frames=None, log_dir="results/runtime_logs"):
    """Invoke main.py as a subprocess and capture its stdout."""
    os.makedirs(log_dir, exist_ok=True)
    seq_name = os.path.basename(seq_path)
    log_path = os.path.join(log_dir, f"{seq_name}_{mode}.log")

    cmd = [sys.executable, "main.py", "--mode", mode, "--sequence", seq_path]
    if frames is not None:
        cmd += ["--frames", str(frames)]

    print(f"\n>>> {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = proc.stdout + "\n" + proc.stderr

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(output)

    if proc.returncode != 0:
        print(f"    [WARN] return code {proc.returncode} "
              f"— see {log_path}")

    return output, log_path


def parse_runtime(text):
    """Return dict of runtime fields, or None if no [Runtime] line was found."""
    for line in text.splitlines():
        m = RUNTIME_RE.search(line)
        if m:
            return {
                "sequence":     m.group("seq"),
                "mode":         m.group("mode"),
                "frames":       int(m.group("frames")),
                "total_sec":    float(m.group("total")),
                "ms_per_frame": float(m.group("ms")),
                "fps":          float(m.group("fps")),
            }
    return None


def format_table(rows):
    if not rows:
        return "(no data)"
    header = ["Sequence", "Mode", "Frames", "Total (s)", "ms/frame", "FPS"]
    widths = [max(len(h), 22) for h in header]
    widths[0] = 28
    lines = []
    lines.append(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    lines.append("-+-".join("-" * w for w in widths))
    for r in rows:
        short = r["sequence"].replace("dataset-", "").replace("_512_16", "")
        cells = [
            short,
            r["mode"],
            str(r["frames"]),
            f"{r['total_sec']:.2f}",
            f"{r['ms_per_frame']:.1f}",
            f"{r['fps']:.2f}",
        ]
        lines.append(" | ".join(c.ljust(w) for c, w in zip(cells, widths)))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=None,
                        help="Limit frames per run (for quick sanity check).")
    parser.add_argument("--skip-run", action="store_true",
                        help="Reparse existing logs in results/runtime_logs/ "
                             "without re-running main.py.")
    parser.add_argument("--output", default="results/runtime_table.csv")
    args = parser.parse_args()

    log_dir = "results/runtime_logs"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows = []
    for seq_name, seq_path in SEQUENCES:
        for mode in MODES:
            if args.skip_run:
                log_path = os.path.join(log_dir, f"{seq_name}_{mode}.log")
                if not os.path.exists(log_path):
                    print(f"    [SKIP] no log: {log_path}")
                    continue
                with open(log_path, "r", encoding="utf-8") as f:
                    output = f.read()
            else:
                output, _ = run_one(seq_path, mode, frames=args.frames,
                                    log_dir=log_dir)

            parsed = parse_runtime(output)
            if parsed is None:
                print(f"    [WARN] no [Runtime] line for {seq_name} {mode}")
                continue
            rows.append(parsed)

    # Write CSV
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["sequence", "mode", "frames",
                        "total_sec", "ms_per_frame", "fps"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\n[CSV ] wrote {len(rows)} rows -> {args.output}")

    print("\n" + "=" * 72)
    print("RUNTIME TABLE")
    print("=" * 72)
    print(format_table(rows))
    print("=" * 72)


if __name__ == "__main__":
    main()
