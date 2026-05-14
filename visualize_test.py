import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Accept sequence name from command line
if len(sys.argv) > 1:
    seq = sys.argv[1]
else:
    seq = "dataset-room2_512_16"

traj_file = f"results/{seq}_mono_traj.txt"

if not os.path.exists(traj_file):
    print(f"File not found: {traj_file}")
    sys.exit(1)

traj = []
with open(traj_file, "r") as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        traj.append([float(parts[1]), float(parts[2]), float(parts[3])])

traj = np.array(traj)
print(f"Sequence     : {seq}")
print(f"Total poses  : {len(traj)}")
print(f"X range      : [{traj[:,0].min():.2f}, {traj[:,0].max():.2f}]")
print(f"Y range      : [{traj[:,1].min():.2f}, {traj[:,1].max():.2f}]")
print(f"Z range      : [{traj[:,2].min():.2f}, {traj[:,2].max():.2f}]")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].plot(traj[:,0], traj[:,2], 'b-', linewidth=1.0)
axes[0].scatter(traj[0,0],  traj[0,2],  c='green', s=100, zorder=5, label='Start')
axes[0].scatter(traj[-1,0], traj[-1,2], c='red',   s=100, zorder=5, label='End')
axes[0].set_xlabel('X'); axes[0].set_ylabel('Z')
axes[0].set_title(f'Top-Down View (X-Z) — {len(traj)} frames')
axes[0].legend(); axes[0].grid(True); axes[0].set_aspect('equal')

axes[1].plot(traj[:,0], traj[:,1], 'b-', linewidth=1.0)
axes[1].scatter(traj[0,0],  traj[0,1],  c='green', s=100, zorder=5, label='Start')
axes[1].scatter(traj[-1,0], traj[-1,1], c='red',   s=100, zorder=5, label='End')
axes[1].set_xlabel('X'); axes[1].set_ylabel('Y')
axes[1].set_title(f'Side View (X-Y) — {len(traj)} frames')
axes[1].legend(); axes[1].grid(True); axes[1].set_aspect('equal')

plt.suptitle(f'Monocular VO — {seq} (up-to-scale)', fontsize=13)
plt.tight_layout()

out = f"results/{seq}_mono_traj.png"
plt.savefig(out, dpi=150)
plt.show()
print(f"Saved: {out}")