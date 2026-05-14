import numpy as np
import matplotlib.pyplot as plt
import sys
import os

seq = sys.argv[1] if len(sys.argv) > 1 else "dataset-room2_512_16"

def load_traj(path):
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return None
    traj = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            traj.append([float(p[1]), float(p[2]), float(p[3])])
    return np.array(traj) if traj else None

mono_path   = f"results/{seq}_mono_traj.txt"
stereo_path = f"results/{seq}_stereo_traj.txt"

print(f"Mono   file: {mono_path}  exists={os.path.exists(mono_path)}")
print(f"Stereo file: {stereo_path}  exists={os.path.exists(stereo_path)}")

mono   = load_traj(mono_path)
stereo = load_traj(stereo_path)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# --- Mono ---
ax = axes[0]
if mono is not None:
    ax.plot(mono[:,0], mono[:,2], 'b-', linewidth=0.8, label='Mono VO (up-to-scale)')
    ax.scatter(mono[0,0],  mono[0,2],  c='green', s=100, zorder=5, label='Start')
    ax.scatter(mono[-1,0], mono[-1,2], c='black', s=100, zorder=5, label='End')
    ax.set_title(f'Mono VO (up-to-scale) — {len(mono)} frames')
else:
    ax.text(0.5, 0.5, 'No mono trajectory found', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_title('Mono VO — No data')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.legend(); ax.grid(True)
if mono is not None:
    ax.set_aspect('equal')

# --- Stereo ---
ax = axes[1]
if stereo is not None:
    ax.plot(stereo[:,0], stereo[:,2], 'r-', linewidth=0.8, label='Stereo VO (metric)')
    ax.scatter(stereo[0,0],  stereo[0,2],  c='green', s=100, zorder=5, label='Start')
    ax.scatter(stereo[-1,0], stereo[-1,2], c='black', s=100, zorder=5, label='End')
    ax.set_title(f'Stereo VO (metric) — {len(stereo)} frames')
    # Print stats
    extent = stereo.max(axis=0) - stereo.min(axis=0)
    print(f"\nStereo trajectory extent:")
    print(f"  X: {stereo[:,0].min():.2f} to {stereo[:,0].max():.2f} m")
    print(f"  Y: {stereo[:,1].min():.2f} to {stereo[:,1].max():.2f} m")
    print(f"  Z: {stereo[:,2].min():.2f} to {stereo[:,2].max():.2f} m")
else:
    ax.text(0.5, 0.5, 'No stereo trajectory found', ha='center', va='center',
            transform=ax.transAxes, fontsize=12)
    ax.set_title('Stereo VO — No data')
ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
ax.legend(); ax.grid(True)
if stereo is not None:
    ax.set_aspect('equal')

plt.suptitle(f'Mono vs Stereo VO — {seq}', fontsize=13)
plt.tight_layout()

out = f"results/{seq}_comparison.png"
plt.savefig(out, dpi=150)
plt.show()
print(f"\nSaved: {out}")