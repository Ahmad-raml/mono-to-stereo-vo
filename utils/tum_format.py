import numpy as np
from scipy.spatial.transform import Rotation


def save_trajectory(filepath, timestamps_ns, poses_T):
    with open(filepath, "w") as f:
        f.write("# timestamp tx ty tz qx qy qz qw\n")
        for ts, T in zip(timestamps_ns, poses_T):
            tx, ty, tz = T[0,3], T[1,3], T[2,3]
            qx, qy, qz, qw = Rotation.from_matrix(T[:3,:3]).as_quat()
            f.write(f"{ts/1e9:.9f} {tx:.9f} {ty:.9f} {tz:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")
    print(f"[TUM] Saved {len(poses_T)} poses to: {filepath}")


def load_trajectory(filepath):
    timestamps, poses_T = [], []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            p = line.split()
            ts = float(p[0])
            tx, ty, tz = float(p[1]), float(p[2]), float(p[3])
            qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3,:3] = R
            T[:3, 3] = [tx, ty, tz]
            timestamps.append(ts)
            poses_T.append(T)
    return np.array(timestamps), poses_T


def load_ground_truth_tum(mocap_csv_path):
    timestamps, poses_T = [], []
    with open(mocap_csv_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or not line:
                continue
            parts = line.split(",")
            ts_ns = int(parts[0])
            tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3,:3] = R
            T[:3, 3] = [tx, ty, tz]
            timestamps.append(ts_ns / 1e9)
            poses_T.append(T)
    return np.array(timestamps), poses_T


def pose_to_translation(poses_T):
    return np.array([T[:3, 3] for T in poses_T])