"""
utils/data_loader.py
Loads TUM VI dataset sequences (EuRoC/DSO 512x512 format).
"""

import os
import numpy as np
import cv2


class TUMVILoader:
    """
    Loads a single TUM VI sequence from the EuRoC/DSO export format.

    Expected folder structure:
        dataset-<name>_512_16/
        └── mav0/
            ├── cam0/data/       <- left camera PNGs
            ├── cam0/data.csv    <- left timestamps
            ├── cam1/data/       <- right camera PNGs
            ├── cam1/data.csv    <- right timestamps
            └── mocap0/data.csv  <- ground truth poses
    """

    def __init__(self, sequence_path):
        self.sequence_path = sequence_path
        self.mav0_path = os.path.join(sequence_path, "mav0")

        self.cam0_dir  = os.path.join(self.mav0_path, "cam0", "data")
        self.cam1_dir  = os.path.join(self.mav0_path, "cam1", "data")
        self.cam0_csv  = os.path.join(self.mav0_path, "cam0", "data.csv")
        self.cam1_csv  = os.path.join(self.mav0_path, "cam1", "data.csv")
        self.mocap_csv = os.path.join(self.mav0_path, "mocap0", "data.csv")

        self.left_timestamps,  self.left_images  = self._load_image_list(self.cam0_csv, self.cam0_dir)
        self.right_timestamps, self.right_images = self._load_image_list(self.cam1_csv, self.cam1_dir)
        self.gt_timestamps,    self.gt_poses      = self._load_ground_truth()

        print(f"[DataLoader] Loaded: {os.path.basename(sequence_path)}")
        print(f"  Left  frames : {len(self.left_images)}")
        print(f"  Right frames : {len(self.right_images)}")
        print(f"  GT poses     : {len(self.gt_poses)}")

    def _load_image_list(self, csv_path, image_dir):
        timestamps, image_paths = [], []

        if not os.path.exists(csv_path):
            for f in sorted(os.listdir(image_dir)):
                if f.endswith(".png"):
                    timestamps.append(int(os.path.splitext(f)[0]))
                    image_paths.append(os.path.join(image_dir, f))
            return np.array(timestamps), image_paths

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(",")
                ts    = int(parts[0])
                fname = parts[1].strip() if len(parts) > 1 else f"{ts}.png"
                timestamps.append(ts)
                image_paths.append(os.path.join(image_dir, fname))

        return np.array(timestamps), image_paths

    def _load_ground_truth(self):
        timestamps, poses = [], []

        if not os.path.exists(self.mocap_csv):
            print("[DataLoader] WARNING: No ground truth file found.")
            return np.array([]), np.array([])

        with open(self.mocap_csv, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split(",")
                ts = int(parts[0])
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qw, qx, qy, qz = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                timestamps.append(ts)
                poses.append([tx, ty, tz, qw, qx, qy, qz])

        return np.array(timestamps), np.array(poses)

    def get_frame(self, idx, grayscale=True):
        """Load stereo pair at index idx."""
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        left_img  = cv2.imread(self.left_images[idx],  flag)
        right_img = cv2.imread(self.right_images[idx], flag)
        if left_img  is None: raise FileNotFoundError(f"Cannot load: {self.left_images[idx]}")
        if right_img is None: raise FileNotFoundError(f"Cannot load: {self.right_images[idx]}")
        return left_img, right_img, self.left_timestamps[idx]

    def get_left_frame(self, idx, grayscale=True):
        """Load only left (cam0) frame — used for monocular VO."""
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(self.left_images[idx], flag)
        if img is None: raise FileNotFoundError(f"Cannot load: {self.left_images[idx]}")
        return img, self.left_timestamps[idx]

    def get_ground_truth_at(self, timestamp_ns):
        """Find closest ground truth pose to a given timestamp."""
        if len(self.gt_timestamps) == 0:
            return None
        idx = np.argmin(np.abs(self.gt_timestamps - timestamp_ns))
        return self.gt_poses[idx]

    def __len__(self):
        return len(self.left_images)


if __name__ == "__main__":
    import sys
    np.random.seed(42)

    seq_path = sys.argv[1] if len(sys.argv) > 1 else "data/dataset-room2_512_16"
    loader = TUMVILoader(seq_path)

    left, right, ts = loader.get_frame(0)
    print(f"\nFirst frame timestamp : {ts} ns")
    print(f"Left  image shape     : {left.shape}, dtype: {left.dtype}")
    print(f"Right image shape     : {right.shape}, dtype: {right.dtype}")

    gt = loader.get_ground_truth_at(ts)
    if gt is not None:
        print(f"Closest GT pose       : tx={gt[0]:.4f} ty={gt[1]:.4f} tz={gt[2]:.4f}")