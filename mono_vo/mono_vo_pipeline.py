"""
mono_vo/mono_vo_pipeline.py
Simple, stable monocular VO using optical flow + essential matrix.
Fixed scale=1 per frame (up-to-scale trajectory, correct for mono VO).

Honest failure handling:
- Frames whose pose could not be estimated are NOT written to the trajectory
  (timestamps and poses are skipped). This avoids hiding tracking failures
  behind frozen poses, which the project guide explicitly warns against and
  which inflates RPE.
- Every frame is logged with status (INIT / OK / TRACK_FAIL / POSE_FAIL) to
  a sibling *_status.csv so the failure rate is auditable after the run.
"""
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.tum_format import save_trajectory
from mono_vo.feature_extractor import FeatureExtractor


class MonoVO:
    def __init__(self, calib, method='ORB', n_features=3000):
        self.calib = calib
        w, h = calib.resolution

        self.K = calib.K0.copy()
        self.D = calib.D0.copy()

        self.K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            calib.K0, calib.D0, (w, h), np.eye(3), balance=1.0
        )
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            calib.K0, calib.D0, np.eye(3), self.K_new, (w, h), cv2.CV_16SC2
        )

        self.extractor = FeatureExtractor(method=method, n_features=n_features)

        # Cumulative pose
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.poses      = [np.eye(4)]
        self.timestamps = []   # one per emitted pose (1:1 with self.poses)

        # Per-frame status log (every frame, success or failure)
        # Each entry: (frame_idx, timestamp_ns, status, n_tracked, n_inliers)
        self.frame_status = []

        # Tracking state
        self.prev_img   = None
        self.prev_pts   = None
        self.frame_idx  = 0
        self.n_failures = 0   # any non-OK, non-INIT outcome

        print(f"[MonoVO] K_orig fx={self.K[0,0]:.1f} fy={self.K[1,1]:.1f}")

    def _preprocess(self, img):
        return cv2.equalizeHist(img)

    def _redetect(self, img, n=2000):
        fast = cv2.FastFeatureDetector_create(threshold=8, nonmaxSuppression=True)
        kps  = fast.detect(img, None)
        if len(kps) > 50:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:n]
            return np.float32([k.pt for k in kps])
        kps, _ = self.extractor.detect_and_compute(img)
        if kps:
            return np.float32([k.pt for k in kps[:n]])
        return np.array([])

    def _build_pose(self):
        T = np.eye(4)
        T[:3, :3] = self.cur_R
        T[:3,  3] = self.cur_t.ravel()
        return T

    def _estimate_pose(self, pts1, pts2):
        if len(pts1) < 8:
            return None, None, None

        pts1_ud = cv2.fisheye.undistortPoints(
            pts1.reshape(-1, 1, 2), self.K, self.D, P=self.K
        ).reshape(-1, 2)
        pts2_ud = cv2.fisheye.undistortPoints(
            pts2.reshape(-1, 1, 2), self.K, self.D, P=self.K
        ).reshape(-1, 2)

        E, mask = cv2.findEssentialMat(
            pts1_ud, pts2_ud,
            cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )
        if E is None or mask is None:
            return None, None, None

        mask = mask.ravel().astype(bool)
        if mask.sum() < 15:
            return None, None, None

        n_good, R, t, _ = cv2.recoverPose(
            E, pts1_ud[mask], pts2_ud[mask], cameraMatrix=self.K
        )
        if n_good < 8:
            return None, None, None

        return R, t, mask

    def _log(self, status, n_tracked=0, n_inliers=0, ts=None):
        self.frame_status.append({
            'frame_idx': self.frame_idx,
            'timestamp_ns': ts,
            'status': status,
            'n_tracked': int(n_tracked),
            'n_inliers': int(n_inliers),
        })

    def process_frame(self, img, timestamp):
        img = self._preprocess(img)

        # First frame: initialize but do not emit a duplicate pose
        # (poses already starts with [I] paired with the first timestamp below).
        if self.prev_img is None:
            self.prev_pts = self._redetect(img)
            self.prev_img = img
            self.timestamps.append(timestamp)   # paired with poses[0] = I
            self._log('INIT', n_tracked=len(self.prev_pts), ts=timestamp)
            self.frame_idx += 1
            return np.eye(4)

        # Track points with Lucas-Kanade optical flow
        if self.prev_pts is not None and len(self.prev_pts) > 0:
            pts1_f = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)
            pts2_f, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_img, img, pts1_f, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            status = status.ravel()
            pts1 = self.prev_pts[status == 1]
            pts2 = pts2_f.reshape(-1, 2)[status == 1]
        else:
            pts1, pts2 = np.array([]), np.array([])

        # TRACK_FAIL: too few points survived optical flow
        if len(pts1) < 50:
            self.prev_pts = self._redetect(img)
            self.prev_img = img
            self._log('TRACK_FAIL', n_tracked=len(pts1), ts=timestamp)
            self.frame_idx  += 1
            self.n_failures += 1
            return self._build_pose()  # last successful pose, NOT appended

        # Estimate pose
        R, t, mask = self._estimate_pose(pts1, pts2)

        if R is None:
            self.prev_img = img
            self.prev_pts = pts2 if len(pts2) > 50 else self._redetect(img)
            self._log('POSE_FAIL', n_tracked=len(pts1), ts=timestamp)
            self.frame_idx  += 1
            self.n_failures += 1
            return self._build_pose()

        # Success — update cumulative pose and emit
        self.cur_t = self.cur_t + self.cur_R @ t
        self.cur_R = R @ self.cur_R

        T = self._build_pose()
        self.poses.append(T)
        self.timestamps.append(timestamp)

        n_inliers = int(mask.sum()) if mask is not None else len(pts1)
        self._log('OK', n_tracked=len(pts1), n_inliers=n_inliers, ts=timestamp)

        # Carry inliers forward; redetect if too sparse
        inlier_pts = pts2[mask] if mask is not None else pts2
        self.prev_pts = inlier_pts if len(inlier_pts) > 200 else self._redetect(img)
        self.prev_img = img
        self.frame_idx += 1
        return T

    def _save_status(self, output_path):
        status_path = output_path.replace('.txt', '_status.csv')
        with open(status_path, 'w') as f:
            f.write("frame_idx,timestamp_ns,status,n_tracked,n_inliers\n")
            for s in self.frame_status:
                ts = '' if s['timestamp_ns'] is None else s['timestamp_ns']
                f.write(f"{s['frame_idx']},{ts},{s['status']},"
                        f"{s['n_tracked']},{s['n_inliers']}\n")
        print(f"[MonoVO] Status log: {status_path}")

    def run(self, loader, output_path, max_frames=None):
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
            exist_ok=True
        )
        n = len(loader) if max_frames is None else min(len(loader), max_frames)
        print(f"\n[MonoVO] Running on {n} frames...")

        for i in range(n):
            img, ts = loader.get_left_frame(i)
            T = self.process_frame(img, ts)
            if i % 100 == 0:
                t = T[:3, 3]
                print(f"  Frame {i:4d}/{n} | "
                      f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] | "
                      f"tracked={len(self.prev_pts) if self.prev_pts is not None else 0} | "
                      f"failures={self.n_failures}")

        save_trajectory(output_path, self.timestamps, self.poses)
        self._save_status(output_path)

        n_emitted = len(self.poses)
        print(f"\n[MonoVO] Done | emitted={n_emitted}/{n} poses | "
              f"failures={self.n_failures}/{n} "
              f"({100.0*self.n_failures/max(n,1):.1f}%)")
        print(f"[MonoVO] Saved: {output_path}")
        return self.poses, self.timestamps
