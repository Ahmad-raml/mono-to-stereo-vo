"""
mono_vo/epipolar.py
Essential matrix, pose recovery, and triangulation.
Ref: Hartley & Zisserman, Multiple View Geometry (2004)
"""
import numpy as np
import cv2

class EpipolarGeometry:
    def __init__(self, K, ransac_threshold=1.0, confidence=0.999):
        self.K = K
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence

    def estimate_essential_matrix(self, pts1, pts2):
        if len(pts1) < 5:
            return None, None, 0
        E, mask = cv2.findEssentialMat(
            pts1, pts2, cameraMatrix=self.K,
            method=cv2.RANSAC,
            prob=self.confidence,
            threshold=self.ransac_threshold
        )
        if E is None or mask is None:
            return None, None, 0
        mask = mask.ravel().astype(bool)
        return E, mask, int(mask.sum())

    def recover_pose(self, E, pts1, pts2, mask=None):
        if E is None:
            return None, None, 0, None
        if mask is not None:
            pts1, pts2 = pts1[mask], pts2[mask]
        n_good, R, t, pose_mask = cv2.recoverPose(E, pts1, pts2, cameraMatrix=self.K)
        return R, t, n_good, pose_mask

    def triangulate_points(self, R, t, pts1, pts2):
        P1 = self.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.K @ np.hstack([R, t.reshape(3, 1)])
        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T.astype(np.float64), pts2.T.astype(np.float64))
        pts_4d /= pts_4d[3]
        return pts_4d[:3].T

    def estimate_pose(self, pts1, pts2):
        E, mask, n_inliers = self.estimate_essential_matrix(pts1, pts2)
        if E is None or n_inliers < 10:
            return None, None, None, None
        R, t, n_good, _ = self.recover_pose(E, pts1, pts2, mask)
        if R is None or n_good < 5:
            return None, None, None, mask
        pts3d = self.triangulate_points(R, t, pts1[mask], pts2[mask])
        pts3d = pts3d[pts3d[:, 2] > 0]  # keep positive depth only
        print(f"[Epipolar] inliers={n_inliers} cheirality={n_good} 3D pts={len(pts3d)}")
        return R, t, pts3d, mask