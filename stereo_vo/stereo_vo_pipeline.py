"""
stereo_vo/stereo_vo_pipeline.py
Metric stereo visual odometry pipeline.
Uses calibrated stereo baseline for metric depth via disparity.
Ref: Section V of project guide — Z = fB/d, 3D-2D PnP pose estimation.

Honest failure handling:
- Frames whose pose could not be estimated, or whose step was rejected by
  sanity checks, are NOT written to the trajectory (the timestamp is also
  skipped). Freezing the previous pose into the trajectory hides tracking
  failures and inflates RPE; the project guide explicitly forbids this.
- Every frame is logged with status to a sibling *_status.csv so the
  failure rate is auditable after the run.

Sequence-specific tuning (depth ranges, disparity validity, RANSAC step
limits) is exposed via constructor arguments — defaults are tuned for
indoor/room2 and overridden per sequence in main.py.
"""
import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.tum_format import save_trajectory
from mono_vo.feature_extractor import FeatureExtractor
from stereo_vo.disparity import DisparityEstimator


class StereoVO:
    def __init__(
        self,
        calib,
        method='ORB',
        n_features=3000,
        disp_method='SGBM',
        # ---- depth and disparity validity ----
        min_depth=0.1,
        max_depth=20.0,
        min_disp_valid=1.0,
        num_disparities=64,
        block_size=7,
        # ---- per-frame motion sanity (20 Hz handheld defaults) ----
        max_step_rot_deg=25.0,
        max_step_trans=0.50,
        min_inlier_ratio=0.25,
        # ---- PnP ----
        pnp_reproj_err=2.0,
        # ---- post-PnP reprojection sanity ----
        # Inliers are guaranteed <= pnp_reproj_err by construction; if their
        # RMSE pushes near that bound the pose is barely fitting even its own
        # subset. Threshold = pnp_reproj_err / 2 catches degenerate motion
        # (pure rotation, near-planar scene, motion blur) without false
        # positives on well-conditioned frames. Empirically: room2 OK-frame
        # reproj_rmse has median 0.53 px and p99 0.88 px.
        max_reproj_rmse=1.0,
        # ---- keyframe selection ----
        # Frame-to-keyframe tracking: 3D landmarks are triangulated once at
        # the keyframe and reused across many frames. Re-keyframing happens
        # only when one of these bounds is exceeded. The motivation is to
        # stop accumulating disparity noise on every frame -- each pose is
        # anchored to a stable keyframe rather than to the (also-noisy)
        # previous frame.
        min_kf_features=200,
        max_kf_trans=0.30,
        max_kf_rot_deg=12.0,
        # ---- feature replenishment within a keyframe ----
        # In translation-heavy scenes (corridor walks, outdoor traversals)
        # close features leave the frame quickly while far ones remain,
        # biasing PnP toward poorly-conditioned distant landmarks. When
        # alive count drops below this fraction of the keyframe's initial
        # count, we detect fresh features in cur, triangulate them, and
        # transform them into keyframe coordinates so they join the alive
        # set without breaking the keyframe pose anchor.
        replenish_ratio=0.5,
    ):
        self.calib    = calib
        self.baseline = calib.baseline  # meters

        # Precompute stereo rectification maps once
        print("[StereoVO] Computing stereo rectification maps...")
        (self.map1l, self.map2l,
         self.map1r, self.map2r,
         self.Q, self.P1, self.P2) = calib.get_stereo_rectification()

        self.K_rect = self.P1[:3, :3]
        self.fx     = self.K_rect[0, 0]

        self.extractor = FeatureExtractor(method=method, n_features=n_features)
        self.disparity = DisparityEstimator(
            method=disp_method,
            num_disparities=num_disparities,
            block_size=block_size,
        )

        # Cumulative pose (camera-to-world)
        self.cur_R = np.eye(3)
        self.cur_t = np.zeros((3, 1))
        self.poses      = [np.eye(4)]
        self.timestamps = []  # 1:1 with self.poses

        # Per-frame status log
        self.frame_status = []

        # Keyframe state. prev_pts3d holds 3D landmarks expressed in the
        # CURRENT KEYFRAME's camera frame -- not the previous frame's, as in
        # pure frame-to-frame VO. prev_pts2d holds the most recent observed
        # 2D positions of those landmarks (in the previous frame's image),
        # which the next KLT step uses as input. Each frame's PnP solves the
        # pose of cur in the keyframe's frame; the world pose is then
        # T_w_cur = T_w_kf @ inv(T_cur_kf).
        self.kf_R = np.eye(3)
        self.kf_t = np.zeros((3, 1))
        self.prev_left_rect = None
        self.prev_pts2d     = None
        self.prev_pts3d     = None
        self.frame_idx      = 0
        self.n_pnp_fail     = 0
        self.n_step_reject  = 0
        self.n_track_fail   = 0
        self.n_keyframes    = 0

        # Tunables
        self.min_depth        = min_depth
        self.max_depth        = max_depth
        self.min_disp_valid   = min_disp_valid
        self.max_step_rot_deg = max_step_rot_deg
        self.max_step_trans   = max_step_trans
        self.min_inlier_ratio = min_inlier_ratio
        self.pnp_reproj_err   = pnp_reproj_err
        self.max_reproj_rmse  = max_reproj_rmse
        self.min_kf_features  = min_kf_features
        self.max_kf_trans     = max_kf_trans
        self.max_kf_rot_deg   = max_kf_rot_deg
        self.replenish_ratio  = replenish_ratio
        # Tracks the alive-feature count we will compare against to decide
        # if replenishment is needed. Reset at each new keyframe and
        # bumped up again whenever replenishment fires.
        self.kf_n_anchor = 0
        self.n_replenish = 0

        # Counters
        self.n_reproj_low_conf = 0   # pose emitted but flagged in status CSV

        print(f"[StereoVO] K_rect fx={self.fx:.2f}  |  baseline={self.baseline*100:.2f} cm")
        print(f"[StereoVO] depth range=[{min_depth:.2f}, {max_depth:.2f}] m  |  "
              f"min_disp_valid={min_disp_valid:.2f} px  |  "
              f"num_disparities={num_disparities}")
        print(f"[StereoVO] sanity: max_step_rot={max_step_rot_deg:.1f}deg  "
              f"max_step_trans={max_step_trans:.2f}m  "
              f"min_inlier_ratio={min_inlier_ratio:.2f}")
        print(f"[StereoVO] keyframe: min_features={min_kf_features}  "
              f"max_trans={max_kf_trans:.2f}m  "
              f"max_rot={max_kf_rot_deg:.1f}deg")

    def _rectify(self, left_img, right_img):
        left_rect  = cv2.remap(left_img,  self.map1l, self.map2l, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right_img, self.map1r, self.map2r, cv2.INTER_LINEAR)
        return left_rect, right_rect

    def _preprocess(self, img):
        return cv2.equalizeHist(img)

    def _detect_features(self, img, n=2000):
        fast = cv2.FastFeatureDetector_create(threshold=8, nonmaxSuppression=True)
        kps  = fast.detect(img, None)
        if len(kps) > 10:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:n]
            return np.float32([k.pt for k in kps])
        kps, _ = self.extractor.detect_and_compute(img)
        return np.float32([k.pt for k in kps[:n]]) if kps else np.array([])

    def _build_pose(self):
        T = np.eye(4)
        T[:3, :3] = self.cur_R
        T[:3,  3] = self.cur_t.ravel()
        return T

    def _sample_disparity(self, disp, u, v):
        """
        Robust per-feature disparity sampler.
        - exact pixel first if it has a valid value
        - else median of valid disparities in a 3x3 neighborhood
        Validity = disparity > self.min_disp_valid (configurable per sequence).
        """
        h, w   = disp.shape
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < w and 0 <= vi < h):
            return 0.0

        thr = self.min_disp_valid
        d0  = disp[vi, ui]
        if d0 > thr:
            return float(d0)

        vals = []
        for dv in range(-1, 2):
            for du in range(-1, 2):
                nu, nv = ui + du, vi + dv
                if 0 <= nu < w and 0 <= nv < h:
                    nd = disp[nv, nu]
                    if nd > thr:
                        vals.append(nd)
        if not vals:
            return 0.0
        return float(np.median(vals))

    def _get_3d_points(self, left_rect, right_rect, pts2d):
        disp = self.disparity.compute(left_rect, right_rect)

        fx = self.fx
        cx = self.K_rect[0, 2]
        cy = self.K_rect[1, 2]
        B  = self.baseline

        pts3d_list, pts2d_list = [], []

        for pt in pts2d:
            u, v = pt[0], pt[1]
            d = self._sample_disparity(disp, u, v)
            if d < self.min_disp_valid:
                continue

            # Z = fB/d  (project guide Eq.4)
            Z = (fx * B) / d
            if Z < self.min_depth or Z > self.max_depth:
                continue

            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fx
            pts3d_list.append([X, Y, Z])
            pts2d_list.append(pt)

        if len(pts3d_list) < 6:
            return np.array([]), np.array([])

        return (np.array(pts3d_list, dtype=np.float64),
                np.array(pts2d_list, dtype=np.float32))

    def _pnp_pose(self, pts3d_prev, pts2d_prev, cur_left):
        """
        Track 2D points prev->cur with optical flow, then solvePnPRansac.

        The 3D points are in the CURRENT KEYFRAME's camera frame (not the
        previous frame's), so R_wc/t_wc returned here is the pose of cur in
        the keyframe frame (T_cur_kf).

        Also returns the FB-survivor subset of pts3d_prev / pts2d (in the
        cur image) so the caller can slide tracking forward without
        re-detecting features.

        Returns:
            {'status': 'OK'|'TRACK_FAIL'|'PNP_FAIL'|'LOW_INLIER',
             'R_wc': ..., 't_wc': ...,
             'n_matched': int, 'n_inliers': int, 'reproj_rmse': float,
             'pts3d_alive': (N',3),   # subset of pts3d_prev that survived FB
             'pts2d_alive': (N',2)}   # cur-image positions of survivors
        """
        out = dict(status='PNP_FAIL', R_wc=None, t_wc=None,
                   n_matched=0, n_inliers=0, reproj_rmse=np.nan,
                   pts3d_alive=None, pts2d_alive=None)

        if len(pts2d_prev) < 6:
            out['status'] = 'TRACK_FAIL'
            return out

        # Forward-backward KLT consistency: track prev->cur, then cur->prev,
        # and drop features whose round-trip distance exceeds 1 px. Standard
        # robustness trick for KLT under occlusion, lighting changes, and
        # large inter-frame displacement (outdoor walking).
        pts2d_prev_f = pts2d_prev.reshape(-1, 1, 2).astype(np.float32)
        lk_kwargs = dict(
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        pts2d_cur_f, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_left_rect, cur_left, pts2d_prev_f, None, **lk_kwargs
        )
        pts2d_back_f, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            cur_left, self.prev_left_rect, pts2d_cur_f, None, **lk_kwargs
        )
        rt_dist = np.linalg.norm(
            pts2d_back_f.reshape(-1, 2) - pts2d_prev, axis=1
        )
        good = ((status_fwd.ravel() == 1) &
                (status_bwd.ravel() == 1) &
                (rt_dist < 1.0))
        out['n_matched'] = int(good.sum())
        if good.sum() < 6:
            out['status'] = 'TRACK_FAIL'
            return out

        pts3d_matched = pts3d_prev[good]
        pts2d_matched = pts2d_cur_f.reshape(-1, 2)[good]
        n_matched = len(pts3d_matched)

        # Survivors are exposed regardless of PnP outcome -- if PnP fails for
        # this frame the caller still wants to know which landmarks tracked
        # so it can keyframe deliberately rather than blindly.
        out['pts3d_alive'] = pts3d_matched
        out['pts2d_alive'] = pts2d_matched

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts3d_matched.astype(np.float64),
            pts2d_matched.astype(np.float64),
            self.K_rect,
            None,
            iterationsCount=200,
            reprojectionError=self.pnp_reproj_err,
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success or inliers is None or len(inliers) < 6:
            return out

        n_inliers = len(inliers)
        out['n_inliers'] = n_inliers

        inlier_idx    = inliers.ravel()
        pts3d_inliers = pts3d_matched[inlier_idx]
        pts2d_inliers = pts2d_matched[inlier_idx]

        # Motion-only bundle adjustment: refine (rvec, tvec) by Levenberg-
        # Marquardt minimization of inlier reprojection error. RANSAC gives
        # a coarse inlier-consistent estimate; LM refinement reduces per-
        # frame pose noise, which dominates long-sequence drift outdoors.
        # Project guide Section IV.C / V.B mentions motion-only BA as an
        # optional refinement step.
        rvec, tvec = cv2.solvePnPRefineLM(
            pts3d_inliers.astype(np.float64),
            pts2d_inliers.astype(np.float64),
            self.K_rect,
            None,
            rvec, tvec,
        )

        # Reprojection RMSE on inliers AFTER refinement.
        projected, _ = cv2.projectPoints(
            pts3d_inliers, rvec, tvec, self.K_rect, None
        )
        projected = projected.reshape(-1, 2)
        reproj_err = np.linalg.norm(pts2d_inliers - projected, axis=1)
        reproj_rmse = float(np.sqrt(np.mean(reproj_err ** 2)))
        out['reproj_rmse'] = reproj_rmse

        if n_matched > 0 and n_inliers / n_matched < self.min_inlier_ratio:
            out['status'] = 'LOW_INLIER'
            return out

        R_pnp, _ = cv2.Rodrigues(rvec)
        out['R_wc']   = R_pnp
        out['t_wc']   = tvec.reshape(3, 1)
        out['status'] = 'OK'
        return out

    def _refresh_map(self, left_rect, right_rect):
        pts2d_new = self._detect_features(left_rect)
        pts3d_new, pts2d_new = self._get_3d_points(left_rect, right_rect, pts2d_new)
        if len(pts3d_new) >= 6:
            self.prev_pts3d = pts3d_new
            self.prev_pts2d = pts2d_new

    def _log(self, status, ts=None, n_matched=0, n_inliers=0,
             reproj_rmse=np.nan, step_rot=np.nan, step_trans=np.nan):
        self.frame_status.append({
            'frame_idx': self.frame_idx,
            'timestamp_ns': ts,
            'status': status,
            'n_matched': int(n_matched),
            'n_inliers': int(n_inliers),
            'reproj_rmse_px': reproj_rmse,
            'step_rot_deg': step_rot,
            'step_trans_m': step_trans,
        })

    def _start_new_keyframe(self, left_rect, right_rect):
        """
        Promote the current frame to a new keyframe:
        - Snapshot the current world pose as the keyframe pose
        - Re-detect features and re-triangulate 3D points in this frame
        - These 3D points are now expressed in the new keyframe's camera frame
        Returns True if the new keyframe has enough landmarks to track from.
        """
        self.kf_R = self.cur_R.copy()
        self.kf_t = self.cur_t.copy()
        self._refresh_map(left_rect, right_rect)
        self.n_keyframes += 1
        self.kf_n_anchor = len(self.prev_pts3d) if self.prev_pts3d is not None else 0
        return self.prev_pts3d is not None and len(self.prev_pts3d) >= 6

    def _replenish_features(self, left_rect, right_rect, R_kf_cur, t_kf_cur):
        """
        Detect fresh features in the current frame, triangulate via
        disparity, and transform their 3D coordinates from the current
        frame into the keyframe frame so they extend the alive set without
        breaking the keyframe pose anchor.

        Transform: x_kf = R_kf_cur @ x_cur + t_kf_cur (T_kf_cur is the
        inverse of the PnP-returned T_cur_kf).
        """
        new_pts2d_seed = self._detect_features(left_rect)
        if len(new_pts2d_seed) == 0:
            return False
        new_pts3d_cur, new_pts2d_valid = self._get_3d_points(
            left_rect, right_rect, new_pts2d_seed
        )
        if len(new_pts3d_cur) == 0:
            return False
        # x_kf = R_kf_cur @ x_cur + t_kf_cur
        new_pts3d_kf = (R_kf_cur @ new_pts3d_cur.T).T + t_kf_cur.ravel()
        if self.prev_pts3d is None or len(self.prev_pts3d) == 0:
            self.prev_pts3d = new_pts3d_kf
            self.prev_pts2d = new_pts2d_valid
        else:
            self.prev_pts3d = np.vstack([self.prev_pts3d, new_pts3d_kf])
            self.prev_pts2d = np.vstack([self.prev_pts2d, new_pts2d_valid])
        self.kf_n_anchor = len(self.prev_pts3d)
        self.n_replenish += 1
        return True

    def process_frame(self, left_img, right_img, timestamp):
        left_img  = self._preprocess(left_img)
        right_img = self._preprocess(right_img)
        left_rect, right_rect = self._rectify(left_img, right_img)

        # First frame: initialize as keyframe #0, emit identity pose
        if self.prev_left_rect is None:
            pts2d = self._detect_features(left_rect)
            pts3d, pts2d_valid = self._get_3d_points(left_rect, right_rect, pts2d)
            self.prev_left_rect = left_rect
            self.prev_pts2d     = pts2d_valid
            self.prev_pts3d     = pts3d
            self.kf_R = np.eye(3)
            self.kf_t = np.zeros((3, 1))
            self.n_keyframes = 1
            self.timestamps.append(timestamp)   # paired with poses[0] = I
            self._log('INIT', ts=timestamp, n_matched=len(pts3d))
            self.frame_idx += 1
            print(f"[StereoVO] Initialized: {len(pts3d)} 3D points (KF#0)")
            return np.eye(4)

        # PnP pose estimation -- 3D points are in keyframe frame, so the
        # returned (R_pnp, t_pnp) is T_cur_kf (transforms KF-frame points to
        # cur-frame: x_cur = R_pnp @ x_kf + t_pnp).
        res          = self._pnp_pose(self.prev_pts3d, self.prev_pts2d, left_rect)
        pnp_status   = res['status']
        n_matched    = res['n_matched']
        n_inliers    = res['n_inliers']
        reproj_rmse  = res['reproj_rmse']
        R_pnp, t_pnp = res['R_wc'], res['t_wc']
        pts3d_alive  = res['pts3d_alive']
        pts2d_alive  = res['pts2d_alive']

        # ---- Hard-skip failure paths ----
        # All three failure modes force a new keyframe so the pipeline
        # doesn't keep PnP'ing against a stale map. We don't emit a pose
        # for this frame (excluded from the trajectory entirely).
        if pnp_status in ('TRACK_FAIL', 'PNP_FAIL', 'LOW_INLIER'):
            log_status = {
                'TRACK_FAIL': 'TRACK_FAIL',
                'PNP_FAIL'  : 'PNP_FAIL',
                'LOW_INLIER': 'PNP_LOW_INLIER',
            }[pnp_status]
            self._log(log_status, ts=timestamp,
                      n_matched=n_matched, n_inliers=n_inliers,
                      reproj_rmse=reproj_rmse)
            self._start_new_keyframe(left_rect, right_rect)
            self.prev_left_rect = left_rect
            self.frame_idx    += 1
            if pnp_status == 'TRACK_FAIL':
                self.n_track_fail += 1
            else:
                self.n_pnp_fail += 1
            return self._build_pose()

        # ---- Compute world pose from keyframe-relative PnP ----
        # T_w_cur = T_w_kf @ inv(T_cur_kf)
        R_kf_cur = R_pnp.T
        t_kf_cur = -(R_pnp.T @ t_pnp)
        cur_R_new = self.kf_R @ R_kf_cur
        cur_t_new = self.kf_t + self.kf_R @ t_kf_cur

        # ---- Single-step motion sanity (cur vs PREVIOUS frame) ----
        # Per-frame caps are still meaningful: a 20 Hz handheld can't
        # actually jump 0.5 m or 25 deg between consecutive frames. We
        # compute prev->cur relative motion from the world poses.
        R_prev_cur = self.cur_R.T @ cur_R_new
        t_prev_cur = self.cur_R.T @ (cur_t_new - self.cur_t)
        step_rot_rad = np.arccos(
            np.clip((np.trace(R_prev_cur) - 1.0) * 0.5, -1.0, 1.0)
        )
        step_rot_deg = float(np.degrees(step_rot_rad))
        step_trans   = float(np.linalg.norm(t_prev_cur))

        if (step_rot_deg > self.max_step_rot_deg or
                step_trans > self.max_step_trans):
            self._log('STEP_REJECT', ts=timestamp,
                      n_matched=n_matched, n_inliers=n_inliers,
                      reproj_rmse=reproj_rmse,
                      step_rot=step_rot_deg, step_trans=step_trans)
            self._start_new_keyframe(left_rect, right_rect)
            self.prev_left_rect = left_rect
            self.frame_idx     += 1
            self.n_step_reject += 1
            return self._build_pose()

        # ---- Success — commit pose ----
        self.cur_R = cur_R_new
        self.cur_t = cur_t_new

        is_low_conf = reproj_rmse > self.max_reproj_rmse
        T = self._build_pose()
        self.poses.append(T)
        self.timestamps.append(timestamp)
        if is_low_conf:
            self.n_reproj_low_conf += 1
            self._log('REPROJ_LOW_CONF', ts=timestamp,
                      n_matched=n_matched, n_inliers=n_inliers,
                      reproj_rmse=reproj_rmse,
                      step_rot=step_rot_deg, step_trans=step_trans)
        else:
            self._log('OK', ts=timestamp,
                      n_matched=n_matched, n_inliers=n_inliers,
                      reproj_rmse=reproj_rmse,
                      step_rot=step_rot_deg, step_trans=step_trans)

        # ---- Keyframe decision ----
        # Three triggers: feature attrition (KLT loss), translation from KF,
        # rotation from KF. Each captures a different reason the current
        # keyframe's landmarks are no longer well-suited for tracking.
        trans_from_kf   = float(np.linalg.norm(t_kf_cur))
        rot_from_kf_rad = np.arccos(
            np.clip((np.trace(R_kf_cur) - 1.0) * 0.5, -1.0, 1.0)
        )
        rot_from_kf_deg = float(np.degrees(rot_from_kf_rad))

        need_new_kf = (
            n_matched < self.min_kf_features or
            trans_from_kf > self.max_kf_trans or
            rot_from_kf_deg > self.max_kf_rot_deg
        )

        if need_new_kf:
            self._start_new_keyframe(left_rect, right_rect)
        else:
            # Slide tracking forward: keep the same keyframe map, but the
            # next frame's KLT input is "where these landmarks were seen in
            # the cur image". The 3D coords are unchanged (still in KF frame).
            self.prev_pts3d = pts3d_alive
            self.prev_pts2d = pts2d_alive

            # Replenishment: if alive features have attrited significantly
            # below the keyframe's anchor count, fold in fresh features
            # from cur. Skipped when the alive set is already healthy so
            # this only fires in translation-heavy scenes (where attrition
            # is asymmetric and bias toward far points hurts PnP).
            if (self.kf_n_anchor > 0 and
                    n_matched < self.replenish_ratio * self.kf_n_anchor):
                self._replenish_features(
                    left_rect, right_rect, R_kf_cur, t_kf_cur
                )

        self.prev_left_rect = left_rect
        self.frame_idx     += 1
        return T

    def _save_status(self, output_path):
        status_path = output_path.replace('.txt', '_status.csv')
        with open(status_path, 'w') as f:
            f.write("frame_idx,timestamp_ns,status,n_matched,n_inliers,"
                    "reproj_rmse_px,step_rot_deg,step_trans_m\n")
            for s in self.frame_status:
                ts  = '' if s['timestamp_ns'] is None else s['timestamp_ns']
                rep = '' if np.isnan(s['reproj_rmse_px']) else f"{s['reproj_rmse_px']:.4f}"
                rot = '' if np.isnan(s['step_rot_deg'])  else f"{s['step_rot_deg']:.4f}"
                tr  = '' if np.isnan(s['step_trans_m'])  else f"{s['step_trans_m']:.4f}"
                f.write(f"{s['frame_idx']},{ts},{s['status']},"
                        f"{s['n_matched']},{s['n_inliers']},{rep},{rot},{tr}\n")
        print(f"[StereoVO] Status log: {status_path}")

    def run(self, loader, output_path, max_frames=None):
        os.makedirs(
            os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
            exist_ok=True
        )
        n = len(loader) if max_frames is None else min(len(loader), max_frames)
        print(f"\n[StereoVO] Running on {n} frames...")

        for i in range(n):
            left_img, right_img, ts = loader.get_frame(i)
            T = self.process_frame(left_img, right_img, ts)
            if i % 100 == 0:
                t = T[:3, 3]
                n3d = len(self.prev_pts3d) if self.prev_pts3d is not None else 0
                print(f"  Frame {i:4d}/{n} | "
                      f"t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]m | "
                      f"3D pts={n3d} | "
                      f"KF={self.n_keyframes} | "
                      f"track-fail={self.n_track_fail} | "
                      f"PnP-fail={self.n_pnp_fail} | "
                      f"low-conf={self.n_reproj_low_conf} | "
                      f"step-reject={self.n_step_reject}")

        save_trajectory(output_path, self.timestamps, self.poses)
        self._save_status(output_path)

        n_emitted = len(self.poses)
        n_skip = (self.n_track_fail + self.n_pnp_fail + self.n_step_reject)
        print(f"\n[StereoVO] Done | emitted={n_emitted}/{n} poses | "
              f"KF={self.n_keyframes} ({n/max(self.n_keyframes,1):.1f} frames/KF) | "
              f"replenish={self.n_replenish} | "
              f"track-fail={self.n_track_fail} PnP-fail={self.n_pnp_fail} "
              f"step-reject={self.n_step_reject} "
              f"low-conf={self.n_reproj_low_conf} "
              f"(skip={100.0*n_skip/max(n,1):.1f}%, "
              f"low-conf={100.0*self.n_reproj_low_conf/max(n,1):.1f}%)")
        print(f"[StereoVO] Saved: {output_path}")
        return self.poses, self.timestamps
