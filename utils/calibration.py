import numpy as np
import cv2
import yaml


class Calibration:
    def __init__(self, camchain_path):
        with open(camchain_path, "r") as f:
            data = yaml.safe_load(f)

        fx0, fy0, cx0, cy0 = data["cam0"]["intrinsics"]
        self.K0 = np.array([[fx0,0,cx0],[0,fy0,cy0],[0,0,1]], dtype=np.float64)
        self.D0 = np.array(data["cam0"]["distortion_coeffs"], dtype=np.float64).reshape(4,1)

        fx1, fy1, cx1, cy1 = data["cam1"]["intrinsics"]
        self.K1 = np.array([[fx1,0,cx1],[0,fy1,cy1],[0,0,1]], dtype=np.float64)
        self.D1 = np.array(data["cam1"]["distortion_coeffs"], dtype=np.float64).reshape(4,1)

        self.resolution = tuple(data["cam0"]["resolution"])

        # Kalibr T_cn_cnm1: transform FROM cam(n-1)=cam0 TO cam(n)=cam1
        # i.e. T_cam1_cam0 — transforms points in cam0 frame to cam1 frame
        T_cam1_cam0 = np.array(data["cam1"]["T_cn_cnm1"], dtype=np.float64)
        self.T_cn_cnm1 = T_cam1_cam0

        # For stereoRectify: R, t describing cam1 pose relative to cam0
        # cv2.stereoRectify wants R, t such that: x_cam1 = R * x_cam0 + t
        # That is exactly T_cam1_cam0
        self.R_stereo = T_cam1_cam0[:3, :3]
        self.t_stereo = T_cam1_cam0[:3,  3].reshape(3, 1)
        self.baseline = abs(T_cam1_cam0[0, 3])  # horizontal baseline in meters

        print(f"[Calibration] Loaded: {camchain_path}")
        print(f"  cam0 fx={fx0:.2f} fy={fy0:.2f} cx={cx0:.2f} cy={cy0:.2f}")
        print(f"  cam1 fx={fx1:.2f} fy={fy1:.2f} cx={cx1:.2f} cy={cy1:.2f}")
        print(f"  Baseline : {self.baseline*100:.2f} cm")

    def undistort_image(self, img, cam='left'):
        K = self.K0 if cam == 'left' else self.K1
        D = self.D0 if cam == 'left' else self.D1
        h, w = img.shape[:2]
        K_new = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            K, D, (w, h), np.eye(3), balance=0.0)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K_new, (w, h), cv2.CV_16SC2)
        return cv2.remap(img, map1, map2, cv2.INTER_LINEAR), K_new

    def get_stereo_rectification(self):
        w, h = self.resolution

        K0 = self.K0.astype(np.float64)
        K1 = self.K1.astype(np.float64)
        D0 = self.D0.reshape(4, 1).astype(np.float64)
        D1 = self.D1.reshape(4, 1).astype(np.float64)
        R  = self.R_stereo.astype(np.float64)
        t  = self.t_stereo.astype(np.float64)

        # Get rectification rotations from fisheye stereoRectify
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            K0, D0, K1, D1,
            (w, h), R, t,
            flags=cv2.CALIB_ZERO_DISPARITY,
            balance=0.0,
            newImageSize=(w, h)
        )

        # cv2.fisheye.stereoRectify gives tiny fx for wide-FOV lenses.
        # Override: use original fx as the rectified focal length.
        # This gives a reasonable perspective projection of the central region.
        fx_new = float(self.K0[0, 0])   # ~191 px
        cx_new = w / 2.0
        cy_new = h / 2.0

        P1_new = np.array([
            [fx_new,   0,    cx_new,                    0],
            [0,      fx_new, cy_new,                    0],
            [0,        0,       1,                      0]
        ], dtype=np.float64)

        P2_new = np.array([
            [fx_new,   0,    cx_new, -fx_new * self.baseline],
            [0,      fx_new, cy_new,                       0],
            [0,        0,       1,                         0]
        ], dtype=np.float64)

        # Q matrix for disparity-to-depth (Z = fx*B/d)
        Q = np.array([
            [1, 0,  0,           -cx_new],
            [0, 1,  0,           -cy_new],
            [0, 0,  0,            fx_new],
            [0, 0, -1/self.baseline, 0  ]
        ], dtype=np.float64)

        # Build rectification maps using R1/R2 from fisheye but our custom P
        map1l, map2l = cv2.fisheye.initUndistortRectifyMap(
            K0, D0, R1, P1_new, (w, h), cv2.CV_16SC2
        )
        map1r, map2r = cv2.fisheye.initUndistortRectifyMap(
            K1, D1, R2, P2_new, (w, h), cv2.CV_16SC2
        )

        print(f"[Calibration] Rectification | fx={fx_new:.1f} | baseline={self.baseline*100:.2f} cm")
        return map1l, map2l, map1r, map2r, Q, P1_new, P2_new

    def rectify_stereo_pair(self, left_img, right_img, maps=None):
        if maps is None:
            map1l, map2l, map1r, map2r, Q, _, _ = self.get_stereo_rectification()
        else:
            map1l, map2l, map1r, map2r, Q = maps
        rect_left  = cv2.remap(left_img,  map1l, map2l, cv2.INTER_LINEAR)
        rect_right = cv2.remap(right_img, map1r, map2r, cv2.INTER_LINEAR)
        return rect_left, rect_right, Q