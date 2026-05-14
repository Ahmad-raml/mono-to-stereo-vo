"""
stereo_vo/disparity.py
Disparity estimation tuned for TUM VI fisheye stereo.

`num_disparities` and `block_size` are exposed because outdoor sequences
need a wider disparity search range (closer-than-room-scale baseline plus
deeper scenes) than indoor ones.
"""
import numpy as np
import cv2


class DisparityEstimator:
    def __init__(self, method='SGBM', num_disparities=64, block_size=7):
        self.method = method
        # OpenCV requires numDisparities to be a multiple of 16
        num_disparities = max(16, int(round(num_disparities / 16.0)) * 16)
        # SGBM block_size must be odd and >= 3
        if block_size < 3:
            block_size = 3
        if block_size % 2 == 0:
            block_size += 1

        if method == 'SGBM':
            # P1/P2 follow OpenCV's recommended scaling: 8/32 * channels * blockSize^2
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=8  * 1 * block_size**2,
                P2=32 * 1 * block_size**2,
                disp12MaxDiff=1,
                uniquenessRatio=5,
                speckleWindowSize=50,
                speckleRange=2,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            # StereoBM block_size must be odd and >= 5
            bm_block = max(5, block_size if block_size % 2 == 1 else block_size + 1)
            self.stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=bm_block
            )

        print(f"[Disparity] Method: {method} | num_disparities={num_disparities} | "
              f"block_size={block_size}")

    def compute(self, left_rect, right_rect):
        disp = self.stereo.compute(left_rect, right_rect)
        disp = disp.astype(np.float32) / 16.0
        disp[disp <= 0] = 0
        return disp

    def disparity_to_depth(self, disparity, focal_length, baseline):
        depth = np.zeros_like(disparity)
        valid = disparity > 0.5
        depth[valid] = (focal_length * baseline) / disparity[valid]
        depth[(depth < 0.1) & (depth > 0)] = 0
        depth[depth > 30.0] = 0
        return depth

    def visualize_disparity(self, disparity):
        disp_vis = disparity.copy()
        disp_vis = cv2.normalize(disp_vis, None, 0, 255, cv2.NORM_MINMAX)
        disp_vis = disp_vis.astype(np.uint8)
        return cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
