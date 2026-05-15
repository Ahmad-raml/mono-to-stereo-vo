import numpy as np
import cv2


class FeatureExtractor:
    def __init__(self, method='ORB', n_features=2000, ratio_thresh=0.75):
        self.method = method.upper()
        self.ratio_thresh = ratio_thresh
        if self.method == 'ORB':
            self.detector = cv2.ORB_create(nfeatures=n_features, scaleFactor=1.2,
                nlevels=8, edgeThreshold=15, WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE, patchSize=31, fastThreshold=10)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        elif self.method == 'SIFT':
            self.detector = cv2.SIFT_create(nfeatures=n_features)
            self.matcher  = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            raise ValueError(f"Unknown method '{method}'. Use 'ORB' or 'SIFT'.")
        print(f"[FeatureExtractor] {self.method} | max_features={n_features} | ratio={ratio_thresh}")

    def detect_and_compute(self, img):
        return self.detector.detectAndCompute(img, None)

    def match(self, desc1, desc2):
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good = []
        for pair in matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)
        return good

    def detect_match(self, img1, img2):
        kps1, desc1 = self.detect_and_compute(img1)
        kps2, desc2 = self.detect_and_compute(img2)
        matches = self.match(desc1, desc2)
        if len(matches) == 0:
            return np.array([]), np.array([]), kps1, kps2, matches
        pts1 = np.float32([kps1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kps2[m.trainIdx].pt for m in matches])
        return pts1, pts2, kps1, kps2, matches

    def track_optical_flow(self, img1, img2, pts1):
        if len(pts1) == 0:
            return np.array([]), np.array([])
        pts1_f = pts1.reshape(-1, 1, 2).astype(np.float32)
        lk_params = dict(winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        pts2_f, status, _ = cv2.calcOpticalFlowPyrLK(img1, img2, pts1_f, None, **lk_params)
        status = status.ravel()
        return pts1[status == 1], pts2_f.reshape(-1, 2)[status == 1]