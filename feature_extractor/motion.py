# feature_extractor/motion.py

import numpy as np
import cv2
from typing import List, Dict, Any
from .utils import safe_mean, safe_var

def _optical_flow_mag(frames_rgb: List[np.ndarray], fps: float) -> float:
    """
    Mean optical flow magnitude over ~first second.
    """
    if fps is None or fps <= 1e-9:
        fps = 1.0
    # how many pairs in first second?
    max_pairs = int(round(min(fps, len(frames_rgb)-1)))
    if max_pairs < 1:
        return 0.0

    mags = []
    for i in range(max_pairs):
        f1 = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        f2 = cv2.cvtColor(frames_rgb[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            f1, f2,
            None,
            pyr_scale=0.5,
            levels=1,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        mags.append(float(np.mean(mag)))
    return safe_mean(mags)

def _scene_cuts(frames_rgb: List[np.ndarray], duration_used_s: float) -> (float, float, float):
    """
    Detect scene cuts by histogram diff. Return:
    - first_cut_time_s
    - cut_rate_per_5s
    - total_cut_count
    """
    if len(frames_rgb) < 2:
        return None, 0.0, 0

    hists = []
    for f in frames_rgb:
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv],[0,1,2],[0,0,0],[16,16,16],[0,180,0,256,0,256])
        hist = cv2.normalize(hist, None).flatten()
        hists.append(hist)

    cut_indices = []
    for i in range(len(hists)-1):
        # 1 - correlation as "difference"
        corr = cv2.compareHist(hists[i].astype(np.float32),
                               hists[i+1].astype(np.float32),
                               cv2.HISTCMP_CORREL)
        if corr < 0.8:  # threshold
            cut_indices.append(i+1)

    if duration_used_s <= 1e-6:
        duration_used_s = max(1.0, len(frames_rgb)/3.0)  # fallback guess

    first_cut_time_s = None
    if len(cut_indices) > 0:
        first_idx = cut_indices[0]
        # map frame index -> time. assume uniform sampling:
        frame_time = duration_used_s * (first_idx / max(1, len(frames_rgb)))
        first_cut_time_s = float(frame_time)

    # cuts per 5s
    total_cuts = len(cut_indices)
    cut_rate_per_5s = 0.0
    if duration_used_s > 0:
        cut_rate_per_5s = total_cuts / (duration_used_s / 5.0)

    return first_cut_time_s, float(cut_rate_per_5s), total_cuts

def _camera_shake(frames_rgb: List[np.ndarray]) -> float:
    """
    Approximate "shake": variance of translation between consecutive frames.
    """
    if len(frames_rgb) < 2:
        return 0.0

    trans_mags = []
    for i in range(len(frames_rgb)-1):
        prev_gray = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
        next_gray = cv2.cvtColor(frames_rgb[i+1], cv2.COLOR_RGB2GRAY)

        # keypoints
        pts_prev = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30)
        pts_next, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, pts_prev, None)
        if pts_prev is None or pts_next is None:
            continue

        # RANSAC estimate affine
        good_prev = pts_prev[st==1]
        good_next = pts_next[st==1]
        if len(good_prev) < 4 or len(good_next) < 4:
            continue

        M, inliers = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC)
        if M is None:
            continue

        dx, dy = M[0,2], M[1,2]
        mag = np.sqrt(dx*dx + dy*dy)
        trans_mags.append(float(mag))

    return safe_var(trans_mags)

def motion_features(frames_rgb: List[np.ndarray],
                    sampling_fps: float,
                    duration_used_s: float) -> Dict[str, Any]:
    """
    Get:
      - motion_intensity_0_1s
      - first_cut_time_s
      - cut_rate_per_5s
      - shake_var
    """
    if len(frames_rgb) == 0:
        return {
            "motion_intensity_0_1s": 0.0,
            "first_cut_time_s": None,
            "cut_rate_per_5s": 0.0,
            "shake_var": 0.0
        }

    mot_int = _optical_flow_mag(frames_rgb, sampling_fps if sampling_fps else 1.0)
    first_cut_t, cut_rate_5s, _ = _scene_cuts(frames_rgb, duration_used_s)
    shake_v = _camera_shake(frames_rgb)

    return {
        "motion_intensity_0_1s": float(mot_int),
        "first_cut_time_s": first_cut_t,
        "cut_rate_per_5s": float(cut_rate_5s),
        "shake_var": float(shake_v)
    }
