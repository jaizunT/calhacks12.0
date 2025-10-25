# feature_extractor/utils.py

import numpy as np
import cv2
from typing import List, Tuple
import torch
import math

def letterbox_resize(img_rgb: np.ndarray, target: int = 256) -> np.ndarray:
    """
    Resize img to fit in target x target, preserving aspect, pad with black.
    img_rgb: uint8 (H,W,3) RGB
    """
    h, w = img_rgb.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target, target, 3), dtype=np.uint8)
    top = (target - nh) // 2
    left = (target - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def michelson_contrast(gray: np.ndarray) -> float:
    """Michelson contrast = (Imax - Imin)/(Imax + Imin)."""
    gmin = float(np.min(gray))
    gmax = float(np.max(gray))
    denom = (gmax + gmin)
    if denom <= 1e-6:
        return 0.0
    return (gmax - gmin) / denom

def edge_density(img_rgb: np.ndarray) -> float:
    """
    Approx "clutter": fraction of pixels that are strong edges.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.mean(edges > 0)  # ratio of edge pixels

def warmth_index(img_rgb: np.ndarray) -> float:
    """
    Crude warmth: mean(R)/mean(B).
    Warm ads (reds/yellows) => higher ratio, cold (blues) => lower.
    """
    r = np.mean(img_rgb[..., 0])
    b = np.mean(img_rgb[..., 2]) + 1e-6
    return float(r / b)

def frame_saliency_score(img_rgb: np.ndarray) -> float:
    """
    Heuristic saliency via Laplacian variance (sharpness = "interesting").
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def pick_representative_indices(frames: List[np.ndarray]) -> List[int]:
    """
    We want: first, middle, last, and most 'salient'.
    Returns unique sorted indices.
    """
    if not frames:
        return []
    n = len(frames)
    idxs = {0, n//2, n-1}
    # find most "salient" by Laplacian variance
    sal_scores = [frame_saliency_score(f) for f in frames]
    sal_idx = int(np.argmax(sal_scores))
    idxs.add(sal_idx)
    return sorted(list(idxs))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-9
    bn = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (an * bn))

def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)

def safe_mean(vals: List[float]) -> float:
    if len(vals) == 0:
        return 0.0
    return float(np.mean(vals))

def safe_var(vals: List[float]) -> float:
    if len(vals) <= 1:
        return 0.0
    return float(np.var(vals))

def dbfs_from_wav(y: np.ndarray) -> float:
    """
    Approx loudness in dBFS from waveform y in [-1,1].
    """
    rms = math.sqrt(float(np.mean(y**2)) + 1e-12)
    dbfs = 20.0 * math.log10(rms + 1e-12)
    return float(dbfs)
