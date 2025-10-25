# feature_extractor/media_io.py

import os
import cv2
import numpy as np
import tempfile
import subprocess
import librosa
from typing import List, Tuple, Optional
from .config import FPS_TARGET, MAX_FRAMES, MAX_VIDEO_SECONDS

def _rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

def load_image_frames(path: str):
    """
    Returns:
        frames_rgb: [np.ndarray(H,W,3) uint8]
        duration_used_s: 0.0
        sampling_fps: None
        duration_total_s: 0.0
        modality: "image"
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image {path}")
    frames_rgb = [_rgb(img_bgr)]
    return frames_rgb, 0.0, None, 0.0, "image"

def load_video_frames(path: str,
                      fps_target: int = FPS_TARGET,
                      max_frames: int = MAX_FRAMES,
                      max_seconds: int = MAX_VIDEO_SECONDS):
    """
    Sample frames from a video by timestamp seeking (uniform over first max_seconds).
    Returns:
        frames_rgb: list of RGB uint8
        duration_used_s: float (<= max_seconds)
        sampling_fps: float (our effective sampling fps)
        duration_total_s: float (full video theoretical duration from metadata)
        modality: "video"
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video {path}")

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 1e-3 or np.isnan(orig_fps):
        orig_fps = 30.0  # fallback guess
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if frame_count <= 0:
        frame_count = orig_fps * 10.0  # fallback guess
    duration_total = frame_count / orig_fps

    clip_duration = min(duration_total, max_seconds)
    target_total_frames = min(max_frames, int(round(fps_target * clip_duration)))
    if target_total_frames < 1:
        target_total_frames = 1

    frames_rgb = []
    timestamps = np.linspace(0.0, clip_duration, num=target_total_frames, endpoint=False)

    for t in timestamps:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(t * 1000.0))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frames_rgb.append(_rgb(frame_bgr))

    cap.release()

    # effective sampling fps from what we actually got
    duration_used = clip_duration
    if duration_used <= 1e-6:
        sampling_fps = None
    else:
        sampling_fps = len(frames_rgb) / duration_used

    return frames_rgb, duration_used, sampling_fps, duration_total, "video"

def extract_audio_wav(path: str,
                      sr: int = 16000,
                      max_seconds: int = MAX_VIDEO_SECONDS) -> Tuple[Optional[np.ndarray], Optional[int], float]:
    """
    Extract mono audio from first max_seconds of video via ffmpeg, resample to sr.
    Returns:
        y: float32 waveform in [-1,1] or None if no audio.
        sr: sample rate or None
        duration_s: float duration actually returned
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        tmp_name = tmp_wav.name

    # Run ffmpeg: first max_seconds, mono, sr Hz
    cmd = [
        "ffmpeg",
        "-y",
        "-i", path,
        "-t", str(max_seconds),
        "-vn",
        "-ac", "1",
        "-ar", str(sr),
        tmp_name
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError:
        # no audio or ffmpeg fail
        try:
            os.remove(tmp_name)
        except Exception:
            pass
        return None, None, 0.0

    # Load with librosa
    try:
        y, sr_out = librosa.load(tmp_name, sr=sr, mono=True)
    except Exception:
        y, sr_out = None, None
    finally:
        try:
            os.remove(tmp_name)
        except Exception:
            pass

    if y is None or len(y) == 0:
        return None, None, 0.0

    dur_s = len(y) / float(sr_out)
    return y.astype(np.float32), sr_out, dur_s
