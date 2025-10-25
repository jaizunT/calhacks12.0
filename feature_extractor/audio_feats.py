# feature_extractor/audio_feats.py

import numpy as np
import webrtcvad
import librosa
import torch
from typing import Dict, Any, List, Optional
from .utils import dbfs_from_wav, safe_mean
from .config import CTA_KEYWORDS

def _vad_ratio(y: np.ndarray, sr: int) -> float:
    """
    Voice Activity Detection fraction using webrtcvad.
    y: float32 [-1,1]
    """
    if y is None or len(y) == 0 or sr != 16000:
        # webrtcvad expects 16k mono PCM16
        return 0.0

    vad = webrtcvad.Vad(2)  # 0-3 aggressiveness
    frame_dur_ms = 30
    frame_len = int(sr * frame_dur_ms / 1000)  # e.g. 480 samples at 16kHz
    total_frames = 0
    speech_frames = 0

    # convert float32 [-1,1] to int16 PCM bytes
    wav16 = (y * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()

    for start in range(0, len(wav16), frame_len*2):  # 2 bytes/int16
        chunk = wav16[start:start+frame_len*2]
        if len(chunk) < frame_len*2:
            break
        total_frames += 1
        if vad.is_speech(chunk, sr):
            speech_frames += 1

    if total_frames == 0:
        return 0.0
    return float(speech_frames / total_frames)

def _tempo_bpm(y: np.ndarray, sr: int) -> Optional[float]:
    if y is None or len(y) == 0:
        return None
    try:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo)
    except Exception:
        return None

def _transcribe_cta(y: np.ndarray,
                    sr: int,
                    whisper_model,
                    max_words=200) -> Dict[str, Any]:
    """
    Run ASR with Whisper tiny/base/etc.
    Returns transcript, cta hits, earliest CTA timestamp.
    """
    if y is None or len(y) == 0:
        return {
            "transcript": "",
            "asr_cta_hits": [],
            "asr_cta_time_s": None
        }

    # Whisper expects float32 16k mono in [-1,1] which we already have.
    # We'll give raw audio directly.
    result = whisper_model.transcribe(y, fp16=False)

    transcript = result.get("text", "").strip()
    hits = []
    earliest = None

    segs = result.get("segments", [])
    for seg in segs:
        seg_text = seg.get("text","").lower()
        start_t = float(seg.get("start", 0.0))
        for kw in CTA_KEYWORDS:
            if kw in seg_text:
                hits.append(kw)
                if earliest is None or start_t < earliest:
                    earliest = start_t

    return {
        "transcript": transcript,
        "asr_cta_hits": list(sorted(set(hits))),
        "asr_cta_time_s": earliest
    }

def audio_features(y: np.ndarray,
                   sr: int,
                   whisper_model) -> Dict[str, Any]:
    """
    Loudness, loudness first 1s, speech_ratio, bpm, transcript, CTA from audio.
    """
    if y is None or len(y) == 0:
        return {
            "loudness_dbfs": None,
            "loudness_dbfs_0_1s": None,
            "speech_ratio": 0.0,
            "bpm": None,
            "transcript": "",
            "asr_cta_hits": [],
            "asr_cta_time_s": None
        }

    full_dbfs = dbfs_from_wav(y)
    first_1s = y[:sr] if len(y) >= sr else y
    first_dbfs = dbfs_from_wav(first_1s)

    speech_r = _vad_ratio(y, sr)
    bpm_val = _tempo_bpm(y, sr)

    asr_info = _transcribe_cta(y, sr, whisper_model)

    return {
        "loudness_dbfs": full_dbfs,
        "loudness_dbfs_0_1s": first_dbfs,
        "speech_ratio": float(speech_r),
        "bpm": bpm_val,
        "transcript": asr_info["transcript"],
        "asr_cta_hits": asr_info["asr_cta_hits"],
        "asr_cta_time_s": asr_info["asr_cta_time_s"]
    }
