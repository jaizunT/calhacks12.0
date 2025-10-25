# feature_extractor/multimodal.py

import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .config import MOTION_THRESH, LOUDNESS_THRESH

def _encode_text_for_align(clip_model,
                           tokenize,
                           backend: str,
                           device: str,
                           text: str) -> np.ndarray:
    if text is None or len(text.strip()) == 0:
        return np.zeros((512,), dtype=np.float32)
    toks = tokenize([text]).to(device)
    with torch.no_grad():
        txt_emb = clip_model.encode_text(toks)
    if backend == "open_clip":
        txt_emb = F.normalize(txt_emb, dim=-1)
    else:
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    return txt_emb[0].detach().cpu().numpy()

def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-9
    bn = np.linalg.norm(b) + 1e-9
    return float(np.dot(a,b)/(an*bn))

def hook_score_fn(motion_intensity_0_1s: float,
                  loudness_dbfs_0_1s: Optional[float],
                  cta_time_s: Optional[float],
                  first_cut_time_s: Optional[float]) -> float:
    """
    Simple interpretable hook score:
    +1 if early motion is "high"
    +1 if audio loudness in first 1s is "high"
    +1 if CTA appears within 1.5s
    +1 if we cut within 0.5s
    Range: 0..4
    """
    score = 0.0
    if motion_intensity_0_1s is not None and motion_intensity_0_1s > MOTION_THRESH:
        score += 1.0
    if loudness_dbfs_0_1s is not None and loudness_dbfs_0_1s > LOUDNESS_THRESH:
        score += 1.0
    if cta_time_s is not None and cta_time_s <= 1.5:
        score += 1.0
    if first_cut_time_s is not None and first_cut_time_s <= 0.5:
        score += 1.0
    return float(score)

def multimodal_features(vision_feats: Dict[str,Any],
                        motion_feats: Dict[str,Any],
                        audio_feats: Dict[str,Any],
                        clip_model,
                        tokenize,
                        backend: str,
                        device: str) -> Dict[str,Any]:
    """
    - Align OCR text ↔ visuals
    - Align ASR transcript ↔ visuals
    - Hook score
    """
    clip_img_mean = np.array(vision_feats["clip_img_mean"], dtype=np.float32)

    ocr_txt = vision_feats.get("ocr_fulltext","")
    asr_txt = audio_feats.get("transcript","")

    ocr_emb = _encode_text_for_align(clip_model, tokenize, backend, device, ocr_txt)
    asr_emb = _encode_text_for_align(clip_model, tokenize, backend, device, asr_txt)

    align_ocr_img = cosine_similarity_np(ocr_emb, clip_img_mean)
    align_asr_img = cosine_similarity_np(asr_emb, clip_img_mean)

    hook_score = hook_score_fn(
        motion_feats.get("motion_intensity_0_1s", 0.0),
        audio_feats.get("loudness_dbfs_0_1s", None),
        vision_feats.get("cta_time_s", None),
        motion_feats.get("first_cut_time_s", None)
    )

    return {
        "align_ocr_img": float(align_ocr_img),
        "align_asr_img": float(align_asr_img),
        "hook_score": hook_score
    }
