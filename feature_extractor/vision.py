# feature_extractor/vision.py

import numpy as np
import cv2
from PIL import Image
from typing import List, Dict, Any, Tuple
import pytesseract
import torch
import torch.nn.functional as F

from .config import (
    VERTICAL_CLASSES,
    CTA_KEYWORDS,
    FRAME_SIZE,
)
from .utils import (
    letterbox_resize,
    michelson_contrast,
    edge_density,
    warmth_index,
    pick_representative_indices,
    softmax,
    cosine_similarity,
    safe_mean,
)
import os

class VisionExtractor:
    """
    Handles:
    - CLIP embeddings (image + text)
    - zero-shot vertical classification
    - OCR, CTA timing/size
    - faces/smile
    - palette / clutter / contrast
    - logo stability
    """

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.backend = None

        # Try open_clip first
        self.open_clip = None
        self.clip = None
        try:
            import open_clip
            self.open_clip = open_clip
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32', pretrained='laion2b_s34b_b79k', device=device
            )
            self.clip_model = model.eval()
            self.clip_preprocess = preprocess
            self.tokenize = open_clip.get_tokenizer('ViT-B-32')
            self.backend = "open_clip"
        except Exception:
            # fallback to original clip
            import clip
            self.clip = clip
            model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
            self.clip_model = model.eval()
            self.clip_preprocess = preprocess
            self.tokenize = clip.tokenize
            self.backend = "clip"

        # cache text embeddings for vertical classes:
        self.vertical_text_emb, self.vertical_prompts = self._encode_vertical_prompts(VERTICAL_CLASSES)

        # Haar cascades
        haar_dir = cv2.data.haarcascades
        self.face_cascade = cv2.CascadeClassifier(os.path.join(
            haar_dir, "haarcascade_frontalface_default.xml"
        ))
        # Smile cascade sometimes not super reliable; we'll try
        self.smile_cascade = cv2.CascadeClassifier(os.path.join(
            haar_dir, "haarcascade_smile.xml"
        ))

    def _encode_text(self, texts: List[str]) -> torch.Tensor:
        if self.backend == "open_clip":
            toks = self.tokenize(texts).to(self.device)
            with torch.no_grad():
                txt = self.clip_model.encode_text(toks)
            txt = F.normalize(txt, dim=-1)
            return txt
        else:
            # self.backend == "clip"
            toks = self.tokenize(texts).to(self.device)
            with torch.no_grad():
                txt = self.clip_model.encode_text(toks)
            txt = txt / txt.norm(dim=-1, keepdim=True)
            return txt

    def _encode_vertical_prompts(self, class_list: List[str]):
        prompts = [f"This is an advertisement for {c}." for c in class_list]
        txt_emb = self._encode_text(prompts)
        return txt_emb, prompts

    def encode_images(self, frames_rgb: List[np.ndarray]) -> torch.Tensor:
        """
        frames_rgb: list of uint8 RGB
        returns torch.Tensor [N, D] normalized
        """
        if len(frames_rgb) == 0:
            return torch.zeros((0, 512), device=self.device)

        pil_list = []
        for f in frames_rgb:
            # preprocess wants PIL RGB of some size
            pil_list.append(Image.fromarray(f))

        batch_tensors = []
        for pil_img in pil_list:
            batch_tensors.append(self.clip_preprocess(pil_img).unsqueeze(0))
        batch = torch.cat(batch_tensors, dim=0).to(self.device)

        with torch.no_grad():
            img_emb = self.clip_model.encode_image(batch)

        # normalize
        if self.backend == "open_clip":
            img_emb = F.normalize(img_emb, dim=-1)
        else:
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        return img_emb  # [N,D]

    def classify_vertical(self, img_emb: torch.Tensor) -> Dict[str, Any]:
        """
        img_emb: [N,D] normalized
        returns dict with top1, top3, entropy, probs
        """
        if img_emb.shape[0] == 0:
            return {
                "vertical_top1": None,
                "vertical_top3": [],
                "vertical_entropy": 0.0,
                "vertical_probs": [0.0]*len(VERTICAL_CLASSES)
            }

        # mean embedding across frames
        mean_emb = img_emb.mean(dim=0, keepdim=True)
        mean_emb = mean_emb / mean_emb.norm(dim=-1, keepdim=True)

        # cosine similarity with each vertical prompt
        sims = (mean_emb @ self.vertical_text_emb.T).cpu().numpy().flatten()
        probs = softmax(sims)
        order = list(np.argsort(probs)[::-1])
        top1 = VERTICAL_CLASSES[order[0]]
        top3 = [VERTICAL_CLASSES[i] for i in order[:3]]

        # entropy
        from .config import entropy as entropy_fn
        ent = entropy_fn(probs.tolist())

        return {
            "vertical_top1": top1,
            "vertical_top3": top3,
            "vertical_entropy": float(ent),
            "vertical_probs": probs.tolist()
        }

    def ocr_frames(self, frames_rgb: List[np.ndarray], frame_indices: List[int]) -> Tuple[List[dict], Dict[int,float], Dict[int,float]]:
        """
        Run OCR on selected frames.
        Returns:
          boxes_all: list of {frame_idx, text, conf, bbox:(x,y,w,h), area}
          text_area_ratio: {frame_idx: float}
          any_text_present: {frame_idx: 0/1}
        """
        boxes_all = []
        text_area_ratio = {}
        any_text_present = {}

        for idx in frame_indices:
            if idx < 0 or idx >= len(frames_rgb):
                continue
            img = frames_rgb[idx]
            h, w = img.shape[:2]
            data = pytesseract.image_to_data(
                Image.fromarray(img),
                output_type=pytesseract.Output.DICT
            )
            total_area = 0.0
            frame_has_text = False
            for i in range(len(data["text"])):
                txt = data["text"][i].strip()
                conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
                x, y, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                if conf < 60 or len(txt) == 0:
                    continue
                area = float(bw * bh)
                total_area += area
                frame_has_text = True
                boxes_all.append({
                    "frame_idx": idx,
                    "text": txt,
                    "conf": conf,
                    "bbox": (x, y, bw, bh),
                    "area": area,
                    "frame_w": w,
                    "frame_h": h
                })
            ratio = total_area / float(h*w)
            text_area_ratio[idx] = ratio
            any_text_present[idx] = 1.0 if frame_has_text else 0.0

        return boxes_all, text_area_ratio, any_text_present

    def compute_cta_features(self,
                             ocr_boxes: List[dict],
                             fps: float) -> Dict[str, Any]:
        """
        Find CTA keywords, earliest frame time, CTA area ratio in that frame.
        """
        if fps is None or fps <= 1e-9:
            fps = 1.0  # fallback for images

        cta_hits = []
        first_frame = None
        frame_area_ratio = {}

        for box in ocr_boxes:
            txt_low = box["text"].lower()
            if any(kw in txt_low for kw in CTA_KEYWORDS):
                frame_i = box["frame_idx"]
                cta_hits.append((frame_i, box["text"]))
                # stash ratio for that frame
                w = box["frame_w"]
                h = box["frame_h"]
                ar = box["area"] / float(w*h)
                frame_area_ratio.setdefault(frame_i, 0.0)
                frame_area_ratio[frame_i] += ar
                if first_frame is None or frame_i < first_frame:
                    first_frame = frame_i

        if first_frame is None:
            return {
                "cta_present": False,
                "cta_time_s": None,
                "cta_area_ratio": 0.0,
                "cta_text_hits": []
            }

        time_s = float(first_frame / fps)
        area_ratio = frame_area_ratio.get(first_frame, 0.0)
        return {
            "cta_present": True,
            "cta_time_s": time_s,
            "cta_area_ratio": float(area_ratio),
            "cta_text_hits": [h[1] for h in cta_hits]
        }

    def face_stats(self, frames_rgb: List[np.ndarray], frame_indices: List[int]) -> Dict[str, Any]:
        """
        Use Haar cascades to count faces and approximate smiles.
        """
        face_counts = []
        smile_counts = []
        for idx in frame_indices:
            if idx < 0 or idx >= len(frames_rgb):
                continue
            img = frames_rgb[idx]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            face_counts.append(len(faces))

            smiles_found = 0
            for (x,y,w,h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(
                    roi_gray, 1.7, 22
                )
                if len(smiles) > 0:
                    smiles_found += 1
            smile_counts.append(smiles_found)

        total_faces = sum(face_counts)
        total_smiles = sum(smile_counts)
        avg_faces = safe_mean(face_counts)
        smile_rate = 0.0
        if total_faces > 0:
            smile_rate = float(total_smiles / total_faces)

        first_frame_faces = 0
        if len(frame_indices) > 0:
            first_idx = frame_indices[0]
            if 0 <= first_idx < len(frames_rgb):
                gray0 = cv2.cvtColor(frames_rgb[first_idx], cv2.COLOR_RGB2GRAY)
                faces0 = self.face_cascade.detectMultiScale(gray0, 1.3, 5)
                first_frame_faces = len(faces0)

        return {
            "face_count": int(total_faces),
            "smile_rate": float(smile_rate),
            "face_present_in_first_frame": bool(first_frame_faces > 0)
        }

    def palette_layout_features(self,
                                frames_rgb: List[np.ndarray],
                                rep_indices: List[int],
                                text_area_ratio_map: Dict[int,float],
                                any_text_present_map: Dict[int,float]) -> Dict[str, Any]:
        warmth_vals = []
        contrast_vals = []
        clutter_vals = []
        text_area_vals = []
        text_first_frame = False

        for i_idx, frame_idx in enumerate(rep_indices):
            if frame_idx < 0 or frame_idx >= len(frames_rgb):
                continue
            img = frames_rgb[frame_idx]
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            warmth_vals.append(warmth_index(img))
            contrast_vals.append(michelson_contrast(gray))
            clutter_vals.append(edge_density(img))
            text_area_vals.append(text_area_ratio_map.get(frame_idx, 0.0))
            if i_idx == 0:
                text_first_frame = any_text_present_map.get(frame_idx, 0.0) > 0.5

        warmth = safe_mean(warmth_vals)
        contrast = safe_mean(contrast_vals)
        clutter = safe_mean(clutter_vals)
        text_area_avg = safe_mean(text_area_vals)

        # crude heuristic for "product/app shot shown clearly"
        product_shot_present = (
            (text_area_avg < 0.10) and (clutter < 0.25)
        )

        return {
            "warmth": warmth,
            "contrast": contrast,
            "clutter": clutter,
            "text_present_in_first_frame": bool(text_first_frame),
            "product_shot_present": bool(product_shot_present)
        }

    def logo_stability(self,
                       ocr_boxes: List[dict],
                       rep_indices: List[int]) -> float:
        """
        Heuristic "brand persistence":
        Find most repeated high-confidence uppercase-ish token.
        Score = fraction of sampled frames containing that token.
        """
        # collect tokens per frame
        per_frame_tokens = {}
        for box in ocr_boxes:
            txt = box["text"].strip()
            if len(txt) < 3:
                continue
            # Heuristic "brand-like": mostly uppercase or TitleCase
            if txt.isupper() or (txt[0].isupper() and txt[1:].islower()):
                fr = box["frame_idx"]
                per_frame_tokens.setdefault(fr, set()).add(txt)

        # count appearances
        token_counts = {}
        for fr, toks in per_frame_tokens.items():
            for t in toks:
                token_counts[t] = token_counts.get(t, 0) + 1

        if len(token_counts) == 0 or len(rep_indices) == 0:
            return 0.0

        # pick most frequent token
        best_token = max(token_counts.items(), key=lambda kv: kv[1])[0]

        # stability = fraction of sampled frames where this token appears
        appearances = 0
        total_considered = 0
        rep_set = set(rep_indices)
        for fr in rep_set:
            total_considered += 1
            toks = per_frame_tokens.get(fr, set())
            if best_token in toks:
                appearances += 1

        if total_considered == 0:
            return 0.0
        return float(appearances / total_considered)

    def extract_vision_features(
        self,
        frames_rgb: List[np.ndarray],
        sampling_fps: float
    ) -> Dict[str, Any]:
        """
        High-level wrapper:
        1) pick representative frames
        2) CLIP embeddings, vertical classification
        3) OCR -> CTA timing, text stats
        4) faces/smiles
        5) palette/layout/clutter
        6) logo stability
        Returns dict with vision-related features + texts for multimodal.
        """
        if len(frames_rgb) == 0:
            return {}

        rep_indices = pick_representative_indices(frames_rgb)

        # 1) CLIP embeddings (on ALL frames, not just reps, for averaging)
        with torch.no_grad():
            img_emb_all = self.encode_images(frames_rgb)  # [N,D]
        img_emb_np = img_emb_all.detach().cpu().numpy()

        clip_img_mean = np.mean(img_emb_np, axis=0)
        clip_img_first = img_emb_np[0] if img_emb_np.shape[0] > 0 else np.zeros_like(clip_img_mean)

        # vertical classification via zero-shot
        vert_info = self.classify_vertical(img_emb_all)

        # 2) OCR on representative frames
        ocr_boxes, text_area_ratio_map, any_text_present_map = self.ocr_frames(frames_rgb, rep_indices)

        # CTA detection
        cta_info = self.compute_cta_features(ocr_boxes, sampling_fps if sampling_fps else 1.0)

        # gather all OCR text for multimodal alignment
        all_ocr_tokens = [b["text"] for b in ocr_boxes]
        ocr_fulltext = " ".join(all_ocr_tokens)

        # 3) face stats (on representative frames)
        face_info = self.face_stats(frames_rgb, rep_indices)

        # 4) palette / layout / clutter
        palette_info = self.palette_layout_features(
            frames_rgb,
            rep_indices,
            text_area_ratio_map,
            any_text_present_map
        )

        # 5) logo stability
        ls = self.logo_stability(ocr_boxes, rep_indices)

        out = {
            # embeddings for downstream recsys
            "clip_img_mean": clip_img_mean.tolist(),
            "clip_img_first": clip_img_first.tolist(),

            # zero-shot category
            "vertical_top1": vert_info["vertical_top1"],
            "vertical_top3": vert_info["vertical_top3"],
            "vertical_entropy": vert_info["vertical_entropy"],

            # CTA / clarity
            "cta_present": bool(cta_info["cta_present"]),
            "cta_time_s": cta_info["cta_time_s"],
            "cta_area_ratio": cta_info["cta_area_ratio"],

            # brand / trust
            "logo_stability": float(ls),

            # faces / warmth / clutter etc.
            "face_count": face_info["face_count"],
            "smile_rate": face_info["smile_rate"],
            "face_present_in_first_frame": face_info["face_present_in_first_frame"],
            "warmth": palette_info["warmth"],
            "contrast": palette_info["contrast"],
            "clutter": palette_info["clutter"],
            "text_present_in_first_frame": palette_info["text_present_in_first_frame"],
            "product_shot_present": palette_info["product_shot_present"],

            # stash OCR text for multimodal step
            "ocr_fulltext": ocr_fulltext,
        }

        return out
