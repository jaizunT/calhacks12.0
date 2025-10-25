# Axon-Signals Feature Extractor

This repo turns raw ad creatives (PNG or MP4) into structured features that a recommendation model could actually use.

## What it does

For each ad:
- samples a few frames (and first ~8 seconds of audio if it's a video)
- extracts 4 buckets of signals:
  1. **Hook / Attention**
     - motion_intensity_0_1s
     - first_cut_time_s
     - loudness_dbfs_0_1s
     - face_present_in_first_frame
     - text_present_in_first_frame
  2. **Clarity / Intent**
     - cta_present
     - cta_time_s
     - cta_area_ratio
     - product_shot_present
  3. **Relevance / Vertical**
     - vertical_top1 / vertical_top3
     - vertical_entropy
     - clip_img_mean / clip_img_first embeddings
  4. **Trust / Polish**
     - logo_stability
     - clutter
     - contrast
     - warmth
     - speech_ratio (testimonial vibe vs hype reel)
     - align_ocr_img / align_asr_img (does the ad *say* what it *shows*)
     - hook_score (simple interpretable engagement heuristic)

Outputs **one row per ad** to `features.csv`.

## Why this is valuable

These features map directly to stuff a real-time ad ranker cares about:
- Will people stop scrolling? (hook_score, motion/audio in first 1s)
- Is the CTA obvious and fast? (cta_time_s, cta_area_ratio)
- Is this ad actually about “car insurance” or “mobile game”? (vertical_top1)
- Does it look like a high-trust brand or a scammy meme edit? (logo_stability, clutter, align_ocr_img)

This satisfies:
- **Signal Extraction Insight**: features are distinct levers (attention, clarity, relevance, trust)
- **Performance**: all signals are computed by pretrained models or classic CV/DSP, batched per creative
- **Robustness**: deterministic frame sampling, OCR+ASR+CLIP handle almost any style of ad
- **Creativity**: alignment features and hook_score are “insight,” not raw pixels

## Install

You need:
- Python 3.10+
- `ffmpeg` on PATH
- `tesseract` on PATH (for OCR)

Then:

```bash
pip install -r requirements.txt
