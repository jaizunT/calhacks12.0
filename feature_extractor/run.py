# run.py
#
# Usage:
#   python run.py --in ads/ --out features.csv --device cpu --whisper tiny
#
# This script:
#   - walks the input dir for .png/.jpg/.jpeg/.mp4
#   - extracts features for each creative
#   - writes a CSV of per-ad features
#
# Parallelization:
#   Right now it's single-process for simplicity (so we reuse models).
#   It's easy to parallelize later with multiprocessing if needed.

import os
import argparse
import pandas as pd
from tqdm import tqdm
import torch

from feature_extractor.media_io import load_image_frames, load_video_frames, extract_audio_wav
from feature_extractor.vision import VisionExtractor
from feature_extractor.motion import motion_features
from feature_extractor.audio_feats import audio_features
from feature_extractor.multimodal import multimodal_features
from feature_extractor.config import WHISPER_MODEL_SIZE

import whisper  # openai-whisper

SUPPORTED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv"}

def discover_media_files(root_dir):
    media_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in SUPPORTED_IMAGE_EXT or ext in SUPPORTED_VIDEO_EXT:
                media_files.append(os.path.join(root, f))
    media_files.sort()
    return media_files

def process_single_ad(
    path: str,
    vision_extractor: VisionExtractor,
    whisper_model
):
    """
    Returns a dict (row) with:
      base metadata
      vision features
      motion features
      audio features
      multimodal features
    """
    ext = os.path.splitext(path.lower())[1]
    is_video = ext in SUPPORTED_VIDEO_EXT

    # Load frames (and duration info, etc)
    if is_video:
        frames_rgb, duration_used_s, sampling_fps, duration_total_s, modality = load_video_frames(path)
        # audio
        y, sr, aud_dur = extract_audio_wav(path)
    else:
        frames_rgb, duration_used_s, sampling_fps, duration_total_s, modality = load_image_frames(path)
        y, sr, aud_dur = None, None, 0.0

    # Vision
    vision_feats = vision_extractor.extract_vision_features(frames_rgb, sampling_fps if sampling_fps else 1.0)

    # Motion (videos only really, but will gracefully handle single frame)
    motion_feats = motion_features(frames_rgb,
                                   sampling_fps if sampling_fps else 1.0,
                                   duration_used_s)

    # Audio
    audio_feats = audio_features(y, sr if sr else 16000, whisper_model)

    # Multimodal
    mm_feats = multimodal_features(
        vision_feats,
        motion_feats,
        audio_feats,
        vision_extractor.clip_model,
        vision_extractor.tokenize,
        vision_extractor.backend,
        vision_extractor.device
    )

    # Base meta
    row = {
        "ad_id": os.path.basename(path),
        "file_path": path,
        "duration_s": float(duration_total_s),
        "duration_used_s": float(duration_used_s),
        "audio_duration_s": float(aud_dur),
        "modality": modality,
    }

    # merge dicts
    row.update({
        # vision subset (excluding internal strings used for multimodal)
        "clip_img_mean": vision_feats.get("clip_img_mean"),
        "clip_img_first": vision_feats.get("clip_img_first"),

        "vertical_top1": vision_feats.get("vertical_top1"),
        "vertical_top3": vision_feats.get("vertical_top3"),
        "vertical_entropy": vision_feats.get("vertical_entropy"),

        "cta_present": vision_feats.get("cta_present"),
        "cta_time_s": vision_feats.get("cta_time_s"),
        "cta_area_ratio": vision_feats.get("cta_area_ratio"),

        "logo_stability": vision_feats.get("logo_stability"),

        "face_count": vision_feats.get("face_count"),
        "smile_rate": vision_feats.get("smile_rate"),
        "face_present_in_first_frame": vision_feats.get("face_present_in_first_frame"),

        "warmth": vision_feats.get("warmth"),
        "contrast": vision_feats.get("contrast"),
        "clutter": vision_feats.get("clutter"),
        "text_present_in_first_frame": vision_feats.get("text_present_in_first_frame"),
        "product_shot_present": vision_feats.get("product_shot_present"),

        # motion
        "motion_intensity_0_1s": motion_feats.get("motion_intensity_0_1s"),
        "first_cut_time_s": motion_feats.get("first_cut_time_s"),
        "cut_rate_per_5s": motion_feats.get("cut_rate_per_5s"),
        "shake_var": motion_feats.get("shake_var"),

        # audio
        "loudness_dbfs": audio_feats.get("loudness_dbfs"),
        "loudness_dbfs_0_1s": audio_feats.get("loudness_dbfs_0_1s"),
        "speech_ratio": audio_feats.get("speech_ratio"),
        "bpm": audio_feats.get("bpm"),
        "transcript": audio_feats.get("transcript"),
        "asr_cta_hits": audio_feats.get("asr_cta_hits"),
        "asr_cta_time_s": audio_feats.get("asr_cta_time_s"),

        # multimodal
        "align_ocr_img": mm_feats.get("align_ocr_img"),
        "align_asr_img": mm_feats.get("align_asr_img"),
        "hook_score": mm_feats.get("hook_score"),
    })

    return row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="indir", required=True,
                        help="Folder containing ads (png/mp4).")
    parser.add_argument("--out", dest="outfile", default="features.csv",
                        help="Output CSV path.")
    parser.add_argument("--device", dest="device", default="cpu",
                        help="cuda or cpu for CLIP.")
    parser.add_argument("--whisper", dest="whisper_size",
                        default=WHISPER_MODEL_SIZE,
                        help="Whisper model size (tiny/base/small...).")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU.")
        device = "cpu"

    # init heavy models ONCE
    vision_extractor = VisionExtractor(device=device)
    whisper_model = whisper.load_model(args.whisper_size, device="cpu")  # whisper tiny/base on CPU is fine for <=8s audio

    media_files = discover_media_files(args.indir)

    rows = []
    for path in tqdm(media_files, desc="Processing ads"):
        try:
            row = process_single_ad(path, vision_extractor, whisper_model)
            rows.append(row)
        except Exception as e:
            print(f"[WARN] failed on {path}: {e}")

    df = pd.DataFrame(rows)

    # Save CSV
    df.to_csv(args.outfile, index=False)
    print(f"Saved {len(df)} rows to {args.outfile}")

if __name__ == "__main__":
    main()
