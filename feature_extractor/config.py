# feature_extractor/config.py

from math import log

# Frame/video sampling config
FPS_TARGET = 3                 # sample ~3 frames per second
MAX_VIDEO_SECONDS = 8          # don't bother past 8s for feature extraction
MAX_FRAMES = 40                # hard cap for frames we run through CLIP etc.
FRAME_SIZE = 256               # we letterbox-resize frames to square of this

# Whisper ASR model size (tiny=fast, base=better)
WHISPER_MODEL_SIZE = "tiny"

# CTA keywords we care about (lowercase match)
CTA_KEYWORDS = [
    "download", "install", "play now", "play free",
    "shop now", "buy now", "learn more", "limited time",
    "sale", "% off", "free", "subscribe", "sign up", "get started",
    "apply now", "quote", "start now"
]

# Zero-shot vertical categories to score with CLIP
VERTICAL_CLASSES = [
    "mobile game / gaming app",
    "finance / investing / trading",
    "insurance / auto insurance / coverage",
    "beauty / skincare / cosmetics",
    "shopping / ecommerce / coupon / sale",
    "automotive / car / dealership",
    "fitness / health / workout",
    "brand awareness / lifestyle / no specific CTA"
]

# Hook thresholds (hand-tuned heuristics)
MOTION_THRESH = 0.5         # optical flow mag ~ "moving fast"
LOUDNESS_THRESH = -25.0     # dBFS-ish. higher (less negative) = louder

def entropy(prob_list):
    """Natural-log entropy."""
    s = 0.0
    for p in prob_list:
        if p > 0:
            s -= p * log(p)
    return s
