"""
Step 1 — Data Augmentation
Reads raw images, applies posture-appropriate augmentations, writes
expanded dataset back so later steps have more data to work with.
"""

import cv2
import numpy as np
import random
import shutil
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DATASET_RAW, AUG_DIR,
    AUG_PER_IMAGE, FLIP_HORIZONTAL,
    BRIGHTNESS_RANGE, CONTRAST_RANGE,
    ROTATION_RANGE, NOISE_STD, BLUR_PROB,
    RANDOM_SEED,
)

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Augmentation primitives ────────────────────────────────────

def adjust_brightness_contrast(img, bright_factor, contrast_factor):
    """Multiply brightness and contrast without clipping harshly."""
    out = img.astype(np.float32)
    out = out * bright_factor
    mean = out.mean()
    out  = (out - mean) * contrast_factor + mean
    return np.clip(out, 0, 255).astype(np.uint8)

def add_gaussian_noise(img, std=NOISE_STD):
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def rotate_image(img, angle):
    h, w = img.shape[:2]
    M   = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                        borderMode=cv2.BORDER_REFLECT_101)

def apply_blur(img, ksize=3):
    return cv2.GaussianBlur(img, (ksize, ksize), 0)

def sharpen(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, k)

def augment_one(img):
    """Return one augmented version of img."""
    out = img.copy()

    # Brightness + Contrast
    b = random.uniform(*BRIGHTNESS_RANGE)
    c = random.uniform(*CONTRAST_RANGE)
    out = adjust_brightness_contrast(out, b, c)

    # Rotation (small — sitting posture)
    angle = random.uniform(*ROTATION_RANGE)
    out = rotate_image(out, angle)

    # Horizontal flip (both sides of sitting are valid)
    if FLIP_HORIZONTAL and random.random() < 0.5:
        out = cv2.flip(out, 1)

    # Gaussian noise
    out = add_gaussian_noise(out)

    # Occasional blur to simulate focus variation
    if random.random() < BLUR_PROB:
        out = apply_blur(out)

    return out


# ── Main runner ────────────────────────────────────────────────

def run(dataset_path: Path = DATASET_RAW):
    images = [p for p in dataset_path.rglob("*") if p.suffix.lower() in SUPPORTED]

    if not images:
        print(f"  [WARN] No images found in {dataset_path}")
        return

    # Wipe old augmentations to avoid stale leftovers
    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)
    AUG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Found {len(images)} raw images — generating {AUG_PER_IMAGE} augmentations each.")
    total_written = 0

    for img_path in tqdm(images, desc="  Augmenting", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [SKIP] Cannot read {img_path.name}")
            continue

        # Always copy the original (sharpened) first
        orig_dst = AUG_DIR / img_path.name
        cv2.imwrite(str(orig_dst), sharpen(img))
        total_written += 1

        # Generate augmented copies
        stem   = img_path.stem
        suffix = img_path.suffix
        for i in range(AUG_PER_IMAGE):
            aug = augment_one(img)
            dst_name = f"{stem}_aug{i:02d}{suffix}"
            cv2.imwrite(str(AUG_DIR / dst_name), aug)
            total_written += 1

    print(f"  ✅ Augmentation complete. Total images: {total_written}")
    print(f"     Saved to: {AUG_DIR}")


if __name__ == "__main__":
    run()
