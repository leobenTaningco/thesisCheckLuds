"""
Step 2 — MediaPipe Feature Extraction
Runs MediaPipe Pose on every augmented image and writes a CSV
of biomechanical features used by RF/MLP downstream.

Features extracted (all from one side, hip→head segment):
  neck_inclination   — angle between ear→shoulder vertical
  torso_inclination  — angle between hip→shoulder vertical
  arm_inclination    — angle between shoulder→elbow vertical
  neck_ratio         — ear-shoulder dist / shoulder-hip dist
  shoulder_ear_y_diff— normalised vertical gap (forward head proxy)
  torso_lean         — horizontal displacement hip→shoulder (normalised)
  head_forward_angle — angle ear–shoulder–hip (3-point)
  shoulder_hip_angle — angle of shoulder→hip from vertical
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from math import atan2, degrees, acos
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    AUG_DIR, FEATURES_CSV,
    MP_MODEL_COMPLEXITY,
    MP_MIN_DETECTION_CONF, MP_MIN_TRACKING_CONF,
    VISIBILITY_THRESHOLD,
    LABEL_GOOD, LABEL_BAD,
)

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_PATH="models/pose_landmarker_heavy.task"
# ── Geometry helpers ────────────────────────────────────────────

def find_inclination(x1, y1, x2, y2):
    return degrees(atan2(abs(x2 - x1), abs(y2 - y1)))

def three_point_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos = np.clip(
        np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6),
        -1.0, 1.0,
    )
    return degrees(acos(cos))


# ── Preprocessing ───────────────────────────────────────────────

def sharpen(img):
    k = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(img, -1, k)

def enhance_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def upscale(img, factor=1.5):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * factor), int(h * factor)),
                    interpolation=cv2.INTER_CUBIC)

def detect_side(pose_landmarks):
    nose_x = pose_landmarks[0].x
    THRESHOLD = 0.08
    if nose_x < (0.5 - THRESHOLD):
        return "RIGHT"
    elif nose_x > (0.5 + THRESHOLD):
        return "LEFT"
    left_vis  = sum(pose_landmarks[i].visibility for i in [11, 23, 25])
    right_vis = sum(pose_landmarks[i].visibility for i in [12, 24, 26])
    return "LEFT" if left_vis >= right_vis else "RIGHT"

# ── Sanity check ────────────────────────────────────────────────

def is_valid_keypoints(kp):
    if 'ear' in kp and 'shoulder' in kp:
        if kp['ear'][1] > kp['shoulder'][1]:
            return False
    if 'shoulder' in kp and 'hip' in kp:
        if kp['hip'][1] < kp['shoulder'][1]:
            return False
        # Hip must be meaningfully below shoulder (not nearly same height)
        if (kp['hip'][1] - kp['shoulder'][1]) < 10:
            return False
    if 'hip' in kp and 'knee' in kp:
        if kp['knee'][1] < kp['hip'][1]:
            return False
    return True


# ── Keypoint extraction ─────────────────────────────────────────

def get_keypoints(image_rgb, landmarker):
    h, w = image_rgb.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(image_rgb)
    )

    results = landmarker.detect(mp_image)

    if not results.pose_landmarks:
        return None

    lms = results.pose_landmarks[0] 
    

    def pt(idx):
        lm = lms[idx]
        if lm.visibility < VISIBILITY_THRESHOLD:
            return None
        return [lm.x * w, lm.y * h]
    
    side = detect_side(lms)

    if side == "LEFT":
        raw = {
            "ear":      pt(7),
            "shoulder": pt(11),
            "hip":      pt(23),
        }
    else:
        raw = {
            "ear":      pt(8),
            "shoulder": pt(12),
            "hip":      pt(24),
        }

    kp = {k: v for k, v in raw.items() if v is not None}

    if len(kp) < 3:
        return None

    if not is_valid_keypoints(kp):
        return None

    return kp, side


# ── Feature engineering ─────────────────────────────────────────

def extract_features(kp, img_h, img_w):
    """Return a flat dict of features for one image."""
    ear      = np.array(kp["ear"])
    shoulder = np.array(kp["shoulder"])
    hip      = np.array(kp["hip"])

    neck_inc  = find_inclination(shoulder[0], shoulder[1], ear[0], ear[1])
    torso_inc = find_inclination(hip[0], hip[1], shoulder[0], shoulder[1])

    sh_dist = np.linalg.norm(shoulder - hip) + 1e-6
    es_dist = np.linalg.norm(ear - shoulder)
    neck_ratio = es_dist / sh_dist

    # Normalised vertical head-forward proxy
    ear_shoulder_y_diff = (shoulder[1] - ear[1]) / img_h

    # Horizontal lean of torso (positive = leaning forward)
    torso_lean = (shoulder[0] - hip[0]) / img_w

    # Three-point angles
    head_forward_angle  = three_point_angle(ear, shoulder, hip)
    shoulder_hip_angle  = find_inclination(hip[0], hip[1], shoulder[0], shoulder[1])

    return {
        "neck_inclination":   neck_inc,
        "torso_inclination":  torso_inc,
        "neck_ratio":         neck_ratio,
        "ear_shoulder_y_diff": ear_shoulder_y_diff,
        "torso_lean":          torso_lean,
        "head_forward_angle":  head_forward_angle,
        "shoulder_hip_angle":  shoulder_hip_angle,
    }


# ── Label helper ────────────────────────────────────────────────

def label_from_name(name: str):
    lower = name.lower()
    if "goodposture" in lower:
        return LABEL_GOOD
    if "badposture" in lower:
        return LABEL_BAD
    return None


# ── Main runner ─────────────────────────────────────────────────

def run(source_dir: Path = AUG_DIR):
    images = sorted([p for p in source_dir.rglob("*") if p.suffix.lower() in SUPPORTED])

    if not images:
        print(f"  [WARN] No images found in {source_dir}")
        return

    print(f"  Running MediaPipe on {len(images)} images…")

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE
    )

    landmarker = PoseLandmarker.create_from_options(options)

    records   = []
    skipped   = 0
    no_label  = 0
    passed = 0

    for img_path in tqdm(images, desc="  Extracting", unit="img"):
        label = label_from_name(img_path.name)
        if label is None:
            no_label += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            skipped += 1
            continue

        img = sharpen(img)
        img = enhance_contrast(img)
        img = upscale(img)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        result = get_keypoints(rgb, landmarker)
        if result is None:
            skipped += 1
            print(f"[SKIP {skipped}] {img_path.name}")
            continue

        if passed % 100 == 0 and passed > 0:
            print(f"[PASS] {passed} images processed successfully")

        kp, side = result
        required = {"ear", "shoulder", "hip"}
        if not required.issubset(kp):
            skipped += 1
            continue

        feats = extract_features(kp, h, w)
        feats["label"]      = label
        feats["filename"]   = img_path.name
        feats["side"]       = side
        records.append(feats)
        passed += 1

    df = pd.DataFrame(records)
    df.to_csv(FEATURES_CSV, index=False)

    print(f"  ✅ Feature extraction done.")
    print(f"     Records : {len(df)}")
    print(f"     Skipped : {skipped} (no pose / invalid)")
    if no_label:
        print(f"     No label: {no_label} (filename lacked 'goodPosture'/'badPosture')")
    print(f"     Saved   : {FEATURES_CSV}")
    print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")


if __name__ == "__main__":
    run()
