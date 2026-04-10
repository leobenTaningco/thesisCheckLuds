"""
config.py — Shared paths and hyper-parameters for the pipeline.
Edit values here; every step reads from this file.
"""

from pathlib import Path

# ── Directories ────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATASET_RAW = ROOT / "dataset" / "raw"          # ← drop your images here
AUG_DIR     = ROOT / "dataset" / "augmented"    # Step 1 output
GOOD_DIR    = ROOT / "dataset" / "sorted" / "goodPosture"
BAD_DIR     = ROOT / "dataset" / "sorted" / "badPosture"
FEATURES_CSV = ROOT / "dataset" / "features.csv"
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

for _d in [AUG_DIR, GOOD_DIR, BAD_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Augmentation ────────────────────────────────────────────────
AUG_PER_IMAGE = 6           # extra copies per original image
FLIP_HORIZONTAL = True      # mirror left/right
BRIGHTNESS_RANGE = (0.6, 1.4)
CONTRAST_RANGE   = (0.7, 1.3)
ROTATION_RANGE   = (-10, 10)   # degrees  (sitting: keep small)
NOISE_STD        = 12          # Gaussian noise σ
BLUR_PROB        = 0.3         # probability of slight Gaussian blur

# ── Deduplication ───────────────────────────────────────────────
HASH_SIZE        = 16          # perceptual hash resolution
SIMILARITY_THRESHOLD = 4       # hamming distance ≤ this → duplicate

# ── MediaPipe ───────────────────────────────────────────────────
MP_MODEL_COMPLEXITY   = 1
MP_MIN_DETECTION_CONF = 0.5
MP_MIN_TRACKING_CONF  = 0.5
VISIBILITY_THRESHOLD  = 0.5

# ── Training ────────────────────────────────────────────────────
TEST_SIZE   = 0.20
VAL_SIZE    = 0.10            # fraction of train set used for val
RANDOM_SEED = 42

RF_PARAMS = {
    "n_estimators":   500,
    "max_depth":      12,
    "min_samples_split": 4,
    "min_samples_leaf":  4,
    "max_features": "sqrt",     # add this — reduces correlation between trees
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

MLP_PARAMS = {
    # "hidden_layer_sizes": (128, 64, 32),
    "hidden_layer_sizes": (64, 32),   # was (128,64,32) — simpler is better with 7-10 features
    "activation":    "relu",
    "solver":        "adam",
    # "alpha":         1e-3,          # L2 regularisation
    "alpha": 1e-2,                    # was 1e-3 — more regularisation to reduce overfitting
    "learning_rate": "adaptive",
    "max_iter":      500,
    "early_stopping": True,
    # "validation_fraction": 0.1,
    # "n_iter_no_change":   20,
    "validation_fraction": 0.15,      # was 0.1 — give more data to val for early stopping
    "n_iter_no_change": 30,           # was 20 — more patient
    "random_state": RANDOM_SEED,
}

# ── Labels ──────────────────────────────────────────────────────
LABEL_GOOD = 1
LABEL_BAD  = 0
LABEL_NAMES = {LABEL_GOOD: "Good Posture", LABEL_BAD: "Bad Posture"}
