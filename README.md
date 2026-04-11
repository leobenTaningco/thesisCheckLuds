# 🧍 Posture Detection Pipeline

**RF + MLP + GBM Ensemble with MediaPipe Pose Landmarks**

Detects whether a seated person has **good** or **bad** posture from static images, using body angles extracted by MediaPipe (Google's pose estimation library).

> **Written for people who know web dev but not Python or ML.**
> If you've worked with JavaScript, PHP, or Java — you already understand the *concepts* here (functions, loops, data structures). This doc focuses on explaining what's unfamiliar: the ML stuff.

---

## Table of Contents

- [How the Pipeline Works (Big Picture)](#how-the-pipeline-works-big-picture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Image Naming Convention](#image-naming-convention)
- [config.py — The Settings File](#configpy--the-settings-file)
- [Step 1 — Data Augmentation](#step-1--data-augmentation-step1_augmentpy)
- [Step 2 — Feature Extraction](#step-2--feature-extraction-step2_extractpy)
- [Step 3 — Deduplication & Balancing](#step-3--deduplication--balancing-step3_dedupe_balancepy)
- [Step 4 — Model Training](#step-4--model-training-step4_trainpy)
- [Step 5 — Analytics & Charts](#step-5--analytics--charts-step5_analyticspy)
- [stepextra.py — Manual Balancer](#stepextrapy--manual-balancer)
- [ML Concepts Explained Simply](#ml-concepts-explained-simply)
- [Running the Pipeline](#running-the-pipeline)

---

## How the Pipeline Works (Big Picture)

Think of this as an **assembly line**. Raw photos go in one end, a trained AI model comes out the other.

```
📸 Raw photos
    ↓
Step 1: Make more photos (augmentation)
    ↓
Step 2: Read body angles from each photo (feature extraction)
    ↓
Step 3: Remove duplicates + balance the dataset
    ↓
Step 4: Train AI models on the angles
    ↓
Step 5: Generate charts showing how well the models performed
    ↓
🤖 Trained model saved to disk
```

**Why not just feed raw pixels to the AI?**

A single 640×480 image has ~921,000 pixels, each with 3 color values. That's nearly 3 million numbers per image. An AI trained on raw pixels would need *millions* of images and hours of compute. Instead, Step 2 distills each photo down to **8 meaningful numbers** (body angles), which is way easier for a small model to learn from.

> **Real-world analogy:** Instead of showing a doctor thousands of raw blood test readings, you hand them a summary card: "cholesterol: 180, blood pressure: 120/80..." — much easier to analyze and act on.

---

## Project Structure

```
posture_pipeline/
├── config.py                ← All paths & settings live here
├── dataset/
│   ├── raw/                 ← ⬅ DROP YOUR PHOTOS HERE
│   ├── augmented/           ← Step 1 output (auto-generated)
│   ├── sorted/
│   │   ├── goodPosture/     ← Step 3 output
│   │   └── badPosture/      ← Step 3 output
│   └── features.csv         ← Step 2/3 output (the "spreadsheet of angles")
├── models/
│   ├── rf_pipeline.joblib          ← Trained Random Forest
│   ├── mlp_pipeline.joblib         ← Trained Neural Network
│   ├── gbm_pipeline.joblib         ← Trained Gradient Boosting
│   ├── rf_mlp_pipeline.joblib      ← RF + MLP ensemble
│   └── rf_mlp_gbm_pipeline.joblib  ← RF + MLP + GBM ensemble
├── outputs/                 ← All charts and results
│   ├── 01_model_comparison.png
│   ├── 02_cv_f1_boxplot.png
│   ├── 03_confusion_matrices.png
│   ├── 04_feature_importance.png
│   ├── 05_roc_curves.png
│   ├── 06_prediction_samples.png
│   ├── 07_scorecard.png
│   └── results.json
└── steps/
    ├── step1_augment.py
    ├── step2_extract.py
    ├── step3_dedupe_balance.py
    ├── step4_train.py
    ├── step5_analytics.py
    └── stepextra.py
```

---

## Setup

```bash
# 1. Create a virtual environment (like node_modules but for Python)
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

> **What's a virtual environment?** Same concept as `npm install` creating a local `node_modules` folder. It isolates your project's Python packages so they don't conflict with other projects on your machine.

---

## Image Naming Convention

Your image filenames **must** contain either `goodPosture` or `badPosture` (case-insensitive). This is how the pipeline knows whether each photo is a good or bad posture example — it literally reads the label from the filename.

```
✅  john_goodPosture_001.jpg
✅  badPosture_frame042.png
✅  session3_goodposture_side.jpg
❌  sitting_upright.jpg     ← will be skipped (no label in name)
❌  bad_001.jpg             ← will be skipped (doesn't contain "badposture")
```

---

## `config.py` — The Settings File

This is the **single source of truth** for the entire pipeline. Every other script imports from here. If you want to change a setting, change it here — not inside the individual step files.

> **Think of it like a `.env` file**, but for ML parameters instead of API keys.

### Paths

```python
ROOT        = Path(__file__).parent       # The folder this file lives in
DATASET_RAW = ROOT / "dataset" / "raw"   # Where you drop your raw photos
AUG_DIR     = ROOT / "dataset" / "augmented"  # Step 1 writes here
GOOD_DIR    = ROOT / "dataset" / "sorted" / "goodPosture"  # Step 3 output
BAD_DIR     = ROOT / "dataset" / "sorted" / "badPosture"   # Step 3 output
FEATURES_CSV = ROOT / "dataset" / "features.csv"  # The angles spreadsheet
MODELS_DIR  = ROOT / "models"            # Trained models saved here
OUTPUTS_DIR = ROOT / "outputs"           # Charts saved here
```

`Path(__file__).parent` is Python's way of saying "the folder this script is saved in." It's relative, so it works on any computer regardless of where the project is located — same idea as `__dirname` in Node.js.

The loop at the bottom auto-creates these folders if they don't exist:

```python
for _d in [AUG_DIR, GOOD_DIR, BAD_DIR, MODELS_DIR, OUTPUTS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
# parents=True  → creates parent folders too (like mkdir -p)
# exist_ok=True → doesn't crash if the folder already exists
```

### Augmentation Settings

```python
AUG_PER_IMAGE    = 6            # How many extra copies to make per original image
FLIP_HORIZONTAL  = True         # Mirror images left-right (valid for sitting posture)
ROTATION_RANGE   = (-10, 10)    # Randomly rotate between -10° and +10°
NOISE_STD        = 12           # Gaussian noise intensity (currently commented out)
BLUR_PROB        = 0.3          # 30% chance of slight blur (currently commented out)
```

### Deduplication Settings

```python
HASH_SIZE            = 16   # Resolution of the perceptual hash (bigger = more precise)
SIMILARITY_THRESHOLD = 4    # Hamming distance threshold — images this close = duplicates
```

### MediaPipe Settings

```python
MP_MODEL_COMPLEXITY   = 1    # 0 = fast/less accurate, 1 = balanced, 2 = accurate/slow
MP_MIN_DETECTION_CONF = 0.5  # MediaPipe only processes images where it's ≥50% confident a person is present
MP_MIN_TRACKING_CONF  = 0.5  # Confidence threshold for tracking landmarks between frames
VISIBILITY_THRESHOLD  = 0.5  # A landmark (e.g. elbow) is only used if MediaPipe is ≥50% sure it can see it
```

### Training Settings

```python
TEST_SIZE   = 0.20   # Hold back 20% of data for testing — the model never sees this during training
VAL_SIZE    = 0.10   # 10% of the training set is used for validation during training
RANDOM_SEED = 42     # Fixed number for all randomness — makes results reproducible
```

`RANDOM_SEED = 42` is important. ML involves a lot of randomness (random splits, random initialization, etc.). By fixing this number, you guarantee the same results every run. It's like using `Math.random()` with a fixed seed — useful for debugging and comparing runs.

### Model Hyperparameters

```python
RF_PARAMS = {
    "n_estimators":      500,       # Build 500 decision trees
    "max_depth":         12,        # Each tree can be at most 12 levels deep
    "min_samples_split": 4,         # A node needs ≥4 samples before it can split
    "min_samples_leaf":  4,         # Each leaf node must have ≥4 samples (prevents overfitting)
    "max_features":      "sqrt",    # Each tree only sees √8 ≈ 2-3 features — reduces correlation between trees
    "class_weight":      "balanced",# Automatically adjusts for imbalanced classes
    "random_state":      RANDOM_SEED,
    "n_jobs":            -1,        # Use all CPU cores in parallel
}
```

```python
MLP_PARAMS = {
    "hidden_layer_sizes": (64, 32), # Neural net: 8 inputs → 64 neurons → 32 neurons → 1 output
    "activation":         "relu",   # Activation function (ReLU is the modern standard)
    "solver":             "adam",   # Optimization algorithm (Adam is the modern standard)
    "alpha":              1e-2,     # L2 regularization strength (prevents overfitting)
    "learning_rate":      "adaptive",# Slows down learning rate when progress stalls
    "max_iter":           500,      # Maximum training epochs
    "early_stopping":     True,     # Stop training if validation score doesn't improve
    "validation_fraction":0.15,     # 15% of training data used for early stopping check
    "n_iter_no_change":   30,       # Stop after 30 epochs with no improvement
    "random_state":       RANDOM_SEED,
}
```

### Labels

```python
LABEL_GOOD  = 1   # Good posture = 1
LABEL_BAD   = 0   # Bad posture  = 0
LABEL_NAMES = {1: "Good Posture", 0: "Bad Posture"}
```

---

## Step 1 — Data Augmentation (`step1_augment.py`)

**Problem:** ML models need a lot of data to learn well. If you only have 50 photos, that's not enough.

**Solution:** Create multiple modified copies of each photo — flipped, rotated, etc. These look different enough that the model treats them as separate training examples, but they still show the same posture.

> **Analogy:** Like a teacher photocopying study worksheets at slightly different angles and brightness so students get variety — not just one perfect version.

With `AUG_PER_IMAGE = 6`, one original photo becomes 7 total images (1 original + 6 augmented copies).

### `rotate_image(img, angle)`

```python
def rotate_image(img, angle):
    h, w = img.shape[:2]   # Get image height and width
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # (w/2, h/2) = rotate around the center of the image
    # angle = degrees to rotate
    # 1.0 = scale factor (1.0 means don't resize)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    # warpAffine applies the rotation matrix
    # BORDER_REFLECT_101 fills empty corners by mirroring edge pixels (no black corners)
```

`img.shape` returns `(height, width, channels)`. The `[:2]` takes just height and width, ignoring the color channels. This is similar to destructuring in JavaScript: `const [h, w] = img.shape`.

### `augment_one(img)`

```python
def augment_one(img):
    """Return one augmented version of img."""
    out = img.copy()   # Don't modify the original — work on a copy

    # Pick a random rotation angle within the configured range
    angle = random.uniform(*ROTATION_RANGE)
    # *ROTATION_RANGE unpacks (-10, 10) into two arguments: random.uniform(-10, 10)
    # This is like the spread operator in JS: Math.random(...rotationRange)
    out = rotate_image(out, angle)

    # 50% chance of flipping horizontally (mirroring left-right)
    if FLIP_HORIZONTAL and random.random() < 0.5:
        out = cv2.flip(out, 1)
        # cv2.flip(img, 1) = horizontal flip
        # cv2.flip(img, 0) = vertical flip

    return out
```

Note: brightness jitter, noise, and blur are **commented out** in the current version. The `# out = adjust_brightness_contrast(...)` lines are disabled — only rotation and horizontal flip are active.

### `run(dataset_path)`

```python
def run(dataset_path: Path = DATASET_RAW):
    # Find all image files in the raw folder
    images = [p for p in dataset_path.rglob("*") if p.suffix.lower() in SUPPORTED]
    # rglob("*") = find all files recursively (like find . -type f)
    # p.suffix.lower() = file extension in lowercase (.jpg, .png, etc.)
    # SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    # Wipe old augmented folder to avoid stale leftovers from previous runs
    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)   # Delete folder and everything in it
    AUG_DIR.mkdir(parents=True, exist_ok=True)  # Create fresh empty folder

    for img_path in tqdm(images, desc="Augmenting", unit="img"):
        # tqdm = progress bar library (like a loading bar in the terminal)
        img = cv2.imread(str(img_path))   # Load image as a numpy array
        if img is None:
            continue   # Skip if the file couldn't be read

        # Always save the original first
        orig_dst = AUG_DIR / img_path.name
        cv2.imwrite(str(orig_dst), img)

        # Generate AUG_PER_IMAGE augmented copies
        stem   = img_path.stem    # Filename without extension, e.g. "photo_goodPosture_01"
        suffix = img_path.suffix  # Just the extension, e.g. ".jpg"
        for i in range(AUG_PER_IMAGE):   # range(6) = 0, 1, 2, 3, 4, 5
            aug = augment_one(img)
            dst_name = f"{stem}_aug{i:02d}{suffix}"
            # f-string: if stem="photo_goodPosture_01", suffix=".jpg", i=3
            # → "photo_goodPosture_01_aug03.jpg"
            # :02d = pad with zeros to 2 digits (so aug03, not aug3)
            cv2.imwrite(str(AUG_DIR / dst_name), aug)
```

> ⚠️ **Why wipe the folder first?** If you removed a photo from `raw/` but didn't wipe `augmented/`, the old augmented copies would still be there. Always start fresh to keep the data consistent.

---

## Step 2 — Feature Extraction (`step2_extract.py`)

**Problem:** We can't feed raw pixel data to our models — there's too much of it, and pixels don't directly encode posture information.

**Solution:** Use MediaPipe to detect body joints in each photo, then calculate 8 angles that meaningfully describe sitting posture. Save all this to a CSV file.

> **What is MediaPipe?** It's a library made by Google that detects 33 body landmarks (joints: nose, shoulders, elbows, hips, knees, etc.) in a photo and returns their (x, y, z) coordinates as fractions of the image size. So if your shoulder is in the middle of a 640px-wide image, its x-coordinate is `0.5`.

### The 8 Features

| Feature | What it measures | Why it matters |
|---|---|---|
| `neck_inclination` | Angle between ear→shoulder line and vertical | Forward head = high angle |
| `torso_inclination` | Angle between hip→shoulder line and vertical | Slouching = high angle |
| `arm_inclination` | Shoulder→elbow vs vertical | Arm position relative to body |
| `neck_ratio` | Ear-shoulder distance / shoulder-hip distance | Another forward head indicator |
| `ear_shoulder_y_diff` | Vertical gap between ear and shoulder (normalized) | Forward head proxy |
| `torso_lean` | Horizontal drift of shoulder relative to hip | Lateral lean |
| `head_forward_angle` | 3-point angle: ear → shoulder → hip | Overall head position |
| `shoulder_hip_angle` | Shoulder→hip deviation from vertical | Overall slouch indicator |

### `angle_with_vertical(p1, p2)`

```python
def angle_with_vertical(p1, p2):
    # p1 and p2 are (x, y) tuples, e.g. (0.45, 0.32)
    dx = p2[0] - p1[0]   # Horizontal difference
    dy = p2[1] - p1[1]   # Vertical difference
    angle = math.degrees(math.atan2(abs(dx), abs(dy)))
    # atan2 computes the arctangent — converts the x/y ratio to an angle
    # math.degrees() converts from radians to degrees
    # This gives the angle this line makes with a perfectly vertical line
    return angle
```

Think of it this way: a perfectly vertical line (straight up-down) has angle 0°. The more the line tilts, the higher the angle. A horizontal line would be 90°.

### `three_point_angle(a, b, c)`

```python
def three_point_angle(a, b, c):
    # Calculates the angle at point B, looking from A to B to C
    # Example: angle at your shoulder, from your ear to your shoulder to your hip
    ba = (a[0] - b[0], a[1] - b[1])   # Vector from B to A
    bc = (c[0] - b[0], c[1] - b[1])   # Vector from B to C

    # Dot product: how much two vectors point in the same direction
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)   # Length of vector BA
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)   # Length of vector BC

    cos_angle = dot / (mag_ba * mag_bc + 1e-6)
    # +1e-6 prevents division by zero if a point has zero length
    cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to [-1, 1] for safety
    return math.degrees(math.acos(cos_angle))
```

This is the dot product formula from linear algebra. Don't worry too much about the math — just know it computes the angle at point B in a triangle.

### `extract_features(landmarks, w, h, side)`

```python
def extract_features(landmarks, w, h, side="right"):
    # landmarks = list of 33 body joints from MediaPipe
    # w, h = image width and height in pixels
    # side = "right" or "left" — which side of the body to measure

    # Pick the right landmarks based on side
    if side == "right":
        ear      = landmarks[mp_pose.PoseLandmark.RIGHT_EAR]
        shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        hip      = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        elbow    = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
    else:
        # Mirror image — use left side landmarks
        ...

    # Check visibility — only use a landmark if MediaPipe is confident it can see it
    if ear.visibility < VISIBILITY_THRESHOLD:
        return None   # Skip this image/side if we can't see the ear clearly

    # Convert from normalized (0–1) coordinates to actual pixels
    ear_pt      = (ear.x * w,      ear.y * h)
    shoulder_pt = (shoulder.x * w, shoulder.y * h)
    hip_pt      = (hip.x * w,      hip.y * h)

    # Calculate all 8 angles
    features = {
        "neck_inclination":   angle_with_vertical(ear_pt, shoulder_pt),
        "torso_inclination":  angle_with_vertical(hip_pt, shoulder_pt),
        "neck_ratio":         dist(ear_pt, shoulder_pt) / (dist(shoulder_pt, hip_pt) + 1e-6),
        "head_forward_angle": three_point_angle(ear_pt, shoulder_pt, hip_pt),
        # ... etc
    }
    return features
```

### `run()`

```python
def run():
    # Set up MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=MP_MODEL_COMPLEXITY,
        min_detection_confidence=MP_MIN_DETECTION_CONF,
        min_tracking_confidence=MP_MIN_TRACKING_CONF,
    )

    rows = []   # Will become rows in the CSV

    for img_path in tqdm(all_images):
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # MediaPipe expects RGB, but OpenCV loads images as BGR (reversed).
        # cvtColor converts between color formats.

        results = pose.process(img_rgb)   # Run MediaPipe pose detection

        if not results.pose_landmarks:
            continue   # MediaPipe found no person — skip this image

        # Determine label from filename
        label = LABEL_GOOD if "goodposture" in img_path.name.lower() else LABEL_BAD

        # Try both left and right sides of the body
        for side in ["right", "left"]:
            features = extract_features(results.pose_landmarks.landmark, w, h, side)
            if features is None:
                continue   # Landmarks not visible enough on this side

            row = {"filename": img_path.name, "label": label, "side": side}
            row.update(features)   # Merge the 8 angles into the row dict
            rows.append(row)

    # Save all rows to CSV
    df = pd.DataFrame(rows)
    df.to_csv(FEATURES_CSV, index=False)
    # pd.DataFrame = create a table (like a 2D array) from a list of dicts
    # .to_csv() saves it as a spreadsheet file
    # index=False = don't add an extra row number column
```

The resulting `features.csv` looks like this:

| filename | label | side | neck_inclination | torso_inclination | ... |
|---|---|---|---|---|---|
| photo_goodPosture_aug00.jpg | 1 | right | 12.3 | 8.1 | ... |
| photo_badPosture_001.jpg | 0 | right | 34.7 | 22.4 | ... |

---

## Step 3 — Deduplication & Balancing (`step3_dedupe_balance.py`)

Two separate jobs happen here:

1. **Deduplication** — remove near-identical images so the model doesn't memorize duplicates
2. **Balancing** — make sure both classes have similar amounts of data

> **Why both matter:**
> - If your dataset is 90% duplicates, the model just memorizes a few examples instead of learning general patterns.
> - If you have 1000 "bad posture" examples and only 10 "good posture" ones, the model will just guess "bad" every single time and report 99% accuracy — while being completely useless. This is called **class imbalance**.

### Perceptual Hashing (Deduplication)

A regular file hash (like MD5) changes if even one pixel is different. A **perceptual hash** is a "fuzzy fingerprint" — two photos that look almost identical will have very similar perceptual hashes, even if they're not pixel-perfect copies.

```python
def compute_hash(img_path: Path):
    if HAS_IMAGEHASH:
        # pHash = "perceptual hash" — compares image structure
        return str(imagehash.phash(PILImage.open(img_path), hash_size=HASH_SIZE))
    return dhash_cv(img_path)   # Fallback if imagehash not installed
```

```python
def dhash_cv(img_path: Path, hash_size: int = HASH_SIZE) -> str:
    # Fallback hash using OpenCV
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # IMREAD_GRAYSCALE = load as black-and-white (faster, structure is what matters)
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    # Shrink to a tiny standard size (e.g. 17×16) — removes noise, keeps structure
    diff = resized[:, 1:] > resized[:, :-1]
    # Compare each pixel to the pixel to its right: True if brighter, False if darker
    # This captures the relative brightness pattern — the "fingerprint"
    return "".join(str(int(b)) for b in diff.flatten())
    # Convert True/False array to a string of 1s and 0s: "1010110011..."
```

```python
def hamming(h1: str, h2: str) -> int:
    # Count how many bits differ between two hash strings
    # 0 = identical images, higher = more different
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))
    # zip() pairs up characters: [('1','1'), ('0','1'), ('1','0'), ...]
    # c1 != c2 = True (1) if they differ, False (0) if same
    # sum() adds up all the differences
```

```python
def deduplicate(filenames: list, source_dir: Path) -> list:
    # Sort so originals come before augmented copies
    # This ensures we keep originals when there's a hash collision
    sorted_names = sorted(hashes.keys(), key=lambda n: (1 if "_aug" in n else 0, n))
    # lambda n: (1 if "_aug" in n else 0, n)
    # Originals get sort key (0, "name") — augmented get (1, "name")
    # Tuples sort element by element, so 0 < 1 = originals sort first

    kept        = []
    seen_hashes = []
    for name in sorted_names:
        h = hashes.get(name, "")
        # Check if this image is too similar to any image we've already kept
        if any(hamming(h, sh) <= SIMILARITY_THRESHOLD for sh in seen_hashes):
            continue   # It's a near-duplicate — skip it
        kept.append(name)
        seen_hashes.append(h)
    return kept
```

### Class Balancing

After deduplication, if one class has more than 1.5× the examples of the other:

**Method A — SMOTE** (preferred, requires `imbalanced-learn`):

```python
if HAS_SMOTE and counts[minority] >= 6:
    k = min(5, counts[minority] - 1)
    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
    X_res, y_res = smote.fit_resample(X, y)
    # SMOTE generates NEW synthetic data points by interpolating between
    # existing minority-class examples.
    # Example: if you have 2 "good posture" rows with angles [12, 8] and [15, 10],
    # SMOTE might generate a new synthetic row at [13.5, 9] — halfway between them.
    # This is smarter than just copy-pasting existing rows.
```

**Method B — Random oversampling** (fallback):

```python
else:
    needed  = len(maj_df) - len(min_df)   # How many rows do we need to add?
    sampled = min_df.sample(needed, replace=True, random_state=RANDOM_SEED)
    # .sample() picks random rows from the minority class
    # replace=True = same row can be picked multiple times (sampling with replacement)
    sampled["filename"] = "synthetic"   # Flag these rows as synthetic
```

> ⚠️ **The synthetic flag matters.** In Step 4, synthetic rows are only used for training — they're never included in the test set. Your test set must contain only real images so your accuracy numbers are honest.

---

## Step 4 — Model Training (`step4_train.py`)

The core of the pipeline. Trains 3 AI models + 2 ensembles, evaluates them, and saves everything to disk.

### The Pipeline Pattern

Every model is wrapped in a 3-stage `sklearn.Pipeline`:

```python
def make_pipeline(clf):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        # Stage 1: Fill in missing values (NaN) with the column median
        # Some images might be missing arm_inclination if the elbow wasn't visible.
        # Instead of crashing, we fill it with the median value of that column.

        ("scaler", StandardScaler()),
        # Stage 2: Normalize all features to the same scale
        # If neck_inclination ranges 0–90° but neck_ratio ranges 0.0–1.0,
        # the neural network will over-weight the larger-scale feature.
        # StandardScaler makes every feature have mean=0 and std=1.

        ("clf", clf),
        # Stage 3: The actual classifier
    ])
```

> **Why normalize?** Imagine training a model where one feature is "height in cm" (160–180) and another is "weight in kg" (60–90). The height feature has bigger numbers, so the model might incorrectly treat it as more important. Normalization puts everything on equal footing.

> **Important:** The scaler is fit on training data only. The learned mean/std is then applied to the test data using those same training values. This prevents "data leakage" — accidentally letting test data influence how you normalize.

### The 3 Models

**Random Forest (RF):**

```python
rf_pipe = make_pipeline(RandomForestClassifier(**RF_PARAMS))
# ** unpacks the dict as keyword arguments:
# RandomForestClassifier(n_estimators=500, max_depth=12, ...)
```

A Random Forest builds 500 decision trees. Each tree asks a series of yes/no questions: *"Is neck_inclination > 25°? → yes → Is torso_inclination > 15°? → yes → Predict BAD."* Each tree sees a random subset of features and training data, which makes them different from each other. The final prediction is a majority vote across all 500 trees.

**MLP (Neural Network):**

```python
mlp_pipe = make_pipeline(MLPClassifier(**MLP_PARAMS))
```

A small neural network: `8 features → 64 neurons → 32 neurons → 1 output (0 or 1)`. Each layer applies a weighted sum + nonlinear activation function. The network learns by adjusting weights to reduce prediction error during training.

**Gradient Boosting (GBM):**

```python
gbm_pipe = make_pipeline(GradientBoostingClassifier(**GBM_PARAMS))
```

Builds decision trees sequentially. Tree 1 makes predictions. Tree 2 focuses specifically on the errors Tree 1 made. Tree 3 focuses on the remaining errors. And so on for 200 trees. This "boosting" strategy often achieves high accuracy but is slower to train.

**Ensembles:**

```python
rf_mlp_gbm_ensemble = VotingClassifier(
    estimators=[
        ("rf",  make_pipeline(RandomForestClassifier(**RF_PARAMS))),
        ("mlp", make_pipeline(MLPClassifier(**MLP_PARAMS))),
        ("gbm", make_pipeline(GradientBoostingClassifier(**GBM_PARAMS))),
    ],
    voting="soft",         # Average the probability scores (not just yes/no votes)
    weights=[2, 1, 2],     # RF and GBM get double weight vs MLP
)
```

Instead of one expert making the call, you have 3 experts vote. `voting="soft"` means they share their confidence levels (e.g., RF says 80% confident it's good, MLP says 60%, GBM says 75%) and the ensemble averages those probabilities.

### Train/Test Split

```python
real_reset = real_df.reset_index(drop=True)
idx_train, idx_test = train_test_split(
    np.arange(len(real_reset)),   # Indices of all real rows
    test_size=TEST_SIZE,          # Hold back 20%
    stratify=real_reset["label"].values,
    # stratify = make sure both splits have the same class ratio
    # Without this, you might get a test set that's 90% bad posture by bad luck
    random_state=RANDOM_SEED,
)

# Training set = real training rows + ALL synthetic rows
train_df = pd.concat([train_real, syn_df], ignore_index=True)
# pd.concat = stack two dataframes vertically (like Array.concat in JS)

# Test set = real rows ONLY — synthetic data never goes here
X_test = test_real[feat_cols].values
y_test = test_real["label"].values
# .values converts the pandas DataFrame column to a plain numpy array
# X = features (the 8 angles), y = labels (0 or 1)
```

### Evaluation

```python
def evaluate(name, clf, X_test, y_test, X_cv, y_cv, cv=CV_FOLDS):
    y_pred = clf.predict(X_test)          # Get model's predictions
    y_prob = clf.predict_proba(X_test)[:, 1]
    # predict_proba returns [[P(bad), P(good)], [P(bad), P(good)], ...]
    # [:, 1] takes the second column (P(good)) for all rows

    acc  = accuracy_score(y_test, y_pred)   # % correct
    f1   = f1_score(y_test, y_pred, ...)    # Balance of precision & recall
    prec = precision_score(y_test, y_pred, ...)
    rec  = recall_score(y_test, y_pred, ...)
    auc  = roc_auc_score(y_test, y_prob)    # Area under the ROC curve

    # Cross-validation: run 5 rounds of train/test on different splits
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(clf, X_cv, y_cv, cv=skf, scoring=["f1", "accuracy", "roc_auc"])
    # Returns a dict with arrays of scores from each fold
```

### Feature Importances

```python
def get_tree_importances(pipeline, feature_names):
    clf = pipeline.named_steps["clf"]
    # .named_steps["clf"] accesses a specific stage in the pipeline by name
    return {n: float(v) for n, v in zip(feature_names, clf.feature_importances_)}
    # clf.feature_importances_ = array of importance scores for each feature
    # zip() pairs feature names with their scores
```

```python
def get_mlp_importance_proxy(pipeline, feature_names):
    mlp    = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    W1     = mlp.coefs_[0]
    # mlp.coefs_ = list of weight matrices for each layer
    # coefs_[0] = weights of the first hidden layer, shape (8, 64)
    std = scaler.scale_
    # scale_ = the standard deviation used to normalize each feature during training
    proxy = (np.abs(W1) * std[:, np.newaxis]).sum(axis=1)
    # np.abs() = absolute value (direction doesn't matter, magnitude does)
    # Multiplying by std "un-normalizes" to get real-world importance
    # .sum(axis=1) = sum across all 64 neurons for each of the 8 input features
    proxy /= proxy.sum() + 1e-6   # Normalize to sum to 1 (convert to percentages)
    return {n: float(v) for n, v in zip(feature_names, proxy)}
```

### Saving Models

```python
joblib.dump(rf_pipe,             MODELS_DIR / "rf_pipeline.joblib")
joblib.dump(rf_mlp_gbm_ensemble, MODELS_DIR / "rf_mlp_gbm_pipeline.joblib")
# joblib.dump = serialize the model object to a file
# Like JSON.stringify but for Python objects — saves the trained model's state
```

To use a saved model later:

```python
import joblib
clf = joblib.load("models/rf_mlp_gbm_pipeline.joblib")
# Input must be an array of 8 values in the same order as feature_names
prediction = clf.predict([[12.3, 8.1, 5.2, 0.41, 0.08, 0.02, 145.0, 4.1]])
# Returns [0] for bad posture or [1] for good posture
```

---

## Step 5 — Analytics & Charts (`step5_analytics.py`)

Loads `results.json` (written by Step 4) and generates 7 chart images. **No training happens here.** This is purely visualization.

> ⚠️ If you retrain (run Step 4 again), always re-run Step 5 to refresh the charts.

### Charts Generated

| File | What it shows |
|---|---|
| `01_model_comparison.png` | Bar chart comparing Accuracy, F1, Precision, Recall, AUC across all models |
| `02_cv_f1_boxplot.png` | Box plots of cross-validation F1 scores — shows consistency, not just average |
| `03_confusion_matrices.png` | Where each model gets confused (false positives vs false negatives) |
| `04_feature_importance.png` | Which of the 8 body angles mattered most to each model |
| `05_roc_curves.png` | ROC curves — overall model quality independent of threshold |
| `06_prediction_samples.png` | Random real images with MediaPipe skeleton + model prediction overlay |
| `07_scorecard.png` | Summary table with best values highlighted in green |

### How to read the Confusion Matrix

```
                 Predicted: Bad   Predicted: Good
Actual: Bad           TN               FP
Actual: Good          FN               TP
```

- **TN (True Negative):** Correctly said "Bad posture" — 
- **TP (True Positive):** Correctly said "Good posture" — 
- **FP (False Positive):** Said "Good" but it was actually Bad — model missed a bad posture
- **FN (False Negative):** Said "Bad" but it was actually Good — unfairly flagged good posture

---

## `stepextra.py` — Manual Balancer

A one-off script that force-balances the dataset by **cutting down** the majority class to match the minority class. This is the opposite of Step 3's approach — instead of adding synthetic rows, it removes real rows.

```python
df = pd.read_csv(FEATURES_CSV)
df = df[df["filename"] != "synthetic"].copy()   # Drop any previous synthetic rows
# df[condition] = filter rows where condition is True (like .filter() in JS)

good_df = df[df["label"] == LABEL_GOOD]
bad_df  = df[df["label"] == LABEL_BAD]

print(f"Before: good={len(good_df)}, bad={len(bad_df)}")

# Randomly cut the bad-posture rows down to match the good-posture count
bad_df = bad_df.sample(n=len(good_df), random_state=RANDOM_SEED)
# .sample(n=X) picks X random rows without replacement

balanced = pd.concat([good_df, bad_df]).sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
# pd.concat stacks them, .sample(frac=1) shuffles all rows randomly
# frac=1 means "sample 100% of the rows" = shuffle everything
# reset_index(drop=True) resets row numbers from 0 to N (drop=True discards the old index)

balanced.to_csv(FEATURES_CSV, index=False)   # Overwrites the CSV
```

> ⚠️ **Use with caution:** This permanently removes rows from your CSV. If your dataset is already small, use SMOTE in Step 3 instead. Also note it assumes good posture is the minority class — if it's the other way around, edit the script accordingly.

---

## ML Concepts Explained Simply

### Training vs Testing Data

You split your data into two piles before training. The model **only ever learns from the training set**. The test set is locked away and only used at the very end to measure real-world performance.

If you tested on your training data, the model could just memorize the answers and report perfect accuracy — useless. It's like giving students the answer key to study, then testing them with those exact same questions.

Your pipeline uses an 80/20 split: 80% training, 20% testing.

### Overfitting

When a model memorizes training data instead of learning general patterns. Signs:
- Training accuracy: 99%
- Test accuracy: 70%

Like a student who memorizes exact practice test questions word-for-word but fails when the exam uses different wording.

Your pipeline fights overfitting with:
- `max_depth=12` in RF — limits tree complexity
- `alpha=1e-2` in MLP — L2 regularization (penalizes large weights)
- `early_stopping=True` in MLP — stops training when validation score stops improving
- `min_samples_leaf=4` in RF — leaf nodes need at least 4 examples (no memorizing single data points)

### Cross-Validation

Instead of one train/test split, split your data 5 different ways and train/test each time, then average the results. Much more reliable because results don't depend on one lucky or unlucky split.

```
Full dataset: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Fold 1: Train=[2-10], Test=[1] → Score A
Fold 2: Train=[1,3-10], Test=[2] → Score B
...
Fold 5: Train=[1-9], Test=[10] → Score E

Final CV Score = average(A, B, C, D, E)
```

### Ensemble Models

Combining multiple models and averaging their predictions. Like getting 3 doctors' opinions before a diagnosis.

Your pipeline uses "soft voting" — models share confidence levels and those are averaged:
- RF: 80% confident it's Good Posture
- MLP: 55% confident it's Good Posture
- GBM: 75% confident it's Good Posture
- Ensemble average: 70% → predicts Good Posture

### Precision vs Recall

Two different ways to measure accuracy:

- **Precision:** "Of all the times I said GOOD — how often was I right?" High precision = few false alarms.
- **Recall:** "Of all the actual GOOD examples — how many did I catch?" High recall = few misses.

**F1 Score** = the harmonic mean of both. Use F1 when you care about both precision and recall equally.

### ROC-AUC

A model predicts a probability (e.g., 73% likely good posture). You need a threshold to convert that to a hard yes/no — usually 50%. The ROC curve shows what happens as you move that threshold from 0% to 100%. AUC (Area Under Curve) summarizes the curve in one number:
- 0.5 = random guessing
- 1.0 = perfect
- Your models should aim for > 0.85

### What `.joblib` files are

`joblib` is Python's way of saving trained model objects to disk. A trained model is just a Python object with a lot of numbers inside (the learned weights, tree structures, etc.). `joblib.dump()` serializes it to a file; `joblib.load()` restores it. Think of it like `localStorage.setItem()` and `localStorage.getItem()` — but for complex Python objects.

---

## Running the Pipeline

### Full pipeline

```bash
python run_pipeline.py --dataset dataset/raw
```

### Run specific steps

```bash
python run_pipeline.py --steps 4 5       # Re-train + re-generate charts only
python run_pipeline.py --steps 1 2 3     # Re-augment + re-extract + re-dedupe
```

### Run individual steps

```bash
python steps/step1_augment.py
python steps/step2_extract.py
python steps/step3_dedupe_balance.py
python steps/step4_train.py
python steps/step5_analytics.py
```

### Recommended order when starting fresh

```bash
# 1. Drop your photos into dataset/raw/
# 2. Run the full pipeline
python run_pipeline.py --dataset dataset/raw
# 3. Check outputs/ for your charts and results
# 4. If class balance looks off, run stepextra.py then re-run steps 4–5
python steps/stepextra.py
python run_pipeline.py --steps 4 5
```
