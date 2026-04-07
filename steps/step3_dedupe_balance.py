"""
Step 3 — Deduplication & Class Balancing  (fixed)

Key fixes:
- SMOTE / oversampling now APPENDS synthetic rows to real rows
  instead of replacing the entire CSV with only synthetic data.
- Deduplication runs within each class independently.
- Synthetic rows are flagged (filename='synthetic') and excluded
  from the image-sort step so no file-copy errors occur.
- arm_inclination column is tolerated whether present or absent.
"""

import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    AUG_DIR, GOOD_DIR, BAD_DIR, FEATURES_CSV,
    HASH_SIZE, SIMILARITY_THRESHOLD,
    LABEL_GOOD, LABEL_BAD, RANDOM_SEED,
)

try:
    import imagehash
    from PIL import Image as PILImage
    HAS_IMAGEHASH = True
except ImportError:
    HAS_IMAGEHASH = False
    print("  [INFO] imagehash not installed — using OpenCV dHash fallback.")

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("  [INFO] imbalanced-learn not installed — will use pandas oversampling.")


# ── Hashing ─────────────────────────────────────────────────────

def dhash_cv(img_path: Path, hash_size: int = HASH_SIZE) -> str:
    import cv2
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return ""
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff    = resized[:, 1:] > resized[:, :-1]
    return "".join(str(int(b)) for b in diff.flatten())

def hamming(h1: str, h2: str) -> int:
    if len(h1) != len(h2):
        return 9999
    return sum(c1 != c2 for c1, c2 in zip(h1, h2))

def compute_hash(img_path: Path):
    if HAS_IMAGEHASH:
        try:
            return str(imagehash.phash(PILImage.open(img_path), hash_size=HASH_SIZE))
        except Exception:
            return dhash_cv(img_path)
    return dhash_cv(img_path)


# ── Deduplication ────────────────────────────────────────────────

def deduplicate(filenames: list, source_dir: Path) -> list:
    paths  = [source_dir / f for f in filenames]
    hashes = {}
    for p in tqdm(paths, desc="    Hashing", unit="img", leave=False):
        if p.exists():
            hashes[p.name] = compute_hash(p)

    # Originals first, then augmented
    sorted_names = sorted(hashes.keys(), key=lambda n: (1 if "_aug" in n else 0, n))

    kept        = []
    seen_hashes = []
    for name in sorted_names:
        h = hashes.get(name, "")
        if not h:
            continue
        if any(hamming(h, sh) <= SIMILARITY_THRESHOLD for sh in seen_hashes):
            continue
        kept.append(name)
        seen_hashes.append(h)

    print(f"    {len(filenames)} → {len(kept)} kept  "
        f"({len(filenames) - len(kept)} near-duplicates removed)")
    return kept


# ── Sorting ──────────────────────────────────────────────────────

def sort_images(kept_names: list, source_dir: Path):
    for dst in [GOOD_DIR, BAD_DIR]:
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True, exist_ok=True)

    for name in tqdm(kept_names, desc="    Sorting", unit="img", leave=False):
        lower = name.lower()
        if "goodposture" in lower:
            dst = GOOD_DIR / name
        elif "badposture" in lower:
            dst = BAD_DIR / name
        else:
            continue
        src = source_dir / name
        if src.exists():
            shutil.copy2(str(src), str(dst))


# ── CSV filtering & balancing ────────────────────────────────────

def balance_csv(kept_set: set):
    if not FEATURES_CSV.exists():
        print("  [WARN] features.csv not found — skipping CSV balance.")
        return

    df = pd.read_csv(FEATURES_CSV)

    # ── Keep only real rows that survived deduplication ──────────
    # Synthetic rows from a previous run are dropped and regenerated fresh.
    real_df = df[df["filename"] != "synthetic"].copy()
    real_df = real_df[real_df["filename"].isin(kept_set)].reset_index(drop=True)

    print(f"  Real rows after dedup filter: {len(real_df)}")
    if real_df.empty:
        print("  [ERROR] No real rows survived — check Step 2 ran correctly.")
        return

    counts = real_df["label"].value_counts()
    print(f"  Class counts: {dict(counts)}")

    feature_cols = [c for c in real_df.columns
                    if c not in ("label", "filename", "side")]

    # Impute any NaN columns (e.g. arm_inclination when elbow absent)
    for col in feature_cols:
        if real_df[col].isna().any():
            median = real_df[col].median()
            real_df[col] = real_df[col].fillna(median)
            print(f"    Imputed NaNs in '{col}' with median {median:.4f}")

    majority = counts.idxmax()
    minority = counts.idxmin()
    ratio    = counts[majority] / (counts[minority] + 1e-6)

    synthetic_rows = []

    if ratio > 1.5:
        print(f"  Imbalance ratio {ratio:.2f} — balancing…")

        if HAS_SMOTE and counts[minority] >= 6:
            X = real_df[feature_cols].values
            y = real_df["label"].values
            k = min(5, counts[minority] - 1)
            smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=k)
            X_res, y_res = smote.fit_resample(X, y)

            # X_res contains the original rows + synthetic rows
            # We only want the NEW synthetic rows (appended after original)
            n_original = len(real_df)
            X_syn = X_res[n_original:]
            y_syn = y_res[n_original:]

            if len(X_syn) > 0:
                df_syn = pd.DataFrame(X_syn, columns=feature_cols)
                df_syn["label"]    = y_syn
                df_syn["filename"] = "synthetic"
                df_syn["side"]     = "N/A"
                synthetic_rows.append(df_syn)
                print(f"    SMOTE added {len(df_syn)} synthetic rows")
            else:
                print("    SMOTE: classes already balanced, no rows added")

        else:
            # Simple oversampling: duplicate minority rows
            min_df  = real_df[real_df["label"] == minority]
            maj_df  = real_df[real_df["label"] == majority]
            needed  = len(maj_df) - len(min_df)
            sampled = min_df.sample(needed, replace=True, random_state=RANDOM_SEED)
            sampled = sampled.copy()
            sampled["filename"] = "synthetic"
            synthetic_rows.append(sampled)
            print(f"    Oversampling added {len(sampled)} rows")
    else:
        print(f"  Classes balanced (ratio {ratio:.2f}) — no resampling needed.")

    # ── Combine real + synthetic and save ────────────────────────
    all_parts = [real_df] + synthetic_rows
    final_df  = pd.concat(all_parts, ignore_index=True)
    final_df  = final_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    final_df.to_csv(FEATURES_CSV, index=False)

    real_count = (final_df["filename"] != "synthetic").sum()
    syn_count  = (final_df["filename"] == "synthetic").sum()
    print(f"  ✅ CSV saved: {len(final_df)} total rows  "
        f"({real_count} real + {syn_count} synthetic)")
    print(f"  Final label distribution:\n{final_df['label'].value_counts().to_string()}")


# ── Main runner ──────────────────────────────────────────────────

def run(source_dir: Path = AUG_DIR):
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_imgs  = [p.name for p in source_dir.rglob("*")
                if p.suffix.lower() in supported]

    if not all_imgs:
        print(f"  [WARN] No images found in {source_dir}")
        return

    good_imgs = [n for n in all_imgs if "goodposture" in n.lower()]
    bad_imgs  = [n for n in all_imgs if "badposture"  in n.lower()]
    other     = len(all_imgs) - len(good_imgs) - len(bad_imgs)

    print(f"  Raw image counts → good: {len(good_imgs)}, "
        f"bad: {len(bad_imgs)}, unlabelled: {other}")

    print("  Deduplicating Good Posture images…")
    good_kept = deduplicate(good_imgs, source_dir)

    print("  Deduplicating Bad Posture images…")
    bad_kept  = deduplicate(bad_imgs,  source_dir)

    kept_names = good_kept + bad_kept

    print("  Sorting into class folders…")
    sort_images(kept_names, source_dir)
    print(f"    Good folder: {len(list(GOOD_DIR.rglob('*.*')))} images")
    print(f"    Bad  folder: {len(list(BAD_DIR.rglob('*.*')))} images")

    print("  Filtering & balancing feature CSV…")
    balance_csv(set(kept_names))

    print("  ✅ Step 3 complete.")


if __name__ == "__main__":
    run()
