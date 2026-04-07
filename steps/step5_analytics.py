"""
Step 5 — Analytics & Visualisation  (fixed: uses MediaPipe Tasks API)

Generates:
  1. Model comparison bar chart (Acc / F1 / Precision / Recall / AUC)
  2. Cross-validation F1 box plots
  3. Confusion matrices (3 side-by-side)
  4. Feature importance charts (RF + MLP proxy)
  5. ROC curves (all 3 models)
  6. Prediction sample grid (randomised each run) with MediaPipe overlay
  7. Summary score card
"""

import json
import random
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from math import atan2, degrees, acos
from pathlib import Path
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import train_test_split

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    GOOD_DIR, BAD_DIR, AUG_DIR,
    LABEL_GOOD, LABEL_BAD, LABEL_NAMES,
    TEST_SIZE, RANDOM_SEED,
    VISIBILITY_THRESHOLD,
)

RESULTS_JSON = OUTPUTS_DIR / "results.json"
MODEL_PATH   = "models/pose_landmarker_heavy.task"

PALETTE = {
    "RF":       "#4C72B0",
    "MLP":      "#DD8452",
    "Ensemble": "#55A868",
}
PALETTE_LIST = list(PALETTE.values())

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.dpi":         130,
})


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def load_results():
    with open(RESULTS_JSON) as f:
        return json.load(f)

def load_features():
    df = pd.read_csv(FEATURES_CSV).dropna()
    meta_cols    = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X   = df[feature_cols].values
    y   = df["label"].values
    fns = df["filename"].values
    return X, y, fns, feature_cols

def load_models():
    return {
        "Random Forest":       joblib.load(MODELS_DIR / "rf_pipeline.joblib"),
        "MLP":                 joblib.load(MODELS_DIR / "mlp_pipeline.joblib"),
        "RF + MLP (Ensemble)": joblib.load(MODELS_DIR / "ensemble_pipeline.joblib"),
    }

def save(fig, name):
    path = OUTPUTS_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  💾 {path.name}")
    return path


# ═══════════════════════════════════════════════════════════════
#  CHART 1 — Model comparison bar chart
# ═══════════════════════════════════════════════════════════════

def chart_comparison(results):
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc"]
    labels  = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC"]
    models  = results["models"]

    x = np.arange(len(metrics))
    w = 0.22
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    for i, (m, col) in enumerate(zip(models, PALETTE_LIST)):
        vals = [m[k] for k in metrics]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=m["name"], color=col, alpha=0.9, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.007,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="white", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white")
    ax.set_title("Model Comparison — All Metrics", color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#444", fontsize=9)
    ax.grid(axis="y", color="#333", linewidth=0.6, zorder=0)
    return save(fig, "01_model_comparison.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 2 — CV F1 box plots
# ═══════════════════════════════════════════════════════════════

def chart_cv_boxplot(results):
    models = results["models"]
    data   = [m["cv_f1"] for m in models]
    names  = [m["name"].replace("RF + MLP ", "Ensemble") for m in models]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, col in zip(bp["boxes"], PALETTE_LIST):
        patch.set_facecolor(col)
        patch.set_alpha(0.85)
    for el in bp["whiskers"] + bp["caps"] + bp["fliers"]:
        el.set_color("#aaa")

    ax.set_xticklabels(names, color="white", fontsize=10)
    ax.set_ylabel("F1 Score (CV fold)", color="white")
    ax.set_title("Cross-Validation F1 Distribution", color="white", fontsize=13)
    ax.grid(axis="y", color="#333", linewidth=0.6)
    return save(fig, "02_cv_f1_boxplot.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 3 — Confusion matrices
# ═══════════════════════════════════════════════════════════════

def chart_confusion_matrices(results):
    models     = results["models"]
    class_names = [LABEL_NAMES[LABEL_BAD], LABEL_NAMES[LABEL_GOOD]]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor("#0f1117")
    fig.suptitle("Confusion Matrices", color="white", fontsize=13, y=1.02)

    for ax, m, col in zip(axes, models, PALETTE_LIST):
        cm      = np.array(m["confusion_matrix"])
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-6)

        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_facecolor("#1a1d2e")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(class_names, color="white", fontsize=9)
        ax.set_yticklabels(class_names, color="white", fontsize=9)
        ax.set_xlabel("Predicted", color="white", fontsize=9)
        ax.set_ylabel("True",      color="white", fontsize=9)
        ax.set_title(m["name"].replace("RF + MLP ", "Ensemble"),
                     color=col, fontsize=10, pad=6)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.1f}%)",
                        ha="center", va="center", fontsize=10,
                        color="white" if cm_norm[i, j] < 0.5 else "black")

    plt.tight_layout()
    return save(fig, "03_confusion_matrices.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 4 — Feature importances
# ═══════════════════════════════════════════════════════════════

def chart_feature_importance(results):
    rf_imp  = results["models"][0].get("feature_importance", {})
    mlp_imp = results["models"][1].get("feature_importance", {})
    if not rf_imp:
        print("  [SKIP] Feature importance data not found.")
        return

    feat_names = list(rf_imp.keys())
    rf_vals    = [rf_imp[f]          for f in feat_names]
    mlp_vals   = [mlp_imp.get(f, 0)  for f in feat_names]

    order      = np.argsort(rf_vals)[::-1]
    feat_names = [feat_names[i] for i in order]
    rf_vals    = [rf_vals[i]    for i in order]
    mlp_vals   = [mlp_vals[i]   for i in order]

    y = np.arange(len(feat_names))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.barh(y + w/2, rf_vals,  w, label="RF",  color=PALETTE["RF"],  alpha=0.9)
    ax.barh(y - w/2, mlp_vals, w, label="MLP", color=PALETTE["MLP"], alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(feat_names, color="white", fontsize=9)
    ax.set_xlabel("Importance", color="white")
    ax.set_title("Feature Importance (RF) & Proxy (MLP)", color="white", fontsize=12)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#444")
    ax.grid(axis="x", color="#333", linewidth=0.6)
    return save(fig, "04_feature_importance.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 5 — ROC curves
# ═══════════════════════════════════════════════════════════════

def chart_roc_curves(models_dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    for (name, clf), col in zip(models_dict.items(), PALETTE_LIST):
        y_prob = (clf.predict_proba(X_test)[:, 1]
                if hasattr(clf, "predict_proba")
                else clf.decision_function(X_test))
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_val     = auc(fpr, tpr)
        short       = name.replace("RF + MLP ", "")
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{short}  (AUC={roc_val:.3f})")

    ax.plot([0, 1], [0, 1], color="#555", lw=1, linestyle="--")
    ax.set_xlabel("False Positive Rate", color="white")
    ax.set_ylabel("True Positive Rate",  color="white")
    ax.set_title("ROC Curves", color="white", fontsize=13)
    ax.legend(facecolor="#1e2130", labelcolor="white", edgecolor="#444", fontsize=9)
    ax.grid(color="#333", linewidth=0.5)
    return save(fig, "05_roc_curves.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 6 — Sample prediction grid with MediaPipe Tasks overlay
# ═══════════════════════════════════════════════════════════════

SKELETON_CONNECTIONS = [
    ("ear", "shoulder"), ("shoulder", "hip"),
]
KP_COLORS = {
    "ear":      (0,   255, 180),
    "shoulder": (0,   180, 255),
    "hip":      (255,  60, 120),
}


def _detect_side(lms):
    nose_x = lms[0].x
    if nose_x < 0.42: return "RIGHT"
    if nose_x > 0.58: return "LEFT"
    lv = sum(lms[i].visibility for i in [11, 23, 25])
    rv = sum(lms[i].visibility for i in [12, 24, 26])
    return "LEFT" if lv >= rv else "RIGHT"


def draw_pose_on_image(img_bgr, landmarker):
    """Draw hip→head skeleton using the Tasks API landmarker."""
    h, w = img_bgr.shape[:2]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np.ascontiguousarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    )
    result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        return img_bgr.copy()

    lms  = result.pose_landmarks[0]
    out  = img_bgr.copy()
    side = _detect_side(lms)

    idx_map = (
        {"ear": 7, "shoulder": 11, "hip": 23}
        if side == "LEFT"
        else {"ear": 8, "shoulder": 12, "hip": 24}
    )

    def pt(name):
        lm = lms[idx_map[name]]
        return (int(lm.x * w), int(lm.y * h)) if lm.visibility >= VISIBILITY_THRESHOLD else None

    kpts = {name: pt(name) for name in idx_map}
    kpts = {k: v for k, v in kpts.items() if v is not None}

    for (n1, n2) in SKELETON_CONNECTIONS:
        if n1 in kpts and n2 in kpts:
            cv2.line(out, kpts[n1], kpts[n2], (200, 200, 200), 2, cv2.LINE_AA)

    for name, pos in kpts.items():
        col = KP_COLORS.get(name, (255, 255, 255))
        cv2.circle(out, pos, 7, col, -1, cv2.LINE_AA)
        cv2.circle(out, pos, 7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(out, name[:3], (pos[0] + 8, pos[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    return out


def find_image_path(filename: str):
    for d in [AUG_DIR, GOOD_DIR, BAD_DIR]:
        p = d / filename
        if p.exists():
            return p
        hits = list(d.rglob(filename))
        if hits:
            return hits[0]
    return None


def make_landmarker():
    BaseOptions             = mp_python.BaseOptions
    PoseLandmarker          = mp_vision.PoseLandmarker
    PoseLandmarkerOptions   = mp_vision.PoseLandmarkerOptions
    VisionRunningMode       = mp_vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
    )
    return PoseLandmarker.create_from_options(options)


def chart_sample_predictions(models_dict, X, y, filenames, n_per_group=4):
    rng  = random.Random()          # unseeded → different every run
    clf  = models_dict["RF + MLP (Ensemble)"]
    pred = clf.predict(X)

    good_correct  = np.where((y == LABEL_GOOD) & (pred == LABEL_GOOD))[0].tolist()
    bad_correct   = np.where((y == LABEL_BAD)  & (pred == LABEL_BAD))[0].tolist()
    misclassified = np.where(y != pred)[0].tolist()

    # Skip synthetic rows (no image file)
    real_mask = np.array([f != "synthetic" for f in filenames])
    good_correct  = [i for i in good_correct  if real_mask[i]]
    bad_correct   = [i for i in bad_correct   if real_mask[i]]
    misclassified = [i for i in misclassified if real_mask[i]]

    rng.shuffle(good_correct)
    rng.shuffle(bad_correct)
    rng.shuffle(misclassified)

    groups = [
        ("Correctly → Good Posture", good_correct[:n_per_group],  "#55A868"),
        ("Correctly → Bad Posture",  bad_correct[:n_per_group],   "#DD8452"),
        ("Misclassified",            misclassified[:n_per_group], "#E74C3C"),
    ]

    n_cols = n_per_group
    n_rows = len(groups)

    landmarker = make_landmarker()

    fig = plt.figure(figsize=(n_cols * 3.2, n_rows * 3.5))
    fig.patch.set_facecolor("#0f1117")

    cell = 1
    for row_i, (group_name, indices, group_col) in enumerate(groups):
        for col_i in range(n_cols):
            ax = fig.add_subplot(n_rows, n_cols, cell)
            ax.set_facecolor("#0f1117")
            ax.axis("off")
            cell += 1

            if col_i < len(indices):
                idx  = indices[col_i]
                fn   = filenames[idx]
                path = find_image_path(fn)

                if path:
                    img = cv2.imread(str(path))
                    if img is not None:
                        img = draw_pose_on_image(img, landmarker)
                        img = cv2.resize(img, (280, 280))
                        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

                true_name = LABEL_NAMES[y[idx]]
                pred_name = LABEL_NAMES[pred[idx]]
                correct   = (y[idx] == pred[idx])
                ax.set_title(f"T:{true_name[:4]}  P:{pred_name[:4]}",
                            color="#55A868" if correct else "#E74C3C",
                            fontsize=8, pad=3)

        fig.text(0.01, 1 - (row_i + 0.5) / n_rows,
                group_name, va="center", ha="left",
                color=group_col, fontsize=10, rotation=90)

    landmarker.close()

    fig.suptitle("Prediction Samples (Ensemble) — MediaPipe Overlay",
                color="white", fontsize=12, y=1.01)
    plt.tight_layout()
    return save(fig, "06_prediction_samples.png")


# ═══════════════════════════════════════════════════════════════
#  CHART 7 — Score card
# ═══════════════════════════════════════════════════════════════

def chart_scorecard(results):
    models  = results["models"]
    metrics = ["accuracy", "f1", "precision", "recall", "roc_auc", "cv_f1_mean", "cv_f1_std"]
    headers = ["Accuracy", "F1", "Precision", "Recall", "ROC-AUC", "CV-F1 μ", "CV-F1 σ"]

    fig, ax = plt.subplots(figsize=(11, 2.2))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.axis("off")

    rows = []
    for m in models:
        short = m["name"].replace("RF + MLP ", "RF+MLP\n")
        rows.append([short] + [f"{m[k]:.4f}" for k in metrics])

    table = ax.table(cellText=rows, colLabels=["Model"] + headers,
                    loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#333")
        if r == 0:
            cell.set_facecolor("#1e2130")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#161922" if r % 2 == 0 else "#1a1d2e")
            cell.set_text_props(color="white")
            if c > 0:
                col_vals = [float(rows[ri][c]) for ri in range(len(rows))]
                best_idx = np.argmin(col_vals) if c == len(headers) else np.argmax(col_vals)
                if r - 1 == best_idx:
                    cell.set_facecolor("#1d3a26")

    ax.set_title("Model Score Card", color="white", fontsize=12, pad=8)
    return save(fig, "07_scorecard.png")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def run():
    if not RESULTS_JSON.exists():
        print("  [ERROR] results.json not found. Run Step 4 first.")
        return

    results     = load_results()
    X, y, fns, feature_names = load_features()
    models_dict = load_models()

    # Reproduce the same real-only test split used in Step 4
    real_mask = np.array([f != "synthetic" for f in fns])
    X_real    = X[real_mask]
    y_real    = y[real_mask]
    fns_real  = fns[real_mask]

    _, X_test, _, y_test, _, fns_test = train_test_split(
        X_real, y_real, fns_real,
        test_size=TEST_SIZE, stratify=y_real, random_state=RANDOM_SEED,
    )

    print("  Generating charts…")
    chart_comparison(results)
    chart_cv_boxplot(results)
    chart_confusion_matrices(results)
    chart_feature_importance(results)
    chart_roc_curves(models_dict, X_test, y_test)
    chart_sample_predictions(models_dict, X, y, fns)
    chart_scorecard(results)

    print(f"\n  ✅ All charts saved to: {OUTPUTS_DIR}")
    for f in sorted(OUTPUTS_DIR.glob("*.png")):
        print(f"    {f.name}")


if __name__ == "__main__":
    run()
