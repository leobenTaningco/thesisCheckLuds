"""
Step 4 — Model Training  (fixed)

Fixes:
- Reads whatever feature columns exist in the CSV (no hardcoded list).
- Imputes NaN columns (e.g. arm_inclination when elbow was absent).
- Guards against empty dataset with a clear error message.
- Cross-validation uses only real rows (not synthetic) to avoid
  data-leakage from oversampled minority rows.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    FEATURES_CSV, MODELS_DIR, OUTPUTS_DIR,
    RF_PARAMS, MLP_PARAMS,
    TEST_SIZE, RANDOM_SEED,
    LABEL_NAMES,
)

RESULTS_JSON = OUTPUTS_DIR / "results.json"
CV_FOLDS     = 5


# ── Data loading ─────────────────────────────────────────────────

def load_data():
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(
            f"Feature CSV not found: {FEATURES_CSV}\n"
            "Run Step 2 first."
        )

    df = pd.read_csv(FEATURES_CSV)
    print(f"  CSV loaded: {len(df)} total rows")

    # Separate real vs synthetic
    real_df = df[df["filename"] != "synthetic"].copy()
    syn_df  = df[df["filename"] == "synthetic"].copy()
    print(f"  Real rows: {len(real_df)}  |  Synthetic rows: {len(syn_df)}")

    if real_df.empty:
        raise ValueError(
            "CSV contains 0 real rows (all rows are synthetic).\n"
            "This means Step 2 produced no valid features.\n"
            "Re-run Step 2 and check the skip reasons printed during extraction."
        )

    # Feature columns = everything except metadata
    meta_cols    = {"label", "filename", "side"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    # Use full dataframe (real + synthetic) for training
    df_train = df.copy()
    X_all = df_train[feature_cols].values
    y_all = df_train["label"].values

    # Real-only arrays for stratified split and CV
    X_real = real_df[feature_cols].values
    y_real = real_df["label"].values

    print(f"\n  Label distribution (real rows):")
    for lbl, name in LABEL_NAMES.items():
        n = (y_real == lbl).sum()
        print(f"    {name}: {n}")

    if len(np.unique(y_real)) < 2:
        raise ValueError(
            "Only one class found in real rows. "
            "Check that both 'goodPosture' and 'badPosture' images were processed."
        )

    return X_all, y_all, X_real, y_real, feature_cols


# ── Evaluation ───────────────────────────────────────────────────

def evaluate(name, clf, X_test, y_test, X_cv, y_cv, cv=CV_FOLDS):
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    else:
        raw = clf.decision_function(X_test)
        y_prob = (raw - raw.min()) / (raw.ptp() + 1e-6)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="binary", zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred).tolist()
    cr   = classification_report(
        y_test, y_pred,
        target_names=list(LABEL_NAMES.values()),
        output_dict=True,
        zero_division=0,
    )

    skf    = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    cv_res = cross_validate(
        clf, X_cv, y_cv,
        cv=skf,
        scoring=["f1", "accuracy", "roc_auc"],
        n_jobs=-1,
    )

    print(f"\n  ── {name} ──")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    ROC-AUC  : {auc:.4f}")
    print(f"    CV F1    : {np.mean(cv_res['test_f1']):.4f} "
          f"± {np.std(cv_res['test_f1']):.4f}")

    return {
        "name":       name,
        "accuracy":   float(acc),
        "f1":         float(f1),
        "precision":  float(prec),
        "recall":     float(rec),
        "roc_auc":    float(auc),
        "confusion_matrix":        cm,
        "classification_report":   cr,
        "cv_f1":      cv_res["test_f1"].tolist(),
        "cv_accuracy":cv_res["test_accuracy"].tolist(),
        "cv_auc":     cv_res["test_roc_auc"].tolist(),
        "cv_f1_mean": float(np.mean(cv_res["test_f1"])),
        "cv_f1_std":  float(np.std(cv_res["test_f1"])),
    }


# ── Feature importance ────────────────────────────────────────────

def get_rf_importances(pipeline, feature_names):
    rf = pipeline.named_steps["clf"]
    return {n: float(v) for n, v in zip(feature_names, rf.feature_importances_)}

def get_mlp_importance_proxy(pipeline, feature_names):
    mlp    = pipeline.named_steps["clf"]
    scaler = pipeline.named_steps["scaler"]
    W1     = mlp.coefs_[0]
    std    = scaler.scale_
    proxy  = (np.abs(W1) * std[:, np.newaxis]).sum(axis=1)
    proxy /= proxy.sum() + 1e-6
    return {n: float(v) for n, v in zip(feature_names, proxy)}


# ── Pipeline builder ──────────────────────────────────────────────

def make_pipeline(clf):
    """Impute → Scale → Classify."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("clf",     clf),
    ])


# ── Main runner ───────────────────────────────────────────────────

def run():
    X_all, y_all, X_real, y_real, feature_names = load_data()

    # Train/test split on real rows only (stratified)
    X_train_r, X_test, y_train_r, y_test = train_test_split(
        X_real, y_real,
        test_size=TEST_SIZE,
        stratify=y_real,
        random_state=RANDOM_SEED,
    )

    # Combine real train rows with all synthetic rows for actual training
    syn_mask  = np.array([True] * len(X_all))  # placeholder
    # Rebuild: real_train rows + all synthetic rows
    df_full   = pd.read_csv(FEATURES_CSV)
    meta_cols = {"label", "filename", "side"}
    feat_cols = [c for c in df_full.columns if c not in meta_cols]

    real_df_full = df_full[df_full["filename"] != "synthetic"]
    syn_df_full  = df_full[df_full["filename"] == "synthetic"]

    # Indices that are in test set: match by position in real_df
    real_reset = real_df_full.reset_index(drop=True)
    test_n     = int(len(real_reset) * TEST_SIZE)
    # Use same random split reproducibly
    from sklearn.model_selection import train_test_split as tts
    idx_train, idx_test = tts(
        np.arange(len(real_reset)),
        test_size=TEST_SIZE, stratify=real_reset["label"].values,
        random_state=RANDOM_SEED,
    )
    train_real = real_reset.iloc[idx_train]
    test_real  = real_reset.iloc[idx_test]

    # Training data = real train rows + synthetic rows
    train_df = pd.concat([train_real, syn_df_full], ignore_index=True)
    X_train  = train_df[feat_cols].values
    y_train  = train_df["label"].values
    X_test   = test_real[feat_cols].values
    y_test   = test_real["label"].values

    print(f"\n  Train: {len(X_train)} (real {len(train_real)} + "
          f"synthetic {len(syn_df_full)})  |  Test: {len(X_test)} (real only)")

    # ── Build pipelines ──────────────────────────────────────────
    rf_pipe  = make_pipeline(RandomForestClassifier(**RF_PARAMS))
    mlp_pipe = make_pipeline(MLPClassifier(**MLP_PARAMS))

    voting_clf = VotingClassifier(
        estimators=[("rf", rf_pipe), ("mlp", mlp_pipe)],
        voting="soft",
        weights=[1, 1],
    )

    # ── Train ────────────────────────────────────────────────────
    print("\n  Training Random Forest…")
    rf_pipe.fit(X_train, y_train)

    print("  Training MLP…")
    mlp_pipe.fit(X_train, y_train)

    print("  Training Soft Voting Ensemble…")
    voting_clf.fit(X_train, y_train)

    # ── Evaluate (CV on real rows only) ─────────────────────────
    rf_metrics  = evaluate("Random Forest",       rf_pipe,    X_test, y_test, X_real, y_real)
    mlp_metrics = evaluate("MLP",                 mlp_pipe,   X_test, y_test, X_real, y_real)
    ens_metrics = evaluate("RF + MLP (Ensemble)", voting_clf, X_test, y_test, X_real, y_real)

    rf_metrics["feature_importance"]  = get_rf_importances(rf_pipe, feature_names)
    mlp_metrics["feature_importance"] = get_mlp_importance_proxy(mlp_pipe, feature_names)

    # ── Save models ──────────────────────────────────────────────
    joblib.dump(rf_pipe,    MODELS_DIR / "rf_pipeline.joblib")
    joblib.dump(mlp_pipe,   MODELS_DIR / "mlp_pipeline.joblib")
    joblib.dump(voting_clf, MODELS_DIR / "ensemble_pipeline.joblib")
    print(f"\n  Models saved → {MODELS_DIR}")

    results = {
        "models":        [rf_metrics, mlp_metrics, ens_metrics],
        "feature_names": feature_names,
        "test_size":     TEST_SIZE,
        "train_n":       int(len(X_train)),
        "test_n":        int(len(X_test)),
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Results saved → {RESULTS_JSON}")
    print("\n  ✅ Training complete.")
    return results


if __name__ == "__main__":
    run()
