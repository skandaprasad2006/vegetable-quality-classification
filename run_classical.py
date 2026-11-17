"""A5 — Random Forest on the 83 handcrafted features.

This is the reference the CNNs must beat.  It is deliberately built to match the
Phase 1 screening protocol exactly — ``StandardScaler`` then
``RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample")`` —
so that the only difference between A5's validation score and the Phase 1
``0.9199`` figure is the *evaluation scheme*: holdout validation versus grouped
cross-validation inside the training split.  Those two numbers are reported
separately and never substituted for one another.

The test split is not touched.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.evaluation.metrics import (
    per_class_frame,
    plot_confusion_matrix,
    save_classification_report,
    save_predictions,
    summarise,
)
from src.features.handcrafted import ALL_FEATURES
from src.training.train import _registry_append
from src.utils import save_json


def _encode_labels(y: np.ndarray) -> np.ndarray:
    """Map quality labels to the canonical integer indices.

    The Phase 1 feature cache stores **string** labels (``"Damaged"`` …), while the
    CNN pipeline uses the integer order in ``config.QUALITY_CLASSES``.  Both must
    agree or the metrics silently compare unlike things.
    """
    y = np.asarray(y)
    # Numeric dtype (int/uint/float/bool) is already encoded.
    if y.dtype.kind in "iufb":
        return y.astype(np.int32)
    # Otherwise the cache holds strings (pandas 3 may persist them as ``object``).
    index = {name: i for i, name in enumerate(config.QUALITY_CLASSES)}
    unknown = {str(v) for v in y} - set(index)
    if unknown:
        raise ValueError(f"unmapped quality labels in feature cache: {unknown}")
    return np.array([index[str(v)] for v in y], dtype=np.int32)


def _cached_features(split: str, smoke: bool = False) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load (or compute and cache) handcrafted features for one split."""
    from src.features.screening import extract_split_features

    cache = config.INTERIM_DIR / f"features_{split}.npz"
    if cache.exists():
        data = np.load(cache, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = _encode_labels(data["y"])
        names = [str(n) for n in data["features"]]
    else:
        df = extract_split_features(split, workers=8)
        X = df[list(ALL_FEATURES)].to_numpy(dtype=np.float32)
        y = _encode_labels(df["quality"].to_numpy())
        config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, y=y, features=np.array(ALL_FEATURES))
        names = list(ALL_FEATURES)

    if smoke:
        X, y = X[:64], y[:64]
    return X, y, names


def run(run_id: str, cfg: dict[str, Any], smoke: bool = False) -> dict[str, Any]:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    seed = cfg.get("seed", config.SEED)
    print(f"[{run_id}] {cfg['label']}")

    X_train, y_train, names = _cached_features("train", smoke)
    X_val, y_val, val_names = _cached_features("validation", smoke)
    if names != val_names:
        raise RuntimeError("feature order mismatch between train and validation caches")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=cfg["n_estimators"],
            class_weight=cfg["class_weight"],
            random_state=seed,
            n_jobs=-1,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    t_inf = time.perf_counter()
    proba = pipeline.predict_proba(X_val)
    pred = pipeline.predict(X_val)
    inference_ms = (time.perf_counter() - t_inf) / max(len(X_val), 1) * 1000

    # Align probability columns to the canonical class order (classes_ are the
    # integer indices the Random Forest was fitted on).
    classes = [int(c) for c in pipeline.named_steps["rf"].classes_]
    full_proba = np.zeros((len(X_val), len(config.QUALITY_CLASSES)), dtype=np.float32)
    for col, cls in enumerate(classes):
        full_proba[:, cls] = proba[:, col]

    quality_metrics = summarise(y_val, pred, config.QUALITY_CLASSES)

    stem = f"{run_id}{'_smoke' if smoke else ''}"
    save_predictions(y_val, pred, full_proba, config.QUALITY_CLASSES,
                     config.METRICS_DIR / f"{stem}_predictions.csv")
    save_classification_report(y_val, pred, config.QUALITY_CLASSES,
                              config.EVAL_REPORT_DIR / f"{stem}_classification_report.txt",
                              title=f"{run_id} quality classification (validation)")
    plot_confusion_matrix(y_val, pred, config.QUALITY_CLASSES,
                          config.CONFUSION_DIR / f"{stem}_quality.png",
                          title=f"{run_id} — quality (validation, n={len(X_val)})")
    per_class_frame(quality_metrics).to_csv(
        config.METRICS_DIR / f"{stem}_per_class.csv", index=False)

    importances = pd.DataFrame({
        "feature": names,
        "importance": pipeline.named_steps["rf"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(config.METRICS_DIR / f"{stem}_feature_importance.csv", index=False)

    model_path = config.FINAL_MODEL_DIR / f"rf_handcrafted{'_smoke' if smoke else ''}.joblib"
    config.FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "features": names,
                 "classes": list(config.QUALITY_CLASSES)}, model_path)

    result: dict[str, Any] = {
        "run_id": run_id,
        "label": cfg["label"],
        "stage": cfg["stage"],
        "config": cfg,
        "model": {
            "name": "StandardScaler + RandomForestClassifier",
            "n_estimators": cfg["n_estimators"],
            "class_weight": cfg["class_weight"],
            "n_features": len(names),
            "features": names,
        },
        "train_images": int(len(X_train)),
        "val_images": int(len(X_val)),
        "train_time_s": round(train_time, 2),
        "inference_ms_per_image_batched": round(inference_ms, 3),
        "latency": {"batch1_ms_median": round(inference_ms, 3)},
        "resources": {},
        "quality": quality_metrics,
        "checkpoints": {"model": str(model_path)},
        "top_features": importances.head(15).to_dict("records"),
    }
    save_json(result, config.METRICS_DIR / f"{stem}_validation.json")

    if not smoke:
        _registry_append({
            "run_id": run_id,
            "date": time.strftime("%Y-%m-%d"),
            "phase": "2",
            "stage": cfg["stage"],
            "backbone": "random_forest",
            "head": "single",
            "preprocessing": "handcrafted_83_features",
            "augmentation": False,
            "class_weight": cfg["class_weight"],
            "img_size": "",
            "batch_size": "",
            "epochs_planned": "",
            "epochs_run": "",
            "optimizer": "closed_form",
            "lr": "",
            "freeze_mode": "",
            "normalization": "",
            "es_patience": "",
            "val_accuracy": round(quality_metrics["accuracy"], 4),
            "val_macro_f1": round(quality_metrics["macro_f1"], 4),
            "val_weighted_f1": round(quality_metrics["weighted_f1"], 4),
            "test_accuracy": "", "test_macro_f1": "", "test_weighted_f1": "",
            "train_time_s": round(train_time, 1),
            "inference_ms_per_image": round(inference_ms, 3),
            "checkpoint_path": str(model_path),
            "notes": "validation metrics only; test split untouched",
        })

    print(f"[{run_id}] quality macro-F1 = {quality_metrics['macro_f1']:.4f} "
          f"(acc {quality_metrics['accuracy']:.4f})")
    return result
