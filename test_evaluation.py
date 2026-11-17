"""Official test-set evaluation of the deployed model — run **once**.

The test split (935 images) has been untouched through Phases 1–3. This script
scores it one time, under the frozen protocol, with the frozen artefact:

* features are extracted by the same function that produced the training matrix
  (``src.features.screening._load_one`` / ``extract_split_features``);
* the model is loaded through the same loader the website uses;
* metrics come from :mod:`src.evaluation.metrics`.

It refuses to overwrite an existing result unless ``--force`` is given, so an
accidental second run cannot quietly replace a score that has already been read.

Outputs (``evaluation/phase4_test/``):
    predictions.csv        per-image truth, prediction, probabilities, confidence
    metrics.json           every aggregate and per-class metric
    per_class.csv          per-class table with support
    confusion_matrix.png   counts and row percentages
    evaluation_config.json model hash, class order, split checks
    evaluation_log.txt     human-readable record of what ran

Run: ``python -m src.evaluation.test_evaluation``
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

import config
from src.evaluation.metrics import (
    per_class_frame,
    plot_confusion_matrix,
    save_classification_report,
    save_predictions,
    summarise,
)
from src.utils import save_json

OUT_DIR = config.EVALUATION_DIR / "phase4_test"

METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1", "macro_f1_ci95")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _encode(y: pd.Series | np.ndarray) -> np.ndarray:
    """Map quality strings to the frozen integer order."""
    arr = np.asarray(y)
    if arr.dtype.kind in "iufb":
        return arr.astype(np.int32)
    index = {name: i for i, name in enumerate(config.QUALITY_CLASSES)}
    unknown = {str(v) for v in arr} - set(index)
    if unknown:
        raise ValueError(f"unmapped quality labels: {unknown}")
    return np.array([index[str(v)] for v in arr], dtype=np.int32)


def _verify_split(manifest: pd.DataFrame) -> dict:
    """Re-prove, immediately before scoring, that the test split is clean."""
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    rest = manifest[manifest["split"] != "test"]

    missing_files = [
        str(r.file_name) for r in test.itertuples(index=False)
        if not (config.PROCESSED_DIR / "test" / r.vegetable / r.quality / r.file_name).exists()
    ]

    checks = {
        "n_test": int(len(test)),
        "n_train": int((manifest["split"] == "train").sum()),
        "n_validation": int((manifest["split"] == "validation").sum()),
        "files_missing_on_disk": len(missing_files),
        "group_ids_shared_with_train_or_validation": int(
            len(set(test["group_id"]) & set(rest["group_id"]))),
        "sha256_shared_with_train_or_validation": int(
            len(set(test["sha256"]) & set(rest["sha256"]))),
        "quality_classes_present": sorted(test["quality"].unique().tolist()),
        "per_class_support": {
            str(k): int(v) for k, v in test["quality"].value_counts().items()
        },
        "per_vegetable_support": {
            str(k): int(v) for k, v in test["vegetable"].value_counts().items()
        },
    }
    checks["pass"] = bool(
        checks["files_missing_on_disk"] == 0
        and checks["group_ids_shared_with_train_or_validation"] == 0
        and checks["sha256_shared_with_train_or_validation"] == 0
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate the deployed model on the test split")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing result (should not be needed)")
    args = ap.parse_args()

    if (OUT_DIR / "metrics.json").exists() and not args.force:
        print(f"Refusing to overwrite {OUT_DIR/'metrics.json'}.\n"
              "The test set is evaluated once; pass --force only for a documented re-run.")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.set_global_seed(config.SEED, include_tf=False)
    t_start = time.perf_counter()
    log: list[str] = []

    def say(line: str) -> None:
        print(line)
        log.append(line)

    say(f"test evaluation started {time.strftime('%Y-%m-%dT%H:%M:%S')}")
    say(f"python {sys.version.split()[0]} on {platform.platform()}")

    # --- manifest ---------------------------------------------------------
    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    checks = _verify_split(manifest)
    say(f"split check: {json.dumps({k: v for k, v in checks.items() if k != 'pass'})}")
    if not checks["pass"]:
        say("ABORT: split verification failed; no predictions were produced.")
        (OUT_DIR / "evaluation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        return 3

    test = manifest[manifest["split"] == "test"].reset_index(drop=True)
    say(f"evaluating {len(test)} test images across "
        f"{len(checks['quality_classes_present'])} quality classes")

    # --- features (training-time extraction) ------------------------------
    from src.features.screening import extract_split_features

    t0 = time.perf_counter()
    feats = extract_split_features("test", workers=8)
    say(f"features extracted in {time.perf_counter() - t0:.1f}s "
        f"({feats.shape[0]} rows x {len(config.QUALITY_CLASSES)} classes)")

    from src.features.handcrafted import ALL_FEATURES

    if list(feats["file_name"]) != list(test["file_name"]):
        say("ABORT: feature rows are not aligned with the manifest order.")
        (OUT_DIR / "evaluation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        return 4

    X = feats[list(ALL_FEATURES)].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = _encode(test["quality"])

    # --- model ------------------------------------------------------------
    from website.utils.model_loader import load_bundle

    model_path = config.FINAL_MODEL_DIR / "rf_handcrafted.joblib"
    bundle = load_bundle()
    say(f"model: {model_path.name} sha256={_sha256(model_path)}")
    say(f"class order: {list(bundle['classes'])}")

    if list(bundle["classes"]) != list(config.QUALITY_CLASSES):
        say("ABORT: artefact class order differs from config.")
        return 5

    t_inf = time.perf_counter()
    proba = bundle["pipeline"].predict_proba(X)
    inference_s = time.perf_counter() - t_inf
    probs = np.zeros((len(X), len(config.QUALITY_CLASSES)), dtype=np.float64)
    for col, cls in enumerate([int(c) for c in bundle["pipeline"].named_steps["rf"].classes_]):
        probs[:, cls] = proba[:, col]
    y_pred = np.argmax(probs, axis=1)

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        say("ABORT: probabilities do not sum to 1.")
        return 6

    say(f"inference: {len(X)} images in {inference_s:.2f}s "
        f"({inference_s / len(X) * 1000:.2f} ms/image, batched)")

    # --- metrics ----------------------------------------------------------
    result = summarise(y_true, y_pred, config.QUALITY_CLASSES)

    weighted = {
        "weighted_precision": float(precision_score(
            y_true, y_pred, average="weighted", zero_division=0)),
        "weighted_recall": float(recall_score(
            y_true, y_pred, average="weighted", zero_division=0)),
    }

    metrics = {
        "n_test_images": int(len(y_true)),
        "accuracy": result["accuracy"],
        "macro_precision": result["macro_precision"],
        "macro_recall": result["macro_recall"],
        "macro_f1": result["macro_f1"],
        "macro_f1_ci95": result["macro_f1_ci95"],
        "weighted_precision": weighted["weighted_precision"],
        "weighted_recall": weighted["weighted_recall"],
        "weighted_f1": result["weighted_f1"],
        "primary_metric": "macro_f1",
        "classes_evaluated": list(config.QUALITY_CLASSES),
        "classes_absent_from_test": result["classes_absent"],
        "per_class": result["per_class"],
        "n_correct": int((y_true == y_pred).sum()),
        "n_incorrect": int((y_true != y_pred).sum()),
    }

    # --- artefacts --------------------------------------------------------
    stem = "A5_test"
    save_predictions(y_true, y_pred, probs, config.QUALITY_CLASSES,
                     OUT_DIR / "predictions.csv",
                     identifiers=test["file_name"].tolist())
    per_class_frame(result).to_csv(OUT_DIR / "per_class.csv", index=False)
    save_classification_report(y_true, y_pred, config.QUALITY_CLASSES,
                              OUT_DIR / "classification_report.txt",
                              title="A5 Random Forest — official test split")
    plot_confusion_matrix(y_true, y_pred, config.QUALITY_CLASSES,
                          OUT_DIR / "confusion_matrix.png",
                          title=f"A5 — official test set (n={len(y_true)})")

    config_payload = {
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": "single pass over the frozen test split; no model selection",
        "model": {
            "run_id": "A5",
            "path": str(model_path),
            "sha256": _sha256(model_path),
            "bytes": model_path.stat().st_size,
            "type": "StandardScaler + RandomForestClassifier",
            "n_estimators": 200,
        },
        "features": {
            "n_features": len(ALL_FEATURES),
            "names": list(ALL_FEATURES),
            "extraction": "src.features.screening.extract_split_features",
            "image_size": config.IMAGE_SIZE,
        },
        "classes": list(config.QUALITY_CLASSES),
        "split_checks": checks,
        "seed": config.SEED,
        "elapsed_s": round(time.perf_counter() - t_start, 2),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    save_json(config_payload, OUT_DIR / "evaluation_config.json")
    save_json(metrics, OUT_DIR / "metrics.json")

    # --- registry ---------------------------------------------------------
    registry = config.ADMIN_DIR / "experiment_registry.csv"
    reg = pd.read_csv(registry)
    mask = reg["run_id"] == "A5"
    reg.loc[mask, "test_accuracy"] = round(metrics["accuracy"], 4)
    reg.loc[mask, "test_macro_f1"] = round(metrics["macro_f1"], 4)
    reg.loc[mask, "test_weighted_f1"] = round(metrics["weighted_f1"], 4)
    reg.loc[mask, "notes"] = (
        "test evaluated once in Phase 4 (evaluation/phase4_test); "
        "validation metrics unchanged")
    reg.to_csv(registry, index=False)
    say("registry: A5 test columns populated")

    say("")
    say(f"TEST MACRO F1 = {metrics['macro_f1']:.4f} "
        f"[{metrics['macro_f1_ci95']['lo']:.4f}, {metrics['macro_f1_ci95']['hi']:.4f}]")
    say(f"TEST ACCURACY = {metrics['accuracy']:.4f}  "
        f"WEIGHTED F1 = {metrics['weighted_f1']:.4f}")
    for row in metrics["per_class"]:
        if row["support"]:
            say(f"  {row['class']:<8} support={row['support']:>3}  "
                f"P={row['precision']:.4f}  R={row['recall']:.4f}  F1={row['f1']:.4f}")
    say(f"elapsed {config_payload['elapsed_s']}s")

    (OUT_DIR / "evaluation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"\nwrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
