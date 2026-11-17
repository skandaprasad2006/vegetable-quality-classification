"""Single-run training entry point for Phase 2.

One process per run, so memory is returned to the OS between experiments and one
failure cannot corrupt the others.

Guarantees enforced here
------------------------
* **Seeds** are fixed from ``config.SEED`` before any model is built.
* **Validation only** is used for early stopping, LR scheduling and checkpoint
  selection.  The test split is never loaded.
* **Resource monitoring** samples RSS and CPU every second; training is stopped
  with a recorded reason if RSS exceeds 85% of physical RAM.
* **Everything is persisted**: config, history, checkpoints, per-sample predictions,
  classification reports, confusion matrices, timings.
* A registry row is appended **only after a run completes**.

Run: ``python -m src.training.train --run-id A2 [--smoke]``
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

import config
from src.evaluation.metrics import (
    per_class_frame,
    plot_confusion_matrix,
    save_classification_report,
    save_predictions,
    summarise,
)
from src.models.architectures import (
    EXPECTED_NORMALIZATION,
    build_model,
    compile_model,
    describe,
)
from src.training import run_configs
from src.training.data import (
    compute_class_weights,
    make_dataset,
    resolve_split,
)
from src.utils import ensure_dir, save_json

REGISTRY = config.ADMIN_DIR / "experiment_registry.csv"

REGISTRY_COLUMNS = [
    "run_id", "date", "phase", "stage", "backbone", "head", "preprocessing",
    "augmentation", "class_weight", "img_size", "batch_size", "epochs_planned",
    "epochs_run", "optimizer", "lr", "freeze_mode", "normalization", "es_patience",
    "val_accuracy", "val_macro_f1", "val_weighted_f1", "test_accuracy",
    "test_macro_f1", "test_weighted_f1", "train_time_s", "inference_ms_per_image",
    "checkpoint_path", "notes",
]


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


class ResourceMonitor(tf.keras.callbacks.Callback):
    """Sample RSS/CPU once per second and abort on unsafe memory pressure."""

    def __init__(self, interval: float = 1.0, max_rss_frac: float = 0.85) -> None:
        super().__init__()
        self.interval = interval
        self.max_rss_frac = max_rss_frac
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.total_ram = None
        self.aborted_reason: str | None = None

    def on_train_begin(self, logs: dict | None = None) -> None:
        import psutil

        self._proc = psutil.Process(os.getpid())
        self.total_ram = psutil.virtual_memory().total
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        import psutil

        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                self.samples.append({
                    "t": time.time(),
                    "rss_mb": rss / 1024 ** 2,
                    "cpu_percent": psutil.cpu_percent(interval=None),
                })
            except Exception:  # noqa: BLE001 - monitoring must never crash training
                pass
            self._stop.wait(self.interval)

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if not self.samples or self.total_ram is None:
            return
        rss = self.samples[-1]["rss_mb"] * 1024 ** 2
        if rss > self.max_rss_frac * self.total_ram:
            self.aborted_reason = (
                f"RSS {rss / 1024 ** 3:.2f} GB exceeded "
                f"{self.max_rss_frac:.0%} of {self.total_ram / 1024 ** 3:.2f} GB"
            )
            self.model.stop_training = True

    def on_train_end(self, logs: dict | None = None) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2 * self.interval)

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {}
        rss = np.array([s["rss_mb"] for s in self.samples])
        cpu = np.array([s["cpu_percent"] for s in self.samples])
        return {
            "samples": len(self.samples),
            "rss_mb_mean": round(float(rss.mean()), 1),
            "rss_mb_max": round(float(rss.max()), 1),
            "cpu_percent_mean": round(float(cpu.mean()), 1),
            "cpu_percent_max": round(float(cpu.max()), 1),
            "total_ram_gb": round(float(self.total_ram) / 1024 ** 3, 2) if self.total_ram else None,
            "aborted_reason": self.aborted_reason,
        }


class MacroF1Callback(tf.keras.callbacks.Callback):
    """Log validation macro-F1 so EarlyStopping can monitor the primary metric."""

    def __init__(self, val_ds: tf.data.Dataset, head: str, y_quality: np.ndarray,
                 y_vegetable: np.ndarray) -> None:
        super().__init__()
        self.val_ds = val_ds
        self.head = head
        self.y_quality = y_quality
        self.y_vegetable = y_vegetable

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        logs = logs if logs is not None else {}
        preds = self.model.predict(self.val_ds, verbose=0)

        if self.head == "single":
            q = np.argmax(preds, axis=1)
            logs["val_macro_f1"] = float(
                f1_score(self.y_quality, q, average="macro", zero_division=0))
        else:
            q = np.argmax(preds["quality"], axis=1)
            v = np.argmax(preds["vegetable"], axis=1)
            logs["val_macro_f1"] = float(
                f1_score(self.y_quality, q, average="macro", zero_division=0))
            logs["val_veg_macro_f1"] = float(
                f1_score(self.y_vegetable, v, average="macro", zero_division=0))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _registry_append(row: dict[str, Any]) -> None:
    exists = REGISTRY.exists()
    with open(REGISTRY, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in REGISTRY_COLUMNS})


def _predict(model: tf.keras.Model, ds: tf.data.Dataset, head: str):
    preds = model.predict(ds, verbose=0)
    if head == "single":
        return {"quality": np.asarray(preds)}
    return {"vegetable": np.asarray(preds["vegetable"]),
            "quality": np.asarray(preds["quality"])}


def _single_image_latency(model: tf.keras.Model, mode: str, size: int,
                          n: int = 20) -> dict[str, float]:
    """Latency for batch-1 inference, which is what the website will do."""
    from src.training.data import normalise_tf

    rng = np.random.default_rng(0)
    raw = (rng.random((n, size, size, 3)) * 255).astype("float32")
    batch = normalise_tf(tf.constant(raw), mode)

    times = []
    for i in range(n):
        x = batch[i:i + 1]
        model(x, training=False)  # warm
        t0 = time.perf_counter()
        model(x, training=False)
        times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "batch1_ms_median": round(float(np.median(times)), 2),
        "batch1_ms_p95": round(float(np.quantile(times, 0.95)), 2),
    }


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------


def run(run_id: str, smoke: bool = False) -> dict[str, Any]:
    cfg = run_configs.get(run_id)

    if cfg["kind"] == "classical":
        from src.training.run_classical import run as run_classical

        return run_classical(run_id, cfg, smoke=smoke)

    config.set_global_seed(cfg.get("seed", config.SEED))

    backbone, head = cfg["backbone"], cfg["head"]
    size, batch = cfg["size"], cfg["batch"]
    epochs = 1 if smoke else cfg["epochs"]
    limit = 64 if smoke else None
    mode = EXPECTED_NORMALIZATION[backbone]

    out_dir = ensure_dir(config.CHECKPOINT_DIR / (f"_smoke_{run_id}" if smoke else run_id))

    print(f"[{run_id}] {cfg['label']}")
    print(f"[{run_id}] backbone={backbone} head={head} size={size} batch={batch} "
          f"epochs={epochs} lr={cfg['lr']} normalize={mode}")

    # --- data -------------------------------------------------------------
    train_df = resolve_split("train")
    val_df = resolve_split("validation")
    if limit:
        train_df, val_df = train_df.head(limit), val_df.head(max(16, limit // 2))

    # Per-sample weights are needed for the multi-task head because Keras 3
    # rejects a per-output ``class_weight`` dict.  Both mechanisms apply the
    # same numerical weights, so fairness between A2/A3 and A4 is preserved.
    ds_weights: dict[str, Any] = {}
    if cfg.get("class_weight", False) and head == "multitask":
        qw_all = compute_class_weights("train")
        vdf = pd.read_csv(config.METADATA_DIR / "splits.csv")
        vt = vdf[vdf["split"] == "train"]
        vc = vt["vegetable"].value_counts()
        kv = len(config.VEGETABLE_CLASSES)
        from src.training.data import VEG_INDEX
        ds_weights = {
            "quality_weights": qw_all,
            "vegetable_weights": {VEG_INDEX[name]: float(len(vt) / (kv * n))
                                  for name, n in vc.items()},
        }

    train_ds = make_dataset(
        "train", preset=cfg["preset"], normalize=mode, size=size, batch_size=batch,
        training=True, head=head, augment=cfg["augment"], seed=cfg["seed"], limit=limit,
        **ds_weights,
    )
    val_ds = make_dataset(
        "validation", preset=cfg["preset"], normalize=mode, size=size, batch_size=batch,
        training=False, head=head, augment=False, limit=(len(val_df) if limit else None),
    )

    y_quality = val_df["quality"].to_numpy()
    y_vegetable = val_df["vegetable"].to_numpy()

    # --- class weights (training split only) ------------------------------
    class_weight: Any = None
    weights_used = {}
    if cfg.get("class_weight", False):
        qw = compute_class_weights("train")
        weights_used = {"quality": {int(k): round(v, 4) for k, v in qw.items()}}
        if head == "single":
            # Keras-native class weighting for the single-output model.
            class_weight = {int(k): v for k, v in qw.items()}
        else:
            # Weights ride on the dataset (see ds_weights above).
            weights_used["vegetable"] = {
                int(k2): round(v2, 4) for k2, v2 in ds_weights["vegetable_weights"].items()
            }
            weights_used["mechanism"] = "per-sample weights on the dataset"

    # --- model ------------------------------------------------------------
    model = build_model(backbone=backbone, head=head, input_size=size,
                        freeze=cfg.get("freeze", True), dropout=0.3)
    model = compile_model(model, learning_rate=cfg["lr"], head=head)
    model_info = describe(model)
    print(f"[{run_id}] params {model_info['total_params']:,} "
          f"(trainable {model_info['trainable_params']:,})")

    # --- callbacks --------------------------------------------------------
    monitor = ResourceMonitor()
    f1_cb = MacroF1Callback(val_ds, head, y_quality, y_vegetable)

    callbacks = [
        monitor,
        f1_cb,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "best.keras"), monitor="val_macro_f1",
            mode="max", save_best_only=True, verbose=0),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_dir / "last.keras"), save_best_only=False, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_macro_f1", mode="max", patience=cfg["es_patience"],
            min_delta=cfg["es_min_delta"], restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_macro_f1", mode="max", factor=cfg["lr_factor"],
            patience=cfg["lr_patience"], min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
    ]

    # --- train ------------------------------------------------------------
    t0 = time.perf_counter()
    # ``class_weight`` must be None when weights already ride on the dataset,
    # otherwise Keras would apply them twice.
    fit_kwargs: dict[str, Any] = {}
    if head == "single" and class_weight is not None:
        fit_kwargs["class_weight"] = class_weight

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        callbacks=callbacks, verbose=2, **fit_kwargs,
    )
    train_time = time.perf_counter() - t0
    epochs_run = len(history.history.get("loss", []))
    print(f"[{run_id}] trained {epochs_run} epochs in {train_time / 60:.1f} min")

    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    save_json(history_data, out_dir / "history.json")

    # --- evaluate with the best checkpoint --------------------------------
    best_path = out_dir / "best.keras"
    model_path = best_path if best_path.exists() else out_dir / "last.keras"
    best_model = tf.keras.models.load_model(model_path)

    t_inf = time.perf_counter()
    preds = _predict(best_model, val_ds, head)
    inference_total = time.perf_counter() - t_inf
    inference_ms = inference_total / max(len(val_df), 1) * 1000

    quality_pred = np.argmax(preds["quality"], axis=1)
    quality_metrics = summarise(y_quality, quality_pred, config.QUALITY_CLASSES)

    # --- persist artefacts ------------------------------------------------
    stem = f"{run_id}{'_smoke' if smoke else ''}"
    save_predictions(y_quality, quality_pred, preds["quality"],
                     config.QUALITY_CLASSES,
                     config.METRICS_DIR / f"{stem}_predictions.csv",
                     identifiers=val_df["path"].str.split("/").str[-1].tolist()
                     if "path" in val_df else None)
    save_classification_report(y_quality, quality_pred, config.QUALITY_CLASSES,
                              config.EVAL_REPORT_DIR / f"{stem}_classification_report.txt",
                              title=f"{run_id} quality classification (validation)")
    plot_confusion_matrix(y_quality, quality_pred, config.QUALITY_CLASSES,
                          config.CONFUSION_DIR / f"{stem}_quality.png",
                          title=f"{run_id} — quality (validation, n={len(val_df)})")
    per_class_frame(quality_metrics).to_csv(
        config.METRICS_DIR / f"{stem}_per_class.csv", index=False)

    result: dict[str, Any] = {
        "run_id": run_id,
        "label": cfg["label"],
        "stage": cfg["stage"],
        "config": cfg,
        "model": model_info,
        "normalization": mode,
        "class_weight_values": weights_used,
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "epochs_planned": epochs,
        "epochs_run": epochs_run,
        "train_time_s": round(train_time, 2),
        "inference_ms_per_image_batched": round(inference_ms, 2),
        "latency": _single_image_latency(best_model, mode, size),
        "resources": monitor.summary(),
        "quality": quality_metrics,
        "checkpoints": {"best": str(model_path),
                        "last": str(out_dir / "last.keras")},
        "history_path": str(out_dir / "history.json"),
    }

    if head == "multitask":
        veg_pred = np.argmax(preds["vegetable"], axis=1)
        veg_metrics = summarise(y_vegetable, veg_pred, config.VEGETABLE_CLASSES)
        result["vegetable"] = veg_metrics

        save_predictions(y_vegetable, veg_pred, preds["vegetable"],
                         config.VEGETABLE_CLASSES,
                         config.METRICS_DIR / f"{stem}_vegetable_predictions.csv")
        save_classification_report(
            y_vegetable, veg_pred, config.VEGETABLE_CLASSES,
            config.EVAL_REPORT_DIR / f"{stem}_vegetable_classification_report.txt",
            title=f"{run_id} vegetable identification (validation)")
        plot_confusion_matrix(
            y_vegetable, veg_pred, config.VEGETABLE_CLASSES,
            config.CONFUSION_DIR / f"{stem}_vegetable.png",
            title=f"{run_id} — vegetable (validation, n={len(val_df)})")
        per_class_frame(veg_metrics).to_csv(
            config.METRICS_DIR / f"{stem}_vegetable_per_class.csv", index=False)

        # Joint agreement: both heads correct on the same sample.
        both = ((quality_pred == y_quality) & (veg_pred == y_vegetable))
        result["joint_both_correct_rate"] = float(both.mean())

    save_json(result, config.METRICS_DIR / f"{stem}_validation.json")

    if not smoke:
        _registry_append({
            "run_id": run_id,
            "date": time.strftime("%Y-%m-%d"),
            "phase": "2",
            "stage": cfg["stage"],
            "backbone": backbone,
            "head": head,
            "preprocessing": cfg["preset"],
            "augmentation": cfg["augment"],
            "class_weight": cfg.get("class_weight", False),
            "img_size": size,
            "batch_size": batch,
            "epochs_planned": cfg["epochs"],
            "epochs_run": epochs_run,
            "optimizer": "Adam",
            "lr": cfg["lr"],
            "freeze_mode": "frozen" if cfg.get("freeze", True) else "full",
            "normalization": mode,
            "es_patience": cfg["es_patience"],
            "val_accuracy": round(quality_metrics["accuracy"], 4),
            "val_macro_f1": round(quality_metrics["macro_f1"], 4),
            "val_weighted_f1": round(quality_metrics["weighted_f1"], 4),
            "test_accuracy": "", "test_macro_f1": "", "test_weighted_f1": "",
            "train_time_s": round(train_time, 1),
            "inference_ms_per_image": round(inference_ms, 2),
            "checkpoint_path": str(model_path),
            "notes": "validation metrics only; test split untouched",
        })

    del best_model, model
    tf.keras.backend.clear_session()
    gc.collect()
    print(f"[{run_id}] quality macro-F1 = {quality_metrics['macro_f1']:.4f} "
          f"(acc {quality_metrics['accuracy']:.4f})")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Run one Phase 2 experiment")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="1 epoch on 64 images to validate the plumbing; writes nothing to the registry")
    args = ap.parse_args()
    run(args.run_id, smoke=args.smoke)


if __name__ == "__main__":
    main()
