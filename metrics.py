"""Evaluation metrics, classification reports and confusion matrices.

Every Phase 2 result goes through this module so that CNN and classical models
are scored by literally the same code on literally the same samples.  The primary
metric is **macro F1** (the corpus is imbalanced up to 45.7×, so accuracy alone is
misleading); accuracy, weighted F1 and per-class precision/recall are reported
alongside it.

A bootstrap confidence interval is attached to every macro F1, because the
validation split holds only 929 images and ~4 in the smallest class — a bare point
estimate would overstate precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def bootstrap_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Percentile bootstrap CI for macro F1 over the evaluated samples."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    stats = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0)
    return {
        "mean": float(stats.mean()),
        "lo": float(np.quantile(stats, alpha / 2)),
        "hi": float(np.quantile(stats, 1 - alpha / 2)),
        "std": float(stats.std()),
    }


def summarise(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    labels: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Full metric bundle for one prediction set."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels) if labels is not None else list(range(len(class_names)))

    per_class = []
    for idx in labels:
        mask = y_true == idx
        support = int(mask.sum())
        per_class.append({
            "class": class_names[idx],
            "index": int(idx),
            "support": support,
            "precision": float(precision_score(y_true == idx, y_pred == idx, zero_division=0)) if support else None,
            "recall": float(recall_score(y_true == idx, y_pred == idx, zero_division=0)) if support else None,
            "f1": float(f1_score(y_true == idx, y_pred == idx, zero_division=0)) if support else None,
        })

    present = [c for c in per_class if c["support"] > 0]
    ci = bootstrap_macro_f1(y_true, y_pred)

    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_f1_ci95": ci,
        "classes_present": len(present),
        "classes_absent": [c["class"] for c in per_class if c["support"] == 0],
        "per_class": per_class,
        "class_names": list(class_names),
    }


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str = "",
) -> str:
    """Write sklearn's text report with a header; returns the text."""
    labels = list(range(len(class_names)))
    text = classification_report(
        y_true, y_pred, labels=labels, target_names=list(class_names),
        zero_division=0, digits=4,
    )
    header = [f"{title}", "=" * len(title), ""] if title else []
    body = "\n".join(header) + text
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body + "\n", encoding="utf-8")
    return body


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str = "Confusion matrix",
) -> None:
    """Counts + row-normalised annotations, written as a PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0)

    annot = np.array([
        [f"{cm[i, j]}\n{normalised[i, j] * 100:.1f}%" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ])

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    sns.heatmap(normalised, annot=annot, fmt="", cmap="Blues", cbar=True,
                xticklabels=list(class_names), yticklabels=list(class_names),
                vmin=0, vmax=1, ax=ax, annot_kws={"fontsize": 8})
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None,
    class_names: Sequence[str],
    path: Path,
    identifiers: Sequence[str] | None = None,
) -> None:
    """Per-sample predictions with full probability vectors, as CSV."""
    data: dict[str, Any] = {}
    if identifiers is not None:
        data["sample"] = list(identifiers)
    data["y_true"] = [class_names[i] for i in y_true]
    data["y_pred"] = [class_names[i] for i in y_pred]
    data["correct"] = (np.asarray(y_true) == np.asarray(y_pred)).astype(int)
    if probabilities is not None:
        for i, name in enumerate(class_names):
            data[f"p_{name}"] = probabilities[:, i]
    data["confidence"] = probabilities.max(axis=1) if probabilities is not None else np.nan

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def per_class_frame(result: dict[str, Any]) -> pd.DataFrame:
    """Flatten a :func:`summarise` result into a per-class DataFrame."""
    return pd.DataFrame(result["per_class"])[
        ["class", "index", "support", "precision", "recall", "f1"]
    ]
