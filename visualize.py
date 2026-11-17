"""Plotting helpers shared by the notebooks and the reporting scripts.

Every function returns a Matplotlib ``Figure`` and can optionally save it, so a
notebook can display it inline and a script can write the same figure to
``evaluation/graphs/`` without duplicating code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

FIG_DPI = 110
plt.rcParams.update({
    "figure.dpi": FIG_DPI,
    "savefig.dpi": FIG_DPI,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

QUALITY_COLOURS = {
    "Damaged": "#d62728",
    "Dried": "#8c564b",
    "Old": "#ff7f0e",
    "Ripe": "#2ca02c",
    "Unripe": "#17becf",
}


def _finish(fig: plt.Figure, out: Path | None) -> plt.Figure:
    fig.tight_layout()
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Dataset composition
# ---------------------------------------------------------------------------


def plot_class_distribution(inventory: pd.DataFrame, out: Path | None = None) -> plt.Figure:
    """Quality counts, vegetable counts, and the joint count matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    q = inventory["quality"].value_counts().reindex(config.QUALITY_CLASSES, fill_value=0)
    axes[0].bar(q.index, q.values, color=[QUALITY_COLOURS[k] for k in q.index])
    for i, v in enumerate(q.values):
        axes[0].text(i, v + max(q.values) * 0.02, str(v), ha="center", fontsize=8)
    axes[0].set_title("Images per quality class")
    axes[0].set_ylabel("images")
    axes[0].tick_params(axis="x", rotation=20)

    v = inventory["vegetable"].value_counts().reindex(config.VEGETABLE_CLASSES, fill_value=0)
    axes[1].barh(v.index[::-1], v.values[::-1], color="#4c72b0")
    for i, val in enumerate(v.values[::-1]):
        axes[1].text(val + 20, i, str(val), va="center", fontsize=8)
    axes[1].set_title("Images per vegetable")
    axes[1].set_xlabel("images")

    # ``unstack`` leaves NaN for combinations that never occur (Tomato/Dried), and
    # ``reindex(fill_value=0)`` does not fill values that are already NaN, so the
    # explicit ``fillna(0)`` is required before the counts can be rendered.
    cells = (
        inventory.groupby(["vegetable", "quality"]).size()
        .unstack("quality")
        .reindex(index=config.VEGETABLE_CLASSES, columns=config.QUALITY_CLASSES, fill_value=0)
        .fillna(0)
    )
    im = axes[2].imshow(cells.values, cmap="YlOrRd", aspect="auto")
    axes[2].set_xticks(range(len(cells.columns)), cells.columns, rotation=30, ha="right")
    axes[2].set_yticks(range(len(cells.index)), cells.index)
    for i in range(cells.shape[0]):
        for j in range(cells.shape[1]):
            val = int(cells.values[i, j])
            axes[2].text(j, i, str(val), ha="center", va="center", fontsize=8,
                         color="white" if val > cells.values.max() * 0.55 else "black")
    axes[2].set_title("(vegetable, quality) cell counts\n0 = empty cell, not a loading bug")
    axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    return _finish(fig, out)


def plot_resolution_and_size(inventory: pd.DataFrame, out: Path | None = None) -> plt.Figure:
    """Resolution (degenerate here) and file-size distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))

    dims = (inventory["width"].astype(str) + "×" + inventory["height"].astype(str))
    counts = dims.value_counts()
    axes[0].bar(counts.index, counts.values, color="#55a868")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Image dimensions")
    axes[0].set_ylabel("images")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].hist(inventory["file_size_bytes"] / 1024, bins=50, color="#4c72b0")
    axes[1].set_title("File size distribution")
    axes[1].set_xlabel("KiB")
    axes[1].set_ylabel("images")
    return _finish(fig, out)


def plot_condition_distributions(inventory: pd.DataFrame, out: Path | None = None) -> plt.Figure:
    """Capture-condition proxies split by quality class."""
    metrics = ["brightness", "contrast", "sharpness", "saturation"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.6))
    for ax, metric in zip(axes, metrics):
        series = [inventory.loc[inventory["quality"] == q, metric].to_numpy()
                  for q in config.QUALITY_CLASSES]
        bp = ax.boxplot(series, tick_labels=[q[:4] for q in config.QUALITY_CLASSES],
                        showfliers=False, patch_artist=True)
        for patch, q in zip(bp["boxes"], config.QUALITY_CLASSES):
            patch.set_facecolor(QUALITY_COLOURS[q])
            patch.set_alpha(0.65)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=0)
    fig.suptitle("Illumination / focus variation per quality class", y=1.02)
    return _finish(fig, out)


def plot_sample_grid(inventory: pd.DataFrame, n_per_class: int = 4,
                     out: Path | None = None, seed: int = config.SEED,
                     root: Path | None = None) -> plt.Figure:
    """One row per quality class, ``n_per_class`` random samples per row."""
    root = Path(root or config.RAW_DIR)
    from src.utils import imread_unicode

    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(len(config.QUALITY_CLASSES), n_per_class,
                             figsize=(2.1 * n_per_class, 2.25 * len(config.QUALITY_CLASSES)))
    for i, quality in enumerate(config.QUALITY_CLASSES):
        part = inventory[inventory["quality"] == quality]
        picks = part.sample(min(n_per_class, len(part)), random_state=int(rng.integers(1 << 31)))
        for j in range(n_per_class):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if j < len(picks):
                row = picks.iloc[j]
                img = imread_unicode(root / row["rel_path"])
                if img is not None:
                    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.set_xlabel(f"{row['vegetable']}\n{j}", fontsize=6)
            if j == 0:
                ax.set_ylabel(quality, fontsize=9, fontweight="bold")
    fig.suptitle("Sample images per quality class", y=0.995)
    return _finish(fig, out)


def plot_split_composition(split_summary: dict, out: Path | None = None) -> plt.Figure:
    """Stacked bars showing how each quality class is distributed across splits."""
    cells = split_summary["per_split_cell_counts"]
    rows = []
    for split, mapping in cells.items():
        for cell, n in mapping.items():
            veg, qual = cell.split("/")
            rows.append({"split": split, "vegetable": veg, "quality": qual, "n": n})
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, group in zip(axes, ["quality", "vegetable"]):
        pivot = df.pivot_table(index=group, columns="split", values="n", aggfunc="sum").fillna(0)
        pivot = pivot.reindex(columns=list(config.SPLITS))
        bottom = np.zeros(len(pivot))
        colours = {"train": "#4c72b0", "validation": "#dd8452", "test": "#55a868"}
        for split in config.SPLITS:
            vals = pivot[split].to_numpy()
            ax.bar(pivot.index, vals, bottom=bottom, label=split, color=colours[split])
            bottom += vals
        ax.set_title(f"Split composition by {group}")
        ax.set_ylabel("images")
        ax.legend()
        ax.tick_params(axis="x", rotation=20)
    return _finish(fig, out)


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------


def plot_stage_grid(stages: Sequence[tuple[str, np.ndarray]], out: Path | None = None,
                    title: str = "Preprocessing stages") -> plt.Figure:
    n = len(stages)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 3.0))
    if n == 1:
        axes = [axes]
    for ax, (label, img) in zip(axes, stages):
        arr = img
        if arr.ndim == 2:
            ax.imshow(arr, cmap="gray", vmin=0, vmax=255)
        elif arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 1)
            ax.imshow(arr)
        else:
            ax.imshow(arr)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(title, y=1.04)
    return _finish(fig, out)


def plot_before_after(pairs: Sequence[tuple[str, np.ndarray, np.ndarray]],
                      out: Path | None = None) -> plt.Figure:
    """Rows of ``before | after`` for each named operation."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(6.2, 2.6 * n), squeeze=False)
    for i, (label, before, after) in enumerate(pairs):
        for j, (img, tag) in enumerate(((before, "before"), (after, "after"))):
            ax = axes[i, j]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.set_title(f"{label} — {tag}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    return _finish(fig, out)


def plot_metric_comparison(df: pd.DataFrame, metric: str,
                           out: Path | None = None) -> plt.Figure:
    """Grouped bar chart of one measured metric across operations.

    ``df`` must be in long form with ``operation``, ``role`` and ``value`` columns
    (the shape produced by ``DataFrame.melt``).
    """
    pivot = df.pivot_table(index="operation", columns="role", values="value", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 3.8))
    pivot.plot(kind="bar", ax=ax, color={"before": "#999999", "after": "#4c72b0"})
    ax.set_title(f"Effect of each operation on {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, out)


def plot_augmentation_grid(images: Sequence[np.ndarray], out: Path | None = None,
                           title: str = "Augmentation examples") -> plt.Figure:
    fig, axes = plt.subplots(1, len(images), figsize=(2.3 * len(images), 2.6))
    if len(images) == 1:
        axes = [axes]
    labels = ["original"] + [f"aug {i}" for i in range(1, len(images))]
    for ax, img, label in zip(axes, images, labels):
        ax.imshow(img if img.dtype == np.uint8 else np.clip(img, 0, 1))
        ax.set_title(label, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(title, y=1.05)
    return _finish(fig, out)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def plot_feature_importance(screen: pd.DataFrame, top_n: int = 20,
                            out: Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    top = screen.nlargest(top_n, "rf_importance").iloc[::-1]
    group_colour = {"colour": "#4c72b0", "texture": "#dd8452", "shape": "#55a868"}
    axes[0].barh(top["feature"], top["rf_importance"],
                 color=[group_colour.get(g, "#999") for g in top["group"]])
    axes[0].set_title(f"Top {top_n} features by Random-Forest importance")
    axes[0].set_xlabel("Gini importance")

    top_mi = screen.nlargest(top_n, "mutual_info").iloc[::-1]
    axes[1].barh(top_mi["feature"], top_mi["mutual_info"],
                 color=[group_colour.get(g, "#999") for g in top_mi["group"]])
    axes[1].set_title(f"Top {top_n} features by mutual information")
    axes[1].set_xlabel("MI (nats)")
    return _finish(fig, out)


def plot_group_benchmark(summary: dict, out: Path | None = None) -> plt.Figure:
    bench = summary["group_benchmark_macro_f1"]
    names = list(bench)
    means = [bench[n]["macro_f1_mean"] for n in names]
    stds = [bench[n]["macro_f1_std"] for n in names]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(names, means, yerr=stds, capsize=4,
                  color=["#4c72b0", "#dd8452", "#55a868", "#8172b3"][:len(names)])
    for b, m, n in zip(bars, means, names):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.01,
                f"{m:.3f}\n({bench[n]['n_features']} feats)", ha="center", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("macro F1 (5-fold CV)")
    ax.set_title("How far each handcrafted feature group gets on quality classification")
    return _finish(fig, out)
