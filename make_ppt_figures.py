"""Figures for the CAT-2 presentation, drawn from the saved artefacts.

Run from the project root:

    python tools/make_ppt_figures.py

Writes PNGs to ``reports/ppt/figures/``. No number is typed here: every value is
read from ``evaluation/metrics/p0_comparison.json``,
``evaluation/phase4_test/metrics.json``, ``evaluation/phase4_robustness/summary.json``
or ``evaluation/metrics/inference_latency.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "reports" / "ppt" / "figures"

INK = "#0d2b21"
EMERALD = "#0f7a4d"
MINT = "#0f9a63"
GREY = "#8aa79a"
PAPER = "#ffffff"

plt.rcParams.update({
    "font.family": "Segoe UI",
    "font.size": 12,
    "axes.edgecolor": "#cfe0d6",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
})


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color="#e6f0ea", linewidth=0.9)
    ax.set_axisbelow(True)


def read(rel: str) -> dict:
    with open(ROOT / rel, "r", encoding="utf-8") as fh:
        return json.load(fh)


def figure_model_comparison() -> Path:
    comp = read("evaluation/metrics/p0_comparison.json")
    runs = comp["runs"]
    order = sorted(runs, key=lambda k: runs[k]["quality"]["macro_f1"])
    names = [runs[k]["display_name"].replace(" (", "\n(") for k in order]
    vals = [runs[k]["quality"]["macro_f1"] for k in order]
    lo = [runs[k]["quality"]["macro_f1_ci95"]["lo"] for k in order]
    hi = [runs[k]["quality"]["macro_f1_ci95"]["hi"] for k in order]
    colours = [EMERALD if k == comp["deployment_decision"]["point_estimate_winner"] else GREY
               for k in order]

    fig, ax = plt.subplots(figsize=(9.2, 4.1), dpi=200)
    y = range(len(order))
    ax.barh(list(y), vals, color=colours, height=0.58)
    ax.errorbar(vals, list(y), xerr=[[v - l for v, l in zip(vals, lo)],
                                     [h - v for v, h in zip(vals, hi)]],
                fmt="none", ecolor=INK, elinewidth=1.3, capsize=4)
    for i, v in enumerate(vals):
        ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=11, color=INK)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlim(min(lo) - 0.012, max(hi) + 0.016)
    ax.set_xlabel("Validation macro F1 (929 images, 95 % bootstrap CI)")
    _clean(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color="#e6f0ea", linewidth=0.9)
    fig.tight_layout()
    path = OUT / "fig_models.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_robustness() -> Path:
    summ = read("evaluation/phase4_robustness/summary.json")
    conds = summ["conditions"]
    order = sorted(conds, key=lambda k: conds[k]["accuracy"], reverse=True)
    labels = [k.replace("_", " ") for k in order]
    vals = [conds[k]["accuracy"] for k in order]

    fig, ax = plt.subplots(figsize=(9.2, 4.3), dpi=200)
    bars = ax.bar(labels, vals, color=[EMERALD if k == "control" else MINT for k in order],
                  width=0.66)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", fontsize=10, color=INK)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Accuracy (61 held-out test images)")
    ax.tick_params(axis="x", rotation=28, labelsize=10)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.axhline(summ["overall_accuracy"], color=GREY, linestyle=":", linewidth=1.2)
    _clean(ax)
    fig.tight_layout()
    path = OUT / "fig_robustness.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_per_class() -> Path:
    m = read("evaluation/phase4_test/metrics.json")
    rows = m["per_class"]
    classes = [r["class"] for r in rows]
    prec = [r["precision"] for r in rows]
    rec = [r["recall"] for r in rows]
    f1 = [r["f1"] for r in rows]

    fig, ax = plt.subplots(figsize=(9.2, 4.3), dpi=200)
    x = range(len(classes))
    w = 0.26
    ax.bar([i - w for i in x], prec, width=w, color="#bfe3d0", label="Precision")
    ax.bar(list(x), rec, width=w, color=MINT, label="Recall")
    ax.bar([i + w for i in x], f1, width=w, color=EMERALD, label="F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{c}\n(n={r['support']})" for c, r in zip(classes, rows)], fontsize=10)
    ax.set_ylim(0.6, 1.03)
    ax.set_ylabel("Test-set score")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.16),
              fontsize=10)
    _clean(ax)
    fig.tight_layout()
    path = OUT / "fig_per_class.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def figure_latency() -> Path:
    lat = read("evaluation/metrics/inference_latency.json")["results"]
    labels = {"A5": "Random Forest\n(deployed)", "A4": "MobileNetV2\nmulti-task",
              "A3": "ResNet50"}
    keys = [k for k in ("A5", "A4", "A3") if k in lat]
    vals = [lat[k]["end_to_end"]["median_ms"] for k in keys]
    colours = [EMERALD if k == "A5" else GREY for k in keys]

    fig, ax = plt.subplots(figsize=(6.2, 3.6), dpi=200)
    bars = ax.bar([labels[k] for k in keys], vals, color=colours, width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 6, f"{v:.0f} ms",
                ha="center", fontsize=11, color=INK)
    ax.set_ylabel("Median end-to-end latency (ms)")
    ax.set_ylim(0, max(vals) * 1.18)
    _clean(ax)
    fig.tight_layout()
    path = OUT / "fig_latency.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (figure_model_comparison, figure_robustness, figure_per_class, figure_latency):
        print("wrote", fn().relative_to(ROOT))


if __name__ == "__main__":
    main()
