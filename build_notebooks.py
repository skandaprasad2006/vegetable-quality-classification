"""Generate the Phase 1 notebooks as valid ``.ipynb`` files.

Notebooks are generated rather than hand-edited JSON so that their structure is
reviewable in version control and reproducible.  They are then executed with
``jupyter nbconvert --execute`` so every stored output is a real execution
result.

Why figures are saved to disk and displayed with ``IPython.display.Image``
instead of being drawn inline: ``src/visualize.py`` selects the non-interactive
Agg backend so the same code works in scripts, CI and notebooks.  Saving to
``evaluation/graphs/`` and embedding the PNG gives identical output everywhere
and guarantees the report images and notebook images are literally the same
files.

Usage:  python tools/build_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = ROOT / "notebooks"
GRAPHS = "evaluation/graphs"

BOOTSTRAP = """\
import sys, json, warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

# Locate the project root by walking upward until config.py is found.  This is
# robust to the kernel's working directory being either the project root or the
# notebooks/ directory, which differs between nbconvert, JupyterLab and VS Code.
_here = Path.cwd().resolve()
ROOT = _here
for _cand in (_here, *_here.parents):
    if (_cand / "config.py").exists():
        ROOT = _cand
        break
else:
    raise RuntimeError(f"could not locate project root from {_here}")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
import config

GRAPHS = "evaluation/graphs"
from IPython.display import Image, display, Markdown
from src.utils import load_json, fmt_bytes, imread_unicode

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 40)
print("kernel cwd :", _here)
print("project root:", ROOT)
print("python      :", sys.version.split()[0])
"""


# ---------------------------------------------------------------------------
# 01 — Dataset audit
# ---------------------------------------------------------------------------


def notebook_audit() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 01 — Dataset Audit and Integrity Verification

**Project:** Vegetable Quality and Freshness Detection Using Image Processing and Deep Learning

This notebook verifies the dataset before any modelling. It reads the artefacts
produced by `src/data/audit.py` — it does not re-invent them — so the numbers
here are the same numbers the tests assert on.

Audit answers four questions:

1. Does every file actually decode?
2. What are the labels, and are they usable?
3. Are there duplicates that would leak across a train/test boundary?
4. What are the image properties the pipeline must handle?
"""))
    c.append(nbf.v4.new_markdown_cell("## 1. Provenance"))
    c.append(nbf.v4.new_code_cell(BOOTSTRAP))
    c.append(nbf.v4.new_code_cell("""\
summary = load_json(config.METADATA_DIR / "audit_summary.json")
source = summary["source"]
display(pd.DataFrame(
    {"field": list(source), "value": [str(v) for v in source.values()]}
))
print("audit wall time:", summary["audit_duration_s"] , "s")"""))
    c.append(nbf.v4.new_markdown_cell("""\
The local corpus is the **Hugging Face redistribution**: 6,150 images already
resized to 256×256. The Mendeley original and the *Data in Brief* paper both
describe **6,850** images, so 700 are absent here. That gap is recorded as a
dataset limitation (see `Admin/decisions.md`, D-004) rather than silently ignored,
and it is the reason no benchmark numbers are claimed against published results.
"""))

    c.append(nbf.v4.new_markdown_cell("## 2. Integrity"))
    c.append(nbf.v4.new_code_cell("""\
inventory = pd.read_csv(config.METADATA_DIR / "image_inventory.csv")
counts, storage, geom = summary["counts"], summary["storage"], summary["geometry"]

integrity = pd.DataFrame({
    "check": ["files enumerated", "decoded successfully", "failed to decode",
              "total size", "mean file size", "distinct dimensions",
              "colour channels", "Pillow modes", "extensions"],
    "value": [
        counts["total_files"], counts["readable"], counts["unreadable"],
        fmt_bytes(storage["total_bytes"]),
        f"{storage['mean_bytes_per_image']:.0f} B",
        geom["unique_dimensions"], geom["channels"], geom["pillow_modes"],
        geom["file_extensions"],
    ],
})
display(integrity)

assert counts["unreadable"] == 0, "corrupt files present"
assert counts["total_files"] == 6150
print("PASS - all files decode; corpus is 6,150 images /", fmt_bytes(storage["total_bytes"]))"""))
    c.append(nbf.v4.new_markdown_cell("""\
Every file was opened twice: Pillow `verify()` (which catches truncated JPEG
data that OpenCV tolerates) and a full OpenCV decode. Zero failures means the
audit's other numbers are computed over the whole corpus.
"""))

    c.append(nbf.v4.new_markdown_cell("## 3. Label structure"))
    c.append(nbf.v4.new_code_cell("""\
cells = pd.DataFrame(counts["cells"])
grid = cells.pivot(index="vegetable", columns="quality", values="count")
grid = grid.reindex(index=list(config.VEGETABLE_CLASSES), columns=list(config.QUALITY_CLASSES))
display(grid.fillna(0).astype(int))

flat = cells["count"].to_numpy()
nonempty = flat[flat > 0]
print("vegetables:", len(config.VEGETABLE_CLASSES), " quality classes:", len(config.QUALITY_CLASSES))
print("populated cells:", counts["populated_cells"], "of 20")
print("empty cells:", counts["empty_cells"])
print(f"imbalance, largest/smallest non-empty cell: {nonempty.max()/nonempty.min():.1f}x")
print(f"largest: {counts['largest_cell']}")
print(f"smallest non-empty: {nonempty.min()} images")"""))
    c.append(nbf.v4.new_markdown_cell("""\
Two structural facts drive every later decision:

* **`Tomato/Dried` contains zero images.** The directory exists but is empty, so
  only 19 of the 20 (vegetable, quality) cells are populated. A 20-way joint
  classifier is therefore impossible without inventing data.
* The corpus is **imbalanced by up to 45×** between the largest cell
  (`Tomato/Old`, 1,234) and the smallest (`Tomato/Damaged`, 27). Accuracy alone
  would be a misleading metric, which is why macro F1 is the headline metric
  throughout this project.

Both facts are pinned by tests in `tests/test_data.py` so they cannot drift
silently.
"""))

    c.append(nbf.v4.new_markdown_cell("## 4. Duplicates and leakage risk"))
    c.append(nbf.v4.new_code_cell("""\
dups = load_json(config.METADATA_DIR / "duplicate_groups.json")
d = summary["duplicates"]
display(pd.DataFrame({
    "kind": ["byte-identical (SHA-256)", "near-duplicate (dHash)", "groups spanning different labels"],
    "groups": [d["exact_group_count"], d["near_group_count"], d["label_conflicting_group_count"]],
    "images": [d["exact_image_count"], d["near_image_count"], d["label_conflicting_image_count"]],
}))

top = pd.DataFrame([{
    "size": g["size"],
    "max_hamming": g["max_hamming_within"],
    "labels": " | ".join(f"{a}/{b}" for a, b in g["labels"]),
    "example": g["members"][0],
} for g in dups["near_duplicate_groups"][:8]])
display(top)"""))
    c.append(nbf.v4.new_markdown_cell("""\
There are **no byte-identical duplicates**, but there are many perceptual
near-duplicates: bursts of photographs of the same physical vegetable taken
seconds apart. This is the single biggest threat to a valid evaluation.

If such a burst is split randomly, the same object appears in both train and
test and the test score measures memorisation. `src/data/splits.py` therefore
treats every near-duplicate cluster as **indivisible**.

The label-conflicting clusters are a separate finding: the same-looking image
carries two different quality labels. Some are genuinely borderline (an "Old"
vegetable that is also "Dried"); others are label noise. They are kept — deleting
them would silently reshape the dataset — but they set a ceiling on achievable
accuracy and are discussed in `reports/limitations.md`.
"""))

    c.append(nbf.v4.new_markdown_cell("## 5. Capture-condition variation"))
    c.append(nbf.v4.new_code_cell("""\
stats = pd.DataFrame(summary["image_statistics"]).T
display(stats[["mean", "std", "p05", "median", "p95"]].round(2))

print("images with sharpness below the 5th percentile:",
      int((inventory["sharpness"] < inventory["sharpness"].quantile(0.05)).sum()))
print("images with >1% clipped highlights:",
      int((inventory["clipped_white_frac"] > 0.01).sum()))
print("images with >1% clipped shadows:",
      int((inventory["clipped_black_frac"] > 0.01).sum()))"""))
    c.append(nbf.v4.new_markdown_cell("""\
Brightness is tightly controlled (mean ≈179 of 255), consistent with studio
photography on white backgrounds. Sharpness varies by two orders of magnitude,
so motion blur and focus are genuine sources of variation — a model that keys on
high-frequency detail will be unstable on the blurred tail. This motivates
measuring a sharpness-retention cost for every denoising operation in notebook 03.
"""))

    c.append(nbf.v4.new_markdown_cell("## 6. Gate 1 verdict"))
    c.append(nbf.v4.new_code_cell("""\
checks = {
    "all images decodable": counts["unreadable"] == 0,
    "label structure verified (19/20 cells)": counts["populated_cells"] == 19,
    "single geometry (256x256)": len(geom["unique_dimensions"]) == 1,
    "duplicate clusters identified": d["near_group_count"] > 0,
    "class labels usable (>=5 classes, >=4 vegetables)":
        inventory["quality"].nunique() == 5 and inventory["vegetable"].nunique() == 4,
}
display(pd.DataFrame({"gate check": list(checks), "passed": list(checks.values())}))
assert all(checks.values())
print("GATE 1 (dataset integrity) SATISFIED")"""))
    return nb


# ---------------------------------------------------------------------------
# 02 — Exploratory analysis
# ---------------------------------------------------------------------------


def notebook_eda() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 02 — Exploratory Analysis

Class balance, sample appearance, capture-condition variation and split
composition. All figures are written to `evaluation/graphs/` and embedded here,
so the report and the notebook show the same images.
"""))
    c.append(nbf.v4.new_code_cell(BOOTSTRAP))
    c.append(nbf.v4.new_code_cell("""\
from src.visualize import (
    plot_class_distribution, plot_resolution_and_size,
    plot_condition_distributions, plot_sample_grid, plot_split_composition,
)

inventory = pd.read_csv(config.METADATA_DIR / "image_inventory.csv")
split_summary = load_json(config.METADATA_DIR / "split_summary.json")
print(len(inventory), "images loaded")"""))

    c.append(nbf.v4.new_markdown_cell("## 1. Class distribution"))
    c.append(nbf.v4.new_code_cell("""\
fig = plot_class_distribution(inventory, out=ROOT / f"{GRAPHS}/eda_class_distribution.png")
display(Image(filename=str(ROOT / f"{GRAPHS}/eda_class_distribution.png")))

q = inventory["quality"].value_counts().reindex(config.QUALITY_CLASSES)
display(pd.DataFrame({
    "count": q, "share": (q / len(inventory) * 100).round(2),
    "vs largest": (q.max() / q).round(2),
}))"""))
    c.append(nbf.v4.new_markdown_cell("""\
`Old` (2,044) and `Ripe` (1,787) together account for 62% of the corpus, while
`Damaged` (317) is 5%. A majority-class predictor that always answered "Old"
would score ~33% accuracy, which is the floor any real model must clear — and
the reason every table in this project leads with macro F1 rather than accuracy.
"""))

    c.append(nbf.v4.new_markdown_cell("## 2. Resolution and file size"))
    c.append(nbf.v4.new_code_cell("""\
plot_resolution_and_size(inventory, out=ROOT / f"{GRAPHS}/eda_resolution_size.png")
display(Image(filename=str(ROOT / f"{GRAPHS}/eda_resolution_size.png")))
print(inventory["file_size_bytes"].describe().round(1))"""))

    c.append(nbf.v4.new_markdown_cell("## 3. Lighting, contrast and focus per class"))
    c.append(nbf.v4.new_code_cell("""\
plot_condition_distributions(inventory, out=ROOT / f"{GRAPHS}/eda_conditions.png")
display(Image(filename=str(ROOT / f"{GRAPHS}/eda_conditions.png")))

agg = inventory.groupby("quality")[["brightness", "contrast", "sharpness", "saturation"]].median()
display(agg.reindex(config.QUALITY_CLASSES).round(1))"""))
    c.append(nbf.v4.new_markdown_cell("""\
Sharpness is markedly lower for `Dried` — dried specimens are wrinkled and
low-contrast, so the Laplacian responds weakly. That is a property of the object,
not a defect, but it means texture features carry real signal here (confirmed by
the feature screening in notebook 03).
"""))

    c.append(nbf.v4.new_markdown_cell("## 4. Sample images"))
    c.append(nbf.v4.new_code_cell("""\
plot_sample_grid(inventory, n_per_class=4, out=ROOT / f"{GRAPHS}/eda_samples.png")
display(Image(filename=str(ROOT / f"{GRAPHS}/eda_samples.png")))"""))
    c.append(nbf.v4.new_markdown_cell("""\
Visual inspection confirms the labels are semantically usable: `Ripe` specimens
are saturated and glossy, `Dried` are wrinkled and matte, `Old` show dulling and
shrivelling, `Unripe` are pale/green, `Damaged` show cuts and lesions.

It also shows why the task is hard at the boundary: `Old` and `Dried` grade into
each other, which is exactly the confusion the Phase 2 error analysis must
quantify.
"""))

    c.append(nbf.v4.new_markdown_cell("## 5. Split composition"))
    c.append(nbf.v4.new_code_cell("""\
plot_split_composition(split_summary, out=ROOT / f"{GRAPHS}/eda_split_composition.png")
display(Image(filename=str(ROOT / f"{GRAPHS}/eda_split_composition.png")))

checks = split_summary["checks"]
display(pd.DataFrame({
    "split": list(config.SPLITS),
    "images": [checks["counts_per_split"][s] for s in config.SPLITS],
    "fraction": [checks["fractions"][s] for s in config.SPLITS],
    "requested": [config.SPLIT_RATIOS[s] for s in config.SPLITS],
}))
print("groups spanning splits:", checks["groups_spanning_splits"])
print("sha256 spanning splits:", checks["sha256_spanning_splits"])
print("duplicate groups in corpus:", checks["groups"])"""))
    c.append(nbf.v4.new_markdown_cell("""\
The split is stratified on the joint (vegetable, quality) cell *and* grouped by
duplicate cluster. Every split contains all four vegetables and all five quality
classes, so per-class metrics on validation and test are meaningful even for the
27-image `Tomato/Damaged` cell — although anything computed on 27 images carries
a huge confidence interval, which the results report states explicitly.
"""))
    return nb


# ---------------------------------------------------------------------------
# 03 — Preprocessing
# ---------------------------------------------------------------------------


def notebook_preprocessing() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    c = nb.cells
    c.append(nbf.v4.new_markdown_cell("""\
# 03 — Image Preprocessing

Both a demonstration of the OpenCV pipeline and the evidence for which operations
are justified. Numerical results come from `src/preprocessing/evaluate.py`, which
measures 190 training images across nine operations.
"""))
    c.append(nbf.v4.new_code_cell(BOOTSTRAP))
    c.append(nbf.v4.new_code_cell("""\
from src.preprocessing.pipeline import (
    PRESETS, get_preset, preprocess_with_stages, display_stages,
    image_quality_metrics, segment_vegetable, bgr_to_rgb, resize,
)
from src.preprocessing.augment import augment, augmentation_grid, AugmentConfig
from src.visualize import plot_before_after, plot_augmentation_grid

study = load_json(config.EVAL_REPORT_DIR / "preprocessing_study.json")
agg = pd.DataFrame(study["aggregate"]).set_index("operation")
inventory = pd.read_csv(config.METADATA_DIR / "image_inventory.csv")
sample = inventory[inventory["quality"] == "Ripe"].iloc[0]
raw_bgr = imread_unicode(config.RAW_DIR / sample["rel_path"])
print("example image:", sample["vegetable"], "/", sample["quality"], sample["file_name"])"""))

    c.append(nbf.v4.new_markdown_cell("## 1. What each operation actually does"))
    c.append(nbf.v4.new_code_cell("""\
display(agg[["noise_sigma", "noise_reduction_pct", "sharpness_retained_pct",
             "contrast_change_pct", "ssim_to_raw", "time_ms"]].round(3))

display(Markdown(
    "- **Noise** is a Laplacian-MAD estimate; lower is smoother.\\n"
    "- **Sharpness kept** compares Laplacian variance to the unprocessed image.\\n"
    "- **SSIM vs raw** shows how much the operation changes the image at all."
))"""))
    c.append(nbf.v4.new_markdown_cell("""\
The measured result is not the textbook default:

* **Bilateral filtering dominates.** It removes 31% of the high-frequency noise
  while retaining 94% of the sharpness measure, at 0.5 ms per image. Gaussian
  blur removes a comparable amount but destroys 76% of the detail.
* **Gaussian blur is the worst choice here** at any kernel size — it is a linear
  low-pass filter and this dataset's artefacts are impulse-like JPEG blocks.
* **CLAHE increases** the Laplacian measure by 160%. That is partly genuine
  local-contrast enhancement and partly amplified noise; the metric cannot tell
  them apart, so CLAHE's value must be settled by validation macro F1, not by
  appearance.
"""))

    c.append(nbf.v4.new_markdown_cell("## 2. Before / after"))
    c.append(nbf.v4.new_code_cell("""\
display(Image(filename=str(ROOT / f"{GRAPHS}/preprocessing_before_after.png")))"""))
    c.append(nbf.v4.new_code_cell("""\
for metric in ("noise_sigma", "sharpness", "contrast"):
    display(Image(filename=str(ROOT / f"{GRAPHS}/preprocessing_{metric}.png")))"""))

    c.append(nbf.v4.new_markdown_cell("## 3. The pipeline presets"))
    c.append(nbf.v4.new_code_cell("""\
display(pd.DataFrame([{"preset": k, **v.describe()} for k, v in PRESETS.items()]))"""))
    c.append(nbf.v4.new_markdown_cell("""\
Each preset is a named, serialisable configuration — the same config is stored
with the trained model, so serving cannot drift from training.
"""))

    c.append(nbf.v4.new_markdown_cell("## 4. Stage-by-stage visualisation"))
    c.append(nbf.v4.new_code_cell("""\
for preset in ("basic", "enhanced", "segmented"):
    _, info = preprocess_with_stages(raw_bgr, get_preset(preset))
    stages = display_stages(info["stages"])
    labels = " → ".join(name for name, _ in stages)
    panel = [img for _, img in stages]
    n = len(panel)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 2.9))
    for ax, (name, img) in zip(np.atleast_1d(axes), stages):
        ax.imshow(img if img.ndim == 3 else img, cmap=None if img.ndim == 3 else "gray")
        ax.set_title(name, fontsize=8); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle(f"preset: {preset}", y=1.03)
    out = ROOT / f"{GRAPHS}/pipeline_stages_{preset}.png"
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    display(Image(filename=str(out)))
    display(Markdown(f"`{preset}`: {labels}"))"""))

    c.append(nbf.v4.new_markdown_cell("## 5. Segmentation behaviour"))
    c.append(nbf.v4.new_code_cell("""\
seg = study["segmentation"]
display(pd.DataFrame([{
    "images sampled": seg["n"],
    "usable masks": f"{seg['usable_fraction']*100:.1f}%",
    "mean foreground fraction": round(seg["area_fraction_mean"], 3),
    "5th percentile": round(seg["area_fraction_p05"], 3),
    "95th percentile": round(seg["area_fraction_p95"], 3),
    "failures by class": str(seg["failures_by_quality"]),
}]))

mask, info = segment_vegetable(raw_bgr)
print("area fraction:", round(info["area_fraction"], 3), "| ok:", info["ok"])

import matplotlib.pyplot as plt
rgb = bgr_to_rgb(raw_bgr)
fig, axes = plt.subplots(1, 3, figsize=(9, 3))
axes[0].imshow(rgb); axes[0].set_title("input")
axes[1].imshow(mask, cmap="gray"); axes[1].set_title("foreground mask")
overlay = rgb.copy(); overlay[mask == 0] = (overlay[mask == 0] * 0.25).astype(np.uint8)
axes[2].imshow(overlay); axes[2].set_title("mask overlay")
for ax in axes: ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
out = ROOT / f"{GRAPHS}/segmentation_example.png"
fig.savefig(out, bbox_inches="tight", dpi=110); plt.close(fig)
display(Image(filename=str(out)))"""))

    c.append(nbf.v4.new_markdown_cell("## 6. Augmentation"))
    c.append(nbf.v4.new_code_cell("""\
base = resize(bgr_to_rgb(raw_bgr), config.IMAGE_SIZE, "squash")
grid = augmentation_grid(base, n=5, seed=config.SEED)
fig, axes = plt.subplots(1, len(grid), figsize=(2.4 * len(grid), 2.7))
for ax, img, label in zip(np.atleast_1d(axes), grid,
                          ["original"] + [f"aug {i}" for i in range(1, len(grid))]):
    ax.imshow(img); ax.set_title(label, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
out = ROOT / f"{GRAPHS}/augmentation_examples.png"
fig.savefig(out, bbox_inches="tight", dpi=110); plt.close(fig)
display(Image(filename=str(out)))

print("config:", AugmentConfig().describe())"""))
    c.append(nbf.v4.new_markdown_cell("""\
Augmentation is applied **only to the training split, only inside the training
pipeline**, never to validation or test, and never before splitting. The
transform is implemented in `src/preprocessing/augment.py` with a single affine
warp (rotation + zoom + translation composed into one matrix) so the image is
resampled once rather than three times.

Horizontal flip is allowed; vertical flip is not, because vegetables have a
canonical upright orientation. Hue rotation is deliberately excluded: ripeness
in this dataset is largely a hue cue, and shifting hue would effectively relabel
the image.
"""))

    c.append(nbf.v4.new_markdown_cell("## 7. Do handcrafted features carry signal?"))
    c.append(nbf.v4.new_code_cell("""\
fs = load_json(config.EVAL_REPORT_DIR / "feature_screening.json")
display(pd.DataFrame({
    "scheme": ["grouped CV (honest)", "random CV (optimistic)"],
    **{
        name: [round(fs["group_benchmark_macro_f1_grouped_cv"][name]["macro_f1_mean"], 4),
               round(fs["group_benchmark_macro_f1_random_cv"][name]["macro_f1_mean"], 4)]
        for name in fs["group_benchmark_macro_f1_grouped_cv"]
    },
}).rename(columns={"all": "all features", "colour": "colour only",
                   "texture": "texture only", "shape": "shape only"}))

print(f"near-duplicate inflation when the group structure is ignored: "
      f"{fs['near_duplicate_inflation_all_features']:+.4f} macro-F1")
print("features screened:", fs["n_features"], "images:", fs["n_images"],
      "duplicate-safe groups:", fs["n_groups"])"""))
    c.append(nbf.v4.new_code_cell("""\
import matplotlib.pyplot as plt
from src.features.handcrafted import ALL_FEATURES

screen = pd.read_csv(config.METRICS_DIR / "feature_screening.csv")
display(screen.head(12).round(4))

top = screen.nlargest(20, "rf_importance").iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 5.5))
colours = {"colour": "#4c72b0", "texture": "#dd8452", "shape": "#55a868"}
ax.barh(top["feature"], top["rf_importance"],
        color=[colours.get(g, "#999") for g in top["group"]])
ax.set_title("Top 20 handcrafted features by Random-Forest importance")
ax.set_xlabel("Gini importance")
out = ROOT / f"{GRAPHS}/feature_importance_top20.png"
fig.savefig(out, bbox_inches="tight", dpi=110); plt.close(fig)
display(Image(filename=str(out)))"""))
    c.append(nbf.v4.new_markdown_cell("""\
This is the most consequential Phase 1 result.

Colour descriptors alone reach **0.897 macro-F1** on quality classification
under grouped cross-validation; all 83 descriptors reach **0.920**. That sets the
bar the deep models must clear in Phase 2. If a CNN lands below this, the honest
conclusion is that the extra capacity is not buying anything on this corpus — not
that the CNN "works".

The gap between grouped and random CV is the measured cost of ignoring duplicate
clusters. It is roughly 2.5 macro-F1 points, which is the size of the leakage
error this project's split design exists to avoid.

Shape features are the weakest group (0.406) as expected: this dataset is a
*condition* classification problem, not a shape problem.
"""))
    return nb


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    builders = {
        "01_dataset_audit.ipynb": notebook_audit,
        "02_exploratory_analysis.ipynb": notebook_eda,
        "03_image_preprocessing.ipynb": notebook_preprocessing,
    }
    for name, builder in builders.items():
        nb = builder()
        nb.metadata.update({
            "kernelspec": {"display_name": "IPCV (Python 3.12)", "language": "python",
                           "name": "ipcv"},
            "language_info": {"name": "python", "version": "3.12.16"},
        })
        path = NOTEBOOKS / name
        nbf.write(nb, path)
        print(f"wrote {path}  ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
