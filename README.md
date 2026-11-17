# VegCheck — Vegetable Quality and Freshness Detection

**Using Image Processing and Deep Learning**

An IPCV project that classifies the visible quality condition of a vegetable from a
single photograph, with a measured, leakage-controlled evaluation and a working web
application — rather than a demo that merely runs.

> **Status:** Phase 1 complete and verified · Phase 2 P0 complete (4 experiments) ·
> Phase 3 web application complete and tested · **Phase 4 final validation complete**.
>
> The deployed model is a **Random Forest on 83 handcrafted descriptors**, chosen
> because it measured best — see *Which model is deployed* below. It predicts
> **quality only**; the application does not claim to identify the vegetable.
>
> **Official test result (one evaluation, never used for selection):
> macro F1 0.9350 · accuracy 0.9497 · weighted F1 0.9492 on 935 images.**

---

## What the application does

From one uploaded JPG/PNG the running app returns:

1. the predicted **quality class** — `Damaged`, `Dried`, `Old`, `Ripe`, `Unripe`;
2. the score for that class and **all five class scores as a bar chart**;
3. the exact **preprocessing stages** the model applies, shown side by side;
4. the **83-descriptor feature vector** that was actually fed to the model;
5. a plain-language explanation of what the result does and does not mean;
6. a **session history** of predictions, clearable at any time.

It does **not** report a vegetable identity, a freshness score, or a food-safety
judgement. Rationale and evidence are in `reports/phase3_website_report.md`.

## Quick start

```bash
cd website
../.venv/Scripts/python.exe -m streamlit run app.py
# or, if the venv is active:
streamlit run app.py
```

Then open <http://localhost:8501>.

The app reads `models/final/rf_handcrafted.joblib`. If that file is missing, the
interface still loads and explains how to produce it:

```bash
.venv/Scripts/python.exe -m src.training.train --run-id A5
```

## Dataset (verified, not assumed)

| Property | Value |
|---|---|
| Images | 6,150 JPEG, RGB |
| Dimensions | **256×256 for every file** |
| Size on disk | 125.5 MiB |
| Vegetables | Bell Pepper, Chile Pepper, New Mexico Green Chile, Tomato |
| Quality classes | Damaged, Dried, Old, Ripe, Unripe |
| Populated (vegetable × quality) cells | **19 of 20** — `Tomato/Dried` is empty |
| Licence | **CC BY 4.0** |
| Corrupt files | 0 |
| Near-duplicate clusters | 572 groups (2,182 images) at threshold 6 |
| Genuine label ambiguity | **41 clusters / 134 images (2.2%)** |

Three verified discrepancies against the published description are documented
rather than smoothed over:

* Mendeley and the paper report **6,850** images; this redistribution has **6,150**.
* Every image is already downsampled to **256×256**, capping useful input size.
* `Tomato/Dried` exists as a directory but contains **zero** images.

### Duplicate-cluster correction

Phase 1 originally reported 42 near-duplicate groups / 866 images carrying
conflicting labels. That was wrong: dHash union-find chained unrelated images, and
one component held **732 images with a diameter of 39** against a threshold of 6.
Genuine ambiguity is **41 clusters / 134 images**. The split was deliberately
**not** rebuilt (over-grouping cannot inflate scores). Full account:
`reports/limitations.md` §L4, `reports/results.md` §2, `Admin/decisions.md` D-007.

## Which model is deployed

Phase 2 ran four experiments on one split (4,286 train / 929 validation; test
untouched). Validation macro F1:

| Run | Model | Macro F1 | 95 % CI | End-to-end latency |
|---|---|---:|---|---:|
| **A5** | **Random Forest, 83 features (deployed)** | **0.9539** | [0.9345, 0.9701] | **19.7 ms** |
| A3 | ResNet50, frozen | 0.9521 | [0.9342, 0.9676] | 302.8 ms |
| A4 | MobileNetV2 multi-task | 0.9401 | [0.9216, 0.9560] | 172.8 ms |
| A2 | MobileNetV2, frozen | 0.9376 | [0.9206, 0.9525] | — |

**Deep learning produced no measurable benefit over the handcrafted baseline.**
A5 vs A3 differ by +0.0018 with a paired bootstrap CI of [−0.0207, +0.0233] and
McNemar p = 0.253 — statistically indistinguishable — while the forest is ~15×
faster end-to-end and 6× smaller. MobileNetV2 in both forms is significantly worse.

The forest was chosen on measurement, not to match the project title. Decision
record: `Admin/decisions.md` D-012. Full results:
`reports/phase2_p0_results.md`.

## Repository layout

```
Admin/          project_plan.md, phase_status.md, decisions.md, experiment_registry.csv
datasets/
  downloads/    SOURCE.md — provenance, no archive stored
  raw/          verified working copy of the dataset
  processed/    train / validation / test (materialised copies)
  interim/      cached feature matrices
  metadata/     inventory, audit summary, duplicates, splits, class map, sha256 manifest
notebooks/      01 audit · 02 EDA · 03 preprocessing  (executed, outputs embedded)
src/
  data/         audit.py, splits.py, verify_raw_copy.py
  preprocessing/ pipeline.py, augment.py, evaluate.py
  features/     handcrafted.py, screening.py
  models/       architectures.py
  training/     data.py, train.py, run_classical.py, run_configs.py, benchmark.py
  evaluation/   metrics.py, compare_p0.py, summarise_p0.py, benchmark_inference.py
  visualize.py  shared plotting helpers
models/
  checkpoints/  A2/ A3/ A4/ — best.keras, last.keras, history, training log
  final/        rf_handcrafted.joblib  ← deployed artefact
evaluation/     metrics/ graphs/ reports/ confusion_matrices/
website/
  app.py        Streamlit entry point
  pages/        dataset · model performance · about
  components/   upload, preprocessing view, results, chart
  utils/        validation, image_processor, model_loader, artefacts, history, cache
  styles/       style.css
tests/          146 tests across data, preprocessing, training, web app
reports/        methodology · results · limitations · references · phase2_p0_results · phase3_website_report
tools/          build_notebooks.py, capture_screenshots.py
config.py, requirements.txt, conftest.py
```

`Dataset/` at the root is the **pristine download** and is never modified. It is
verified byte-identical to `datasets/raw/` by `src/data/verify_raw_copy.py`.

## Setup

TensorFlow publishes **no wheels for Python 3.14/3.15**, and Python 3.12 is not
installed on this machine by default. The project therefore provisions CPython
3.12 with `uv` and builds a project-local virtual environment.

```bash
# 1. install uv, then let uv provision CPython 3.12 (user-scope, no admin)
py -3.15 -m pip install uv
py -3.15 -m uv python install 3.12

# 2. create the project-local venv from that interpreter
py -3.15 -m uv venv .venv --python 3.12 --seed

# 3. install dependencies
.venv/Scripts/python.exe -m pip install -r requirements.txt

# 4. register the notebook kernel (once, project-local)
.venv/Scripts/python.exe -m ipykernel install \
  --prefix "$(cygpath -w "$PWD/.venv")" --name ipcv --display-name "IPCV (Python 3.12)"
```

This workspace is already provisioned: `.venv/` holds Python 3.12.16 with all
dependencies installed, and the `ipcv` kernelspec is registered inside the venv
prefix. Steps 1–4 are only needed on a fresh clone or a different machine.

On Linux/macOS replace `.venv/Scripts/python.exe` with `.venv/bin/python`.

## Model artefact requirements

The web app loads exactly one artefact and refuses to serve it if its contract has
drifted:

| Requirement | Value |
|---|---|
| Path | `models/final/rf_handcrafted.joblib` |
| Contents | `{"pipeline", "features", "classes"}` |
| Pipeline | `StandardScaler` → `RandomForestClassifier(n_estimators=200)` |
| Features | the 83 names in `src.features.handcrafted.ALL_FEATURES`, **in order** |
| Classes | `config.QUALITY_CLASSES`, **in that order** |
| Input | **224×224** RGB, raw 0–255 (no ImageNet normalisation) |
| Size | ~15 MB |

If any of these disagree, startup shows a clear error instead of predicting
garbage. Verification lives in `website/utils/model_loader.py`.

## Reproducing the project

```bash
PY=.venv/Scripts/python.exe

# Phase 1 — data
$PY -m src.data.audit              # inventory, integrity, duplicate detection
$PY -m src.data.verify_raw_copy    # prove the working copy is byte-identical
$PY -m src.data.splits             # grouped stratified split + materialise
$PY -m src.features.screening      # 83 features, grouped vs random CV
$PY -m src.preprocessing.evaluate  # 9 operations measured for quality and latency

# Phase 2 — P0 experiments (each in its own process)
$PY -m src.training.train --run-id A5
$PY -m src.training.train --run-id A2
$PY -m src.training.train --run-id A4
$PY -m src.training.train --run-id A3

# Phase 2 — comparison artefacts the website reads
$PY -m src.evaluation.compare_p0
$PY -m src.evaluation.benchmark_inference

# Phase 3 — web application
cd website && ../.venv/Scripts/python.exe -m streamlit run app.py

# Phase 4 — validation (from the project root)
$PY -m src.evaluation.phase4_audit          # 8 integrity checks
$PY -m src.evaluation.test_evaluation       # OFFICIAL test pass (refuses to overwrite)
$PY -m src.evaluation.robustness            # controlled degradation study
$PY -m src.evaluation.performance_audit     # stage timings and failure paths
$PY tools/accessibility_audit.py            # contrast, keyboard, mobile
tools/make_release_manifest.py              # release manifest with hashes
```

`src/training/run_configs.py` enforces stage authorisation: only `P0` run ids are
permitted, and requesting a P1/P2 configuration raises `PermissionError`.

## Testing

```bash
.venv/Scripts/python.exe -m pytest tests/ -q     # 178 passed
```

| File | Covers |
|---|---|
| `test_data.py` | dataset composition, zero cross-split leakage, split reproducibility, processed-tree integrity |
| `test_preprocessing.py` | colour-space contracts, filters, CLAHE, normalisation, segmentation, augmentation determinism |
| `test_web_validation.py` | valid/corrupt/oversized/unsupported uploads, extension spoofing, dimension limits |
| `test_web_preprocessing.py` | output dimensions, channel order, feature count and order, finiteness, no normalisation |
| `test_web_inference.py` | model contract, class mapping, probability validation, NaN/inf rejection, real end-to-end prediction |
| `test_web_app.py` | all four pages render, upload→predict→history flow, invalid input never reaches the model |

## Reported results

| Measurement | Result |
|---|---|
| Images audited / corrupt | 6,150 / **0** |
| Byte-identical duplicates | **0** |
| Groups or hashes crossing splits | **0** |
| Split sizes | 4,286 / 929 / 935 |
| Handcrafted features, grouped CV macro F1 (quality, Phase 1) | **0.920 ± 0.033** |
| Handcrafted features, random CV (optimistic) | 0.945 |
| Near-duplicate inflation | **+0.025 macro F1** |
| Deployed model, holdout validation macro F1 (Phase 2) | **0.9539** |
| Deployed model, official test macro F1 (Phase 4) | **0.9350** [0.9137, 0.9553] |
| Deployed model, official test accuracy | **0.9497** (47 errors / 935) |
| Deployed model, end-to-end inference | **27.2 ms** median (CPU), cold start 2.06 s |
| Robustness study (61 images × 10 conditions) | 0.902 control → 0.426 downscaled |
| Tests | **178 passed** |

The Phase 1 grouped-CV figure (0.9199) and the Phase 2 holdout figure (0.9539) are
**different evaluation schemes on different populations** and are never combined.
See `reports/phase2_p0_results.md` §3.

## Image-processing pipeline

The deployed model uses: resize to 256×256 → foreground segmentation (LAB distance
from the border-estimated background) → 83 descriptors (colour moments and hue
histograms, GLCM Haralick texture, uniform-LBP, shape morphology) → standardise →
Random Forest.

CLAHE, bilateral/median filtering and ImageNet normalisation are **not** applied at
inference: they belong to experiments that were not deployed.

| Preset | Operations |
|---|---|
| `basic` | resize → ImageNet normalisation |
| `enhanced` | median 3×3 → CLAHE on LAB-L → normalisation |
| `bilateral` | bilateral 5/50/50 → CLAHE → normalisation |
| `segmented` | enhanced + LAB-distance background removal |

Operations were selected by measurement, not convention: bilateral filtering
dominates Gaussian and median on the noise/sharpness trade-off, and Gaussian blur is
the worst option here — the opposite of the usual default.

## What this project does **not** claim

* It does **not** detect internal rot. The dataset is external photographs only.
* It makes **no food-safety or edibility judgement**.
* `Ripe` and `Unripe` describe **maturity**, not "fresh" and "rotten".
* It does **not** report an ordinal freshness score — the five classes are distinct
  categories, not points on one scale.
* The app does **not** identify the vegetable: the deployed model cannot.
* Model scores are **not calibrated probabilities**; no calibration was performed.
* It does **not** reproduce published benchmark numbers — the corpus is a smaller,
  downsampled redistribution.

See `reports/limitations.md` for all documented limitations with consequences.

## Documentation

| Document | Contents |
|---|---|
| `reports/methodology.md` | how and why, environment, pipeline design |
| `reports/results.md` | every measured Phase 1 number |
| `reports/phase2_p0_results.md` | Phase 2 P0 experiments, statistics, decision |
| `reports/phase3_website_report.md` | the application: architecture, evidence, tests |
| `reports/phase4_final_validation.md` | **final research report: validation, test, limits** |
| `reports/phase4_integrity_audit.md` | 8 integrity checks; the 224 px defect |
| `reports/phase4_real_world_robustness.md` | degradation study, failure cases |
| `reports/phase4_performance_audit.md` | stage timings, cold start, failure paths |
| `reports/phase4_accessibility_audit.md` | contrast, keyboard, mobile |
| `reports/demo_checklist.md` | 10-minute demonstration script |
| `reports/phase4_viva_questions.md` | 24 viva questions with evidence |
| `reports/VegCheck_CAT2_Presentation.pptx` | 20-slide CAT-2 deck with speaker notes |
| `reports/phase4_release_manifest.csv` | every release artefact with SHA-256 |
| `reports/limitations.md` | limitations and what each implies |
| `reports/references.md` | verified sources |
| `Admin/decisions.md` | dated decisions with evidence (D-001 … D-012) |
| `Admin/phase_status.md` | live task board |

## Attribution

Dataset: Suryawanshi, Patil & Chumchu (2022), *VegNet*, **Data in Brief** 45:108657,
licensed **CC BY 4.0**. Reuse of the images requires attribution.
