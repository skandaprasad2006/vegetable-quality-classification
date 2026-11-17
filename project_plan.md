# Project Plan — Vegetable Quality and Freshness Detection

**Working title:** VegCheck — Intelligent Vegetable Quality Detection
**Domain:** Image Processing and Computer Vision (IPCV)
**Owner:** Skanda
**Created:** 2026-11-14

---

## 1. Problem statement

Manual grading of vegetables is slow, subjective and inconsistent between graders.
This project builds and *empirically evaluates* a two-head computer-vision system that,
from a single RGB photograph of a vegetable, predicts:

1. **which vegetable** it is (4-way), and
2. **what visible quality condition** it is in (5-way).

The deliverable is not "a model that runs". It is a documented, reproducible study:
a verified dataset, a measured preprocessing study, three compared architectures,
an ablation matrix, honest error analysis, and a working web application that serves
the *actually selected* model.

## 2. Verified starting conditions (measured, not assumed)

| Item | Value | How it was established |
|---|---|---|
| Dataset images | 6,150 `.jpg` | recursive filesystem enumeration |
| Bytes on disk | 131,638,293 (125.5 MiB) | `os.path.getsize` over all files |
| Image resolution | 256×256, **all** files | JPEG SOF marker parse |
| Vegetables | 4 | directory enumeration |
| Quality categories | 5 | directory enumeration |
| Populated (vegetable, quality) cells | **19 of 20** | `Tomato/Dried` is empty |
| Licence | CC BY 4.0 | Mendeley record + Hugging Face card |
| Upstream claimed count | 6,850 | Mendeley v1 and Data in Brief article |
| Local variant | 6,150, downsampled to 256 px | HF `Project-AgML/VegNet_quality_classification` |

The 700-image gap and the 256 px ceiling are treated as *dataset limitations*, not
something to paper over.

## 3. Phases

### Phase 1 — Data, image processing, features  ← current
1.1 Acquisition + audit (integrity, duplicates, dimensions, corrupt files)
1.2 Exploratory analysis (class distribution, imbalance, sample grids)
1.3 OpenCV preprocessing pipeline, evaluated op-by-op
1.4 Leakage-safe grouped stratified split + augmentation
1.5 Handcrafted feature extraction and feature screening

### Phase 2 — Models
2.1 Task definition (single-task vs two-head multi-task)
2.2 Custom CNN baseline
2.3 MobileNetV2, ResNet50 transfer learning
2.4 Training harness (early stopping, LR schedule, checkpoints, class weights)
2.5 Evaluation (macro F1 first, per-class metrics, confusion matrices, timings)
2.6 Ablations (class weights, preprocessing, augmentation, frozen vs fine-tuned)
2.7 Final model selection + reusable prediction function

### Phase 3 — Website
Streamlit application with Home, Detection, Image Processing, Model Information
and About pages, driven by the Phase 2 artefacts, plus automated tests.

## 4. Quality gates

* **Gate 1:** every image opens, labels verified, zero cross-split leakage,
  splits reproducible from a recorded seed, preprocessing demonstrably runs.
* **Gate 2:** all three architectures trained and evaluated on validation,
  the test set touched exactly once for the selected model, prediction pipeline
  works without Streamlit.
* **Gate 3:** the app serves real model output, rejects invalid uploads, and the
  full test suite passes.

## 5. Explicit non-goals

* Detecting **internal** rot or chemical contamination. The dataset contains only
  external photographs; no such claim will be made.
* Any food-safety or edibility verdict.
* A 20-way joint classifier — one cell has no data.
* Beating published benchmark numbers — the corpus differs (see D-004).

## 6. Resource budget

| Budget | Limit | Current |
|---|---|---|
| Storage | 10 GB | dataset 125.5 MiB + working copy + env (~1.5 GiB) |
| Compute | CPU, 16 logical cores, 15.7 GB RAM | measured in Phase 2 |

## 7. Recording conventions

* `Admin/phase_status.md` — live task board, updated at every checkpoint.
* `Admin/decisions.md` — dated, evidence-backed decisions; superseding entries are
  appended, never rewritten.
* `Admin/experiment_registry.csv` — one row per **actually executed** experiment.
* `reports/` — the long-form write-up.
