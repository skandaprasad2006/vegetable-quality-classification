# Decisions Log

Chronological record of technical decisions, the evidence behind them, and reversals.
Every entry states *why*, not just *what*.

---

## D-001 — Workspace root is the project root

**Date:** 2026-11-14
**Status:** Accepted

The master prompt specifies a `Vegetable_Quality_Detection/` root folder. The workspace
(`Skanda's IPCV proj`) already contains the downloaded `Dataset/` and several empty
placeholder folders. Creating a nested `Vegetable_Quality_Detection/` directory inside the
workspace would bury the data one level deeper and duplicate the existing tree.

**Decision:** the workspace root *is* the project root. Rule 1 ("work in the actual project
directory, do not scatter files") is satisfied by building the canonical structure directly
in the workspace. All paths in `config.py` are derived from the location of `config.py`, so
the project stays relocatable.

The pre-existing empty folders (`Notebooks/`, `application/`, `models/`, `research/`,
`results/`, `src/`) were consolidated into the canonical lowercase tree from the spec and
removed **only after** confirming each contained zero files.

---

## D-002 — Python 3.12 provisioned via uv, project-local virtual environment

**Date:** 2026-11-14
**Status:** Accepted

**Evidence gathered:**
- `python` on PATH is the Microsoft Store alias stub and does not run.
- `py -0p` lists exactly two interpreters: **3.15.2** and **3.14.3**. No 3.12 or older.
- `pip index versions tensorflow-cpu` returns `No matching distribution found` for **both**
  3.14 and 3.15. TensorFlow publishes no wheels for those interpreter versions.
- `scikit-image`, `tensorflow` and the rest of the scientific stack are likewise absent.

**Decision:** install CPython 3.12.16 with `uv` and create `.venv` inside the project.
Python 3.12 is within TensorFlow's supported range, so the master prompt's stated
preference for TensorFlow/Keras can be honoured instead of substituting a framework.

**Consequence:** the venv must be created with `.venv/Scripts/python.exe`; the system
`python` alias remains unusable and is never referenced again.
**Reversal cost:** low — deleting `.venv` and recreating it is ~4 minutes.

---

## D-003 — TensorFlow/Keras retained as the deep-learning framework (CPU execution)

**Date:** 2026-11-14
**Status:** Accepted, with documented limitation

**Evidence gathered:**
- GPU: NVIDIA RTX 4050 Laptop, 6 GB VRAM (present).
- CPU: Intel i5-13450HX, 16 logical cores. RAM 15.7 GB.
- TensorFlow on **native Windows** has had no GPU support since 2.10; the `tensorflow[and-cuda]`
  extras target Linux/WSL2 only. Enabling the RTX 4050 for TensorFlow therefore requires a
  WSL2 + CUDA + cuDNN stack (multi-GB, administrator involvement, and a different filesystem
  for the dataset).

**Decision:** use TensorFlow/Keras on **CPU**, as the prompt prefers, and size the
experiments to the measured CPU throughput rather than pretending GPU acceleration exists.
Training budgets are set from a benchmark run before the experiment matrix is fixed
(see `Admin/experiment_registry.csv` and the throughput note recorded in Phase 2).

**Rejected alternative:** switching to PyTorch, whose Windows wheels bundle CUDA, would be
faster. Rejected for now because the prompt explicitly names TensorFlow and CPU-only training
is feasible at 256×256 inputs. `/workspace` note: if measured throughput makes the required
ablation matrix infeasible, this decision is revisited **with the measurement as evidence**
and recorded as a superseding entry — not silently.

---

## D-004 — Dataset provenance: 6,150-image 256 px variant, not the 6,850-image original

**Date:** 2026-11-14
**Status:** Accepted (documented limitation)

**Evidence gathered:**
- Local copy: 6,150 `.jpg` files, 131,638,293 bytes (125.5 MiB), **every file exactly 256×256**.
- Mendeley Data record 10.17632/6nxnjbn9w6.1 (v1, 2022-09-21) and the Data in Brief article
  (10.1016/j.dib.2022.108657) both state **6,850 images**.
- Hugging Face `Project-AgML/VegNet_quality_classification` reports `6.15k rows`, column
  `image width (px) = 256`, licence `cc-by-4.0`.
- Licence confirmed as **CC BY 4.0** on both Mendeley and Hugging Face.

**Conclusion:** the local copy is the Hugging Face redistribution (6,150 images, downsampled
to 256×256), not the original full-resolution 6,850-image Mendeley release. 700 images are
absent relative to the published dataset.

**Consequences, all of which must be stated in the final report:**
1. Reported results cannot be compared numerically to papers that used the 6,850-image release.
2. Input resolution above 256 px cannot recover detail that no longer exists. Model input is
   therefore capped at ≤256 px; 224 px is the upper candidate.
3. No attempt is made to "top up" the dataset from the Mendeley original, because mixing two
   differently-preprocessed copies of the same dataset risks duplicate/near-duplicate leakage
   across splits and would make the corpus impossible to describe honestly.

---

## D-005 — Missing class: `Tomato/Dried` contains zero images

**Date:** 2026-11-14
**Status:** Accepted (documented limitation)

`4. Tomato/Dried/` exists as a directory but contains **0** files, so only **19 of the
20 possible (vegetable, quality) cells are populated**. Every class count was enumerated
directly from the filesystem (see `datasets/metadata/dataset_card.json`).

**Consequences:**
1. A naive 20-way joint classifier is not possible; the joint label space has 19 classes.
2. Quality classification is still well-posed as a global 5-class problem because all five
   quality categories occur for the other three vegetables, and "Dried" is represented by
   1,389 images overall.
3. The website must not silently rank "Tomato + Dried" as a plausible outcome; the prediction
   layer reports the model's actual distribution but the UI states which combinations are
   unsupported by training data.
4. `Tomato/Damaged` has only 27 images, the smallest cell in the dataset. Any claim about that
   cell's per-class accuracy is statistically fragile and is reported with its support count.

---

## D-006 — Pristine download preserved separately from the working copy

**Date:** 2026-11-14
**Status:** Accepted

The prompt forbids modifying the original downloaded dataset, and also asks for `datasets/raw/`.
Duplicating 125 MiB is negligible against the 10 GB budget and removes any chance of the
audit/preprocessing steps mutating the only copy.

**Decision:** `Dataset/` stays byte-identical and untouched as the pristine download.
`datasets/raw/vegnet/` holds a verified working copy. A SHA-256 manifest of the pristine tree
is written to `datasets/metadata/` and both trees are compared after copying.

---

## D-007 — Duplicate clustering over-merged; Phase 1 split retained as instructed

**Date:** 2026-11-14
**Status:** Accepted, with recorded risk

**Evidence (`src/data/cluster_diagnostics.py`):**

Union-find computes the transitive closure of "dHash Hamming ≤ threshold", so it
chains: A~B and B~C merge A and C even when they are far apart. On this corpus of
near-identical white-background studio photographs the effect is severe.

| Threshold | Largest component | Diameter | Distinct labels |
|---:|---:|---:|---:|
| 2 | 5 | 4 | 1 |
| 4 | 25 | 18 | 5 |
| **6 (Phase 1)** | **732** | **39** | **18** |
| 8 | 2,349 | 45 | 20 |

At the threshold Phase 1 used, one component holds **732 images (11.9% of the
corpus)** across all four vegetables. Its diameter (39) is 6.5× its own threshold,
which is impossible for a genuine duplicate burst.

**Consequences:**
1. The Phase 1 claim "42 groups / 866 images carry conflicting labels" is
   **corrected**: 732 of those images are the single chained component, not label
   noise. The genuine ambiguity figure is **41 clusters / 134 images (2.2%)**.
2. The 732-image component sits entirely in the training split. Rebuilding at
   threshold 2 relocates 2,813 of 6,150 images (45.8%).

**Decision:** keep `datasets/metadata/splits.csv` **unchanged**, because the
Phase 2 authorisation explicitly requires it and because changing it would
invalidate comparability with any result already described. The over-grouping is
*conservative* — it cannot inflate test scores — but it does mean 11.9% of the
corpus is absent from validation and test. This is recorded as risk R1 and every
Phase 2 conclusion is conditioned on it.

A threshold-2 split is pre-specified as an optional robustness check. It will only
be produced if approved, and it will be written as a **new** artefact, never over
the Phase 1 manifest.

---

## D-008 — Pretrained input normalisation lives in the data pipeline

**Date:** 2026-11-14
**Status:** Accepted

MobileNetV2 expects `x/127.5 − 1`; ResNet50 expects Caffe-style BGR mean
subtraction. Feeding either the wrong distribution silently degrades transfer
learning without raising an error.

**Decision:** implement both as explicit `normalize()` modes (`tf`, `caffe`) in
`src/preprocessing/pipeline.py`, mirrored in `src/training/data.py`, instead of
wrapping `keras.applications.*.preprocess_input` in a `Lambda` layer.

**Reason:** a `Lambda` layer makes the saved `.keras` file depend on
`custom_objects` at load time, which is exactly the kind of fragility that breaks
the website in Phase 3. The normalisation mode is stored in model metadata and
asserted at serving time, so training/serving drift is detectable.

---

## D-009 — Experiment budget derived from measurement, not ambition

**Date:** 2026-11-14
**Status:** Accepted

Benchmark (`evaluation/reports/hardware_benchmark.md`) measured, per epoch on
4,286 training images: custom CNN **2.87 min**, MobileNetV2 frozen **0.81**,
MobileNetV2 last-30 fine-tune **0.92**, ResNet50 frozen **1.61**, MobileNetV2
multitask **0.57**, MobileNetV2 full fine-tune **4.12**.

Two results inverted the expected ranking:
* the **from-scratch CNN is the most expensive model**, 3.5× a frozen ResNet50,
  because a frozen backbone only back-propagates a small head;
* fine-tuning the last 30 layers costs **0.92** vs **4.12** min/epoch for full
  fine-tuning — a 4.5× saving for the mode that is usually better behaved on
  small datasets.

**Decision:** a 12-run matrix (Stage A primary comparison, Stage B transfer depth,
Stage C five one-at-a-time ablations) totalling **≈3.5 h worst case**, with
priority tiers P0–P3 so a valid primary result exists after ~45 min. No full
factorial sweep: 32 runs would be needed for information the one-at-a-time design
already provides.

---

## D-010 — Two-head multi-task, not a 20-way joint classifier

**Date:** 2026-11-14
**Status:** Accepted

Every image carries exactly one vegetable and one quality label, so a shared trunk
with two heads is supported by the annotations. The joint (vegetable, quality)
label space has only **19** cells because `Tomato/Dried` is empty (D-005), so a
20-way classifier would require a class with no data.

**Decision:** two heads — 4-way vegetable (auxiliary, loss weight 0.3) and 5-way
quality (primary, weight 1.0). The auxiliary head discourages the trunk from
collapsing onto a single cue; quality remains the reported task. The weight split
is a documented judgement call, not a tuned optimum, and is not presented as one.

---

## D-011 — Handcrafted baseline re-evaluated on the identical protocol

**Date:** 2026-11-14
**Status:** Accepted

Phase 1 reported **0.9199** for 83 handcrafted descriptors under grouped 5-fold CV
*within* the training split. That is not directly comparable to a CNN scored on the
held-out validation split.

**Decision:** run A5 — a Random Forest on the same 83 features, fitted on the full
training split, scored on the same validation split with the same macro-F1 — so the
CNN-vs-classical comparison is like-for-like. The Phase 1 number is retained as the
CV-side reference; the two are reported side by side and never conflated.

---

## D-012 — Deployed model for the Phase 3 application

**Date:** 2026-11-14
**Status:** Accepted

P0 produced four models on the same validation split:

| Run | Model | Macro F1 | 95 % CI | End-to-end latency |
|---|---|---:|---|---:|
| A5 | Random Forest, 83 features | 0.9539 | [0.9345, 0.9701] | 44 ms |
| A3 | ResNet50 frozen | 0.9521 | [0.9342, 0.9676] | 235 ms |
| A4 | MobileNetV2 multi-task | 0.9401 | [0.9216, 0.9560] | 129 ms |
| A2 | MobileNetV2 frozen | 0.9376 | [0.9206, 0.9525] | — |

**Finding:** deep learning produced no measurable benefit over the handcrafted
baseline. A5 vs A3 differ by +0.0018 with a paired bootstrap CI of
[−0.0207, +0.0233] and McNemar p = 0.253 — statistically indistinguishable.
MobileNetV2 in both forms is significantly *worse* than the forest.

**Decision:** deploy **A5** in the website.

Rationale, in order:
1. Highest point estimate of the four.
2. Tied with the best CNN on a paired test, so accuracy cannot separate them.
3. 5× lower end-to-end latency (44 ms vs 235 ms), 6× smaller artefact
   (15 MB vs 94 MB), 0.02 s load vs 1.66 s, and no TensorFlow requirement at
   serving time.
4. Trains in 0.65 s, so the deployed artefact is trivially reproducible.

**Consequence:** A5 predicts quality only. The website must **not** display a
vegetable prediction — A4's vegetable head exists but A4 is not deployed, and
fabricating a vegetable label from the quality model would be dishonest.

**Rejected alternative:** deploying A3 because the project title mentions "deep
learning". Selecting a model to fit the title, when a paired test cannot separate
it and it is 5× slower, would be motivated reasoning. The DL experiments remain in
the report as measured results.
