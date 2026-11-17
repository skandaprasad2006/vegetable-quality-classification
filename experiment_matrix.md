# Experiment Matrix — Phase 2

Every time estimate below is derived from the **measured** per-epoch costs in
`evaluation/reports/hardware_benchmark.md` (10 physical / 16 logical cores,
CPU-only TensorFlow 2.21.0, 4,286 training images, 134 steps per epoch at
batch 32). Nothing is estimated from specifications.

**Fixed for every run** (fairness controls, per the brief):

| Control | Value |
|---|---|
| Splits | `datasets/metadata/splits.csv` — unchanged from Phase 1 |
| Primary metric | **macro F1 on validation** (test untouched until final evaluation) |
| Seed | 42 (`config.SEED`) |
| Optimiser | Adam |
| Loss | sparse categorical cross-entropy |
| Early stopping | on `val_macro_f1`, `restore_best_weights=True` |
| LR schedule | `ReduceLROnPlateau(factor=0.3, patience=2, min_lr=1e-6)` |
| Checkpointing | best + last per run, `ModelCheckpoint(save_best_only=False)` |
| Class weights | balanced, computed from the **training split only** |
| Batch size | 32 |

---

## Stage A — Primary comparison (5 runs, ~102 min worst case)

Identical data, identical preprocessing (`tf_native` resize + backbone-appropriate
normalisation), augmentation on, class weights on.

| ID | Model | Head | Input | Epochs (cap) | ES patience | LR | min/epoch | Worst-case min |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| A1 | Custom CNN (4 blocks, 0.42 M params) | quality | 224 | 20 | 5 | 1e-3 | 2.87 | **57.4** |
| A2 | MobileNetV2, frozen trunk | quality | 224 | 15 | 4 | 1e-3 | 0.81 | **12.2** |
| A3 | ResNet50, frozen trunk | quality | 224 | 15 | 4 | 1e-3 | 1.61 | **24.2** |
| A4 | MobileNetV2, multitask | veg + quality | 224 | 15 | 4 | 1e-3 | 0.57 | **8.6** |
| A5 | Handcrafted RF (83 features) | quality | — | — | — | — | <0.02 | **<1** |

A1 is deliberately given the largest epoch budget: a from-scratch CNN on ~4.3k
images converges slowly, and under-training it would make transfer learning look
better than it is.

## Stage B — Transfer-learning depth (2 runs, ~42 min)

Only for the best transfer backbone from Stage A.

| ID | Config | Epochs | LR | min/epoch | Worst-case min |
|---|---|---:|---:|---:|---:|
| B1 | Fine-tune last 30 layers | 10 | 1e-4 | 0.92 | **9.2** |
| B2 | Full fine-tune | 8 | 1e-4 | 4.12 | **33.0** |

B2 uses a low LR because full fine-tuning of an ImageNet trunk on 4.3k images at
1e-3 destroys the pretrained features.

## Stage C — Targeted ablations (5 runs, ~65 min)

Each run changes **one** factor relative to the Stage A/B winner. No combinatorial
sweep: with five binary factors a full grid would be 32 runs (~7 h) for
information the one-at-a-time design already provides.

| ID | Factor changed | Setting | min/epoch | Worst-case min |
|---|---|---|---:|---:|
| C1 | Class weighting | **off** (vs balanced) | 0.81 | **12.2** |
| C2 | Preprocessing | `enhanced` (median 3×3 + CLAHE) | ~0.91 | **13.7** |
| C3 | Preprocessing | `bilateral` + CLAHE | ~0.91 | **13.7** |
| C4 | Preprocessing | `segmented` (background removed) | ~0.91 | **13.7** |
| C5 | Augmentation | **off** | 0.81 | **12.2** |

The OpenCV presets add ≈0.10 min/epoch (718 img/s measured end-to-end).

## Totals

| Stage | Runs | Worst-case |
|---|---:|---:|
| A | 5 | 102 min |
| B | 2 | 42 min |
| C | 5 | 65 min |
| **Total** | **12** | **≈ 3 h 30 min** |

Early stopping typically removes 20–40% of the worst case, so the realistic
expectation is **2–2.5 hours**. Stage C is only run after Stage A/B, and only if
interrupted mid-way does the ordering matter — A1–A4 complete first and are
sufficient for a valid primary comparison.

## Storage

| Artefact | Estimate |
|---|---|
| MobileNetV2 checkpoints (best + last, 11 runs) | ~220 MB |
| ResNet50 checkpoints (best + last, 1 run) | ~182 MB |
| Custom CNN checkpoints | ~3 MB |
| Training histories + configs + metrics | < 10 MB |
| **Total** | **< 0.5 GB** |

Well inside the 10 GB budget; dataset + splits already occupy ~0.4 GB.

## What is deliberately excluded

* **No 160 px variant.** ResNet50@160 is 0.91 min/epoch (vs 1.61 at 224), but
  changing input size across models breaks the "identical data" fairness rule.
  It is recorded as an option, not run.
* **No hyperparameter search.** Learning rates are the standard transfer-learning
  defaults, not tuned. Tuning on validation and then reporting validation would
  overstate performance; tuning on test is forbidden.
* **No full factorial ablation.** Justified above.
* **No test-set usage.** The test split is evaluated exactly once, on the final
  selected model, at the end of Stage C.
