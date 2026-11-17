# Phase Status

**Last updated:** 2026-11-14
**Current phase:** Phase 2 — **preflight complete, awaiting approval to train**
**No training run has been launched.** `Admin/experiment_registry.csv` has a header
and zero rows, because zero experiments have been executed.

---

## Phase 1 — COMPLETE, gate passed

All Phase 1 deliverables verified. Artefacts unchanged: 12 files checksummed in
`Admin/phase1_artifacts.sha256` and re-verified during Phase 2 preflight — **0
changes**.

Key results: 6,150 images / 0 corrupt / 0 exact duplicates · splits 4,286/929/935
with 0 groups crossing · 83 handcrafted features reach **0.9199** grouped-CV macro
F1 · 73 tests pass · notebooks 01–03 executed with 0 errors.

---

## Phase 2 — Preflight (Stages 1–3) — COMPLETE

| Stage | Task | Status | Evidence |
|---|---|---|---|
| 1 | Inspect Phase 1 artefacts | ✅ Done | 12/12 checksums unchanged |
| 1 | Review handcrafted baseline (0.9199) | ✅ Done | `evaluation/reports/feature_screening.json` |
| 1 | Audit screening for CV leakage | ✅ Done | `src/features/leakage_audit.py` |
| 1 | Verify fold-local fitting | ✅ Done | 0 group violations; Δscaling +0.0000 |
| 1 | Investigate 42 label-conflicting groups | ✅ Done | `src/data/label_conflicts.py` → 732-image artefact identified |
| 1 | Confirm splits unchanged | ✅ Done | `splits.csv` hash identical to Phase 1 |
| 2 | CPU / RAM / disk measured | ✅ Done | `evaluation/reports/hardware_benchmark.md` |
| 2 | TF device configuration | ✅ Done | CPU:0 only; 0 GPU visible |
| 2 | Images/sec, s/epoch, memory, inference | ✅ Done | 7 architectures benchmarked in 90 s |
| 3 | Experiment matrix prepared | ✅ Done | `Admin/experiment_matrix.md` |
| 3 | Multi-task feasibility assessed | ✅ Done | Supported; 19-cell joint space |

### Preflight findings that changed the plan

1. **Duplicate clustering over-merged.** One component holds 732 images (11.9% of
   the corpus) with diameter 39 against a threshold of 6. The "42 conflicting
   groups / 866 images" figure is corrected to a genuine **41 clusters / 134
   images (2.2%)**. Risk R1; split retained unchanged as instructed.
2. **The from-scratch CNN is the most expensive model** (2.87 min/epoch), 3.5× a
   frozen ResNet50 — a frozen backbone only back-propagates a small head.
3. **Fine-tuning the last 30 layers costs 4.5× less** than full fine-tuning
   (0.92 vs 4.12 min/epoch).
4. **Data loading is not the bottleneck** (3,891 img/s native, 718 img/s OpenCV).

---

## Phase 2 — Training (Stages 4–5) — NOT STARTED

Blocked on approval.

| Stage | Runs | Worst case | Status |
|---|---|---:|---|
| A — primary comparison | A1–A5 | 102 min | ⛔ Blocked |
| B — transfer depth | B1–B2 | 42 min | ⛔ Blocked |
| C — targeted ablations | C1–C5 | 65 min | ⛔ Blocked |

**Bar to beat: 0.9199 macro F1** (handcrafted features, grouped CV). A5 re-runs
that baseline on the identical validation protocol so the comparison is fair.

## Phase 3 — Website

Not started. Requires a selected model from Phase 2.

---

## Open items

1. **R1 (high):** split distorted by duplicate over-merging. Threshold-2 split
   pre-specified as an optional robustness check; requires approval.
2. `tests/test_models.py` and `tests/test_prediction.py` still absent — they belong
   to the training and serving stages, and are not stubbed to look complete.
3. Peak available RAM was 4.49 GB at benchmark time; monitor during fine-tuning.
4. B2 (full fine-tune) is optional and the most memory-intensive run.

## How to resume

```bash
.venv/Scripts/python.exe -m pytest tests/ -q            # 73 tests
.venv/Scripts/python.exe -m src.features.leakage_audit  # re-verify no leakage
.venv/Scripts/python.exe -m src.data.cluster_diagnostics
.venv/Scripts/python.exe -m src.training.benchmark      # re-measure if hardware changed
```

---

# Phase 2 (P0) — COMPLETE

Executed 2026-11-14: A5, A2, A4, A3 — one process each, all exit 0.

| Run | Model | Val macro F1 | 95 % CI | End-to-end |
|---|---|---:|---|---:|
| A5 | Random Forest, 83 features | **0.9539** | [0.9345, 0.9701] | 19.7 ms |
| A3 | ResNet50 frozen | 0.9521 | [0.9342, 0.9676] | 302.8 ms |
| A4 | MobileNetV2 multi-task | 0.9401 | [0.9216, 0.9560] | 172.8 ms |
| A2 | MobileNetV2 frozen | 0.9376 | [0.9206, 0.9525] | — |

**Finding:** deep learning gave no measurable benefit over the handcrafted
baseline. A5 vs A3: Δ = +0.0018, paired CI [−0.0207, +0.0233], McNemar p = 0.253.

Registry: 4 rows. Test split untouched. Report: `reports/phase2_p0_results.md`.

**Not executed (as instructed):** A1, B1, B2, C1–C5.

# Phase 3 — COMPLETE

Streamlit application built on the deployed A5 model.

- `website/` — 4 pages, 4 components, 8 utility modules.
- 66 new tests (146 total, all passing).
- Screenshots: `reports/screenshots/`.
- Report: `reports/phase3_website_report.md`.
- Phase 1 artefacts re-verified: 12/12 checksums unchanged.

## Open items

1. **R1 (high):** split distorted by duplicate over-merging. Threshold-2 split
   remains pre-specified as an optional robustness check; requires approval.
2. Test split still not evaluated — awaiting approval under the approved protocol.
3. Seed variance unmeasured: one seed per configuration.
4. `experiment_matrix.md` P1/P2 runs remain authorised-off in `run_configs.py`.

---

# Phase 4 — COMPLETE (final validation)

| Stage | Outcome |
|---|---|
| 1 · Integrity audit | 8 checks; **1 critical defect found and fixed** (serving 256 px vs training 224 px) |
| 2 · Official test evaluation | **macro F1 0.9350** [0.9137, 0.9553], accuracy 0.9497, 935 images, run once |
| 3 · Robustness | 610 trials, control 0.902 → 0.426 downscaled; confidence unreliable out of distribution |
| 4 · Performance | 27.19 ms median end-to-end, 2.06 s cold start, 0 errors in 200 runs |
| 5 · Accessibility | 0 contrast failures (min 6.7:1), alt text added, mobile 390 px clean |
| 6 · Integration tests | **178 passed** (32 new) |
| 7 · Final report | `reports/phase4_final_validation.md` |
| 8 · Demo + viva | `reports/demo_checklist.md`, `reports/phase4_viva_questions.md`, 6 screenshots |
| 9 · Release freeze | `reports/phase4_release_manifest.csv` (48 artefacts) |

**Test split is now spent.** It was scored once; no further model comparison may use it.

**Not executed:** A1, B1, B2, C1–C5 — still unauthorised.
