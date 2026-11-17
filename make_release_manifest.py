"""Build ``reports/phase4_release_manifest.csv``.

Lists every artefact that constitutes the release with its SHA-256, size, purpose
and status. The frozen Phase 1 artefacts are included so the manifest itself
doubles as an integrity record.

Run: ``python tools/make_release_manifest.py``
"""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402

OUT = ROOT / "reports" / "phase4_release_manifest.csv"

# (path, purpose, status)
ENTRIES: list[tuple[str, str, str]] = [
    # --- entry points -------------------------------------------------------
    ("config.py", "Central configuration: paths, label order, seed", "release"),
    ("requirements.txt", "Pinned dependency set", "release"),
    ("README.md", "Project documentation and run instructions", "release"),

    # --- deployed model -----------------------------------------------------
    ("models/final/rf_handcrafted.joblib", "Deployed A5 Random Forest (200 trees, 83 features)", "release"),

    # --- frozen Phase 1 data artefacts -------------------------------------
    ("datasets/metadata/splits.csv", "Frozen split manifest (4,286 / 929 / 935)", "frozen-phase1"),
    ("datasets/metadata/split_summary.json", "Split checks and per-cell counts", "frozen-phase1"),
    ("datasets/metadata/class_map.json", "Frozen class order", "frozen-phase1"),
    ("datasets/metadata/audit_summary.json", "Dataset audit (raw duplicate figures)", "frozen-phase1"),
    ("datasets/metadata/duplicate_groups.json", "Near-duplicate cluster assignment", "frozen-phase1"),
    ("datasets/metadata/image_inventory.csv", "Per-image inventory and hashes", "frozen-phase1"),
    ("evaluation/metrics/feature_screening.csv", "Phase 1 feature screening table", "frozen-phase1"),
    ("evaluation/reports/feature_screening.json", "Phase 1 grouped-CV benchmark (0.9199)", "frozen-phase1"),
    ("evaluation/metrics/preprocessing_comparison.csv", "Phase 1 preprocessing study", "frozen-phase1"),
    ("notebooks/01_dataset_audit.ipynb", "Executed audit notebook", "frozen-phase1"),
    ("notebooks/02_exploratory_analysis.ipynb", "Executed EDA notebook", "frozen-phase1"),
    ("notebooks/03_image_preprocessing.ipynb", "Executed preprocessing notebook", "frozen-phase1"),

    # --- Phase 2 experiment artefacts --------------------------------------
    ("Admin/experiment_registry.csv", "Four executed experiments with validation and test metrics", "phase2"),
    ("evaluation/metrics/p0_comparison.json", "Paired comparison, CIs, deployment decision", "phase2"),
    ("evaluation/metrics/A5_validation.json", "A5 validation results", "phase2"),
    ("evaluation/metrics/A3_validation.json", "A3 validation results", "phase2"),
    ("evaluation/metrics/A4_validation.json", "A4 validation results", "phase2"),
    ("evaluation/metrics/A2_validation.json", "A2 validation results", "phase2"),
    ("evaluation/metrics/inference_latency.json", "End-to-end latency benchmark", "phase2"),

    # --- Phase 4 official test evaluation ----------------------------------
    ("evaluation/phase4_test/metrics.json", "OFFICIAL test metrics (macro F1 0.9350)", "phase4-authoritative"),
    ("evaluation/phase4_test/predictions.csv", "Per-image test predictions and probabilities", "phase4-authoritative"),
    ("evaluation/phase4_test/per_class.csv", "Test per-class precision/recall/F1/support", "phase4-authoritative"),
    ("evaluation/phase4_test/confusion_matrix.png", "Test confusion matrix", "phase4-authoritative"),
    ("evaluation/phase4_test/evaluation_config.json", "Model hash, class order, split checks", "phase4-authoritative"),
    ("evaluation/phase4_test/evaluation_log.txt", "Run log for the single test evaluation", "phase4-authoritative"),

    # --- Phase 4 audits -----------------------------------------------------
    ("evaluation/phase4/integrity_audit.json", "Eight integrity checks", "phase4-evidence"),
    ("evaluation/phase4_robustness/summary.json", "Robustness study results", "phase4-evidence"),
    ("evaluation/phase4_robustness/trials.csv", "610 robustness trials", "phase4-evidence"),
    ("evaluation/phase4_performance/performance.json", "Stage timings and failure paths", "phase4-evidence"),
    ("evaluation/phase4_accessibility/audit.json", "Contrast, keyboard, mobile checks", "phase4-evidence"),

    # --- application --------------------------------------------------------
    ("website/app.py", "Streamlit entry point", "release"),
    ("website/utils/model_loader.py", "Model load, contract verification, inference guards", "release"),
    ("website/utils/image_processor.py", "Training-time preprocessing reused at serving", "release"),
    ("website/utils/paths.py", "Centralised paths; 224 px input size", "release"),
    ("website/.streamlit/config.toml", "15 MB upload limit matching the app", "release"),

    # --- reports ------------------------------------------------------------
    ("reports/phase2_p0_results.md", "Phase 2 P0 results", "report"),
    ("reports/phase3_website_report.md", "Phase 3 application report", "report"),
    ("reports/phase4_integrity_audit.md", "Phase 4 integrity audit", "report"),
    ("reports/phase4_real_world_robustness.md", "Phase 4 robustness", "report"),
    ("reports/phase4_performance_audit.md", "Phase 4 performance", "report"),
    ("reports/phase4_accessibility_audit.md", "Phase 4 accessibility", "report"),
    ("reports/phase4_final_validation.md", "Final research report", "report"),
    ("reports/demo_checklist.md", "Demonstration checklist", "report"),
    ("reports/phase4_viva_questions.md", "Viva preparation", "report"),
    ("reports/VegCheck_CAT2_Presentation.pptx", "CAT-2 presentation deck", "report"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    rows = []
    missing = []
    for rel, purpose, status in ENTRIES:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        rows.append({
            "path": rel,
            "sha256": sha256(p),
            "bytes": p.stat().st_size,
            "purpose": purpose,
            "status": status,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["path", "sha256", "bytes", "purpose", "status"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {OUT} with {len(rows)} entries")
    if missing:
        print(f"WARNING missing {len(missing)}: {missing}")


if __name__ == "__main__":
    main()
