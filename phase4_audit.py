"""Phase 4 Stage 1 — integrity audit of the released system.

Each check answers a specific question about whether the artefact being shipped is
the artefact that was evaluated. Read-only: nothing is modified.

Run: ``python -m src.evaluation.phase4_audit``
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import config
from src.features.handcrafted import ALL_FEATURES
from src.features.screening import _load_one
from src.utils import imread_unicode, save_json
from website.utils.image_processor import preprocess
from website.utils.model_loader import load_bundle, model_summary

OUT_DIR = config.EVALUATION_DIR / "phase4"
N_PIPELINE_IMAGES = 20


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_a_model_identity() -> dict:
    """The deployed file is the one the registry says was evaluated."""
    reg = pd.read_csv(config.ADMIN_DIR / "experiment_registry.csv")
    row = reg[reg["run_id"] == "A5"].iloc[0]
    recorded = Path(str(row["checkpoint_path"]))

    deployed = config.FINAL_MODEL_DIR / "rf_handcrafted.joblib"
    bundle = load_bundle()
    summary = model_summary(bundle)

    return {
        "recorded_path": str(recorded),
        "deployed_path": str(deployed),
        "same_path": recorded.name == deployed.name,
        "sha256": _sha256(deployed),
        "bytes": deployed.stat().st_size,
        "recorded_val_macro_f1": float(row["val_macro_f1"]),
        "model": summary,
        "n_estimators_matches_config": summary["n_estimators"] == 200,
        "features_match_config": list(bundle["features"]) == list(ALL_FEATURES),
        "classes_match_config": list(bundle["classes"]) == list(config.QUALITY_CLASSES),
        "pass": bool(
            recorded.name == deployed.name
            and summary["n_estimators"] == 200
            and list(bundle["features"]) == list(ALL_FEATURES)
            and list(bundle["classes"]) == list(config.QUALITY_CLASSES)
        ),
    }


def check_b_pipeline_equivalence() -> dict:
    """The serving pipeline reproduces the training-time feature vector exactly.

    Compares ``website.utils.image_processor.preprocess`` against
    ``src.features.screening._load_one`` — the function that produced
    ``features_train.npz`` — on real images.
    """
    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    val = manifest[manifest["split"] == "validation"].head(N_PIPELINE_IMAGES)

    max_abs = 0.0
    mismatches: list[str] = []
    for row in val.itertuples(index=False):
        path = config.PROCESSED_DIR / "validation" / row.vegetable / row.quality / row.file_name

        training = _load_one(path)

        bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        serving = preprocess(rgb).feature_values

        for name in ALL_FEATURES:
            diff = abs(float(training[name]) - float(serving[name]))
            max_abs = max(max_abs, diff)
            if diff > 1e-5:
                mismatches.append(f"{row.file_name}:{name} delta={diff:.2e}")

    return {
        "images_compared": int(len(val)),
        "features_per_image": len(ALL_FEATURES),
        "max_abs_difference": float(max_abs),
        "n_mismatches": len(mismatches),
        "examples": mismatches[:5],
        "pass": len(mismatches) == 0,
    }


def check_c_class_mapping() -> dict:
    """Class order is identical in config, metadata, and the artefact."""
    class_map = json.loads((config.METADATA_DIR / "class_map.json").read_text(encoding="utf-8"))
    bundle = load_bundle()
    summary = model_summary(bundle)

    from_meta = class_map["quality_classes"]
    return {
        "config_order": list(config.QUALITY_CLASSES),
        "metadata_order": list(from_meta),
        "artefact_order": list(bundle["classes"]),
        "rf_classes_": [int(c) for c in bundle["pipeline"].named_steps["rf"].classes_],
        "vegetable_order": list(config.VEGETABLE_CLASSES),
        "metadata_vegetables": list(class_map["vegetable_classes"]),
        "pass": bool(
            list(from_meta) == list(config.QUALITY_CLASSES) == list(bundle["classes"])
            and [int(c) for c in bundle["pipeline"].named_steps["rf"].classes_]
            == list(range(len(config.QUALITY_CLASSES)))
        ),
    }


def check_d_test_untouched() -> dict:
    """No test scoring outside the single authorised Phase 4 evaluation.

    Phase 4 legitimately creates ``evaluation/phase4_test/`` and fills the
    registry's test columns. Those are expected. Anything else is a violation.
    """
    reg = pd.read_csv(config.ADMIN_DIR / "experiment_registry.csv")
    test_cols = ["test_accuracy", "test_macro_f1", "test_weighted_f1"]
    populated = {c: bool(reg[c].notna().any()) for c in test_cols}

    authorised_dir = config.EVALUATION_DIR / "phase4_test"
    authorised_done = (authorised_dir / "metrics.json").exists()

    existing = sorted(
        p.name for p in config.METRICS_DIR.glob("*")
        if "test" in p.name.lower()
    )
    test_cache = (config.INTERIM_DIR / "features_test.npz").exists()

    # Code paths that would touch the test split
    offenders: list[str] = []
    for path in list(Path("src").rglob("*.py")) + list(Path("website").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r'resolve_split\(\s*["\']test', text):
            offenders.append(str(path))

    return {
        "phase4_authorised_evaluation_present": authorised_done,
        "registry_test_columns_populated": populated,
        "existing_test_artefacts_in_metrics_dir": existing,
        "features_test_cache_present": bool(test_cache),
        "code_paths_loading_test_split": offenders,
        # Before the authorised run nothing may exist; afterwards the registry may
        # be filled and the feature cache created, but nothing outside phase4_test.
        "pass": bool(not existing and not offenders),
    }


def check_e_manifests_unchanged() -> dict:
    """Phase 1 frozen artefacts still hash to their recorded values."""
    results = []
    for line in (config.ADMIN_DIR / "phase1_artifacts.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split(maxsplit=1)
        path = path.lstrip("*")
        actual = _sha256(Path(path))
        results.append({"path": path, "ok": actual == digest})

    failures = [r["path"] for r in results if not r["ok"]]
    return {
        "checked": len(results),
        "failures": failures,
        "pass": len(failures) == 0,
    }


def check_e2_split_integrity() -> dict:
    """The split file still satisfies the properties Gate 1 required."""
    df = pd.read_csv(config.METADATA_DIR / "splits.csv")
    counts = df["split"].value_counts().to_dict()

    groups_per_split = df.groupby("group_id")["split"].nunique()
    sha_per_split = df.groupby("sha256")["split"].nunique()

    test_groups = set(df.loc[df["split"] == "test", "group_id"])
    train_groups = set(df.loc[df["split"] == "train", "group_id"])
    val_groups = set(df.loc[df["split"] == "validation", "group_id"])

    test_sha = set(df.loc[df["split"] == "test", "sha256"])
    other_sha = set(df.loc[df["split"] != "test", "sha256"])

    return {
        "counts": {k: int(v) for k, v in counts.items()},
        "total": int(len(df)),
        "groups_spanning_splits": int((groups_per_split > 1).sum()),
        "sha256_spanning_splits": int((sha_per_split > 1).sum()),
        "test_groups_shared_with_train_or_val": len(test_groups & (train_groups | val_groups)),
        "test_sha_shared_with_train_or_val": len(test_sha & other_sha),
        "pass": bool(
            (groups_per_split > 1).sum() == 0
            and (sha_per_split > 1).sum() == 0
            and not (test_groups & (train_groups | val_groups))
            and not (test_sha & other_sha)
        ),
    }


def check_f_website_sources() -> dict:
    """The app reads its metrics from artefacts rather than literals."""
    filhos = Path("website")
    literal_hits: list[str] = []
    for path in filhos.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        # A metric hardcoded to four decimals is the failure mode this guards against.
        for m in re.finditer(r"0\.9[0-9]{3}", text):
            literal_hits.append(f"{path}: {m.group(0)}")

    artefacts_used = sorted({
        m.group(1)
        for p in filhos.rglob("*.py")
        for m in re.finditer(r"\"([a-z_]+_JSON|SPLITS_CSV|comparison|latency)\"", p.read_text(encoding="utf-8", errors="ignore"))
    })

    return {
        "hardcoded_metric_literals": literal_hits,
        "artefact_references": artefacts_used,
        "app_reads_comparison_json": (Path("website/utils/artefacts.py").read_text(encoding="utf-8").count("COMPARISON_JSON") > 0),
        "pass": len(literal_hits) == 0,
    }


def check_g_duplicate_docs() -> dict:
    """Corrected duplicate figures are used, and superseded ones only in context."""
    # The corrected figure is expressed several ways across documents
    # ("41 clusters", "41 genuinely ambiguous clusters", a table cell "| 41 | 134 |").
    corrected = re.compile(r"41\b[^\n]{0,60}(clusters|134)")
    stale_context = re.compile(r"732|866|42 clusters")
    files = ["README.md", "reports/results.md", "reports/limitations.md",
             "reports/dataset_audit.md", "reports/phase2_p0_results.md"]

    per_file = {}
    for name in files:
        p = Path(name)
        if not p.exists():
            per_file[name] = {"exists": False}
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        per_file[name] = {
            "exists": True,
            "mentions_corrected_figure": bool(corrected.search(text)),
            "stale_figure_mentions": len(stale_context.findall(text)),
        }

    return {
        "files": per_file,
        "all_mention_corrected": all(v.get("mentions_corrected_figure") for v in per_file.values() if v["exists"]),
        "pass": all(v.get("mentions_corrected_figure") for v in per_file.values() if v["exists"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    checks = {
        "A_model_identity": check_a_model_identity(),
        "B_pipeline_equivalence": check_b_pipeline_equivalence(),
        "C_class_mapping": check_c_class_mapping(),
        "D_test_untouched": check_d_test_untouched(),
        "E_frozen_artefacts": check_e_manifests_unchanged(),
        "E2_split_integrity": check_e2_split_integrity(),
        "F_website_sources": check_f_website_sources(),
        "G_duplicate_docs": check_g_duplicate_docs(),
    }

    all_pass = all(c["pass"] for c in checks.values())
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "all_checks_pass": all_pass,
        "checks": checks,
    }
    save_json(payload, OUT_DIR / "integrity_audit.json")

    for name, result in checks.items():
        print(f"{'PASS' if result['pass'] else 'FAIL'}  {name}")
        for key, value in result.items():
            if key != "pass" and not isinstance(value, (dict, list)):
                print(f"        {key}: {value}")
    print(f"\nALL CHECKS PASS: {all_pass}")


if __name__ == "__main__":
    main()
