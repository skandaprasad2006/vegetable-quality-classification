"""Audit the Phase 1 feature-screening procedure for cross-validation leakage.

The headline Phase 1 number is a grouped-CV macro-F1 of 0.9199 for the 83
handcrafted descriptors.  Before any model is trained, that number must be shown
to be *honest*: every data-dependent step inside the cross-validation loop must be
fitted on the training fold only.

Four audits are run:

1. **Group integrity** — do the grouped folds actually keep every duplicate
   cluster on one side of the boundary?
2. **Scaling leakage** — a ``StandardScaler`` fitted on the full training set
   rather than per fold.  Quantified, because the size of the effect decides
   whether it matters.
3. **Selection leakage** — choosing the top-k features by importance computed on
   the full training set (the common mistake) versus nested selection inside the
   pipeline.  This is the leak the brief specifically asks about.
4. **Manifest integrity** — the split file must be byte-identical to Phase 1 and
   no test-split feature cache may exist.

The audit reports what the procedure *actually did*, including any place where a
transformation is fitted outside the loop and why that is or is not consequential.

Run: ``python -m src.features.leakage_audit``
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.features.handcrafted import ALL_FEATURES
from src.utils import load_json, save_json


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_train_features() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    cache = config.INTERIM_DIR / "features_train.npz"
    if not cache.exists():
        raise FileNotFoundError(
            f"{cache} missing; run `python -m src.features.screening` first"
        )
    data = np.load(cache, allow_pickle=True)
    X = np.nan_to_num(data["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"]
    groups = data["groups"]
    features = [str(f) for f in data["features"]]
    if features != list(ALL_FEATURES):
        raise RuntimeError("cached feature order does not match ALL_FEATURES")
    return X, y, groups, features


def _rf():
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=200, random_state=config.SEED, n_jobs=-1, class_weight="balanced_subsample",
    )


def _cv(groups: np.ndarray, cv: int = 5):
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=config.SEED)


# ---------------------------------------------------------------------------
# Audit 1 — group integrity
# ---------------------------------------------------------------------------


def audit_group_integrity(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          cv: int = 5) -> dict[str, Any]:
    splitter = _cv(groups, cv)
    folds: list[dict[str, Any]] = []
    violations = 0

    for i, (tr, te) in enumerate(splitter.split(X, y, groups)):
        g_tr, g_te = set(groups[tr].tolist()), set(groups[te].tolist())
        shared = g_tr & g_te
        violations += len(shared)
        folds.append({
            "fold": i,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "groups_train": len(g_tr),
            "groups_test": len(g_te),
            "groups_shared": len(shared),
            "classes_in_test": int(len(set(y[te].tolist()))),
        })

    return {
        "cv": cv,
        "n_groups": int(len(np.unique(groups))),
        "folds": folds,
        "group_violations": int(violations),
        "passed": violations == 0,
    }


# ---------------------------------------------------------------------------
# Audit 2 — scaling leakage
# ---------------------------------------------------------------------------


def audit_scaling_leakage(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                          cv: int = 5) -> dict[str, Any]:
    """Per-fold scaler (correct) versus a scaler fitted on all training data."""
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = _cv(groups, cv)

    correct = Pipeline([("scale", StandardScaler()), ("rf", _rf())])
    score_correct = cross_val_score(correct, X, y, cv=splitter, groups=groups,
                                    scoring="f1_macro", n_jobs=1)

    # Deliberately leaky: subtract the global training mean/std before the loop.
    leaky_scaler = StandardScaler().fit(X)
    X_leaky = leaky_scaler.transform(X)
    score_leaky = cross_val_score(_rf(), X_leaky, y, cv=splitter, groups=groups,
                                  scoring="f1_macro", n_jobs=1)

    delta = float(score_leaky.mean() - score_correct.mean())
    return {
        "correct_per_fold": [float(s) for s in score_correct],
        "correct_mean": float(score_correct.mean()),
        "correct_std": float(score_correct.std()),
        "leaky_per_fold": [float(s) for s in score_leaky],
        "leaky_mean": float(score_leaky.mean()),
        "leaky_std": float(score_leaky.std()),
        "delta": delta,
        "note": (
            "Scaling is a linear, label-independent transform; leakage here cannot "
            "invent signal, only shift the estimate. The small delta is expected and "
            "is reported to show the magnitude rather than to dismiss the issue."
        ),
    }


# ---------------------------------------------------------------------------
# Audit 3 — feature-selection leakage
# ---------------------------------------------------------------------------


def audit_selection_leakage(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                            k: int = 20, cv: int = 5) -> dict[str, Any]:
    """Selecting top-k on the full training set versus nested selection.

    The flat/flat comparison is the one that matters: an analyst who ranks features
    once on all training data and then cross-validates that subset reports an
    optimistic score, because the ranking already saw the held-out folds.
    """
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    splitter = _cv(groups, cv)

    # Each model class needs its own estimator instance per cross_val_score call.
    def pipeline_nested():
        return Pipeline([
            ("scale", StandardScaler()),
            ("select", SelectKBest(f_classif, k=k)),
            ("rf", _rf()),
        ])

    def pipeline_flat_orders():
        # Selection already happened outside; the pipeline still scales correctly
        # but the chosen columns were decided using every fold.
        return Pipeline([("scale", StandardScaler()), ("rf", _rf())])

    nested = cross_val_score(pipeline_nested(), X, y, cv=splitter, groups=groups,
                             scoring="f1_macro", n_jobs=1)

    selector = SelectKBest(f_classif, k=k).fit(X, y)
    idx = np.where(selector.get_support())[0]
    flat = cross_val_score(pipeline_flat_orders(), X[:, idx], y, cv=splitter,
                           groups=groups, scoring="f1_macro", n_jobs=1)

    return {
        "k": int(k),
        "nested_selection_mean": float(nested.mean()),
        "nested_selection_std": float(nested.std()),
        "flat_selection_mean": float(flat.mean()),
        "flat_selection_std": float(flat.std()),
        "inflation": float(flat.mean() - nested.mean()),
        "selected_features": [ALL_FEATURES[i] for i in idx],
        "explanation": (
            "nested = selection re-fitted inside every fold (honest). "
            "flat = selection performed once on all training data (leaky). "
            "The Phase 1 report used neither: it cross-validated all 83 features "
            "with no selection at all, which is free of this leak."
        ),
    }


# ---------------------------------------------------------------------------
# Audit 4 — manifest integrity
# ---------------------------------------------------------------------------


def audit_manifests() -> dict[str, Any]:
    out: dict[str, Any] = {}

    import hashlib

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    baseline_file = config.ADMIN_DIR / "phase1_artifacts.sha256"
    if baseline_file.exists():
        expected: dict[str, str] = {}
        for line in baseline_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            digest, path = line.split(maxsplit=1)
            # GNU sha256sum emits "<digest> *<path>" in binary mode.
            expected[path.strip().lstrip("*")] = digest
        current = {}
        for path in expected:
            p = config.PROJECT_ROOT / path
            current[path] = sha(p) if p.exists() else "MISSING"
        changed = [p for p in expected if expected[p] != current[p]]
        out["baseline_snapshot"] = {
            "files_checked": len(expected),
            "changed": changed,
            "unchanged": len(changed) == 0,
        }
    else:
        out["baseline_snapshot"] = {"files_checked": 0, "unchanged": None,
                                    "note": "no baseline snapshot recorded"}

    splits = pd.read_csv(config.METADATA_DIR / "splits.csv")
    out["splits"] = {
        "rows": int(len(splits)),
        "counts": {k: int(v) for k, v in splits["split"].value_counts().items()},
        "unique_rel_paths": int(splits["rel_path"].nunique()),
        "groups": int(splits["group_id"].nunique()),
    }

    interim = sorted(p.name for p in config.INTERIM_DIR.glob("*"))
    out["interim_artefacts"] = interim
    out["test_split_was_featurised"] = any("test" in n for n in interim)
    out["only_training_split_used"] = all("test" not in n for n in interim)

    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(cv: int = 5, k: int = 20) -> dict[str, Any]:
    X, y, groups, features = load_train_features()
    print(f"auditing {X.shape[0]} training images x {X.shape[1]} features, "
          f"{len(np.unique(groups))} groups")

    print("1/4 group integrity ...")
    groups_ok = audit_group_integrity(X, y, groups, cv)
    print(f"    shared groups across folds: {groups_ok['group_violations']}")

    print("2/4 scaling leakage ...")
    scaling = audit_scaling_leakage(X, y, groups, cv)
    print(f"    correct {scaling['correct_mean']:.4f} vs leaky "
          f"{scaling['leaky_mean']:.4f} (delta {scaling['delta']:+.4f})")

    print("3/4 feature-selection leakage ...")
    selection = audit_selection_leakage(X, y, groups, k, cv)
    print(f"    nested {selection['nested_selection_mean']:.4f} vs flat "
          f"{selection['flat_selection_mean']:.4f} "
          f"(inflation {selection['inflation']:+.4f})")

    print("4/4 manifest integrity ...")
    manifests = audit_manifests()

    phase1 = load_json(config.EVAL_REPORT_DIR / "feature_screening.json")
    headline = phase1["group_benchmark_macro_f1_grouped_cv"]["all"]

    verdict = (
        groups_ok["passed"]
        and not manifests["test_split_was_featurised"]
        and abs(headline["macro_f1_mean"] - scaling["correct_mean"]) < 1e-9
    )

    summary = {
        "n_images": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "group_integrity": groups_ok,
        "scaling_leakage": scaling,
        "selection_leakage": selection,
        "manifest_integrity": manifests,
        "phase1_headline": {
            "macro_f1_mean": headline["macro_f1_mean"],
            "macro_f1_std": headline["macro_f1_std"],
            "n_features": headline["n_features"],
            "scheme": headline["scheme"],
        },
        "reproduced_exactly": bool(abs(headline["macro_f1_mean"] - scaling["correct_mean"]) < 1e-9),
        "verdict": bool(verdict),
    }
    save_json(summary, config.EVAL_REPORT_DIR / "leakage_audit.json")
    return summary


def main() -> None:
    summary = run()
    print(json.dumps({
        "group_violations": summary["group_integrity"]["group_violations"],
        "scaling_delta": round(summary["scaling_leakage"]["delta"], 5),
        "selection_inflation": round(summary["selection_leakage"]["inflation"], 5),
        "headline_reproduced": summary["reproduced_exactly"],
        "phase1_artifacts_unchanged": summary["manifest_integrity"]["baseline_snapshot"].get("unchanged"),
        "verdict": summary["verdict"],
    }, indent=2))


if __name__ == "__main__":
    main()
