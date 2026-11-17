"""Screening handcrafted features: which descriptors actually carry signal?

For every feature this reports one-way ANOVA F and p-value against the quality
label, mutual information, and Random-Forest Gini importance.

The important part is the *model* comparison.  A 5-fold cross-validated Random
Forest is fitted per feature group, twice:

``random CV``
    Plain stratified K-fold inside the training split.  Near-duplicate clusters
    land on both sides of the fold boundary, so this number is **optimistic**.
``grouped CV``
    Stratified *Group* K-fold with the duplicate-cluster ids from Phase 1.  No
    cluster is ever split across folds.  This is the honest estimate and is the
    one quoted in the report.

Reporting both makes the size of the near-duplicate effect visible instead of
letting the flattering number stand alone.

Only the training split is used, so feature selection cannot leak into
validation or test.

Run: ``python -m src.features.screening [--force]``
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import config
from src.features.handcrafted import ALL_FEATURES, extract_features, group_indices
from src.preprocessing.pipeline import segment_vegetable
from src.utils import imread_unicode, save_json

FEATURE_ORDER = ALL_FEATURES


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _load_one(path: Path) -> dict[str, Any]:
    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    mask, info = segment_vegetable(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not info["ok"]:
        mask = None
    feats = extract_features(rgb, mask)
    feats["_seg_area_frac"] = float(info["area_fraction"])
    feats["_seg_ok"] = float(info["ok"])
    return feats


def extract_split_features(split: str = "train", workers: int = 8,
                           root: Path | None = None) -> pd.DataFrame:
    root = Path(root or config.PROCESSED_DIR)
    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    part = manifest[manifest["split"] == split].reset_index(drop=True)

    paths = [root / split / r.vegetable / r.quality / r.file_name
             for r in part.itertuples(index=False)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_load_one, paths))

    df = pd.DataFrame(rows, columns=list(FEATURE_ORDER) + ["_seg_area_frac", "_seg_ok"])
    df.insert(0, "vegetable", part["vegetable"].values)
    df.insert(1, "quality", part["quality"].values)
    df.insert(2, "file_name", part["file_name"].values)
    df.insert(3, "group_id", part["group_id"].values)
    print(f"extracted {len(df)} rows from '{split}' in {time.perf_counter() - t0:.1f}s")
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def univariate_screen(X: np.ndarray, y: np.ndarray, names: tuple[str, ...]) -> pd.DataFrame:
    from sklearn.feature_selection import f_classif, mutual_info_classif

    f_vals, p_vals = f_classif(X, y)
    f_vals = np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
    p_vals = np.nan_to_num(p_vals, nan=1.0)
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=config.SEED)

    groups = group_indices(names)
    group_of = {names[i]: g for g, idx in groups.items() for i in idx}

    return pd.DataFrame({
        "feature": list(names),
        "group": [group_of.get(n, "other") for n in names],
        "anova_f": f_vals,
        "p_value": p_vals,
        "mutual_info": mi,
    }).sort_values("anova_f", ascending=False).reset_index(drop=True)


def forest_importance(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=300, random_state=config.SEED, n_jobs=-1, class_weight="balanced_subsample",
    )
    rf.fit(X, y)
    return rf.feature_importances_


def _make_pipeline():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200, random_state=config.SEED, n_jobs=-1,
            class_weight="balanced_subsample",
        )),
    ])


def group_benchmark(
    X: np.ndarray,
    y: np.ndarray,
    names: tuple[str, ...],
    cv: int = 5,
    groups: np.ndarray | None = None,
    label: str = "",
) -> dict[str, Any]:
    """Cross-validated macro-F1 per feature group.

    ``groups`` switches from stratified K-fold to stratified *group* K-fold.
    """
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_score

    groups_idx = group_indices(names)
    subsets: dict[str, list[int]] = {**groups_idx, "all": list(range(len(names)))}
    subsets = {k: v for k, v in subsets.items() if len(v) >= 2}

    if groups is None:
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=config.SEED)
    else:
        splitter = StratifiedGroupKFold(n_splits=cv, shuffle=True, random_state=config.SEED)

    out: dict[str, Any] = {}
    for name, idx in subsets.items():
        kwargs = {"groups": groups} if groups is not None else {}
        scores = cross_val_score(
            _make_pipeline(), X[:, idx], y, cv=splitter,
            scoring="f1_macro", n_jobs=1, **kwargs,
        )
        out[name] = {
            "n_features": len(idx),
            "macro_f1_mean": float(scores.mean()),
            "macro_f1_std": float(scores.std()),
            "folds": [float(s) for s in scores],
            "scheme": "grouped" if groups is not None else "random",
        }
        print(f"  [{label:<8}] {name:<8} ({len(idx):>3} feats) "
              f"macro-F1 = {scores.mean():.4f} ± {scores.std():.4f}")
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_cached(cache: Path, split: str) -> pd.DataFrame | None:
    if not cache.exists():
        return None
    data = np.load(cache, allow_pickle=True)
    n = data["X"].shape[0]
    if "groups" not in data:
        return None
    return pd.DataFrame({
        "vegetable": data["vegetable"], "quality": data["y"],
        "group_id": data["groups"],
        **{name: data["X"][:, i] for i, name in enumerate(data["features"])},
        "_seg_area_frac": data["seg_area_frac"], "_seg_ok": data["seg_ok"],
    }) if n else None


def run_screening(split: str = "train", workers: int = 8, cv: int = 5,
                  force: bool = False) -> dict[str, Any]:
    config.ensure_project_dirs()
    cache = config.INTERIM_DIR / f"features_{split}.npz"

    df = None if force else _load_cached(cache, split)
    if df is None:
        df = extract_split_features(split, workers=workers)
        np.savez_compressed(
            cache,
            X=df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32),
            y=df["quality"].to_numpy(),
            vegetable=df["vegetable"].to_numpy(),
            groups=df["group_id"].to_numpy(),
            features=np.array(FEATURE_ORDER),
            seg_area_frac=df["_seg_area_frac"].to_numpy(),
            seg_ok=df["_seg_ok"].to_numpy(),
        )
        df.to_csv(config.INTERIM_DIR / f"features_{split}.csv", index=False)
    else:
        print(f"loaded cached features from {cache}")

    X = np.nan_to_num(df[list(FEATURE_ORDER)].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = df["quality"].to_numpy()
    groups = df["group_id"].to_numpy()

    print(f"screening {X.shape[0]} images x {X.shape[1]} features "
          f"in {len(np.unique(groups))} duplicate-safe groups")

    screen = univariate_screen(X, y, FEATURE_ORDER)
    screen["rf_importance"] = forest_importance(X, y)
    screen = screen.sort_values("rf_importance", ascending=False).reset_index(drop=True)
    screen.to_csv(config.METRICS_DIR / "feature_screening.csv", index=False)

    print("random stratified CV (optimistic, near-duplicates may cross folds):")
    bench_random = group_benchmark(X, y, FEATURE_ORDER, cv=cv, label="random")
    print("grouped CV (honest, duplicate clusters kept whole):")
    bench_grouped = group_benchmark(X, y, FEATURE_ORDER, cv=cv, groups=groups, label="grouped")

    print("vegetable identity, grouped CV:")
    veg_grouped = group_benchmark(X, df["vegetable"].to_numpy(), FEATURE_ORDER,
                                  cv=cv, groups=groups, label="veg")

    summary = {
        "split": split,
        "n_images": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_groups": int(len(np.unique(groups))),
        "quality_support": {k: int(v) for k, v in pd.Series(y).value_counts().items()},
        "feature_groups": {g: len(v) for g, v in group_indices(FEATURE_ORDER).items()},
        "group_benchmark_macro_f1_grouped_cv": bench_grouped,
        "group_benchmark_macro_f1_random_cv": bench_random,
        "group_benchmark_vegetable_macro_f1_grouped_cv": veg_grouped,
        "near_duplicate_inflation_all_features": float(
            bench_random["all"]["macro_f1_mean"] - bench_grouped["all"]["macro_f1_mean"]
        ),
        "top_10_by_anova": screen.head(10)[
            ["feature", "group", "anova_f", "p_value", "mutual_info", "rf_importance"]
        ].to_dict("records"),
        "n_features_p_below_0_01": int((screen["p_value"] < 0.01).sum()),
        "segmentation": {
            "usable_fraction": float(df["_seg_ok"].mean()),
            "mean_area_fraction": float(df["_seg_area_frac"].mean()),
            "min_area_fraction": float(df["_seg_area_frac"].min()),
            "max_area_fraction": float(df["_seg_area_frac"].max()),
        },
    }
    save_json(summary, config.EVAL_REPORT_DIR / "feature_screening.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Screen handcrafted features")
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--force", action="store_true", help="re-extract instead of reusing the cache")
    args = ap.parse_args()
    s = run_screening(args.split, args.workers, args.cv, args.force)
    print(json.dumps({
        "n_images": s["n_images"],
        "n_features": s["n_features"],
        "n_groups": s["n_groups"],
        "grouped": {k: round(v["macro_f1_mean"], 4)
                    for k, v in s["group_benchmark_macro_f1_grouped_cv"].items()},
        "random": {k: round(v["macro_f1_mean"], 4)
                   for k, v in s["group_benchmark_macro_f1_random_cv"].items()},
        "inflation": round(s["near_duplicate_inflation_all_features"], 4),
    }, indent=2))


if __name__ == "__main__":
    main()
