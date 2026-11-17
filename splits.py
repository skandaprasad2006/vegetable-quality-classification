"""Leakage-safe split construction.

Design decisions, in order of importance:

1. **Groups before splits.**  Every near-duplicate cluster from the audit is an
   indivisible unit.  Two photos of the same vegetable from the same session are
   near-identical; if one is in train and the other in test, the test score is
   inflated by memorisation rather than generalisation.
2. **Stratification on the joint (vegetable, quality) cell.**  The dataset is
   strongly imbalanced (27 → 1,234 images per cell), so a plain random split can
   leave a class almost absent from validation/test.
3. **Deterministic.**  Randomness is drawn from a generator seeded with
   ``(config.SEED, stratum)``, so the split is reproducible and independent of
   iteration order.
4. **Augmentation is never applied here.**  Only original images are split;
   augmentation happens inside the training pipeline on the training split.

Outputs
-------
``datasets/metadata/splits.csv``          one row per image with its split
``datasets/metadata/split_summary.json``  per-split / per-class counts + checks
``datasets/processed/{split}/...``         materialised copies for training

Run:  ``python -m src.data.splits``
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.utils import load_json, save_json


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def attach_groups(inventory: pd.DataFrame, duplicate_data: dict[str, Any]) -> pd.DataFrame:
    """Add an integer ``group_id``; duplicate clusters share one id.

    Images that belong to no near-duplicate cluster get a unique negative id so
    that group ids never collide with cluster ids.
    """
    df = inventory.copy()
    group_of: dict[str, int] = duplicate_data.get("group_of_rel_path", {})
    df["group_id"] = [group_of.get(p, -(i + 1)) for i, p in enumerate(df["rel_path"])]
    return df


# ---------------------------------------------------------------------------
# Stratified group assignment
# ---------------------------------------------------------------------------


def _assign_stratum(
    sizes: list[tuple[int, int]],
    total: int,
    ratios: dict[str, float],
    rng: np.random.Generator,
    min_groups_for_split: int = 1,
) -> dict[int, str]:
    """Greedily distribute groups of a single stratum across splits.

    ``sizes`` is ``[(group_id, count), ...]``.  Groups are visited largest first
    (hardest to place well) and each goes to the split with the largest remaining
    deficit relative to its target.  At least one group is forced into every
    split when the stratum has enough groups to allow it, so no class is absent
    from validation or test.
    """
    order = list(range(len(sizes)))
    rng.shuffle(order)
    order.sort(key=lambda i: -sizes[i][1])  # stable: shuffle breaks size ties

    target = {s: ratios[s] * total for s in config.SPLITS}
    current = {s: 0 for s in config.SPLITS}
    assignment: dict[int, str] = {}
    remaining_groups = len(sizes)
    forced = {s: False for s in config.SPLITS}

    for pos, i in enumerate(order):
        gid, n = sizes[i]

        # Reserve groups so that smaller splits are not starved by the greedy pass.
        must_fill = [
            s for s in config.SPLITS
            if not forced[s]
            and remaining_groups <= (len(config.SPLITS) - pos)
            and min_groups_for_split
        ]

        if must_fill:
            split = must_fill[0]
        else:
            deficit = {s: target[s] - current[s] for s in config.SPLITS}
            split = max(config.SPLITS, key=lambda s: (deficit[s], -config.SPLITS.index(s)))

        assignment[gid] = split
        current[split] += n
        forced[split] = True
        remaining_groups -= 1

    return assignment


def build_splits(
    inventory: pd.DataFrame,
    duplicate_data: dict[str, Any],
    seed: int = config.SEED,
    ratios: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Assign every image exactly one split, keeping duplicate clusters intact."""
    ratios = dict(ratios or config.SPLIT_RATIOS)
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")

    df = attach_groups(inventory, duplicate_data)
    df = df[df["readable"]].copy()

    assignment: dict[int, str] = {}
    for stratum, part in df.groupby(["vegetable", "quality"], sort=True):
        counts = part.groupby("group_id").size()
        sizes = [(int(g), int(n)) for g, n in counts.items()]
        total = sum(n for _, n in sizes)
        rng = np.random.default_rng([seed, *[ord(ch) for ch in "/".join(stratum)]])
        assignment.update(_assign_stratum(sizes, total, ratios, rng))

    df["split"] = df["group_id"].map(assignment)
    if df["split"].isna().any():
        raise RuntimeError("some images were not assigned a split")
    return df


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def verify_splits(df: pd.DataFrame) -> dict[str, Any]:
    """Assert the properties Gate 1 depends on.  Returns a check report."""
    split_counts = df["split"].value_counts().to_dict()

    groups_per_split = df.groupby("group_id")["split"].nunique()
    leaking_groups = groups_per_split[groups_per_split > 1]

    sha_per_split = df.groupby("sha256")["split"].nunique()
    leaking_hashes = sha_per_split[sha_per_split > 1]

    missing = [s for s in config.SPLITS if split_counts.get(s, 0) == 0]

    per_split_classes = {
        s: sorted(df.loc[df["split"] == s, "quality"].unique().tolist())
        for s in config.SPLITS
    }
    per_split_vegetables = {
        s: sorted(df.loc[df["split"] == s, "vegetable"].unique().tolist())
        for s in config.SPLITS
    }

    return {
        "counts_per_split": {s: int(split_counts.get(s, 0)) for s in config.SPLITS},
        "fractions": {
            s: round(split_counts.get(s, 0) / len(df), 4) for s in config.SPLITS
        },
        "groups": int(df["group_id"].nunique()),
        "groups_spanning_splits": int(len(leaking_groups)),
        "sha256_spanning_splits": int(len(leaking_hashes)),
        "empty_splits": missing,
        "classes_present_per_split": per_split_classes,
        "vegetables_present_per_split": per_split_vegetables,
        "all_groups_intact": bool(len(leaking_groups) == 0),
        "no_exact_duplicate_across_splits": bool(len(leaking_hashes) == 0),
        "total_images": int(len(df)),
    }


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------


def materialise(df: pd.DataFrame, raw_root: Path, out_root: Path) -> int:
    """Copy each image into ``out_root/<split>/<vegetable>/<quality>/``.

    Copies (not symlinks/hardlinks) keep the processed tree self-contained and
    portable; 125 MiB is well inside the storage budget.
    """
    written = 0
    for row in df.itertuples(index=False):
        dst = out_root / row.split / row.vegetable / row.quality / row.file_name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_root / row.rel_path, dst)
        written += 1
    return written


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_splits(
    inventory: pd.DataFrame,
    duplicate_data: dict[str, Any],
    raw_root: Path | None = None,
    out_root: Path | None = None,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_root = Path(raw_root or config.RAW_DIR)
    out_root = Path(out_root or config.PROCESSED_DIR)

    df = build_splits(inventory, duplicate_data, seed=seed)
    checks = verify_splits(df)

    df_out = df[[
        "rel_path", "file_name", "vegetable", "quality", "split", "group_id",
        "sha256", "width", "height", "sharpness", "brightness",
    ]].copy()
    df_out["is_in_duplicate_group"] = [
        int(g) >= 0 for g in df["group_id"]
    ]

    summary = {
        "seed": seed,
        "ratios_requested": config.SPLIT_RATIOS,
        "checks": checks,
        "per_split_cell_counts": {
            s: {
                f"{v}/{q}": int(n)
                for (v, q), n in df[df["split"] == s].groupby(["vegetable", "quality"]).size().items()
            }
            for s in config.SPLITS
        },
        "per_split_vegetable_counts": {
            s: {k: int(v) for k, v in df[df["split"] == s]["vegetable"].value_counts().items()}
            for s in config.SPLITS
        },
        "per_split_quality_counts": {
            s: {k: int(v) for k, v in df[df["split"] == s]["quality"].value_counts().items()}
            for s in config.SPLITS
        },
    }

    config.ensure_project_dirs()
    df_out.to_csv(config.METADATA_DIR / "splits.csv", index=False)
    save_json(summary, config.METADATA_DIR / "split_summary.json")
    save_json(config.class_map(), config.METADATA_DIR / "class_map.json")

    copied = materialise(df, raw_root, out_root)
    summary["materialised_files"] = copied
    save_json(summary, config.METADATA_DIR / "split_summary.json")

    return df_out, summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Build leakage-safe train/val/test splits")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--no-copy", action="store_true",
                    help="only write manifests, do not copy images into datasets/processed")
    args = ap.parse_args()

    inventory = pd.read_csv(config.METADATA_DIR / "image_inventory.csv")
    duplicate_data = load_json(config.METADATA_DIR / "duplicate_groups.json")

    if args.no_copy:
        df = build_splits(inventory, duplicate_data, seed=args.seed)
        print(json.dumps(verify_splits(df), indent=2))
        return

    _, summary = run_splits(inventory, duplicate_data, seed=args.seed)
    print(json.dumps(summary["checks"], indent=2))
    print(f"materialised {summary['materialised_files']} files")


if __name__ == "__main__":
    main()
