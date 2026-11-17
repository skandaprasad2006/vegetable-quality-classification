"""Investigate near-duplicate groups whose members carry different labels.

Phase 1 found 42 perceptual clusters (866 images) where visually near-identical
images are labelled differently.  Those images sit on the decision boundary or
are mislabelled, and either way they set a ceiling on achievable accuracy and
distort per-class metrics.  This module characterises them rather than assuming
they are noise.

Outputs
-------
``evaluation/metrics/label_conflicts.csv``              one row per conflicting group
``evaluation/metrics/label_conflict_pairs.csv``         label-pair co-occurrence
``evaluation/graphs/label_conflict_examples.png``       visual montage
``evaluation/reports/label_conflicts.md``               written analysis

Run: ``python -m src.data.label_conflicts``
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

import config
from src.utils import imread_unicode, load_json, save_json

MAX_MONTAGE_GROUPS = 6
IMAGES_PER_GROUP = 4


# ---------------------------------------------------------------------------
# Characterisation
# ---------------------------------------------------------------------------


def _pairs(labels: list[list[str]]) -> list[tuple[str, str]]:
    """All unordered label pairs present inside one group (as 'Veg/Quality')."""
    names = sorted({f"{v}/{q}" for v, q in labels})
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out.append((names[i], names[j]))
    return out


def build_table(dups: dict[str, Any], inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group table and label-pair co-occurrence table."""
    lookup = inventory.set_index("rel_path")
    rows: list[dict[str, Any]] = []

    for gid, group in enumerate(dups["label_conflicting_groups"]):
        members = group["members"]
        labels = group["labels"]
        qualities = sorted({q for _, q in labels})
        vegetables = sorted({v for v, _ in labels})

        shapes = [
            (int(lookup.loc[m, "width"]), int(lookup.loc[m, "height"]))
            for m in members if m in lookup.index
        ]
        sharp = [float(lookup.loc[m, "sharpness"]) for m in members if m in lookup.index]
        bright = [float(lookup.loc[m, "brightness"]) for m in members if m in lookup.index]

        rows.append({
            "conflict_id": gid,
            "size": group["size"],
            "max_hamming_within": group["max_hamming_within"],
            "vegetables": " | ".join(vegetables),
            "qualities": " | ".join(qualities),
            "same_vegetable": len(vegetables) == 1,
            "n_distinct_labels": len(labels),
            "mean_sharpness": float(np.mean(sharp)) if sharp else np.nan,
            "mean_brightness": float(np.mean(bright) if bright else np.nan),
            "consistent_dimensions": len(set(shapes)) <= 1,
            "example_member": members[0],
            "members": " ; ".join(members),
        })

    df = pd.DataFrame(rows)

    pair_counts: Counter[tuple[str, str]] = Counter()
    pair_images: Counter[tuple[str, str]] = Counter()
    for group in dups["label_conflicting_groups"]:
        for a, b in _pairs(group["labels"]):
            pair_counts[(a, b)] += 1
            pair_images[(a, b)] += group["size"]

    pairs_df = pd.DataFrame(
        [{"label_a": a, "label_b": b, "groups": n, "images_involved": pair_images[(a, b)]}
         for (a, b), n in pair_counts.most_common()]
    )
    return df, pairs_df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def montage(dups: dict[str, Any], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = sorted(dups["label_conflicting_groups"], key=lambda g: -g["size"])[:MAX_MONTAGE_GROUPS]
    if not groups:
        return

    fig, axes = plt.subplots(len(groups), IMAGES_PER_GROUP,
                             figsize=(2.4 * IMAGES_PER_GROUP, 2.3 * len(groups)),
                             squeeze=False)

    for row, group in enumerate(groups):
        members = group["members"][:IMAGES_PER_GROUP]
        for col in range(IMAGES_PER_GROUP):
            ax = axes[row, col]
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if col >= len(members):
                ax.axis("off")
                continue
            rel = members[col]
            img = imread_unicode(config.RAW_DIR / rel)
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            veg, qual = rel.split("/")[0], rel.split("/")[1]
            ax.set_xlabel(f"{veg}\n{qual}", fontsize=6)
        ax0 = axes[row, 0]
        ax0.set_ylabel(f"group {row + 1}\n({group['size']} imgs, "
                       f"dH≤{group['max_hamming_within']})", fontsize=7)

    fig.suptitle("Near-duplicate groups carrying conflicting quality labels", y=1.0)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(df: pd.DataFrame, pairs: pd.DataFrame, dups: dict[str, Any],
                 summary: dict[str, Any], path: Path) -> None:
    total_images = summary["total_images"]
    conflict_images = int(df["size"].sum())
    same_veg = int(df["same_vegetable"].sum())
    cross_veg = len(df) - same_veg

    lines = [
        "# Label-Conflicting Near-Duplicate Groups", "",
        f"Derived from `datasets/metadata/duplicate_groups.json` "
        f"(dHash Hamming threshold ≤ {dups['threshold']}).", "",
        "## Scale", "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Conflicting groups | {len(df)} |",
        f"| Images involved | {conflict_images} |",
        f"| Share of corpus | {conflict_images / total_images * 100:.2f}% |",
        f"| Groups where all members share one vegetable | {same_veg} |",
        f"| Groups spanning different vegetables | {cross_veg} |",
        f"| Median group size | {df['size'].median():.0f} |",
        f"| Largest group | {int(df['size'].max())} |",
        "",
        "## Most common conflicting label pairs", "",
        "| Label A | Label B | Groups | Images |",
        "|---|---|---:|---:|",
    ]
    for r in pairs.head(12).itertuples(index=False):
        lines.append(f"| {r.label_a} | {r.label_b} | {r.groups} | {r.images_involved} |")

    top_quality_pairs = Counter()
    for r in pairs.itertuples(index=False):
        qa, qb = r.label_a.split("/")[1], r.label_b.split("/")[1]
        top_quality_pairs[tuple(sorted((qa, qb)))] += r.groups

    lines += ["", "## Quality pairs regardless of vegetable", "",
              "| Quality pair | Groups |", "|---|---:|"]
    for (a, b), n in top_quality_pairs.most_common():
        lines.append(f"| {a} ↔ {b} | {n} |")

    lines += [
        "",
        "## Interpretation", "",
        "Two distinct phenomena are present and must not be conflated:", "",
        "1. **Genuine visual ambiguity.** `Old` and `Dried` describe overlapping",
        "   states — a shrivelled specimen can reasonably be called either. Groups",
        "   pairing these classes are boundary cases, not annotation errors.",
        "2. **Frames spanning different specimens.** Some 'clusters' are actually",
        "   several similar-looking vegetables photographed in one session. A",
        "   difference in label is then correct: the images look alike because of",
        "   consistent lighting and identical backgrounds, not because they are the",
        "   same object.",
        "",
        "The dHash similarity that defines a cluster cannot distinguish these cases.",
        "That is precisely why the splitter keeps clusters whole: it is better to",
        "over-group and lose a handful of training examples than to leak a specimen",
        "across the train/test boundary.",
        "",
        "## Consequences for Phase 2", "",
        f"* **A ceiling exists.** {conflict_images} images "
        f"({conflict_images / total_images * 100:.1f}% of the corpus) are in groups where "
        "no model can satisfy every label simultaneously. Perfect accuracy is",
        "  unattainable by construction, independent of model quality.",
        "* **Per-class metrics inherit the noise.** `Dried` and `Old` per-class recall",
        "  will be depressed by boundary cases; this is reported alongside support",
        "  counts and not presented as pure model failure.",
        "* **Error analysis must stratify by this flag.** Phase 2's confusion analysis",
        "  will separate errors on conflict-flagged images from errors on clean images,",
        "  so the reported confusion matrix is not blamed entirely on the model.",
        "* **Relabelling was rejected.** Deciding which label is 'correct' in an",
        "  ambiguous pair requires a domain expert and would silently reshape the",
        "  dataset. Retaining and documenting is the defensible choice; the flag is",
        "  available in `datasets/metadata/duplicate_groups.json` for anyone who",
        "  wants to re-run the analysis with these images excluded.",
        "* **The test split is affected too.** Conflicts were distributed by the",
        "  grouped split, so a conflicting cluster sits entirely in one split; the",
        "  test set therefore contains some of these groups and the reported test",
        "  score carries the same inherent noise. This is stated, not corrected.",
        "",
        "## Example groups",
        "",
        "See `evaluation/graphs/label_conflict_examples.png` for the six largest",
        "clusters, and `evaluation/metrics/label_conflicts.csv` for every group.",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run() -> dict[str, Any]:
    config.ensure_project_dirs()
    dups = load_json(config.METADATA_DIR / "duplicate_groups.json")
    inventory = pd.read_csv(config.METADATA_DIR / "image_inventory.csv")

    df, pairs = build_table(dups, inventory)
    df.to_csv(config.METRICS_DIR / "label_conflicts.csv", index=False)
    pairs.to_csv(config.METRICS_DIR / "label_conflict_pairs.csv", index=False)

    montage(dups, config.GRAPHS_DIR / "label_conflict_examples.png")

    quality_pairs: Counter[tuple[str, str]] = Counter()
    for r in pairs.itertuples(index=False):
        quality_pairs[tuple(sorted((r.label_a.split("/")[1], r.label_b.split("/")[1])))] += r.groups

    summary = {
        "total_images": int(len(inventory)),
        "conflict_groups": int(len(df)),
        "conflict_images": int(df["size"].sum()),
        "share_of_corpus": float(df["size"].sum() / len(inventory)),
        "groups_same_vegetable": int(df["same_vegetable"].sum()),
        "groups_cross_vegetable": int((~df["same_vegetable"]).sum()),
        "median_group_size": float(df["size"].median()) if len(df) else 0.0,
        "largest_group": int(df["size"].max()) if len(df) else 0,
        "top_quality_pairs": [
            {"pair": list(k), "groups": int(v)} for k, v in quality_pairs.most_common()
        ],
        "top_label_pairs": pairs.head(10).to_dict("records"),
    }
    save_json(summary, config.EVAL_REPORT_DIR / "label_conflicts.json")
    write_report(df, pairs, dups, summary, config.EVAL_REPORT_DIR / "label_conflicts.md")
    return summary


def main() -> None:
    s = run()
    print(json.dumps({k: s[k] for k in
                      ("conflict_groups", "conflict_images", "share_of_corpus",
                       "groups_same_vegetable", "groups_cross_vegetable",
                       "median_group_size", "largest_group")}, indent=2))
    print("top quality pairs:", [(tuple(p["pair"]), p["groups"]) for p in s["top_quality_pairs"][:6]])


if __name__ == "__main__":
    main()
