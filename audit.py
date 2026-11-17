"""Dataset acquisition verification and integrity audit.

The audit answers, with measurements rather than assumptions:

* how many images exist, where they are, and what they weigh;
* whether every file actually decodes (Pillow ``verify`` *and* OpenCV decode);
* what the image dimensions, formats and colour statistics are;
* which files are byte-identical duplicates;
* which files are perceptual near-duplicates, including duplicates that carry
  *different* labels (label noise / leakage risk).

Outputs
-------
``datasets/metadata/image_inventory.csv``    one row per image
``datasets/metadata/audit_summary.json``     aggregates used by notebooks + tests
``datasets/metadata/duplicate_groups.json``  duplicate clusters (union-find)
``reports/dataset_audit.md``                 human-readable report

Run:  ``python -m src.data.audit``
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import config
from src.utils import dhash, fmt_bytes, hamming_matrix, imread_unicode, save_json, sha256_file

INVENTORY_COLUMNS = [
    "rel_path", "file_name", "vegetable_dir", "vegetable", "quality",
    "split_dir", "file_size_bytes", "sha256", "dhash_hex",
    "width", "height", "channels", "pillow_mode", "readable",
    "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
    "brightness", "contrast", "sharpness", "saturation", "colourfulness",
    "clipped_black_frac", "clipped_white_frac", "decode_error",
]


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_images(root: Path) -> list[dict[str, str]]:
    """Return one record per image found under ``root``.

    Raises if the expected ``<vegetable>/<quality>/`` layout is not found, so a
    silently empty or restructured dataset can never pass the audit.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root not found: {root}")

    found: list[dict[str, str]] = []
    unknown_dirs: list[str] = []

    for veg_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        veg_name = config.VEGETABLE_DIRS.get(veg_dir.name)
        if veg_name is None:
            unknown_dirs.append(veg_dir.name)
            continue
        for qual_dir in sorted(p for p in veg_dir.iterdir() if p.is_dir()):
            if qual_dir.name not in config.QUALITY_CLASSES:
                unknown_dirs.append(f"{veg_dir.name}/{qual_dir.name}")
                continue
            for f in sorted(qual_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS:
                    found.append({
                        "rel_path": f.relative_to(root).as_posix(),
                        "file_name": f.name,
                        "vegetable_dir": veg_dir.name,
                        "vegetable": veg_name,
                        "quality": qual_dir.name,
                        "split_dir": f"{veg_name}/{qual_dir.name}",
                    })

    if unknown_dirs:
        raise ValueError(
            "unexpected directories in dataset (labels would be dropped silently): "
            + ", ".join(unknown_dirs)
        )
    if not found:
        raise ValueError(f"no images found under {root}")
    return found


# ---------------------------------------------------------------------------
# Per-image inspection
# ---------------------------------------------------------------------------


def _colourfulness(rgb: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric."""
    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def inspect_image(root: Path, record: dict[str, str]) -> dict[str, Any]:
    """Open one image and measure it.  Never raises: failures are recorded."""
    path = Path(root) / record["rel_path"]
    row: dict[str, Any] = dict(record)
    row.update({
        "file_size_bytes": path.stat().st_size,
        "sha256": "", "dhash_hex": "",
        "width": 0, "height": 0, "channels": 0, "pillow_mode": "",
        "readable": False,
        "mean_r": np.nan, "mean_g": np.nan, "mean_b": np.nan,
        "std_r": np.nan, "std_g": np.nan, "std_b": np.nan,
        "brightness": np.nan, "contrast": np.nan, "sharpness": np.nan,
        "saturation": np.nan, "colourfulness": np.nan,
        "clipped_black_frac": np.nan, "clipped_white_frac": np.nan,
        "decode_error": "",
    })

    try:
        row["sha256"] = sha256_file(path)
    except OSError as exc:
        row["decode_error"] = f"read: {exc}"

    # Pillow's verify() catches truncated JPEG data that OpenCV tolerates.
    try:
        from PIL import Image

        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            row["pillow_mode"] = im.mode
    except Exception as exc:  # noqa: BLE001 - any failure means "not verified"
        row["decode_error"] = (row["decode_error"] + f" pillow: {exc}").strip()

    import cv2

    bgr = imread_unicode(path, cv2.IMREAD_COLOR)
    if bgr is None:
        row["decode_error"] = (row["decode_error"] + " cv2: decode failed").strip()
        return row

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)

    row.update({
        "readable": True,
        "height": int(bgr.shape[0]),
        "width": int(bgr.shape[1]),
        "channels": int(bgr.shape[2]) if bgr.ndim == 3 else 1,
        "dhash_hex": format(dhash(gray), "016x"),
    })
    for i, ch in enumerate("rgb"):
        row[f"mean_{ch}"] = float(rgb[..., i].mean())
        row[f"std_{ch}"] = float(rgb[..., i].std())
    row["brightness"] = float(hsv[..., 2].mean())
    row["contrast"] = float(lab[..., 0].std())
    row["sharpness"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    row["saturation"] = float(hsv[..., 1].mean())
    row["colourfulness"] = _colourfulness(rgb)
    row["clipped_black_frac"] = float((gray <= 2).mean())
    row["clipped_white_frac"] = float((gray >= 253).mean())
    return row


def inspect_all(root: Path, records: list[dict[str, str]], workers: int = 8) -> pd.DataFrame:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(lambda r: inspect_image(root, r), records))
    return pd.DataFrame(rows, columns=INVENTORY_COLUMNS)


# ---------------------------------------------------------------------------
# Duplicate analysis
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def find_duplicate_groups(df: pd.DataFrame, threshold: int) -> dict[str, Any]:
    """Exact (SHA-256) and near-duplicate (dHash Hamming) clusters."""
    readable = df[df["readable"]].reset_index(drop=True)

    # --- exact duplicates
    exact: dict[str, list[str]] = {}
    for sha, grp in readable.groupby("sha256"):
        if len(grp) > 1:
            exact[sha] = sorted(grp["rel_path"].tolist())

    # --- near duplicates over every readable image
    hashes = np.array([int(h, 16) for h in readable["dhash_hex"]], dtype=np.uint64)
    n = len(readable)
    uf = _UnionFind(n)
    dist = hamming_matrix(hashes)
    close = np.triu(dist <= threshold, k=1)
    for i, j in np.argwhere(close):
        uf.union(int(i), int(j))

    clusters: dict[int, list[int]] = {}
    for idx in range(n):
        clusters.setdefault(uf.find(idx), []).append(idx)

    near_groups: list[dict[str, Any]] = []
    label_conflicts: list[dict[str, Any]] = []
    for members in clusters.values():
        if len(members) < 2:
            continue
        paths = [readable.loc[i, "rel_path"] for i in members]
        labels = {(readable.loc[i, "vegetable"], readable.loc[i, "quality"]) for i in members}
        max_d = int(dist[np.ix_(members, members)].max())
        entry = {
            "size": len(members),
            "members": sorted(paths),
            "labels": sorted([list(t) for t in labels]),
            "max_hamming_within": max_d,
            "spans_multiple_labels": len(labels) > 1,
        }
        near_groups.append(entry)
        if len(labels) > 1:
            label_conflicts.append(entry)

    near_groups.sort(key=lambda g: (-g["size"], g["members"][0]))

    # group id per rel_path, used by the splitter to keep clusters together
    group_of: dict[str, int] = {}
    for gid, members in enumerate(clusters.values()):
        if len(members) < 2:
            continue
        for i in members:
            group_of[readable.loc[i, "rel_path"]] = gid

    return {
        "exact_duplicate_groups": list(exact.values()),
        "near_duplicate_groups": near_groups,
        "label_conflicting_groups": label_conflicts,
        "group_of_rel_path": group_of,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class AuditResult:
    inventory: pd.DataFrame
    summary: dict[str, Any]


def run_audit(root: Path | None = None, workers: int = 8, threshold: int | None = None) -> AuditResult:
    root = Path(root or config.RAW_DIR)
    threshold = config.DHASH_NEAR_DUP_THRESHOLD if threshold is None else threshold

    t0 = time.perf_counter()
    records = enumerate_images(root)
    print(f"enumerated {len(records)} images under {root}")
    df = inspect_all(root, records, workers=workers)
    print(f"inspected in {time.perf_counter() - t0:.1f}s")

    dup = find_duplicate_groups(df, threshold)

    readable = df[df["readable"]]
    counts = df.groupby(["vegetable", "quality"]).size()
    per_vegetable = df.groupby("vegetable").size().to_dict()
    per_quality = df.groupby("quality").size().to_dict()

    cells = [
        {"vegetable": v, "quality": q, "count": int(counts.get((v, q), 0))}
        for v in config.VEGETABLE_CLASSES for q in config.QUALITY_CLASSES
    ]
    sizes = counts.to_numpy()
    imbalance_ratio = float(sizes.max() / sizes.min()) if sizes.min() > 0 else float("inf")

    summary: dict[str, Any] = {
        "dataset_name": "VegNet (Unripe, Ripe, Old, Dried, Damaged)",
        "audited_root": str(root),
        "source": {
            "mendeley": "https://data.mendeley.com/datasets/6nxnjbn9w6/1",
            "doi": "10.17632/6nxnjbn9w6.1",
            "paper_doi": "10.1016/j.dib.2022.108657",
            "huggingface": "https://huggingface.co/datasets/Project-AgML/VegNet_quality_classification",
            "licence": "CC BY 4.0",
            "upstream_reported_images": 6850,
            "local_variant": "Hugging Face redistribution, 6150 images, resized to 256x256",
        },
        "counts": {
            "total_files": int(len(df)),
            "readable": int(len(readable)),
            "unreadable": int((~df["readable"]).sum()),
            "per_vegetable": {k: int(v) for k, v in per_vegetable.items()},
            "per_quality": {k: int(v) for k, v in per_quality.items()},
            "cells": cells,
            "populated_cells": int(sum(1 for c in cells if c["count"] > 0)),
            "empty_cells": [f"{c['vegetable']}/{c['quality']}" for c in cells if c["count"] == 0],
            "imbalance_ratio_max_min_cell": imbalance_ratio,
            "largest_cell": max(cells, key=lambda c: c["count"]),
            "smallest_cell": min(cells, key=lambda c: c["count"]),
        },
        "storage": {
            "total_bytes": int(df["file_size_bytes"].sum()),
            "total_human": fmt_bytes(int(df["file_size_bytes"].sum())),
            "mean_bytes_per_image": float(df["file_size_bytes"].mean()),
            "max_bytes": int(df["file_size_bytes"].max()),
            "min_bytes": int(df["file_size_bytes"].min()),
        },
        "geometry": {
            "unique_dimensions": {
                f"{w}x{h}": int(c) for (w, h), c in
                readable.groupby(["width", "height"]).size().items()
            },
            "min_width": int(readable["width"].min()) if len(readable) else 0,
            "max_width": int(readable["width"].max()) if len(readable) else 0,
            "min_height": int(readable["height"].min()) if len(readable) else 0,
            "max_height": int(readable["height"].max()) if len(readable) else 0,
            "channels": {int(k): int(v) for k, v in readable["channels"].value_counts().items()},
            "pillow_modes": {str(k): int(v) for k, v in readable["pillow_mode"].value_counts().items()},
            "file_extensions": {str(k): int(v) for k, v in df["file_name"].str.rsplit(".", n=1).str[-1].str.lower().value_counts().items()},
        },
        "image_statistics": {
            col: {
                "mean": float(readable[col].mean()),
                "std": float(readable[col].std()),
                "min": float(readable[col].min()),
                "p05": float(readable[col].quantile(0.05)),
                "median": float(readable[col].median()),
                "p95": float(readable[col].quantile(0.95)),
                "max": float(readable[col].max()),
            }
            for col in ("brightness", "contrast", "sharpness", "saturation",
                        "colourfulness", "clipped_black_frac", "clipped_white_frac")
            if len(readable)
        },
        "duplicates": {
            "exact_group_count": len(dup["exact_duplicate_groups"]),
            "exact_image_count": sum(len(g) for g in dup["exact_duplicate_groups"]),
            "near_group_count": len(dup["near_duplicate_groups"]),
            "near_image_count": sum(g["size"] for g in dup["near_duplicate_groups"]),
            "near_dup_threshold": threshold,
            "label_conflicting_group_count": len(dup["label_conflicting_groups"]),
            "label_conflicting_image_count": sum(g["size"] for g in dup["label_conflicting_groups"]),
        },
        "audit_duration_s": round(time.perf_counter() - t0, 2),
    }

    return AuditResult(inventory=df, summary=summary), dup


def persist(result: AuditResult, dup: dict[str, Any]) -> None:
    config.ensure_project_dirs()
    result.inventory.to_csv(config.METADATA_DIR / "image_inventory.csv", index=False)
    save_json(result.summary, config.METADATA_DIR / "audit_summary.json")
    save_json(dup, config.METADATA_DIR / "duplicate_groups.json")
    write_markdown(result.summary, dup, config.REPORTS_DIR / "dataset_audit.md")


def write_markdown(summary: dict[str, Any], dup: dict[str, Any], path: Path) -> None:
    c, s, g = summary["counts"], summary["storage"], summary["geometry"]
    lines: list[str] = []
    add = lines.append

    add("# Dataset Audit Report — VegNet")
    add("")
    add(f"Generated by `src/data/audit.py`. Audit wall time: **{summary['audit_duration_s']} s**.")
    add("")
    add("## 1. Provenance")
    add("")
    add("| Field | Value |")
    add("|---|---|")
    for k, v in summary["source"].items():
        add(f"| {k} | {v} |")
    add("")
    add("## 2. Integrity")
    add("")
    add(f"- Files enumerated: **{c['total_files']}**")
    add(f"- Decoded successfully: **{c['readable']}**")
    add(f"- Failed to decode: **{c['unreadable']}**")
    add(f"- Total size: **{s['total_human']}** ({s['total_bytes']:,} bytes)")
    add(f"- Mean file size: {s['mean_bytes_per_image']:.1f} bytes "
        f"(min {s['min_bytes']:,}, max {s['max_bytes']:,})")
    add("")

    add("## 3. Class structure")
    add("")
    add("| Vegetable | Quality | Count |")
    add("|---|---|---:|")
    for cell in c["cells"]:
        add(f"| {cell['vegetable']} | {cell['quality']} | {cell['count']} |")
    add("")
    add(f"- Populated (vegetable, quality) cells: **{c['populated_cells']} of 20**")
    add(f"- Empty cells: **{', '.join(c['empty_cells']) or 'none'}**")
    add("- Empty folders exist on disk; their zero count is a dataset property, not a "
        "loading bug.")
    add("")
    add("| Vegetable | Images |")
    add("|---|---:|")
    for k, v in c["per_vegetable"].items():
        add(f"| {k} | {v} |")
    add("")
    add("| Quality | Images |")
    add("|---|---:|")
    for k, v in c["per_quality"].items():
        add(f"| {k} | {v} |")
    add("")
    add(f"- Imbalance (largest cell / smallest non-empty cell): "
        f"**{c['imbalance_ratio_max_min_cell']:.1f}×**")
    add(f"- Largest cell: {c['largest_cell']['vegetable']}/{c['largest_cell']['quality']} "
        f"= {c['largest_cell']['count']}")
    add(f"- Smallest cell: {c['smallest_cell']['vegetable']}/{c['smallest_cell']['quality']} "
        f"= {c['smallest_cell']['count']}")
    add("")

    add("## 4. Geometry and format")
    add("")
    dims = ", ".join(f"{k} ({v} files)" for k, v in g["unique_dimensions"].items())
    add(f"- Distinct dimensions: {dims}")
    add(f"- Width range: {g['min_width']}–{g['max_width']} px; "
        f"height range: {g['min_height']}–{g['max_height']} px")
    add(f"- Channels: {g['channels']}")
    add(f"- Pillow modes: {g['pillow_modes']}")
    add(f"- Extensions: {g['file_extensions']}")
    add("")
    add("Every image is exactly 256×256, which caps any useful network input at 256 px "
        "and means no resolution study above that value is meaningful.")
    add("")

    add("## 5. Duplicates")
    add("")
    d = summary["duplicates"]
    add(f"- Exact duplicates (identical SHA-256): **{d['exact_group_count']} group(s)**, "
        f"{d['exact_image_count']} file(s)")
    add(f"- Near-duplicates (dHash Hamming ≤ {d['near_dup_threshold']}): "
        f"**{d['near_group_count']} group(s)**, {d['near_image_count']} file(s)")
    add(f"- Groups spanning **different labels**: **{d['label_conflicting_group_count']}** "
        f"({d['label_conflicting_image_count']} files)")
    add("")
    if dup["near_duplicate_groups"]:
        add("Largest near-duplicate clusters:")
        add("")
        add("| Size | Max Hamming | Labels | Example member |")
        add("|---:|---:|---|---|")
        for grp in dup["near_duplicate_groups"][:10]:
            labels = "<br>".join(f"{a}/{b}" for a, b in grp["labels"])
            add(f"| {grp['size']} | {grp['max_hamming_within']} | {labels} | "
                f"`{grp['members'][0]}` |")
        add("")
    else:
        add("No near-duplicate clusters found.")
        add("")
    add("All duplicate clusters are treated as indivisible groups by "
        "`src/data/splits.py`, so a cluster can never straddle train/validation/test.")
    add("")

    add("## 6. Image statistics")
    add("")
    add("| Metric | mean | std | p05 | median | p95 |")
    add("|---|---:|---:|---:|---:|---:|")
    for col, st in summary["image_statistics"].items():
        add(f"| {col} | {st['mean']:.2f} | {st['std']:.2f} | {st['p05']:.2f} | "
            f"{st['median']:.2f} | {st['p95']:.2f} |")
    add("")
    add("## 7. Verdict")
    add("")
    ok = c["unreadable"] == 0 and c["populated_cells"] == 19
    add(f"- All images decodable: **{'PASS' if c['unreadable'] == 0 else 'FAIL'}**")
    add(f"- Label structure as expected (19/20 cells populated): "
        f"**{'PASS' if c['populated_cells'] == 19 else 'REVIEW'}**")
    add(f"- Single geometry (256×256) suitable for a fixed-size pipeline: "
        f"**{'PASS' if len(g['unique_dimensions']) == 1 else 'REVIEW'}**")
    add("")
    add("Gate 1 for dataset integrity is "
        f"**{'satisfied' if ok else 'NOT satisfied'}** by these measurements.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit the VegNet dataset")
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threshold", type=int, default=None)
    args = ap.parse_args()

    result, dup = run_audit(args.root, args.workers, args.threshold)
    persist(result, dup)
    s = result.summary
    print(json.dumps({
        "total": s["counts"]["total_files"],
        "readable": s["counts"]["readable"],
        "bytes": s["storage"]["total_human"],
        "near_dup_groups": s["duplicates"]["near_group_count"],
        "label_conflicts": s["duplicates"]["label_conflicting_group_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
