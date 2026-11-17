"""Verify that the working copy is byte-identical to the pristine download.

The project promises never to modify the original dataset.  This script proves
it by hashing both trees and comparing, rather than trusting that ``cp -r`` did
what it was told.

The pristine download carries an extra wrapper directory
(``Dataset/VegNet (...)/New VegNet/``) that the working copy does not, so both
trees are first normalised to the directory that directly holds the vegetable
folders; otherwise every relative path would differ and the comparison would be
meaningless.

Writes ``datasets/metadata/pristine_copy_check.json`` and the full SHA-256
manifest of the pristine tree to ``datasets/metadata/sha256_manifest.csv``.

Run: ``python -m src.data.verify_raw_copy``
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd

import config
from src.utils import fmt_bytes, save_json, sha256_file


def find_image_root(root: Path, max_depth: int = 4) -> Path:
    """Return the directory that directly contains the vegetable folders."""
    root = Path(root)
    expected = set(config.VEGETABLE_DIRS)

    candidates: list[Path] = [root]
    for depth in range(1, max_depth + 1):
        for parts in product("*", repeat=depth):
            candidates.extend(p for p in root.glob("/".join(parts)) if p.is_dir())

    for candidate in candidates:
        try:
            children = {p.name for p in candidate.iterdir() if p.is_dir()}
        except OSError:
            continue
        if expected <= children:
            return candidate

    raise FileNotFoundError(
        f"no directory under {root} contains the expected vegetable folders {sorted(expected)}"
    )


def _hash_tree(root: Path, workers: int = 8) -> dict[str, tuple[str, int]]:
    """Map ``relative posix path -> (sha256, size)`` for every image under ``root``."""
    files = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in config.IMAGE_EXTENSIONS
    )

    def one(p: Path) -> tuple[str, tuple[str, int]]:
        return p.relative_to(root).as_posix(), (sha256_file(p), p.stat().st_size)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return dict(pool.map(one, files))


def run(pristine: Path | None = None, working: Path | None = None) -> dict:
    pristine_root = find_image_root(Path(pristine or config.PRISTINE_DIR))
    working_root = find_image_root(Path(working or config.RAW_DIR))

    print(f"pristine image root: {pristine_root}")
    a = _hash_tree(pristine_root)
    print(f"working image root : {working_root}")
    b = _hash_tree(working_root)

    only_pristine = sorted(set(a) - set(b))
    only_working = sorted(set(b) - set(a))
    differing = sorted(k for k in set(a) & set(b) if a[k][0] != b[k][0])

    total_bytes = sum(v[1] for v in a.values())
    report = {
        "pristine_root": str(pristine_root),
        "working_root": str(working_root),
        "files_pristine": len(a),
        "files_working": len(b),
        "total_bytes_pristine": total_bytes,
        "total_human": fmt_bytes(total_bytes),
        "missing_from_working": only_pristine[:50],
        "extra_in_working": only_working[:50],
        "content_mismatches": differing[:50],
        "n_missing_from_working": len(only_pristine),
        "n_extra_in_working": len(only_working),
        "n_content_mismatches": len(differing),
        "identical": not (only_pristine or only_working or differing),
    }

    config.ensure_project_dirs()
    save_json(report, config.METADATA_DIR / "pristine_copy_check.json")
    pd.DataFrame(
        [{"rel_path": k, "sha256": v[0], "bytes": v[1]} for k, v in sorted(a.items())]
    ).to_csv(config.METADATA_DIR / "sha256_manifest.csv", index=False)

    print(json.dumps({k: report[k] for k in
                      ("files_pristine", "files_working", "total_human",
                       "n_missing_from_working", "n_extra_in_working",
                       "n_content_mismatches", "identical")}, indent=2))
    return report


def main() -> None:
    report = run()
    if not report["identical"]:
        raise SystemExit("working copy differs from the pristine download")


if __name__ == "__main__":
    main()
