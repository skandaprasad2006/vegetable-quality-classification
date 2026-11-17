"""Dataset integrity, audit consistency and split-leakage tests.

These tests read the artefacts produced by ``src.data.audit`` and
``src.data.splits``.  They are the executable form of the Phase 1 gate: if any of
them fails, model training must not begin.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import config
from src.data.audit import enumerate_images
from src.data.splits import build_splits, verify_splits
from src.utils import load_json

pytestmark = pytest.mark.skipif(
    not (config.METADATA_DIR / "image_inventory.csv").exists(),
    reason="run `python -m src.data.audit` first",
)


@pytest.fixture(scope="module")
def inventory() -> pd.DataFrame:
    return pd.read_csv(config.METADATA_DIR / "image_inventory.csv")


@pytest.fixture(scope="module")
def summary() -> dict:
    return load_json(config.METADATA_DIR / "audit_summary.json")


@pytest.fixture(scope="module")
def splits() -> pd.DataFrame:
    return pd.read_csv(config.METADATA_DIR / "splits.csv")


# ---------------------------------------------------------------------------
# Acquisition
# ---------------------------------------------------------------------------


def test_raw_dataset_is_present() -> None:
    assert config.RAW_DIR.is_dir(), f"missing raw dataset at {config.RAW_DIR}"
    files = list((config.RAW_DIR).rglob("*.jpg"))
    assert len(files) == 6150, f"expected 6150 images, found {len(files)}"


def test_pristine_download_is_untouched() -> None:
    """The original download must still be present and readable."""
    assert config.PRISTINE_DIR.is_dir()
    assert len(list(config.PRISTINE_DIR.rglob("*.jpg"))) == 6150


def test_enumerate_rejects_unknown_layout(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        enumerate_images(tmp_path / "does-not-exist")
    (tmp_path / "surprise").mkdir()
    with pytest.raises(ValueError):
        enumerate_images(tmp_path)


def test_enumerate_rejects_unknown_class_folder(tmp_path: Path) -> None:
    veg = tmp_path / "1. Bell Pepper"
    (veg / "Mystery").mkdir(parents=True)
    (veg / "Mystery" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    with pytest.raises(ValueError, match="unexpected directories"):
        enumerate_images(tmp_path)


# ---------------------------------------------------------------------------
# Audit integrity
# ---------------------------------------------------------------------------


def test_every_image_is_listed(inventory: pd.DataFrame) -> None:
    on_disk = list(config.RAW_DIR.rglob("*.jpg"))
    assert len(inventory) == len(on_disk) == 6150
    assert inventory["rel_path"].is_unique


def test_no_corrupt_images(inventory: pd.DataFrame, summary: dict) -> None:
    assert inventory["readable"].all(), (
        "unreadable files: " + ", ".join(inventory.loc[~inventory["readable"], "rel_path"].head())
    )
    assert summary["counts"]["unreadable"] == 0
    assert summary["counts"]["readable"] == summary["counts"]["total_files"] == 6150


def test_all_hashes_and_dimensions_recorded(inventory: pd.DataFrame) -> None:
    assert (inventory["sha256"].str.len() == 64).all()
    assert (inventory["dhash_hex"].str.len() == 16).all()
    assert (inventory["width"] == 256).all()
    assert (inventory["height"] == 256).all()


def test_label_counts_match_summary(inventory: pd.DataFrame, summary: dict) -> None:
    per_veg = inventory["vegetable"].value_counts().to_dict()
    assert per_veg == summary["counts"]["per_vegetable"]
    per_qual = inventory["quality"].value_counts().to_dict()
    assert per_qual == summary["counts"]["per_quality"]
    assert sum(per_veg.values()) == 6150


def test_known_dataset_composition(inventory: pd.DataFrame) -> None:
    """Pin the composition that the report quotes, so silent changes fail loudly."""
    assert inventory["vegetable"].nunique() == 4
    assert inventory["quality"].nunique() == 5
    assert sorted(inventory["vegetable"].unique()) == sorted(config.VEGETABLE_CLASSES)
    assert sorted(inventory["quality"].unique()) == sorted(config.QUALITY_CLASSES)

    cells = inventory.groupby(["vegetable", "quality"]).size()
    # Documented dataset limitation: this cell is empty in this release.
    assert cells.get(("Tomato", "Dried"), 0) == 0
    assert len(cells) == 19
    assert cells.max() == 1234 and cells.min() == 27


def test_no_byte_identical_duplicates(inventory: pd.DataFrame) -> None:
    assert inventory["sha256"].nunique() == len(inventory)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def test_splits_cover_each_image_exactly_once(splits: pd.DataFrame, inventory: pd.DataFrame) -> None:
    assert len(splits) == 6150
    assert splits["rel_path"].is_unique
    assert set(splits["rel_path"]) == set(inventory["rel_path"])
    assert set(splits["split"]) == set(config.SPLITS)


def test_no_group_spans_splits(splits: pd.DataFrame) -> None:
    per_group = splits.groupby("group_id")["split"].nunique()
    leaking = per_group[per_group > 1]
    assert leaking.empty, f"{len(leaking)} duplicate groups cross splits"


def test_no_identical_image_spans_splits(splits: pd.DataFrame) -> None:
    per_hash = splits.groupby("sha256")["split"].nunique()
    assert (per_hash == 1).all()


def test_split_fractions_are_respected(splits: pd.DataFrame) -> None:
    fractions = splits["split"].value_counts(normalize=True)
    assert abs(fractions["train"] - 0.70) < 0.02
    assert abs(fractions["validation"] - 0.15) < 0.02
    assert abs(fractions["test"] - 0.15) < 0.02


def test_every_split_contains_every_class(splits: pd.DataFrame) -> None:
    for split in config.SPLITS:
        part = splits[splits["split"] == split]
        assert set(part["quality"]) == set(config.QUALITY_CLASSES), f"{split} misses a quality class"
        assert set(part["vegetable"]) == set(config.VEGETABLE_CLASSES), f"{split} misses a vegetable"


def test_every_cell_has_test_coverage(splits: pd.DataFrame) -> None:
    """Each populated cell must appear in at least train and test."""
    for (veg, qual), part in splits.groupby(["vegetable", "quality"]):
        present = set(part["split"])
        assert "train" in present, f"{veg}/{qual} absent from train"
        assert "test" in present, f"{veg}/{qual} absent from test"


def test_split_is_reproducible(inventory: pd.DataFrame) -> None:
    """Same seed and inputs must give an identical assignment."""
    dup = load_json(config.METADATA_DIR / "duplicate_groups.json")
    a = build_splits(inventory, dup, seed=config.SEED)
    b = build_splits(inventory, dup, seed=config.SEED)
    assert a["split"].tolist() == b["split"].tolist()

    c = build_splits(inventory, dup, seed=config.SEED + 1)
    assert a["split"].tolist() != c["split"].tolist()


def test_verify_splits_reports_no_leakage(splits: pd.DataFrame) -> None:
    checks = verify_splits(splits)
    assert checks["all_groups_intact"]
    assert checks["no_exact_duplicate_across_splits"]
    assert checks["groups_spanning_splits"] == 0
    assert checks["sha256_spanning_splits"] == 0
    assert checks["empty_splits"] == []


def test_processed_tree_matches_manifest(splits: pd.DataFrame) -> None:
    if not (config.PROCESSED_DIR / "train").exists():
        pytest.skip("processed images not materialised")
    for split in config.SPLITS:
        expected = int((splits["split"] == split).sum())
        found = len(list((config.PROCESSED_DIR / split).rglob("*.jpg")))
        assert found == expected, f"{split}: {found} files on disk, {expected} in manifest"


# ---------------------------------------------------------------------------
# Label-space metadata
# ---------------------------------------------------------------------------


def test_class_map_matches_config() -> None:
    path = config.METADATA_DIR / "class_map.json"
    if not path.exists():
        pytest.skip("class_map.json not written yet")
    cm = load_json(path)
    assert cm["vegetable_classes"] == list(config.VEGETABLE_CLASSES)
    assert cm["quality_classes"] == list(config.QUALITY_CLASSES)
    assert len(cm["joint_classes"]) == 19
    assert ["Tomato", "Dried"] not in cm["joint_classes"]
