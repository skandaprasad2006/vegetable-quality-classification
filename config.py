"""Central, single-source-of-truth configuration for the project.

Every path is derived from the location of this file, so the project is
relocatable and no script needs a hard-coded absolute path.

Nothing in this module performs I/O at import time.  ``ensure_project_dirs()``
is the only function that creates directories, and it is never called
implicitly.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent

PRISTINE_DIR = PROJECT_ROOT / "Dataset"          # never modified after download
RAW_ROOT = PROJECT_ROOT / "datasets" / "raw" / "vegnet"
RAW_DIR = RAW_ROOT / "New VegNet"                # verified working copy

DOWNLOAD_DIR = PROJECT_ROOT / "datasets" / "downloads"
INTERIM_DIR = PROJECT_ROOT / "datasets" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"
METADATA_DIR = PROJECT_ROOT / "datasets" / "metadata"

NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"

MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = MODELS_DIR / "checkpoints"
FINAL_MODEL_DIR = MODELS_DIR / "final"
MODEL_METADATA_DIR = MODELS_DIR / "model_metadata"

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
METRICS_DIR = EVALUATION_DIR / "metrics"
CONFUSION_DIR = EVALUATION_DIR / "confusion_matrices"
GRAPHS_DIR = EVALUATION_DIR / "graphs"
EVAL_REPORT_DIR = EVALUATION_DIR / "reports"

WEBSITE_DIR = PROJECT_ROOT / "website"
REPORTS_DIR = PROJECT_ROOT / "reports"
TESTS_DIR = PROJECT_ROOT / "tests"
ADMIN_DIR = PROJECT_ROOT / "Admin"


def ensure_project_dirs() -> None:
    """Create every directory the pipeline writes to.  Idempotent."""
    for d in (
        DOWNLOAD_DIR, RAW_ROOT, INTERIM_DIR, PROCESSED_DIR, METADATA_DIR,
        NOTEBOOK_DIR, CHECKPOINT_DIR, FINAL_MODEL_DIR, MODEL_METADATA_DIR,
        METRICS_DIR, CONFUSION_DIR, GRAPHS_DIR, EVAL_REPORT_DIR,
        REPORTS_DIR, ADMIN_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (PROCESSED_DIR / split).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------
# The raw directory names carry an ordinal prefix ("1. Bell Pepper").  These
# two dicts are the ONLY place that mapping lives.
VEGETABLE_DIRS: dict[str, str] = {
    "1. Bell Pepper": "Bell Pepper",
    "2. Chile Pepper": "Chile Pepper",
    "3. New Mexico Green Chile": "New Mexico Green Chile",
    "4. Tomato": "Tomato",
}

# Explicit, alphabetically sorted, frozen order.  Model output indices refer to
# this order; changing it invalidates every saved model.
VEGETABLE_CLASSES: tuple[str, ...] = (
    "Bell Pepper",
    "Chile Pepper",
    "New Mexico Green Chile",
    "Tomato",
)

QUALITY_CLASSES: tuple[str, ...] = (
    "Damaged",
    "Dried",
    "Old",
    "Ripe",
    "Unripe",
)

NUM_VEGETABLE_CLASSES = len(VEGETABLE_CLASSES)
NUM_QUALITY_CLASSES = len(QUALITY_CLASSES)

#: (vegetable, quality) cells that contain images.  ``Tomato/Dried`` is empty in
#: this release of the dataset and is therefore excluded from the joint space.
JOINT_CLASSES: tuple[tuple[str, str], ...] = tuple(
    (v, q) for v in VEGETABLE_CLASSES for q in QUALITY_CLASSES if not (v == "Tomato" and q == "Dried")
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
SEED = 42
SPLIT_RATIOS: dict[str, float] = {"train": 0.70, "validation": 0.15, "test": 0.15}
SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: Maximum Hamming distance between 64-bit dHashes for two images to be treated
#: as near-duplicates (and therefore forced into the same split).
DHASH_NEAR_DUP_THRESHOLD = 6

# ---------------------------------------------------------------------------
# Model input
# ---------------------------------------------------------------------------
# The source images are exactly 256x256 (see Admin/decisions.md D-004), so a
# larger network input cannot add information.
SOURCE_IMAGE_SIZE = 256
IMAGE_SIZE = 224
IMAGE_CHANNELS = 3

# ImageNet statistics: required by MobileNetV2 / ResNet50 Keras weights.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ---------------------------------------------------------------------------
# Preprocessing defaults
# ---------------------------------------------------------------------------
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID = (8, 8)
GAUSSIAN_KSIZE = (3, 3)
MEDIAN_KSIZE = 3

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def set_global_seed(seed: int = SEED, include_tf: bool = True) -> None:
    """Seed every RNG this project uses.

    ``include_tf=False`` keeps the cheap path (tests, data tooling) from paying
    the ~3 s cost of importing TensorFlow.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency
        pass
    if include_tf:
        try:
            import tensorflow as tf

            tf.random.set_seed(seed)
            tf.keras.utils.set_random_seed(seed)
        except Exception:  # pragma: no cover - TF optional for data-only paths
            pass


def class_map() -> dict[str, object]:
    """The label-space description persisted next to every trained model."""
    return {
        "vegetable_classes": list(VEGETABLE_CLASSES),
        "quality_classes": list(QUALITY_CLASSES),
        "joint_classes": [list(c) for c in JOINT_CLASSES],
        "image_size": IMAGE_SIZE,
        "source_image_size": SOURCE_IMAGE_SIZE,
        "seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "imagenet_mean": list(IMAGENET_MEAN),
        "imagenet_std": list(IMAGENET_STD),
    }
