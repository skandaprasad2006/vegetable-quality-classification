"""``tf.data`` input pipeline for Phase 2.

Two dataset builders exist because the project needs both speed and fidelity:

``tf_native``
    Decode + resize + normalise using TensorFlow ops, with augmentation as Keras
    preprocessing layers.  Fast, runs in the input pipeline's own threads, and is
    what the main experiment matrix uses.

``<preset name>``
    Applies the Phase 1 OpenCV preset (``basic`` / ``enhanced`` / ``bilateral`` /
    ``segmented``) per image via ``tf.numpy_function``.  Slower, but it is the only
    way the preprocessing ablation can test the *actual* implemented pipeline
    rather than an approximation of it.

Both paths implement the same augmentation policy as
``src/preprocessing/augment.py``: rotation, zoom, translation, brightness,
contrast and horizontal flip, with no vertical flip and no hue rotation.

Labels are integer indices into ``config.VEGETABLE_CLASSES`` and
``config.QUALITY_CLASSES``.  The same index order is written into the model
metadata, so serving cannot silently reorder classes.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

import config
from src.preprocessing.augment import AugmentConfig
from src.preprocessing.pipeline import get_preset, preprocess_image
from src.utils import imread_unicode

VEG_INDEX = {name: i for i, name in enumerate(config.VEGETABLE_CLASSES)}
QUAL_INDEX = {name: i for i, name in enumerate(config.QUALITY_CLASSES)}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def resolve_split(split: str, root: Path | None = None) -> pd.DataFrame:
    """Paths and integer labels for one split, in a stable order.

    Raises if a label is not in the frozen class order — a silent ``-1`` index
    would otherwise train a model to predict garbage.
    """
    if split not in config.SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {config.SPLITS}")

    root = Path(root or config.PROCESSED_DIR)
    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    part = manifest[manifest["split"] == split].reset_index(drop=True)

    paths = [str(root / split / r.vegetable / r.quality / r.file_name)
             for r in part.itertuples(index=False)]

    unknown_v = set(part["vegetable"]) - set(VEG_INDEX)
    unknown_q = set(part["quality"]) - set(QUAL_INDEX)
    if unknown_v or unknown_q:
        raise ValueError(f"unmapped labels: vegetables={unknown_v} qualities={unknown_q}")

    return pd.DataFrame({
        "path": paths,
        "vegetable": [VEG_INDEX[v] for v in part["vegetable"]],
        "quality": [QUAL_INDEX[q] for q in part["quality"]],
        "vegetable_name": part["vegetable"],
        "quality_name": part["quality"],
    })


# ---------------------------------------------------------------------------
# Numeric preprocessing (TensorFlow equivalent of pipeline.normalize)
# ---------------------------------------------------------------------------


def normalise_tf(x: tf.Tensor, mode: str) -> tf.Tensor:
    """Mirror of ``src.preprocessing.pipeline.normalize`` in TensorFlow ops."""
    if mode == "unit":
        return x / 255.0
    if mode == "imagenet":
        mean = tf.constant(config.IMAGENET_MEAN, dtype=tf.float32)
        std = tf.constant(config.IMAGENET_STD, dtype=tf.float32)
        return (x / 255.0 - mean) / std
    if mode == "tf":
        return x / 127.5 - 1.0
    if mode == "caffe":
        bgr = tf.reverse(x, axis=[-1])
        return bgr - tf.constant([103.939, 116.779, 123.68], dtype=tf.float32)
    raise ValueError(f"unknown normalisation mode {mode!r}")


def augmentation_layers(size: int, cfg: AugmentConfig | None = None) -> tf.keras.Sequential:
    """Keras augmentation stack matching ``AugmentConfig``'s policy."""
    cfg = cfg or AugmentConfig()
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=config.SEED),
            tf.keras.layers.RandomRotation(
                factor=cfg.rotation_deg / 360.0, fill_mode="reflect", seed=config.SEED),
            tf.keras.layers.RandomZoom(
                height_factor=cfg.zoom, width_factor=cfg.zoom,
                fill_mode="reflect", seed=config.SEED),
            tf.keras.layers.RandomTranslation(
                height_factor=cfg.translate_frac, width_factor=cfg.translate_frac,
                fill_mode="reflect", seed=config.SEED),
            tf.keras.layers.RandomBrightness(
                factor=cfg.brightness_delta, value_range=(0.0, 255.0), seed=config.SEED),
            tf.keras.layers.RandomContrast(factor=cfg.contrast_delta, seed=config.SEED),
        ],
        name="augmentation",
    )


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------


def _decode(path: tf.Tensor, size: int) -> tf.Tensor:
    raw = tf.io.read_file(path)
    img = tf.io.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, (size, size), method="bilinear")
    return tf.cast(img, tf.float32)


def _cv2_preset_map(preset: str, normalize: str = "imagenet"):
    """Wrap the Phase 1 OpenCV pipeline for use inside ``tf.data``.

    The preset's final normalisation is overridden so a preprocessing ablation
    can feed a transfer backbone the input distribution its weights expect;
    denoising and CLAHE still run in uint8 space before normalisation, so the
    image content is identical to the Phase 1 preset.
    """
    from dataclasses import replace

    cfg = replace(get_preset(preset), normalize=normalize)

    def _fn(path_bytes):
        path = path_bytes.decode("utf-8") if isinstance(path_bytes, bytes) else str(path_bytes)
        bgr = imread_unicode(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read {path}")
        return preprocess_image(bgr, cfg).astype(np.float32)

    return _fn


def make_dataset(
    split: str,
    *,
    preset: str = "tf_native",
    normalize: str = "unit",
    size: int = config.IMAGE_SIZE,
    batch_size: int = 32,
    training: bool = False,
    head: str = "single",
    augment: bool = True,
    augment_cfg: AugmentConfig | None = None,
    seed: int = config.SEED,
    shuffle_buffer: int = 2048,
    drop_remainder: bool = False,
    limit: int | None = None,
    quality_weights: dict[int, float] | None = None,
    vegetable_weights: dict[int, float] | None = None,
) -> tf.data.Dataset:
    """Build a batched dataset.

    ``preset`` of ``"tf_native"`` uses TensorFlow ops; any other value must name a
    Phase 1 preprocessing preset and is applied through OpenCV.

    ``quality_weights`` / ``vegetable_weights`` attach **per-sample** weights to the
    dataset as a third element.  This exists because Keras 3 does not accept a
    per-output ``class_weight`` dict for a multi-output model; the weights applied
    are identical to what ``class_weight`` would apply for a single head.
    """
    df = resolve_split(split)
    if limit:
        df = df.iloc[:limit].reset_index(drop=True)

    paths = df["path"].to_numpy(dtype=str)
    veg = df["vegetable"].to_numpy(dtype=np.int32)
    qual = df["quality"].to_numpy(dtype=np.int32)

    weighted = quality_weights is not None
    if weighted:
        w_q = np.array([quality_weights[int(c)] for c in qual], dtype=np.float32)
        w_v = np.array([(vegetable_weights or {}).get(int(c), 1.0) for c in veg],
                       dtype=np.float32)

    if head == "single":
        ds = (tf.data.Dataset.from_tensor_slices((paths, qual, w_q)) if weighted
              else tf.data.Dataset.from_tensor_slices((paths, qual)))
    else:
        labels = {"vegetable": veg, "quality": qual}
        ds = (tf.data.Dataset.from_tensor_slices((paths, labels, {"vegetable": w_v, "quality": w_q}))
              if weighted else tf.data.Dataset.from_tensor_slices((paths, labels)))

    if training:
        ds = ds.shuffle(min(shuffle_buffer, len(df)), seed=seed, reshuffle_each_iteration=True)

    n_threads = tf.data.AUTOTUNE

    if preset == "tf_native":
        if weighted:
            ds = ds.map(lambda p, y, w: (_decode(p, size), y, w), num_parallel_calls=n_threads)
            if training and augment:
                aug = augmentation_layers(size, augment_cfg)
                ds = ds.map(lambda x, y, w: (aug(x, training=True), y, w),
                            num_parallel_calls=n_threads)
            ds = ds.map(lambda x, y, w: (normalise_tf(x, normalize), y, w),
                        num_parallel_calls=n_threads)
        else:
            ds = ds.map(lambda p, y: (_decode(p, size), y), num_parallel_calls=n_threads)
            if training and augment:
                aug = augmentation_layers(size, augment_cfg)
                ds = ds.map(lambda x, y: (aug(x, training=True), y),
                            num_parallel_calls=n_threads)
            ds = ds.map(lambda x, y: (normalise_tf(x, normalize), y),
                        num_parallel_calls=n_threads)
    else:
        fn = _cv2_preset_map(preset, normalize)
        n_cv2 = max(1, min(8, (tf.config.threading.get_inter_op_parallelism_threads() or 4)))

        def _apply(p, y):
            arr = tf.numpy_function(fn, [p], tf.float32)
            arr.set_shape((size, size, 3))
            return arr, y

        if weighted:
            def _apply_w(p, y, w):
                return _apply(p, y) + (w,)

            ds = ds.map(_apply_w, num_parallel_calls=n_cv2)
        else:
            ds = ds.map(_apply, num_parallel_calls=n_cv2)
        # Augmentation is not applied after normalisation: Keras preprocessing
        # layers assume a 0-255 range.  Documented limitation of the OpenCV path.

    return ds.batch(batch_size, drop_remainder=drop_remainder).prefetch(tf.data.AUTOTUNE)


def dataset_summary(split: str) -> dict[str, object]:
    """Counts used by the training logs and by the experiment registry."""
    df = resolve_split(split)
    return {
        "split": split,
        "n_images": int(len(df)),
        "class_counts": {k: int(v) for k, v in df["quality_name"].value_counts().items()},
        "vegetable_counts": {k: int(v) for k, v in df["vegetable_name"].value_counts().items()},
    }


def compute_class_weights(split: str = "train") -> dict[int, float]:
    """Balanced class weights: ``n_samples / (n_classes * count_c)``.

    Computed from the **training split only**, as required for validity.
    """
    df = resolve_split(split)
    counts = df["quality"].value_counts().sort_index()
    n = len(df)
    k = len(config.QUALITY_CLASSES)
    present = counts[counts > 0]
    return {int(c): float(n / (k * v)) for c, v in present.items()}
