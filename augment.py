"""Augmentation implemented with OpenCV + NumPy.

Why not ``tf.keras.layers.RandomFlip`` etc.?  Because Phase 1 must be runnable
and testable without TensorFlow, and because an explicit implementation makes the
geometry (one warp for rotate+zoom+translate, not three resamplings) reviewable.

Deliberate omissions
--------------------
* **No vertical flip.**  Vegetables are photographed in a consistent upright
  orientation; a vertical flip produces an image that does not occur in the
  distribution.
* **No hue rotation by default.**  Ripeness is largely a hue cue in this dataset
  (green → red).  Shifting hue would relabel the image.
* **No elastic deformation.**  With ~4.3k training images, strong geometric
  distortion increased validation loss in pilot runs and it is left out.

All randomness comes from a caller-supplied ``numpy.random.Generator``, so an
augmentation sequence is reproducible from ``(seed, call order)``.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class AugmentConfig:
    rotation_deg: float = 12.0
    zoom: float = 0.10
    translate_frac: float = 0.06
    brightness_delta: float = 0.12
    contrast_delta: float = 0.12
    hflip_p: float = 0.5
    apply_p: float = 0.9

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def rotate(image: np.ndarray, angle: float, border: int = cv2.BORDER_REFLECT_101) -> np.ndarray:
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=border)


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return cv2.flip(image, 1)


def zoom(image: np.ndarray, factor: float) -> np.ndarray:
    """Scale about the centre and crop back to the original size."""
    if factor <= 0:
        raise ValueError("zoom factor must be positive")
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), 0.0, factor)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def translate(image: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = image.shape[:2]
    m = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def adjust_brightness(image: np.ndarray, delta: float) -> np.ndarray:
    """``delta`` in [-1, 1], applied as an additive shift in 8-bit units."""
    return np.clip(image.astype(np.float32) + delta * 255.0, 0, 255).astype(np.uint8)


def adjust_contrast(image: np.ndarray, delta: float) -> np.ndarray:
    """``delta`` in [-1, 1] as a multiplicative change about mid-grey."""
    factor = 1.0 + delta
    out = (image.astype(np.float32) - 127.5) * factor + 127.5
    return np.clip(out, 0, 255).astype(np.uint8)


def augment(image: np.ndarray, rng: np.random.Generator,
            cfg: AugmentConfig | None = None) -> np.ndarray:
    """Apply a random composition of geometry + photometry to an RGB uint8 image."""
    cfg = cfg or AugmentConfig()
    if rng.random() > cfg.apply_p:
        return image.copy()

    out = image
    h, w = out.shape[:2]

    angle = float(rng.uniform(-cfg.rotation_deg, cfg.rotation_deg))
    scale = float(1.0 + rng.uniform(-cfg.zoom, cfg.zoom))
    tx = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * w)
    ty = float(rng.uniform(-cfg.translate_frac, cfg.translate_frac) * h)

    # One affine warp for all three geometric operations: a single interpolation
    # instead of three successive ones, which would compound blur.
    m = cv2.getRotationMatrix2D((w / 2 - 0.5, h / 2 - 0.5), angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    out = cv2.warpAffine(out, m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    if rng.random() < cfg.hflip_p:
        out = horizontal_flip(out)

    if cfg.brightness_delta > 0:
        out = adjust_brightness(out, float(rng.uniform(-cfg.brightness_delta, cfg.brightness_delta)))
    if cfg.contrast_delta > 0:
        out = adjust_contrast(out, float(rng.uniform(-cfg.contrast_delta, cfg.contrast_delta)))

    return out


def augment_batch(images: np.ndarray, seed: int, cfg: AugmentConfig | None = None) -> np.ndarray:
    """Augment a ``(N, H, W, 3)`` uint8 batch deterministically from ``seed``."""
    rng = np.random.default_rng(seed)
    return np.stack([augment(img, rng, cfg) for img in images], axis=0)


def augmentation_grid(image: np.ndarray, n: int = 6, seed: int = 0,
                      cfg: AugmentConfig | None = None) -> list[np.ndarray]:
    """``[original] + n augmented variants`` — used by the notebooks."""
    rng = np.random.default_rng(seed)
    return [image.copy()] + [augment(image, rng, cfg) for _ in range(n)]
