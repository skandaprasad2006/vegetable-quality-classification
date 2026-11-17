"""OpenCV image-processing pipeline.

Every operation here is implemented as a small, testable function so that each
one can be measured independently before being adopted.  The project does **not**
stack all of them by default: :data:`PRESETS` defines three concrete pipelines
(basic / enhanced / segmented) and Phase 2 ablates them against each other.

Conventions
-----------
* Files are read as **BGR** (OpenCV's native order) and the pipeline returns
  **RGB float32**, because Keras pretrained weights expect RGB.
* ``uint8`` is used for display stages; normalisation happens last.
* Hue in OpenCV is 0–179 (degrees / 2), which matters for histogram ranges.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np

import config


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """Declarative description of one preprocessing pipeline."""

    size: int = config.IMAGE_SIZE
    resize_mode: str = "squash"          # squash | crop | pad
    denoise: str | None = None           # None | gaussian | median | bilateral
    denoise_ksize: int = 3
    clahe: bool = False
    clahe_clip: float = config.CLAHE_CLIP_LIMIT
    clahe_tiles: tuple[int, int] = config.CLAHE_TILE_GRID
    segment: bool = False
    segment_fill: int = 255              # neutral fill for removed background
    normalize: str = "imagenet"          # none | unit | imagenet

    def describe(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "resize_mode": self.resize_mode,
            "denoise": self.denoise,
            "denoise_ksize": self.denoise_ksize,
            "clahe": self.clahe,
            "clahe_clip": self.clahe_clip,
            "clahe_tiles": list(self.clahe_tiles),
            "segment": self.segment,
            "normalize": self.normalize,
        }


#: Pipelines that Phase 2 compares.  ``basic`` is deliberately the cheapest
#: defensible baseline so that any gain from the others is attributable.
#: Pipelines that Phase 2 compares, fixed by the measured study in
#: ``evaluation/reports/preprocessing_study.md``.
#: ``bilateral`` is added because it measured the best noise/sharpness trade-off;
#: ``enhanced`` keeps median as the cheaper reference arm so the ablation has a
#: like-for-like denoiser comparison rather than a single promoted choice.
PRESETS: dict[str, PreprocessConfig] = {
    "basic": PreprocessConfig(),
    "enhanced": PreprocessConfig(denoise="median", denoise_ksize=3, clahe=True),
    "bilateral": PreprocessConfig(denoise="bilateral", clahe=True),
    "segmented": PreprocessConfig(
        denoise="median", denoise_ksize=3, clahe=True, segment=True,
    ),
}


def get_preset(name: str) -> PreprocessConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown preprocessing preset {name!r}; choose from {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Colour-space helpers
# ---------------------------------------------------------------------------


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def to_hsv(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def resize_squash(rgb: np.ndarray, size: int) -> np.ndarray:
    """Direct resize, ignoring aspect ratio.

    The source images are square (256×256) so this introduces no distortion on
    this dataset; it is the default for that reason.
    """
    return cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)


def resize_center_crop(rgb: np.ndarray, size: int) -> np.ndarray:
    """Resize the short side to ``size`` then centre-crop — preserves aspect."""
    h, w = rgb.shape[:2]
    scale = size / min(h, w)
    nh, nw = max(size, int(round(h * scale))), max(size, int(round(w * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    top = (nh - size) // 2
    left = (nw - size) // 2
    return resized[top:top + size, left:left + size]


def resize_pad(rgb: np.ndarray, size: int, fill: int = 0) -> np.ndarray:
    """Resize the long side to ``size`` then pad to a square.

    Used only by the segmentation pathway, where squashing would distort the
    object whose shape is being described.
    """
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, rgb.shape[2]), fill, dtype=rgb.dtype)
    top = (size - nh) // 2
    left = (size - nw) // 2
    out[top:top + nh, left:left + nw] = resized
    return out


def resize(rgb: np.ndarray, size: int, mode: str = "squash", fill: int = 0) -> np.ndarray:
    if mode == "squash":
        return resize_squash(rgb, size)
    if mode == "crop":
        return resize_center_crop(rgb, size)
    if mode == "pad":
        return resize_pad(rgb, size, fill)
    raise ValueError(f"unknown resize mode {mode!r}")


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------


def _validate_ksize(ksize: int) -> int:
    if int(ksize) < 1 or int(ksize) % 2 == 0:
        raise ValueError(f"kernel size must be a positive odd integer, got {ksize}")
    return int(ksize)


def gaussian_filter(rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Linear low-pass filter.  Cheap, but blurs edges uniformly."""
    k = _validate_ksize(ksize)
    return cv2.GaussianBlur(rgb, (k, k), 0)


def median_filter(rgb: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Non-linear filter: removes salt-and-pepper noise while keeping edges.

    Preferred over Gaussian for this dataset, whose JPEG artefacts are impulse-
    like rather than Gaussian.
    """
    k = _validate_ksize(ksize)
    return cv2.medianBlur(rgb, k)


def bilateral_filter(rgb: np.ndarray, d: int = 5, sigma_colour: float = 50.0,
                     sigma_space: float = 50.0) -> np.ndarray:
    """Edge-preserving smoothing.  ~10× slower than the other two on CPU."""
    return cv2.bilateralFilter(rgb, d, sigma_colour, sigma_space)


def denoise(rgb: np.ndarray, kind: str | None, ksize: int = 3) -> np.ndarray:
    if kind is None:
        return rgb
    if kind == "gaussian":
        return gaussian_filter(rgb, ksize)
    if kind == "median":
        return median_filter(rgb, ksize)
    if kind == "bilateral":
        return bilateral_filter(rgb)
    raise ValueError(f"unknown denoiser {kind!r}")


# ---------------------------------------------------------------------------
# Contrast enhancement
# ---------------------------------------------------------------------------


def clahe_enhance(rgb: np.ndarray, clip_limit: float = 2.0,
                  tiles: tuple[int, int] = (8, 8)) -> np.ndarray:
    """CLAHE applied to the **L channel of LAB** only.

    Equalising RGB channels independently would shift white balance and hue.
    Operating on lightness keeps chromaticity intact, which matters because
    ripeness is principally a colour cue in this dataset.
    """
    if clip_limit <= 0:
        raise ValueError("clip_limit must be > 0")
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tiles))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2RGB)


# ---------------------------------------------------------------------------
# Segmentation / background handling
# ---------------------------------------------------------------------------


def estimate_background_colour(bgr: np.ndarray, border: int = 8) -> np.ndarray:
    """Median BGR of the image border, used as the background reference."""
    h, w = bgr.shape[:2]
    b = max(1, min(border, h // 2, w // 2))
    strips = [bgr[:b, :], bgr[-b:, :], bgr[:, :b], bgr[:, -b:]]
    px = np.concatenate([s.reshape(-1, bgr.shape[2]) for s in strips], axis=0)
    return np.median(px, axis=0)


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    flood = mask.copy()
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, ff_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, holes)


def segment_vegetable(bgr: np.ndarray, border: int = 8,
                      min_area_frac: float = 0.02) -> tuple[np.ndarray, dict[str, Any]]:
    """Foreground mask by LAB distance from the border-derived background colour.

    Why LAB distance rather than a fixed HSV range: the dataset is shot on white
    backgrounds, but exposure varies between sessions, so a hard HSV threshold
    either eats highlights on pale vegetables or keeps shadow.  Distance from
    the *locally estimated* background adapts per image.

    Returns ``(mask, info)`` where ``info['ok']`` is False when the result is
    implausible (almost nothing or almost everything selected); callers should
    fall back to the unsegmented image in that case.
    """
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg_bgr = estimate_background_colour(bgr, border)
    bg_lab = cv2.cvtColor(bg_bgr.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    dist = np.linalg.norm(lab - bg_lab[None, None, :], axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n_labels > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    mask = _fill_holes(mask)
    area_frac = float(mask.mean() / 255.0)

    ys, xs = np.nonzero(mask)
    if ys.size:
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    else:
        bbox = (0, 0, bgr.shape[1] - 1, bgr.shape[0] - 1)

    ok = min_area_frac <= area_frac <= 0.98
    return mask, {
        "ok": bool(ok),
        "area_fraction": area_frac,
        "bbox": bbox,
        "background_bgr": [float(v) for v in bg_bgr],
    }


def apply_mask(rgb: np.ndarray, mask: np.ndarray, fill: int = 255) -> np.ndarray:
    """Replace everything outside ``mask`` with a neutral fill."""
    out = rgb.copy()
    out[mask == 0] = fill
    return out


def crop_to_mask(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tight crop around the mask bounding box (no resampling)."""
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return rgb
    return rgb[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize(rgb: np.ndarray, mode: str = "imagenet") -> np.ndarray:
    """Convert a uint8 RGB image to a float32 model input.

    The mode must match the expectation of whatever produced the weights:

    * ``unit``     — divide by 255 (range 0–1). Raw input for a from-scratch CNN.
    * ``imagenet`` — divide by 255 then standardise with the ImageNet mean/std.
    * ``tf``       — scale to −1…1, i.e. ``x/127.5 - 1``. This is exactly what
      ``keras.applications.mobilenet_v2.preprocess_input`` computes.
    * ``caffe``    — RGB→BGR then subtract the ImageNet BGR means, no rescaling.
      This is exactly what ``keras.applications.resnet50.preprocess_input``
      computes in Caffe mode.

    Handling this explicitly in the data pipeline (rather than with a ``Lambda``
    layer inside the model) keeps saved models serialisable without custom
    objects, and records the choice in the model metadata.
    """
    if mode in ("none", None):
        return rgb.astype(np.float32)
    if mode in ("unit", "imagenet"):
        arr = rgb.astype(np.float32) / 255.0
        if mode == "unit":
            return arr
        mean = np.asarray(config.IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(config.IMAGENET_STD, dtype=np.float32)
        return (arr - mean) / std
    if mode == "tf":
        return rgb.astype(np.float32) / 127.5 - 1.0
    if mode == "caffe":
        bgr = rgb.astype(np.float32)[..., ::-1]
        return bgr - np.array([103.939, 116.779, 123.68], dtype=np.float32)
    raise ValueError(f"unknown normalisation mode {mode!r}")


def denormalize(arr: np.ndarray, mode: str = "imagenet") -> np.ndarray:
    """Inverse of :func:`normalize`; returns uint8 RGB for display."""
    arr = np.asarray(arr, dtype=np.float32)
    if mode == "imagenet":
        mean = np.asarray(config.IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(config.IMAGENET_STD, dtype=np.float32)
        return np.clip((arr * std + mean) * 255.0, 0, 255).astype(np.uint8)
    if mode == "tf":
        return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)
    if mode == "caffe":
        rgb = arr + np.array([103.939, 116.779, 123.68], dtype=np.float32)
        return np.clip(rgb[..., ::-1], 0, 255).astype(np.uint8)
    if mode in ("unit", "none", None):
        return np.clip(arr * 255.0 if mode != "none" else arr, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown normalisation mode {mode!r}")


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------


def preprocess_with_stages(bgr: np.ndarray, cfg: PreprocessConfig | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the full pipeline and keep every intermediate for inspection.

    Returns ``(model_input, stages)`` where ``stages`` holds the ordered visual
    stages (RGB uint8) plus a ``meta`` dict.  The website renders exactly these.
    """
    cfg = cfg or PreprocessConfig()
    stages: dict[str, Any] = {}

    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError(f"expected a 3-channel BGR image, got shape {bgr.shape}")

    rgb = bgr_to_rgb(bgr)
    stages["original"] = rgb

    working = resize(rgb, cfg.size, cfg.resize_mode)
    stages["resized"] = working

    hsv = to_hsv(working)
    stages["hsv"] = hsv

    # Segmentation (and the crop it enables) is decided before enhancement so
    # that CLAHE is not computed on background pixels.
    seg_info: dict[str, Any] = {"ok": False, "area_fraction": 1.0}
    mask = None
    if cfg.segment:
        bgr_working = rgb_to_bgr(working)
        mask, seg_info = segment_vegetable(bgr_working)
        if seg_info["ok"]:
            working = crop_to_mask(working, mask)
            mask = None  # recomputed after enhancement on the cropped image
            working = resize(working, cfg.size, "pad", cfg.segment_fill)

    working = denoise(working, cfg.denoise, cfg.denoise_ksize)
    if cfg.clahe:
        working = clahe_enhance(working, cfg.clahe_clip, cfg.clahe_tiles)
    stages["enhanced"] = working

    if cfg.segment:
        bgr_working = rgb_to_bgr(working)
        mask, seg_info = segment_vegetable(bgr_working)
        stages["mask"] = mask
        if seg_info["ok"]:
            working = apply_mask(working, mask, cfg.segment_fill)
        stages["segmented"] = working

    model_input = normalize(working, cfg.normalize)
    stages["model_input"] = model_input

    meta = {
        "config": cfg.describe(),
        "segmentation": seg_info,
        "input_shape": list(model_input.shape),
        "input_dtype": str(model_input.dtype),
        "input_min": float(model_input.min()),
        "input_max": float(model_input.max()),
        "input_mean": float(model_input.mean()),
    }
    return model_input, dict(stages=stages, meta=meta)


def preprocess_image(bgr: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Model input only — the hot path used by training and serving."""
    arr, _ = preprocess_with_stages(bgr, cfg)
    return arr


def display_stages(stages: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    """Ordered ``(label, RGB uint8)`` pairs for visualisation."""
    out: list[tuple[str, np.ndarray]] = []
    for key in ("original", "resized", "hsv", "enhanced", "segmented", "mask"):
        if key not in stages:
            continue
        img = stages[key]
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        out.append((key, img))
    return out


# ---------------------------------------------------------------------------
# Quality measurements used to *decide* whether an operation is worth using
# ---------------------------------------------------------------------------


def image_quality_metrics(rgb: np.ndarray) -> dict[str, float]:
    """Quantitative descriptors of an image's condition.

    ``sharpness`` is the variance of the Laplacian (higher = more high-frequency
    detail).  ``noise_sigma`` is a robust estimate from the median absolute
    deviation of the Laplacian, following Immerkær's approach.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())

    return {
        "contrast": float(lab[..., 0].std()),
        "brightness": float(hsv[..., 2].mean()),
        "saturation": float(hsv[..., 1].mean()),
        "sharpness": float(lap.var()),
        "noise_sigma": float(np.median(np.abs(lap)) / 0.6745 / np.sqrt(6.0)),
        "entropy": entropy,
    }
