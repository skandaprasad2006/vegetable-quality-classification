"""Handcrafted feature extraction: colour, texture, shape.

These features exist to answer a specific question in the report — *how much of
this task is solvable with classical IPCV descriptors, without a CNN?*  They are
therefore extracted and screened (see :mod:`src.features.screening`) rather than
silently fed into a model.

Feature groups
--------------
``colour``  RGB moments, HSV statistics (hue handled circularly), hue histogram,
            Hasler–Süsstrunk colourfulness, excess-green / excess-red indices.
``texture`` GLCM Haralick descriptors at three distances, plus uniform-LBP
            histograms at two radii.
``shape``   Morphology of the segmented foreground: area, circularity, solidity,
            aspect ratio, extent, equivalent diameter.

Texture is measured on the **mask bounding box** when a mask is supplied.  Filling
the background with a constant instead would manufacture a strong artificial edge
and inflate GLCM contrast.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

HUE_BINS = 12
GLCM_DISTANCES = (1, 2, 3)
GLCM_ANGLES = (0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4)
GLCM_LEVELS = 32
LBP_SETTINGS = ((8, 1), (16, 2))

GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")

COLOUR_FEATURES = (
    [f"rgb_{c}_{s}" for c in "rgb" for s in ("mean", "std", "skew")]
    + ["hue_cos_mean", "hue_sin_mean", "sat_mean", "sat_std", "val_mean", "val_std"]
    + [f"hue_hist_{i}" for i in range(HUE_BINS)]
    + ["colourfulness", "excess_green", "excess_red"]
)

TEXTURE_FEATURES = (
    [f"glcm_{p}_d{d}" for p in GLCM_PROPS for d in GLCM_DISTANCES]
    + [f"lbp{p}_{i}" for p, _ in LBP_SETTINGS for i in range(p + 2)]
)

SHAPE_FEATURES = (
    "shape_area_frac", "shape_perimeter_norm", "shape_circularity",
    "shape_aspect_ratio", "shape_solidity", "shape_extent",
    "shape_equiv_diameter_norm",
)

ALL_FEATURES = tuple(COLOUR_FEATURES) + tuple(TEXTURE_FEATURES) + tuple(SHAPE_FEATURES)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "colour": tuple(COLOUR_FEATURES),
    "texture": tuple(TEXTURE_FEATURES),
    "shape": tuple(SHAPE_FEATURES),
}


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def _skew(channel: np.ndarray) -> float:
    c = channel.astype(np.float32)
    std = c.std()
    if std < 1e-6:
        return 0.0
    return float((((c - c.mean()) / std) ** 3).mean())


def _colourfulness(rgb: np.ndarray) -> float:
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    rg, yb = r - g, 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2)
                 + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def colour_features(rgb: np.ndarray) -> dict[str, float]:
    """Colour descriptors of an RGB uint8 image.

    Hue is a circular quantity, so its "mean" is reported as the mean sine and
    cosine of the hue angle; averaging hue degrees directly would place red at
    both 0° and 180° and average to cyan.
    """
    out: dict[str, float] = {}
    for i, ch in enumerate("rgb"):
        c = rgb[..., i].astype(np.float32)
        out[f"rgb_{ch}_mean"] = float(c.mean() / 255.0)
        out[f"rgb_{ch}_std"] = float(c.std() / 255.0)
        out[f"rgb_{ch}_skew"] = _skew(c)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = hsv[..., 0].astype(np.float32) * (2.0 * np.pi / 180.0)
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)

    out["hue_cos_mean"] = float(np.cos(h).mean())
    out["hue_sin_mean"] = float(np.sin(h).mean())
    out["sat_mean"] = float(s.mean() / 255.0)
    out["sat_std"] = float(s.std() / 255.0)
    out["val_mean"] = float(v.mean() / 255.0)
    out["val_std"] = float(v.std() / 255.0)

    hist = cv2.calcHist([hsv], [0], None, [HUE_BINS], [0, 180]).ravel().astype(np.float64)
    total = hist.sum()
    hist = hist / total if total > 0 else np.zeros(HUE_BINS)
    for i, val in enumerate(hist):
        out[f"hue_hist_{i}"] = float(val)

    out["colourfulness"] = _colourfulness(rgb) / 100.0
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    out["excess_green"] = float((2 * g - r - b).mean() / 255.0)
    out["excess_red"] = float((2 * r - g - b).mean() / 255.0)
    return out


# ---------------------------------------------------------------------------
# Texture
# ---------------------------------------------------------------------------


def glcm_features(gray: np.ndarray) -> dict[str, float]:
    """Haralick descriptors from a symmetric, normalised GLCM.

    The image is quantised to :data:`GLCM_LEVELS` levels to keep the co-occurrence
    matrix small (32² per distance/angle); this is standard practice and costs
    little discriminative power while keeping extraction to a few milliseconds.
    """
    from skimage.feature import graycoprops, graycomatrix

    q = np.clip(gray.astype(np.float32) / 256.0 * GLCM_LEVELS, 0, GLCM_LEVELS - 1).astype(np.uint8)
    matrix = graycomatrix(
        q, distances=list(GLCM_DISTANCES), angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS, symmetric=True, normed=True,
    )
    out: dict[str, float] = {}
    for prop in GLCM_PROPS:
        vals = graycoprops(matrix, prop)  # (n_distances, n_angles)
        for di, d in enumerate(GLCM_DISTANCES):
            out[f"glcm_{prop}_d{d}"] = float(np.nan_to_num(vals[di]).mean())
    return out


def lbp_features(gray: np.ndarray) -> dict[str, float]:
    """Normalised histograms of uniform local binary patterns.

    Uniform LBP is used because of its rotation-invariant binning and because the
    histogram is cheap and illumination-robust.
    """
    from skimage.feature import local_binary_pattern

    out: dict[str, float] = {}
    for p, r in LBP_SETTINGS:
        codes = local_binary_pattern(gray, p, r, method="uniform")
        n_bins = p + 2
        hist, _ = np.histogram(codes, bins=n_bins, range=(0, n_bins))
        hist = hist.astype(np.float64)
        total = hist.sum()
        hist = hist / total if total > 0 else np.zeros(n_bins)
        for i, val in enumerate(hist):
            out[f"lbp{p}_{i}"] = float(val)
    return out


def texture_features(gray: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """GLCM + LBP on the foreground region (mask bounding box when available)."""
    if mask is not None:
        ys, xs = np.nonzero(mask)
        if ys.size > 8:
            gray = gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if gray.shape[0] < 8 or gray.shape[1] < 8:
        return {**{k: 0.0 for k in TEXTURE_FEATURES}}
    out = glcm_features(gray)
    out.update(lbp_features(gray))
    return out


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def shape_features(mask: np.ndarray) -> dict[str, float]:
    """Morphological descriptors of the largest foreground contour.

    Returns zeros when the mask is empty, which is the honest encoding of
    "no foreground found" for a feature vector.
    """
    zeros = {k: 0.0 for k in SHAPE_FEATURES}
    if mask is None:
        return zeros

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return zeros
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area <= 0:
        return zeros

    perimeter = float(cv2.arcLength(contour, True))
    x, y, w, h = cv2.boundingRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    h_img, w_img = mask.shape[:2]

    return {
        "shape_area_frac": area / float(h_img * w_img),
        "shape_perimeter_norm": perimeter / float(2 * (h_img + w_img)),
        "shape_circularity": float(4 * np.pi * area / (perimeter ** 2 + 1e-9)),
        "shape_aspect_ratio": float(max(w, h) / (min(w, h) + 1e-9)),
        "shape_solidity": float(area / (hull_area + 1e-9)),
        "shape_extent": float(area / (w * h + 1e-9)),
        "shape_equiv_diameter_norm": float(np.sqrt(4 * area / np.pi) / max(h_img, w_img)),
    }


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------


def extract_features(rgb: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """All handcrafted features for one RGB uint8 image."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    feats: dict[str, float] = {}
    feats.update(colour_features(rgb))
    feats.update(texture_features(gray, mask))
    feats.update(shape_features(mask))
    return feats


def extract_feature_matrix(images: list[np.ndarray],
                           masks: list[np.ndarray] | None = None) -> tuple[np.ndarray, tuple[str, ...]]:
    """Stack :func:`extract_features` over a list of images.

    The column order is :data:`ALL_FEATURES`; the first successful extraction is
    used to assert that order matches, so a drifted feature set fails loudly.
    """
    rows: list[dict[str, float]] = []
    for i, img in enumerate(images):
        m = masks[i] if masks is not None else None
        rows.append(extract_features(img, m))
    if not rows:
        return np.zeros((0, len(ALL_FEATURES)), dtype=np.float32), ALL_FEATURES

    key_order = tuple(rows[0].keys())
    if key_order != ALL_FEATURES:
        missing = set(ALL_FEATURES) - set(key_order)
        extra = set(key_order) - set(ALL_FEATURES)
        raise RuntimeError(f"feature set drift: missing={sorted(missing)} extra={sorted(extra)}")

    matrix = np.array([[r[k] for k in ALL_FEATURES] for r in rows], dtype=np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return matrix, ALL_FEATURES


def group_indices(feature_names: tuple[str, ...]) -> dict[str, list[int]]:
    """Map each feature group to the column indices it occupies."""
    return {
        group: [i for i, name in enumerate(feature_names) if name in members]
        for group, members in FEATURE_GROUPS.items()
    }
