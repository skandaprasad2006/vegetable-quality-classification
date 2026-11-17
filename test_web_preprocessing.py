"""Preprocessing must reproduce the training-time feature extraction exactly."""

from __future__ import annotations

import numpy as np
import pytest

from src.features.handcrafted import ALL_FEATURES
from website.utils.image_processor import Preprocessed, overlay, preprocess
from website.utils.paths import FEATURE_IMAGE_SIZE


@pytest.fixture
def photo() -> np.ndarray:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 255, (300, 420, 3), dtype=np.uint8)
    # A bright blob on a plain background, so segmentation has something to find.
    img[80:260, 120:300] = (70, 150, 60)
    return img


def test_output_is_preprocessed_dataclass(photo):
    pre = preprocess(photo)
    assert isinstance(pre, Preprocessed)


def test_resize_dimensions(photo):
    pre = preprocess(photo)
    assert pre.resized_rgb.shape == (FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE, 3)


def test_original_preserved_untouched(photo):
    pre = preprocess(photo)
    assert pre.original_rgb.shape == photo.shape
    assert np.array_equal(pre.original_rgb, photo)


def test_channel_order_is_rgb(photo):
    """A red-dominant image must stay red-dominant: no BGR/RGB swap."""
    red = np.zeros((200, 200, 3), dtype=np.uint8)
    red[..., 0] = 220
    red[..., 1] = 30
    pre = preprocess(red)
    r, g, b = pre.resized_rgb[..., 0].mean(), pre.resized_rgb[..., 1].mean(), \
        pre.resized_rgb[..., 2].mean()
    assert r > g and r > b


def test_feature_count_and_dtype(photo):
    pre = preprocess(photo)
    assert pre.feature_row.shape == (1, len(ALL_FEATURES))
    assert pre.feature_row.dtype == np.float32


def test_feature_order_matches_canonical_order(photo):
    pre = preprocess(photo)
    assert tuple(pre.feature_values.keys()) == ALL_FEATURES


def test_features_finite(photo):
    pre = preprocess(photo)
    assert np.all(np.isfinite(pre.feature_row))


def test_feature_values_match_row(photo):
    """The displayed values must be the values the model receives."""
    pre = preprocess(photo)
    for i, name in enumerate(ALL_FEATURES):
        assert pre.feature_row[0, i] == pytest.approx(pre.feature_values[name], abs=1e-6)


def test_overlay_shape_and_dtype(photo):
    pre = preprocess(photo)
    assert pre.overlay_rgb.shape == pre.resized_rgb.shape
    assert pre.overlay_rgb.dtype == np.uint8


def test_overlay_without_mask_is_copy():
    img = np.full((256, 256, 3), 120, dtype=np.uint8)
    out = overlay(img, None)
    assert np.array_equal(out, img)
    assert out is not img


def test_segmentation_reports_area(photo):
    pre = preprocess(photo)
    assert 0.0 <= pre.segmentation["area_fraction"] <= 1.0
    assert pre.segmentation["mask_pixels"] >= 0
    # A uint8 mask summed rather than counted reports 255x this bound.
    assert pre.segmentation["mask_pixels"] <= FEATURE_IMAGE_SIZE ** 2


def test_uniform_image_does_not_crash():
    """A flat image has no foreground; the pipeline must still return features."""
    flat = np.full((256, 256, 3), 200, dtype=np.uint8)
    pre = preprocess(flat)
    assert pre.feature_row.shape == (1, len(ALL_FEATURES))
    assert np.all(np.isfinite(pre.feature_row))


def test_preprocess_rejects_bad_input():
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        preprocess(np.zeros((10, 10, 3), dtype=np.uint8)[..., :2])


def test_no_normalisation_applied(photo):
    """Deployed preprocessing must not rescale pixels — features use raw 0–255."""
    pre = preprocess(photo)
    assert pre.resized_rgb.dtype == np.uint8
    assert pre.resized_rgb.max() > 1
