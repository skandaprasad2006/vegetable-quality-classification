"""Tests for the OpenCV pipeline, augmentation and handcrafted features.

All tests build synthetic images, so they are deterministic, fast, and do not
depend on the dataset being present.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

import config
from src.features.handcrafted import (
    ALL_FEATURES,
    FEATURE_GROUPS,
    colour_features,
    extract_feature_matrix,
    extract_features,
    glcm_features,
    lbp_features,
    shape_features,
)
from src.preprocessing.augment import (
    AugmentConfig,
    adjust_brightness,
    adjust_contrast,
    augment,
    augment_batch,
    augmentation_grid,
    horizontal_flip,
    rotate,
    translate,
    zoom,
)
from src.preprocessing.pipeline import (
    PRESETS,
    PreprocessConfig,
    bgr_to_rgb,
    clahe_enhance,
    denormalize,
    display_stages,
    estimate_background_colour,
    gaussian_filter,
    get_preset,
    image_quality_metrics,
    median_filter,
    normalize,
    preprocess_image,
    preprocess_with_stages,
    resize,
    rgb_to_bgr,
    segment_vegetable,
    to_hsv,
    to_lab,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def square_rgb() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)


@pytest.fixture
def blob_on_white() -> np.ndarray:
    """A dark elliptical object on a white background (BGR), like the dataset."""
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    cv2.ellipse(img, (128, 128), (60, 45), 15, 0, 360, (40, 60, 200), -1)
    return img


# ---------------------------------------------------------------------------
# Colour handling
# ---------------------------------------------------------------------------


def test_bgr_rgb_roundtrip(square_rgb: np.ndarray) -> None:
    bgr = rgb_to_bgr(square_rgb)
    back = bgr_to_rgb(bgr)
    assert np.array_equal(back, square_rgb)


def test_bgr_to_rgb_swaps_channels() -> None:
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 0] = 255  # blue in BGR
    rgb = bgr_to_rgb(bgr)
    assert rgb[0, 0, 2] == 255 and rgb[0, 0, 0] == 0


def test_hsv_of_pure_red() -> None:
    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0] = 255
    hsv = to_hsv(rgb)
    assert hsv[0, 0, 0] == 0            # hue of red is 0 in OpenCV
    assert hsv[0, 0, 1] == 255          # fully saturated
    assert hsv[0, 0, 2] == 255          # fully bright


def test_lab_lightness_of_grey() -> None:
    """OpenCV encodes L* in 0-255 as L*100/255; sRGB 128 is L*~57, not 128."""
    rgb = np.full((8, 8, 3), 128, np.uint8)
    lab = to_lab(rgb)
    assert 130 <= lab[0, 0, 0] <= 150
    assert abs(int(lab[0, 0, 1]) - 128) <= 1   # neutral: a*, b* at midpoint
    assert abs(int(lab[0, 0, 2]) - 128) <= 1


def test_lab_lightness_increases_with_intensity() -> None:
    levels = [to_lab(np.full((2, 2, 3), v, np.uint8))[0, 0, 0] for v in (32, 128, 240)]
    assert levels == sorted(levels) and levels[0] < levels[-1]


# ---------------------------------------------------------------------------
# Resizing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["squash", "crop", "pad"])
def test_resize_output_shape(mode: str, square_rgb: np.ndarray) -> None:
    out = resize(square_rgb, 224, mode)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_resize_crop_preserves_aspect_of_wide_image() -> None:
    wide = np.zeros((100, 200, 3), np.uint8)
    wide[:, :100] = 255
    out = resize(wide, 64, "crop")
    assert out.shape == (64, 64, 3)


def test_resize_pad_centres_content() -> None:
    tall = np.zeros((200, 100, 3), np.uint8)
    tall[...] = 200
    out = resize(tall, 64, "pad")
    assert out.shape == (64, 64, 3)
    assert out[0, 0].tolist() == [0, 0, 0]      # padding
    assert out[32, 32].tolist() == [200, 200, 200]  # content


def test_resize_rejects_unknown_mode(square_rgb: np.ndarray) -> None:
    with pytest.raises(ValueError, match="unknown resize mode"):
        resize(square_rgb, 64, "magic")


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _noisy_image() -> np.ndarray:
    rng = np.random.default_rng(7)
    base = np.full((128, 128, 3), 128, np.uint8)
    noise = rng.normal(0, 25, base.shape)
    img = np.clip(base + noise, 0, 255).astype(np.uint8)
    # salt-and-pepper
    mask = rng.random((128, 128)) < 0.02
    img[mask] = 0
    img[rng.random((128, 128)) < 0.02] = 255
    return img


def test_gaussian_reduces_noise_variance() -> None:
    noisy = _noisy_image()
    smoothed = gaussian_filter(noisy, 5)
    assert smoothed.std() < noisy.std()


def test_median_removes_impulse_noise() -> None:
    """Median filtering must suppress salt-and-pepper impulses far better than noise."""
    noisy = _noisy_image()
    grey = cv2.cvtColor(noisy, cv2.COLOR_RGB2GRAY)
    impulses_before = int((grey > 240).sum() + (grey < 15).sum())
    assert impulses_before > 100, "test image should contain many impulses"

    grey_med = cv2.cvtColor(median_filter(noisy, 5), cv2.COLOR_RGB2GRAY)
    impulses_after = int((grey_med > 240).sum() + (grey_med < 15).sum())
    assert impulses_after < impulses_before * 0.05


def test_median_preserves_edges_better_than_gaussian() -> None:
    """Same kernel size: the median keeps the step edge steeper."""
    img = np.zeros((64, 64, 3), np.uint8)
    img[:, 32:] = 220
    rng = np.random.default_rng(11)
    img = np.clip(img.astype(np.int16) + rng.integers(-12, 13, img.shape), 0, 255).astype(np.uint8)

    def max_step(x: np.ndarray) -> int:
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.int16)
        return int(np.abs(np.diff(g, axis=1)).max())

    med, gau = median_filter(img, 5), gaussian_filter(img, 5)
    assert max_step(med) > max_step(gau)


@pytest.mark.parametrize("ksize", [0, 2, 4, -3])
def test_invalid_kernel_sizes_rejected(ksize: int, square_rgb: np.ndarray) -> None:
    with pytest.raises(ValueError, match="positive odd"):
        gaussian_filter(square_rgb, ksize)
    with pytest.raises(ValueError, match="positive odd"):
        median_filter(square_rgb, ksize)


# ---------------------------------------------------------------------------
# CLAHE
# ---------------------------------------------------------------------------


def test_clahe_increases_local_contrast_on_flat_image() -> None:
    # A low-contrast gradient: CLAHE should spread the histogram.
    x = np.linspace(110, 140, 256, dtype=np.uint8)
    img = np.tile(x, (256, 1))
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    before = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].std()
    after = cv2.cvtColor(clahe_enhance(rgb, 2.0, (8, 8)), cv2.COLOR_RGB2LAB)[..., 0].std()
    assert after > before


def test_clahe_preserves_chromaticity_of_grey() -> None:
    grey = np.full((64, 64, 3), 120, np.uint8)
    out = clahe_enhance(grey, 2.0, (8, 8))
    lab = cv2.cvtColor(out, cv2.COLOR_RGB2LAB)
    assert np.abs(lab[..., 1].astype(int) - 128).max() <= 2
    assert np.abs(lab[..., 2].astype(int) - 128).max() <= 2


def test_clahe_rejects_bad_clip_limit(square_rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        clahe_enhance(square_rgb, 0.0)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def test_normalize_unit_range(square_rgb: np.ndarray) -> None:
    out = normalize(square_rgb, "unit")
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_normalize_imagenet_statistics() -> None:
    """A patch of exactly the ImageNet mean colour must normalise to zero."""
    mean_rgb = np.round(np.array(config.IMAGENET_MEAN) * 255).astype(np.uint8)
    rgb = mean_rgb.reshape(1, 1, 3).repeat(4, 0).repeat(4, 1)
    out = normalize(rgb, "imagenet")
    assert np.all(np.abs(out) < 0.01)


def test_normalize_imagenet_is_affine(square_rgb: np.ndarray) -> None:
    unit = normalize(square_rgb, "unit")
    imagenet = normalize(square_rgb, "imagenet")
    recovered = imagenet * np.array(config.IMAGENET_STD, np.float32) + np.array(config.IMAGENET_MEAN, np.float32)
    assert np.allclose(recovered, unit, atol=1e-5)


def test_normalize_rejects_unknown_mode(square_rgb: np.ndarray) -> None:
    with pytest.raises(ValueError):
        normalize(square_rgb, "banana")


def test_normalize_tf_mode_matches_mobilenet_preprocessing() -> None:
    """``tf`` mode must equal x/127.5-1, which is what MobileNetV2 expects."""
    rgb = np.array([[[0, 127, 255]]], dtype=np.uint8)
    out = normalize(rgb, "tf")
    assert np.allclose(out, [[[-1.0, 127 / 127.5 - 1.0, 1.0]]], atol=1e-6)


def test_normalize_caffe_mode_is_bgr_mean_subtracted() -> None:
    """``caffe`` mode must reverse channels and subtract the BGR means, no scaling."""
    rgb = np.array([[[10, 20, 30]]], dtype=np.uint8)
    out = normalize(rgb, "caffe")[0, 0]
    expected = np.array([30, 20, 10], dtype=np.float32) - np.array([103.939, 116.779, 123.68])
    assert np.allclose(out, expected, atol=1e-4)


@pytest.mark.parametrize("mode", ["unit", "imagenet", "tf", "caffe", "none"])
def test_denormalize_inverts_normalize(mode: str) -> None:
    rng = np.random.default_rng(3)
    rgb = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    back = denormalize(normalize(rgb, mode), mode)
    assert np.abs(back.astype(int) - rgb.astype(int)).max() <= 2


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_background_estimate_of_white_border(blob_on_white: np.ndarray) -> None:
    bg = estimate_background_colour(blob_on_white)
    assert np.allclose(bg, [255, 255, 255], atol=2)


def test_segmentation_finds_the_blob(blob_on_white: np.ndarray) -> None:
    mask, info = segment_vegetable(blob_on_white)
    assert info["ok"]
    assert mask.shape == (256, 256)
    assert set(np.unique(mask)).issubset({0, 255})
    # centre inside the ellipse, corner outside
    assert mask[128, 128] == 255
    assert mask[5, 5] == 0
    # area close to the ellipse's pi*a*b
    assert 0.05 < info["area_fraction"] < 0.20


def test_segmentation_of_uniform_image_is_flagged_unusable() -> None:
    flat = np.full((128, 128, 3), 200, np.uint8)
    _, info = segment_vegetable(flat)
    assert info["ok"] is False


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_output_shape_and_dtype(blob_on_white: np.ndarray) -> None:
    out = preprocess_image(blob_on_white)
    assert out.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_pipeline_reports_all_stages(blob_on_white: np.ndarray) -> None:
    _, info = preprocess_with_stages(blob_on_white, get_preset("segmented"))
    stages = info["stages"]
    for required in ("original", "resized", "hsv", "enhanced", "segmented", "model_input"):
        assert required in stages, f"missing stage {required}"
    assert info["meta"]["input_shape"] == [config.IMAGE_SIZE, config.IMAGE_SIZE, 3]
    assert len(display_stages(stages)) >= 5


def test_preset_is_reproducible(blob_on_white: np.ndarray) -> None:
    a = preprocess_image(blob_on_white, PRESETS["enhanced"])
    b = preprocess_image(blob_on_white, PRESETS["enhanced"])
    assert np.array_equal(a, b)


def test_pipeline_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((10, 10), np.uint8))
    with pytest.raises(ValueError):
        preprocess_image(np.zeros((0, 0, 3), np.uint8))


def test_unknown_preset_raises() -> None:
    with pytest.raises(KeyError):
        get_preset("nope")


def test_quality_metrics_are_finite(square_rgb: np.ndarray) -> None:
    m = image_quality_metrics(square_rgb)
    assert all(np.isfinite(v) for v in m.values())
    assert m["entropy"] > 0


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


def test_rotation_changes_pixels_but_keeps_shape() -> None:
    img = np.zeros((64, 64, 3), np.uint8)
    cv2.rectangle(img, (16, 24), (48, 40), (255, 255, 255), -1)
    out = rotate(img, 20)
    assert out.shape == img.shape
    assert not np.array_equal(out, img)


def test_horizontal_flip_is_involution() -> None:
    img = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    assert np.array_equal(horizontal_flip(horizontal_flip(img)), img)


def test_zoom_keeps_shape() -> None:
    img = np.full((50, 50, 3), 90, np.uint8)
    assert zoom(img, 1.3).shape == img.shape
    with pytest.raises(ValueError):
        zoom(img, 0.0)


def test_translate_keeps_shape() -> None:
    img = np.full((40, 40, 3), 10, np.uint8)
    assert translate(img, 5, -3).shape == img.shape


def test_brightness_and_contrast_clip_safely() -> None:
    img = np.full((16, 16, 3), 250, np.uint8)
    assert adjust_brightness(img, 0.5).max() == 255
    assert adjust_brightness(img, -5.0).min() == 0
    assert adjust_contrast(img, 3.0).max() <= 255


def test_augment_is_reproducible_from_seed(square_rgb: np.ndarray) -> None:
    cfg = AugmentConfig()
    a = augment(square_rgb, np.random.default_rng(123), cfg)
    b = augment(square_rgb, np.random.default_rng(123), cfg)
    c = augment(square_rgb, np.random.default_rng(124), cfg)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_augment_preserves_shape_and_dtype(square_rgb: np.ndarray) -> None:
    out = augment(square_rgb, np.random.default_rng(0))
    assert out.shape == square_rgb.shape
    assert out.dtype == np.uint8


def test_augment_apply_p_zero_is_identity(square_rgb: np.ndarray) -> None:
    out = augment(square_rgb, np.random.default_rng(0), AugmentConfig(apply_p=0.0))
    assert np.array_equal(out, square_rgb)


def test_augment_batch_is_deterministic(square_rgb: np.ndarray) -> None:
    batch = np.stack([square_rgb] * 4)
    a = augment_batch(batch, seed=5)
    b = augment_batch(batch, seed=5)
    assert np.array_equal(a, b)
    assert a.shape == batch.shape


def test_augmentation_grid_returns_requested_count(square_rgb: np.ndarray) -> None:
    grid = augmentation_grid(square_rgb, n=3, seed=1)
    assert len(grid) == 4
    assert np.array_equal(grid[0], square_rgb)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_colour_features_are_bounded(square_rgb: np.ndarray) -> None:
    f = colour_features(square_rgb)
    assert set(f) <= set(FEATURE_GROUPS["colour"])
    assert all(np.isfinite(v) for v in f.values())
    hist_sum = sum(v for k, v in f.items() if k.startswith("hue_hist_"))
    assert abs(hist_sum - 1.0) < 1e-6


def test_colour_features_discriminate_red_and_green() -> None:
    red = np.zeros((32, 32, 3), np.uint8); red[..., 0] = 220
    green = np.zeros((32, 32, 3), np.uint8); green[..., 1] = 220
    fr, fg = colour_features(red), colour_features(green)
    assert fr["excess_red"] > fg["excess_red"]
    assert fg["excess_green"] > fr["excess_green"]
    assert fr["hue_cos_mean"] > 0.9    # red sits at hue 0
    assert fg["hue_cos_mean"] < -0.5   # green sits near hue 120 -> cos < 0


def test_glcm_features_present_and_finite() -> None:
    rng = np.random.default_rng(1)
    gray = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    f = glcm_features(gray)
    assert len(f) == 18
    assert all(np.isfinite(v) for v in f.values())
    assert f["glcm_energy_d1"] > 0


def test_lbp_histogram_sums_to_one() -> None:
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    f = lbp_features(gray)
    assert len(f) == 10 + 18
    assert abs(sum(f.values()) - 2.0) < 1e-6  # two independent histograms


def test_shape_features_of_circle_are_circular() -> None:
    """Circularity is 4*pi*A/P^2.  ``cv2.arcLength`` measures the digital stair-
    step boundary, which over-estimates a circle's perimeter, so even a perfect
    disc reads near 0.89 rather than 1.0.  The useful property is therefore that a
    disc scores clearly higher than a square, not that it reaches 1.0."""
    disc = np.zeros((200, 200), np.uint8)
    cv2.circle(disc, (100, 100), 60, 255, -1)
    square = np.zeros((200, 200), np.uint8)
    cv2.rectangle(square, (40, 40), (160, 160), 255, -1)

    f_disc = shape_features(disc)
    f_square = shape_features(square)

    assert 0.82 < f_disc["shape_circularity"] <= 1.0
    assert abs(f_disc["shape_aspect_ratio"] - 1.0) < 0.05
    assert 0.97 < f_disc["shape_solidity"] <= 1.0
    assert f_disc["shape_circularity"] > f_square["shape_circularity"]
    assert f_square["shape_circularity"] == pytest.approx(4 * np.pi * 120 ** 2 / 480 ** 2, rel=0.05)


def test_shape_features_of_empty_mask_are_zero() -> None:
    assert all(v == 0.0 for v in shape_features(np.zeros((32, 32), np.uint8)).values())
    assert all(v == 0.0 for v in shape_features(None).values())


def test_extract_features_matches_declared_order(square_rgb: np.ndarray) -> None:
    f = extract_features(square_rgb)
    assert tuple(f.keys()) == ALL_FEATURES


def test_feature_matrix_shape_and_dtype(square_rgb: np.ndarray) -> None:
    X, names = extract_feature_matrix([square_rgb] * 3)
    assert X.shape == (3, len(ALL_FEATURES))
    assert names == ALL_FEATURES
    assert np.isfinite(X).all()
    assert X.dtype == np.float32


def test_feature_groups_partition_all_features() -> None:
    grouped = [n for members in FEATURE_GROUPS.values() for n in members]
    assert sorted(grouped) == sorted(ALL_FEATURES)
    assert len(grouped) == len(set(grouped))
