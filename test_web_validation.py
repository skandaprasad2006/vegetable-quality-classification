"""Upload validation: file type, content, size and dimension limits."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from website.utils.validation import (
    ValidationError,
    decode_image,
    describe_upload,
    extension_of,
    validate_upload,
)


def _encode(rgb: np.ndarray, ext: str = ".png") -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(ext, bgr)
    assert ok
    return buf.tobytes()


@pytest.fixture
def photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)
    img[40:90, 40:90] = (60, 170, 70)
    return img


def test_extension_parsing():
    assert extension_of("a.JPG") == ".jpg"
    assert extension_of("a.tar.gz") == ".gz"
    assert extension_of("noext") == ""


def test_valid_png_accepted(photo):
    data = _encode(photo, ".png")
    validate_upload("veg.png", data)
    rgb = decode_image(data)
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.uint8


def test_valid_jpeg_accepted(photo):
    data = _encode(photo, ".jpg")
    validate_upload("veg.jpg", data)
    assert decode_image(data).shape[2] == 3


def test_uppercase_extension_accepted(photo):
    validate_upload("VEG.JPEG", _encode(photo, ".jpg"))


def test_empty_upload_rejected():
    with pytest.raises(ValidationError, match="empty"):
        validate_upload("veg.png", b"")


def test_none_upload_rejected():
    with pytest.raises(ValidationError, match="empty"):
        validate_upload("veg.png", None)


def test_unsupported_extension_rejected(photo):
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload("veg.gif", _encode(photo, ".png"))


def test_extension_without_dot_rejected(photo):
    with pytest.raises(ValidationError, match="Unsupported file type"):
        validate_upload("vegetable", _encode(photo))


def test_extension_spoofing_rejected():
    """A .png name over JPEG bytes is still checked by content, not extension."""
    # PNG-named text: passes the extension check, fails the magic-number check.
    with pytest.raises(ValidationError, match="not a readable"):
        validate_upload("fake.png", b"this is not an image at all")


def test_corrupted_image_rejected():
    # Valid JPEG header, no image data: the cheap checks pass by design,
    # and the decoder — the real gate in the app's flow — must reject it.
    broken = b"\xff\xd8\xff" + b"\x00" * 64
    validate_upload("broken.jpg", broken)
    with pytest.raises(ValidationError, match="could not be decoded"):
        decode_image(broken)


def test_truncated_real_jpeg_rejected(photo):
    data = _encode(photo, ".jpg")
    with pytest.raises(ValidationError):
        decode_image(data[: len(data) // 8])


def test_extremely_small_image_rejected():
    tiny = np.full((16, 16, 3), 100, dtype=np.uint8)
    with pytest.raises(ValidationError, match="minimum size"):
        decode_image(_encode(tiny))


def test_oversized_dimensions_rejected():
    wide = np.full((64, 12001, 3), 100, dtype=np.uint8)
    with pytest.raises(ValidationError, match="maximum size"):
        decode_image(_encode(wide))


def test_oversized_file_rejected(monkeypatch, photo):
    import website.utils.validation as validation

    monkeypatch.setattr(validation, "MAX_UPLOAD_BYTES", 10)
    with pytest.raises(ValidationError, match="limit is"):
        validate_upload("veg.png", _encode(photo))


def test_describe_upload_reports_dimensions(photo):
    data = _encode(photo)
    info = describe_upload("veg.png", data, decode_image(data))
    assert info["width"] == 128 and info["height"] == 128
    assert info["bytes"] == len(data)
    assert info["extension"] == ".png"


def test_non_square_image_preserved(photo):
    wide = np.full((100, 300, 3), 90, dtype=np.uint8)
    rgb = decode_image(_encode(wide))
    assert rgb.shape[:2] == (100, 300)
