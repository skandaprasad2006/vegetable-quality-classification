"""Application-level tests: pages render, and the upload→predict flow works.

Streamlit's ``AppTest`` runs the real script and reports exceptions, but it cannot
drive a file uploader, so the interaction flow is exercised through the exact
functions the app calls in the same order.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import config
from website.utils.image_processor import preprocess
from website.utils.model_loader import load_bundle, predict_quality
from website.utils.validation import decode_image, validate_upload

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "website" / "app.py"
PAGES = [
    ROOT / "website" / "pages" / "1_Dataset_Information.py",
    ROOT / "website" / "pages" / "2_Model_Performance.py",
    ROOT / "website" / "pages" / "3_About_Project.py",
]


def test_main_app_renders_without_exception():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert not at.exception, at.exception
    text = " ".join(str(m.value) for m in at.markdown)
    assert "VegCheck" in text


def test_main_app_exposes_uploader():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert len(at.get("file_uploader")) == 1


def test_main_app_shows_no_predictions_before_upload():
    at = AppTest.from_file(str(APP), default_timeout=90).run()
    assert at.session_state.get("last_prediction") is None


@pytest.mark.parametrize("page", PAGES, ids=[p.stem for p in PAGES])
def test_each_page_renders_without_exception(page: Path):
    at = AppTest.from_file(str(page), default_timeout=90).run()
    assert not at.exception, at.exception


def test_dataset_page_reports_large_int_metric():
    at = AppTest.from_file(str(PAGES[0]), default_timeout=90).run()
    labels = [m.label for m in at.metric]
    assert "Total images" in labels


def test_model_page_reports_macro_f1():
    at = AppTest.from_file(str(PAGES[1]), default_timeout=90).run()
    labels = [m.label for m in at.metric]
    assert "Macro F1" in labels
    value = next(m.value for m in at.metric if m.label == "Macro F1")
    assert 0.0 < float(value) <= 1.0


# ---------------------------------------------------------------------------
# The upload → preprocess → predict → history flow, in the app's order
# ---------------------------------------------------------------------------


def test_full_prediction_flow_matches_app_order():
    from website.utils.history import add_entry, clear_entries, get_entries

    df = pd.read_csv(config.METADATA_DIR / "splits.csv")
    row = df[df["split"] == "validation"].iloc[7]
    path = (config.PROCESSED_DIR / "validation" / row["vegetable"]
            / row["quality"] / row["file_name"])
    data = path.read_bytes()

    validate_upload(path.name, data)          # 1. validation
    rgb = decode_image(data)                  # 2. decoding
    pre = preprocess(rgb)                     # 3. approved preprocessing
    pred = predict_quality(load_bundle(), pre.feature_row)   # 4-8

    state: dict = {}
    add_entry(state, filename=path.name, quality=pred.label,
              confidence=pred.confidence, model="A5", latency_ms=pred.latency_ms)

    entries = get_entries(state)
    assert len(entries) == 1
    assert entries[0]["quality"] in config.QUALITY_CLASSES
    assert entries[0]["vegetable"] is None      # never fabricated
    assert 0.0 <= entries[0]["confidence"] <= 1.0

    clear_entries(state)
    assert get_entries(state) == []


def test_history_is_capped():
    from website.utils.history import MAX_ENTRIES, add_entry, get_entries

    state: dict = {}
    for i in range(MAX_ENTRIES + 7):
        add_entry(state, filename=f"f{i}.jpg", quality="Ripe",
                  confidence=0.5, model="A5", latency_ms=1.0)
    entries = get_entries(state)
    assert len(entries) == MAX_ENTRIES
    assert entries[-1]["filename"] == f"f{MAX_ENTRIES + 6}.jpg"


def test_history_clear_on_empty_state_is_safe():
    from website.utils.history import clear_entries, get_entries

    state: dict = {}
    clear_entries(state)
    assert get_entries(state) == []


def test_invalid_upload_does_not_reach_the_model():
    """A rejected file must raise before any inference happens."""
    from website.utils.validation import ValidationError

    with pytest.raises(ValidationError):
        validate_upload("notes.txt", b"hello")


def test_generated_image_predicts_without_error():
    """A synthetic image must produce a valid result rather than a crash."""
    img = np.full((256, 256, 3), 210, dtype=np.uint8)
    cv2.circle(img, (128, 128), 70, (70, 150, 60), -1)
    pre = preprocess(img)
    pred = predict_quality(load_bundle(), pre.feature_row)
    assert pred.label in config.QUALITY_CLASSES
    assert pred.probabilities.sum() == pytest.approx(1.0, abs=1e-6)
