"""Model loading and inference: contract checks, error paths, real integration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

import config
from src.features.handcrafted import ALL_FEATURES
from website.utils.image_processor import preprocess
from website.utils.model_loader import (
    ModelUnavailable,
    Prediction,
    PredictionError,
    load_bundle,
    model_summary,
    predict_quality,
    verify_bundle,
)
from website.utils.paths import DEPLOYED_MODEL_PATH, QUALITY_CLASSES
from website.utils.validation import decode_image


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_model_file_exists():
    assert DEPLOYED_MODEL_PATH.exists(), (
        "the deployed artefact is missing; run `python -m src.training.train --run-id A5`")


def test_bundle_loads_and_verifies():
    bundle = load_bundle()
    assert set(bundle) >= {"pipeline", "features", "classes"}


def test_feature_contract_matches_project():
    bundle = load_bundle()
    assert list(bundle["features"]) == list(ALL_FEATURES)


def test_class_order_matches_frozen_config():
    bundle = load_bundle()
    assert list(bundle["classes"]) == list(config.QUALITY_CLASSES)


def test_summary_reports_structure():
    summary = model_summary(load_bundle())
    assert summary["n_features"] == len(ALL_FEATURES)
    assert summary["n_classes"] == len(QUALITY_CLASSES)
    assert summary["n_trees_fitted"] == summary["n_estimators"] > 0


def test_missing_file_raises_clear_error(tmp_path: Path):
    with pytest.raises(ModelUnavailable, match="not found"):
        load_bundle(tmp_path / "does_not_exist.joblib")


def test_verify_rejects_wrong_feature_list():
    bundle = load_bundle()
    broken = dict(bundle)
    broken["features"] = list(ALL_FEATURES)[:-1]
    with pytest.raises(ModelUnavailable, match="feature list"):
        verify_bundle(broken)


def test_verify_rejects_wrong_class_order():
    bundle = load_bundle()
    broken = dict(bundle)
    broken["classes"] = list(reversed(list(QUALITY_CLASSES)))
    with pytest.raises(ModelUnavailable, match="class order"):
        verify_bundle(broken)


def test_verify_rejects_missing_keys():
    with pytest.raises(ModelUnavailable, match="missing"):
        verify_bundle({"pipeline": object()})


# ---------------------------------------------------------------------------
# Prediction output
# ---------------------------------------------------------------------------


@pytest.fixture
def real_features() -> np.ndarray:
    bundle = load_bundle()
    df = pd.read_csv(config.METADATA_DIR / "splits.csv")
    row = df[df["split"] == "validation"].iloc[0]
    path = (config.PROCESSED_DIR / "validation" / row["vegetable"]
            / row["quality"] / row["file_name"])
    bgr = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return preprocess(rgb).feature_row


def test_prediction_is_valid_distribution(real_features):
    pred = predict_quality(load_bundle(), real_features)
    assert isinstance(pred, Prediction)
    assert pred.probabilities.shape == (len(QUALITY_CLASSES),)
    assert np.all(np.isfinite(pred.probabilities))
    assert np.all(pred.probabilities >= 0)
    assert pred.probabilities.sum() == pytest.approx(1.0, abs=1e-6)


def test_prediction_label_matches_argmax(real_features):
    pred = predict_quality(load_bundle(), real_features)
    assert pred.label == QUALITY_CLASSES[int(np.argmax(pred.probabilities))]
    assert pred.label == pred.class_names[pred.index]
    assert pred.confidence == pytest.approx(float(pred.probabilities.max()))


def test_prediction_label_is_a_known_class(real_features):
    pred = predict_quality(load_bundle(), real_features)
    assert pred.label in QUALITY_CLASSES


def test_second_choice_is_distinct(real_features):
    pred = predict_quality(load_bundle(), real_features)
    assert pred.second_label != pred.label
    assert pred.second_confidence <= pred.confidence


def test_negative_latency_is_impossible(real_features):
    pred = predict_quality(load_bundle(), real_features)
    assert pred.latency_ms >= 0


# ---------------------------------------------------------------------------
# Invalid inputs
# ---------------------------------------------------------------------------


def test_wrong_feature_count_rejected():
    with pytest.raises(PredictionError, match="feature row"):
        predict_quality(load_bundle(), np.zeros((1, 10), dtype=np.float32))


def test_two_rows_rejected():
    with pytest.raises(PredictionError, match="feature row"):
        predict_quality(load_bundle(), np.zeros((2, len(ALL_FEATURES)), dtype=np.float32))


def test_nan_features_rejected():
    row = np.zeros((1, len(ALL_FEATURES)), dtype=np.float32)
    row[0, 5] = np.nan
    with pytest.raises(PredictionError, match="NaN"):
        predict_quality(load_bundle(), row)


def test_infinite_features_rejected():
    row = np.zeros((1, len(ALL_FEATURES)), dtype=np.float32)
    row[0, 7] = np.inf
    with pytest.raises(PredictionError, match="NaN"):
        predict_quality(load_bundle(), row)


class _BadPipeline:
    """Stand-in pipeline that returns something a probability vector cannot be."""

    def __init__(self, output):
        self._output = output

    def predict_proba(self, X):  # noqa: N803 - mirrors sklearn's signature
        return self._output


def _bundle_with(pipeline) -> dict:
    return {"pipeline": pipeline, "features": list(ALL_FEATURES),
            "classes": list(QUALITY_CLASSES)}


@pytest.mark.parametrize("bad", [
    np.array([[0.5, 0.5]]),                       # wrong number of classes
    np.array([[0.6, 0.6, 0.0, 0.0, 0.0]]),        # does not sum to 1
    np.array([[np.nan] * 5]),                     # NaN
    np.array([[-0.1, 0.4, 0.3, 0.2, 0.2]]),       # negative
])
def test_malformed_probabilities_rejected(bad):
    with pytest.raises(PredictionError):
        predict_quality(_bundle_with(_BadPipeline(bad)),
                        np.zeros((1, len(ALL_FEATURES)), dtype=np.float32))


# ---------------------------------------------------------------------------
# End-to-end with the real artefact and a real photograph
# ---------------------------------------------------------------------------


def test_end_to_end_real_image_returns_plausible_result():
    df = pd.read_csv(config.METADATA_DIR / "splits.csv")
    row = df[df["split"] == "validation"].iloc[3]
    path = (config.PROCESSED_DIR / "validation" / row["vegetable"]
            / row["quality"] / row["file_name"])
    data = path.read_bytes()

    rgb = decode_image(data)              # as the app does from uploaded bytes
    pre = preprocess(rgb)
    pred = predict_quality(load_bundle(), pre.feature_row)

    assert pred.label in QUALITY_CLASSES
    assert 0.2 <= pred.confidence <= 1.0   # 5 classes, so a blind guess is 0.2
    assert pred.probabilities.sum() == pytest.approx(1.0, abs=1e-6)
