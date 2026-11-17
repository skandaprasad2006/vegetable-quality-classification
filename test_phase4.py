"""Phase 4 coverage: test-evaluation integrity, robustness transforms, audit state.

These tests lock down the things Phase 4 introduced, so a later change cannot
silently re-open the test set, alter a transform, or leave the integrity audit
failing.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import config
from src.evaluation import test_evaluation as te
from src.evaluation import robustness as rb

TEST_DIR = config.EVALUATION_DIR / "phase4_test"
AUDIT_JSON = config.EVALUATION_DIR / "phase4" / "integrity_audit.json"


# ---------------------------------------------------------------------------
# Test-set evaluation: guards and artefacts
# ---------------------------------------------------------------------------


def test_evaluator_refuses_to_overwrite(monkeypatch, tmp_path):
    """A second run must exit non-zero rather than quietly replace the score."""
    monkeypatch.setattr(te, "OUT_DIR", tmp_path)
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["test_evaluation"])
    assert te.main() == 2


def test_encode_maps_names_in_frozen_order():
    labels = pd.Series(["Ripe", "Damaged", "Unripe", "Dried", "Old"])
    encoded = te._encode(labels)
    assert encoded.tolist() == [3, 0, 4, 1, 2]


def test_encode_passes_through_integers():
    assert te._encode(np.array([0, 4], dtype=np.int32)).tolist() == [0, 4]


def test_encode_rejects_unknown_label():
    with pytest.raises(ValueError, match="unmapped"):
        te._encode(np.array(["Rotten"]))


def test_split_verifier_detects_group_leakage():
    """A group id appearing in train and test must fail the pre-flight check."""
    frame = pd.DataFrame({
        "split": ["train", "test"],
        "group_id": [7, 7],
        "sha256": ["a", "b"],
        "quality": ["Ripe", "Ripe"],
        "vegetable": ["Tomato", "Tomato"],
        "file_name": ["x.jpg", "y.jpg"],
        "rel_path": ["x", "y"],
    })
    checks = te._verify_split(frame)
    assert checks["group_ids_shared_with_train_or_validation"] == 1
    assert checks["pass"] is False


def test_test_metrics_artefact_is_complete():
    path = TEST_DIR / "metrics.json"
    if not path.exists():
        pytest.skip("Phase 4 test evaluation has not been run in this checkout")
    m = json.loads(path.read_text(encoding="utf-8"))

    assert m["n_test_images"] == 935
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "weighted_precision", "weighted_recall", "weighted_f1"):
        assert 0.0 <= m[key] <= 1.0, key

    assert m["primary_metric"] == "macro_f1"
    assert len(m["per_class"]) == 5
    assert sum(c["support"] for c in m["per_class"]) == m["n_test_images"]
    # Every reported class has test examples; none is claimed from zero support.
    assert all(c["support"] > 0 for c in m["per_class"])


def test_test_predictions_are_internally_consistent():
    path = TEST_DIR / "predictions.csv"
    if not path.exists():
        pytest.skip("Phase 4 test evaluation has not been run in this checkout")
    df = pd.read_csv(path)

    assert len(df) == 935
    prob_cols = [c for c in df.columns if c.startswith("p_")]
    assert len(prob_cols) == 5

    sums = df[prob_cols].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert (df[prob_cols].to_numpy() >= 0).all()
    assert df["confidence"].between(0, 1).all()

    # Stored prediction equals the argmax of the stored probabilities.
    class_order = list(config.QUALITY_CLASSES)
    argmax = df[prob_cols].to_numpy().argmax(axis=1)
    assert [class_order[i] for i in argmax] == df["y_pred"].tolist()
    assert df["correct"].isin([0, 1]).all()


# ---------------------------------------------------------------------------
# Robustness transforms
# ---------------------------------------------------------------------------


@pytest.fixture
def photo() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.mark.parametrize("name", sorted(rb.CONDITIONS))
def test_transform_preserves_shape_and_dtype(name, photo):
    _, fn = rb.CONDITIONS[name]
    out = fn(photo)
    assert out.shape == photo.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize("name", sorted(rb.CONDITIONS))
def test_transform_is_deterministic(name, photo):
    _, fn = rb.CONDITIONS[name]
    assert np.array_equal(fn(photo), fn(photo))


def test_control_is_identity(photo):
    _, fn = rb.CONDITIONS["control"]
    assert np.array_equal(fn(photo), photo)


def test_robustness_summary_covers_every_condition():
    path = config.EVALUATION_DIR / "phase4_robustness" / "summary.json"
    if not path.exists():
        pytest.skip("Phase 4 robustness study has not been run in this checkout")
    s = json.loads(path.read_text(encoding="utf-8"))
    assert set(s["conditions"]) == set(rb.CONDITIONS)
    for result in s["conditions"].values():
        assert 0.0 <= result["accuracy"] <= 1.0
        assert result["n_trials"] > 0


# ---------------------------------------------------------------------------
# Integrity audit state
# ---------------------------------------------------------------------------


def test_integrity_audit_exists_and_passes():
    path = AUDIT_JSON
    if not path.exists():
        pytest.skip("Phase 4 audit has not been run in this checkout")
    audit = json.loads(path.read_text(encoding="utf-8"))
    failing = [k for k, v in audit["checks"].items() if not v["pass"]]
    assert not failing, f"integrity checks failing: {failing}"


def test_serving_input_size_matches_training():
    """Regression guard for the Phase 4 defect: 224, not the 256 file size."""
    from website.utils.paths import FEATURE_IMAGE_SIZE

    assert FEATURE_IMAGE_SIZE == config.IMAGE_SIZE == 224


def test_deployed_model_is_the_registered_artefact():
    reg = pd.read_csv(config.ADMIN_DIR / "experiment_registry.csv")
    row = reg[reg["run_id"] == "A5"].iloc[0]
    from pathlib import Path

    assert Path(str(row["checkpoint_path"])).name == "rf_handcrafted.joblib"
