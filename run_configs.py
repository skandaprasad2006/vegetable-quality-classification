"""Authoritative run configurations.

These are the configurations approved in ``Admin/experiment_matrix.md``.  They
are defined here, in code, so a run cannot drift from what was documented, and so
the exact configuration is written into each run's metadata.

Only the **P0** runs are authorised for execution.  ``P1_RUNS`` and ``P2_RUNS``
are defined for completeness and must not be executed without approval.
"""

from __future__ import annotations

from typing import Any

import config

# Shared fairness controls. Changing any of these for one model and not another
# would break the comparison.
COMMON: dict[str, Any] = {
    "preset": "tf_native",
    "size": config.IMAGE_SIZE,
    "batch": 32,
    "seed": config.SEED,
    "augment": True,
    "class_weight": True,
    "es_patience": 4,
    "es_min_delta": 1e-4,
    "lr_patience": 2,
    "lr_factor": 0.3,
}

P0_RUNS: dict[str, dict[str, Any]] = {
    "A2": {
        "stage": "P0", "kind": "cnn", "backbone": "mobilenetv2", "head": "single",
        "freeze": True, "epochs": 15, "lr": 1e-3,
        "label": "MobileNetV2 frozen, quality only",
        **COMMON,
    },
    "A3": {
        "stage": "P0", "kind": "cnn", "backbone": "resnet50", "head": "single",
        "freeze": True, "epochs": 15, "lr": 1e-3,
        "label": "ResNet50 frozen, quality only",
        **COMMON,
    },
    "A4": {
        "stage": "P0", "kind": "cnn", "backbone": "mobilenetv2", "head": "multitask",
        "freeze": True, "epochs": 15, "lr": 1e-3,
        "label": "MobileNetV2 frozen, multi-task (vegetable + quality)",
        **COMMON,
    },
    "A5": {
        "stage": "P0", "kind": "classical", "head": "single",
        "n_estimators": 200, "class_weight": "balanced_subsample",
        "label": "Random Forest on 83 handcrafted features",
        "seed": config.SEED,
    },
}

#: Defined for the record only. NOT authorised for execution.
P1_RUNS: dict[str, dict[str, Any]] = {
    "A1": {"stage": "P1", "kind": "cnn", "backbone": "custom_cnn", "head": "single",
           "freeze": True, "epochs": 20, "lr": 1e-3,
           "label": "Custom CNN from scratch", **COMMON, "es_patience": 5},
    "C5": {"stage": "P1", "kind": "cnn", "backbone": "mobilenetv2", "head": "single",
           "freeze": True, "epochs": 15, "lr": 1e-3, "augment": False,
           "label": "Ablation: no augmentation", **{**COMMON, "augment": False}},
}

P2_RUNS: dict[str, dict[str, Any]] = {
    "C1": {"stage": "P2", "kind": "cnn", "backbone": "mobilenetv2", "head": "single",
           "freeze": True, "epochs": 15, "lr": 1e-3,
           "label": "Ablation: no class weighting", **{**COMMON, "class_weight": False}},
    "C2": {"stage": "P2", "kind": "cnn", "backbone": "mobilenetv2", "head": "single",
           "freeze": True, "epochs": 15, "lr": 1e-3, "preset": "enhanced",
           "label": "Ablation: median+CLAHE preprocessing", **{**COMMON, "preset": "enhanced"}},
    "C3": {"stage": "P2", "kind": "cnn", "backbone": "mobilenetv2", "head": "single",
           "freeze": True, "epochs": 15, "lr": 1e-3, "preset": "bilateral",
           "label": "Ablation: bilateral+CLAHE preprocessing", **{**COMMON, "preset": "bilateral"}},
    "C4": {"stage": "P2", "kind": "cnn", "backbone": "mobilenetv2", "head": "single",
           "freeze": True, "epochs": 15, "lr": 1e-3, "preset": "segmented",
           "label": "Ablation: background removed", **{**COMMON, "preset": "segmented"}},
}

ALL_RUNS: dict[str, dict[str, Any]] = {**P0_RUNS, **P1_RUNS, **P2_RUNS}

AUTHORISED_STAGES = ("P0",)


def get(run_id: str) -> dict[str, Any]:
    if run_id not in ALL_RUNS:
        raise KeyError(f"unknown run id {run_id!r}; known: {sorted(ALL_RUNS)}")
    cfg = dict(ALL_RUNS[run_id])
    if cfg["stage"] not in AUTHORISED_STAGES:
        raise PermissionError(
            f"run {run_id} belongs to stage {cfg['stage']}, which is not authorised. "
            f"Authorised stages: {AUTHORISED_STAGES}"
        )
    return cfg
