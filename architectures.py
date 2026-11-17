"""Model definitions for Phase 2.

Three architectures are compared, all able to serve either a single
quality-classification head or a two-head multi-task head (vegetable + quality):

``custom_cnn``   A from-scratch convolutional network.  This is the genuine
                 baseline: it establishes what the task is worth without
                 pretrained weights, so the benefit of transfer learning is
                 measurable rather than assumed.
``mobilenetv2``  Depthwise-separable transfer learning, ImageNet weights.
``resnet50``     Residual transfer learning, ImageNet weights.

Input convention
----------------
Models accept **already-normalised** float input.  The matching
``preprocess_input`` equivalent is applied in the data pipeline, not with a
``Lambda`` layer, so a saved ``.keras`` model loads without ``custom_objects``.
Each builder records the normalisation mode it expects, and the training script
must pass that mode to the dataset builder.

Heads
-----
``single``      one softmax over the 5 quality classes
``multitask``   shared trunk, then a 4-way vegetable head and a 5-way quality
                head.  Justified by the annotations: every image carries both
                labels (verified in Phase 1 — 19/20 cells populated, but each
                image has exactly one vegetable and one quality).
"""

from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers, models

import config

#: Normalisation mode each backbone's ImageNet weights expect.
EXPECTED_NORMALIZATION = {
    "custom_cnn": "unit",
    "mobilenetv2": "tf",
    "resnet50": "caffe",
}


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def _quality_head(x: tf.Tensor, dropout: float = 0.3, name: str = "quality") -> tf.Tensor:
    x = layers.Dropout(dropout, name=f"{name}_drop")(x)
    x = layers.Dense(128, activation="relu", name=f"{name}_fc")(x)
    x = layers.Dropout(dropout / 2, name=f"{name}_drop2")(x)
    return layers.Dense(config.NUM_QUALITY_CLASSES, activation="softmax", name=name)(x)


def _vegetable_head(x: tf.Tensor, dropout: float = 0.3, name: str = "vegetable") -> tf.Tensor:
    x = layers.Dense(128, activation="relu", name=f"{name}_fc")(x)
    x = layers.Dropout(dropout / 2, name=f"{name}_drop")(x)
    return layers.Dense(config.NUM_VEGETABLE_CLASSES, activation="softmax", name=name)(x)


# ---------------------------------------------------------------------------
# Backbones
# ---------------------------------------------------------------------------


def _custom_cnn_trunk(inputs: tf.Tensor, width: int = 32, dropout: float = 0.4,
                      l2: float = 1e-4) -> tf.Tensor:
    """Four conv blocks (32→64→128→256), each Conv-BN-ReLU-MaxPool.

    BatchNorm is used because it makes a from-scratch CNN trainable at a
    reasonable learning rate on ~4.3k images; without it the baseline would fail
    for optimisation reasons rather than representational ones, which would
    overstate the value of transfer learning.
    """
    x = inputs
    for i, filters in enumerate((width, width * 2, width * 4, width * 8)):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False,
                          kernel_regularizer=tf.keras.regularizers.l2(l2),
                          name=f"block{i + 1}_conv")(x)
        x = layers.BatchNormalization(name=f"block{i + 1}_bn")(x)
        x = layers.Activation("relu", name=f"block{i + 1}_relu")(x)
        x = layers.MaxPooling2D(2, name=f"block{i + 1}_pool")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="trunk_dropout")(x)
    return x


def _transfer_trunk(inputs: tf.Tensor, backbone: str, freeze: bool,
                    unfreeze_last: int = 0) -> tf.Tensor:
    """ImageNet backbone, optionally frozen, followed by global average pooling."""
    weights = "imagenet"
    if backbone == "mobilenetv2":
        base = tf.keras.applications.MobileNetV2(
            include_top=False, weights=weights, input_shape=inputs.shape[1:])
    elif backbone == "resnet50":
        base = tf.keras.applications.ResNet50(
            include_top=False, weights=weights, input_shape=inputs.shape[1:])
    else:
        raise ValueError(f"unknown backbone {backbone!r}")

    base.trainable = not freeze
    if not freeze and unfreeze_last > 0:
        # Keep everything frozen except the last ``unfreeze_last`` layers.
        for layer in base.layers[:-unfreeze_last]:
            layer.trainable = False
    # BatchNorm layers stay frozen whenever the backbone is frozen: unfreezing
    # them on small batches corrupts the running statistics.
    if freeze:
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False

    x = base(inputs, training=False)
    return layers.GlobalAveragePooling2D(name="gap")(x)


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_model(
    backbone: str = "custom_cnn",
    head: str = "single",
    input_size: int = config.IMAGE_SIZE,
    freeze: bool = True,
    unfreeze_last: int = 0,
    width: int = 32,
    dropout: float = 0.3,
) -> tf.keras.Model:
    """Build one model.

    Parameters
    ----------
    backbone
        ``custom_cnn`` | ``mobilenetv2`` | ``resnet50``
    head
        ``single`` (quality only) | ``multitask`` (vegetable + quality)
    freeze
        Transfer backbones only. ``True`` trains the head on a frozen trunk.
    unfreeze_last
        When ``freeze=False``, unfreeze only the last N backbone layers instead
        of the whole trunk — the cheap fine-tuning mode.
    """
    if head not in ("single", "multitask"):
        raise ValueError(f"unknown head {head!r}")

    inputs = layers.Input(shape=(input_size, input_size, 3), name="image")

    if backbone == "custom_cnn":
        trunk = _custom_cnn_trunk(inputs, width=width, dropout=dropout + 0.1)
    else:
        trunk = _transfer_trunk(inputs, backbone, freeze, unfreeze_last)

    if head == "single":
        outputs: Any = _quality_head(trunk, dropout=dropout)
    else:
        outputs = {
            "vegetable": _vegetable_head(trunk, dropout=dropout),
            "quality": _quality_head(trunk, dropout=dropout),
        }

    name = f"{backbone}_{head}"
    if backbone != "custom_cnn":
        name += "_frozen" if freeze else ("_finetune" if unfreeze_last else "_full")
    return models.Model(inputs=inputs, outputs=outputs, name=name)


def compile_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    head: str = "single",
    class_weight: dict[int, float] | None = None,
    multitask_weights: tuple[float, float] = (0.3, 1.0),
) -> tf.keras.Model:
    """Compile with sparse categorical cross-entropy.

    For the multi-task head the quality loss dominates (weight 1.0) and the
    vegetable loss is a 0.3 auxiliary term: quality is the primary task of the
    project, and the vegetable head exists mainly to force the trunk to keep
    identity information rather than collapsing onto colour alone.
    """
    metrics = ["accuracy"]
    if head == "single":
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=metrics,
        )
    else:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss={
                "vegetable": "sparse_categorical_crossentropy",
                "quality": "sparse_categorical_crossentropy",
            },
            loss_weights={"vegetable": multitask_weights[0],
                          "quality": multitask_weights[1]},
            metrics={k: metrics for k in ("vegetable", "quality")},
        )
    return model


def describe(model: tf.keras.Model) -> dict[str, Any]:
    """Compact description used in metadata and the experiment registry."""
    total = int(model.count_params())
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    return {
        "name": model.name,
        "input_shape": list(model.input_shape[1:]),
        "output_names": list(model.output_names),
        "total_params": total,
        "trainable_params": trainable,
        "frozen_params": total - trainable,
        "size_mb_fp32": round(total * 4 / (1024 ** 2), 2),
    }
