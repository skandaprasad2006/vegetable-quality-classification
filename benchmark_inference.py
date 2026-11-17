"""End-to-end single-image inference latency for the deployable models.

The training harness measured ``model.predict`` on **pre-processed arrays**, which
flatters a classical model: its feature extraction is the expensive part and was
done offline.  A website receives a JPEG, so the honest number is
``decode -> preprocess -> features -> predict``.

Writes ``evaluation/metrics/inference_latency.json``.

Run: ``python -m src.evaluation.benchmark_inference``
"""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pandas as pd

import config
from src.utils import imread_unicode, save_json

N_IMAGES = 25


def _paths(n: int) -> list:
    df = pd.read_csv(config.METADATA_DIR / "splits.csv")
    val = df[df["split"] == "validation"].head(n)
    return [config.PROCESSED_DIR / "validation" / r.vegetable / r.quality / r.file_name
            for r in val.itertuples(index=False)]


def _stats(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms)
    return {
        "n": int(arr.size),
        "median_ms": round(float(np.median(arr)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "p95_ms": round(float(np.quantile(arr, 0.95)), 2),
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


def bench_random_forest(paths: list) -> dict:
    # Use the same loader the website uses, so the measurement reflects the
    # configuration that is actually served (including its n_jobs setting).
    from website.utils.model_loader import load_bundle

    bundle = load_bundle()
    pipeline, features = bundle["pipeline"], bundle["features"]

    from src.features.handcrafted import extract_features
    from src.preprocessing.pipeline import segment_vegetable

    decode_t, seg_t, feat_t, pred_t, total_t = [], [], [], [], []
    for p in paths:
        t0 = time.perf_counter()
        bgr = imread_unicode(p, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                         interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()

        mask, info = segment_vegetable(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not info["ok"]:
            mask = None
        t2 = time.perf_counter()

        feats = extract_features(rgb, mask)
        X = np.array([[feats[k] for k in features]], dtype=np.float32)
        t3 = time.perf_counter()

        pipeline.predict_proba(X)
        t4 = time.perf_counter()

        decode_t.append((t1 - t0) * 1000)
        seg_t.append((t2 - t1) * 1000)
        feat_t.append((t3 - t2) * 1000)
        pred_t.append((t4 - t3) * 1000)
        total_t.append((t4 - t0) * 1000)

    return {
        "model": "A5 Random Forest",
        "components_ms": {
            "decode_resize": _stats(decode_t),
            "segmentation": _stats(seg_t),
            "feature_extraction": _stats(feat_t),
            "predict": _stats(pred_t),
        },
        "end_to_end": _stats(total_t),
    }


def bench_keras(paths: list, run_id: str, size: int, normalization: str) -> dict:
    import tensorflow as tf

    from src.training.data import normalise_tf

    t_load = time.perf_counter()
    model = tf.keras.models.load_model(config.CHECKPOINT_DIR / run_id / "best.keras")
    load_s = time.perf_counter() - t_load

    decode_t, pre_t, pred_t, total_t = [], [], [], []
    model(tf.zeros((1, size, size, 3)), training=False)

    for p in paths:
        t0 = time.perf_counter()
        bgr = imread_unicode(p, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        t1 = time.perf_counter()

        x = normalise_tf(tf.constant(rgb[None].astype(np.float32)), normalization)
        t2 = time.perf_counter()

        model(x, training=False)
        t3 = time.perf_counter()

        decode_t.append((t1 - t0) * 1000)
        pre_t.append((t2 - t1) * 1000)
        pred_t.append((t3 - t2) * 1000)
        total_t.append((t3 - t0) * 1000)

    return {
        "model": run_id,
        "model_load_s": round(load_s, 2),
        "components_ms": {
            "decode_resize": _stats(decode_t),
            "normalise": _stats(pre_t),
            "forward_pass": _stats(pred_t),
        },
        "end_to_end": _stats(total_t),
    }


def main() -> None:
    paths = _paths(N_IMAGES)
    print(f"benchmarking {len(paths)} single images, sequentially\n")

    payload = {"n_images": len(paths), "device": "CPU (TensorFlow CPU:0)", "results": {}}

    rf = bench_random_forest(paths)
    payload["results"]["A5"] = rf
    print(f"A5 random forest   end-to-end median {rf['end_to_end']['median_ms']:.1f} ms "
          f"(segmentation {rf['components_ms']['segmentation']['median_ms']:.1f} ms, "
          f"features {rf['components_ms']['feature_extraction']['median_ms']:.1f} ms)")

    a3 = bench_keras(paths, "A3", 224, "caffe")
    payload["results"]["A3"] = a3
    print(f"A3 ResNet50        end-to-end median {a3['end_to_end']['median_ms']:.1f} ms "
          f"(load {a3['model_load_s']} s)")

    a4 = bench_keras(paths, "A4", 224, "tf")
    payload["results"]["A4"] = a4
    print(f"A4 MobileNetV2 MT  end-to-end median {a4['end_to_end']['median_ms']:.1f} ms")

    save_json(payload, config.METRICS_DIR / "inference_latency.json")
    print("\nwrote evaluation/metrics/inference_latency.json")


if __name__ == "__main__":
    main()
