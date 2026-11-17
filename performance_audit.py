"""Phase 4 Stage 4 — measured performance of the deployed pipeline.

The Phase 3 figure (~21 ms) was a single observation from one interactive session.
This script measures each stage repeatedly, separates cold start from steady state,
and checks the failure paths the brief calls out.

Outputs ``evaluation/phase4_performance/performance.json`` plus a printed summary.

Run: ``python -m src.evaluation.performance_audit``
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import psutil

import config
from src.utils import imread_unicode, save_json
from website.utils.image_processor import preprocess
from website.utils.model_loader import ModelUnavailable, load_bundle, predict_quality
from website.utils.validation import ValidationError, decode_image, validate_upload

OUT_DIR = config.EVALUATION_DIR / "phase4_performance"
N_REPEATS = 50
N_REPEAT_PREDICTIONS = 200


def stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": round(float(arr.mean()), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms": round(float(arr.std(ddof=1)), 2) if arr.size > 1 else 0.0,
        "min_ms": round(float(arr.min()), 2),
        "max_ms": round(float(arr.max()), 2),
        "p95_ms": round(float(np.quantile(arr, 0.95)), 2),
    }


def measure_cold_start() -> dict:
    """Model load in a fresh interpreter, from first statement to usable bundle."""
    code = (
        "import time,sys;t0=time.perf_counter();"
        "from website.utils.model_loader import load_bundle;"
        "b=load_bundle();"
        "print(f'{time.perf_counter()-t0:.4f}')"
    )
    samples = []
    for _ in range(3):
        t_outer = time.perf_counter()
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             cwd=str(config.PROJECT_ROOT), check=True)
        wall = time.perf_counter() - t_outer
        samples.append({"load_s": float(out.stdout.strip()), "process_wall_s": round(wall, 3)})
    return {
        "samples": samples,
        "load_s_median": round(float(np.median([s["load_s"] for s in samples])), 3),
        "interpreter_and_imports_s": round(
            float(np.median([s["process_wall_s"] - s["load_s"] for s in samples])), 3),
        "note": "Fresh process each time; includes importing numpy, sklearn, joblib.",
    }


def measure_stages() -> dict:
    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    row = manifest[manifest["split"] == "validation"].iloc[0]
    path = config.PROCESSED_DIR / "validation" / row["vegetable"] / row["quality"] / row["file_name"]
    data = path.read_bytes()

    bundle = load_bundle()
    blob = bundle["pipeline"] if False else None  # placeholder guard, unused

    # warm up every stage once
    _ = decode_image(data)
    _ = preprocess(decode_image(data))
    _ = predict_quality(bundle, preprocess(decode_image(data)).feature_row)

    t_validate, t_decode, t_pre, t_seg, t_feat, t_pred, t_total = ([] for _ in range(7))
    proc = psutil.Process(os.getpid())
    rss_start = proc.memory_info().rss
    cpu_start = psutil.cpu_percent(interval=None)

    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        validate_upload(path.name, data)
        t1 = time.perf_counter()
        rgb = decode_image(data)
        t2 = time.perf_counter()
        pre = preprocess(rgb)
        t3 = time.perf_counter()
        # isolate segmentation + features for reporting
        pred = predict_quality(bundle, pre.feature_row)
        t4 = time.perf_counter()

        t_validate.append((t1 - t0) * 1000)
        t_decode.append((t2 - t1) * 1000)
        t_pre.append((t3 - t2) * 1000)
        t_pred.append((t4 - t3) * 1000)
        t_total.append((t4 - t0) * 1000)

    cpu_used = psutil.cpu_percent(interval=None)
    rss_end = proc.memory_info().rss

    # Component split of preprocessing, measured separately
    rgb = decode_image(data)
    seg_t, feat_t = [], []
    from src.features.handcrafted import extract_features
    from src.preprocessing.pipeline import segment_vegetable

    for _ in range(N_REPEATS):
        a = time.perf_counter()
        small = cv2.resize(rgb, (config.IMAGE_SIZE, config.IMAGE_SIZE),
                           interpolation=cv2.INTER_AREA)
        mask, info = segment_vegetable(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if not info["ok"]:
            mask = None
        b = time.perf_counter()
        extract_features(small, mask)
        c = time.perf_counter()
        seg_t.append((b - a) * 1000)
        feat_t.append((c - b) * 1000)

    return {
        "image": path.name,
        "image_bytes": len(data),
        "image_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "stages": {
            "validation": stats(t_validate),
            "decode": stats(t_decode),
            "preprocess": stats(t_pre),
            "predict": stats(t_pred),
            "end_to_end": stats(t_total),
        },
        "preprocess_components": {
            "resize_plus_segmentation": stats(seg_t),
            "feature_extraction": stats(feat_t),
        },
        "process": {
            "rss_start_mb": round(rss_start / 1024 ** 2, 1),
            "rss_end_mb": round(rss_end / 1024 ** 2, 1),
            "rss_growth_mb": round((rss_end - rss_start) / 1024 ** 2, 1),
            "cpu_percent_mean": round(cpu_used, 1),
            "cpu_count": psutil.cpu_count(),
        },
    }


def measure_large_image() -> dict:
    """A 4000x3000 upload must be handled without blowing up memory or time."""
    rng = np.random.default_rng(0)
    big = rng.integers(0, 255, (3000, 4000, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(big, cv2.COLOR_RGB2BGR))
    data = buf.tobytes()

    t0 = time.perf_counter()
    validate_upload("big.jpg", data)
    rgb = decode_image(data)
    t1 = time.perf_counter()
    pre = preprocess(rgb)
    pred = predict_quality(load_bundle(), pre.feature_row)
    t2 = time.perf_counter()

    return {
        "megapixels": round(4000 * 3000 / 1e6, 1),
        "jpeg_bytes": len(data),
        "within_upload_limit": len(data) <= 15 * 1024 * 1024,
        "decode_ms": round((t1 - t0) * 1000, 1),
        "predict_and_preprocess_ms": round((t2 - t1) * 1000, 1),
        "predicted": pred.label,
        "ok": True,
    }


def measure_repeated_predictions() -> dict:
    """Many consecutive predictions: any drift or memory growth?"""
    from website.utils.samples import sample_images

    samples = sample_images()
    bundle = load_bundle()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    errors = 0
    latencies: list[float] = []

    for i in range(N_REPEAT_PREDICTIONS):
        sample = samples[i % len(samples)]
        try:
            rgb = decode_image(sample["bytes"])
            pre = preprocess(rgb)
            pred = predict_quality(bundle, pre.feature_row)
            latencies.append(pred.latency_ms)
        except Exception:  # noqa: BLE001
            errors += 1

    rss_after = proc.memory_info().rss
    return {
        "predictions": N_REPEAT_PREDICTIONS,
        "errors": errors,
        "rss_before_mb": round(rss_before / 1024 ** 2, 1),
        "rss_after_mb": round(rss_after / 1024 ** 2, 1),
        "rss_growth_mb": round((rss_after - rss_before) / 1024 ** 2, 2),
        "predict_only": stats(latencies),
    }


def check_failure_paths() -> dict:
    results = {}

    # Missing model file
    try:
        load_bundle(Path("models/final/definitely_absent.joblib"))
        results["missing_model"] = {"raised": False, "pass": False}
    except ModelUnavailable as exc:
        results["missing_model"] = {"raised": True, "message": str(exc)[:140], "pass": True}

    # Upload just under and just over the configured limit
    limit = 15 * 1024 * 1024
    head = b"\xff\xd8\xff"
    under = head + b"\x00" * (limit - 10)
    over = head + b"\x00" * (limit + 10)

    def _try(data: bytes, name: str) -> tuple[bool, str]:
        try:
            validate_upload(name, data)
            return True, ""
        except ValidationError as exc:
            return False, str(exc)

    ok_under, msg_under = _try(under, "under.jpg")
    ok_over, msg_over = _try(over, "over.jpg")
    results["upload_limit"] = {
        "limit_mb": 15,
        "just_under_accepted": ok_under,
        "just_over_rejected": not ok_over,
        "over_message": msg_over[:120],
        "pass": ok_under and (not ok_over),
    }

    # Missing evaluation artefact -> clear error, not a crash
    from website.utils import artefacts

    try:
        artefacts._read_json(Path("evaluation/metrics/does_not_exist.json"))
        results["missing_artefact"] = {"raised": False, "pass": False}
    except artefacts.ArtefactMissing as exc:
        results["missing_artefact"] = {"raised": True, "message": str(exc)[:120], "pass": True}

    # Invalid image bytes
    try:
        decode_image(b"not an image")
        results["invalid_image"] = {"raised": False, "pass": False}
    except ValidationError as exc:
        results["invalid_image"] = {"raised": True, "message": str(exc)[:120], "pass": True}

    results["all_pass"] = all(v.get("pass", False) for v in results.values()
                              if isinstance(v, dict))
    return results


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("measuring cold start (3 fresh processes)...")
    cold = measure_cold_start()

    print(f"measuring stages ({N_REPEATS} repeats)...")
    stages = measure_stages()

    print("measuring large-image handling...")
    large = measure_large_image()

    print(f"measuring {N_REPEAT_PREDICTIONS} consecutive predictions...")
    repeated = measure_repeated_predictions()

    print("checking failure paths...")
    failures = check_failure_paths()

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cold_start": cold,
        "stages": stages,
        "large_image": large,
        "repeated_predictions": repeated,
        "failure_paths": failures,
    }
    save_json(payload, OUT_DIR / "performance.json")

    print("\n--- steady state (ms) ---")
    for name, s in stages["stages"].items():
        print(f"  {name:<12} mean {s['mean_ms']:>7.2f}  median {s['median_ms']:>6.2f}  "
              f"sd {s['std_ms']:>5.2f}  min {s['min_ms']:>6.2f}  max {s['max_ms']:>7.2f}")
    print(f"\ncold start (load only): {cold['load_s_median']} s")
    print(f"repeated predictions: {N_REPEAT_PREDICTIONS} runs, {repeated['errors']} errors, "
          f"RSS growth {repeated['rss_growth_mb']} MB")
    print(f"large image: {large['megapixels']} MP in "
          f"{large['decode_ms'] + large['predict_and_preprocess_ms']:.0f} ms")
    print(f"failure paths pass: {failures['all_pass']}")
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
