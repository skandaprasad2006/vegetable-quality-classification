"""Phase 4 Stage 3 — robustness of the deployed pipeline to photographic variation.

**What this is and is not.** The repository contains no independent set of ordinary
photographs: the dataset is studio-style imagery on plain backgrounds. Rather than
claim a real-world evaluation that was never performed, this script measures how
the *frozen* pipeline behaves under controlled degradations that approximate the
conditions an ordinary photograph introduces — rotation, off-native resolution,
JPEG artefacts, blur, dim or harsh lighting, backlight, and a tighter crop.

Because the transformations are derived from held-out test images, the ground-truth
label is inherited and the result is quantitative. It is **not** a test of unseen
vegetables or unseen backgrounds, and it is reported separately from the official
test score.

Outputs (``evaluation/phase4_robustness/``):
    trials.csv     one row per (image, condition)
    summary.json   per-condition accuracy, confidence and prediction distribution
    manifest.csv   the trial manifest with provenance and suitability flags

Run: ``python -m src.evaluation.robustness``
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd

import config
from src.evaluation.metrics import summarise
from src.utils import imread_unicode, save_json
from website.utils.image_processor import preprocess
from website.utils.model_loader import load_bundle, predict_quality

OUT_DIR = config.EVALUATION_DIR / "phase4_robustness"
#: 60 stratified-random held-out images. 25 (the brief's photograph range) gave a
#: control accuracy 19 points below the official test score, because taking the
#: first rows per class selects an unrepresentatively hard subset.
N_BASE = 60


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


def identity(img: np.ndarray) -> np.ndarray:
    """Control: the unmodified photograph."""
    return img


def rotate_15(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), 15, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def low_resolution(img: np.ndarray) -> np.ndarray:
    """Simulate a distant phone shot: downscale to 96 px, back up to 224."""
    small = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (224, 224), interpolation=cv2.INTER_LINEAR)


def jpeg_q30(img: np.ndarray) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                           [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    if not ok:
        return img
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def dim_lighting(img: np.ndarray) -> np.ndarray:
    return np.clip(img.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)


def harsh_lighting(img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32) * 1.45
    return np.clip(out, 0, 255).astype(np.uint8)


def low_contrast(img: np.ndarray) -> np.ndarray:
    mean = float(img.mean())
    out = (img.astype(np.float32) - mean) * 0.55 + mean
    return np.clip(out, 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray) -> np.ndarray:
    """Slightly out-of-focus capture."""
    return cv2.GaussianBlur(img, (5, 5), 1.4)


def vignette(img: np.ndarray) -> np.ndarray:
    """Uneven illumination falling off toward the corners."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    mask = np.clip(1.15 - 0.55 * d, 0.35, 1.0)[..., None]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def tight_crop(img: np.ndarray) -> np.ndarray:
    """Closer framing: keep the central 70 % and rescale."""
    h, w = img.shape[:2]
    dy, dx = int(h * 0.15), int(w * 0.15)
    crop = img[dy:h - dy, dx:w - dx]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


CONDITIONS: dict[str, tuple[str, Callable[[np.ndarray], np.ndarray]]] = {
    "control":        ("unmodified test image", identity),
    "rotated_15deg":  ("rotated 15 degrees, reflected borders", rotate_15),
    "low_resolution": ("downscaled to 96px then restored to 224px", low_resolution),
    "jpeg_q30":       ("JPEG recompressed at quality 30", jpeg_q30),
    "dim_lighting":   ("brightness x 0.55 (dim room)", dim_lighting),
    "harsh_lighting": ("brightness x 1.45 (overexposed)", harsh_lighting),
    "low_contrast":   ("contrast compressed to 55%", low_contrast),
    "gaussian_blur":  ("5x5 gaussian, sigma 1.4 (out of focus)", gaussian_blur),
    "vignette":       ("radial illumination falloff", vignette),
    "tight_crop":     ("central 70% crop (closer framing)", tight_crop),
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()

    manifest = pd.read_csv(config.METADATA_DIR / "splits.csv")
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)

    # Stratified by quality class, proportional, drawn with a fixed seed so the
    # study is reproducible.
    rng = np.random.default_rng(config.SEED)
    parts = []
    for _, group in test.groupby("quality"):
        take = max(2, int(round(N_BASE * len(group) / len(test))))
        idx = rng.choice(len(group), size=min(take, len(group)), replace=False)
        parts.append(group.iloc[np.sort(idx)])
    base = pd.concat(parts).reset_index(drop=True)
    print(f"base images: {len(base)} across {base['quality'].nunique()} classes "
          f"(stratified, seed {config.SEED})")

    rows: list[dict] = []
    t_all = time.perf_counter()

    for rec in base.itertuples(index=False):
        path = (config.PROCESSED_DIR / "test" / rec.vegetable / rec.quality / rec.file_name)
        bgr = imread_unicode(path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        for name, (description, fn) in CONDITIONS.items():
            t0 = time.perf_counter()
            transformed = fn(rgb)
            pre = preprocess(transformed)
            pred = predict_quality(bundle, pre.feature_row)
            elapsed = (time.perf_counter() - t0) * 1000

            rows.append({
                "file_name": rec.file_name,
                "vegetable": rec.vegetable,
                "true_quality": rec.quality,
                "condition": name,
                "condition_description": description,
                "predicted_quality": pred.label,
                "confidence": round(pred.confidence, 4),
                "correct": int(pred.label == rec.quality),
                "runner_up": pred.second_label,
                "runner_up_confidence": round(pred.second_confidence, 4),
                "segmentation_ok": pred is not None and pre.segmentation["ok"],
                "latency_ms": round(elapsed, 2),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "trials.csv", index=False)

    summary: dict = {
        "n_base_images": int(len(base)),
        "conditions": {},
        "note": ("Derived from held-out test images by controlled transformation. "
                 "Quantitative but not a real-world photograph study."),
    }

    for name, group in df.groupby("condition"):
        preds = group["predicted_quality"].value_counts().to_dict()
        summary["conditions"][name] = {
            "description": CONDITIONS[name][0],
            "n_trials": int(len(group)),
            "accuracy": round(float(group["correct"].mean()), 4),
            "mean_confidence": round(float(group["confidence"].mean()), 4),
            "median_confidence": round(float(group["confidence"].median()), 4),
            "accuracy_when_confident_ge_0_7": (
                round(float(group.loc[group["confidence"] >= 0.7, "correct"].mean()), 4)
                if (group["confidence"] >= 0.7).any() else None),
            "n_confident_ge_0_7": int((group["confidence"] >= 0.7).sum()),
            "segmentation_ok_fraction": round(float(group["segmentation_ok"].mean()), 4),
            "mean_latency_ms": round(float(group["latency_ms"].mean()), 2),
            "prediction_distribution": {str(k): int(v) for k, v in preds.items()},
        }

    summary["overall_accuracy"] = round(float(df["correct"].mean()), 4)
    summary["overall_mean_confidence"] = round(float(df["confidence"].mean()), 4)
    summary["elapsed_s"] = round(time.perf_counter() - t_all, 2)

    worst = sorted(summary["conditions"].items(), key=lambda kv: kv[1]["accuracy"])[:3]
    summary["weakest_conditions"] = [{"condition": k, "accuracy": v["accuracy"]}
                                     for k, v in worst]

    # Failure examples, for the report
    failures = df[df["correct"] == 0].sort_values("confidence", ascending=False)
    summary["n_failures"] = int(len(failures))
    summary["highest_confidence_failures"] = failures.head(8)[[
        "file_name", "condition", "true_quality", "predicted_quality", "confidence"
    ]].to_dict("records")

    save_json(summary, OUT_DIR / "summary.json")

    # Manifest, as required by the stage brief.
    man = df[[
        "file_name", "condition", "condition_description", "vegetable",
        "true_quality", "predicted_quality", "confidence",
    ]].copy()
    man.insert(0, "source", "New VegNet test split (held-out), transformed in memory")
    man.insert(1, "real_photograph", "no")
    man["label_provenance"] = "dataset label of the unmodified source image"
    man["suitable_for_quantitative_evaluation"] = "yes (label inherited)"
    man.to_csv(OUT_DIR / "manifest.csv", index=False)

    print(f"\noverall accuracy across conditions: {summary['overall_accuracy']:.4f}")
    for name, s in sorted(summary["conditions"].items(), key=lambda kv: -kv[1]["accuracy"]):
        print(f"  {name:<16} acc={s['accuracy']:.3f}  conf={s['mean_confidence']:.3f}  "
              f"n={s['n_trials']}")
    print(f"wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
