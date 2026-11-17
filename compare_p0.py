"""Compare the four P0 runs and decide which model is deployed.

The comparison is **paired**: every run was scored on the same 929 validation
images in the same order, so differences can be bootstrapped over samples and
tested with McNemar's test on the discordant pairs.  Reporting a bare 0.0018 gap
as a win would be indefensible.

Writes ``evaluation/metrics/p0_comparison.json`` — the single artefact the website
reads for its performance page, so no metric is ever retyped by hand.

Run: ``python -m src.evaluation.compare_p0``
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.stats import binomtest

import config

RUNS = ["A5", "A3", "A4", "A2"]
DISPLAY = {
    "A5": "Random Forest (83 handcrafted features)",
    "A3": "ResNet50 (frozen backbone)",
    "A4": "MobileNetV2 multi-task (vegetable + quality)",
    "A2": "MobileNetV2 (frozen backbone)",
}
N_BOOT = 2000


def _load(run: str) -> dict:
    return json.loads((config.METRICS_DIR / f"{run}_validation.json").read_text(encoding="utf-8"))


def _preds(run: str) -> pd.DataFrame:
    return pd.read_csv(config.METRICS_DIR / f"{run}_predictions.csv")


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import f1_score

    return float(f1_score(y, p, average="macro", zero_division=0))


def paired_difference(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Bootstrap the macro-F1 difference ``a - b`` over the shared samples."""
    y = a["y_true"].to_numpy()
    pa = a["y_pred"].to_numpy()
    pb = b["y_pred"].to_numpy()
    if not np.array_equal(y, b["y_true"].to_numpy()):
        raise RuntimeError("runs were not evaluated on identical samples in identical order")

    rng = np.random.default_rng(config.SEED)
    n = len(y)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas[i] = _macro_f1(y[idx], pa[idx]) - _macro_f1(y[idx], pb[idx])

    # McNemar on discordant pairs: does one model err where the other is right?
    a_right = (pa == y)
    b_right = (pb == y)
    n01 = int(np.sum(~a_right & b_right))   # a wrong, b right
    n10 = int(np.sum(a_right & ~b_right))   # a right, b wrong
    p_mcnemar = 1.0
    if n01 + n10 > 0:
        p_mcnemar = float(binomtest(n10, n01 + n10, 0.5).pvalue)

    return {
        "macro_f1_a": _macro_f1(y, pa),
        "macro_f1_b": _macro_f1(y, pb),
        "delta": float(_macro_f1(y, pa) - _macro_f1(y, pb)),
        "delta_ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
        "p_delta_le_zero": float(np.mean(deltas <= 0)),
        "mcnemar": {
            "a_wrong_b_right": n01,
            "a_right_b_wrong": n10,
            "p_value": p_mcnemar,
        },
        "significant_at_0.05": bool(p_mcnemar < 0.05),
    }


def main() -> None:
    runs = {r: _load(r) for r in RUNS}
    preds = {r: _preds(r) for r in RUNS}

    table = {}
    for r, d in runs.items():
        q = d["quality"]
        entry = {
            "display_name": DISPLAY[r],
            "run_id": r,
            "stage": d["stage"],
            "model_family": "classical" if d["config"]["kind"] == "classical" else "deep learning",
            "n_features": d.get("model", {}).get("n_features"),
            "n_parameters": d.get("model", {}).get("total_params"),
            "config": {
                "backbone": d["config"].get("backbone"),
                "head": d["config"].get("head"),
                "input_size": d["config"].get("size"),
                "batch_size": d["config"].get("batch"),
                "epochs": d["config"].get("epochs"),
                "epochs_run": d.get("epochs_run"),
                "learning_rate": d["config"].get("lr"),
                "normalization": d.get("normalization"),
                "class_weighted": d["config"].get("class_weight"),
                "n_estimators": d["config"].get("n_estimators"),
            },
            "quality": {
                "macro_f1": q["macro_f1"],
                "macro_f1_ci95": q.get("macro_f1_ci95"),
                "accuracy": q["accuracy"],
                "weighted_f1": q["weighted_f1"],
                "macro_precision": q.get("macro_precision"),
                "macro_recall": q.get("macro_recall"),
                "n_samples": q["n_samples"],
                "per_class": q["per_class"],
                "classes_absent": q.get("classes_absent", []),
            },
            "timing": {
                "train_time_s": d["train_time_s"],
                "inference_ms_per_image": d["inference_ms_per_image_batched"],
                "batch1_ms_median": d.get("latency", {}).get("batch1_ms_median"),
            },
            "resources": d.get("resources") or {},
            "artefacts": d.get("checkpoints", {}),
        }
        if "vegetable" in d:
            v = d["vegetable"]
            entry["vegetable"] = {
                "macro_f1": v["macro_f1"],
                "macro_f1_ci95": v.get("macro_f1_ci95"),
                "accuracy": v["accuracy"],
                "weighted_f1": v["weighted_f1"],
                "per_class": v["per_class"],
                "joint_both_correct_rate": d.get("joint_both_correct_rate"),
            }
        table[r] = entry

    pairs = {}
    for i, a in enumerate(RUNS):
        for b in RUNS[i + 1:]:
            pairs[f"{a}_vs_{b}"] = paired_difference(preds[a], preds[b])

    ranking = sorted(RUNS, key=lambda r: runs[r]["quality"]["macro_f1"], reverse=True)
    best = ranking[0]
    runner_up = ranking[1]
    head_to_head = pairs[f"{best}_vs_{runner_up}"] if f"{best}_vs_{runner_up}" in pairs \
        else pairs[f"{runner_up}_vs_{best}"]

    deployment = {
        "ranking_by_macro_f1": ranking,
        "point_estimate_winner": best,
        "runner_up": runner_up,
        "winner_beats_runner_up_significantly": bool(
            head_to_head["significant_at_0.05"] and head_to_head["delta"] > 0),
        "head_to_head": head_to_head,
    }

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "protocol": {
            "evaluation": "single holdout validation split",
            "train_images": int(runs["A2"]["train_images"]),
            "validation_images": int(runs["A2"]["val_images"]),
            "test_images_used": 0,
            "seed": config.SEED,
            "primary_metric": "macro F1",
            "note": ("All runs scored on the identical 929 validation images in identical "
                     "order. The test split was never loaded. The Phase 1 grouped-CV figure "
                     "(0.9199) is a separate historical benchmark and is not comparable."),
        },
        "runs": table,
        "pairwise": pairs,
        "deployment_decision": deployment,
    }

    out = config.METRICS_DIR / "p0_comparison.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}\n")

    print("| Run | Model | Macro F1 | 95% CI | Accuracy | Weighted F1 | Train | Inference |")
    print("|---|---|---:|---|---:|---:|---:|---:|")
    for r in ranking:
        e = table[r]
        ci = e["quality"]["macro_f1_ci95"]
        print(f"| {r} | {e['display_name']} | {e['quality']['macro_f1']:.4f} | "
              f"[{ci['lo']:.4f}, {ci['hi']:.4f}] | {e['quality']['accuracy']:.4f} | "
              f"{e['quality']['weighted_f1']:.4f} | {e['timing']['train_time_s']:.1f} s | "
              f"{e['timing']['inference_ms_per_image']:.2f} ms |")
    print()
    for name, p in pairs.items():
        print(f"{name}: delta {p['delta']:+.4f} "
              f"CI [{p['delta_ci95'][0]:+.4f}, {p['delta_ci95'][1]:+.4f}] "
              f"McNemar p={p['mcnemar']['p_value']:.4g}")

    print()
    print(f"ranked: {ranking}")
    print(f"best: {best}, runner-up: {runner_up}, "
          f"significant: {deployment['winner_beats_runner_up_significantly']}")


if __name__ == "__main__":
    main()
