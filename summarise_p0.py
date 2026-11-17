"""Build the markdown tables for ``reports/phase2_p0_results.md``.

Every number is read from the artefacts written by the runs themselves
(``evaluation/metrics/<run>_validation.json`` and ``<run>_per_class.csv``), so the
report cannot drift from the measurements.

Run: ``python -m src.evaluation.summarise_p0``
"""

from __future__ import annotations

import json

import pandas as pd

import config

RUNS = ["A5", "A2", "A4", "A3"]
NAMES = {
    "A5": "Random Forest (83 handcrafted)",
    "A2": "MobileNetV2 frozen",
    "A3": "ResNet50 frozen",
    "A4": "MobileNetV2 multi-task",
}


def _load(run: str) -> dict:
    path = config.METRICS_DIR / f"{run}_validation.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    results = {r: _load(r) for r in RUNS if (config.METRICS_DIR / f"{r}_validation.json").exists()}
    if not results:
        raise SystemExit("no completed runs found")

    print("<!-- CONFIG TABLE -->")
    print("| Run | Model | Backbone state | Input | Batch | Epochs (run/planned) | LR | Normalisation | Class weighting |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r, d in results.items():
        c = d["config"]
        if c["kind"] == "classical":
            print(f"| {r} | {NAMES[r]} | — | — | — | — | — | — | {c['class_weight']} |")
        else:
            print(f"| {r} | {NAMES[r]} | {'frozen' if c.get('freeze') else 'full'} | "
                  f"{c['size']}² | {c['batch']} | {d['epochs_run']}/{c['epochs']} | "
                  f"{c['lr']} | {d['normalization']} | {c.get('class_weight')} |")
    print()

    print("<!-- HEADLINE TABLE -->")
    print("| Run | Model | **Macro F1** | Accuracy | Weighted F1 | Vs. RF (macro F1) |")
    print("|---|---|---:|---:|---:|---:|")
    rf = results.get("A5", {}).get("quality", {}).get("macro_f1")
    for r, d in results.items():
        q = d["quality"]
        delta = "—" if r == "A5" or rf is None else f"{q['macro_f1'] - rf:+.4f}"
        print(f"| {r} | {NAMES[r]} | **{q['macro_f1']:.4f}** | {q['accuracy']:.4f} | "
              f"{q['weighted_f1']:.4f} | {delta} |")
    print()

    print("<!-- PER CLASS -->")
    for r, d in results.items():
        df = pd.read_csv(config.METRICS_DIR / f"{r}_per_class.csv")
        print(f"\n**{r} — {NAMES[r]}**\n")
        print("| Class | Support | Precision | Recall | F1 |")
        print("|---|---:|---:|---:|---:|")
        for _, row in df.iterrows():
            print(f"| {row['class']} | {int(row['support'])} | {row['precision']:.4f} | "
                  f"{row['recall']:.4f} | {row['f1']:.4f} |")
        print(f"| **Macro avg** | {int(df['support'].sum())} | — | — | "
              f"**{d['quality']['macro_f1']:.4f}** |")
    print()

    print("<!-- TIMINGS -->")
    print("| Run | Train time | Inference (batched, ms/img) | Batch-1 latency median (ms) |")
    print("|---|---:|---:|---:|")
    for r, d in results.items():
        t = d["train_time_s"]
        tt = f"{t / 60:.1f} min" if t >= 60 else f"{t:.2f} s"
        b1 = d.get("latency", {}).get("batch1_ms_median")
        print(f"| {r} | {tt} | {d['inference_ms_per_image_batched']:.2f} | "
              f"{b1 if b1 is not None else '—'} |")
    print()

    print("<!-- RESOURCES -->")
    print("| Run | Peak RSS (MB) | Mean RSS (MB) | Mean CPU % | Total RAM (GB) | Aborted |")
    print("|---|---:|---:|---:|---:|---|")
    for r, d in results.items():
        res = d.get("resources") or {}
        if not res:
            print(f"| {r} | — | — | — | — | n/a (scikit-learn, in-process) |")
        else:
            print(f"| {r} | {res.get('rss_mb_max')} | {res.get('rss_mb_mean')} | "
                  f"{res.get('cpu_percent_mean')} | {res.get('total_ram_gb')} | "
                  f"{res.get('aborted_reason') or 'no'} |")
    print()

    print("<!-- MULTITASK -->")
    if "A4" in results and "vegetable" in results["A4"]:
        v = results["A4"]["vegetable"]
        print(f"vegetable macro F1 {v['macro_f1']:.4f} | accuracy {v['accuracy']:.4f} | "
              f"weighted F1 {v['weighted_f1']:.4f}")
        print(f"joint both-heads-correct {results['A4'].get('joint_both_correct_rate', float('nan')):.4f}")
        vdf = pd.read_csv(config.METRICS_DIR / "A4_vegetable_per_class.csv")
        print("\n| Vegetable | Support | Precision | Recall | F1 |")
        print("|---|---:|---:|---:|---:|")
        for _, row in vdf.iterrows():
            print(f"| {row['class']} | {int(row['support'])} | {row['precision']:.4f} | "
                  f"{row['recall']:.4f} | {row['f1']:.4f} |")


if __name__ == "__main__":
    main()
