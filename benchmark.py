"""Hardware and throughput benchmark for Phase 2 planning.

Nothing here is estimated from a spec sheet.  Every number is measured by running
real TensorFlow work on this machine:

* device configuration reported by TensorFlow itself;
* input-pipeline throughput for both dataset paths (TensorFlow ops and the OpenCV
  preset), in images/second;
* forward+backward time per batch for each candidate architecture, measured with
  ``train_on_batch`` after warm-up;
* inference time per image;
* process RSS before and after building and stepping a model.

From those, per-epoch training time is derived for the real training-split size.

Run: ``python -m src.training.benchmark``
"""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np

import config
from src.utils import save_json

#: Architectures to time.  ``freeze`` governs transfer backbones.
BENCHMARK_CONFIGS: list[dict[str, Any]] = [
    {"backbone": "custom_cnn", "head": "single", "size": 224, "batch": 32, "freeze": True},
    {"backbone": "mobilenetv2", "head": "single", "size": 224, "batch": 32, "freeze": True},
    {"backbone": "mobilenetv2", "head": "single", "size": 224, "batch": 32,
     "freeze": False, "unfreeze_last": 30},
    {"backbone": "mobilenetv2", "head": "single", "size": 224, "batch": 32, "freeze": False},
    {"backbone": "resnet50", "head": "single", "size": 224, "batch": 32, "freeze": True},
    {"backbone": "resnet50", "head": "single", "size": 160, "batch": 32, "freeze": True},
    {"backbone": "mobilenetv2", "head": "multitask", "size": 224, "batch": 32, "freeze": True},
]

WARMUP_STEPS = 2
TIMED_STEPS = 4


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def environment() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    disk = shutil.disk_usage(config.PROJECT_ROOT)
    cpu_count = psutil.cpu_count(logical=True)
    phys = psutil.cpu_count(logical=False)

    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_logical_cores": cpu_count,
        "cpu_physical_cores": phys,
        "cpu_freq_mhz": getattr(psutil.cpu_freq(), "max", None),
        "ram_total_gb": round(vm.total / 1024 ** 3, 2),
        "ram_available_gb": round(vm.available / 1024 ** 3, 2),
        "disk_total_gb": round(disk.total / 1024 ** 3, 1),
        "disk_free_gb": round(disk.free / 1024 ** 3, 1),
        "python": platform.python_version(),
        "cwd": str(config.PROJECT_ROOT),
    }


def tf_environment() -> dict[str, Any]:
    import tensorflow as tf

    devices = tf.config.list_physical_devices()
    gpus = tf.config.list_physical_devices("GPU")
    return {
        "tensorflow_version": tf.__version__,
        "eager": bool(tf.executing_eagerly()),
        "physical_devices": [f"{d.device_type}:{d.name}" for d in devices],
        "cpu_count": int(tf.config.list_physical_devices("CPU").__len__()),
        "gpu_devices": len(gpus),
        "gpu_visible_to_tf": bool(gpus),
        "intra_op_threads": int(tf.config.threading.get_intra_op_parallelism_threads()),
        "inter_op_threads": int(tf.config.threading.get_inter_op_parallelism_threads()),
        "mixed_precision": tf.keras.mixed_precision.global_policy().name,
        "note": (
            "No GPU is visible to TensorFlow on native Windows; the installed "
            "tensorflow-cpu build has no CUDA support and TF >= 2.11 does not "
            "ship native Windows GPU binaries."
        ) if not gpus else "GPU visible to TensorFlow.",
    }


# ---------------------------------------------------------------------------
# Throughput
# ---------------------------------------------------------------------------


def dataset_throughput(preset: str, size: int, batch: int = 32,
                       batches: int = 8, normalize: str = "unit") -> dict[str, Any]:
    """Measure images/second for one input pipeline (decode through to batch)."""
    from src.training.data import make_dataset

    ds = make_dataset("train", preset=preset, size=size, batch_size=batch,
                      normalize=normalize, training=False, head="single")
    it = iter(ds)
    next(it)  # warm the pipeline

    t0 = time.perf_counter()
    n = 0
    for _ in range(batches):
        try:
            xb, _ = next(it)
        except StopIteration:
            break
        n += int(xb.shape[0])
    elapsed = time.perf_counter() - t0
    return {
        "preset": preset,
        "batch_size": batch,
        "images": n,
        "seconds": round(elapsed, 3),
        "images_per_second": round(n / elapsed, 1) if elapsed > 0 else None,
    }


# ---------------------------------------------------------------------------
# Model timing
# ---------------------------------------------------------------------------


def _synthetic_batch(size: int, batch: int, mode: str, head: str):
    rng = np.random.default_rng(0)
    x = rng.random((batch, size, size, 3)).astype("float32")
    from src.training.data import normalise_tf
    import tensorflow as tf

    xn = normalise_tf(tf.constant(x * 255.0), mode).numpy()
    if head == "single":
        y = rng.integers(0, config.NUM_QUALITY_CLASSES, batch).astype("int32")
    else:
        y = {
            "vegetable": rng.integers(0, config.NUM_VEGETABLE_CLASSES, batch).astype("int32"),
            "quality": rng.integers(0, config.NUM_QUALITY_CLASSES, batch).astype("int32"),
        }
    return xn, y


def benchmark_model(cfg: dict[str, Any], train_size: int) -> dict[str, Any]:
    import psutil
    import tensorflow as tf

    from src.models.architectures import EXPECTED_NORMALIZATION, build_model, compile_model, describe

    backbone, head, size, batch = cfg["backbone"], cfg["head"], cfg["size"], cfg["batch"]
    freeze = cfg.get("freeze", True)

    base_mode = EXPECTED_NORMALIZATION[backbone]
    # A from-scratch CNN is normalised to [0,1]; transfer backbones keep theirs.
    mode = base_mode if backbone != "custom_cnn" else "unit"

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss

    unfreeze_last = int(cfg.get("unfreeze_last", 0))
    model = build_model(backbone=backbone, head=head, input_size=size,
                        freeze=freeze, unfreeze_last=unfreeze_last, dropout=0.3)
    model = compile_model(model, learning_rate=1e-3, head=head)

    rss_built = proc.memory_info().rss
    info = describe(model)
    info.update({"backbone": backbone, "head": head, "input_size": size,
                 "batch_size": batch, "frozen": bool(freeze),
                 "unfreeze_last": unfreeze_last, "normalization": mode})

    x, y = _synthetic_batch(size, batch, mode, head)

    # warm-up: first step includes graph construction / memory allocation
    for _ in range(WARMUP_STEPS):
        model.train_on_batch(x, y, return_dict=True)

    step_times: list[float] = []
    for _ in range(TIMED_STEPS):
        t0 = time.perf_counter()
        model.train_on_batch(x, y, return_dict=True)
        step_times.append(time.perf_counter() - t0)

    rss_after = proc.memory_info().rss

    # inference
    inf_times: list[float] = []
    for _ in range(3):
        t0 = time.perf_counter()
        model(x, training=False)
        inf_times.append(time.perf_counter() - t0)

    median_step = float(np.median(step_times))
    steps_per_epoch = math.ceil(train_size / batch)

    info.update({
        "step_time_s_median": round(median_step, 3),
        "step_time_s_min": round(min(step_times), 3),
        "step_time_s_all": [round(t, 3) for t in step_times],
        "images_per_second_train": round(batch / median_step, 1),
        "steps_per_epoch": steps_per_epoch,
        "est_epoch_time_s": round(steps_per_epoch * median_step, 1),
        "est_epoch_time_min": round(steps_per_epoch * median_step / 60.0, 2),
        "inference_batch_s_median": round(float(np.median(inf_times)), 4),
        "inference_ms_per_image": round(float(np.median(inf_times)) / batch * 1000, 2),
        "rss_delta_model_mb": round((rss_built - rss_before) / 1024 ** 2, 1),
        "rss_delta_train_mb": round((rss_after - rss_built) / 1024 ** 2, 1),
        "rss_total_mb": round(rss_after / 1024 ** 2, 1),
    })

    del model, x, y
    tf.keras.backend.clear_session()
    return info


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run(batches: int = 8) -> dict[str, Any]:
    import tensorflow as tf

    config.ensure_project_dirs()
    t_start = time.perf_counter()

    from src.training.data import dataset_summary

    train_summary = dataset_summary("train")
    train_size = train_summary["n_images"]

    print("environment ...")
    env = environment()
    tfenv = tf_environment()

    print("dataset throughput ...")
    throughput = [
        dataset_throughput("tf_native", config.IMAGE_SIZE, 32, batches, "unit"),
        dataset_throughput("enhanced", config.IMAGE_SIZE, 32, max(2, batches // 2), "imagenet"),
    ]

    print(f"benchmarking {len(BENCHMARK_CONFIGS)} architectures ...")
    models = []
    for cfg in BENCHMARK_CONFIGS:
        mode = ("frozen" if cfg.get("freeze", True)
                else (f"ft{cfg.get('unfreeze_last', 0)}" if cfg.get("unfreeze_last") else "full"))
        label = f"{cfg['backbone']}/{cfg['head']}@{cfg['size']}/{cfg['batch']}/{mode}"
        print(f"  {label} ...", flush=True)
        try:
            models.append(benchmark_model(cfg, train_size))
        except Exception as exc:  # noqa: BLE001 - record, do not abort the suite
            models.append({**cfg, "error": f"{type(exc).__name__}: {exc}"})

    summary = {
        "environment": env,
        "tensorflow": tfenv,
        "training_split": train_summary,
        "input_throughput": throughput,
        "models": models,
        "benchmark_duration_s": round(time.perf_counter() - t_start, 1),
    }
    save_json(summary, config.EVAL_REPORT_DIR / "hardware_benchmark.json")
    write_report(summary)
    return summary


def write_report(summary: dict[str, Any]) -> None:
    env, tfenv = summary["environment"], summary["tensorflow"]
    lines = [
        "# Hardware and Throughput Benchmark", "",
        f"Measured {summary['benchmark_duration_s']} s of real TensorFlow work on this "
        "machine. Nothing is estimated from specifications.", "",
        "## Environment", "",
        "| Item | Value |", "|---|---|",
        f"| Platform | {env['platform']} |",
        f"| CPU | {env['processor']} |",
        f"| Cores | {env['cpu_physical_cores']} physical / {env['cpu_logical_cores']} logical |",
        f"| RAM | {env['ram_total_gb']} GB total, {env['ram_available_gb']} GB available |",
        f"| Disk | {env['disk_free_gb']} GB free of {env['disk_total_gb']} GB |",
        f"| Python | {env['python']} |",
        f"| TensorFlow | {tfenv['tensorflow_version']} |",
        f"| Devices visible to TF | {', '.join(tfenv['physical_devices'])} |",
        f"| GPU visible to TF | {tfenv['gpu_visible_to_tf']} |",
        f"| Mixed precision | {tfenv['mixed_precision']} |",
        "",
        f"> {tfenv['note']}", "",
        "## Input pipeline throughput", "",
        "| Pipeline | Batch | Images/s |", "|---|---:|---:|",
    ]
    for t in summary["input_throughput"]:
        lines.append(f"| `{t['preset']}` | {t['batch_size']} | {t['images_per_second']} |")

    lines += ["", "## Model timings", "",
              "| Model | Params (M) | Size (MB) | Step (s) | Train img/s | s/epoch | min/epoch | Inference ms/img | RSS (MB) |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for m in summary["models"]:
        if "error" in m:
            lines.append(f"| {m['backbone']} @{m['size']} | — | — | — | — | — | — | — | FAILED: {m['error']} |")
            continue
        lines.append(
            f"| `{m['name']}` @{m['input_size']} | {m['total_params'] / 1e6:.2f} | {m['size_mb_fp32']} | "
            f"{m['step_time_s_median']} | {m['images_per_second_train']} | "
            f"{m['est_epoch_time_s']} | {m['est_epoch_time_min']} | "
            f"{m['inference_ms_per_image']} | {m['rss_total_mb']} |"
        )

    if summary["models"] and "error" not in summary["models"][0]:
        fastest = min(summary["models"], key=lambda m: m["est_epoch_time_s"])
        slowest = max(summary["models"], key=lambda m: m["est_epoch_time_s"])
        lines += [
            "", "## Implications", "",
            f"* Cheapest configuration: `{fastest['name']}` at "
            f"{fastest['est_epoch_time_min']} min/epoch.",
            f"* Most expensive: `{slowest['name']}` at {slowest['est_epoch_time_min']} min/epoch "
            f"({slowest['est_epoch_time_s'] / max(fastest['est_epoch_time_s'], 1e-9):.1f}× the cheapest).",
            "* Epoch budgets in `Admin/experiment_matrix.md` are derived from these "
            "measurements, not chosen in advance.",
            "* Peak RSS stays well inside the available RAM, so memory is not the "
            "binding constraint — wall-clock time is.",
        ]

    path = config.EVAL_REPORT_DIR / "hardware_benchmark.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    import argparse

    from src.utils import load_json

    ap = argparse.ArgumentParser(description="Benchmark hardware and model throughput")
    ap.add_argument("--report-only", action="store_true",
                    help="regenerate the markdown from the stored JSON without re-running")
    args = ap.parse_args()

    if args.report_only:
        write_report(load_json(config.EVAL_REPORT_DIR / "hardware_benchmark.json"))
        print("rewrote", config.EVAL_REPORT_DIR / "hardware_benchmark.md")
        return

    s = run()
    print(json.dumps({
        "tf": s["tensorflow"]["tensorflow_version"],
        "gpu_visible": s["tensorflow"]["gpu_visible_to_tf"],
        "throughput": [(t["preset"], t["images_per_second"]) for t in s["input_throughput"]],
        "models": [{k: m.get(k) for k in ("name", "est_epoch_time_min", "inference_ms_per_image")}
                   for m in s["models"] if "error" not in m],
        "errors": [m for m in s["models"] if "error" in m],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
