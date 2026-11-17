# Hardware and Throughput Benchmark

Measured 84.8 s of real TensorFlow work on this machine. Nothing is estimated from specifications.

## Environment

| Item | Value |
|---|---|
| Platform | Windows-11-10.0.26300-SP0 |
| CPU | Intel64 Family 6 Model 191 Stepping 2, GenuineIntel |
| Cores | 10 physical / 16 logical |
| RAM | 15.71 GB total, 6.24 GB available |
| Disk | 40.1 GB free of 474.7 GB |
| Python | 3.12.16 |
| TensorFlow | 2.21.0 |
| Devices visible to TF | CPU:/physical_device:CPU:0 |
| GPU visible to TF | False |
| Mixed precision | float32 |

> No GPU is visible to TensorFlow on native Windows; the installed tensorflow-cpu build has no CUDA support and TF >= 2.11 does not ship native Windows GPU binaries.

## Input pipeline throughput

| Pipeline | Batch | Images/s |
|---|---:|---:|
| `tf_native` | 32 | 3614.7 |
| `enhanced` | 32 | 668.0 |

## Model timings

| Model | Params (M) | Size (MB) | Step (s) | Train img/s | s/epoch | min/epoch | Inference ms/img | RSS (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `custom_cnn_single` @224 | 0.42 | 1.62 | 1.285 | 24.9 | 172.2 | 2.87 | 8.47 | 1980.0 |
| `mobilenetv2_single_frozen` @224 | 2.42 | 9.24 | 0.361 | 88.6 | 48.4 | 0.81 | 19.39 | 2140.5 |
| `mobilenetv2_single_finetune` @224 | 2.42 | 9.24 | 0.41 | 78.0 | 54.9 | 0.92 | 18.46 | 3554.1 |
| `mobilenetv2_single_full` @224 | 2.42 | 9.24 | 1.843 | 17.4 | 247.0 | 4.12 | 15.95 | 3855.9 |
| `resnet50_single_frozen` @224 | 23.85 | 90.98 | 0.723 | 44.3 | 96.8 | 1.61 | 37.4 | 4028.9 |
| `resnet50_single_frozen` @160 | 23.85 | 90.98 | 0.408 | 78.3 | 54.7 | 0.91 | 24.95 | 6087.2 |
| `mobilenetv2_multitask_frozen` @224 | 2.59 | 9.87 | 0.257 | 124.6 | 34.4 | 0.57 | 15.62 | 6220.4 |

## Implications

* Cheapest configuration: `mobilenetv2_multitask_frozen` at 0.57 min/epoch.
* Most expensive: `mobilenetv2_single_full` at 4.12 min/epoch (7.2× the cheapest).
* Epoch budgets in `Admin/experiment_matrix.md` are derived from these measurements, not chosen in advance.
* Peak RSS stays well inside the available RAM, so memory is not the binding constraint — wall-clock time is.
