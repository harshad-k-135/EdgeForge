# EdgeForge

EdgeForge is a collection of experiments and utilities for reducing the cost of
YOLO object-detection models and measuring TensorFlow Lite inference on
resource-constrained devices. The repository focuses on YOLO11n and YOLO26n,
with workflows for distillation, pruning, TFLite conversion, Raspberry Pi
benchmarking, and offline evaluation of benchmark outputs.

The checkout contains scripts and selected evaluation reports. It does **not**
include the training/evaluation datasets or the detector checkpoints expected
by several scripts.

## What is here

| Area | Purpose |
| --- | --- |
| Distillation | A YOLO26n student-training example using a YOLO26x teacher. |
| Structured pruning | Uses Torch-Pruning dependency graphs and group magnitude importance to remove channels while retaining a runnable network. |
| Taylor-guided pruning | Scores dependency groups using one calibration forward/backward pass, then selects low-scoring channels for pruning. |
| Unstructured pruning | Applies gradual magnitude pruning during Ultralytics training using a cubic sparsity schedule. |
| TFLite export | Converts checkpoints to FP16, FP32, or INT8 TFLite models. |
| Device benchmarking | Runs TFLite models over images on a Raspberry Pi (or another supported host) and saves per-image timings and raw outputs. |
| Evaluation | Decodes saved model outputs and calculates detection metrics on a host machine. |

## Repository layout

```text
EdgeForge/
├── README.md
├── .gitignore
├── scripts/
│   ├── depgraphpru.py
│   ├── distill.py
│   ├── download_seg_models.py
│   ├── evaluate_pi_report.py
│   ├── pi_tflite_benchmark.py
│   ├── pt_to_tflite.py
│   ├── prune_structured.py
│   ├── prune_unstructured.py
│   ├── prune_utils.py
│   ├── test_depgraphpru.py
│   ├── taylor_group_scores.json
│   └── runs/detect/                 # saved Ultralytics run configurations
└── results/
    ├── base/                        # baseline TFLite evaluation reports
    ├── distilled/                   # distilled model reports
    └── quantized/                   # INT8 model reports
```

Generated model weights, datasets, and full benchmark captures are not tracked.
The checked-in result directories retain evaluation reports; the benchmark
script's raw `.npz` outputs and metadata need to be generated separately.

## Workflow

### 1. Prepare model weights and data

Install a Python environment with the packages needed for the workflow you plan
to run (see [Requirements](#requirements)). The pruning scripts default to
detector checkpoints at:

```text
models/base/v11/yolo11n.pt
models/base/v26/yolo26n.pt
```

Those detector checkpoints are not included. Supply compatible weights at
those paths or pass a checkpoint with the script's `--weights` or `--model`
option. Training and evaluation data are also user-provided; pruning commands
expect an Ultralytics detection dataset YAML.

`download_seg_models.py` is a separate helper that downloads **segmentation**
weights (`yolo11n-seg.pt`, `yolo11x-seg.pt`, `yolo26n-seg.pt`, and
`yolo26x-seg.pt`). It does not provide the detection checkpoint files used by
the pruning defaults:

```bash
python3 scripts/download_seg_models.py --out-dir models/base
```

### 2. Distill or prune

The distillation example is configured for a Kaggle notebook environment. It
trains a YOLO26n student from a YOLO26x teacher and expects a GPU, a
`coco_minitrain_10k.yaml` dataset file, and an Ultralytics installation that
supports the `distill_model` training argument:

```bash
python3 scripts/distill.py
```

Adapt the paths and training settings in `scripts/distill.py` for another
environment.

Structured channel pruning defaults to both nano detectors and targets a 30%
reduction in multiply-accumulate operations (MACs):

```bash
python3 scripts/prune_structured.py --model both --macs-reduction 0.30
```

Use `--model yolo11n` or `--model yolo26n` to select one model. `--weights`,
`--out-dir`, `--imgsz`, and the pruning parameters can also be supplied; run
`python3 scripts/prune_structured.py --help` for the full options. Checkpoints
are written under `models/pruned/structured/` by default.

Taylor-guided pruning calibrates on 32 validation images at 640 pixels by
default and uses the dataset configured by `--data` (default: Ultralytics'
`coco128.yaml`). The model and dataset must have the same number of classes:

```bash
python3 scripts/depgraphpru.py \
  --model models/base/v11/yolo11n.pt \
  --data path/to/dataset.yaml
```

The script writes pruned checkpoints, per-model score JSON, and pruning
summaries under `models/pruned/taylor/` by default. `--group-fraction` controls
the fraction of eligible dependency groups selected; `--channels-per-layer`
sets how many low-scoring channels to remove from a selected group.

Unstructured magnitude pruning runs a pruning-aware training job. For example:

```bash
python3 scripts/prune_unstructured.py \
  --model yolo11n \
  --data path/to/dataset.yaml \
  --epochs 50 \
  --final-sparsity 0.70
```

It saves the resulting checkpoint under `models/pruned/unstructured/` by
default. The model, dataset YAML, and training environment must be available
locally.

### 3. Export to TensorFlow Lite

`pt_to_tflite.py` accepts one or more checkpoints. For example:

```bash
python3 scripts/pt_to_tflite.py \
  --models models/pruned/structured/yolo11n_structured_30%.pt \
  --precision int8
```

Available precision choices are `fp16`, `fp32`, `int8`, and `both`. `fp16`
exports FP16 only, and `fp32` exports FP32 only. Both `int8` and `both` also
write an FP32 export before producing INT8; neither includes FP16. The default
output directory is `models/pruned/structured/`. INT8 conversion uses
representative images from `Datasets/coco128/images/train2017/`, so provide
those calibration images or adjust the path in the script. `--help` lists
output and reuse options.

### 4. Benchmark TFLite models on-device

On the benchmark device, install a compatible TFLite runtime, NumPy, and an
image backend (OpenCV or Pillow). Then pass `.tflite` files or directories
containing them:

```bash
python3 scripts/pi_tflite_benchmark.py \
  --models models/pruned/structured/yolo11n_int8.tflite \
  --images path/to/test/images \
  --threads 4
```

The default image directory is `./coco2017_test100/images`. The script saves a
timestamped `pi_report_*` directory containing system information, run
configuration, per-image timing metadata, and raw TFLite outputs. Use `--out`
to choose a destination, `--limit` for a short run, or `--cooldown` to add a
pause between models.

The recorded `invoke_ms` is the TFLite interpreter invocation only. `total_ms`
includes preprocessing and tensor copies; JPEG decoding and output
decode/NMS are excluded. Copy the report directory to the evaluation host if
the benchmark and evaluation run on different machines.

### 5. Evaluate the benchmark report

The evaluation dataset must contain YOLO-format labels in `labels/` and a
`classes.txt` file. Run the host-side evaluator with NumPy:

```bash
python3 scripts/evaluate_pi_report.py \
  --report path/to/pi_report_YYYYMMDD_HHMMSS \
  --dataset path/to/test_dataset
```

By default, output files are written to the report's `evaluation/` directory:
an evaluation summary in Markdown, CSV, and JSON, plus per-image and
per-class-AP CSV files. The evaluator supports common YOLO output layouts,
de-quantizes integer outputs, and maps boxes back from letterboxed input
coordinates. Precision, recall, F1, and detection accuracy use the configured
confidence threshold (`0.25` by default) and IoU 0.5; mAP uses ten IoU
thresholds from 0.50 to 0.95 and averages over classes present in the ground
truth.

## Example results

The table below summarizes selected checked-in reports. Model and device
variants differ across runs; these values are reference measurements, not
reproducible guarantees. The reports use 100 single-class images, 229 ground
truth boxes, and 35 classes. Their small sample size means per-class AP and
small differences should be interpreted cautiously.

| Report group | Device as identified by report | Model | Format | Model size (MB) | Total mean (ms/image) | mAP50 | mAP50-95 | F1 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| [Base FP16](results/base/macm4/pi_report_v11n_v26n_fp16_20261022/evaluation/evaluation_report.md) | macOS Apple Silicon (report folder tagged `macm4`) | YOLO11n | FP16 | 5.435 | 65.1 | 0.804 | 0.634 | 0.688 |
| [Base FP16](results/base/macm4/pi_report_v11n_v26n_fp16_20261022/evaluation/evaluation_report.md) | macOS Apple Silicon (report folder tagged `macm4`) | YOLO26n | FP16 | 5.038 | 61.4 | 0.784 | 0.642 | 0.720 |
| [Quantized Mac report](results/quantized/macm4/pi_report_v8n_v11n_v26n_20261013_0037/evaluation/evaluation_report.md) | macOS Apple Silicon (report folder tagged `macm4`) | YOLO11n | INT8 | 3.086 | 32.7 | 0.779 | 0.572 | 0.637 |
| [Quantized Mac report](results/quantized/macm4/pi_report_v8n_v11n_v26n_20261013_0037/evaluation/evaluation_report.md) | macOS Apple Silicon (report folder tagged `macm4`) | YOLO26n | INT8 | 2.931 | 25.9 | 0.751 | 0.580 | 0.689 |
| [Quantized Raspberry Pi report](results/quantized/raspi/pi_report_yolo11n_int8_20261013_134748/evaluation/evaluation_report.md) | Raspberry Pi 3 Model B Plus | YOLO11n | INT8 | 3.086 | 962.5 | 0.779 | 0.574 | 0.637 |
| [Quantized Raspberry Pi report](results/quantized/raspi/pi_report_yolo26n_int8_20261013_135851/evaluation/evaluation_report.md) | Raspberry Pi 3 Model B Plus | YOLO26n | INT8 | 2.931 | 867.9 | 0.779 | 0.613 | 0.679 |
| [Distilled Raspberry Pi report](results/distilled/raspi/pi_report_v11_distilled_float16_20261013_124006/evaluation/evaluation_report.md) | Raspberry Pi 3 Model B Plus | YOLO11n distilled | FP16 | 5.430 | 1,378.6 | 0.644 | 0.476 | 0.605 |
| [Distilled Raspberry Pi report](results/distilled/raspi/pi_report_v26_distilled_float16_20261013_124434/evaluation/evaluation_report.md) | Raspberry Pi 3 Model B Plus | YOLO26n distilled | FP16 | 5.029 | 1,223.1 | 0.779 | 0.613 | 0.655 |

`Total mean` includes preprocessing and tensor copies, but not output decoding
or NMS. The corresponding report files contain invocation-only latency,
precision, recall, confidence settings, and device telemetry. In particular,
the quantized Mac report's heading says latency was measured on a Pi, while its
device metadata identifies macOS Apple Silicon; verify the report source when
making cross-device comparisons.

## Requirements

There is no checked-in dependency manifest or pinned environment. Install
versions compatible with your Python version, operating system, and hardware.
Requirements vary by task:

- **Pruning and training:** PyTorch, Ultralytics, and Torch-Pruning. CUDA or
  Apple MPS may accelerate work where supported; CPU execution may be slower.
- **Taylor-guided pruning:** additionally uses Ultralytics dataset/training
  APIs and requires a validation dataset with matching model class count.
- **TFLite export:** Ultralytics export dependencies; INT8 conversion also uses
  TensorFlow, NumPy, and OpenCV.
- **Device benchmarking:** NumPy, a TFLite runtime (`ai-edge-litert` or
  `tflite-runtime`), and OpenCV or Pillow.
- **Offline report evaluation:** NumPy only.

## Tests

The repository includes unit tests for Taylor-guided runtime group selection:

```bash
python3 -m unittest discover -s scripts
```

The test imports the pruning script and therefore requires its PyTorch,
Ultralytics, and Torch-Pruning dependencies.

## Notes and limitations

- Several scripts use fixed default model, dataset, or calibration-image paths.
  Supply the required files or adapt those paths before running.
- `scripts/runs/detect/*/args.yaml` are saved training configurations, not a
  complete training dataset or a portable environment specification.
- The Taylor score file under `scripts/` is one saved calibration result; it is
  not a substitute for rerunning calibration on your data and weights.
- Evaluation reports cover different model variants and devices. Do not treat
  results from separate runs as a controlled comparison unless the exact
  weights, dataset, runtime, and settings are matched.
- No project license file is included. Check the licenses for model weights,
  datasets, and dependencies separately before using or redistributing them.
