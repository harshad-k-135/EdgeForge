# Taylor-pruned YOLO TFLite Raspberry Pi benchmark report

- Device: **Raspberry Pi 3 Model B Plus Rev 1.3** (same Raspberry Pi setup as the existing reports)
- Runtime: `ai_edge_litert.interpreter`; other evaluation settings unchanged, as confirmed by the user
- Test set: 100 single-class images from the COCO validation set, 229 ground-truth boxes, 35 classes present
- Input resolution: 640×640; warm-up: 5 invocations per model
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7
- Results source: user-reported Raspberry Pi latency and confirmation that accuracy matches the corresponding Mac Taylor results

## Latency (ms per image, reported from the Raspberry Pi)

| Model | Size MB | Input | dtype | Reported latency (s) | Reported latency (ms) | FPS (derived) |
|---|---|---|---|---|---|---|
| pruned_taylor_11_fp32 | 10.72 | 640x640 | float32 | 1.4 | 1400 | 0.71 |
| pruned_taylor_11_int8 | 2.978 | 640x640 | int8 | 0.9 | 900 | 1.11 |
| pruned_taylor_26_fp32 | 9.917 | 640x640 | float32 | 1.3 | 1300 | 0.77 |
| pruned_taylor_26_int8 | 2.801 | 640x640 | int8 | 0.8 | 800 | 1.25 |

## Accuracy

The user confirmed that Raspberry Pi accuracy matches the corresponding Mac results. The values below are reproduced from the [Mac Taylor evaluation report](../../../macm4/pi_report_v11n_v26n_taylor_fp32_int8_20261108/evaluation/evaluation_report.md); they were not independently recomputed from Raspberry Pi raw outputs for this report.

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| pruned_taylor_11_fp32 | 0.714 | 0.536 | 0.636 | 0.642 | 0.639 | 0.470 | 0.890 | 0.681 @ 0.39 |
| pruned_taylor_11_int8 | 0.699 | 0.510 | 0.597 | 0.659 | 0.627 | 0.456 | 0.880 | 0.668 @ 0.39 |
| pruned_taylor_26_fp32 | 0.762 | 0.582 | 0.590 | 0.686 | 0.634 | 0.464 | 0.900 | 0.701 @ 0.40 |
| pruned_taylor_26_int8 | 0.780 | 0.564 | 0.610 | 0.690 | 0.648 | 0.479 | 0.890 | 0.668 @ 0.51 |

## INT8 latency improvement over Taylor FP32

| Model | FP32 latency (ms) | INT8 latency (ms) | Latency reduction | Speedup |
|---|---|---|---|---|
| YOLO11n Taylor | 1400 | 900 | 35.7% | 1.56× |
| YOLO26n Taylor | 1300 | 800 | 38.5% | 1.63× |

## Notes

- Latencies are the rounded seconds-per-image values supplied by the user. Their timing scope (invoke-only versus total) was not specified, so they are labeled reported latency rather than invoke mean or total mean.
- FPS is calculated as 1 / reported latency in seconds. Latency reduction is (FP32 latency − INT8 latency) / FP32 latency; speedup is FP32 latency / INT8 latency. These comparisons measure quantization within the Taylor-pruned models.
- Per-image timings, preprocessing time, median, standard deviation, p95, temperature, and throttling telemetry were not supplied for this run and are not inferred from earlier runs.
- Model sizes, input details, and accuracy values correspond to the same model files evaluated on the Mac.
- mAP is averaged over the 35 classes present in this subset. Predictions of absent classes count as false positives in precision, F1, and detection accuracy, but do not affect mAP.
- Precision, recall, and F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy is the fraction of images whose most confident detection above the confidence threshold matches the image's single ground-truth class.
- The models predict all 80 COCO classes. This 100-image subset contains 35 classes; evaluation on the full COCO validation set remains future work.
