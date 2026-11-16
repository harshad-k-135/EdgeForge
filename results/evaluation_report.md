# Taylor-pruned YOLO TFLite Mac benchmark report

- Device: **macOS-26.6.2-arm64-arm-64bit** (10 cores, governor `None`, None MHz)
- Runtime: `ai_edge_litert.interpreter` (4 CPU threads; 5 warm-up invocations per model)
- Temperature start/end: None / None C; throttled flags start/end: None / None
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on this Mac)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| pruned_taylor_11_fp32 | 10.72 | 640x640 | float32 | 2.4 | 87.2 | 85.5 | 10.7 | 103.8 | 90.7 | 11.0 |
| pruned_taylor_11_int8 | 2.978 | 640x640 | int8 | 3.2 | 34.9 | 33.4 | 6.3 | 40.9 | 38.3 | 26.1 |
| pruned_taylor_26_fp32 | 9.917 | 640x640 | float32 | 2.2 | 75.1 | 65.2 | 18.4 | 109.2 | 78.1 | 12.8 |
| pruned_taylor_26_int8 | 2.801 | 640x640 | int8 | 3.1 | 24.6 | 24.8 | 3.5 | 30.7 | 27.9 | 35.9 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| pruned_taylor_11_fp32 | 0.714 | 0.536 | 0.636 | 0.642 | 0.639 | 0.470 | 0.890 | 0.681 @ 0.39 |
| pruned_taylor_11_int8 | 0.699 | 0.510 | 0.597 | 0.659 | 0.627 | 0.456 | 0.880 | 0.668 @ 0.39 |
| pruned_taylor_26_fp32 | 0.762 | 0.582 | 0.590 | 0.686 | 0.634 | 0.464 | 0.900 | 0.701 @ 0.40 |
| pruned_taylor_26_int8 | 0.780 | 0.564 | 0.610 | 0.690 | 0.648 | 0.479 | 0.890 | 0.668 @ 0.51 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.

## Run details

- Converted with `scripts/pt_to_tflite.py --precision both --beside-source` at 640×640; INT8 calibration used all 128 COCO128 images.
- Benchmarked with `scripts/pi_tflite_benchmark.py` on this Mac; evaluated with `scripts/evaluate_pi_report.py` on `Datasets/coco2017_test100`.
- All four models completed all 100 images with no benchmark failures.
- Source hashes and package versions: `../provenance.json`. Raw outputs, per-image timings, CSV summaries, and per-class AP files are saved alongside this report.
