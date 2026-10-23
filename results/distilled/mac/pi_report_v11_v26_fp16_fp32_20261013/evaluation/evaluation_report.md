# YOLO TFLite benchmark report

- Device: **macOS-26.6.2-arm64-arm-64bit** (10 cores, governor `None`, None MHz)
- Runtime: `ai_edge_litert.interpreter` (default threading)
- Temperature start/end: None / None C; throttled flags start/end: None / None
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on the Pi)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| v11_distilled_float16 | 5.43 | 640x640 | float32 | 1.8 | 67.0 | 66.7 | 1.8 | 70.1 | 69.4 | 14.4 |
| v11_distilled_float32 | 10.743 | 640x640 | float32 | 1.7 | 68.4 | 66.6 | 4.7 | 78.8 | 70.6 | 14.2 |
| v26_distilled_float16 | 5.029 | 640x640 | float32 | 1.7 | 59.8 | 57.8 | 9.8 | 66.1 | 62.0 | 16.1 |
| v26_distilled_float32 | 9.924 | 640x640 | float32 | 1.7 | 58.2 | 57.2 | 3.8 | 64.0 | 60.5 | 16.5 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| v11_distilled_float16 | 0.644 | 0.476 | 0.583 | 0.629 | 0.605 | 0.434 | 0.850 | 0.647 @ 0.44 |
| v11_distilled_float32 | 0.644 | 0.475 | 0.583 | 0.629 | 0.605 | 0.434 | 0.850 | 0.647 @ 0.44 |
| v26_distilled_float16 | 0.779 | 0.613 | 0.682 | 0.629 | 0.655 | 0.486 | 0.880 | 0.686 @ 0.43 |
| v26_distilled_float32 | 0.779 | 0.611 | 0.682 | 0.629 | 0.655 | 0.486 | 0.880 | 0.686 @ 0.43 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.
