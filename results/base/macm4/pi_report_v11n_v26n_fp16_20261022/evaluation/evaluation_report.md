# YOLO TFLite benchmark report

- Device: **macOS-26.6.2-arm64-arm-64bit** (10 cores, governor `None`, None MHz)
- Runtime: `ai_edge_litert.interpreter` (default threading)
- Temperature start/end: None / None C; throttled flags start/end: None / None
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on the Pi)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_float16 | 5.435 | 640x640 | float32 | 1.5 | 63.1 | 62.2 | 4.3 | 66.2 | 65.1 | 15.4 |
| yolo26n_float16 | 5.038 | 640x640 | float32 | 1.6 | 59.3 | 56.9 | 8.6 | 79.4 | 61.4 | 16.3 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| yolo11n_float16 | 0.804 | 0.634 | 0.661 | 0.716 | 0.688 | 0.524 | 0.950 | 0.744 @ 0.49 |
| yolo26n_float16 | 0.784 | 0.642 | 0.733 | 0.707 | 0.720 | 0.562 | 0.950 | 0.740 @ 0.27 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.
