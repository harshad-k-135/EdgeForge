# YOLO TFLite benchmark report

- Device: **macOS-26.6.2-arm64-arm-64bit** (10 cores, governor `None`, None MHz)
- Runtime: `ai_edge_litert.interpreter` (default threading)
- Temperature start/end: None / None C; throttled flags start/end: None / None
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on the Pi)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_int8 | 3.086 | 640x640 | float32 | 3.2 | 28.9 | 28.4 | 4.7 | 36.3 | 32.7 | 30.6 |
| yolo26n_int8 | 2.931 | 640x640 | float32 | 2.9 | 22.5 | 22.1 | 2.2 | 26.2 | 25.9 | 38.6 |
| yolov8n_int8 | 3.485 | 640x640 | float32 | 2.3 | 21.9 | 20.8 | 2.9 | 30.3 | 24.6 | 40.6 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| yolo11n_int8 | 0.779 | 0.572 | 0.573 | 0.716 | 0.637 | 0.467 | 0.960 | 0.716 @ 0.51 |
| yolo26n_int8 | 0.751 | 0.580 | 0.668 | 0.712 | 0.689 | 0.526 | 0.930 | 0.702 @ 0.32 |
| yolov8n_int8 | 0.729 | 0.506 | 0.588 | 0.686 | 0.633 | 0.463 | 0.910 | 0.672 @ 0.43 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.
