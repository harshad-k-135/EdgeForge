# YOLO TFLite benchmark report

- Device: **Raspberry Pi 3 Model B Plus Rev 1.3** (4 cores, governor `ondemand`, 1400 MHz)
- Runtime: `ai_edge_litert.interpreter` (default threading)
- Temperature start/end: 47.2 / 52.1 C; throttled flags start/end: throttled=0x80000 / throttled=0x80000
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on the Pi)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo11n_int8 | 3.086 | 640x640 | float32 | 23.2 | 932.7 | 932.5 | 0.9 | 934.7 | 962.5 | 1.0 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| yolo11n_int8 | 0.779 | 0.574 | 0.576 | 0.712 | 0.637 | 0.467 | 0.960 | 0.709 @ 0.47 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.
