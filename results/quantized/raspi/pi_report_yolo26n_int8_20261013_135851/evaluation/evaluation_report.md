# YOLO TFLite benchmark report

- Device: **Raspberry Pi 3 Model B Plus Rev 1.3** (4 cores, governor `ondemand`, 1400 MHz)
- Runtime: `ai_edge_litert.interpreter` (default threading)
- Temperature start/end: 47.8 / 50.5 C; throttled flags start/end: throttled=0x80000 / throttled=0x80000
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on the Pi)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n_int8 | 2.931 | 640x640 | float32 | 23.3 | 838.0 | 837.8 | 1.1 | 840.2 | 867.9 | 1.2 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| yolo26n_int8 | 0.779 | 0.613 | 0.653 | 0.707 | 0.679 | 0.514 | 0.930 | 0.701 @ 0.32 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.
