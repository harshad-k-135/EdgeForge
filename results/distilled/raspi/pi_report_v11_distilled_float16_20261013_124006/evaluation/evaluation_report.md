# YOLO TFLite benchmark report

- Device: **Raspberry Pi 3 Model B Plus Rev 1.3** (4 cores, governor `ondemand`, 1400 MHz)
- Runtime: `ai_edge_litert.interpreter` (default threading)
- Temperature start/end: 46.2 / 48.3 C; throttled flags start/end: throttled=0x0 / throttled=0x0
- Test set: 100 single-class images, 229 ground-truth boxes, 35 classes present
- Confidence threshold for P/R/F1/accuracy: 0.25; NMS IoU: 0.7

## Latency (ms per image, measured on the Pi)

| Model | Size MB | Input | dtype | Preproc | Invoke mean | Invoke median | Invoke std | Invoke p95 | Total mean | FPS (total) |
|---|---|---|---|---|---|---|---|---|---|---|
| v11_distilled_float16 | 5.43 | 640x640 | float32 | 16.0 | 1355.7 | 1348.8 | 16.5 | 1399.9 | 1378.6 | 0.7 |

## Accuracy

| Model | mAP50 | mAP50-95 | Precision | Recall | F1 | Det. accuracy | Image top-1 acc. | Best F1 @ conf |
|---|---|---|---|---|---|---|---|---|
| v11_distilled_float16 | 0.644 | 0.476 | 0.583 | 0.629 | 0.605 | 0.434 | 0.850 | 0.647 @ 0.44 |

## Notes

- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.
- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.
- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).
- Image top-1 accuracy = the most confident detection has the image's (single) class.
- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.
