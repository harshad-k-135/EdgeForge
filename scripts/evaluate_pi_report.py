#!/usr/bin/env python3
"""
evaluate_pi_report.py  --  RUN THIS ON YOUR MAC (needs only numpy)
===================================================================
Takes the report folder produced by `pi_tflite_benchmark.py` (after you scp it over) plus the
test dataset's labels, and computes everything the Pi deliberately did not:

    * latency per image / summary stats (from the Pi timings)
    * decoding + NMS of the raw TFLite outputs
    * mAP@0.5, mAP@0.5:0.95 (Ultralytics-style: 101-point AP, 10 IoU thresholds)
    * precision, recall, F1 (at a confidence threshold, plus the best-F1 operating point)
    * detection accuracy and image-level top-1 class accuracy

Usage:
    python3 evaluate_pi_report.py --report ~/Downloads/pi_report_20261011_120000 \
                                  --dataset ./coco2017_test100

--dataset must contain labels/*.txt (YOLO format) and classes.txt, as written by
build_coco_single_class_test.py. Results go to <report>/evaluation/.

Supported raw output layouts (auto-detected, int8/uint8 outputs are de-quantised):
    [1, 4+nc, N] / [1, N, 4+nc]   YOLOv8 / v9 / v11 style (class scores, no objectness)
    [1, 5+nc, N] / [1, N, 5+nc]   YOLOv5 / v7 style (objectness + class scores)
    [1, 300, 6]  / [1, 6, 300]    end-to-end / NMS-free heads: x1, y1, x2, y2, conf, class
Boxes may be normalised (0-1) or in input pixels; this is detected automatically.

Metric definitions (all IoU thresholds on original-resolution boxes):
    mAP50 / mAP50-95 : mean over classes that appear in the ground truth (same as Ultralytics val)
    P / R / F1       : micro-averaged over all instances, IoU 0.5, predictions with conf >= --conf
    Detection acc.   : TP / (TP + FP + FN) at IoU 0.5, conf >= --conf
    Image top-1 acc. : the most confident detection (conf >= --conf) has the image's class
                       (valid because every test image contains a single class)
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

TRAPZ = getattr(np, "trapezoid", None) or np.trapz
IOUV = np.linspace(0.5, 0.95, 10)


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_classes(dataset: Path):
    f = dataset / "classes.txt"
    if not f.is_file():
        sys.exit(f"ERROR: {f} not found")
    return [l.strip() for l in f.read_text().splitlines() if l.strip()]


def load_gt(label_file: Path, W: int, H: int):
    if not label_file.is_file():
        return np.zeros(0, int), np.zeros((0, 4))
    rows = [l.split() for l in label_file.read_text().splitlines() if l.strip()]
    if not rows:
        return np.zeros(0, int), np.zeros((0, 4))
    a = np.array(rows, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h = a[:, 1] * W, a[:, 2] * H, a[:, 3] * W, a[:, 4] * H
    return a[:, 0].astype(int), np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


# --------------------------------------------------------------------------- #
# Decoding + NMS
# --------------------------------------------------------------------------- #
def dequantize(raw: np.ndarray, scale: float, zero_point: float) -> np.ndarray:
    if raw.dtype in (np.int8, np.uint8) and scale:
        return (raw.astype(np.float32) - zero_point) * scale
    return raw.astype(np.float32)


def greedy_nms(boxes: np.ndarray, iou_thres: float, max_det: int) -> np.ndarray:
    """boxes sorted by score (desc). Returns kept indices."""
    x1, y1, x2, y2 = boxes.T
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    idx, keep = np.arange(len(boxes)), []
    while idx.size and len(keep) < max_det:
        k = idx[0]
        keep.append(k)
        rest = idx[1:]
        w = np.clip(np.minimum(x2[k], x2[rest]) - np.maximum(x1[k], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[k], y2[rest]) - np.maximum(y1[k], y1[rest]), 0, None)
        inter = w * h
        iou = inter / (areas[k] + areas[rest] - inter + 1e-9)
        idx = rest[iou <= iou_thres]
    return np.array(keep, dtype=int)


def decode_and_nms(out: np.ndarray, nc: int, in_w: int, in_h: int, nms_iou: float, max_det: int, conf_floor: float = 0.001):
    """Returns (xyxy in model-input pixels, conf, cls), sorted by confidence (desc)."""
    a = out[0] if out.ndim == 3 else out
    if a.ndim != 2:
        raise ValueError(f"unexpected output shape {out.shape}")
    r, c = a.shape
    scale = np.array([in_w, in_h, in_w, in_h], dtype=np.float32)

    # --- end-to-end head: (N, 6) = x1 y1 x2 y2 conf cls
    if 6 in (r, c) and nc + 4 != 6 and nc + 5 != 6:
        if r == 6 and c != 6:
            a = a.T
        boxes = a[:, :4] * (scale if a[:, :4].max() <= 2.0 else 1.0)
        conf, cls = a[:, 4], a[:, 5].astype(int)
        m = (conf > conf_floor) & (cls >= 0) & (cls < nc)
        boxes, conf, cls = boxes[m], conf[m], cls[m]
        o = np.argsort(-conf)[:max_det]
        return boxes[o], conf[o], cls[o]

    # --- anchor-based head
    ch = (nc + 4, nc + 5)
    if r in ch and c not in ch:
        a = a.T  # channels-first -> (N, C)
    elif c in ch and r not in ch:
        pass  # already (N, C)
    elif r in ch and c in ch:
        a = a.T if c > r else a  # ambiguous: treat the larger axis as the anchor axis
    else:
        raise ValueError(f"output shape {out.shape} does not match nc={nc} (expected {nc + 4} or {nc + 5} channels)")
    boxes = a[:, :4]
    scores = a[:, 4:5] * a[:, 5:] if a.shape[1] == nc + 5 else a[:, 4:]
    if boxes.max() <= 2.0:  # normalised xywh -> input pixels
        boxes = boxes * scale
    xyxy = np.concatenate([boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2], 1)

    # multi-label candidates (as in Ultralytics validation), class-aware NMS
    i, j = np.nonzero(scores > conf_floor)
    if i.size == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
    conf = scores[i, j]
    if conf.size > 30000:
        top = np.argpartition(-conf, 30000)[:30000]
        i, j, conf = i[top], j[top], conf[top]
    o = np.argsort(-conf)
    i, j, conf = i[o], j[o], conf[o]
    b = xyxy[i]
    keep = greedy_nms(b + j[:, None].astype(np.float32) * 8192.0, nms_iou, max_det)
    return b[keep], conf[keep], j[keep]


def to_original(xyxy: np.ndarray, rec: dict) -> np.ndarray:
    b = xyxy.astype(np.float64).copy()
    b[:, [0, 2]] = (b[:, [0, 2]] - rec["pad_left"]) / rec["ratio"]
    b[:, [1, 3]] = (b[:, [1, 3]] - rec["pad_top"]) / rec["ratio"]
    b[:, [0, 2]] = b[:, [0, 2]].clip(0, rec["orig_w"])
    b[:, [1, 3]] = b[:, [1, 3]].clip(0, rec["orig_h"])
    return b


# --------------------------------------------------------------------------- #
# Matching + AP (Ultralytics-compatible)
# --------------------------------------------------------------------------- #
def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match_predictions(pred_cls: np.ndarray, gt_cls: np.ndarray, iou: np.ndarray) -> np.ndarray:
    """(P, 10) boolean matrix: prediction i is a true positive at IoU threshold j."""
    correct = np.zeros((len(pred_cls), len(IOUV)), dtype=bool)
    if len(pred_cls) == 0 or len(gt_cls) == 0:
        return correct
    iou = iou * (gt_cls[:, None] == pred_cls[None, :])
    for k, thr in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= thr)).T  # (gt_idx, pred_idx)
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1], k] = True
    return correct


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(TRAPZ(np.interp(x, mrec, mpre), x))


def ap_per_class(tp: np.ndarray, conf: np.ndarray, pred_cls: np.ndarray, target_cls: np.ndarray):
    o = np.argsort(-conf)
    tp, pred_cls = tp[o].astype(float), pred_cls[o]
    classes, n_gt = np.unique(target_cls, return_counts=True)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, c in enumerate(classes):
        m = pred_cls == c
        if not m.any():
            continue
        tpc = tp[m].cumsum(0)
        fpc = (1 - tp[m]).cumsum(0)
        recall = tpc / (n_gt[ci] + 1e-16)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return classes, n_gt, ap


def prf_at(conf: np.ndarray, tp50: np.ndarray, n_gt: int, t: float):
    m = conf >= t
    tp = int(tp50[m].sum())
    fp = int(m.sum()) - tp
    fn = n_gt - tp
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / n_gt if n_gt else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    acc = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "det_accuracy": acc}


# --------------------------------------------------------------------------- #
# Evaluate one model
# --------------------------------------------------------------------------- #
def evaluate_model(mdir: Path, dataset: Path, names: list, args):
    meta = json.loads((mdir / "meta.json").read_text())
    npz = np.load(mdir / "raw_outputs.npz")
    nc = len(names)
    qs, qz = meta["output"]["quantization"]
    in_w, in_h = meta["input"]["width"], meta["input"]["height"]

    tp_all, conf_all, cls_all, gt_cls_all, rows = [], [], [], [], []
    for rec in meta["images"]:
        out = dequantize(npz[f"img_{rec['stem']}"], qs, qz)
        xyxy, conf, cls = decode_and_nms(out, nc, in_w, in_h, args.nms_iou, args.max_det)
        xyxy = to_original(xyxy, rec)
        gt_cls, gt_xyxy = load_gt(dataset / "labels" / f"{rec['stem']}.txt", rec["orig_w"], rec["orig_h"])

        correct = match_predictions(cls, gt_cls, box_iou(gt_xyxy, xyxy))
        tp_all.append(correct); conf_all.append(conf); cls_all.append(cls); gt_cls_all.append(gt_cls)

        sel = conf >= args.conf
        tp = int(correct[sel, 0].sum())
        fp = int(sel.sum()) - tp
        gt_name = names[np.bincount(gt_cls).argmax()] if len(gt_cls) else ""
        top_cls = int(cls[sel][0]) if sel.any() else -1  # arrays are sorted by confidence
        rows.append({
            "image": rec["stem"], "gt_class": gt_name, "n_gt": len(gt_cls),
            "n_pred_at_conf": int(sel.sum()), "tp": tp, "fp": fp, "fn": len(gt_cls) - tp,
            "top1_pred_class": names[top_cls] if top_cls >= 0 else "(none)",
            "top1_conf": round(float(conf[sel][0]), 4) if sel.any() else 0.0,
            "top1_correct": bool(len(gt_cls) and top_cls == np.bincount(gt_cls).argmax()),
            "preprocess_ms": round(rec["preprocess_ms"], 3), "invoke_ms": round(rec["invoke_ms"], 3),
            "total_ms": round(rec["total_ms"], 3), "temp_c": rec.get("temp_c"),
        })

    tp = np.concatenate(tp_all) if tp_all else np.zeros((0, 10), bool)
    conf = np.concatenate(conf_all) if conf_all else np.zeros(0)
    pcls = np.concatenate(cls_all) if cls_all else np.zeros(0, int)
    gcls = np.concatenate(gt_cls_all) if gt_cls_all else np.zeros(0, int)
    n_gt = len(gcls)

    classes, n_per_class, ap = ap_per_class(tp, conf, pcls, gcls)
    fixed = prf_at(conf, tp[:, 0], n_gt, args.conf)
    sweep = [(t, prf_at(conf, tp[:, 0], n_gt, t)) for t in np.linspace(0.01, 0.99, 99)]
    best_t, best = max(sweep, key=lambda s: s[1]["f1"])

    lat = meta["latency_summary_ms"]
    top1 = float(np.mean([r["top1_correct"] for r in rows]))
    summary = {
        "model": meta["model_name"], "size_mb": meta["model_size_mb"],
        "input": f"{in_w}x{in_h}", "in_dtype": meta["input"]["dtype"],
        "preprocess_ms_mean": lat["preprocess_ms"]["mean"],
        "invoke_ms_mean": lat["invoke_ms"]["mean"], "invoke_ms_median": lat["invoke_ms"]["median"],
        "invoke_ms_std": lat["invoke_ms"]["std"], "invoke_ms_p95": lat["invoke_ms"]["p95"],
        "invoke_ms_min": lat["invoke_ms"]["min"], "invoke_ms_max": lat["invoke_ms"]["max"],
        "total_ms_mean": lat["total_ms"]["mean"], "total_ms_p95": lat["total_ms"]["p95"],
        "fps_invoke": 1000 / lat["invoke_ms"]["mean"], "fps_total": 1000 / lat["total_ms"]["mean"],
        "mAP50": float(ap[:, 0].mean()) if len(ap) else 0.0,
        "mAP50_95": float(ap.mean()) if len(ap) else 0.0,
        "conf_threshold": args.conf,
        "precision": fixed["precision"], "recall": fixed["recall"], "f1": fixed["f1"],
        "detection_accuracy": fixed["det_accuracy"], "image_top1_accuracy": top1,
        "tp": fixed["tp"], "fp": fixed["fp"], "fn": fixed["fn"],
        "best_f1": best["f1"], "best_f1_conf": float(best_t),
        "best_f1_precision": best["precision"], "best_f1_recall": best["recall"],
        "num_images": len(rows), "num_gt_instances": n_gt, "num_gt_classes": len(classes),
    }
    per_class = [{
        "class": names[int(c)], "class_id": int(c), "instances": int(n), "AP50": float(a[0]), "AP50_95": float(a.mean()),
    } for c, n, a in zip(classes, n_per_class, ap)]
    return summary, rows, per_class


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def write_csv(path: Path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def write_markdown(path: Path, summaries: list, sysinfo: dict, args, failures: list):
    f2, f1 = (lambda v: f"{v:.2f}"), (lambda v: f"{v:.1f}")
    lat_rows = [[s["model"], s["size_mb"], s["input"], s["in_dtype"], f1(s["preprocess_ms_mean"]), f1(s["invoke_ms_mean"]),
                 f1(s["invoke_ms_median"]), f1(s["invoke_ms_std"]), f1(s["invoke_ms_p95"]), f1(s["total_ms_mean"]),
                 f1(s["fps_total"])] for s in summaries]
    acc_rows = [[s["model"], f"{s['mAP50']:.3f}", f"{s['mAP50_95']:.3f}", f"{s['precision']:.3f}", f"{s['recall']:.3f}",
                 f"{s['f1']:.3f}", f"{s['detection_accuracy']:.3f}", f"{s['image_top1_accuracy']:.3f}",
                 f"{s['best_f1']:.3f} @ {s['best_f1_conf']:.2f}"] for s in summaries]
    dev = sysinfo.get("device_model") or sysinfo.get("platform", "unknown device")
    lines = [
        "# YOLO TFLite benchmark report", "",
        f"- Device: **{dev}** ({sysinfo.get('cpu_count')} cores, governor `{sysinfo.get('cpu_governor')}`, "
        f"{sysinfo.get('cpu_freq_mhz')} MHz)",
        f"- Runtime: `{sysinfo.get('tflite_runtime_module')}` (default threading)",
        f"- Temperature start/end: {sysinfo.get('temp_c_start')} / {sysinfo.get('temp_c_end')} C; "
        f"throttled flags start/end: {sysinfo.get('throttled_start')} / {sysinfo.get('throttled_end')}",
        f"- Test set: {summaries[0]['num_images'] if summaries else 0} single-class images, "
        f"{summaries[0]['num_gt_instances'] if summaries else 0} ground-truth boxes, "
        f"{summaries[0]['num_gt_classes'] if summaries else 0} classes present",
        f"- Confidence threshold for P/R/F1/accuracy: {args.conf}; NMS IoU: {args.nms_iou}", "",
        "## Latency (ms per image, measured on the Pi)", "",
        md_table(["Model", "Size MB", "Input", "dtype", "Preproc", "Invoke mean", "Invoke median", "Invoke std",
                  "Invoke p95", "Total mean", "FPS (total)"], lat_rows), "",
        "## Accuracy", "",
        md_table(["Model", "mAP50", "mAP50-95", "Precision", "Recall", "F1", "Det. accuracy", "Image top-1 acc.",
                  "Best F1 @ conf"], acc_rows), "",
        "## Notes", "",
        "- *Invoke* is `interpreter.invoke()` only; *Total* adds preprocessing and tensor copies. Decode/NMS time is not included.",
        "- mAP is averaged over classes present in the ground truth (Ultralytics convention). Predictions of classes absent from "
        "the 100 images do not affect mAP, but they do count as false positives in precision, F1 and detection accuracy.",
        "- Precision / recall / F1 are micro-averaged at IoU 0.5; detection accuracy = TP / (TP + FP + FN).",
        "- Image top-1 accuracy = the most confident detection has the image's (single) class.",
        "- With only 100 images, many per-class APs rest on a handful of boxes; treat small differences with caution.",
    ]
    if failures:
        lines += ["", "## Models that failed on the Pi", ""] + [f"- `{x['model']}`: {x['error']}" for x in failures]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True, help="folder copied from the Pi (pi_report_...)")
    ap.add_argument("--dataset", type=Path, required=True, help="test dataset folder with labels/ and classes.txt")
    ap.add_argument("--out", type=Path, default=None, help="default: <report>/evaluation")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold for P/R/F1/accuracy (default 0.25)")
    ap.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold (default 0.7, Ultralytics val default)")
    ap.add_argument("--max-det", type=int, default=300)
    args = ap.parse_args()

    names = load_classes(args.dataset)
    model_dirs = sorted(d for d in args.report.iterdir() if (d / "meta.json").is_file())
    if not model_dirs:
        sys.exit(f"ERROR: no model result folders (with meta.json) found in {args.report}")
    out = args.out or args.report / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    sysinfo = json.loads((args.report / "system_info.json").read_text()) if (args.report / "system_info.json").is_file() else {}
    cfg = json.loads((args.report / "run_config.json").read_text()) if (args.report / "run_config.json").is_file() else {}

    summaries = []
    for d in model_dirs:
        print(f"Evaluating {d.name} ...", flush=True)
        try:
            s, rows, per_class = evaluate_model(d, args.dataset, names, args)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            continue
        summaries.append(s)
        write_csv(out / f"{d.name}_per_image.csv", rows)
        write_csv(out / f"{d.name}_per_class_ap.csv", sorted(per_class, key=lambda r: -r["AP50_95"]))
        print(f"  mAP50 {s['mAP50']:.3f} | mAP50-95 {s['mAP50_95']:.3f} | P {s['precision']:.3f} | R {s['recall']:.3f} | "
              f"F1 {s['f1']:.3f} | invoke {s['invoke_ms_mean']:.1f} ms")

    if not summaries:
        sys.exit("ERROR: no model could be evaluated.")
    write_csv(out / "evaluation_summary.csv", summaries)
    (out / "evaluation_summary.json").write_text(json.dumps(summaries, indent=2))
    write_markdown(out / "evaluation_report.md", summaries, sysinfo, args, cfg.get("failures", []))
    print(f"\nWrote results to {out.resolve()}:")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()