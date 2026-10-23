#!/usr/bin/env python3
"""
pi_tflite_benchmark.py  --  RUN THIS ON THE RASPBERRY PI
=========================================================
Runs one or more YOLO .tflite models over the 100-image test set and writes a report
folder that you copy to your Mac with scp. The Pi does ONLY what must happen on the
device: image preprocessing, TFLite inference, and timing. All the expensive work
(decoding, NMS, mAP, precision/recall/F1, tables) is done afterwards on the Mac by
`evaluate_pi_report.py`.

What gets timed (per image, perf_counter, after warm-up):
    preprocess_ms : letterbox resize + normalise/quantise the input
    invoke_ms     : interpreter.invoke() only            <- headline model latency
    total_ms      : preprocess + set_tensor + invoke + get_tensor
  (JPEG decoding and decode/NMS post-processing are NOT included; decoding is done once up
   front, and post-processing happens on the Mac.)

Report folder layout:
    pi_report_<timestamp>/
        system_info.json            Pi model, OS, CPU governor/frequency, temperature, throttling
        run_config.json             CLI args, models run, failures
        <model_name>/
            meta.json               model I/O details, per-image timings + letterbox params
            raw_outputs.npz         raw model output tensor for every image (as returned by TFLite)

Usage:
    python3 pi_tflite_benchmark.py --models yolov8n_float32.tflite yolov8n_int8.tflite
    python3 pi_tflite_benchmark.py --models ./models/ --images ./coco2017_test100/images --threads 4

Install (inside a venv on the Pi):
    pip install numpy opencv-python-headless ai-edge-litert      # or: tflite-runtime
"""

import argparse
import gc
import getpass
import importlib
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# --------------------------------------------------------------------------- #
# TFLite runtime + image backend (first one that imports wins)
# --------------------------------------------------------------------------- #
def get_interpreter_class():
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter", "tensorflow.lite"):
        try:
            return importlib.import_module(mod).Interpreter, mod
        except Exception:
            continue
    sys.exit("ERROR: no TFLite runtime found. Install one:  pip install ai-edge-litert  (or tflite-runtime)")


try:
    import cv2

    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
    try:
        from PIL import Image
    except ImportError:
        sys.exit("ERROR: need opencv (pip install opencv-python-headless) or Pillow.")
    print("WARNING: OpenCV not found, using Pillow (resizing differs very slightly from OpenCV).")


def read_rgb(path: Path) -> np.ndarray:
    if HAVE_CV2:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"cannot read {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.asarray(Image.open(path).convert("RGB"))


def resize(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if HAVE_CV2:
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.asarray(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def letterbox(img: np.ndarray, new_w: int, new_h: int, pad_value: int = 114):
    """Ultralytics-style letterbox. Returns (canvas, ratio, left_pad, top_pad)."""
    h, w = img.shape[:2]
    r = min(new_h / h, new_w / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (new_w - nw) / 2, (new_h - nh) / 2
    left, top = int(round(dw - 0.1)), int(round(dh - 0.1))
    canvas = np.full((new_h, new_w, 3), pad_value, np.uint8)
    canvas[top:top + nh, left:left + nw] = img if (nw, nh) == (w, h) else resize(img, nw, nh)
    return canvas, r, left, top


# --------------------------------------------------------------------------- #
# Input preparation (float32 / float16 / int8 / uint8 models)
# --------------------------------------------------------------------------- #
def make_input_fn(in_det: dict):
    shape = list(in_det["shape"])
    dtype = in_det["dtype"]
    scale, zero_point = in_det["quantization"]
    if len(shape) != 4:
        raise ValueError(f"unsupported input shape {shape}")
    nchw = shape[1] == 3 and shape[3] != 3
    h, w = (shape[2], shape[3]) if nchw else (shape[1], shape[2])

    if dtype == np.float32:
        def convert(c):
            return c.astype(np.float32) * np.float32(1 / 255.0)
    elif dtype == np.float16:
        def convert(c):
            return (c.astype(np.float32) * np.float32(1 / 255.0)).astype(np.float16)
    elif dtype in (np.int8, np.uint8):
        if scale and scale > 0:
            info = np.iinfo(dtype)
            lut = np.clip(np.round(np.arange(256) / 255.0 / scale + zero_point), info.min, info.max).astype(dtype)

            def convert(c):
                return lut[c]
        else:  # raw uint8 model, expects 0..255
            def convert(c):
                return c.astype(dtype)
    else:
        raise ValueError(f"unsupported input dtype {dtype}")

    def prep(canvas: np.ndarray) -> np.ndarray:
        if nchw:
            canvas = canvas.transpose(2, 0, 1)
        return np.ascontiguousarray(convert(canvas)[None])

    return prep, w, h, nchw


# --------------------------------------------------------------------------- #
# Cheap system telemetry (read once per image, outside the timed region)
# --------------------------------------------------------------------------- #
def read_text(path: str):
    try:
        with open(path) as f:
            return f.read().strip().replace("\x00", "")
    except OSError:
        return None


def cpu_temp_c():
    v = read_text("/sys/class/thermal/thermal_zone0/temp")
    return round(int(v) / 1000, 1) if v and v.lstrip("-").isdigit() else None


def vcgencmd(*a):
    try:
        return subprocess.run(["vcgencmd", *a], capture_output=True, text=True, timeout=2).stdout.strip() or None
    except Exception:
        return None


def system_info(interp_module: str) -> dict:
    cur = read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
    mx = read_text("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq")
    mem = read_text("/proc/meminfo") or ""
    mem_kb = next((int(l.split()[1]) for l in mem.splitlines() if l.startswith("MemTotal")), None)
    return {
        "hostname": socket.gethostname(),
        "device_model": read_text("/proc/device-tree/model"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__ if HAVE_CV2 else None,
        "tflite_runtime_module": interp_module,
        "cpu_count": os.cpu_count(),
        "cpu_governor": read_text("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "cpu_freq_mhz": int(cur) // 1000 if cur and cur.isdigit() else None,
        "cpu_max_freq_mhz": int(mx) // 1000 if mx and mx.isdigit() else None,
        "mem_total_mb": mem_kb // 1024 if mem_kb else None,
        "temp_c_start": cpu_temp_c(),
        "throttled_start": vcgencmd("get_throttled"),
    }


def latency_stats(vals):
    a = np.asarray(vals, dtype=np.float64)
    return {
        "mean": float(a.mean()), "median": float(np.median(a)), "std": float(a.std()),
        "min": float(a.min()), "max": float(a.max()),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
    }


# --------------------------------------------------------------------------- #
# Benchmark one model
# --------------------------------------------------------------------------- #
def benchmark_model(model_path: Path, name: str, images: list, Interpreter, args, out_root: Path) -> None:
    t = time.perf_counter()
    interp = Interpreter(model_path=str(model_path), num_threads=args.threads)
    interp.allocate_tensors()
    load_ms = (time.perf_counter() - t) * 1e3

    in_det = interp.get_input_details()[0]
    out_dets = interp.get_output_details()
    if len(out_dets) > 1:
        print(f"  WARNING: model has {len(out_dets)} outputs; only output 0 is saved (detection models should have 1).")
    out_det = out_dets[0]
    in_idx, out_idx = in_det["index"], out_det["index"]
    prep, in_w, in_h, nchw = make_input_fn(in_det)

    # warm-up (not recorded)
    x0 = prep(letterbox(images[0][1], in_w, in_h)[0])
    for _ in range(args.warmup):
        interp.set_tensor(in_idx, x0)
        interp.invoke()

    records, arrays = [], {}
    gc.collect()
    gc.disable()
    try:
        for stem, img in images:
            t0 = time.perf_counter()
            canvas, r, left, top = letterbox(img, in_w, in_h)
            x = prep(canvas)
            t1 = time.perf_counter()
            interp.set_tensor(in_idx, x)
            t2 = time.perf_counter()
            interp.invoke()
            t3 = time.perf_counter()
            out = interp.get_tensor(out_idx)  # TFLite returns a copy
            t4 = time.perf_counter()

            arrays[f"img_{stem}"] = out
            records.append({
                "stem": stem,
                "orig_w": int(img.shape[1]), "orig_h": int(img.shape[0]),
                "ratio": float(r), "pad_left": int(left), "pad_top": int(top),
                "preprocess_ms": (t1 - t0) * 1e3,
                "invoke_ms": (t3 - t2) * 1e3,
                "total_ms": (t4 - t0) * 1e3,
                "temp_c": cpu_temp_c(),
            })
    finally:
        gc.enable()

    model_dir = out_root / name
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "raw_outputs.npz", **arrays)  # uncompressed: no CPU spent on zlib

    summary = {k: latency_stats([rec[k] for rec in records]) for k in ("preprocess_ms", "invoke_ms", "total_ms")}
    meta = {
        "model_name": name,
        "model_file": str(model_path),
        "model_size_mb": round(model_path.stat().st_size / 1e6, 3),
        "threads": args.threads,
        "warmup_runs": args.warmup,
        "load_ms": load_ms,
        "input": {
            "shape": [int(v) for v in in_det["shape"]], "dtype": np.dtype(in_det["dtype"]).name,
            "quantization": [float(v) for v in in_det["quantization"]],
            "width": int(in_w), "height": int(in_h), "nchw": bool(nchw),
        },
        "output": {
            "shape": [int(v) for v in out_det["shape"]], "dtype": np.dtype(out_det["dtype"]).name,
            "quantization": [float(v) for v in out_det["quantization"]],
        },
        "latency_summary_ms": summary,
        "images": records,
    }
    (model_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    s_inv, s_tot = summary["invoke_ms"], summary["total_ms"]
    print(f"  invoke  mean {s_inv['mean']:.1f} ms | median {s_inv['median']:.1f} | p95 {s_inv['p95']:.1f}"
          f"   total mean {s_tot['mean']:.1f} ms  ({1000 / s_tot['mean']:.1f} FPS)")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def find_models(paths):
    found = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(p.glob("*.tflite"))
        elif p.suffix == ".tflite" and p.is_file():
            found.append(p)
        else:
            print(f"WARNING: skipping {p} (not a .tflite file or folder)")
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help=".tflite files and/or folders containing them")
    ap.add_argument("--images", type=Path, default=Path("./coco2017_test100/images"), help="folder with the test images")
    ap.add_argument("--out", type=Path, default=None, help="report folder (default: ./pi_report_<timestamp>)")
    ap.add_argument("--threads", type=int, default=4, help="TFLite CPU threads (default 4)")
    ap.add_argument("--warmup", type=int, default=5, help="untimed warm-up invocations per model (default 5)")
    ap.add_argument("--limit", type=int, default=0, help="only use the first N images (quick smoke test)")
    ap.add_argument("--cooldown", type=float, default=0.0, help="seconds to sleep between models (thermal settling)")
    args = ap.parse_args()

    models = find_models(args.models)
    if not models:
        sys.exit("ERROR: no .tflite models found.")
    img_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMG_EXTS) if args.images.is_dir() else []
    if args.limit:
        img_paths = img_paths[: args.limit]
    if not img_paths:
        sys.exit(f"ERROR: no images found in {args.images}")

    Interpreter, interp_module = get_interpreter_class()
    started = datetime.now()
    out_root = args.out or Path(f"./pi_report_{started:%Y%m%d_%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Runtime: {interp_module} | threads: {args.threads} | models: {len(models)} | images: {len(img_paths)}")
    info = system_info(interp_module)
    print(f"Device: {info['device_model']} | governor: {info['cpu_governor']} | temp: {info['temp_c_start']} C")

    # decode every image ONCE and share across all models (decoding is not part of latency)
    images = [(p.stem, read_rgb(p)) for p in img_paths]

    used, failures = set(), []
    for i, mp in enumerate(models, 1):
        name = mp.stem
        while name in used:  # same file name in two folders
            name += "_2"
        used.add(name)
        print(f"[{i}/{len(models)}] {mp.name}")
        try:
            benchmark_model(mp, name, images, Interpreter, args, out_root)
        except Exception as e:  # keep going with the other models
            print(f"  FAILED: {type(e).__name__}: {e}")
            failures.append({"model": str(mp), "error": f"{type(e).__name__}: {e}"})
        if args.cooldown and i < len(models):
            time.sleep(args.cooldown)

    info["temp_c_end"] = cpu_temp_c()
    info["throttled_end"] = vcgencmd("get_throttled")
    (out_root / "system_info.json").write_text(json.dumps(info, indent=2))
    (out_root / "run_config.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()},
        "started": started.isoformat(), "finished": datetime.now().isoformat(),
        "num_images": len(images), "models": [str(m) for m in models], "failures": failures,
    }, indent=2))

    size_mb = sum(f.stat().st_size for f in out_root.rglob("*") if f.is_file()) / 1e6
    print(f"\nDone. Report: {out_root.resolve()}  ({size_mb:.0f} MB)")
    print("Copy it to your Mac with (run on the Mac):")
    print(f"  scp -r {getpass.getuser()}@{socket.gethostname()}.local:{out_root.resolve()} ~/Downloads/")


if __name__ == "__main__":
    main()
