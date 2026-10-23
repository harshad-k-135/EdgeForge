"""Export YOLO checkpoints to FP16, FP32, or INT8 TensorFlow Lite."""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def convert_int8(saved_model, destination, imgsz, legacy):
    # Run without importing PyTorch to avoid TF/PyTorch thread-pool deadlocks.
    import cv2
    import numpy as np
    import tensorflow as tf
    from pi_tflite_benchmark import letterbox, read_rgb

    images = sorted((ROOT / "Datasets/coco128/images/train2017").glob("*.jpg"))
    if not images:
        raise FileNotFoundError("No COCO128 calibration images found")

    def representative_data():
        for index, image in enumerate(images, 1):
            canvas = letterbox(read_rgb(image), imgsz, imgsz)[0]
            yield [canvas[None].astype(np.float32) / 255.0]
            if index % 32 == 0:
                print(f"Calibrated {index}/{len(images)} images", flush=True)

    cv2.setNumThreads(1)
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    converter.experimental_new_quantizer = not legacy
    destination.write_bytes(converter.convert())
    print(f"INT8 -> {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision", choices=("fp16", "fp32", "int8", "both"), default="both")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models/pruned/structured")
    parser.add_argument("--beside-source", action="store_true", help="Save exports beside each source checkpoint")
    parser.add_argument("--reuse-saved-model", action="store_true")
    parser.add_argument("--int8-timeout", type=int, default=900)
    parser.add_argument("--int8-worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--destination", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--legacy", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.int8_worker:
        convert_int8(args.int8_worker, args.destination, args.imgsz, args.legacy)
        return

    import prune_utils  # noqa: F401: classes embedded in pruned checkpoints
    from ultralytics import YOLO

    models = args.models or sorted((ROOT / "models/pruned/structured").glob("*.pt"))
    if not models:
        parser.error("No checkpoints found")
    for weights in models:
        if not weights.is_file():
            parser.error(f"Checkpoint not found: {weights}")
    if not args.beside_source:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for weights in models:
        out_dir = weights.parent if args.beside_source else args.out_dir
        if args.precision == "fp16":
            model = YOLO(str(weights))
            exported = Path(model.export(
                format="tflite", imgsz=args.imgsz, half=True, int8=False,
                device="cpu", end2end=False, nms=False,
            ))
            destination = out_dir / f"{weights.stem}_fp16.tflite"
            if exported.resolve() != destination.resolve():
                shutil.copy2(exported, destination)
            print(f"FP16 -> {destination}", flush=True)
            continue
        saved_model = weights.parent / f"{weights.stem}_saved_model"
        destination = out_dir / f"{weights.stem}_fp32.tflite"
        if not (args.reuse_saved_model and (saved_model / "saved_model.pb").is_file() and destination.is_file()):
            model = YOLO(str(weights))
            exported = Path(model.export(
                format="tflite", imgsz=args.imgsz, half=False, int8=False,
                device="cpu", end2end=False, nms=False,
            ))
            shutil.copy2(exported, destination)
        print(f"FP32 -> {destination}", flush=True)
        if args.precision == "fp32":
            continue

        destination = out_dir / f"{weights.stem}_int8.tflite"
        env = {**os.environ, "TF_NUM_INTRAOP_THREADS": "1", "TF_NUM_INTEROP_THREADS": "1", "OMP_NUM_THREADS": "1"}
        for legacy in (False, True):
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()),
                   "--int8-worker", str(saved_model), "--destination", str(destination),
                   "--imgsz", str(args.imgsz)]
            if legacy:
                cmd.append("--legacy")
            try:
                subprocess.run(cmd, env=env, check=True, timeout=args.int8_timeout)
                break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
                print(f"INT8 conversion failed ({legacy=}, {exc})", flush=True)
                if legacy:
                    raise


if __name__ == "__main__":
    main()
