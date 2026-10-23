"""Run strategy 3 (gradual unstructured pruning) on YOLO11n / YOLO26n.

This performs a pruning-aware training run using GradualMagnitudePruner from
prune_utils.py (Zhu & Gupta cubic sparsity schedule). The final checkpoint is
copied to --out-dir and can be exported or used for subsequent fine-tuning.

Example:
  python scripts/prune_unstructured.py --model yolo11n --data data.yaml --epochs 50
  python scripts/prune_unstructured.py --model both --data data.yaml --final-sparsity 0.7
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prune_utils import GradualMagnitudePruner, sparsity_report  # noqa: E402

DEFAULTS = {
    "yolo11n": ROOT / "models/base/v11/yolo11n.pt",
    "yolo26n": ROOT / "models/base/v26/yolo26n.pt",
}


def prune_one(name, source, args):
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    yolo = YOLO(str(source), task="detect")
    pruner = GradualMagnitudePruner(
        final_sparsity=args.final_sparsity,
        initial_sparsity=args.initial_sparsity,
        begin_frac=args.begin_frac,
        end_frac=args.end_frac,
        frequency=args.frequency,
    )
    pruner.attach(yolo)

    run_name = f"{name}_unstructured_pruning"
    yolo.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=run_name,
        exist_ok=args.exist_ok,
        save=True,
        plots=False,
    )

    run_dir = Path(yolo.trainer.save_dir)
    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce the expected final checkpoint: {last}")
    destination = args.out_dir / f"{name}_unstructured_{args.final_sparsity:.0%}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(last, destination)

    # Read final sparsity from the saved EMA model, which is the model used for validation.
    final_model = YOLO(str(destination), task="detect").model
    report = sparsity_report(final_model)
    print(f"[{name}] saved pruning checkpoint: {destination}")
    print(f"[{name}] prunable Conv2d sparsity: {report['prunable_sparsity']:.1%}; "
          f"all Conv2d sparsity: {report['all_conv_sparsity']:.1%}")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*DEFAULTS, "both"], default="both")
    parser.add_argument("--weights", type=Path, help="Override weights for one selected model")
    parser.add_argument("--data", type=Path, required=True, help="Ultralytics detection dataset YAML")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--initial-sparsity", type=float, default=0.0)
    parser.add_argument("--final-sparsity", type=float, default=0.70)
    parser.add_argument("--begin-frac", type=float, default=0.0)
    parser.add_argument("--end-frac", type=float, default=0.75)
    parser.add_argument("--frequency", type=int, default=None,
                        help="Mask update interval in batches (default is chosen automatically)")
    parser.add_argument("--project", type=Path, default=ROOT / "runs/pruning/unstructured")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models/pruned/unstructured")
    parser.add_argument("--exist-ok", action="store_true")
    args = parser.parse_args()

    if not args.data.is_file():
        parser.error(f"dataset YAML not found: {args.data}")
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs/batch must be positive and workers cannot be negative")
    if not 0 <= args.initial_sparsity <= args.final_sparsity < 1:
        parser.error("sparsities must satisfy 0 <= initial <= final < 1")
    if not 0 <= args.begin_frac < args.end_frac <= 1:
        parser.error("schedule fractions must satisfy 0 <= begin < end <= 1")
    if args.frequency is not None and args.frequency < 1:
        parser.error("--frequency must be positive")

    names = list(DEFAULTS) if args.model == "both" else [args.model]
    if args.weights and len(names) != 1:
        parser.error("--weights can only be used with one --model")
    for name in names:
        prune_one(name, args.weights or DEFAULTS[name], args)


if __name__ == "__main__":
    main()
