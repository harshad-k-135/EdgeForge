"""Run strategy 1 (structured channel pruning) on YOLO11n / YOLO26n.

Example:
  python scripts/prune_structured.py --model both --macs-reduction 0.30

The checkpoint can be loaded with Ultralytics for inference/export. Because
structured pruning changes the network graph and uses C2fSplit from
prune_utils.py, import that module before loading the checkpoint in another
Python process. For fine-tuning, use prune_utils.make_trainer(pruned_model)
with an Ultralytics DetectionTrainer so training keeps the pruned graph.
"""

import argparse
import copy
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prune_utils import model_complexity, structured_prune  # noqa: E402

DEFAULTS = {
    "yolo11n": ROOT / "models/base/v11/yolo11n.pt",
    "yolo26n": ROOT / "models/base/v26/yolo26n.pt",
}


def save_checkpoint(model, source: Path, destination: Path) -> None:
    """Save an Ultralytics-compatible checkpoint containing the pruned graph."""
    model = model.float().cpu().eval()
    checkpoint = {
        "model": model,
        "ema": copy.deepcopy(model),
        "optimizer": None,
        "updates": 0,
        "epoch": -1,
        "best_fitness": None,
        "train_args": {"model": str(source), "task": "detect"},
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def prune_one(name, source, args):
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {source}")
    yolo = YOLO(str(source), task="detect")
    model = yolo.model
    base_gmacs, base_mparams = model_complexity(model, args.imgsz)
    print(f"[{name}] starting at {base_gmacs:.3f} GMACs, {base_mparams:.3f} M parameters")

    stats = structured_prune(
        model,
        imgsz=args.imgsz,
        macs_reduction=args.macs_reduction,
        max_ratio=args.max_channel_ratio,
        steps=args.steps,
        round_to=args.round_to,
        verbose=True,
    )
    destination = args.out_dir / f"{name}_structured_{stats['macs_reduction']:.0%}.pt"
    save_checkpoint(model, source, destination)
    print(f"[{name}] saved {destination}")
    print(f"[{name}] {stats['macs_reduction']:.1%} MAC reduction, "
          f"{stats['params_reduction']:.1%} parameter reduction")
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=[*DEFAULTS, "both"], default="both")
    parser.add_argument("--weights", type=Path, help="Override weights for one selected model")
    parser.add_argument("--macs-reduction", type=float, default=0.30,
                        help="Target fraction of baseline MACs to remove (default: 0.30)")
    parser.add_argument("--max-channel-ratio", type=float, default=0.60,
                        help="Maximum per-layer channel pruning ratio")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--round-to", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "models/pruned/structured")
    args = parser.parse_args()

    if not 0 < args.macs_reduction < 1:
        parser.error("--macs-reduction must be between 0 and 1")
    if not 0 < args.max_channel_ratio < 1:
        parser.error("--max-channel-ratio must be between 0 and 1")
    if args.steps < 1 or args.round_to < 1:
        parser.error("--steps and --round-to must be positive")
    names = list(DEFAULTS) if args.model == "both" else [args.model]
    if args.weights and len(names) != 1:
        parser.error("--weights can only be used with one --model")
    for name in names:
        prune_one(name, args.weights or DEFAULTS[name], args)


if __name__ == "__main__":
    main()
