"""Calibrate and prune YOLO11n, then YOLO26n, selecting about 20% of groups.

By default remove the lowest-scoring channel from each matching group. Use
--channels-per-layer to change this count. Whole layers are never deleted.
Thresholds are computed separately for each model, prioritizing low std,
then low mean. Scores, checkpoints and pruning reports are saved per model.
"""

from pathlib import Path
import argparse
import gc
import json
import math

import torch
from torch.utils.data import DataLoader, Subset
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
import torch_pruning as tp

from prune_utils import build_ignored_layers, swap_c2f_for_split
from prune_structured import save_checkpoint


ROOT = Path(__file__).resolve().parents[1]
MODELS = [
    ROOT / "models/base/v11/yolo11n.pt",
    ROOT / "models/base/v26/yolo26n.pt",
]
MODEL_PATH = MODELS[-1]
DATA = "coco128.yaml"
NUM_IMAGES = 32
IMG_SIZE = 640
SCORES_PATH = Path(__file__).resolve().parent / "taylor_group_scores.json"
OUTPUT_DIR = ROOT / "models/pruned/taylor"
GROUP_FRACTION = 0.20
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)


def calibrate(
    model_path=MODEL_PATH, data=DATA, num_images=NUM_IMAGES,
    img_size=IMG_SIZE, device=DEVICE, seed=0, scores_path=SCORES_PATH,
):
    """Return model, criterion, DepGraph and ranked groups, retaining gradients.

    Perform exactly one forward/backward pass, without an optimizer or pruning.
    The entire calibration batch must fit in device memory.
    """
    if num_images < 1:
        raise ValueError("num_images must be positive")

    print("Using device:", device)
    model = YOLO(str(model_path)).model.to(device).float()
    # This equivalent rewrite removes chunk(2) ambiguity from DepGraph's
    # channel mapping. Use these same rewritten modules for subsequent pruning.
    swapped = swap_c2f_for_split(model)
    print(f"C2f/C3k2 blocks rewritten for DepGraph: {swapped}")
    # Loaded inference checkpoints may have frozen parameters. Training mode
    # selects the raw detection outputs; requires_grad enables their graph.
    model.train()
    model.requires_grad_(True)
    # YOLO11's DFL projection is fixed and used only during inference. Keep it
    # frozen so the gradient validation checks only trainable loss parameters.
    dfl = getattr(model.model[-1], "dfl", None)
    if dfl is not None:
        dfl.requires_grad_(False)
    # Use pretrained running statistics without updating BatchNorm buffers.
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()

    # Keep raw scores: the default mean normalization makes every group's
    # mean equal to one, which is uninformative for comparing groups.
    importance = tp.importance.GroupTaylorImportance(normalizer=None)
    print("Importance criterion:", type(importance).__name__)

    # Build validation transforms directly, without creating a training run.
    args = get_cfg(overrides={"task": "detect", "imgsz": img_size})
    dataset_info = check_det_dataset(data, autodownload=False)
    if dataset_info["nc"] != model.model[-1].nc:
        raise ValueError("Calibration dataset and model must have the same class count")
    model.args = args
    model.nc = dataset_info["nc"]
    model.names = dataset_info["names"]
    dataset = build_yolo_dataset(
        args, dataset_info["val"], num_images, dataset_info,
        mode="val", rect=False, stride=max(int(model.stride.max()), 32),
    )
    if len(dataset) < num_images:
        raise ValueError(f"Need {num_images} images, but validation has {len(dataset)}")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:num_images].tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=num_images, shuffle=False,
        num_workers=0, collate_fn=dataset.collate_fn,
    )
    batch = next(iter(loader))
    batch = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }
    batch["img"] = batch["img"].float() / 255.0

    model.zero_grad(set_to_none=True)
    # Protect attention reshapes, detection heads and head-input widths. The
    # YOLO26 one-to-one head detaches its inputs, hiding their coupling from DG.
    ignored = build_ignored_layers(model)
    with torch.enable_grad():
        predictions = None
        training_modes = {module: module.training for module in model.modules()}

        def calibration_forward(traced_model, images):
            nonlocal predictions
            # DepGraph internally calls eval(); restore our training/BN modes.
            for module, training in training_modes.items():
                module.training = training
            predictions = traced_model(images)
            return predictions

        # Trace the actual calibration forward; no additional model pass.
        graph = tp.DependencyGraph().build_dependency(
            model, example_inputs=batch["img"], forward_fn=calibration_forward,
        )
        loss_components, _ = model.loss(batch, preds=predictions)
        # Ultralytics returns [box, cls, dfl] losses scaled by batch size.
        # Reduce to a scalar and normalize once to obtain a per-image loss.
        loss = loss_components.sum() / num_images
        if not loss.requires_grad:
            raise RuntimeError("Detection loss has no autograd graph")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite calibration loss: {loss.detach().item()}")
        loss.backward()

    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, p in trainable if p.grad is None]
    nonfinite = [
        name for name, p in trainable
        if p.grad is not None and not torch.isfinite(p.grad).all()
    ]
    if missing or nonfinite:
        raise RuntimeError(f"Invalid gradients: missing={missing}, nonfinite={nonfinite}")
    if not any(torch.count_nonzero(p.grad).item() for _, p in trainable):
        raise RuntimeError("All calibration gradients are zero")

    print(f"Calibration images processed: {batch['img'].shape[0]}")
    print(f"Calibration loss per image: {loss.detach().item():.6f}")
    print("Forward/backward passes completed: 1/1")
    print(f"Parameters with finite gradients: {len(trainable)}/{len(trainable)}")
    ranked_groups = score_groups(model, graph, importance, ignored)
    report = {
        "model_path": str(model_path),
        "criterion": "GroupTaylorImportance",
        "normalizer": None,
        "group_reduction": "mean",
        "std_definition": "population standard deviation across channel scores (correction=0)",
        "ranking": "minimum channel score in each dependency group",
        "num_images": num_images, "img_size": img_size, "seed": seed,
        "excluded": "attention, detection heads, and head-input output widths",
        "groups": [{k: v for k, v in row.items() if k != "group"} for row in ranked_groups],
    }
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Full channel scores and dependencies saved to: {scores_path}")
    return model, importance, graph, ranked_groups


@torch.no_grad()
def score_groups(model, graph, importance, ignored):
    """Call the actual TP criterion on each eligible DepGraph pruning group.

    TP returns one score per root channel, aggregating all dependent weights.
    Rank by the lowest channel score, since pruning a channel removes its
    coupled slices throughout the group, rather than the entire group width.
    """
    names = {module: name for name, module in model.named_modules()}
    rows = []
    for group_id, group in enumerate(graph.get_all_groups(ignored_layers=ignored)):
        scores = importance(group)
        if scores is None or not torch.isfinite(scores).all():
            raise RuntimeError(f"Invalid GroupTaylorImportance scores for group {group_id}")
        root = group[0].dep.target.module
        root_indices = list(group[0].idxs)
        order = torch.argsort(scores).tolist()
        # A full-width group would delete the layer. Validate the single-channel
        # candidate that is actually being ranked, not removal of every channel.
        candidate = graph.get_pruning_group(
            root, group[0].dep.handler, [root_indices[order[0]]],
        )
        if not graph.check_pruning_group(candidate):
            raise RuntimeError(f"Invalid lowest-channel pruning candidate in group {group_id}")
        rows.append({
            "group_id": group_id, "root": names[root], "channels": len(root_indices),
            "min_score": scores.min().item(), "mean_score": scores.mean().item(),
            "std_score": scores.std(correction=0).item(),
            "lowest_channel": root_indices[order[0]],
            "channel_indices": root_indices, "channel_scores": scores.cpu().tolist(),
            "channels_lowest_first": [root_indices[i] for i in order],
            "dependencies": [
                {"layer": names.get(dep.target.module, str(dep.target.module)),
                 "operation": dep.handler.__name__, "indices": list(idxs)}
                for dep, idxs in group
            ],
            "group": group,
        })
    if not rows:
        raise RuntimeError("No eligible pruning groups found")
    rows.sort(key=lambda row: row["min_score"])
    print_group_scores(rows)
    return rows


def print_group_scores(rows):
    """Print group summaries, using population standard deviation over channels."""
    print(f"\nActual GroupTaylorImportance scores: {len(rows)} eligible groups")
    print("Raw scores (normalizer=None); ranked by the lowest coupled channel score.")
    print("Std is the population standard deviation across each group's channel scores.")
    print(f"{'Group':>5} {'Root layer':<40} {'Channels':>8} {'Min score':>12} {'Mean score':>12} {'Std score':>12} {'Pick channel':>12}")
    for row in rows:
        print(
            f"{row['group_id']:>5} {row['root']:<40} {row['channels']:>8} "
            f"{row['min_score']:>12.6g} {row['mean_score']:>12.6g} "
            f"{row['std_score']:>12.6g} "
            f"{row['lowest_channel']:>12}"
        )


def choose_thresholds(rows, group_fraction=GROUP_FRACTION):
    """Choose strict score cutoffs, lowering std first and then mean.

    Round the requested group count to the nearest integer (at least one).
    First retain the target lowest-std groups, including ties at the boundary.
    Then tighten mean within that pool to the closest attainable count. Equal
    score pairs cannot be separated by thresholds, so counts may differ on ties.
    """
    if not rows:
        raise ValueError("No eligible groups to select")
    if not 0 < group_fraction <= 1:
        raise ValueError("group_fraction must be in (0, 1]")
    if any(not math.isfinite(row[key]) or row[key] < 0
           for row in rows for key in ("mean_score", "std_score")):
        raise ValueError("Group scores must be finite and nonnegative")
    target = max(1, math.floor(len(rows) * group_fraction + 0.5))
    ordered = sorted(rows, key=lambda row: (row["std_score"], row["mean_score"]))
    std_threshold = math.nextafter(ordered[target - 1]["std_score"], math.inf)
    pool = [row for row in rows if row["std_score"] < std_threshold]
    # Each distinct mean defines an attainable count under a strict cutoff.
    mean_counts = {}
    for row in pool:
        mean_counts[row["mean_score"]] = mean_counts.get(row["mean_score"], 0) + 1
    count = 0
    candidates = []
    for mean, frequency in sorted(mean_counts.items()):
        count += frequency
        candidates.append((abs(count - target), count, mean))
    _, selected, mean_cutoff = min(candidates)
    mean_threshold = math.nextafter(mean_cutoff, math.inf)
    print(f"Runtime selection: target {target}/{len(rows)} ({group_fraction:.0%}), "
          f"selected {selected}/{len(rows)} ({selected / len(rows):.1%}); "
          f"std < {std_threshold:.8g}, mean < {mean_threshold:.8g}")
    return mean_threshold, std_threshold, target


def prune_low_score_groups(
    model, graph, ranked_groups, mean_threshold,
    std_threshold, channels_per_layer=1,
):
    """Prune lowest Taylor channels where BOTH strict thresholds hold.

    Use the calibration scores before any pruning. Recreate each dependency group
    at its current width and track original root indices when groups overlap.
    """
    if channels_per_layer < 1:
        raise ValueError("channels_per_layer must be positive")
    if not (0 < mean_threshold < float("inf") and 0 < std_threshold < float("inf")):
        raise ValueError("Score thresholds must be positive and finite")
    modules = dict(model.named_modules())
    remaining = {
        modules[row["root"]]: list(row["channel_indices"])
        for row in ranked_groups
    }
    ignored = set(build_ignored_layers(model))
    selected = [
        row for row in ranked_groups
        if row["mean_score"] < mean_threshold and row["std_score"] < std_threshold
    ]
    print(f"\nPruning {len(selected)}/{len(ranked_groups)} groups: "
          f"mean < {mean_threshold}, std < {std_threshold}")
    records = []
    for row in selected:
        root = modules[row["root"]]
        surviving = remaining[root]
        # Coupled pruning may already have removed some of these channels.
        removed = set(row["channel_indices"]) - set(surviving)
        count = min(max(0, channels_per_layer - len(removed)), len(surviving) - 1)
        original_indices = [
            idx for idx in row["channels_lowest_first"] if idx in surviving
        ][:count]
        record = {
            "root": row["root"], "mean_score": row["mean_score"],
            "std_score": row["std_score"], "channels_before": len(surviving),
            "original_indices": original_indices,
        }
        if not original_indices:
            record.update(status="skipped", reason="pruning budget reached or only one channel remains")
        else:
            indices = [surviving.index(idx) for idx in original_indices]
            group = graph.get_pruning_group(root, row["group"][0].dep.handler, indices)
            touches_protected_output = any(
                dep.target.module in ignored and graph.is_out_channel_pruning_fn(dep.handler)
                for dep, _ in group
            )
            if touches_protected_output or not graph.check_pruning_group(group):
                record.update(status="skipped", reason="protected output or invalid dependency group")
            else:
                # Collect output changes before pruning mutates module widths.
                output_changes = {}
                for dep, idxs in group:
                    module = dep.target.module
                    if module in remaining and graph.is_out_channel_pruning_fn(dep.handler):
                        output_changes.setdefault(module, set()).update(idxs)
                group.prune()
                for module, idxs in output_changes.items():
                    remaining[module] = [
                        idx for position, idx in enumerate(remaining[module]) if position not in idxs
                    ]
                record.update(status="pruned", indices=indices)
        record["channels_after"] = len(remaining[root])
        records.append(record)
        print(f"  {record['root']}: {record['status']}, "
              f"{record['channels_before']} -> {record['channels_after']} channels")
    return records


def prune_one(source, destination, scores_path, args):
    """Finish calibration, pruning, validation and saving before the next model."""
    print(f"\nProcessing {source.stem}: {source}")
    model, _, graph, ranked_groups = calibrate(
        source, args.data, args.num_images, args.img_size, args.device,
        scores_path=scores_path,
    )
    mean_threshold, std_threshold, target = choose_thresholds(
        ranked_groups, args.group_fraction,
    )
    parameters_before = sum(p.numel() for p in model.parameters())
    records = prune_low_score_groups(
        model, graph, ranked_groups, mean_threshold,
        std_threshold, args.channels_per_layer,
    )
    model.zero_grad(set_to_none=True)
    # Verify both raw detection outputs and inference after all coupled changes.
    example = torch.zeros(1, 3, args.img_size, args.img_size, device=args.device)
    with torch.no_grad():
        model.train()
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        model(example)
        model.eval()
        model(example)
    if hasattr(model.model[-1], "shape"):
        model.model[-1].shape = None
    parameters_after = sum(p.numel() for p in model.parameters())
    save_checkpoint(model, source, destination)
    report = {
        "model_path": str(source),
        "mean_threshold": mean_threshold, "std_threshold": std_threshold,
        "threshold_priority": "standard deviation first, then mean",
        "group_fraction": args.group_fraction, "eligible_groups": len(ranked_groups),
        "target_groups": target, "selected_groups": len(records),
        "pruned_groups": sum(row["status"] == "pruned" for row in records),
        "selection": "mean_score < mean_threshold AND std_score < std_threshold",
        "channels_per_layer": args.channels_per_layer,
        "scores_path": str(scores_path), "parameters_before": parameters_before,
        "parameters_after": parameters_after, "groups": records,
    }
    report_path = destination.with_suffix(".pruning.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Parameters: {parameters_before:,} -> {parameters_after:,}")
    print(f"Pruned checkpoint saved to: {destination}")
    print(f"Pruning report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, nargs="+", default=MODELS,
                        help="Checkpoint paths, processed in order (default: YOLO11n, YOLO26n)")
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--group-fraction", type=float, default=GROUP_FRACTION,
                        help="Fraction of eligible groups to select per model (default: 0.20)")
    parser.add_argument("--channels-per-layer", type=int, default=1)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path, help="Checkpoint output override for a single model")
    args = parser.parse_args()
    if args.channels_per_layer < 1:
        parser.error("--channels-per-layer must be positive")
    if not 0 < args.group_fraction <= 1:
        parser.error("--group-fraction must be in (0, 1]")
    if args.output and len(args.model) != 1:
        parser.error("--output requires exactly one --model; use --out-dir for multiple models")
    if len({source.stem for source in args.model}) != len(args.model):
        parser.error("Model checkpoint names must be unique to keep outputs separate")
    jobs = [
        (source, args.output or args.out_dir / f"{source.stem}_taylor.pt",
         args.out_dir / f"{source.stem}_taylor_group_scores.json")
        for source in args.model
    ]
    for source, destination, _ in jobs:
        if not source.is_file():
            parser.error(f"Checkpoint not found: {source}")
        if destination.resolve() in {path.resolve() for path in args.model}:
            parser.error("Output must differ from all original model paths")
    for source, destination, scores_path in jobs:
        prune_one(source, destination, scores_path, args)
        # DepGraph holds reference cycles; release them between model runs.
        gc.collect()
        if str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()
        elif str(args.device).startswith("mps"):
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
