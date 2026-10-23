"""Pruning helpers for Ultralytics YOLO11n / YOLO26n (detection).

Strategy 1: structured channel pruning with Torch-Pruning (DepGraph + group-magnitude importance)
Strategy 3: gradual unstructured magnitude pruning (Zhu & Gupta 2017 cubic schedule) run as
            Ultralytics callbacks, so zeros are exact in the saved checkpoint.
"""
import copy

import torch
import torch.nn as nn

# =====================================================================================
# 0. Small utilities
# =====================================================================================
C2F_NAMES = {"C2f", "C3k2"}
# Modules whose forward hard-codes channel counts (reshape / split with fixed sizes).
ATTN_NAMES = {"Attention", "PSABlock", "C2PSA", "AAttn", "ABlock", "A2C2f"}


class _FreezeBN:
    """Make BatchNorm running statistics immune to dummy forward passes (momentum -> 0)."""

    def __init__(self, model):
        self.model = model
        self.saved = []

    def __enter__(self):
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                self.saved.append((m, m.momentum))
                m.momentum = 0.0
        return self

    def __exit__(self, *exc):
        for m, mom in self.saved:
            m.momentum = mom
        self.saved = []


def _flatten(o):
    """Flatten nested list / tuple / dict model outputs into a flat list of tensors."""
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, dict):
        return [t for v in o.values() for t in _flatten(v)]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in _flatten(v)]
    return []


@torch.no_grad()
def count_macs_params(model, example):
    """MACs of all Conv2d layers for one image (train-mode forward, BN stats untouched)."""
    macs = [0]

    def hook(m, inp, out):
        macs[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    hooks = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    try:
        with _FreezeBN(model):
            model(example)
    finally:
        for h in hooks:
            h.remove()
    params = sum(p.numel() for p in model.parameters())
    return macs[0] / example.shape[0], params


def model_complexity(model, imgsz=640):
    """(GMACs, M params) of the inference path; works on a copy, original is untouched."""
    m = copy.deepcopy(model).cpu().float().train()
    head = m.model[-1]
    if hasattr(head, "end2end"):
        head.end2end = False  # count one head, not both
    macs, params = count_macs_params(m, torch.randn(1, 3, imgsz, imgsz))
    return macs / 1e9, params / 1e6


# =====================================================================================
# 1. C2f / C3k2 without torch.chunk  (needed so every channel group can be pruned)
# =====================================================================================
def _slice_conv(block, lo, hi):
    """Ultralytics Conv (conv + bn + act) restricted to output channels [lo, hi)."""
    new = copy.deepcopy(block)
    w = block.conv.weight.detach()[lo:hi].clone()
    new.conv.weight = nn.Parameter(w)
    new.conv.out_channels = hi - lo
    if block.conv.bias is not None:
        new.conv.bias = nn.Parameter(block.conv.bias.detach()[lo:hi].clone())
    bn, nbn = block.bn, new.bn
    nbn.weight = nn.Parameter(bn.weight.detach()[lo:hi].clone())
    nbn.bias = nn.Parameter(bn.bias.detach()[lo:hi].clone())
    nbn.running_mean = bn.running_mean[lo:hi].clone()
    nbn.running_var = bn.running_var[lo:hi].clone()
    nbn.num_features = hi - lo
    return new


class C2fSplit(nn.Module):
    """Drop-in replacement for C2f / C3k2: the first 1x1 conv is stored as two convs
    (cv0, cv1) instead of one conv followed by chunk(2, 1). Numerically identical."""

    def __init__(self, orig):
        super().__init__()
        c = orig.c
        if orig.cv1.conv.out_channels != 2 * c or not hasattr(orig.cv1, "bn"):
            raise RuntimeError("Unexpected C2f layout (already fused or modified?)")
        self.c = c
        self.cv0 = _slice_conv(orig.cv1, 0, c)
        self.cv1 = _slice_conv(orig.cv1, c, 2 * c)
        self.cv2 = orig.cv2
        self.m = orig.m
        for k in ("f", "i", "type", "np"):  # Ultralytics layer bookkeeping
            if hasattr(orig, k):
                setattr(self, k, getattr(orig, k))

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def swap_c2f_for_split(module):
    """Recursively replace every C2f / C3k2 by C2fSplit. Returns the number replaced."""
    n = 0
    for name, child in list(module.named_children()):
        if type(child).__name__ in C2F_NAMES:
            setattr(module, name, C2fSplit(child))
            n += 1
        else:
            n += swap_c2f_for_split(child)
    return n


# =====================================================================================
# 2. Strategy 1: structured pruning with Torch-Pruning
# =====================================================================================
def build_ignored_layers(model):
    """Layers Torch-Pruning must not touch.

    * the whole detection head (box/cls branches, DFL, one2one branch of YOLO26)
    * the last conv of each neck output that feeds the head, so the head's input width
      never changes (YOLO26's one2one branch sees *detached* features, which would hide
      that coupling from the dependency graph)
    * attention blocks (C2PSA / PSABlock / Attention): they reshape with fixed sizes
    """
    top = [model.model[-1]]
    head = model.model[-1]
    froms = head.f if isinstance(head.f, (list, tuple)) else [head.f]
    for i in froms:
        producer = model.model[i]
        out_conv = getattr(producer, "cv2", None)
        if out_conv is None:
            raise RuntimeError(f"Don't know how to protect the output of layer {i}: {type(producer)}")
        top.append(out_conv)
    for m in model.modules():
        if type(m).__name__ in ATTN_NAMES:
            top.append(m)
    flat = {}
    for t in top:
        for sub in t.modules():
            flat[id(sub)] = sub
    return list(flat.values())


def structured_prune(model, imgsz=640, macs_reduction=0.30, max_ratio=0.6, steps=30,
                     round_to=8, verbose=True):
    """Prune `model` (an Ultralytics DetectionModel, modified in place) until MACs have
    dropped by `macs_reduction` (e.g. 0.30 = 30 % fewer MACs) or `max_ratio` is hit.

    Group-magnitude importance (L2) + DepGraph, local pruning, channels rounded to `round_to`.
    Returns a dict of statistics.
    """
    import torch_pruning as tp

    model = model.cpu().float()
    for p in model.parameters():
        p.requires_grad_(True)
    n_swapped = swap_c2f_for_split(model)

    head = model.model[-1]
    prev_e2e = getattr(head, "end2end", None)
    if prev_e2e is not None:
        head.end2end = False  # trace only the one2many branch (the other sees detached inputs)
    model.train()

    example = torch.randn(1, 3, imgsz, imgsz)
    base_macs, base_params = count_macs_params(model, example)
    target = base_macs * (1.0 - macs_reduction)
    ignored = build_ignored_layers(model)

    imp_name = next(n for n in ("GroupMagnitudeImportance", "GroupNormImportance", "MagnitudeImportance")
                    if hasattr(tp.importance, n))
    importance = getattr(tp.importance, imp_name)(p=2)
    pruner_cls = getattr(tp.pruner, "BasePruner", None) or tp.pruner.MetaPruner
    kwargs = dict(importance=importance, iterative_steps=steps, ignored_layers=ignored,
                  round_to=round_to, output_transform=_flatten)

    history = []
    with _FreezeBN(model):
        try:
            pruner = pruner_cls(model, example, pruning_ratio=max_ratio, **kwargs)
        except TypeError:  # torch-pruning < 1.3 called it ch_sparsity
            pruner = pruner_cls(model, example, ch_sparsity=max_ratio, **kwargs)

        macs, params = base_macs, base_params
        for i in range(steps):
            pruner.step()
            macs, params = count_macs_params(model, example)
            history.append((i + 1, macs / 1e9, params / 1e6))
            if verbose:
                print(f"  prune step {i + 1:>2}: {macs / 1e9:6.3f} GMACs  {params / 1e6:6.3f} M params")
            if macs <= target:
                break
        else:
            print(f"  WARNING: target not reached after {steps} steps "
                  f"({1 - macs / base_macs:.1%} reduction). Increase max_ratio.")

    if prev_e2e is not None:
        head.end2end = prev_e2e

    # sanity: both train-mode and inference-mode forward passes must work
    with torch.no_grad(), _FreezeBN(model):
        model.train()
        model(example)
        model.eval()
        model(example)
    if hasattr(head, "shape"):
        head.shape = None  # force anchors to be rebuilt for the real input size later
    model.train()

    return dict(base_gmacs=base_macs / 1e9, base_mparams=base_params / 1e6,
                gmacs=macs / 1e9, mparams=params / 1e6,
                macs_reduction=1 - macs / base_macs, params_reduction=1 - params / base_params,
                c2f_swapped=n_swapped, history=history)


def make_trainer(pruned_model):
    """Ultralytics trainer that fine-tunes the given (already pruned) model instead of
    rebuilding the network from its YAML (which would undo the pruning)."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class PrunedDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            return pruned_model

    return PrunedDetectionTrainer


# =====================================================================================
# 3. Strategy 3: gradual unstructured magnitude pruning (Zhu & Gupta, 2017)
# =====================================================================================
def select_layers(model, skip_first=True, skip_depthwise=True, skip_head_output=True):
    """Conv2d layers that get pruned: everything except the 3-channel stem, grouped /
    depthwise convs, the final box / class prediction convs and the fixed DFL conv."""
    head = model.model[-1]
    skip = set()
    if skip_head_output:
        for sub in head.modules():
            if isinstance(sub, nn.Sequential) and len(sub) and isinstance(sub[-1], nn.Conv2d):
                skip.add(id(sub[-1]))
    dfl = getattr(head, "dfl", None)
    if dfl is not None:
        skip.update(id(m) for m in dfl.modules())
    layers = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d) or id(m) in skip:
            continue
        if skip_first and m.in_channels == 3:
            continue
        if skip_depthwise and m.groups > 1:
            continue
        layers.append((name, m))
    return layers


def sparsity_report(model, **select_kwargs):
    layers = select_layers(model, **select_kwargs)
    z = sum(int((m.weight == 0).sum()) for _, m in layers)
    t = sum(m.weight.numel() for _, m in layers)
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    gz = sum(int((m.weight == 0).sum()) for m in convs)
    gt = sum(m.weight.numel() for m in convs)
    return dict(pruned_layers=len(layers), prunable_sparsity=z / max(t, 1),
                all_conv_sparsity=gz / max(gt, 1))


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


class GradualMagnitudePruner:
    """Per-layer magnitude pruning with the cubic schedule of Zhu & Gupta (2017):

        s_t = s_f + (s_i - s_f) * (1 - (t - t0) / (t_end - t0)) ** 3

    Masks are recomputed every `frequency` iterations, re-applied after every optimizer
    step, and applied to the EMA copy as well (Ultralytics validates / saves the EMA model),
    so zeros are exact in last.pt.
    """

    def __init__(self, final_sparsity=0.70, initial_sparsity=0.0, begin_frac=0.0, end_frac=0.75,
                 frequency=None, skip_first=True, skip_depthwise=True, skip_head_output=True):
        self.sf, self.si = final_sparsity, initial_sparsity
        self.begin_frac, self.end_frac = begin_frac, end_frac
        self.frequency = frequency
        self.sel = dict(skip_first=skip_first, skip_depthwise=skip_depthwise,
                        skip_head_output=skip_head_output)
        self.step = 0
        self.masks = {}
        self.finalized = False
        self.log = []

    def attach(self, yolo):
        yolo.add_callback("on_train_start", self.on_train_start)
        yolo.add_callback("on_train_batch_end", self.on_train_batch_end)
        return self

    # ---- schedule --------------------------------------------------------------
    def sparsity_at(self, step):
        if step < self.begin:
            return 0.0
        frac = min(1.0, (step - self.begin) / max(1, self.end - self.begin))
        return self.sf + (self.si - self.sf) * (1.0 - frac) ** 3

    # ---- callbacks -------------------------------------------------------------
    def on_train_start(self, trainer):
        total = trainer.epochs * len(trainer.train_loader)
        self.total = total
        self.begin = int(self.begin_frac * total)
        self.end = max(self.begin + 1, int(self.end_frac * total))
        self.freq = self.frequency or max(1, min(100, total // 200))
        self.model = _unwrap(trainer.model)
        self.layers = select_layers(self.model, **self.sel)
        ema = getattr(getattr(trainer, "ema", None), "ema", None)
        self.ema_mods = dict(ema.named_modules()) if ema is not None else {}
        print(f"[GMP] {len(self.layers)} conv layers, {total} iterations, "
              f"sparsity {self.si:.0%} -> {self.sf:.0%} between iter {self.begin} and {self.end}, "
              f"mask update every {self.freq} iters")

    @torch.no_grad()
    def _update_masks(self, s):
        for name, m in self.layers:
            w = m.weight
            n = w.numel()
            k = int(s * n)
            if k <= 0:
                continue
            idx = torch.topk(w.detach().abs().flatten(), k, largest=False, sorted=False).indices
            mask = torch.ones(n, dtype=torch.bool, device=w.device)
            mask[idx] = False
            mask = mask.view_as(w)
            prev = self.masks.get(name)
            if prev is not None:
                mask &= prev  # once pruned, always pruned
            self.masks[name] = mask

    @torch.no_grad()
    def _apply(self):
        for name, m in self.layers:
            mask = self.masks.get(name)
            if mask is None:
                continue
            m.weight.data.masked_fill_(~mask, 0.0)
            em = self.ema_mods.get(name)
            if em is not None:
                em.weight.data.masked_fill_(~mask, 0.0)

    def on_train_batch_end(self, trainer):
        self.step += 1
        due = self.step >= self.begin and (
            (self.step - self.begin) % self.freq == 0 or (self.step >= self.end and not self.finalized))
        if due:
            s = self.sparsity_at(self.step)
            self._update_masks(s)
            self.log.append((self.step, s))
            if self.step >= self.end:
                self.finalized = True
        if self.masks:
            self._apply()