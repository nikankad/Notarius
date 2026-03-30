"""Writes a human-readable training config / model spec to the checkpoint directory."""

from __future__ import annotations

import datetime
import torch.nn as nn
from pathlib import Path


def _layer_summary(model: nn.Module) -> list[str]:
    lines = []
    for name, module in model.named_modules():
        if name == "":
            continue
        params = sum(p.numel() for p in module.parameters(recurse=False))
        param_str = f"  [{params:,} params]" if params > 0 else ""
        lines.append(f"  {name:<60} {type(module).__name__}{param_str}")
    return lines


def _decoder_lines(decoder: dict | None) -> list[str]:
    if decoder is None:
        return [
            f"  Type               : Greedy (argmax)",
            f"  Language model     : None",
        ]
    return [
        f"  Type               : {decoder.get('type', 'Greedy')}",
        f"  Language model     : {decoder.get('lm', 'None')}",
        f"  LM alpha           : {decoder.get('alpha', 'N/A')}",
        f"  LM beta            : {decoder.get('beta', 'N/A')}",
        f"  Beam width         : {decoder.get('beam_width', 'N/A')}",
    ]


def _augmentation_lines(augmentation: dict | None) -> list[str]:
    def _flag(val: bool) -> str:
        return "ON " if val else "OFF"

    if augmentation is None:
        augmentation = {}
    return [
        f"  Speed perturbation : {_flag(augmentation.get('speed_perturb', False))}",
        f"  SpecAugment        : {_flag(augmentation.get('spec_augment', False))}",
        f"  SpecCutout         : {_flag(augmentation.get('spec_cutout', False))}",
    ]


def write_training_config(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    R: int,
    n_mels: int,
    n_classes: int,
    num_epochs: int,
    warmup_epochs: int,
    lr: float,
    batch_size: int,
    optimizer_name: str,
    train_size: int,
    val_size: int,
    test_size: int,
    device: str,
    B: int | None = None,
    C: int | None = None,
    expand: int | None = None,
    augmentation: dict | None = None,
    decoder: dict | None = None,
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_dir / "training_config.txt"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if C is not None:
        header = "  IBNET TRAINING CONFIG"
        arch_lines = [
            f"  Architecture  : IBNet (Inverted Bottleneck ASR)",
            f"  R             : {R}",
            f"  C             : {C}",
            f"  expand        : {expand}",
            f"  n_mels        : {n_mels}",
            f"  n_classes     : {n_classes}",
            f"  Total params  : {total_params:,}",
            f"  Trainable     : {trainable_params:,}",
        ]
    else:
        header = "  QUARTZNET TRAINING CONFIG"
        arch_lines = [
            f"  Architecture  : QuartzNet-{B}x{R} (Depthwise-Separable ASR)" if B else f"  Architecture  : QuartzNet (Depthwise-Separable ASR)",
            f"  B             : {B}" if B else None,
            f"  R             : {R}",
            f"  n_mels        : {n_mels}",
            f"  n_classes     : {n_classes}",
            f"  Total params  : {total_params:,}",
            f"  Trainable     : {trainable_params:,}",
        ]
        # Filter out None values
        arch_lines = [line for line in arch_lines if line is not None]

    lines = [
        "=" * 70,
        header,
        f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "[ Model ]",
        *arch_lines,
        "",
        "[ Training ]",
        f"  Epochs        : {num_epochs}",
        f"  Warmup epochs : {warmup_epochs}",
        f"  Learning rate : {lr}",
        f"  Batch size    : {batch_size}",
        f"  Optimizer     : {optimizer_name}",
        f"  LR schedule   : LinearLR warmup → CosineAnnealingLR",
        f"  Loss          : CTCLoss (blank=28, zero_infinity=True)",
        f"  Grad clip     : 1.0",
        f"  Mixed prec.   : bfloat16 (autocast)",
        "",
        "[ Augmentation ]",
        *_augmentation_lines(augmentation),
        "",
        "[ Decoder ]",
        *_decoder_lines(decoder),
        "",
        "[ Dataset ]",
        f"  Train size    : {train_size:,}",
        f"  Val size      : {val_size:,}",
        f"  Test size     : {test_size:,}",
        "",
        "[ Hardware ]",
        f"  Device        : {device}",
        "",
        "[ Architecture ]",
        *_layer_summary(model),
        "",
        "=" * 70,
    ]

    out_path.write_text("\n".join(lines))
    print(f"Training config written to: {out_path}")
