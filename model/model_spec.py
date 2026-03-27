"""Writes a human-readable training config / model spec to the checkpoint directory."""

import datetime
import torch
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


def write_training_config(
    model: nn.Module,
    checkpoint_dir: Path,
    *,
    R: int,
    expand: int,
    C: int,
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
):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    out_path = checkpoint_dir / "training_config.txt"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    lines = [
        "=" * 70,
        "  NOTARIUS TRAINING CONFIG",
        f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        "[ Model ]",
        f"  Architecture  : Notarius (Inverted Bottleneck ASR)",
        f"  R             : {R}",
        f"  expand        : {expand}",
        f"  C             : {C}  (C2={C*2}, C3={C*4})",
        f"  n_mels        : {n_mels}",
        f"  n_classes     : {n_classes}",
        f"  Total params  : {total_params:,}",
        f"  Trainable     : {trainable_params:,}",
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
