import datetime
import os
import random
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from dotenv import load_dotenv
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Subset
from torch_optimizer import NovoGrad
from torchaudio.datasets import LIBRISPEECH

from ..helpers import (
    BucketBatchSampler,
    DistributedBucketBatchSampler,
    batch_word_errors_and_count,
    collate_fn_test,
    get_dataset_lengths,
    log_epoch,
    collate_fn_speed_perturb,
)
from ..qnmodel import QuartzNetBxR
from ..scripts.model_spec import write_training_config

load_dotenv()
root = os.getenv("ROOT")


def _generate_run_id(B: int, R: int, aug_label: str = "") -> str:
    slurm_id = os.environ.get("SLURM_JOB_ID")
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = f"-{aug_label}" if aug_label else ""
    model_spec = f"{B}x{R}"
    return f"quartznet-{model_spec}{suffix}-{slurm_id}" if slurm_id else f"quartznet-{model_spec}{suffix}-{ts}"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _is_distributed() -> bool:
    return _env_int("WORLD_SIZE", 1) > 1


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _setup_distributed():
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)

    if world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP launch requested, but CUDA is not available.")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, local_rank, world_size, device


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _resolve_checkpoint_dir(checkpoint_dir: str) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    if checkpoint_path.is_absolute():
        return checkpoint_path
    project_root = Path(__file__).resolve().parents[1]
    return project_root / checkpoint_path


def _save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _build_datasets():
    train_ds = LIBRISPEECH(root=root, url="train-other-500", download=False)
    val_ds = LIBRISPEECH(root=root, url="dev-clean", download=False)
    test_ds = LIBRISPEECH(root=root, url="test-clean", download=False)

    rng = random.Random(42)
    indices = list(range(len(train_ds)))
    rng.shuffle(indices)
    train_ds = Subset(train_ds, indices[:2 * len(train_ds) // 5])
    return train_ds, val_ds, test_ds


def _loader_kwargs(num_workers: int) -> dict:
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return kwargs


def _build_dataloaders(train_ds, val_ds, test_ds, batch_size, num_workers, rank, world_size):
    if _is_main_process(rank):
        print("Pre-computing dataset lengths for bucket batching...")
        train_lengths_all = get_dataset_lengths(train_ds.dataset)
        val_lengths = get_dataset_lengths(val_ds)
        test_lengths = get_dataset_lengths(test_ds)
    _barrier()

    if not _is_main_process(rank):
        train_lengths_all = get_dataset_lengths(train_ds.dataset)
        val_lengths = None
        test_lengths = None

    per_device_batch_size = batch_size if world_size == 1 else max(1, batch_size // world_size)
    train_lengths = [train_lengths_all[i] for i in train_ds.indices]

    if world_size > 1:
        train_sampler = DistributedBucketBatchSampler(
            train_lengths,
            batch_size=per_device_batch_size,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=42,
        )
    else:
        train_sampler = BucketBatchSampler(train_lengths, batch_size=per_device_batch_size, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_fn_speed_perturb,
        **_loader_kwargs(num_workers),
    )

    val_loader = None
    test_loader = None
    if _is_main_process(rank):
        val_sampler = BucketBatchSampler(val_lengths, batch_size=batch_size, shuffle=False)
        test_sampler = BucketBatchSampler(test_lengths, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(
            val_ds,
            batch_sampler=val_sampler,
            collate_fn=collate_fn_test,
            **_loader_kwargs(num_workers),
        )
        test_loader = DataLoader(
            test_ds,
            batch_sampler=test_sampler,
            collate_fn=collate_fn_test,
            **_loader_kwargs(num_workers),
        )

    return train_loader, val_loader, test_loader, per_device_batch_size


def _build_inference_payload(model, B, R, epoch, best_val_loss, best_val_wer, run_id=None):
    return {
        "run_id": run_id,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "best_val_wer": best_val_wer,
        "model_state_dict": model.state_dict(),
        "config": {
            "B": B,
            "R": R,
            "n_mels": 64,
            "n_classes": 29,
        },
    }


def _reduce_train_metrics(total_loss, total_word_errors, total_ref_words, num_batches, device):
    metrics = torch.tensor(
        [total_loss, total_word_errors, total_ref_words, num_batches],
        dtype=torch.float64,
        device=device,
    )
    if _is_distributed():
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    return metrics.tolist()


def train_model(
    B=5,
    R=5,
    num_epochs=50,
    warmup_epochs=2,
    lr=0.005,
    output_base="outputs/quartznet",
    save_every=10,
    resume_from=None,
    batch_size=180,
    num_workers=16,
    compile_model=True,
    run_id=None,
    augmentation=None,
):
    rank, local_rank, world_size, device = _setup_distributed()
    is_main = _is_main_process(rank)

    if run_id is None:
        run_id = _generate_run_id(B, R)

    run_dir = _resolve_checkpoint_dir(output_base) / run_id
    log_csv = str(run_dir / "training_log.csv")

    try:
        train_ds, val_ds, test_ds = _build_datasets()
        train_loader, val_loader, test_loader, per_device_batch_size = _build_dataloaders(
            train_ds, val_ds, test_ds, batch_size, num_workers, rank, world_size
        )

        if is_main:
            print(f"Run ID: {run_id}")
            print(f"Using device: {device}")
            print(f"Distributed training | enabled: {world_size > 1} | world_size: {world_size}")
            print(f"Dataset sizes | train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}")
            print(f"Dataloader batches | train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}")
            print(f"Batch sizes | global target: {batch_size} | per GPU: {per_device_batch_size}")
            print(f"Starting training for {num_epochs} epochs with QuartzNet-{B}x{R}")

        base_model = QuartzNetBxR(n_mels=64, n_classes=29, B=B, R=R).to(device)
        total_params = sum(p.numel() for p in base_model.parameters())
        if is_main:
            print(f"Model parameters: {total_params:,}")

        train_model_ref = base_model
        if compile_model and world_size == 1:
            train_model_ref = torch.compile(base_model)
        elif compile_model and world_size > 1 and is_main:
            print("Skipping torch.compile under DDP for stability.")

        if world_size > 1:
            train_model_ref = DDP(train_model_ref, device_ids=[local_rank], output_device=local_rank)

        optimizer = NovoGrad(
            train_model_ref.parameters(),
            lr=lr,
            betas=(0.95, 0.5),
            weight_decay=0.001,
        )
        criterion = nn.CTCLoss(blank=28, zero_infinity=True)

        steps_per_epoch = len(train_loader)
        total_steps = num_epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        cosine_steps = total_steps - warmup_steps

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, cosine_steps), eta_min=1e-4
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        if is_main:
            print(f"Output directory: {run_dir}")
            write_training_config(
                model=base_model,
                checkpoint_dir=run_dir,
                B=B,
                R=R,
                n_mels=64,
                n_classes=29,
                num_epochs=num_epochs,
                warmup_epochs=warmup_epochs,
                lr=lr,
                batch_size=batch_size,
                optimizer_name="NovoGrad(betas=(0.95,0.5), wd=0.001)",
                train_size=len(train_ds),
                val_size=len(val_ds),
                test_size=len(test_ds),
                device=f"{device} | world_size={world_size} | per_device_batch_size={per_device_batch_size}",
                augmentation=augmentation,
            )

        start_epoch = 0
        best_val_loss = float("inf")
        best_val_wer = float("inf")
        train_losses = []
        val_losses = []
        train_wers = []
        val_wers = []
        log_interval = max(1, len(train_loader) // 20)

        if resume_from:
            resume_path = Path(resume_from)
            if not resume_path.is_absolute():
                resume_path = run_dir / resume_path
            checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
            base_model.load_state_dict(checkpoint["model_state_dict"])
            optimizer_state = checkpoint.get("optimizer_state_dict")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            elif is_main:
                print("Resume checkpoint has no optimizer state; optimizer starts fresh.")
            start_epoch = checkpoint["epoch"] + 1
            train_losses = checkpoint.get("train_losses", [])
            val_losses = checkpoint.get("val_losses", [])
            train_wers = checkpoint.get("train_wers", [])
            val_wers = checkpoint.get("val_wers", [])
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            best_val_wer = checkpoint.get("best_val_wer", float("inf"))
            scheduler_state = checkpoint.get("scheduler_state_dict")
            if scheduler_state is not None:
                scheduler.load_state_dict(scheduler_state)
            if is_main:
                print(f"Resumed from checkpoint: {resume_path} (starting at epoch {start_epoch + 1})")

        if is_main:
            print("-" * 90)

        autocast_enabled = device.type == "cuda"

        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()

            if world_size > 1 and hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            train_model_ref.train()
            total_train_loss = 0.0
            total_train_word_errors = 0
            total_train_ref_words = 0
            if is_main:
                print(f"\nEpoch {epoch+1}/{num_epochs} | Training")
            train_start_time = time.time()

            for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
                if is_main and batch_idx == 0:
                    print(f"First batch shapes | inputs: {tuple(inputs.shape)} | targets: {tuple(targets.shape)}")

                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                input_lengths = input_lengths.to(device, non_blocking=True)
                target_lengths = target_lengths.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                    outputs = train_model_ref(inputs)

                outputs = outputs.float().permute(2, 0, 1).log_softmax(dim=2)
                adjusted_lengths = ((input_lengths - 1) // 2) + 1
                loss = criterion(outputs, targets, adjusted_lengths, target_lengths)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model_ref.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                total_train_loss += loss.item()
                wer_logits = outputs.detach().permute(1, 2, 0)
                batch_word_errors, batch_ref_words = batch_word_errors_and_count(wer_logits, targets, target_lengths)
                total_train_word_errors += batch_word_errors
                total_train_ref_words += batch_ref_words

                batch_number = batch_idx + 1
                if is_main and (batch_number % log_interval == 0 or batch_number == len(train_loader)):
                    running_avg_loss = total_train_loss / batch_number
                    elapsed = time.time() - train_start_time
                    avg_batch_time = elapsed / batch_number
                    eta_seconds = avg_batch_time * (len(train_loader) - batch_number)
                    progress = (batch_number / len(train_loader)) * 100
                    running_train_wer = (total_train_word_errors / max(1, total_train_ref_words)) * 100
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Train {batch_number:>4}/{len(train_loader)} ({progress:5.1f}%) | "
                        f"Loss: {loss.item():.4f} | Avg: {running_avg_loss:.4f} | "
                        f"WER: {running_train_wer:.2f}% | "
                        f"ETA: {_format_seconds(eta_seconds)}"
                    )

            reduced_train_loss, reduced_train_word_errors, reduced_train_ref_words, reduced_batches = (
                _reduce_train_metrics(total_train_loss, total_train_word_errors, total_train_ref_words, len(train_loader), device)
            )
            avg_train_loss = reduced_train_loss / max(1.0, reduced_batches)
            avg_train_wer = (reduced_train_word_errors / max(1.0, reduced_train_ref_words)) * 100
            train_duration = time.time() - train_start_time

            if is_main:
                train_losses.append(avg_train_loss)
                train_wers.append(avg_train_wer)
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Train done | "
                    f"Avg Loss: {avg_train_loss:.4f} | WER: {avg_train_wer:.2f}% | "
                    f"Time: {_format_seconds(train_duration)}"
                )

            _barrier()

            if is_main:
                base_model.eval()
                total_val_loss = 0.0
                total_val_word_errors = 0
                total_val_ref_words = 0
                print(f"Epoch {epoch+1}/{num_epochs} | Validation")
                val_start_time = time.time()
                with torch.no_grad():
                    for val_batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(val_loader):
                        inputs = inputs.to(device, non_blocking=True)
                        targets = targets.to(device, non_blocking=True)
                        input_lengths = input_lengths.to(device, non_blocking=True)
                        target_lengths = target_lengths.to(device, non_blocking=True)

                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled):
                            outputs = base_model(inputs)

                        outputs = outputs.float().permute(2, 0, 1).log_softmax(dim=2)
                        adjusted_lengths = ((input_lengths - 1) // 2) + 1
                        val_loss = criterion(outputs, targets, adjusted_lengths, target_lengths)
                        total_val_loss += val_loss.item()

                        wer_logits = outputs.detach().permute(1, 2, 0)
                        val_word_errors, val_ref_words = batch_word_errors_and_count(wer_logits, targets, target_lengths)
                        total_val_word_errors += val_word_errors
                        total_val_ref_words += val_ref_words

                        val_batch_number = val_batch_idx + 1
                        if val_batch_number % log_interval == 0 or val_batch_number == len(val_loader):
                            running_val_avg_loss = total_val_loss / val_batch_number
                            progress = (val_batch_number / len(val_loader)) * 100
                            running_val_wer = (total_val_word_errors / max(1, total_val_ref_words)) * 100
                            print(
                                f"Epoch {epoch+1}/{num_epochs} | "
                                f"Val   {val_batch_number:>4}/{len(val_loader)} ({progress:5.1f}%) | "
                                f"Loss: {val_loss.item():.4f} | Avg: {running_val_avg_loss:.4f} | "
                                f"WER: {running_val_wer:.2f}%"
                            )

                avg_val_loss = total_val_loss / len(val_loader)
                avg_val_wer = (total_val_word_errors / max(1, total_val_ref_words)) * 100
                val_losses.append(avg_val_loss)
                val_wers.append(avg_val_wer)
                val_duration = time.time() - val_start_time
                epoch_duration = time.time() - epoch_start_time
                eta_epochs_seconds = (num_epochs - (epoch + 1)) * epoch_duration
                current_lr = scheduler.get_last_lr()[0]

                print(
                    f"Epoch {epoch+1}/{num_epochs} complete | "
                    f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                    f"Train WER: {avg_train_wer:.2f}% | Val WER: {avg_val_wer:.2f}% | "
                    f"LR: {current_lr:.2e} | "
                    f"Epoch Time: {_format_seconds(epoch_duration)} | "
                    f"Val Time: {_format_seconds(val_duration)} | "
                    f"ETA to finish: {_format_seconds(eta_epochs_seconds)}"
                )

                log_epoch(
                    csv_path=log_csv,
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=avg_val_loss,
                    train_wer=avg_train_wer,
                    val_wer=avg_val_wer,
                    prev_train_loss=train_losses[-2] if len(train_losses) > 1 else None,
                    prev_val_loss=val_losses[-2] if len(val_losses) > 1 else None,
                    prev_train_wer=train_wers[-2] if len(train_wers) > 1 else None,
                    prev_val_wer=val_wers[-2] if len(val_wers) > 1 else None,
                    run_id=run_id,
                )

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                if avg_val_wer < best_val_wer:
                    best_val_wer = avg_val_wer

                checkpoint_payload = {
                    "run_id": run_id,
                    "epoch": epoch,
                    "model_state_dict": base_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_wers": train_wers,
                    "val_wers": val_wers,
                    "best_val_loss": best_val_loss,
                    "best_val_wer": best_val_wer,
                    "config": {
                        "B": B,
                        "R": R,
                        "n_mels": 64,
                        "n_classes": 29,
                    },
                }

                last_ckpt_path = run_dir / "last.pt"
                _save_checkpoint(last_ckpt_path, checkpoint_payload)
                print(f"Saved last checkpoint: {last_ckpt_path}")

                if avg_val_wer <= best_val_wer:
                    best_ckpt_path = run_dir / "best.pt"
                    _save_checkpoint(best_ckpt_path, _build_inference_payload(
                        base_model, B, R, epoch, best_val_loss, best_val_wer, run_id
                    ))
                    print(f"New best checkpoint (val WER: {avg_val_wer:.2f}%): {best_ckpt_path}")

                if save_every > 0 and ((epoch + 1) % save_every == 0):
                    epoch_ckpt_path = run_dir / f"epoch_{epoch + 1:03d}.pt"
                    _save_checkpoint(epoch_ckpt_path, checkpoint_payload)
                    print(f"Saved periodic checkpoint: {epoch_ckpt_path}")

                print("-" * 90)

            _barrier()

        if is_main:
            final_model_path = run_dir / "final_model.pt"
            torch.save(_build_inference_payload(base_model, B, R, num_epochs - 1, best_val_loss, best_val_wer, run_id), final_model_path)
            print(f"Saved final model weights: {final_model_path}")

        return base_model, train_losses, val_losses
    finally:
        _cleanup_distributed()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--B", type=int, default=5, help="Model depth: 5 (QuartzNet-5x5), 10 (10x5), or 15 (15x5)")
    parser.add_argument("--R", type=int, default=5, help="Number of modules per block")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=180)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--output-dir", default="outputs/quartznet", help="Base output dir; run saved to <output-dir>/<run-id>/")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--log-aug-speed", action="store_true", help="Label this run as using speed perturbation (config/checkpoint only)")
    parser.add_argument("--log-aug-specaugment", action="store_true", help="Label this run as using SpecAugment (config/checkpoint only)")
    parser.add_argument("--log-aug-speccutout", action="store_true", help="Label this run as using SpecCutout (config/checkpoint only)")
    args = parser.parse_args()

    errors = []

    if not root:
        errors.append("ROOT is not set. Add it to your environment or .env file.")

    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute() and not resume_path.exists():
            errors.append(f"--resume: file not found: {resume_path}")

    output_base_path = _resolve_checkpoint_dir(args.output_dir)
    if output_base_path.exists() and not output_base_path.is_dir():
        errors.append(f"--output-dir: '{output_base_path}' exists but is not a directory")

    if args.batch_size < 1:
        errors.append("--batch-size must be >= 1")
    if args.num_workers < 0:
        errors.append("--num-workers must be >= 0")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    train_model(
        B=args.B,
        R=args.R,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup,
        lr=args.lr,
        output_base=args.output_dir,
        save_every=args.save_every,
        resume_from=args.resume,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        compile_model=not args.no_compile,
        run_id=_generate_run_id(args.B, args.R, "-".join(filter(None, [
            "speed" if args.log_aug_speed else "",
            "specaug" if args.log_aug_specaugment else "",
            "cutout" if args.log_aug_speccutout else "",
        ]))),
        augmentation={
            "speed_perturb": args.log_aug_speed,
            "spec_augment": args.log_aug_specaugment,
            "spec_cutout": args.log_aug_speccutout,
        },
    )
