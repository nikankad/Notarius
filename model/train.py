import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch.nn as nn
import torch
import time
from pathlib import Path
from torchaudio.datasets import LIBRISPEECH
from helpers import collate_fn_test, get_dataset_lengths, BucketBatchSampler, log_epoch, collate_fn, batch_word_errors_and_count
from model import Notarius
from model_old import QuartzNetBxR as QuartzNet
from torch.utils.data import DataLoader, Subset
from torch_optimizer import NovoGrad
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Access the variables using os.getenv()
root = str(os.getenv("ROOT"))
# settings

train_ds = LIBRISPEECH(root=root, url="train-other-500", download=False)
val_ds = LIBRISPEECH(root=root, url="dev-clean", download=False)
test_ds = LIBRISPEECH(root=root, url="test-clean", download=False)

# Create subsets (40% of training dataset)
train_ds = Subset(train_ds, range(2 * len(train_ds) // 5))
# initialize dataloader
print("Pre-computing dataset lengths for bucket batching...")
_train_lengths = get_dataset_lengths(train_ds.dataset)
train_sampler = BucketBatchSampler([_train_lengths[i] for i in train_ds.indices], batch_size=128, shuffle=True)
val_sampler   = BucketBatchSampler(get_dataset_lengths(val_ds),   batch_size=128 , shuffle=False)
test_sampler  = BucketBatchSampler(get_dataset_lengths(test_ds),  batch_size=128 , shuffle=False)

train_loader = DataLoader(train_ds, batch_sampler=train_sampler, collate_fn=collate_fn,
                          num_workers=16, pin_memory=True, persistent_workers=True, prefetch_factor=4)
val_loader = DataLoader(val_ds,   batch_sampler=val_sampler,
                        collate_fn=collate_fn_test, num_workers=16, persistent_workers=True, pin_memory=True)
test_loader = DataLoader(test_ds,  batch_sampler=test_sampler,
                         collate_fn=collate_fn_test, num_workers=16, persistent_workers=True,pin_memory=True)


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



def _build_inference_payload(model, B: int, R: int, epoch: int, best_val_loss: float):
    return {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "config": {
            "B": B,
            "R": R,
            "n_mels": 64,
            "n_classes": 29,
        },
    }


def train_model(B=5, R=5, num_epochs=10, warmup_epochs=5, lr=0.04, checkpoint_dir="outputs/checkpoints", save_every=10, resume_from=None, log_csv="outputs/training_log.csv"):
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(
        f"Dataset sizes | train: {len(train_ds)} | val: {len(val_ds)} | test: {len(test_ds)}"
    )
    print(
        f"Dataloader batches | train: {len(train_loader)} | val: {len(val_loader)} | test: {len(test_loader)}"
    )
    print(
        f"Starting training for {num_epochs} epochs with QuartzNet B={B}, R={R}")

    model = QuartzNet(n_mels=64, n_classes=29, B=B, R=R).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    model = torch.compile(model)
    optimizer = NovoGrad(model.parameters(), lr=lr,
                         betas=(0.95, 0.5), weight_decay=0.001)
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    # Calculate step-based scheduling
    steps_per_epoch = len(train_loader)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_steps = total_steps - warmup_steps

    print(f"Learning rate schedule | warmup_steps: {warmup_steps} | cosine_steps: {cosine_steps}")

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cosine_steps), eta_min=1e-4
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    checkpoint_dir_path = _resolve_checkpoint_dir(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir_path}")

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
            resume_path = checkpoint_dir_path / resume_path
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        else:
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
        print(f"Resumed from checkpoint: {resume_path} (starting at epoch {start_epoch + 1})")

    print("-" * 90)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        total_train_loss = 0.0
        total_train_word_errors = 0
        total_train_ref_words = 0
        print(f"\nEpoch {epoch+1}/{num_epochs} | Training")
        train_start_time = time.time()

        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            if batch_idx == 0:
                print(
                    f"First batch shapes | inputs: {tuple(inputs.shape)} | targets: {tuple(targets.shape)}"
                )
            inputs = inputs.to(device)
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(inputs)

            outputs = outputs.float().permute(2, 0, 1).log_softmax(dim=2)
            adjusted_lengths = ((input_lengths - 1) // 2) + 1
            loss = criterion(outputs, targets,
                             adjusted_lengths, target_lengths)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

            batch_number = batch_idx + 1
            if batch_number % log_interval == 0 or batch_number == len(train_loader):
                wer_logits = outputs.detach().permute(1, 2, 0)  # (B, C, T)
                train_word_errors, train_ref_words = batch_word_errors_and_count(
                    wer_logits, targets, target_lengths
                )
                total_train_word_errors += train_word_errors
                total_train_ref_words += train_ref_words

                running_avg_loss = total_train_loss / (batch_idx + 1)
                elapsed = time.time() - train_start_time
                avg_batch_time = elapsed / batch_number
                eta_seconds = avg_batch_time * \
                    (len(train_loader) - batch_number)
                progress = (batch_number / len(train_loader)) * 100
                running_train_wer = (
                    total_train_word_errors / max(1, total_train_ref_words)
                ) * 100
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train {batch_number:>4}/{len(train_loader)} ({progress:5.1f}%) | "
                    f"Loss: {loss.item():.4f} | Avg: {running_avg_loss:.4f} | "
                    f"WER: {running_train_wer:.2f}% | "
                    f"ETA: {_format_seconds(eta_seconds)}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_wer = (total_train_word_errors / max(1, total_train_ref_words)) * 100
        train_losses.append(avg_train_loss)
        train_wers.append(avg_train_wer)
        train_duration = time.time() - train_start_time
        print(
            f"Epoch {epoch+1}/{num_epochs} | Train done | "
            f"Avg Loss: {avg_train_loss:.4f} | WER: {avg_train_wer:.2f}% | "
            f"Time: {_format_seconds(train_duration)}"
        )

        # Validation
        model.eval()
        total_val_loss = 0.0
        total_val_word_errors = 0
        total_val_ref_words = 0
        print(f"Epoch {epoch+1}/{num_epochs} | Validation")
        val_start_time = time.time()
        with torch.no_grad():
            for val_batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                input_lengths = input_lengths.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(inputs)
                outputs = outputs.permute(2, 0, 1).log_softmax(dim=2)

                adjusted_lengths = ((input_lengths - 1) // 2) + 1
                val_loss = criterion(
                    outputs, targets, adjusted_lengths, target_lengths)
                total_val_loss += val_loss.item()

                val_batch_number = val_batch_idx + 1
                if val_batch_number % log_interval == 0 or val_batch_number == len(val_loader):
                    wer_logits = outputs.detach().permute(1, 2, 0)  # (B, C, T)
                    val_word_errors, val_ref_words = batch_word_errors_and_count(
                        wer_logits, targets, target_lengths
                    )
                    total_val_word_errors += val_word_errors
                    total_val_ref_words += val_ref_words

                    running_val_avg_loss = total_val_loss / (val_batch_idx + 1)
                    progress = (val_batch_number / len(val_loader)) * 100
                    running_val_wer = (
                        total_val_word_errors / max(1, total_val_ref_words)
                    ) * 100
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
            train_loss=avg_train_loss, val_loss=avg_val_loss,
            train_wer=avg_train_wer,   val_wer=avg_val_wer,
            prev_train_loss=train_losses[-2] if len(train_losses) > 1 else None,
            prev_val_loss=val_losses[-2]     if len(val_losses)   > 1 else None,
            prev_train_wer=train_wers[-2]    if len(train_wers)   > 1 else None,
            prev_val_wer=val_wers[-2]        if len(val_wers)     > 1 else None,
            total_params=total_params,
        )

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_wers": train_wers,
            "val_wers": val_wers,
            "best_val_loss": best_val_loss,
            "config": {
                "B": B,
                "R": R,
                "n_mels": 64,
                "n_classes": 29,
            },
        }

        last_ckpt_path = checkpoint_dir_path / "last.pt"
        _save_checkpoint(last_ckpt_path, checkpoint_payload)
        print(f"Saved last checkpoint: {last_ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        if avg_val_wer < best_val_wer:
            best_val_wer = avg_val_wer
            checkpoint_payload["best_val_wer"] = best_val_wer
            checkpoint_payload["best_val_loss"] = best_val_loss

            best_ckpt_path = checkpoint_dir_path / "best.pt"
            best_payload = _build_inference_payload(
                model=model,
                B=B,
                R=R,
                epoch=epoch,
                best_val_loss=best_val_loss,
            )
            _save_checkpoint(best_ckpt_path, best_payload)
            print(f"New best checkpoint (val WER: {avg_val_wer:.2f}%): {best_ckpt_path}")

        if save_every > 0 and ((epoch + 1) % save_every == 0):
            epoch_ckpt_path = checkpoint_dir_path / f"epoch_{epoch + 1:03d}.pt"
            _save_checkpoint(epoch_ckpt_path, checkpoint_payload)
            print(f"Saved periodic checkpoint: {epoch_ckpt_path}")

        print("-" * 90)

    final_model_path = checkpoint_dir_path / "final_model.pt"
    final_payload = _build_inference_payload(
        model=model,
        B=B,
        R=R,
        epoch=num_epochs - 1,
        best_val_loss=best_val_loss,
    )
    torch.save(final_payload, final_model_path)
    print(f"Saved final model weights: {final_model_path}")

    return model, train_losses, val_losses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",      default=None,                        help="Checkpoint to resume from")
    parser.add_argument("--epochs",      type=int,   default=50)
    parser.add_argument("--B",           type=int,   default=5)
    parser.add_argument("--R",           type=int,   default=5)
    parser.add_argument("--lr",          type=float, default=0.005)
    parser.add_argument("--warmup",      type=int,   default=2)
    parser.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    parser.add_argument("--log-csv",     default="outputs/training_log.csv")
    args = parser.parse_args()
    train_model(B=args.B, R=args.R, num_epochs=args.epochs, warmup_epochs=args.warmup, lr=args.lr,
                checkpoint_dir=args.checkpoint_dir, resume_from=args.resume, log_csv=args.log_csv)
