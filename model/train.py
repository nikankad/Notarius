import torch.nn as nn
import torch
import time
from torchaudio.datasets import LIBRISPEECH
from helpers import collate_fn, collate_fn_test
from model import QuartzNetBxR
# from dataset import LocalLibriSpeechDataset
from torch.utils.data import DataLoader
from torch_optimizer import NovoGrad
import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

# Access the variables using os.getenv()
root = str(os.getenv("ROOT"))
# settings
torch.set_num_threads(12)
torch.backends.cudnn.benchmark = True

train_ds = LIBRISPEECH(root=root, url="train-clean-100", download=False)
val_ds = LIBRISPEECH(root=root, url="dev-clean", download=False)
test_ds = LIBRISPEECH(root=root, url="test-clean", download=False)


# initialize dataloader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn,
                          num_workers=32, pin_memory=True, persistent_workers=True, prefetch_factor=2)
val_loader = DataLoader(val_ds,   batch_size=64, shuffle=False,
                        collate_fn=collate_fn_test, num_workers=32, persistent_workers=True)
test_loader = DataLoader(test_ds,  batch_size=64, shuffle=False,
                         collate_fn=collate_fn_test, num_workers=32, persistent_workers=True)


def _format_seconds(seconds: float) -> str:
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def train_model(B=5, R=5, num_epochs=10):
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

    model = QuartzNetBxR(n_mels=64, n_classes=29).to(device)
    optimizer = NovoGrad(model.parameters(), lr=0.01,
                         betas=(0.95, 0.5), weight_decay=0.001)
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    train_losses = []
    val_losses = []
    log_interval = max(1, len(train_loader) // 20)

    print("-" * 90)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training
        model.train()
        total_train_loss = 0.0
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

            outputs = model(inputs)
            outputs = outputs.permute(2, 0, 1).log_softmax(dim=2)

            adjusted_lengths = ((input_lengths - 1) // 2) + 1
            loss = criterion(outputs, targets,
                             adjusted_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            batch_number = batch_idx + 1
            if batch_number % log_interval == 0 or batch_number == len(train_loader):
                running_avg_loss = total_train_loss / (batch_idx + 1)
                elapsed = time.time() - train_start_time
                avg_batch_time = elapsed / batch_number
                eta_seconds = avg_batch_time * \
                    (len(train_loader) - batch_number)
                progress = (batch_number / len(train_loader)) * 100
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train {batch_number:>4}/{len(train_loader)} ({progress:5.1f}%) | "
                    f"Loss: {loss.item():.4f} | Avg: {running_avg_loss:.4f} | "
                    f"ETA: {_format_seconds(eta_seconds)}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_duration = time.time() - train_start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Train done | Avg Loss: {avg_train_loss:.4f} | Time: {_format_seconds(train_duration)}")

        # Validation
        model.eval()
        total_val_loss = 0.0
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
                    running_val_avg_loss = total_val_loss / (val_batch_idx + 1)
                    progress = (val_batch_number / len(val_loader)) * 100
                    print(
                        f"Epoch {epoch+1}/{num_epochs} | "
                        f"Val   {val_batch_number:>4}/{len(val_loader)} ({progress:5.1f}%) | "
                        f"Loss: {val_loss.item():.4f} | Avg: {running_val_avg_loss:.4f}"
                    )

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_duration = time.time() - val_start_time
        epoch_duration = time.time() - epoch_start_time
        eta_epochs_seconds = (num_epochs - (epoch + 1)) * epoch_duration

        print(
            f"Epoch {epoch+1}/{num_epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Epoch Time: {_format_seconds(epoch_duration)} | "
            f"Val Time: {_format_seconds(val_duration)} | "
            f"ETA to finish: {_format_seconds(eta_epochs_seconds)}"
        )
        print("-" * 90)

    return model, train_losses, val_losses


if __name__ == "__main__":
    train_model(B=5, R=5, num_epochs=10)
