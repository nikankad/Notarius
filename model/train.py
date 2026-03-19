
import torch.nn as nn 
import torch
from helpers import collate_fn, collate_fn_test
from model import QuartzNetBxR
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader
from torch_optimizer import NovoGrad
root = "/home/student/GOATS422/Notarius/datasets"
# settings
torch.set_num_threads(24)
torch.backends.cudnn.benchmark = True

train_ds = LIBRISPEECH(root=root, url="train-clean-100",download=False)
val_ds   = LIBRISPEECH(root=root, url="dev-clean",download=False)
test_ds  = LIBRISPEECH(root=root, url="test-clean",download=False)


#initialize dataloader
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True,  collate_fn=collate_fn, num_workers=24, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=24)
test_loader  = DataLoader(test_ds,  batch_size=64, shuffle=False, collate_fn=collate_fn_test, num_workers=24)


def train_model(B=5, R=5, num_epochs=10):
    # Initialize model, optimizer, and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = QuartzNetBxR(n_mels=64, n_classes=29, B=B, R=R).to(device)
    optimizer = NovoGrad(model.parameters(), lr=0.01, betas=(0.95, 0.5), weight_decay=0.001)
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            adjusted_lengths = ((input_lengths - 1) // 2) + 1
            loss = criterion(outputs, targets, adjusted_lengths, target_lengths)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | "
                    f"Batch {batch_idx}/{len(train_loader)} | "
                    f"Batch Loss: {loss.item():.4f}",
                    end='\r'
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets, input_lengths, target_lengths in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                adjusted_lengths = ((input_lengths - 1) // 2) + 1
                val_loss = criterion(outputs, targets, adjusted_lengths, target_lengths)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} complete | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f}"
        )


    return model, train_losses, val_losses


if __name__ == "__main__":
    train_model(num_epochs=10)
    
