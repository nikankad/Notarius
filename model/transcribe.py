import argparse
from pathlib import Path

import torch
import torchaudio

from helpers import spec_transform, ctc_greedy_decode
from model import Notarius as QuartzNetBxR
from model_old import QuartzNetBxR as OldQuartzNetBxR


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True)

    config = checkpoint.get("config", {})

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Strip _orig_mod. prefix from torch.compile() checkpoints
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Detect old vs new architecture by checking key names
    is_old = any("net.3.net." in k for k in state_dict.keys())
    ModelClass = OldQuartzNetBxR if is_old else QuartzNetBxR

    model = ModelClass(
        n_mels=config.get("n_mels", 64),
        n_classes=config.get("n_classes", 29),
        B=config.get("B", 5),
        R=config.get("R", 5),
    ).to(device)

    model.load_state_dict(state_dict)

    model.eval()
    return model


def transcribe_audio(audio_path: Path, checkpoint_path: Path, device: str = "auto") -> str:
    if device == "auto":
        resolved_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model = _load_model(checkpoint_path, resolved_device)

    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    features = spec_transform(waveform).squeeze(0)
    inputs = features.unsqueeze(0).to(resolved_device)

    with torch.no_grad():
        outputs = model(inputs)
        token_ids = outputs.argmax(dim=1).squeeze(0).tolist()

    return ctc_greedy_decode(token_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with a trained QuartzNet checkpoint")
    parser.add_argument("--audio", required=True,
                        help="Path to input wav file")
    parser.add_argument(
        "--checkpoint",
        default="/home/xz/GOATS422/Notarius/outputs/checkpoints_speed_perturb2/epoch_050.pt",
        help="Path to model checkpoint (default: outputs/checkpoints/best.pt)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    checkpoint_path = Path(args.checkpoint)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint file not found: {checkpoint_path}")

    transcript = transcribe_audio(audio_path, checkpoint_path, args.device)
    print(transcript)


if __name__ == "__main__":
    main()
