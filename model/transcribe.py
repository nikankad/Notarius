import argparse
from pathlib import Path

import torch
import torchaudio

from helpers import blank, idx2char, spec_transform
from model import QuartzNetBxR


def _ctc_greedy_decode(token_ids):
    decoded = []
    prev = None
    for token in token_ids:
        if token != blank and token != prev:
            decoded.append(idx2char[token])
        prev = token
    return "".join(decoded)


def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True)

    config = checkpoint.get("config", {})
    model = QuartzNetBxR(
        n_mels=config.get("n_mels", 64),
        n_classes=config.get("n_classes", 29),
        B=config.get("B", 5),
        R=config.get("R", 5),
    ).to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

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

    return _ctc_greedy_decode(token_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with a trained QuartzNet checkpoint")
    parser.add_argument("--audio", required=True,
                        help="Path to input wav file")
    parser.add_argument(
        "--checkpoint",
        default="/home/xz/GOATS422/Notarius/outputs/checkpoints/epoch_080.pt",
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
