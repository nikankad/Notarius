import argparse
from pathlib import Path

import torch
import torchaudio
import torchaudio.functional as F

from helpers import blank, idx2char, spec_transform, chars
from qnmodel import QuartzNetBxR
from model import IBNet

# ---------------------------------------------------------------------------
# Greedy CTC decoder (no LM) — same as original transcribe.py
# ---------------------------------------------------------------------------

def _ctc_greedy_decode(token_ids):
    decoded = []
    prev = None
    for token in token_ids:
        if token != blank and token != prev:
            decoded.append(idx2char[token])
        prev = token
    return "".join(decoded)


# ---------------------------------------------------------------------------
# Build pyctcdecode beam search decoder with LM
# ---------------------------------------------------------------------------

def _build_lm_decoder(arpa_path: Path, alpha: float = 0.5, beta: float = 1.5, beam_width: int = 100):
    """
    Builds a pyctcdecode BeamSearchDecoderCTC loaded with a KenLM .arpa file.

    alpha : LM weight (higher = trust LM more)
    beta  : word insertion bonus (higher = encourages longer hypotheses)
    beam_width : number of beams to keep at each step
    """
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        raise ImportError(
            "pyctcdecode is not installed. Run: pip install pyctcdecode pypi-kenlm"
        )

    # pyctcdecode expects a flat list of vocab tokens in token-index order.
    # Blank must be the empty string "" at its index position.
    vocab = list(chars)          # indices 0-27: a-z, space, apostrophe
    vocab.append("")             # index 28: CTC blank → empty string

    decoder = build_ctcdecoder(
        vocab,
        str(arpa_path),
        unigrams=None,
        alpha=alpha,
        beta=beta,
        unk_score_offset=-10.0,
    )
    return decoder


# ---------------------------------------------------------------------------
# Model loading (identical to original transcribe.py)
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    is_ibnet = any("layer1" in k for k in state_dict.keys())
    if is_ibnet:
        model = IBNet(
            n_mels=config.get("n_mels", 64),
            n_classes=config.get("n_classes", 29),
            R=config.get("R", 3),
            expand=2,
            C=192,
        ).to(device)
    else:
        model = QuartzNetBxR(
            n_mels=config.get("n_mels", 64),
            n_classes=config.get("n_classes", 29),
            R=config.get("R", 5),
        ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Core transcription function
# ---------------------------------------------------------------------------

def transcribe_audio(
    audio_path: Path,
    checkpoint_path: Path,
    arpa_path: Path = None,
    alpha: float = 0.5,
    beta: float = 1.5,
    beam_width: int = 100,
    device: str = "auto",
) -> dict:
    """
    Transcribes audio and returns both greedy and (optionally) LM-decoded results.

    Returns a dict:
        {
            "greedy": "...",
            "lm":     "..."   # only present if arpa_path is given
        }
    """
    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model = _load_model(checkpoint_path, resolved_device)

    # Load and preprocess audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        waveform = F.resample(waveform, sample_rate, 16000)

    features = spec_transform(waveform).squeeze(0)   # (n_mels, time)
    inputs = features.unsqueeze(0).to(resolved_device)  # (1, n_mels, time)

    with torch.no_grad():
        logits = model(inputs)   # (1, n_classes, time)

    results = {}

    # --- Greedy decode ---
    token_ids = logits.argmax(dim=1).squeeze(0).tolist()
    results["greedy"] = _ctc_greedy_decode(token_ids)

    # --- LM beam search decode ---
    if arpa_path is not None:
        decoder = _build_lm_decoder(arpa_path, alpha=alpha, beta=beta, beam_width=beam_width)

        # pyctcdecode expects log-softmax probs shaped (time, vocab)
        log_probs = logits.squeeze(0).log_softmax(dim=0).T.cpu().numpy()  # (time, n_classes)
        results["lm"] = decoder.decode(log_probs, beam_width=beam_width)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio with QuartzNet — greedy and/or LM beam search"
    )
    parser.add_argument("--audio",      required=True,  help="Path to input .wav file")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--arpa",
        default=None,
        help="Path to .arpa LM file (e.g. 3-gram.pruned.3e-7.arpa). "
             "If omitted, only greedy decoding is run.",
    )
    parser.add_argument("--alpha",      type=float, default=0.5,  help="LM weight (default: 0.5)")
    parser.add_argument("--beta",       type=float, default=1.5,  help="Word insertion bonus (default: 1.5)")
    parser.add_argument("--beam-width", type=int,   default=100,  help="Beam width (default: 100)")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )

    args = parser.parse_args()

    audio_path      = Path(args.audio)
    checkpoint_path = Path(args.checkpoint)
    arpa_path       = Path(args.arpa) if args.arpa else None

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if arpa_path and not arpa_path.exists():
        raise FileNotFoundError(f"ARPA file not found: {arpa_path}")

    results = transcribe_audio(
        audio_path=audio_path,
        checkpoint_path=checkpoint_path,
        arpa_path=arpa_path,
        alpha=args.alpha,
        beta=args.beta,
        beam_width=args.beam_width,
        device=args.device,
    )

    print("\n=== Greedy (no LM) ===")
    print(results["greedy"])

    if "lm" in results:
        print("\n=== Beam search + LM ===")
        print(results["lm"])
        print()


if __name__ == "__main__":
    main()