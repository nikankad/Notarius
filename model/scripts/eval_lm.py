"""Evaluate a trained checkpoint on test-clean and test-other using greedy + LM beam search."""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchaudio.datasets import LIBRISPEECH

from helpers import (
    BucketBatchSampler,
    batch_word_errors_and_count,
    collate_fn_test,
    get_dataset_lengths,
    chars,
    idx2char,
    word_edit_distance,
)
from model import IBNet
from qnmodel import QuartzNetBxR

root = os.getenv("ROOT")

DEFAULT_ARPA = "lm/6gram.arpa"


def _build_lm_decoder(arpa_path, alpha=0.5, beta=1.5, beam_width=100):
    from pyctcdecode import build_ctcdecoder
    vocab = list(chars)
    vocab.append("")  # CTC blank
    return build_ctcdecoder(vocab, arpa_path, unigrams=None,
                            alpha=alpha, beta=beta, unk_score_offset=-10.0)


def _batch_wer_lm(logits, targets, target_lengths, lm_decoder, beam_width=100):
    log_probs_batch = logits.permute(0, 2, 1).detach().cpu().numpy()
    targets_cpu = targets.detach().cpu()
    target_lengths_cpu = target_lengths.detach().cpu()
    total_errors = 0
    total_words = 0
    for i in range(log_probs_batch.shape[0]):
        hyp_text = lm_decoder.decode(log_probs_batch[i], beam_width=beam_width)
        target_len = int(target_lengths_cpu[i].item())
        ref_text = "".join(idx2char[t] for t in targets_cpu[i, :target_len].tolist())
        total_errors += word_edit_distance(ref_text.split(), hyp_text.split())
        total_words += len(ref_text.split())
    return total_errors, total_words


def _load_model(checkpoint_path, device):
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
            expand=config.get("expand", 2),
            C=config.get("C", 192),
        ).to(device)
        model_name = "IBNet"
    else:
        B = config.get("B", 5)
        R = config.get("R", 5)
        model = QuartzNetBxR(
            n_mels=config.get("n_mels", 64),
            n_classes=config.get("n_classes", 29),
            B=B,
            R=R,
        ).to(device)
        model_name = f"QuartzNet-{B}x{R}"

    model.load_state_dict(state_dict)
    model.eval()
    return model, model_name


def _evaluate_dataset(model, loader, device, lm_decoder, beam_width):
    total_greedy_errors = 0
    total_greedy_words = 0
    total_lm_errors = 0
    total_lm_words = 0

    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            input_lengths = input_lengths.to(device, non_blocking=True)
            target_lengths = target_lengths.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                outputs = model(inputs)

            outputs = outputs.float().permute(2, 0, 1).log_softmax(dim=2)
            logits = outputs.detach().permute(1, 2, 0)

            # Greedy WER
            ge, gw = batch_word_errors_and_count(logits, targets, target_lengths)
            total_greedy_errors += ge
            total_greedy_words += gw

            # LM WER
            le, lw = _batch_wer_lm(logits, targets, target_lengths, lm_decoder, beam_width)
            total_lm_errors += le
            total_lm_words += lw

    greedy_wer = (total_greedy_errors / max(1, total_greedy_words)) * 100
    lm_wer = (total_lm_errors / max(1, total_lm_words)) * 100
    return greedy_wer, lm_wer


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoint with greedy + LM decoding")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--arpa", default=DEFAULT_ARPA, help="Path to .arpa LM file")
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--lm-alpha", type=float, default=0.5)
    parser.add_argument("--lm-beta", type=float, default=1.5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--csv", default=None, help="Path to shared CSV file to append results to")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"LM: {args.arpa} (alpha={args.lm_alpha}, beta={args.lm_beta}, beam={args.beam_width})")
    print()

    model, model_name = _load_model(args.checkpoint, device)
    print(f"Model: {model_name} ({sum(p.numel() for p in model.parameters()):,} params)")

    print(f"Loading LM decoder...")
    lm_decoder = _build_lm_decoder(args.arpa, alpha=args.lm_alpha, beta=args.lm_beta, beam_width=args.beam_width)

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    # CSV output — shared file if --csv provided, otherwise next to checkpoint
    if args.csv:
        csv_path = Path(args.csv)
    else:
        checkpoint_dir = Path(args.checkpoint).resolve().parent
        csv_path = checkpoint_dir / "eval_lm.csv"
    write_header = not csv_path.exists()

    results = []
    for split_name, split_url in [("dev-clean", "dev-clean"), ("dev-other", "dev-other"), ("test-clean", "test-clean"), ("test-other", "test-other")]:
        print(f"\n{'='*60}")
        print(f"Evaluating: {split_name}")
        print(f"{'='*60}")

        ds = LIBRISPEECH(root=root, url=split_url, download=False)
        lengths = get_dataset_lengths(ds)
        sampler = BucketBatchSampler(lengths, batch_size=args.batch_size, shuffle=False)
        loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn_test, **loader_kwargs)

        start = time.time()
        greedy_wer, lm_wer = _evaluate_dataset(model, loader, device, lm_decoder, args.beam_width)
        elapsed = time.time() - start

        print(f"  Greedy WER : {greedy_wer:.2f}%")
        print(f"  LM WER    : {lm_wer:.2f}%")
        print(f"  Time       : {elapsed:.1f}s")

        results.append({
            "split": split_name,
            "greedy_wer": greedy_wer,
            "lm_wer": lm_wer,
            "time_s": elapsed,
        })

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "model", "checkpoint", "split",
                "greedy_wer", "lm_wer",
                "lm_arpa", "lm_alpha", "lm_beta", "beam_width",
            ])
        for r in results:
            writer.writerow([
                model_name,
                Path(args.checkpoint).name,
                r["split"],
                f"{r['greedy_wer']:.4f}",
                f"{r['lm_wer']:.4f}",
                args.arpa,
                args.lm_alpha,
                args.lm_beta,
                args.beam_width,
            ])

    print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()
