import csv
import os
import json
import random
import torch.nn as nn
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, SpeedPerturbation
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Sampler
import torchaudio
# define vocabulary
chars = "abcdefghijklmnopqrstuvwxyz '"  # 26 + space + apostrophe = 28 chars
blank = len(chars)                       # 28 = CTC blank token
num_classes = len(chars) + 1            # 29 total

# char to index
char2idx = {c: i for i, c in enumerate(chars)}
idx2char = {i: c for i, c in enumerate(chars)}

spec_transform = nn.Sequential(
    MelSpectrogram(n_fft=400, sample_rate=16000,  hop_length=160, n_mels=64),
    AmplitudeToDB(stype="power", top_db=80)
)

speed_perturb = SpeedPerturbation(
    orig_freq=16000,
    factors=[0.9, 1.0, 1.1]
)

spec_aug_cut = nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=12),
    torchaudio.transforms.TimeMasking(time_mask_param=20),
)


def encode(transcript):
    transcript = transcript.lower()
    return [char2idx[c] for c in transcript if c in char2idx]


def decode(indices):
    return ''.join([idx2char[i] for i in indices if i != blank])

# collate function: pad waveforms and keep transcripts as targets while also returning the orignal lengths of data


def collate_fn_speed_perturb(batch):
    waveforms, _, transcripts, *_ = zip(*batch)

    # Apply random speed perturbation per utterance for training-time augmentation.
    perturbed_waveforms = [speed_perturb(w)[0] for w in waveforms]

    # Compute mel features per sample (no raw-audio padding first)
    feats = [spec_transform(w).squeeze(0).transpose(0, 1) for w in perturbed_waveforms]
    # each feat: (time, n_mels)

    # Frame lengths for CTC input_lengths
    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

    #  Pad along time and convert to model shape (batch, n_mels, time)
    tensors = pad_sequence(feats, batch_first=True)          # (B, T, M)
    tensors = tensors.transpose(1, 2).contiguous()           # (B, M, T)

    # Encode transcripts
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths




def get_dataset_lengths(dataset):
    """Read audio lengths in frames, cached to disk after the first run."""
    cache_path = os.path.join(dataset._path, "_lengths_cache.json")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)

    print(f"Building lengths cache at {cache_path} (one-time)...")
    lengths = []
    for fileid in dataset._walker:
        speaker_id, chapter_id, _ = fileid.split("-", 2)
        file_path = os.path.join(dataset._path, speaker_id, chapter_id, f"{fileid}.flac")
        lengths.append(torchaudio.info(file_path).num_frames)

    with open(cache_path, "w") as f:
        json.dump(lengths, f)

    return lengths


class BucketBatchSampler(Sampler):
    """Batch indices so that each batch contains similarly-lengthed sequences."""
    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        sorted_indices = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        batches = [sorted_indices[i:i + self.batch_size]
                   for i in range(0, len(sorted_indices), self.batch_size)]
        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def log_epoch(csv_path, epoch, train_loss, val_loss, train_wer, val_wer,
              prev_train_loss=None, prev_val_loss=None, prev_train_wer=None, prev_val_wer=None):
    def pct(current, previous):
        if previous is None:
            return ""
        return f"{((current - previous) / previous) * 100:.2f}"

    path = os.path.join(csv_path)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "epoch",
                "train_loss", "val_loss", "train_loss_delta_%", "val_loss_delta_%",
                "train_wer",  "val_wer",  "train_wer_delta_%",  "val_wer_delta_%",
            ])
        writer.writerow([
            epoch,
            f"{train_loss:.6f}", f"{val_loss:.6f}",
            pct(train_loss, prev_train_loss), pct(val_loss, prev_val_loss),
            f"{train_wer:.4f}",  f"{val_wer:.4f}",
            pct(train_wer,  prev_train_wer),  pct(val_wer,  prev_val_wer),
        ])


def collate_fn_test(batch):
    waveforms, _, transcripts, *_ = zip(*batch)

    # Compute mel features per sample (no raw-audio padding first)
    feats = [spec_transform(w).squeeze(0).transpose(0, 1) for w in waveforms]
    # each feat: (time, n_mels)

    # Frame lengths for CTC input_lengths
    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

    #  Pad along time and convert to model shape (batch, n_mels, time)
    tensors = pad_sequence(feats, batch_first=True)          # (B, T, M)
    tensors = tensors.transpose(1, 2).contiguous()           # (B, M, T)

    # Encode transcripts
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths
