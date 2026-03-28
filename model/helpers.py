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



class SpecSquareCutout(nn.Module):
    """Zeros out num_holes random square patches per sample in a spectrogram batch.

    Args:
        num_holes: number of squares to cut per sample
        hole_size: side length of each square in bins/frames
    """
    def __init__(self, num_holes: int = 1, hole_size: int = 100):
        super().__init__()
        self.num_holes = num_holes
        self.hole_size = hole_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        B, F, T = x.shape
        x = x.clone()
        for b in range(B):
            for _ in range(self.num_holes):
                f0 = random.randint(0, max(0, F - self.hole_size))
                t0 = random.randint(0, max(0, T - self.hole_size))
                x[b, f0:f0 + self.hole_size, t0:t0 + self.hole_size] = 0.0
        return x


def encode(transcript):
    transcript = transcript.lower()
    return [char2idx[c] for c in transcript if c in char2idx]


def decode(indices):
    return ''.join([idx2char[i] for i in indices if i != blank])

# collate function: pad waveforms and keep transcripts as targets while also returning the orignal lengths of data


def collate_fn_speed_perturb(batch):
    waveforms, _, transcripts, *_ = zip(*batch)

    # Apply random speed perturbation per utterance for training-time augmentation.
    waveforms = [speed_perturb(w)[0] for w in waveforms]

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


def collate_fn(batch):
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

_spec_square_cutout = SpecSquareCutout(num_holes=1, hole_size=20)

def collate_fn_cutout(batch):
    waveforms, _, transcripts, *_ = zip(*batch)

    feats = [spec_transform(w).squeeze(0).transpose(0, 1) for w in waveforms]
    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tensors = pad_sequence(feats, batch_first=True).transpose(1, 2).contiguous()

    tensors = _spec_square_cutout(tensors)

    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths


def make_train_collate_fn(use_speed_perturb=False, use_spec_augment=False, use_spec_cutout=False):
    """Returns a training collate function with the requested augmentations.

    - use_spec_augment: FrequencyMasking (horizontal strips)
    - use_spec_cutout:  SpecSquareCutout (random squares per sample)
    """
    aug_steps = []
    if use_spec_augment:
        aug_steps.append(torchaudio.transforms.FrequencyMasking(freq_mask_param=12))
    if use_spec_cutout:
        aug_steps.append(SpecSquareCutout(num_holes=1, hole_size=20))
    spec_aug = nn.Sequential(*aug_steps) if aug_steps else None

    def _collate(batch):
        waveforms, _, transcripts, *_ = zip(*batch)

        if use_speed_perturb and random.random() < 0.667:
            waveforms = [speed_perturb(w)[0] for w in waveforms]

        feats = [spec_transform(w).squeeze(0).transpose(0, 1) for w in waveforms]
        input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
        tensors = pad_sequence(feats, batch_first=True).transpose(1, 2).contiguous()

        if spec_aug is not None:
            tensors = spec_aug(tensors)

        encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
        target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
        targets = pad_sequence(encoded, batch_first=True, padding_value=0)

        return tensors, targets, input_lengths, target_lengths

    return _collate


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
    """Batch indices so that each batch contains similarly-lengthed sequences.

    Samples are first sorted by length and divided into `num_buckets` equal
    buckets.  Batches are built within each bucket (not across boundaries), so
    within-batch length variance is governed by `num_buckets` rather than
    `batch_size`.  This keeps padding overhead constant regardless of batch size.
    """
    def __init__(self, lengths, batch_size, shuffle=True, num_buckets=200):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_buckets = num_buckets
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])
        self._total = len(lengths)

    def __iter__(self):
        sorted_indices = self.sorted_indices
        bucket_size = max(self.batch_size, len(sorted_indices) // self.num_buckets)
        buckets = [sorted_indices[i:i + bucket_size]
                   for i in range(0, len(sorted_indices), bucket_size)]

        batches = []
        for bucket in buckets:
            if self.shuffle:
                random.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batches.append(bucket[i:i + self.batch_size])

        if self.shuffle:
            random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self):
        return (self._total + self.batch_size - 1) // self.batch_size


class DistributedBucketBatchSampler(Sampler):
    """Shard bucketed batches across distributed workers."""

    def __init__(
        self,
        lengths,
        batch_size,
        num_replicas,
        rank,
        shuffle=True,
        num_buckets=200,
        drop_last=False,
        seed=42,
    ):
        if num_replicas < 1:
            raise ValueError("num_replicas must be >= 1")
        if rank < 0 or rank >= num_replicas:
            raise ValueError("rank must be in [0, num_replicas)")

        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.num_buckets = num_buckets
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _build_batches(self):
        sorted_indices = list(self.sorted_indices)
        bucket_size = max(self.batch_size, len(sorted_indices) // self.num_buckets)
        buckets = [
            sorted_indices[i:i + bucket_size]
            for i in range(0, len(sorted_indices), bucket_size)
        ]

        generator = random.Random(self.seed + self.epoch)
        batches = []
        for bucket in buckets:
            if self.shuffle:
                generator.shuffle(bucket)
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        if self.shuffle:
            generator.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._build_batches()
        if self.drop_last:
            total_size = (len(batches) // self.num_replicas) * self.num_replicas
            batches = batches[:total_size]
        elif batches:
            remainder = len(batches) % self.num_replicas
            if remainder != 0:
                batches.extend(batches[: self.num_replicas - remainder])

        for batch in batches[self.rank::self.num_replicas]:
            yield batch

    def __len__(self):
        batches = len(self._build_batches())
        if self.drop_last:
            return batches // self.num_replicas
        return (batches + self.num_replicas - 1) // self.num_replicas


def log_epoch(csv_path, epoch, train_loss, val_loss, train_wer, val_wer,
              prev_train_loss=None, prev_val_loss=None, prev_train_wer=None, prev_val_wer=None,
              run_id=None):
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
                "run_id",
                "epoch",
                "train_loss", "val_loss", "train_loss_delta_%", "val_loss_delta_%",
                "train_wer",  "val_wer",  "train_wer_delta_%",  "val_wer_delta_%",
            ])
        writer.writerow([
            run_id or "",
            epoch,
            f"{train_loss:.6f}", f"{val_loss:.6f}",
            pct(train_loss, prev_train_loss), pct(val_loss, prev_val_loss),
            f"{train_wer:.4f}",  f"{val_wer:.4f}",
            pct(train_wer,  prev_train_wer),  pct(val_wer,  prev_val_wer),
        ])


def ctc_greedy_decode(token_ids):
    decoded = []
    prev = None
    for token in token_ids:
        if token != blank and token != prev:
            decoded.append(idx2char[token])
        prev = token
    return "".join(decoded)


def word_edit_distance(ref_words, hyp_words):
    rows = len(ref_words) + 1
    cols = len(hyp_words) + 1
    dp = [[0] * cols for _ in range(rows)]

    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j

    for i in range(1, rows):
        for j in range(1, cols):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],
                    dp[i][j - 1],
                    dp[i - 1][j - 1],
                )

    return dp[-1][-1]


def batch_word_errors_and_count(logits, targets, target_lengths):
    pred_ids = logits.argmax(dim=1).detach().cpu()
    targets_cpu = targets.detach().cpu()
    target_lengths_cpu = target_lengths.detach().cpu()

    total_word_errors = 0
    total_ref_words = 0

    for i in range(pred_ids.size(0)):
        pred_text = ctc_greedy_decode(pred_ids[i].tolist())

        target_len = int(target_lengths_cpu[i].item())
        target_tokens = targets_cpu[i, :target_len].tolist()
        ref_text = "".join(idx2char[token] for token in target_tokens)

        ref_words = ref_text.split()
        hyp_words = pred_text.split()

        total_word_errors += word_edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)

    return total_word_errors, total_ref_words


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
