import torch.nn as nn
import torch
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, SpeedPerturbation
from torch.nn.utils.rnn import pad_sequence
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

spec_aug_mask = nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=12),
    torchaudio.transforms.TimeMasking(time_mask_param=20),
)


def spec_aug(spectrogram):
    return spec_aug_mask(spectrogram)


def encode(transcript):
    transcript = transcript.lower()
    return [char2idx[c] for c in transcript if c in char2idx]


def decode(indices):
    return ''.join([idx2char[i] for i in indices if i != blank])

# collate function: pad waveforms and keep transcripts as targets while also returning the orignal lengths of data


def collate_fn(batch):
    waveforms, _, transcripts, *_ = zip(*batch)

    # Compute mel features per sample (no raw-audio padding first)
    feats = [spec_transform(w).squeeze(0).transpose(0, 1) for w in waveforms]
    # each feat: (time, n_mels)

    # Frame lengths for CTC input_lengths
    input_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)

    # SpecAug on mel features before padding
    feats = [spec_aug(f.transpose(0, 1)).transpose(0, 1) for f in feats]

    #  Pad along time and convert to model shape (batch, n_mels, time)
    tensors = pad_sequence(feats, batch_first=True)          # (B, T, M)
    tensors = tensors.transpose(1, 2).contiguous()           # (B, M, T)

    # Encode transcripts
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths

# this collate method will not apply the spec augment transformationl.


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

    # SpecAug on mel features
    # tensors = spec_aug(tensors)

    # Encode transcripts
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t in transcripts]
    target_lengths = torch.tensor([len(e) for e in encoded], dtype=torch.long)
    targets = pad_sequence(encoded, batch_first=True, padding_value=0)

    return tensors, targets, input_lengths, target_lengths
