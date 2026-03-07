import torchaudio

dataset = torchaudio.datasets.LIBRISPEECH(
    root="datasets/",
    url="train-clean-100",
    download=False
)

waveform, sample_rate, transcript, speaker_id, chapter_id, utt_id = dataset[0]

print(speaker_id)