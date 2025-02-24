from typing import Union

import numpy as np
import torch
import torchaudio
from scipy.signal import wiener
from scipy import signal


def get_wav_metadata(file_path: str) -> dict[str, Union[int, float]]:
    metadata = torchaudio.info(file_path)
    return {
        "sr": metadata.sample_rate,
        "num_frames": metadata.num_frames,
        "duration": metadata.num_frames / metadata.sample_rate,
    }


def get_sr(file_path: str) -> int:
    return get_wav_metadata(file_path)["sr"]


def get_duration(file_path: str) -> float:
    return get_wav_metadata(file_path)["duration"]


def resample_file(
    audio_path: str, save_path: str, target_sr: int, to_mono: bool = False
):
    waveform, sr = torchaudio.load(audio_path)

    if to_mono and waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)

    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resampler(waveform)
    torchaudio.save(save_path, waveform.to(torch.float32), sample_rate=target_sr)


def apply_filters(
    waveform: np.array, sr: int, highpass: bool = True, lowpass: bool = True
) -> np.array:
    if highpass:
        hp_cutoff = 80
        b, a = signal.butter(4, hp_cutoff / (sr / 2), btype="highpass")
        waveform = signal.filtfilt(b, a, waveform)

    if lowpass:
        lp_cutoff = 7000
        b, a = signal.butter(4, lp_cutoff / (sr / 2), btype="lowpass")
        waveform = signal.filtfilt(b, a, waveform)

    return waveform


def apply_wiener(waveform: np.array, win_length: int = 7):
    return wiener(waveform, mysize=win_length)
