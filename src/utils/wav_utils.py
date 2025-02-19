from typing import Union

import torch
import torchaudio


def get_wav_metadata(file_path: str) -> dict[str, Union[int, float]]:
    metadata = torchaudio.info(file_path)
    return {
        "sr": metadata.sample_rate,
        "num_frames": metadata.num_frames,
        "duration": metadata.num_frames / metadata.sample_rate,
    }


def resample_file(
    audio_path: str, save_path: str, target_sr: int, to_mono: bool = False
):
    waveform, sr = torchaudio.load(audio_path)

    if to_mono and waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)

    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    waveform = resampler(waveform)
    torchaudio.save(save_path, waveform.to(torch.float32), sample_rate=target_sr)
