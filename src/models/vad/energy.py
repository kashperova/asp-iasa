import numpy as np
from scipy.io import wavfile

from models.vad.base import BaseVAD


class EnergyVAD(BaseVAD):
    def __init__(
        self,
        window: int = 25,
        shift: int = 20,
        pre_emphasis: float = 0.95,
        energy_th: float = 0.05,
    ):
        """
        args
        ---------
        window: int
            frame length in milliseconds

        shift: int
            frame shift in milliseconds

        pre_emphasis: float
            pre-emphasis factor

        energy_th: float
            threshold for vad predictions
        """
        super().__init__()
        self.window = window
        self.shift = shift
        self.pre_emphasis = pre_emphasis
        self.energy_th = energy_th

    def _get_boundaries(self, audio_path: str, *args) -> list[tuple[float, float]]:
        """
        N.B. Energy VAD  works only with mono channel audio

        args
        ---------
        audio_path: str
            path to .wav file

        returns
        ---------
        list with boundaries (speech segments in seconds)
        """
        sr, waveform = wavfile.read(audio_path)
        # pre-emphasis (high-pass filter)
        waveform = np.append(
            waveform[0], waveform[1:] - self.pre_emphasis * waveform[:-1]
        )
        # compute energy
        energy = self.__compute_energy(waveform, sr)
        # apply energy threshold
        vad = np.zeros(energy.shape)
        vad[energy > self.energy_th] = 1

        return self.__to_boundaries(waveform, vad, sr)

    def __compute_energy(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        # todo: loudness normalization, low-pass filter
        # compute sample numbers of frame length and frame shift
        frame_len = self.window * sr // 1000
        frame_shift = self.shift * sr // 1000

        energy = np.zeros((waveform.shape[0] - frame_len + frame_shift) // frame_shift)
        for i in range(energy.shape[0]):
            energy[i] = np.sum(
                waveform[i * frame_shift : i * frame_shift + frame_len] ** 2
            )

        return energy

    def __to_boundaries(
        self, waveform: np.ndarray, vad: np.ndarray, sr: int
    ) -> list[tuple[float, float]]:
        boundaries, start = [], None

        for i, v in enumerate(vad):
            time = round(i * self.shift / sr, 4)
            if v == 1 and start is None:
                start = time
            elif v == 0 and start is not None:
                boundaries.append((start, time))
                start = None

        if start is not None:  # handle case when speech continues to the end
            boundaries.append((start, round(len(waveform) / sr, 4)))

        return  self._merge_boundaries(boundaries=boundaries, close_th=150)
