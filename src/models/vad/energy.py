import librosa
import numpy as np

from models.vad.base import BaseVAD


class EnergyVAD(BaseVAD):
    def __init__(self, low_energy: float = 4.0):
        super().__init__()
        self.low_energy = low_energy

    def _get_boundaries(
        self, audio_path: str, close_th: int = 450, *args
    ) -> list[tuple[float, float]]:
        waveform, sr = librosa.load(audio_path, sr=None)

        # split the signal into frames and compute energy and decisions
        frame_length = int(np.ceil(0.010 * sr))  # 10 ms
        frame_overlap = int(np.ceil(0.005 * sr))  # 5 ms
        frames = self.__buffer_waveform(waveform, frame_length, frame_overlap)
        num_frames = frames.shape[0]

        energy = np.zeros(num_frames)
        classes = np.zeros(num_frames, dtype=int)

        for k in range(num_frames):
            frame = frames[k]
            energy[k] = np.sum(frame**2)
            classes[k] = self.__classify(energy[k])

        boundaries = self.__create_boundaries(classes, frame_length, frame_overlap, sr)
        return self._merge_boundaries(boundaries=boundaries, close_th=close_th)

    def __classify(self, energy: float) -> int:
        # classify the frame
        # 0 – noise, 1 – voice
        return 0 if energy < self.low_energy else 1

    @staticmethod
    def __buffer_waveform(
        waveform: np.array, frame_length: int, overlap: int
    ) -> np.ndarray:
        # split the signal into overlapping frames.
        step = frame_length - overlap
        num_frames = int(np.floor((len(waveform) - overlap) / step))
        frames = np.empty((num_frames, frame_length))
        for i in range(num_frames):
            start = i * step
            frames[i] = waveform[start : start + frame_length]
        return frames

    @staticmethod
    def __create_boundaries(
        classes: np.array, frame_length: int, frame_overlap: int, sr: int
    ) -> list[tuple[float, float]]:
        boundaries = []
        frame_step = frame_length - frame_overlap
        num_frames = len(classes)
        i = 0
        while i < num_frames:
            if classes[i] != 0:
                start_frame = i
                while i < num_frames and classes[i] != 0:
                    i += 1
                end_frame = i - 1
                start_time = start_frame * frame_step / sr
                end_time = end_frame * frame_step / sr + frame_length / sr
                boundaries.append((start_time, end_time))
            else:
                i += 1

        return boundaries
