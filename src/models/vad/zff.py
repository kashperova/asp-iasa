from zff import utils
from zff.zff import zff_vad

from models.vad.base import BaseVAD


class ZffVAD(BaseVAD):
    def _get_boundaries(self, audio_path: str, *args) -> list[tuple[float, float]]:
        """
        Zero frequency filter from paper https://arxiv.org/abs/2206.13420

        args
        ---------
        audio_path: str
            path to .wav file

        returns
        ---------
        list with boundaries (speech segments in seconds)
        """
        sr, waveform = utils.load_audio(audio_path)
        boundaries = zff_vad(waveform, sr)
        boundaries = utils.smooth_decision(boundaries, sr)
        segments = utils.sample2time(waveform, sr, boundaries)
        segments = [(item[0], item[1]) for item in segments]
        return self._merge_boundaries(boundaries=segments, close_th=150)
