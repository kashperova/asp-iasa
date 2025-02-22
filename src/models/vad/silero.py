import torch

from models.vad.base import BaseVAD


class SileroVAD(BaseVAD):
    def __init__(self):
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )
        self.__model = model
        self.__get_speech_timestamps = utils[0]
        self.__read_audio = utils[2]

    def _get_boundaries(
        self, audio_path: str, merge_th: int = None, *args, **kwargs
    ) -> list[tuple[float, float]]:
        """
        args
        ---------
        audio_path: str
            path to .wav file

        returns
        ---------
        list with boundaries (speech segments in seconds)
        """
        waveform = self.__read_audio(audio_path)
        boundaries = self.__get_speech_timestamps(waveform, self.__model, return_seconds=True)
        boundaries = [(item["start"], item["end"]) for item in boundaries]
        if merge_th:
            boundaries = self._merge_boundaries(boundaries=boundaries, close_th=merge_th)

        return boundaries
