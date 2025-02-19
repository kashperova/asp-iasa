import os
from typing import Union

from utils.singleton import Singleton


class BaseVAD(metaclass=Singleton):
    def _get_boundaries(
        self, audio_path: str, *args, **kwargs
    ) -> list[tuple[float, float]]:
        raise NotImplementedError

    def get_boundaries(
        self, audio_path: str, *args, **kwargs
    ) -> list[dict[str, Union[float, str]]]:
        """
        extract boundaries with speech labels in such a format:
        [
            {"start": sec, "end": sec},
        ]

        args
        ---------
        audio_path: str
            path to .wav file

        returns
        ---------
        list with boundaries
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"{audio_path} is not a valid path")

        boundaries = self._get_boundaries(audio_path, *args, **kwargs)
        return [{"start": start, "end": end} for start, end in boundaries]
