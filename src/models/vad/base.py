import os
from typing import Union

from utils.singleton import Singleton


class BaseVAD(metaclass=Singleton):
    def _get_boundaries(
        self, audio_path: str, *args, **kwargs
    ) -> list[tuple[float, float]]:
        raise NotImplementedError

    def get_boundaries(
        self, audio_path: str, round_factor: int = None, *args, **kwargs
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

        round_factor: int
            factor to round preds

        returns
        ---------
        list with boundaries
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"{audio_path} is not a valid path")

        boundaries = self._get_boundaries(audio_path, *args, **kwargs)
        return [{"start": start, "end": end} for start, end in boundaries]

    @staticmethod
    def _merge_boundaries(
        close_th: int, boundaries: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """
        Merge final speech boundaries

        args
        ---------
            close_th: int
                threshold in milliseconds.
                If the distance between boundaries is smaller than this value,
                the segments will be merged.

            boundaries: list[tuple[float, float]]
                extracted speech boundaries
        returns
        ---------
            list with boundaries
        """
        if len(boundaries) == 0:
            return []

        close_th_sec = close_th / 1000.0
        merged_boundaries = [boundaries[0]]

        for start, end in boundaries[1:]:
            last_start, last_end = merged_boundaries[-1]

            if start - last_end <= close_th_sec:
                merged_boundaries[-1] = (last_start, max(last_end, end))
            else:
                merged_boundaries.append((start, end))

        return merged_boundaries
