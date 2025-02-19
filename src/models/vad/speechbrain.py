from speechbrain.inference import VAD

from models.vad.base import BaseVAD
from utils.wav_utils import get_wav_metadata


class SpeechbrainVAD(BaseVAD):
    def __init__(
        self, source: str = "speechbrain/vad-crdnn-libriparty", save_dir: str = "models"
    ):
        super().__init__()
        self.__model = VAD.from_hparams(source=source, savedir=save_dir)

    def _get_boundaries(
        self,
        audio_path: str,
        large_chunk_size: float = None,
        small_chunk_size: float = 0.5,
        overlap_small_chunk: bool = False,
        activation_th: float = 0.75,
        deactivation_th: float = 0.6,
        apply_energy_vad: bool = True,
        merge_close_th: float = 0.5,
        remove_short_len_th: float = 0.35,
    ) -> list[tuple[float, float]]:
        """
        N.B. VAD from Speechbrain toolkit works only with 16kHz sr & mono channel

        args
        ---------
            large_chunk_size: float
                size (in seconds) of the large chunks that are read sequentially
                from the audio file.

            small_chunk_size: float
                size (in seconds) of the small chunks extracted from the large ones.
                The audio signal is processed in parallel within the small chunks.
                N.B. large_chunk_size/small_chunk_size must be an integer.

            overlap_small_chunk: bool
                if true, creates overlapped small chunks. The probabilities of the
                overlapped chunks are combined using hamming windows.

            activation_th:  float
                threshold for starting a speech segment.

            deactivation_th: float
                threshold for ending a speech segment.

            apply_energy_vad: bool
                if true, applies energy-based VAD within the detected speech segments.
                The VAD neural net often creates longer segments and tends to merge segments
                that are close with each other; so it can be useful for more accurate detection.

            merge_close_th: float
                threshold to merge segments. If the distance between boundaries
                is smaller than this value, the segments will be merged.

            remove_short_len_th: float
                threshold to remove short segments. If the length of the segment
                is smaller than this value, the segments will be merged.

        returns
        ---------
        list with boundaries
        """

        metadata = get_wav_metadata(audio_path)
        if metadata["sr"] != 16000:
            raise ValueError(
                f"Audio should be resampled to 16KHz, currently {metadata['sr']}"
            )

        prob_chunks = self.__model.get_speech_prob_file(
            audio_path,
            large_chunk_size=large_chunk_size or int(metadata["duration"]),
            small_chunk_size=small_chunk_size,
        )
        prob_th = self.__model.apply_threshold(
            prob_chunks, activation_th=activation_th, deactivation_th=deactivation_th
        ).float()
        boundaries = self.__model.get_boundaries(prob_th)

        if apply_energy_vad:
            # remove speech segments shorter than 10 ms before applying VAD energy
            boundaries = self.__model.remove_short_segments(boundaries, len_th=0.01)

            boundaries = self.__model.energy_VAD(audio_path, boundaries)
            boundaries = self.__model.merge_close_segments(
                boundaries, close_th=merge_close_th
            )
            boundaries = self.__model.remove_short_segments(
                boundaries, len_th=remove_short_len_th
            )

        return [
            (round(item[0].item(), 4), round(item[1].item(), 4)) for item in boundaries
        ]
