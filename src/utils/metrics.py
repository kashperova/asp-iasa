from pyannote.core import Annotation, Segment
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
    DetectionRecall,
    DetectionPrecision,
)


def annotate_vad_results(results: list[dict]) -> Annotation:
    annotation = Annotation()
    for speech in results:
        annotation[Segment(speech["start"], speech["end"])] = "SPEECH"

    return annotation


class DetectionMetric:
    def __init__(self, metric: BaseMetric, name: str):
        self.__metric = metric
        self.__name = name

    @property
    def name(self):
        return self.__name

    def __call__(self, targets: list[dict], predictions: list[dict]):
        return self.__metric(
            reference=annotate_vad_results(targets),
            hypothesis=annotate_vad_results(predictions),
            detailed=False,
        )

    @classmethod
    def create(cls, metric_name: str):
        if metric_name == "error_rate":
            return cls(DetectionErrorRate(), metric_name)
        elif metric_name == "precision":
            return cls(DetectionPrecision(), metric_name)
        elif metric_name == "recall":
            return cls(DetectionRecall(), metric_name)
        elif metric_name == "f1":
            return cls(DetectionPrecisionRecallFMeasure(), metric_name)
