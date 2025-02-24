from typing import Callable

import numpy as np
from pyannote.core import Annotation, Segment
from pyannote.metrics.base import BaseMetric
from pyannote.metrics.detection import (
    DetectionErrorRate,
    DetectionPrecisionRecallFMeasure,
    DetectionRecall,
    DetectionPrecision,
)
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    recall_score,
    precision_score,
    accuracy_score,
)


def evaluate_clustering(
    vad_labels: np.ndarray, cluster_labels: np.ndarray, align_func: Callable = None
):
    aligned_labels = (
        align_func(cluster_labels, vad_labels)
        if align_func is not None
        else cluster_labels
    )

    accuracy = accuracy_score(vad_labels, aligned_labels)
    precision = precision_score(vad_labels, aligned_labels, zero_division=0)
    recall = recall_score(vad_labels, aligned_labels, zero_division=0)
    f1 = f1_score(vad_labels, aligned_labels, zero_division=0)
    conf_matrix = confusion_matrix(vad_labels, aligned_labels)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "aligned_labels": aligned_labels,
    }


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
