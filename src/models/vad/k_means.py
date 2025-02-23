import pickle
from typing import Callable

import librosa
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.vad.base import BaseVAD
from utils.metrics import evaluate_clustering


class KMeansVAD(BaseVAD):
    _model: KMeans = None

    def __init__(self, chunk_dur: float, features_list: list[str] = None):
        if features_list is None:
            self.features_list = [
                "mfcc",
                "spectral_centroid",
                "zcr",
                "spectral_flux",
                "rms",
            ]
        else:
            self.features_list = features_list

        self._scaler = StandardScaler()
        self.chunk_dur = chunk_dur

    def fit(
        self, audio_data: list[dict], align_func: Callable = None, verbose: bool = True
    ):
        all_features, silero_labels = [], []
        for item in audio_data:
            waveform, sr = librosa.load(item["file"], sr=None)
            features = self.extract_features(waveform, sr)
            print(f"FEAT SHAPE: {features.shape}")
            all_features.append(features)
            silero_labels.append(
                self._get_target_labels(
                    silero_segments=item["silero_segments"],
                    num_frames=features.shape[0],
                    sr=sr,
                )
            )

        max_frames = max(features.shape[0] for features in all_features)
        all_features = [
            np.pad(f, ((0, max_frames - f.shape[0]), (0, 0)), mode="constant")  # Pad
            if f.shape[0] < max_frames
            else f[:max_frames]  # Truncate
            for f in all_features
        ]

        features_scaled = self._scaler.fit_transform(
            np.array(all_features).reshape(1, -1)
        )
        self._model = KMeans(n_clusters=2, random_state=42)
        kmeans_labels = self._model.fit_predict(features_scaled)
        results = evaluate_clustering(
            np.array(silero_labels), kmeans_labels, align_func=align_func
        )

        if verbose:
            print("Evaluation Results for KMeans Clustering:")
            print(f"Accuracy: {results['accuracy']:.4f}")
            print(f"Precision: {results['precision']:.4f}")
            print(f"Recall: {results['recall']:.4f}")
            print(f"F1 Score: {results['f1_score']:.4f}")

    def extract_features(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        features_list = []
        if "mfcc" in self.features_list:
            features_list.append(librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13))

        if "spectral_centroid" in self.features_list:
            features_list.append(librosa.feature.spectral_centroid(y=waveform, sr=sr))

        if "zcr" in self.features_list:
            features_list.append(librosa.feature.zero_crossing_rate(y=waveform))

        if "spectral_flux" in self.features_list:
            spectral_flux = librosa.onset.onset_strength(y=waveform, sr=sr)
            features_list.append(spectral_flux.reshape(1, -1))

        if "rms" in self.features_list:
            features_list.append(librosa.feature.rms(y=waveform))

        num_frames = min(f.shape[1] for f in features_list)
        features = np.vstack([f[:, :num_frames] for f in features_list]).T

        return features

    def _get_boundaries(
        self, audio_path: str, *args, **kwargs
    ) -> list[tuple[float, float]]:
        if not self._model:
            raise ValueError("KMeans not fitted or loaded yet.")

        waveform, sr = librosa.load(audio_path, sr=None)
        features = self.extract_features(waveform, sr)
        scaled = self._scaler.transform(features)
        # cluster = self._model.predict(scaled)
        # todo:

    def _get_target_labels(
        self, silero_segments: list[dict], sr: int, num_frames: int
    ) -> list[int]:
        labels = []
        for i in range(num_frames):
            chunk_start = int(i * self.chunk_dur * sr)
            chunk_end = chunk_start + int(self.chunk_dur * sr)
            is_speech = any(
                s["start"] <= chunk_start and s["end"] >= chunk_end
                for s in silero_segments
            )
            labels.append(1 if is_speech else 0)

        return labels

    def save(self):
        with open("kmeans_vad.pkl", "wb") as f:
            pickle.dump(self._model, f)

        with open("kmeans_scaler.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

    def load(self, kmeans_path: str, scaler_path: str, features_list: list[str]):
        with open(kmeans_path, "rb") as f:
            self._model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            self._scaler = pickle.load(f)

        self.features_list = features_list
