import json
import pickle

import librosa
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from models.vad.base import BaseVAD
from utils.wav_utils import apply_filters, apply_wiener


class KMeansVAD(BaseVAD):
    def __init__(
        self,
        features_list: list[str],
        chunk_dur: float = 0.025,
        hop_dur: float = 0.01,
        n_mfcc: int = 13,
        context_size: int = 5,
        model: KMeans = None,
        scaler: StandardScaler = None,
        output_csv_path: str = None,
        df_data: pd.DataFrame = None,
    ):
        self.features_list = features_list
        self.chunk_dur = chunk_dur
        self.hop_dur = hop_dur
        self.n_mfcc = n_mfcc
        self.context_size = context_size

        self._mfcc_features = [f for f in features_list if "mfcc" in f]
        self._cls_features = [f for f in features_list if f not in self._mfcc_features]

        self._model = model
        self._scaler = scaler or StandardScaler()
        self._output_csv_path = output_csv_path
        self._df_data = df_data

    def fit(self, audio_files: list[str], csv_path: str):
        all_data = []
        for item in audio_files:
            df = self._process_audio(item)
            all_data.append(df)
            print(f"Processing {item}")

        combined_df = pd.concat(all_data, ignore_index=True)

        feature_columns = [c for c in combined_df.columns if c not in ["file_path"]]
        features = combined_df[feature_columns].values
        X = self._scaler.fit_transform(features)

        self._model = KMeans(n_clusters=2, random_state=42)
        labels = self._model.fit_predict(X)
        cluster_energies = [X[labels == i, -1].mean() for i in range(2)]
        if cluster_energies[0] > cluster_energies[1]:
            labels = 1 - labels

        combined_df["vad_pred"] = labels
        combined_df.to_csv(csv_path, index=False)
        self._output_csv_path = csv_path
        self._df_data = combined_df

    def _get_mfcc(
        self, waveform: np.ndarray, sr: int, n_fft: int, hop_length: int
    ) -> np.ndarray:
        return librosa.feature.mfcc(
            y=waveform, sr=sr, n_mfcc=self.n_mfcc, n_fft=n_fft, hop_length=hop_length
        )

    def extract_features(
        self, waveform: np.ndarray, sr: int, chunk_start: int, chunk_end: int
    ) -> np.ndarray:
        features_list = []
        current_chunk = waveform[chunk_start:chunk_end]

        context_samples = int(self.context_size * (chunk_end - chunk_start))
        # might need a better context calculation (pad with something like zeros or mean values)
        left_start = max(0, chunk_start - context_samples)
        right_end = min(len(waveform), chunk_end + context_samples)

        left_context = waveform[left_start:chunk_start]
        right_context = waveform[chunk_end:right_end]

        full_context = np.concatenate([left_context, current_chunk, right_context])

        n_fft = min(512, len(current_chunk))
        hop_length = n_fft // 4
        mfccs_with_context = None

        if "mfccs" in self.features_list:
            mfccs = self._get_mfcc(current_chunk, sr, n_fft, hop_length)
            if mfccs.shape[1] == 0:
                mfccs = np.zeros((13, 1))
            features_list.append(np.mean(mfccs, axis=1))

        if "mfcc_delta" in self.features_list:
            mfccs_with_context = self._get_mfcc(full_context, sr, n_fft, hop_length)
            mfcc_delta = librosa.feature.delta(mfccs_with_context)
            if mfcc_delta.shape[1] == 0:
                mfcc_delta = np.zeros((13, 1))
            features_list.append(np.mean(mfcc_delta, axis=1))

        if "mfcc_delta2" in self.features_list:
            if mfccs_with_context is None:
                mfccs_with_context = self._get_mfcc(full_context, sr, n_fft, hop_length)

            mfcc_delta2 = librosa.feature.delta(mfccs_with_context, order=2)
            if mfcc_delta2.shape[1] == 0:
                mfcc_delta2 = np.zeros((13, 1))
            features_list.append(np.mean(mfcc_delta2, axis=1))

        if len(self._cls_features) > 0:
            # classical features (we need to add pitch or something related!!!)
            features_list.append([])
            spec = np.abs(librosa.stft(current_chunk, n_fft=n_fft, hop_length=hop_length))

            if "spectral_centroid" in self.features_list:
                spectral_centroid = 0 if spec.shape[1] == 0 else librosa.feature.spectral_centroid(S=spec, sr=sr).mean()
                features_list[-1].append(spectral_centroid)

            if "spectral_flux" in self.features_list:
                spectral_flux = 0 if spec.shape[1] == 0 else (np.mean(np.diff(spec, axis=1) ** 2) if spec.shape[1] > 1 else 0)
                features_list[-1].append(spectral_flux)

            if "zcr" in self.features_list:
                zcr = librosa.feature.zero_crossing_rate(current_chunk, frame_length=n_fft, hop_length=hop_length).mean()
                features_list[-1].append(zcr)

            if "ste" in self.features_list:
                ste = np.mean(current_chunk**2)
                features_list[-1].append(ste)

            if "pitch" in self.features_list:
                f0 = librosa.yin(current_chunk, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7')).mean()
                features_list[-1].append(f0)

        return np.concatenate(features_list)

    def _get_boundaries(
        self, audio_path: str, round_factor: int = None, *args, **kwargs
    ) -> list[tuple[float, float]]:
        if not self._model:
            raise ValueError("KMeans not fitted or loaded yet.")

        file_df = self._df_data[self._df_data["file_path"] == audio_path]

        predictions = file_df["vad_pred"].values
        boundaries = []
        changes = np.diff(predictions.astype(int))
        change_points = np.where(changes != 0)[0] + 1

        if predictions[0] == 1:
            change_points = np.concatenate(([0], change_points))
        if predictions[-1] == 1:
            change_points = np.concatenate((change_points, [len(predictions)]))

        for i in range(0, len(change_points), 2):
            if i + 1 < len(change_points):
                start = change_points[i] * self.hop_dur  # todo: ?? chunk dur
                end = change_points[i + 1] * self.hop_dur
                boundaries.append((start, end))

        boundaries = self._merge_boundaries(boundaries=boundaries, close_th=100)
        if round_factor is not None:
            boundaries = [(round(s, round_factor), round(e, round_factor)) for s, e in boundaries]
        return boundaries

    def _process_audio(self, audio_path: str) -> pd.DataFrame:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = apply_filters(waveform=waveform, sr=sr)
        waveform = librosa.util.normalize(waveform)
        waveform = apply_wiener(waveform)

        chunk_samples = int(self.chunk_dur * sr)
        hop_samples = int(self.hop_dur * sr)
        num_chunks = (len(waveform) - chunk_samples) // hop_samples + 1

        feature_dim = self.n_mfcc * len(self._mfcc_features) + len(self._cls_features)
        all_features = np.zeros((num_chunks, feature_dim))

        for i in range(num_chunks):
            chunk_start = i * hop_samples
            chunk_end = chunk_start + chunk_samples

            all_features[i] = self.extract_features(
                waveform, sr, chunk_start, chunk_end
            )

        feature_names = []
        if "mfccs" in self._mfcc_features:
            feature_names += [f"mfcc_{i + 1}" for i in range(self.n_mfcc)]
        if "mfcc_delta" in self._mfcc_features:
            feature_names += [f"mfcc_delta_{i + 1}" for i in range(self.n_mfcc)]
        if "mfcc_delta2" in self._mfcc_features:
            feature_names += [f"mfcc_delta2_{i + 1}" for i in range(self.n_mfcc)]
        feature_names += self._cls_features

        df = pd.DataFrame(all_features, columns=feature_names)
        df["file_path"] = audio_path

        return df

    def save(self, version: str = ""):
        with open(f"kmeans_vad_{version}.pkl", "wb") as f:
            pickle.dump(self._model, f)

        with open(f"kmeans_scaler_{version}.pkl", "wb") as f:
            pickle.dump(self._scaler, f)

        with open(f"kmeans_cfg_{version}.json", "w") as f:
            json.dump(
                {
                    "chunk_dur": self.chunk_dur,
                    "hop_dur": self.hop_dur,
                    "features_list": self.features_list,
                    "n_mfcc": self.n_mfcc,
                    "context_size": self.context_size,
                    "output_csv_path": self._output_csv_path,
                },
                f,
            )

    @classmethod
    def load(cls, kmeans_path: str, scaler_path: str, cfg_path: str) -> "KMeansVAD":
        with open(kmeans_path, "rb") as f:
            model = pickle.load(f)

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        with open(cfg_path, "rb") as f:
            data = json.load(f)
            features_list = data["features_list"]
            chunk_dur = data["chunk_dur"]
            hop_dur = data["hop_dur"]
            n_mfcc = data["n_mfcc"]
            context_size = data["context_size"]
            output_csv_path = data["output_csv_path"]

        df_data = pd.read_csv(output_csv_path)

        return cls(
            chunk_dur=chunk_dur,
            hop_dur=hop_dur,
            features_list=features_list,
            n_mfcc=n_mfcc,
            context_size=context_size,
            model=model,
            scaler=scaler,
            output_csv_path=output_csv_path,
            df_data=df_data,
        )
