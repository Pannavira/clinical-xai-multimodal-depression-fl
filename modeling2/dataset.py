"""
dataset.py
==========
PyTorch Dataset dan DataLoader untuk Multimodal Depression Detection (E-DAIC)
dengan Z-Score Normalization (StandardScaler) berbasis Training Split (Zero Leakage).

Memuat 3 modalitas:
1. Audio (X_a) : Tensor (512, 80) -> log-Mel spectrogram 80 dimensi, panjang temporal T=512.
2. Visual (X_v): Tensor (512, 72, 3) -> 68 3D Facial Landmarks + 4 Gaze Vectors (72 nodes, 3 coordinates, min-max normalized).
3. Text (X_t)  : Tensor (512, 768) -> Sentence-BERT embeddings 768 dimensi, panjang temporal T=512.
4. Target      : PHQ-8 continuous score (float) dan Depression Label biner (0 / 1).
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


def _pad_or_truncate_sequence(arr: np.ndarray, target_length: int = 512) -> np.ndarray:
    """Pad (dengan nol) atau potong sequence ke panjang tepat target_length."""
    seq_len = arr.shape[0]
    if seq_len == target_length:
        return arr
    elif seq_len > target_length:
        return arr[:target_length]
    else:
        pad_width = [(0, target_length - seq_len)] + [(0, 0)] * (arr.ndim - 1)
        return np.pad(arr, pad_width, mode="constant", constant_values=0.0)


def _adapt_to_sequence_shape(
    arr: np.ndarray,
    target_shape: Tuple[int, ...],
    modality: str,
) -> np.ndarray:
    """
    Mengadaptasi array fitur dari penyimpanan ke bentuk target temporal (T=512).
    Mendukung array 1D (tabular agregat), 2D sequence, maupun 3D tensor.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # 1. Audio target: (512, 80)
    if modality == "audio":
        if arr.shape == (512, 80):
            return arr
        if arr.ndim == 1:
            in_dim = arr.shape[0]
            tile_dim = int(np.ceil(80 / in_dim))
            vec_80 = np.tile(arr, tile_dim)[:80]
            time_steps = np.linspace(0.8, 1.2, 512)[:, None]
            out = np.tile(vec_80, (512, 1)) * time_steps
            return out.astype(np.float32)
        elif arr.ndim == 2:
            T, D = arr.shape
            if D < 80:
                pad_d = np.zeros((T, 80 - D), dtype=np.float32)
                arr = np.concatenate([arr, pad_d], axis=1)
            elif D > 80:
                arr = arr[:, :80]
            return _pad_or_truncate_sequence(arr, target_length=512)

    # 2. Visual target: (512, 72, 3)
    elif modality == "visual":
        if arr.shape == (512, 72, 3):
            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                arr = (arr - min_val) / (max_val - min_val)
            return arr
        if arr.ndim == 1:
            in_dim = arr.shape[0]
            target_flat_dim = 72 * 3  # 216
            tile_dim = int(np.ceil(target_flat_dim / in_dim))
            vec_216 = np.tile(arr, tile_dim)[:target_flat_dim]
            vmin, vmax = vec_216.min(), vec_216.max()
            if vmax > vmin:
                vec_216 = (vec_216 - vmin) / (vmax - vmin)
            time_steps = np.linspace(0.9, 1.1, 512)[:, None]
            seq_216 = np.tile(vec_216, (512, 1)) * time_steps
            return seq_216.reshape(512, 72, 3).astype(np.float32)
        elif arr.ndim == 2:
            T, D = arr.shape
            target_flat_dim = 72 * 3
            if D < target_flat_dim:
                pad_d = np.zeros((T, target_flat_dim - D), dtype=np.float32)
                arr = np.concatenate([arr, pad_d], axis=1)
            elif D > target_flat_dim:
                arr = arr[:, :target_flat_dim]
            arr = _pad_or_truncate_sequence(arr, target_length=512)
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = (arr - vmin) / (vmax - vmin)
            return arr.reshape(512, 72, 3).astype(np.float32)
        elif arr.ndim == 3 and arr.shape[1:] == (72, 3):
            arr = _pad_or_truncate_sequence(arr, target_length=512)
            vmin, vmax = arr.min(), arr.max()
            if vmax > vmin:
                arr = (arr - vmin) / (vmax - vmin)
            return arr.astype(np.float32)

    # 3. Text target: (512, 768)
    elif modality == "text":
        if arr.shape == (512, 768):
            return arr
        if arr.ndim == 1:
            in_dim = arr.shape[0]
            if in_dim < 768:
                pad_d = np.zeros(768 - in_dim, dtype=np.float32)
                vec_768 = np.concatenate([arr, pad_d])
            else:
                vec_768 = arr[:768]
            time_steps = np.linspace(0.95, 1.05, 512)[:, None]
            out = np.tile(vec_768, (512, 1)) * time_steps
            return out.astype(np.float32)
        elif arr.ndim == 2:
            T, D = arr.shape
            if D < 768:
                pad_d = np.zeros((T, 768 - D), dtype=np.float32)
                arr = np.concatenate([arr, pad_d], axis=1)
            elif D > 768:
                arr = arr[:, :768]
            return _pad_or_truncate_sequence(arr, target_length=512)

    return arr


class EDAICMultimodalDataset(Dataset):
    """
    PyTorch Dataset untuk multimodal depression detection (E-DAIC).

    Memuat fitur Audio (512, 80), Visual (512, 72, 3), dan Text (512, 768)
    beserta target PHQ-8 score (continuous) dan label biner.
    """

    def __init__(
        self,
        index_csv: Union[str, Path] = "modeling/01_Input_From_Partition/multimodal_feature_index.csv",
        labels_csv: Union[str, Path] = "data/detailed_lables.csv",
        feature_base_dir: Union[str, Path] = "modeling/01_Input_From_Partition",
        tensor_cache_dir: Optional[Union[str, Path]] = None,
        split: Optional[str] = "train",
        normalize: bool = True,
        synthetic_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.index_csv = Path(index_csv)
        self.labels_csv = Path(labels_csv)
        self.feature_base_dir = Path(feature_base_dir)
        self.tensor_cache_dir = Path(tensor_cache_dir) if tensor_cache_dir else None
        self.split = split.lower() if split else None
        self.normalize = normalize
        self._norm_stats: Dict[str, Dict[str, np.ndarray]] = {}

        if synthetic_samples is not None:
            self._df = self._generate_synthetic_metadata(synthetic_samples)
            self._is_synthetic = True
        else:
            self._df = self._load_and_merge_metadata()
            self._is_synthetic = False

            if self.normalize and self.split == "train":
                self.fit_normalizer()

    def fit_normalizer(self) -> None:
        """Menghitung mean dan std per modalitas dari split train."""
        for mod in ["audio", "visual", "text"]:
            arrays = []
            for _, row in self._df.iterrows():
                arr = self._load_raw_array(row, mod)
                if arr is not None:
                    arrays.append(arr)
            if arrays:
                stk = np.stack(arrays, axis=0)
                mean = stk.mean(axis=0, keepdims=False).astype(np.float32)
                std = stk.std(axis=0, keepdims=False).clip(min=1e-7).astype(np.float32)
                self._norm_stats[mod] = {"mean": mean, "std": std}

    def set_normalizer(self, norm_stats: Dict[str, Dict[str, np.ndarray]]) -> None:
        """Mengatur statistik normalisasi dari training split ke val/test split."""
        self._norm_stats = norm_stats

    def get_normalizer(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Mendapatkan statistik normalisasi."""
        return self._norm_stats

    def _generate_synthetic_metadata(self, num_samples: int) -> pd.DataFrame:
        np.random.seed(42)
        scores = np.random.uniform(0, 24, size=num_samples).astype(np.float32)
        labels = (scores >= 10.0).astype(np.int64)
        return pd.DataFrame({
            "participant_id": list(range(1000, 1000 + num_samples)),
            "split": [self.split or "train"] * num_samples,
            "phq8_score": scores,
            "depression_label": labels,
        })

    def _load_and_merge_metadata(self) -> pd.DataFrame:
        if not self.index_csv.exists():
            raise FileNotFoundError(f"Index CSV tidak ditemukan di: {self.index_csv}")

        df_index = pd.read_csv(self.index_csv)

        if "participant_id" not in df_index.columns and "Participant" in df_index.columns:
            df_index = df_index.rename(columns={"Participant": "participant_id"})
        df_index["participant_id"] = df_index["participant_id"].astype(int)

        if self.labels_csv.exists():
            df_labels = pd.read_csv(self.labels_csv)
            id_col = "Participant" if "Participant" in df_labels.columns else "participant_id"
            sev_col = "Depression_severity" if "Depression_severity" in df_labels.columns else "depression_label"

            labels_subset = df_labels[[id_col, sev_col]].copy()
            labels_subset = labels_subset.rename(columns={id_col: "participant_id", sev_col: "phq8_score"})
            labels_subset["participant_id"] = labels_subset["participant_id"].astype(int)

            df_merged = pd.merge(df_index, labels_subset, on="participant_id", how="left")
            if "phq8_score" not in df_merged.columns or df_merged["phq8_score"].isna().all():
                df_merged["phq8_score"] = df_merged["depression_label"].astype(float)
        else:
            df_merged = df_index.copy()
            df_merged["phq8_score"] = df_merged["depression_label"].astype(float)

        df_merged["depression_label"] = (df_merged["phq8_score"] >= 10.0).astype(int)

        if self.split is not None:
            split_key = "dev" if self.split in ["val", "validation", "dev"] else self.split
            df_merged = df_merged[df_merged["split"].isin([split_key, self.split])].reset_index(drop=True)

        return df_merged

    def __len__(self) -> int:
        return len(self._df)

    def _load_raw_array(self, row: pd.Series, modality: str) -> Optional[np.ndarray]:
        pid = int(row["participant_id"])
        col_name = f"{modality}_feature_path"
        if col_name in row and pd.notna(row[col_name]):
            file_path = self.feature_base_dir / str(row[col_name])
            if file_path.exists():
                return np.load(file_path).astype(np.float32)

        mod_dir_map = {
            "audio": "final_audio_features",
            "visual": "final_visual_features",
            "text": "final_text_features",
        }
        default_path = self.feature_base_dir / mod_dir_map[modality] / f"{pid}.npy"
        if default_path.exists():
            return np.load(default_path).astype(np.float32)
        return None

    def _load_sample_tensor(self, row: pd.Series, modality: str) -> np.ndarray:
        pid = int(row["participant_id"])

        if self.tensor_cache_dir is not None:
            cache_file = self.tensor_cache_dir / modality / f"{pid}.pt"
            if cache_file.exists():
                return torch.load(cache_file, map_location="cpu", weights_only=False).numpy()

            cache_npy = self.tensor_cache_dir / modality / f"{pid}.npy"
            if cache_npy.exists():
                return np.load(cache_npy)

        arr = self._load_raw_array(row, modality)
        if arr is not None:
            # Terapkan Z-score Normalization jika statistik tersedia
            if self.normalize and modality in self._norm_stats:
                mean = self._norm_stats[modality]["mean"]
                std = self._norm_stats[modality]["std"]
                arr = (arr - mean) / std

            return _adapt_to_sequence_shape(arr, (512,), modality)

        if modality == "audio":
            return np.zeros((512, 80), dtype=np.float32)
        elif modality == "visual":
            return np.zeros((512, 72, 3), dtype=np.float32)
        else:
            return np.zeros((512, 768), dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int, float]]:
        row = self._df.iloc[idx]
        pid = int(row["participant_id"])
        phq8 = float(row["phq8_score"])
        label = int(row["depression_label"])

        if self._is_synthetic:
            torch.manual_seed(pid)
            audio = torch.randn(512, 80)
            visual = torch.rand(512, 72, 3)
            text = torch.randn(512, 768)
        else:
            audio_arr = self._load_sample_tensor(row, "audio")
            visual_arr = self._load_sample_tensor(row, "visual")
            text_arr = self._load_sample_tensor(row, "text")

            audio = torch.from_numpy(audio_arr).float()
            visual = torch.from_numpy(visual_arr).float()
            text = torch.from_numpy(text_arr).float()

        return {
            "audio": audio,          # (512, 80)
            "visual": visual,        # (512, 72, 3)
            "text": text,            # (512, 768)
            "phq8_score": torch.tensor(phq8, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "participant_id": pid,
        }


def build_dataloaders(
    index_csv: Union[str, Path] = "modeling/01_Input_From_Partition/multimodal_feature_index.csv",
    labels_csv: Union[str, Path] = "data/detailed_lables.csv",
    feature_base_dir: Union[str, Path] = "modeling/01_Input_From_Partition",
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membangun PyTorch DataLoader untuk Train, Validation, dan Test split E-DAIC.
    Normalisasi dihitung dari split train dan diterapkan ke val dan test tanpa leakage.
    """
    train_dataset = EDAICMultimodalDataset(
        index_csv=index_csv,
        labels_csv=labels_csv,
        feature_base_dir=feature_base_dir,
        split="train",
        normalize=normalize,
    )
    val_dataset = EDAICMultimodalDataset(
        index_csv=index_csv,
        labels_csv=labels_csv,
        feature_base_dir=feature_base_dir,
        split="dev",
        normalize=normalize,
    )
    test_dataset = EDAICMultimodalDataset(
        index_csv=index_csv,
        labels_csv=labels_csv,
        feature_base_dir=feature_base_dir,
        split="test",
        normalize=normalize,
    )

    # Pass normalizer dari train ke val dan test
    if normalize:
        train_norm_stats = train_dataset.get_normalizer()
        val_dataset.set_normalizer(train_norm_stats)
        test_dataset.set_normalizer(train_norm_stats)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    logger.info(
        f"[EDAIC DataLoader] Loaded splits -> Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)} (Normalized: {normalize})"
    )
    return train_loader, val_loader, test_loader
