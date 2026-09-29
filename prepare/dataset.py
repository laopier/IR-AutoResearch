from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from prepare.split_manifest import read_manifest


EXPECTED_FEATURE_CHANNELS = 24


class IRDropDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
    ):
        self.data_root = Path(data_root).resolve()
        self.manifest_path = Path(manifest_path).resolve()

        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"Data root does not exist: {self.data_root}"
            )

        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest does not exist: {self.manifest_path}"
            )

        self.rows = read_manifest(self.manifest_path)

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve_data_path(self, relative_path: str) -> Path:
        path = (self.data_root / relative_path).resolve()

        try:
            path.relative_to(self.data_root)
        except ValueError as error:
            raise ValueError(
                f"Data path escapes data root: {relative_path}"
            ) from error

        if not path.is_file():
            raise FileNotFoundError(
                f"Data file does not exist: {path}"
            )

        return path

    def _load_array(self, path: Path) -> np.ndarray:
        array = np.load(path, allow_pickle=False)

        if not np.issubdtype(array.dtype, np.number):
            raise TypeError(
                f"Expected a numeric array at {path}, "
                f"got {array.dtype}"
            )

        if not np.isfinite(array).all():
            raise ValueError(
                f"Array contains NaN or Inf: {path}"
            )

        return array

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        feature_relative, label_relative = self.rows[index]

        feature_path = self._resolve_data_path(feature_relative)
        label_path = self._resolve_data_path(label_relative)

        feature = self._load_array(feature_path)
        label = self._load_array(label_path)

        if feature.ndim != 3:
            raise ValueError(
                f"Expected feature [H, W, C], got {feature.shape}"
            )

        if feature.shape[-1] != EXPECTED_FEATURE_CHANNELS:
            raise ValueError(
                f"Expected {EXPECTED_FEATURE_CHANNELS} feature "
                f"channels, got {feature.shape[-1]}"
            )

        if label.ndim == 3 and label.shape[-1] == 1:
            label = label[..., 0]
        elif label.ndim != 2:
            raise ValueError(
                f"Expected label [H, W, 1] or [H, W], "
                f"got {label.shape}"
            )

        if feature.shape[:2] != label.shape:
            raise ValueError(
                f"Feature spatial shape {feature.shape[:2]} "
                f"does not match label shape {label.shape}"
            )

        feature = np.ascontiguousarray(
            feature.transpose(2, 0, 1)[None, ...],
            dtype=np.float32,
        )
        label = np.ascontiguousarray(
            label,
            dtype=np.float32,
        )

        sample_id = Path(feature_relative).stem

        return (
            torch.from_numpy(feature),
            torch.from_numpy(label),
            sample_id,
        )