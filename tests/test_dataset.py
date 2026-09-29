import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare.dataset import IRDropDataset


class TestIRDropDataset(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temporary_directory.name)

        self.feature_directory = self.data_root / "feature"
        self.label_directory = self.data_root / "label"
        self.feature_directory.mkdir()
        self.label_directory.mkdir()

        self.filename = (
            "1-RISCY-a-1-c2-u0.8-m1-p1-f0.npy"
        )
        self.feature_path = (
            self.feature_directory / self.filename
        )
        self.label_path = (
            self.label_directory / self.filename
        )
        self.manifest_path = self.data_root / "manifest.csv"

        self.manifest_path.write_text(
            f"feature/{self.filename},"
            f"label/{self.filename}\n",
            encoding="utf-8",
        )

        self.save_valid_sample()

    def tearDown(self):
        self.temporary_directory.cleanup()

    def save_valid_sample(self):
        feature = np.zeros((8, 8, 24), dtype=np.float64)
        label = np.zeros((8, 8, 1), dtype=np.float64)

        np.save(self.feature_path, feature)
        np.save(self.label_path, label)

    def test_valid_sample_and_batch_contract(self):
        dataset = IRDropDataset(
            self.data_root,
            self.manifest_path,
        )

        feature, label, sample_id = dataset[0]

        self.assertEqual(feature.shape, (1, 24, 8, 8))
        self.assertEqual(label.shape, (8, 8))
        self.assertEqual(feature.dtype, torch.float32)
        self.assertEqual(label.dtype, torch.float32)
        self.assertEqual(
            sample_id,
            "1-RISCY-a-1-c2-u0.8-m1-p1-f0",
        )

        loader = DataLoader(dataset, batch_size=1)
        batch_feature, batch_label, batch_ids = next(iter(loader))

        self.assertEqual(
            batch_feature.shape,
            (1, 1, 24, 8, 8),
        )
        self.assertEqual(batch_label.shape, (1, 8, 8))
        self.assertEqual(batch_ids[0], sample_id)

    def test_rejects_nonfinite_feature(self):
        feature = np.zeros((8, 8, 24), dtype=np.float64)
        feature[0, 0, 0] = np.nan
        np.save(self.feature_path, feature)

        dataset = IRDropDataset(
            self.data_root,
            self.manifest_path,
        )

        with self.assertRaisesRegex(
            ValueError,
            "contains NaN or Inf",
        ):
            dataset[0]

    def test_rejects_wrong_channel_count(self):
        feature = np.zeros((8, 8, 23), dtype=np.float64)
        np.save(self.feature_path, feature)

        dataset = IRDropDataset(
            self.data_root,
            self.manifest_path,
        )

        with self.assertRaisesRegex(
            ValueError,
            "Expected 24 feature channels",
        ):
            dataset[0]

    def test_rejects_spatial_shape_mismatch(self):
        label = np.zeros((7, 8, 1), dtype=np.float64)
        np.save(self.label_path, label)

        dataset = IRDropDataset(
            self.data_root,
            self.manifest_path,
        )

        with self.assertRaisesRegex(
            ValueError,
            "does not match label shape",
        ):
            dataset[0]

    def test_rejects_path_outside_data_root(self):
        self.manifest_path.write_text(
            f"../outside.npy,label/{self.filename}\n",
            encoding="utf-8",
        )

        dataset = IRDropDataset(
            self.data_root,
            self.manifest_path,
        )

        with self.assertRaisesRegex(
            ValueError,
            "escapes data root",
        ):
            dataset[0]


if __name__ == "__main__":
    unittest.main()