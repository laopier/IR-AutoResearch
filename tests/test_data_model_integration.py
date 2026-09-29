import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare.dataset import IRDropDataset
from train.mavi import MAVI


class TestDataModelIntegration(unittest.TestCase):
    def test_dataset_batch_can_run_through_mavi(self):
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            feature_directory = data_root / "feature"
            label_directory = data_root / "label"
            feature_directory.mkdir()
            label_directory.mkdir()

            filename = (
                "1-RISCY-a-1-c2-u0.8-m1-p1-f0.npy"
            )

            feature = np.random.default_rng(0).random(
                (32, 32, 24),
                dtype=np.float32,
            )
            label = np.random.default_rng(1).random(
                (32, 32, 1),
                dtype=np.float32,
            )

            np.save(feature_directory / filename, feature)
            np.save(label_directory / filename, label)

            manifest_path = data_root / "manifest.csv"
            manifest_path.write_text(
                f"feature/{filename},label/{filename}\n",
                encoding="utf-8",
            )

            dataset = IRDropDataset(
                data_root,
                manifest_path,
            )
            loader = DataLoader(dataset, batch_size=1)

            batch_feature, batch_label, _ = next(iter(loader))

            model = MAVI()
            model.init_weights()
            model.eval()

            with torch.no_grad():
                prediction = model(batch_feature)

            self.assertEqual(
                prediction.shape,
                batch_label.shape,
            )
            self.assertTrue(
                torch.isfinite(prediction).all().item()
            )


if __name__ == "__main__":
    unittest.main()