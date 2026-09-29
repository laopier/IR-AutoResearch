import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from program.runner import run_experiment


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(
            torch.tensor(0.5)
        )

    def forward(self, feature):
        return feature[:, 0, 0] * self.scale


def build_tiny_model():
    return TinyModel()


def build_tiny_optimizer(model):
    return torch.optim.SGD(
        model.parameters(),
        lr=0.1,
    )


def build_tiny_scheduler(optimizer, total_steps):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )


def train_tiny_batch(
    model,
    optimizer,
    feature,
    target,
):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    prediction = model(feature)
    loss = torch.nn.functional.l1_loss(
        prediction,
        target,
    )
    loss.backward()
    optimizer.step()

    return float(loss.detach().item())


class TestRunner(unittest.TestCase):
    def test_run_experiment_writes_auditable_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_directory = root / "feature"
            label_directory = root / "label"
            feature_directory.mkdir()
            label_directory.mkdir()

            train_name = "train.npy"
            validation_name = "validation.npy"

            np.save(
                feature_directory / train_name,
                np.ones((12, 12, 24), dtype=np.float32),
            )
            np.save(
                label_directory / train_name,
                np.zeros((12, 12, 1), dtype=np.float32),
            )
            np.save(
                feature_directory / validation_name,
                np.ones((12, 12, 24), dtype=np.float32),
            )
            np.save(
                label_directory / validation_name,
                np.zeros((12, 12, 1), dtype=np.float32),
            )

            train_manifest = root / "train.csv"
            validation_manifest = root / "validation.csv"
            output_path = root / "result.json"

            train_manifest.write_text(
                f"feature/{train_name},label/{train_name}\n",
                encoding="utf-8",
            )
            validation_manifest.write_text(
                f"feature/{validation_name},"
                f"label/{validation_name}\n",
                encoding="utf-8",
            )

            with (
                patch(
                    "program.runner.build_model",
                    side_effect=build_tiny_model,
                ),
                patch(
                    "program.runner.build_optimizer",
                    side_effect=build_tiny_optimizer,
                ),
                patch(
                    "program.runner.build_scheduler",
                    side_effect=build_tiny_scheduler,
                ),
                patch(
                    "program.runner.train_batch",
                    side_effect=train_tiny_batch,
                ),
            ):
                result = run_experiment(
                    data_root=root,
                    train_manifest=train_manifest,
                    validation_manifest=validation_manifest,
                    output_path=output_path,
                    run_kind="smoke",
                    description="runner unit test",
                    total_steps=2,
                    batch_size=1,
                    seed=0,
                    device_name="cpu",
                )

            saved_result = json.loads(
                output_path.read_text(encoding="utf-8")
            )

            self.assertEqual(result, saved_result)
            self.assertEqual(result["run_kind"], "smoke")
            self.assertEqual(result["budget"]["total_steps"], 2)
            self.assertEqual(result["data"]["train_samples"], 1)
            self.assertEqual(
                result["data"]["validation_samples"],
                1,
            )
            self.assertIn("commit", result["git"])
            self.assertIn("mae_float", result["validation"])
            self.assertIn("elapsed_seconds", result["resources"])

    def test_rejects_invalid_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            with self.assertRaisesRegex(
                ValueError,
                "total_steps must be positive",
            ):
                run_experiment(
                    data_root=root,
                    train_manifest=root / "train.csv",
                    validation_manifest=root / "validation.csv",
                    output_path=root / "result.json",
                    run_kind="smoke",
                    description="invalid budget",
                    total_steps=0,
                    batch_size=1,
                    seed=0,
                    device_name="cpu",
                )


if __name__ == "__main__":
    unittest.main()