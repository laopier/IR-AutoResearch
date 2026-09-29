import unittest

import torch

from train.experiment import (
    build_model,
    build_optimizer,
    build_scheduler,
    compute_loss,
    train_batch,
)


class TestTrainExperiment(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = build_model()
        self.optimizer = build_optimizer(self.model)

    def test_builders_return_expected_interfaces(self):
        self.assertIsInstance(
            self.model,
            torch.nn.Module,
        )
        self.assertIsInstance(
            self.optimizer,
            torch.optim.Optimizer,
        )

        scheduler = build_scheduler(
            self.optimizer,
            total_steps=10,
        )

        self.assertTrue(hasattr(scheduler, "step"))

    def test_compute_loss_returns_finite_scalar(self):
        prediction = torch.zeros(1, 16, 16)
        target = torch.ones(1, 16, 16)

        loss = compute_loss(prediction, target)

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss).item())

    def test_compute_loss_rejects_shape_mismatch(self):
        prediction = torch.zeros(1, 16, 16)
        target = torch.zeros(1, 15, 16)

        with self.assertRaisesRegex(
            ValueError,
            "does not match target shape",
        ):
            compute_loss(prediction, target)

    def test_train_batch_updates_parameters(self):
        feature = torch.rand(1, 1, 24, 32, 32)
        target = torch.rand(1, 32, 32)

        parameters_before = [
            parameter.detach().clone()
            for parameter in self.model.parameters()
        ]

        loss = train_batch(
            self.model,
            self.optimizer,
            feature,
            target,
        )

        parameters_after = list(self.model.parameters())

        changed = any(
            not torch.equal(before, after.detach())
            for before, after
            in zip(parameters_before, parameters_after)
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())
        self.assertTrue(
            changed,
            "train_batch did not update any model parameter",
        )


if __name__ == "__main__":
    unittest.main()