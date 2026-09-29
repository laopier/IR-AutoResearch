import unittest

import torch

from prepare.evaluator import (
    evaluate_batch,
    mae_float,
    nrms_official,
    ssim_official,
)


class TestEvaluator(unittest.TestCase):
    def test_perfect_prediction(self):
        target = torch.linspace(
            0,
            1,
            16 * 16,
        ).reshape(1, 16, 16)

        metrics = evaluate_batch(target.clone(), target)

        self.assertEqual(metrics["mae_float"], 0.0)
        self.assertEqual(metrics["nrms_official"], 0.0)
        self.assertAlmostEqual(
            metrics["ssim_official"],
            1.0,
            places=12,
        )

    def test_matches_official_circuitnet_reference(self):
        target = torch.linspace(
            0,
            1,
            16 * 16,
        ).reshape(1, 16, 16)
        prediction = (target * 0.8 + 0.05).clamp(0, 1)

        self.assertAlmostEqual(
            mae_float(prediction, target),
            0.06264704465866089,
            places=7,
        )
        self.assertAlmostEqual(
            nrms_official(prediction, target),
            0.0764293346492368,
            places=12,
        )
        self.assertAlmostEqual(
            ssim_official(prediction, target),
            0.9717933877995407,
            places=12,
        )

    def test_accepts_supported_tensor_shapes(self):
        target_2d = torch.linspace(
            0,
            1,
            16 * 16,
        ).reshape(16, 16)
        prediction_4d = target_2d.reshape(1, 1, 16, 16)

        self.assertEqual(
            mae_float(prediction_4d, target_2d),
            0.0,
        )

    def test_rejects_shape_mismatch(self):
        prediction = torch.zeros(1, 16, 16)
        target = torch.zeros(1, 15, 16)

        with self.assertRaisesRegex(
            ValueError,
            "does not match target shape",
        ):
            evaluate_batch(prediction, target)

    def test_rejects_nonfinite_values(self):
        prediction = torch.zeros(1, 16, 16)
        target = torch.zeros(1, 16, 16)
        prediction[0, 0, 0] = float("nan")

        with self.assertRaisesRegex(
            ValueError,
            "contains NaN or Inf",
        ):
            evaluate_batch(prediction, target)

    def test_constant_target_nrms_policy(self):
        target = torch.zeros(1, 16, 16)
        prediction = torch.ones(1, 16, 16)

        self.assertEqual(
            nrms_official(prediction, target),
            0.05,
        )


if __name__ == "__main__":
    unittest.main()