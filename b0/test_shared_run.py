import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from b0.shared_run import command, latest_checkpoint


class SharedRunTests(unittest.TestCase):
    def test_resume_keeps_cumulative_target_and_seed(self):
        config = {"data_root": "data", "train_manifest": "train.csv", "val_manifest": "val.csv",
                  "steps": 1000, "batch_size": 2, "lr_horizon_steps": 200000,
                  "checkpoint_every": 100, "device": "cuda"}
        args = command(config, 2, Path("new"), Path("old/checkpoint_step600.pt"))
        self.assertEqual(args[args.index("--steps") + 1], "1000")
        self.assertEqual(args[args.index("--seed") + 1], "2")
        self.assertEqual(args[args.index("--resume") + 1], "old/checkpoint_step600.pt")

    def test_final_checkpoint_without_validation_uses_earlier_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("checkpoint_step900.pt", "checkpoint_final.pt"):
                (root / name).touch()
            with patch("b0.checkpoint.load_checkpoint", side_effect=lambda p: {"step": 1000 if "final" in p.name else 900}):
                step, path = latest_checkpoint([root], 1000)
            self.assertEqual(step, 900)
            (root / "result.json").write_text('{"status":')
            with patch("b0.checkpoint.load_checkpoint", side_effect=lambda p: {"step": 1000 if "final" in p.name else 900}):
                self.assertEqual(latest_checkpoint([root], 1000)[0], 900)


if __name__ == "__main__": unittest.main()
