import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from b0 import pilot_run


class PilotRunTests(unittest.TestCase):
    def test_wrapper_preserves_recipe_and_targets_500_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "b0").mkdir()
            config = {"data_root": "data", "dataset_report": "receipt", "train_manifest": "train.csv",
                      "val_manifest": "val.csv", "output": "new_output", "steps": 500, "seed": 0,
                      "batch_size": 2, "lr_horizon_steps": 200000, "checkpoint_every": 100}
            (root / "b0/pilot_config.json").write_text(json.dumps(config))
            with patch.object(pilot_run, "ROOT", root), patch.object(pilot_run, "verify"), \
                 patch.object(pilot_run.subprocess, "run") as run:
                pilot_run.main()
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("--steps") + 1], "500")
            self.assertEqual(command[command.index("--batch-size") + 1], "2")
            self.assertEqual(command[command.index("--lr-horizon-steps") + 1], "200000")
            self.assertNotIn("--resume", command)


if __name__ == "__main__": unittest.main()
