"""Tiny CPU-only worker checks; real loss/optimizer/scheduler, no GPU training."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from sa0 import controller as c


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"), "CPU torch/numpy/opencv required")
class WorkerTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.temp = tempfile.TemporaryDirectory(prefix="ir_cpu_worker_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        c.freeze_sources(self.workspace)
        shutil.copy2(c.ROOT / "sa0/research/worker.py", self.workspace / "worker.py")
        (self.workspace / "train/feature_transform.py").write_text("def transform(feature):\n    return feature\n")
        (self.workspace / "train/mavi.py").write_text('''import torch
class MAVI(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
    def init_weights(self):
        torch.nn.init.constant_(self.conv.weight, .1)
        torch.nn.init.constant_(self.conv.bias, .1)
    def forward(self, feature):
        return self.conv(feature.mean(2))[:,0]
''')
        data = self.root / "data"
        data.mkdir()
        rng = np.random.default_rng(0)
        rows = []
        for i in range(2):
            np.save(data / f"x{i}.npy", rng.random((16,16,24), dtype=np.float32))
            np.save(data / f"y{i}.npy", rng.random((16,16), dtype=np.float32))
            rows.append(f"x{i}.npy,y{i}.npy")
        (self.root / "manifest.csv").write_text("\n".join(rows) + "\n")
        self.config = {"workspace": str(self.workspace), "data_root": str(data),
                       "train_manifest": str(self.root / "manifest.csv"), "val_manifest": str(self.root / "manifest.csv"),
                       "out_dir": str(self.root / "out"), "seed": 0, "steps": 2,
                       "batch_size": 2, "val_batch_size": 1, "lr_horizon_steps": 200000,
                       "recipe": {"initial_lr": .0002}, "device": "cpu", "custom_scheduler": False}

    def execute(self):
        path = self.root / "config.json"
        path.write_text(json.dumps(self.config))
        env = {**os.environ, "IR_SA0_CONFIG": str(path), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        result = subprocess.run([sys.executable, str(self.workspace / "worker.py")], cwd=self.workspace,
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        return c.read_json(self.root / "out/result.json")

    def test_b0_recipe_remains_l1_adamw_official_cosine(self):
        result = self.execute()
        self.assertEqual(result["optimizer"], "AdamW")
        self.assertEqual(result["initialized_from"], "fresh")
        first = json.loads((self.root / "out/training.jsonl").read_text().splitlines()[0])
        self.assertEqual(first["lr"], .0002)
        self.assertEqual(result["steps"], 2)
        self.assertEqual([item["step"] for item in result["validation_probes"]], [2])
        self.assertTrue((self.root / "out/validation_probes.jsonl").is_file())

    def test_loss_optimizer_scheduler_overrides_and_frozen_inference(self):
        self.config["recipe"].update(loss={"name": "smooth_l1", "beta": .01, "scale": 100},
                                     optimizer={"name": "sgd", "momentum": .9}, scheduler={"name": "constant"})
        (self.workspace / "train/feature_transform.py").write_text("def transform(feature):\n    return feature * .9\n")
        result = self.execute()
        self.assertEqual(result["optimizer"], "SGD")
        self.assertEqual(result["recipe"]["loss"]["name"], "smooth_l1")
        frozen = self.root / "final"
        frozen.mkdir()
        shutil.copytree(self.workspace, frozen / "source")
        shutil.copy2(self.root / "out/checkpoint.pt", frozen / "checkpoint.pt")
        (frozen / "FROZEN.json").write_text(json.dumps({"source_hashes": c.source_hashes(frozen / "source"),
            "checkpoint_sha256": c.sha256(frozen / "checkpoint.pt")}))
        result = subprocess.run([sys.executable, "-m", "sa0.research.inference", "--frozen", str(frozen),
                                 "--feature", str(self.root / "data/x0.npy"), "--output", str(self.root / "prediction.npy"),
                                 "--device", "cpu"], cwd=c.ROOT, capture_output=True, text=True, encoding="utf-8", timeout=90)
        self.assertEqual(result.returncode, 0, result.stderr)
        import numpy as np
        self.assertEqual(np.load(self.root / "prediction.npy").shape, (16,16))

    def test_custom_scheduler_function_is_actually_used(self):
        file = self.workspace / "train/experiment.py"
        file.write_text(file.read_text(encoding="utf-8") + '\ndef research_lr(step, horizon, initial_lr):\n    return 0.0003\n', encoding="utf-8")
        self.execute()
        logs = [json.loads(line) for line in (self.root / "out/training.jsonl").read_text().splitlines()]
        self.assertTrue(all(x["lr"] == .0003 for x in logs))

    def test_native_candidate_scheduler_honors_recipe_initial_lr(self):
        file = self.workspace / "train/experiment.py"
        file.write_text(file.read_text(encoding="utf-8").replace("eta_min=1e-7", "eta_min=2e-7"), encoding="utf-8")
        self.config["custom_scheduler"] = True
        self.config["recipe"]["initial_lr"] = .0008
        self.execute()
        first = json.loads((self.root / "out/training.jsonl").read_text().splitlines()[0])
        self.assertEqual(first["lr"], .0008)


if __name__ == "__main__":
    unittest.main()
