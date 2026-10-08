"""Real tiny CPU native resume -> shared cache -> independent worker equivalence."""
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from sa0 import controller as c
from sa0.research import baseline_cache
from sa0.research import test_worker


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
class BaselineCacheTests(unittest.TestCase):
    def setUp(self):
        test_worker.WorkerTests.setUp(self)
        shutil.copy2(self.root / "data/x0.npy", self.root / "data/val_x.npy")
        shutil.copy2(self.root / "data/y0.npy", self.root / "data/val_y.npy")
        (self.root / "val.csv").write_text("val_x.npy,val_y.npy\n")
        self.config["val_manifest"] = str(self.root / "val.csv")

    def test_native_resume_matches_fresh_worker_and_rejects_tampering(self):
        import torch
        from b0.run import run
        shutil.copytree(c.ROOT / "b0", self.workspace / "b0", ignore=shutil.ignore_patterns("__pycache__"))
        spec = importlib.util.spec_from_file_location("cache_tiny", self.workspace / "train/mavi.py")
        tiny = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tiny)
        def build():
            model = tiny.MAVI()
            model.init_weights()
            return model
        self.config["steps"] = 4
        base = {**self.config, "baseline_initial_lr": .0002, "baseline_cache_dir": str(self.root / "cache")}
        paths = [self.root / "native2", self.root / "native4"]
        with patch("b0.run.ROOT", self.workspace), patch("b0.run.build_model", side_effect=build), \
             patch("train.experiment.build_model", side_effect=build), patch.object(c, "ROOT", self.workspace):
            for i, steps in enumerate((2, 4)):
                args = SimpleNamespace(data_root=Path(base["data_root"]), train_manifest=Path(base["train_manifest"]),
                    validation_manifest=Path(base["val_manifest"]), output=paths[i], scope="development_baseline",
                    steps=steps, lr_horizon_steps=200000, batch_size=2, seed=0, device="cpu", num_workers=0,
                    cpu_threads=2, checkpoint_every=1, log_every=10, preflight_only=False, check_values=True,
                    resume=paths[0] / "checkpoint_final.pt" if i else None)
                run(args)
            env = baseline_cache.runtime_environment("cpu", 2)
            baseline_cache.publish(c, base, 0, paths, self.root / "cache/seed_0", env)
            folder, receipt, imported = baseline_cache.read(c, base, self.workspace, 0, env)
            with self.assertRaises(ValueError):
                baseline_cache.read(c, {**base, "steps": 5}, self.workspace, 0, env)
            with self.assertRaises(ValueError):
                baseline_cache.read(c, base, self.workspace, 0, {**env, "torch_num_threads": 1})
        fresh = test_worker.WorkerTests.execute(self)
        self.assertEqual(imported["final_metrics"], fresh["final_metrics"])
        direct = torch.load(self.root / "out/checkpoint.pt", weights_only=False, map_location="cpu")
        cached = torch.load(folder / "artifacts/checkpoint.pt", weights_only=False, map_location="cpu")
        self.assertEqual(direct["model"].keys(), cached["model"].keys())
        self.assertTrue(all(torch.equal(direct["model"][k], cached["model"][k]) for k in direct["model"]))
        self.assertEqual(len((folder / "artifacts/training.jsonl").read_text().splitlines()), 4)
        (folder / "artifacts/training.jsonl").write_text("changed")
        with self.assertRaises(ValueError):
            baseline_cache.read(c, base, self.workspace, 0, env)


if __name__ == "__main__": unittest.main()
