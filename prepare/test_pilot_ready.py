import csv
import json
from pathlib import Path
import tempfile
import unittest

from prepare.pilot_ready import verify, sha


class ReadyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        (self.data / "feature").mkdir(parents=True)
        (self.data / "label").mkdir()
        samples, files = {}, {}
        self.spec = {}
        for split, count in (("train", 128), ("validation", 32)):
            samples[split] = []
            for i in range(count):
                name = f"{split}_{i}.npy"
                row = {"feature": "feature/" + name, "label": "label/" + name}
                samples[split].append(row)
                for value in row.values():
                    path = self.data / value
                    path.write_bytes(value.encode())  # hash-gate fixture, not real training data
                    files[value] = {"sha256": sha(path)}
            manifest = self.root / (split + ".csv")
            with manifest.open("w", newline="") as f:
                csv.writer(f).writerows((r["feature"], r["label"]) for r in samples[split])
            self.spec["train_manifest" if split == "train" else "val_manifest"] = str(manifest)
        plan = self.root / "SELECTION.json"
        plan.write_text(json.dumps({"samples": samples}))
        self.report = self.root / "DATASET.json"
        self.report.write_text(json.dumps({"ready": True, "train_samples": 128, "validation_samples": 32,
            "selection_sha256": sha(plan), "files": files}))

    def test_ready_and_complete_hashes(self):
        self.assertTrue(verify(self.data, self.report, [self.spec])["ready"])

    def test_changed_array_rejected(self):
        (self.data / "feature/train_0.npy").write_bytes(b"changed")
        with self.assertRaises(ValueError): verify(self.data, self.report, [self.spec])

    def test_changed_manifest_rejected(self):
        Path(self.spec["val_manifest"]).write_text("feature/train_0.npy,label/train_0.npy\n")
        with self.assertRaises(ValueError): verify(self.data, self.report, [self.spec])

    def test_missing_report_rejected(self):
        with self.assertRaises(RuntimeError): verify(self.data, self.root / "missing.json", [self.spec])

    def test_incomplete_receipt_rejected(self):
        report = json.loads(self.report.read_text())
        report["files"].pop("label/train_0.npy")
        self.report.write_text(json.dumps(report))
        with self.assertRaises(ValueError): verify(self.data, self.report, [self.spec])


if __name__ == "__main__": unittest.main()
