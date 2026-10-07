import csv
import gzip
import io
import importlib.util
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from prepare import pilot_stream as p


class PilotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.csv"
        self.test = self.root / "test.csv"
        self.test.write_text("feature/10000-zero-riscy-b-3-c2-u0.85-m1-p6-f1.npy,label/10000-zero-riscy-b-3-c2-u0.85-m1-p6-f1.npy\n")
        rows = []
        i = 0
        for design in ("RISCY-a", "RISCY-b", "RISCY-FPU-a", "RISCY-FPU-b"):
            for layout in range(60):
                for mesh in (1, 2):
                    i += 1
                    name = f"{i}-{design}-{layout+1}-c{(2,5,20)[layout%3]}-u{(.7,.75,.8,.85,.9)[layout%5]}-m{layout%3+1}-p{mesh}-f0.npy"
                    rows.append(("feature/" + name, "label/" + name))
        with self.source.open("w", newline="") as f:
            csv.writer(f).writerows(rows)

    def test_selection_balanced_disjoint_reproducible(self):
        a = p.selection(self.source, self.test)
        self.assertEqual(a, p.selection(self.source, self.test))
        for split, total, each in (("train", 128, 32), ("validation", 32, 8)):
            self.assertEqual(len(a["samples"][split]), total)
            for design in ("RISCY-a", "RISCY-b", "RISCY-FPU-a", "RISCY-FPU-b"):
                self.assertEqual(sum(x["design"] == design for x in a["samples"][split]), each)
        self.assertFalse({x["layout_group"] for x in a["samples"]["train"]} & {x["layout_group"] for x in a["samples"]["validation"]})
        self.assertNotEqual(a["samples"], p.selection(self.source, self.test, 1)["samples"])

    def test_test_overlap_rejected(self):
        self.test.write_text(self.source.read_text().splitlines()[0] + "\n")
        with self.assertRaises(ValueError):
            p.selection(self.source, self.test)

    def test_split_gzip_stream_and_selective_extraction(self):
        plan = p.selection(self.source, self.test)
        name = plan["samples"]["train"][0]["name"]
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode="w:gz") as t:
            for path, content in (("power_t/unselected.npy", b"discard"), ("power_t/" + name, b"selected")):
                info = tarfile.TarInfo(path)
                info.size = len(content)
                t.addfile(info, io.BytesIO(content))
        blob = packed.getvalue()
        chunks = [blob[:len(blob)//2], blob[len(blob)//2:]]
        class Reader:
            def __init__(self, item, cache, progress):
                self.stream = io.BytesIO(chunks[item[1]])
            def read(self, n):
                return self.stream.read(n)
            def close(self):
                pass
        raw, cache = self.root / "raw", self.root / "cache"
        cache.mkdir()
        with patch.object(p, "CachedFile", Reader), patch.dict(p.ARCHIVES, {"power_t": [("a",0,1),("b",1,1)]}):
            receipt = p.extract_component("power_t", {name}, raw, cache, lambda *a, **k: None)
        self.assertEqual((raw / "power_t" / name).read_bytes(), b"selected")
        self.assertFalse((raw / "power_t/unselected.npy").exists())
        self.assertIn(name, receipt)

    def test_official_double_gzip_and_extensionless_raw_names(self):
        plan = p.selection(self.source, self.test)
        name = plan["samples"]["validation"][0]["name"]
        packed = io.BytesIO()
        with tarfile.open(fileobj=packed, mode="w:gz") as t:
            info = tarfile.TarInfo("power_i/" + name.removesuffix(".npy"))
            info.size = 5
            t.addfile(info, io.BytesIO(b"array"))
        blob = gzip.compress(packed.getvalue())
        class Reader:
            def __init__(self, *args): self.stream = io.BytesIO(blob)
            def read(self, n): return self.stream.read(n)
            def close(self): pass
        raw, cache = self.root / "raw", self.root / "cache"
        cache.mkdir()
        with patch.object(p, "CachedFile", Reader):
            p.extract_component("power_i", {name}, raw, cache, lambda *a, **k: None)
        self.assertEqual((raw / "power_i" / name).read_bytes(), b"array")
        self.assertEqual(json.loads((raw / "power_i/ARCHIVE_FORMAT.json").read_text())["gzip_layers"], 2)

    @unittest.skipUnless(importlib.util.find_spec("numpy") and importlib.util.find_spec("cv2"), "numpy/opencv required")
    def test_official_preprocessing_order_shape_and_label_scale(self):
        import numpy as np
        name = "synthetic-test-only.npy"
        raw = self.root / "raw"
        grid = np.arange(64, dtype=np.float64).reshape(8,8)
        for component in p.COMPONENTS:
            (raw / component).mkdir(parents=True)
            array = np.stack([grid + i for i in range(20)]) if component == "power_t" else grid
            if component == "IR_drop": array = np.full((8,8), 10., dtype=np.float64)
            np.save(raw / component / name, array)
        plan = {"samples": {"train": [{"name": name}], "validation": []}}
        p.preprocess(plan, raw, self.root / "data")
        feature = np.load(self.root / "data/feature" / name)
        label = np.load(self.root / "data/label" / name)
        self.assertEqual(feature.shape, (256,256,24))
        self.assertEqual(label.shape, (256,256,1))
        self.assertEqual(float(feature.min()), 0.)
        self.assertEqual(float(feature.max()), 1.)
        self.assertTrue(np.allclose(label, 7/(np.log10(50)+6)))
        self.assertTrue(p.audit(plan, self.root / "data")["ready"])
        p.preprocess(plan, raw, self.root / "data")  # matching existing arrays aren't overwritten


if __name__ == "__main__":
    unittest.main()
