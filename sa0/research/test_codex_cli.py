import os
from pathlib import Path
import tempfile
import unittest

from sa0.research.codex_cli import resolve


class CodexCliTests(unittest.TestCase):
    def test_explicit_path_is_preserved(self):
        self.assertEqual(resolve({"codex_path": "mock"}), "mock")

    def test_auto_selects_newest_desktop_binary(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / "old" / "codex.exe"
            new = root / "new" / "codex.exe"
            old.parent.mkdir()
            new.parent.mkdir()
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            os.utime(old, ns=(1, 1))
            os.utime(new, ns=(2, 2))
            self.assertEqual(resolve({"codex_path": "auto", "codex_bin_root": str(root)}), str(new))

    def test_auto_fails_clearly_when_binary_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                resolve({"codex_path": "auto", "codex_bin_root": temporary})


if __name__ == "__main__":
    unittest.main()
