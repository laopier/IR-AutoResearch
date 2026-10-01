import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from b0.run import check_data, run
from prepare.dataset import IRDropDataset


class TestPreflight(unittest.TestCase):
    def make_data(self, root):
        (root / 'feature').mkdir(); (root / 'label').mkdir()
        for name in ['a', 'b', 'c']:
            np.save(root / f'feature/{name}.npy', np.zeros((16, 16, 24), dtype=np.float32))
            np.save(root / f'label/{name}.npy', np.zeros((16, 16, 1), dtype=np.float32))
        train = root / 'train.csv'; val = root / 'val.csv'
        train.write_text('feature/a.npy,label/a.npy\nfeature/b.npy,label/b.npy\n')
        val.write_text('feature/c.npy,label/c.npy\n')
        return train, val

    def test_rejects_shared_label_even_with_distinct_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, val = self.make_data(root)
            val.write_text('feature/c.npy,label/a.npy\n')
            with self.assertRaisesRegex(ValueError, 'share a resolved data file'):
                check_data(IRDropDataset(root, train), IRDropDataset(root, val), 2, False)

    def test_rejects_duplicate_aliases_after_path_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, val = self.make_data(root)
            train.write_text('feature/a.npy,label/a.npy\nfeature/./a.npy,label/b.npy\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate resolved'):
                check_data(IRDropDataset(root, train), IRDropDataset(root, val), 2, False)

    def test_rejects_values_that_overflow_float32(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, val = self.make_data(root)
            np.save(root / 'feature/a.npy', np.full((16, 16, 24), 1e100))
            with self.assertRaisesRegex(ValueError, 'Nonfinite float32'):
                check_data(IRDropDataset(root, train), IRDropDataset(root, val), 2, True)

    def test_rejects_no_full_training_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, val = self.make_data(root)
            with self.assertRaisesRegex(ValueError, 'smaller than batch_size'):
                check_data(IRDropDataset(root, train), IRDropDataset(root, val), 3, False)

    def test_existing_run_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); train, val = self.make_data(root)
            output = root / 'existing'; output.mkdir()
            sentinel = output / 'result.json'; sentinel.write_text('keep existing result')
            args = SimpleNamespace(steps=1, batch_size=2, num_workers=0, lr_horizon_steps=200000,
                                   checkpoint_every=10000, log_every=100, data_root=root,
                                   train_manifest=train, validation_manifest=val,
                                   check_values=False, preflight_only=False, device='cpu', output=output)
            with self.assertRaises(FileExistsError):
                run(args)
            self.assertEqual(sentinel.read_text(), 'keep existing result')


if __name__ == '__main__':
    unittest.main()
