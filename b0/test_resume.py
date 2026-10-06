"""Behavioral tests: replay across epoch boundaries, RNG and Adam state."""
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from b0.checkpoint import load_checkpoint
from b0.run import run


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x):
        x = x.mean(2)
        if self.training:
            # Exercise all three CPU RNG streams, not just sample shuffling.
            x = self.dropout(x) * (0.9 + 0.1 * random.random() + 0.1 * np.random.rand())
        return self.conv(x)[:, 0]


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right), 'Tensor states differ'
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right, (left, right)


class TestResume(unittest.TestCase):
    def make_data(self, root):
        (root / 'feature').mkdir()
        (root / 'label').mkdir()
        rng = np.random.default_rng(18)
        for i in range(6):
            np.save(root / f'feature/{i}.npy', rng.random((16, 16, 24), dtype=np.float32))
            np.save(root / f'label/{i}.npy', rng.random((16, 16, 1), dtype=np.float32))
        (root / 'train.csv').write_text(''.join(f'feature/{i}.npy,label/{i}.npy\n' for i in range(5)))
        (root / 'val.csv').write_text('feature/5.npy,label/5.npy\n')

    def args(self, root, name, steps, resume=None):
        return SimpleNamespace(data_root=root, train_manifest=root / 'train.csv',
                               validation_manifest=root / 'val.csv', output=root / name,
                               scope='mini', steps=steps, lr_horizon_steps=200000,
                               batch_size=2, seed=0, device='cpu', num_workers=0,
                               cpu_threads=1, checkpoint_every=2, log_every=100,
                               preflight_only=False, check_values=True, resume=resume)

    def test_continuous_equals_resumed_inside_epoch_and_at_boundary(self):
        with tempfile.TemporaryDirectory() as tmp, patch('b0.run.build_model', TinyModel):
            root = Path(tmp)
            self.make_data(root)
            run(self.args(root, 'continuous', 7))
            continuous = load_checkpoint(root / 'continuous/checkpoint_final.pt')
            all_logs = (root / 'continuous/training.jsonl').read_text().splitlines()
            for cut in [2, 3]:
                with self.subTest(cut=cut):
                    run(self.args(root, f'first{cut}', cut))
                    checkpoint = root / f'first{cut}/checkpoint_final.pt'
                    run(self.args(root, f'resumed{cut}', 7, checkpoint))
                    resumed = load_checkpoint(root / f'resumed{cut}/checkpoint_final.pt')
                    for key in ['state_dict', 'optimizer_state_dict', 'rng', 'sampling']:
                        assert_state_equal(continuous[key], resumed[key])
                    combined = (root / f'first{cut}/training.jsonl').read_text().splitlines()
                    combined += (root / f'resumed{cut}/training.jsonl').read_text().splitlines()
                    self.assertEqual(all_logs, combined)
                    report = json.loads((root / f'resumed{cut}/result.json').read_text())
                    self.assertEqual(report['updates_this_run'], 7 - cut)
            # Also exercise a periodic checkpoint, not only the final checkpoint.
            run(self.args(root, 'periodic_resume', 7, root / 'continuous/checkpoint_step4.pt'))
            periodic = load_checkpoint(root / 'periodic_resume/checkpoint_final.pt')
            assert_state_equal(continuous['state_dict'], periodic['state_dict'])
            assert_state_equal(continuous['optimizer_state_dict'], periodic['optimizer_state_dict'])

    def test_rejects_changed_data_and_training_contract(self):
        with tempfile.TemporaryDirectory() as tmp, patch('b0.run.build_model', TinyModel):
            root = Path(tmp)
            self.make_data(root)
            run(self.args(root, 'first', 2))
            checkpoint = root / 'first/checkpoint_final.pt'
            args = self.args(root, 'wrong_batch', 3, checkpoint)
            args.batch_size = 1
            with self.assertRaisesRegex(ValueError, 'Resume contract changed: batch_size'):
                run(args)
            np.save(root / 'feature/0.npy', np.zeros((16, 16, 24), dtype=np.float32))
            with self.assertRaisesRegex(ValueError, 'data_files'):
                run(self.args(root, 'changed_data', 3, checkpoint))

    def test_rejects_old_weights_and_nonincreasing_target(self):
        with tempfile.TemporaryDirectory() as tmp, patch('b0.run.build_model', TinyModel):
            root = Path(tmp)
            self.make_data(root)
            old = root / 'old.pt'
            torch.save({'state_dict': {}, 'step': 100}, old)
            with self.assertRaisesRegex(ValueError, 'old weights-only'):
                run(self.args(root, 'old_resume', 101, old))
            run(self.args(root, 'first', 2))
            with self.assertRaisesRegex(ValueError, 'cumulative target'):
                run(self.args(root, 'same_target', 2, root / 'first/checkpoint_final.pt'))


if __name__ == '__main__':
    unittest.main()
