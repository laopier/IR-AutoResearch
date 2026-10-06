"""Compare actual MAVI continuous training with a stop/restart, in fresh processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

from b0.checkpoint import load_checkpoint
from b0.run import ROOT, sha256, write_json


def equal(left, right):
    if isinstance(left, torch.Tensor):
        if not torch.equal(left, right):
            raise AssertionError('Checkpoint tensors differ')
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right, (left, right)


def verify(args):
    args.output.mkdir(parents=True, exist_ok=False)
    outputs = {name: args.output / name for name in ['continuous', 'first', 'resumed']}
    common = [sys.executable, '-m', 'b0.run', '--data-root', str(args.data_root.resolve()),
              '--train-manifest', str(ROOT / 'prepare/manifests/smoke_train.csv'),
              '--validation-manifest', str(ROOT / 'prepare/manifests/smoke_validation.csv'),
              '--scope', 'mini', '--batch-size', '2', '--seed', '0', '--device', args.device,
              '--lr-horizon-steps', '200000', '--num-workers', '0', '--cpu-threads', '2',
              '--checkpoint-every', '2', '--log-every', '1', '--check-values']
    for name, steps in [('continuous', 4), ('first', 2), ('resumed', 4)]:
        command = common + ['--output', str(outputs[name].resolve()), '--steps', str(steps)]
        if name == 'resumed':
            command += ['--resume', str((outputs['first'] / 'checkpoint_final.pt').resolve())]
        print('VERIFY:', name, flush=True)
        subprocess.run(command, cwd=ROOT, check=True,
                       env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2'))
    continuous = load_checkpoint(outputs['continuous'] / 'checkpoint_final.pt')
    resumed = load_checkpoint(outputs['resumed'] / 'checkpoint_final.pt')
    for key in ['state_dict', 'optimizer_state_dict', 'rng', 'sampling']:
        equal(continuous[key], resumed[key])
    logs = lambda name: (outputs[name] / 'training.jsonl').read_text().splitlines()
    assert logs('continuous') == logs('first') + logs('resumed')
    reports = {name: json.loads((path / 'result.json').read_text()) for name, path in outputs.items()}
    assert reports['continuous']['validation'] == reports['resumed']['validation']
    report = {'status': 'passed', 'model': 'MAVI', 'device': args.device,
              'comparison': '4 continuous updates versus 2 + resumed to cumulative 4',
              'model_optimizer_rng_sampling_bitwise_equal': True,
              'sample_ids_loss_lr_logs_equal': True, 'validation_equal': True,
              'environment': reports['continuous']['environment'],
              'validation': reports['continuous']['validation'],
              'result_sha256': {name: sha256(path / 'result.json') for name, path in outputs.items()}}
    write_json(args.output / 'verification.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    verify(parser.parse_args())
