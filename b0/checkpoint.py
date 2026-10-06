"""Training state at a completed update; load only checkpoints you produced."""
from __future__ import annotations

import os
import random
from pathlib import Path
import numpy as np
import torch
from train.experiment import build_model, build_optimizer, train_batch
from torch.utils.data import DataLoader
import random
FORMAT_VERSION = 1


def capture_rng() -> dict:
    name, keys, position, gaussian, cached = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': {'name': name, 'keys': torch.from_numpy(keys.astype(np.int64)),
                      'position': position, 'gaussian': gaussian, 'cached': cached},
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state: dict) -> None:
    random.setstate(state['python'])
    np_state = state['numpy']
    np.random.set_state((np_state['name'], np_state['keys'].numpy().astype(np.uint32),
                         np_state['position'], np_state['gaussian'], np_state['cached']))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same visible CUDA device count')
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(path: Path, state: dict) -> None:
    # A failed write must not leave a partial checkpoint with the final name.
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_checkpoint(path: Path) -> dict:
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('resume_format_version') != FORMAT_VERSION:
        raise ValueError('Not a resumable B0 checkpoint; old weights-only files cannot resume')
    return state


def restore_iterator(loader, generator, sampling):
    """Replay the current permutation without updating the model (workers=0)."""
    consumed = sampling['batches_consumed']
    if not 0 < consumed <= len(loader):
        raise ValueError('Invalid checkpoint batch cursor')
    generator.set_state(sampling['epoch_generator_state'])
    iterator = iter(loader)
    for _ in range(consumed):
        next(iterator)
    if not torch.equal(generator.get_state(), sampling['generator_state']):
        raise ValueError('DataLoader replay differs from saved sampler state')
    return iterator


    
    
    