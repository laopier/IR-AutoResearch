"""Controller-launched train/evaluation worker; candidate code lives separately."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
from torch.utils.data import DataLoader


def install_audit(roots):
    allowed = [Path(p).resolve() for p in roots]
    # Python audit checks are a guardrail, not an OS security sandbox.
    def audit(event, args):
        if event.startswith('socket.') or event in ['subprocess.Popen', 'os.system', 'os.fork', 'os.posix_spawn']:
            raise PermissionError(f'Offline worker blocked {event}')
        if event == 'open' and args and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(args[0])).resolve()
            if not any(path == root or root in path.parents for root in allowed):
                raise PermissionError(f'Worker file access outside allowed roots: {path}')
    sys.addaudithook(audit)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--steps', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.workspace.resolve()))
    # Protected data/evaluation modules are checked against the frozen snapshot by the controller.
    from prepare.dataset import IRDropDataset
    from prepare.evaluator import evaluate_batch
    # Initialize torch's optimizer machinery before installing file-access guards.
    torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))])
    install_audit([args.workspace, args.data_root, args.manifest.parent, args.output,
                   sys.prefix, sys.base_prefix, tempfile.gettempdir(), '/usr', '/dev', '/proc', '/sys',
                   Path(__file__).resolve().parent])
    from train.experiment import build_model, build_optimizer, build_scheduler, train_batch
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('This mini rehearsal requires CUDA')
    dataset = IRDropDataset(args.data_root, args.manifest)
    if not len(dataset):
        raise ValueError('Empty dataset')
    model = build_model().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if args.mode == 'train':
        optimizer = build_optimizer(model)
        scheduler = build_scheduler(optimizer, total_steps=200000)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(args.seed), num_workers=0)
        iterator = iter(loader)
        with (args.output / 'training.jsonl').open('x', encoding='utf-8') as log:
            for step in range(args.steps):
                try:
                    x, y, ids = next(iterator)
                except StopIteration:
                    iterator = iter(loader); x, y, ids = next(iterator)
                lr = optimizer.param_groups[0]['lr']
                loss = train_batch(model, optimizer, x.to(device), y.to(device))
                scheduler.step()
                log.write(json.dumps({'step': step+1, 'loss': loss, 'lr': lr,
                                      'sample_ids': list(ids)}, allow_nan=False)+'\n'); log.flush()
                print(f'train step {step+1}/{args.steps}: loss={loss:.6f}', flush=True)
        torch.save({'state_dict': model.state_dict()}, args.output / 'checkpoint.pt')
        payload = {'mode': 'train', 'steps_completed': args.steps,
                   'parameter_count': sum(p.numel() for p in model.parameters())}
    else:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        model.eval()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        metrics_sum = {}; count = 0
        with torch.no_grad():
            for x, y, _ in loader:
                metrics = evaluate_batch(model(x.to(device)), y.to(device))
                for key, value in metrics.items():
                    metrics_sum[key] = metrics_sum.get(key, 0.0) + value
                count += 1
        payload = {'mode': 'evaluate', 'metrics': {k:v/count for k,v in metrics_sum.items()},
                   'samples': count, 'parameter_count': sum(p.numel() for p in model.parameters())}
    torch.cuda.synchronize(device)
    payload.update(elapsed_seconds=time.perf_counter()-started,
                   peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
                   environment={'python': sys.version.split()[0], 'torch': str(torch.__version__),
                                'numpy': np.__version__, 'cuda': torch.version.cuda,
                                'gpu': torch.cuda.get_device_name(device)},
                   python_audit_guard_enabled=True)
    with (args.output / f'{args.mode}.json').open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
