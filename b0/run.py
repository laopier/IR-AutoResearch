from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
import traceback
import zipfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare.dataset import IRDropDataset
from program.runner import set_seed, validate
from train.experiment import build_model, build_optimizer, train_batch

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def official_lr(step: int, horizon: int) -> float:
    # Official train.py sets the LR BEFORE update, with zero-based iter_num.
    if not 0 <= step < horizon:
        raise ValueError('step must be in [0, lr_horizon_steps)')
    return 1e-7 + 0.5 * (2e-4 - 1e-7) * (math.cos(math.pi * step / horizon) + 1)


def check_data(train: IRDropDataset, validation: IRDropDataset,
               batch_size: int, check_values: bool) -> dict:
    if len(train) < batch_size:
        raise ValueError('Training set is smaller than batch_size with drop_last=True')
    if not len(validation):
        raise ValueError('Validation set is empty')
    paths = {}
    report = {}
    for name, dataset in [('train', train), ('validation', validation)]:
        features, labels = [], []
        shape_counts = {}
        for i, (feature, label) in enumerate(dataset.rows):
            f, l = dataset._resolve_data_path(feature), dataset._resolve_data_path(label)
            features.append(f)
            labels.append(l)
            for path, is_feature in [(f, True), (l, False)]:
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if not np.issubdtype(array.dtype, np.number):
                    raise TypeError(f'Non-numeric array: {path}')
                if is_feature:
                    if array.ndim != 3 or array.shape[-1] != 24:
                        raise ValueError(f'Invalid feature shape {array.shape}: {path}')
                    spatial = array.shape[:2]
                elif array.shape not in [spatial, (*spatial, 1)]:
                    raise ValueError(f'Invalid label shape {array.shape}: {path}')
            key = str(spatial)
            shape_counts[key] = shape_counts.get(key, 0) + 1
            if min(spatial) < 16 or any(size % 8 for size in spatial):
                raise ValueError(f'B0 requires H,W >=16 and divisible by 8: {f}')
            if check_values:
                x, y, _ = dataset[i]
                if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                    raise ValueError(f'Nonfinite float32 conversion: {f}')
        if len(features) != len(set(features)) or len(labels) != len(set(labels)):
            raise ValueError(f'Duplicate resolved feature/label paths in {name}')
        if len(shape_counts) != 1:
            raise ValueError(f'Mixed shapes need an explicit batching protocol: {name}')
        paths[name] = (set(features), set(labels))
        report[name] = {'samples': len(dataset), 'spatial_shapes': shape_counts,
                        'manifest_sha256': sha256(dataset.manifest_path)}
    if (paths['train'][0] | paths['train'][1]) & (paths['validation'][0] | paths['validation'][1]):
        raise ValueError('Train and validation share a resolved data file')
    report['finite_values_checked_in_preflight'] = check_values
    report['design_isolation_verified'] = False
    return report


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def capture_source(output: Path) -> dict:
    files = sorted(p for folder in ['prepare', 'train', 'program', 'b0']
                   for p in (ROOT / folder).rglob('*')
                   if p.is_file() and p.suffix in ['.py', '.csv', '.json', '.md'])
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p) for p in files}
    with zipfile.ZipFile(output / 'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    def git(*args):
        return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args],
                                       cwd=ROOT, text=True).strip()
    try:
        state = {'commit': git('rev-parse', 'HEAD'), 'status': git('status', '--porcelain')}
        state['dirty'] = bool(state['status'])
    except (OSError, subprocess.CalledProcessError):
        state = {'commit': None, 'dirty': None, 'status': 'Git metadata unavailable; use source hashes'}
    return {'git': state, 'file_sha256': hashes,
            'snapshot_sha256': sha256(output / 'source_snapshot.zip')}


def run(args) -> None:
    if args.steps <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError('Invalid training budget or worker count')
    if args.steps > args.lr_horizon_steps:
        raise ValueError('steps cannot exceed lr_horizon_steps')
    if args.checkpoint_every <= 0 or args.log_every <= 0:
        raise ValueError('Checkpoint and log intervals must be positive')
    train = IRDropDataset(args.data_root, args.train_manifest)
    validation = IRDropDataset(args.data_root, args.validation_manifest)
    print('Preflight: checking all manifest files and array headers', flush=True)
    data_report = check_data(train, validation, args.batch_size, args.check_values)
    if args.preflight_only:
        print(json.dumps(data_report, indent=2), flush=True)
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    # Never overwrite a previous run, even when its output directory is empty.
    args.output.mkdir(parents=True, exist_ok=False)
    source = capture_source(args.output)
    config = {key: str(value.resolve()) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config['recipe'] = {'model': 'MAVI(in_channels=1,out_channels=4,bilinear=False)',
                        'optimizer': 'AdamW', 'lr': 2e-4, 'weight_decay': 1e-2,
                        'betas': [0.9, 0.999], 'loss': '100 * mean L1',
                        'augmentation': None, 'precision': 'float32',
                        'drop_last': True, 'lr_set_before_update': True,
                        'validation_batch_size': 1}
    write_json(args.output / 'config.json', config)
    write_json(args.output / 'source.json', source)
    write_json(args.output / 'data_report.json', data_report)
    (args.output / 'train_manifest.csv').write_bytes(args.train_manifest.read_bytes())
    (args.output / 'validation_manifest.csv').write_bytes(args.validation_manifest.read_bytes())
    device = torch.device(args.device)
    set_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=args.num_workers, generator=generator,
                        pin_memory=device.type == 'cuda')
    val_loader = DataLoader(validation, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=device.type == 'cuda')
    model = build_model().to(device)
    optimizer = build_optimizer(model)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    environment = {'python': platform.python_version(), 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'numpy': np.__version__, 'opencv': cv2.__version__,
                   'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                   'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   'torch_threads': torch.get_num_threads()}
    write_json(args.output / 'environment.json', environment)
    started = time.perf_counter()
    try:
        initial_metrics = validate(model, val_loader, device)
        print('Initial validation:', initial_metrics, flush=True)
        iterator = iter(loader)
        losses = []
        training_started = time.perf_counter()
        with (args.output / 'training.jsonl').open('x', encoding='utf-8') as log:
            for step in range(args.steps):
                try:
                    feature, target, ids = next(iterator)
                except StopIteration:
                    iterator = iter(loader)
                    feature, target, ids = next(iterator)
                lr = official_lr(step, args.lr_horizon_steps)
                for group in optimizer.param_groups:
                    group['lr'] = lr
                loss = train_batch(model, optimizer, feature.to(device), target.to(device))
                losses.append(loss)
                log.write(json.dumps({'step': step + 1, 'loss': loss, 'lr': lr,
                                      'sample_ids': list(ids)}, allow_nan=False) + '\n')
                log.flush()
                if (step + 1) % args.log_every == 0 or step + 1 == args.steps:
                    print(f'Step {step+1}/{args.steps}, loss={loss:.6f}, lr={lr:.9g}', flush=True)
                if (step + 1) % args.checkpoint_every == 0 and step + 1 < args.steps:
                    torch.save({'state_dict': model.state_dict(), 'step': step + 1,
                                'config': config, 'source': source},
                               args.output / f'checkpoint_step{step+1}.pt')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - training_started
        final_metrics = validate(model, val_loader, device)
        checkpoint_path = args.output / 'checkpoint_final.pt'
        torch.save({'state_dict': model.state_dict(), 'step': args.steps,
                    'config': config, 'source': source}, checkpoint_path)
        # Reload into a NEW model and compare all validation metrics.
        reloaded = build_model().to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        reloaded.load_state_dict(checkpoint['state_dict'], strict=True)
        reload_metrics = validate(reloaded, val_loader, device)
        if not all(math.isclose(final_metrics[k], reload_metrics[k], abs_tol=1e-7, rel_tol=1e-7)
                   for k in final_metrics):
            raise RuntimeError('Checkpoint reload validation does not match')
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        result = {'status': 'completed', 'group': 'B0', 'scope': args.scope,
                  'performance_claims_allowed': False, 'seed': args.seed,
                  'steps': args.steps, 'batch_size': args.batch_size,
                  'lr_horizon_steps': args.lr_horizon_steps,
                  'source': source, 'data': data_report, 'environment': environment,
                  'initial_validation': initial_metrics, 'validation': final_metrics,
                  'checkpoint_reload_validation': reload_metrics, 'checkpoint_reload_passed': True,
                  'training': {'initial_loss': losses[0], 'final_loss': losses[-1],
                               'mean_loss': float(np.mean(losses)), 'last_update_lr': lr},
                  'model': {'parameter_count': sum(p.numel() for p in model.parameters())},
                  'resources': {'elapsed_seconds': time.perf_counter() - started,
                                'training_seconds': training_seconds,
                                'single_gpu_training_hours': training_seconds / 3600 if device.type == 'cuda' else None,
                                'peak_allocated_mib': torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == 'cuda' else None},
                  'checkpoint': {'path': checkpoint_path.name, 'sha256': sha256(checkpoint_path),
                                 'weights_only_no_optimizer_resume': True}}
        write_json(args.output / 'result.json', result)
        print('COMPLETED:', final_metrics, 'checkpoint reload verified', flush=True)
    except Exception:
        write_json(args.output / 'failure.json', {'status': 'failed', 'traceback': traceback.format_exc(),
                                                 'elapsed_seconds': time.perf_counter() - started})
        raise


def main():
    parser = argparse.ArgumentParser(description='B0 fixed-recipe training; no agent search or hidden test.')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--train-manifest', type=Path, required=True)
    parser.add_argument('--validation-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', choices=['mini', 'development_baseline'], required=True)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--lr-horizon-steps', type=int, default=200000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=2)
    parser.add_argument('--checkpoint-every', type=int, default=10000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--check-values', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
