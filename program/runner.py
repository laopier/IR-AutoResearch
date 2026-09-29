from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from prepare.dataset import IRDropDataset
from prepare.evaluator import evaluate_batch
from train.experiment import (
    build_model,
    build_optimizer,
    build_scheduler,
    train_batch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def get_git_state() -> dict[str, object]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    return {
        "commit": commit,
        "dirty": bool(status),
    }


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available"
        )

    return torch.device(requested)


def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    metric_sums = {
        "mae_float": 0.0,
        "nrms_official": 0.0,
        "ssim_official": 0.0,
    }
    sample_count = 0

    with torch.no_grad():
        for feature, target, _ in loader:
            feature = feature.to(device)
            target = target.to(device)

            prediction = model(feature)
            metrics = evaluate_batch(prediction, target)
            batch_size = feature.shape[0]

            for name, value in metrics.items():
                metric_sums[name] += value * batch_size

            sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("Validation dataset is empty")

    return {
        name: value / sample_count
        for name, value in metric_sums.items()
    }


def run_experiment(
    *,
    data_root: Path,
    train_manifest: Path,
    validation_manifest: Path,
    output_path: Path,
    run_kind: str,
    description: str,
    total_steps: int,
    batch_size: int,
    seed: int,
    device_name: str,
) -> dict[str, object]:
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    set_seed(seed)
    device = resolve_device(device_name)

    train_dataset = IRDropDataset(
        data_root,
        train_manifest,
    )
    validation_dataset = IRDropDataset(
        data_root,
        validation_manifest,
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    optimizer = build_optimizer(model)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_steps,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )
    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    started_at = time.perf_counter()
    train_losses = []
    train_iterator = iter(train_loader)

    for _ in range(total_steps):
        try:
            feature, target, _ = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            feature, target, _ = next(train_iterator)

        feature = feature.to(device)
        target = target.to(device)

        loss = train_batch(
            model,
            optimizer,
            feature,
            target,
        )
        train_losses.append(loss)
        scheduler.step()

    validation_metrics = validate(
        model,
        validation_loader,
        device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(device)
            / 1024
            / 1024
        )
    else:
        peak_memory_mib = None

    elapsed_seconds = time.perf_counter() - started_at
    git_state = get_git_state()

    result = {
        "schema_version": 1,
        "run_kind": run_kind,
        "description": description,
        "git": git_state,
        "seed": seed,
        "device": str(device),
        "budget": {
            "total_steps": total_steps,
            "batch_size": batch_size,
        },
        "data": {
            "root": str(data_root.resolve()),
            "train_manifest": str(train_manifest.resolve()),
            "train_manifest_sha256": sha256_file(train_manifest),
            "validation_manifest": str(
                validation_manifest.resolve()
            ),
            "validation_manifest_sha256": sha256_file(
                validation_manifest
            ),
            "train_samples": len(train_dataset),
            "validation_samples": len(validation_dataset),
        },
        "model": {
            "parameter_count": parameter_count,
            "trainable_parameter_count": (
                trainable_parameter_count
            ),
        },
        "training": {
            "mean_loss": float(np.mean(train_losses)),
            "final_loss": float(train_losses[-1]),
            "final_learning_rate": float(
                optimizer.param_groups[0]["lr"]
            ),
        },
        "validation": validation_metrics,
        "resources": {
            "elapsed_seconds": elapsed_seconds,
            "peak_allocated_mib": peak_memory_mib,
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded IR-drop experiment."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validation-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--run-kind",
        choices=("smoke", "search"),
        required=True,
    )
    parser.add_argument("--description", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cuda",
    )
    args = parser.parse_args()

    result = run_experiment(
        data_root=args.data_root,
        train_manifest=args.train_manifest,
        validation_manifest=args.validation_manifest,
        output_path=args.output,
        run_kind=args.run_kind,
        description=args.description,
        total_steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device_name=args.device,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()