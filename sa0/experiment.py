import json
import math
import os
import platform
import sys
import time
from pathlib import Path


def main():
    config_path = Path(os.environ["IR_SA0_CONFIG"])
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # 必须在导入模型、数据和评价模块前设置。
    workspace = Path(config["workspace"]).resolve()
    sys.path.insert(0, str(workspace))

    import torch
    import numpy as np
    import cv2
    from torch.utils.data import DataLoader

    from prepare.dataset import IRDropDataset
    from program.runner import set_seed, validate
    from train.experiment import (
        build_model,
        build_optimizer,
        train_batch,
    )

    for module_name in ("train.experiment", "train.mavi", "prepare.dataset", "program.runner"):
        Path(sys.modules[module_name].__file__).resolve().relative_to(workspace)

    seed = config["seed"]
    steps = config["steps"]
    horizon = config["lr_horizon_steps"]
    initial_lr = config["initial_lr"]

    if type(steps) is not int or steps <= 0:
        raise ValueError("steps必须是正整数")

    if type(horizon) is not int or horizon < steps:
        raise ValueError("lr_horizon_steps必须不小于steps")

    if (
        type(initial_lr) not in (int, float)
        or not math.isfinite(initial_lr)
        or initial_lr <= 1e-7
    ):
        raise ValueError("initial_lr必须是大于1e-7的有限数值")

    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda")
    set_seed(seed)

    train_dataset = IRDropDataset(
        config["data_root"],
        config["train_manifest"],
    )
    val_dataset = IRDropDataset(
        config["data_root"],
        config["val_manifest"],
    )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=0,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["val_batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        raise ValueError("训练或验证DataLoader为空")

    model = build_model().to(device)
    optimizer = build_optimizer(model)

    initial_metrics = validate(model, val_loader, device)
    print("initial_metrics:", initial_metrics, flush=True)

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    iterator = iter(train_loader)

    with (out_dir / "training.jsonl").open(
        "x", encoding="utf-8"
    ) as log_handle:
        for step in range(steps):
            try:
                feature, target, ids = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                feature, target, ids = next(iterator)

            feature = feature.to(device)
            target = target.to(device)

            lr = (
                1e-7
                + 0.5
                * (initial_lr - 1e-7)
                * (math.cos(math.pi * step / horizon) + 1)
            )

            for group in optimizer.param_groups:
                group["lr"] = lr

            loss = train_batch(
                model,
                optimizer,
                feature,
                target,
            )

            entry = {
                "step": step + 1,
                "lr": lr,
                "loss": loss,
                "sample_ids": list(ids),
            }
            log_handle.write(
                json.dumps(entry, allow_nan=False) + "\n"
            )
            log_handle.flush()
            print(entry, flush=True)

    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    peak_allocated_mib = (
        torch.cuda.max_memory_allocated(device) / 1024**2
    )

    final_metrics = validate(model, val_loader, device)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": steps,
            "config": config,
        },
        out_dir / "checkpoint.pt",
    )

    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "device_total_bytes": torch.cuda.get_device_properties(device).total_memory,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "torch_num_threads": torch.get_num_threads(),
        "optimizer": type(optimizer).__name__,
    }
    result = {
        "environment": environment,
        "steps": steps,
        "seed": seed,
        "batch_size": config["batch_size"],
        "val_batch_size": config["val_batch_size"],
        "lr_horizon_steps": horizon,
        "initial_lr": initial_lr,
        "parameter_count": sum(
            p.numel() for p in model.parameters()
        ),
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "training_seconds": training_seconds,
        "peak_allocated_mib": peak_allocated_mib,
    }

    with (out_dir / "result.json").open(
        "x", encoding="utf-8"
    ) as handle:
        json.dump(
            result,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )

    print("final_metrics:", final_metrics, flush=True)


if __name__ == "__main__":
    main()
