"""Independent initialization for every job; frozen evaluator, candidate recipe."""
import json
import math
import os
from pathlib import Path
import platform
import sys
import time


def main():
    config = json.loads(Path(os.environ["IR_SA0_CONFIG"]).read_text(encoding="utf-8"))
    workspace = Path(config["workspace"]).resolve()
    sys.path.insert(0, str(workspace))
    import torch
    import numpy as np
    import cv2
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from prepare.dataset import IRDropDataset
    from program.runner import set_seed, validate
    from train.experiment import build_model, build_optimizer, compute_loss
    from train.feature_transform import transform
    import train.experiment as training
    for name in ("train.experiment", "train.mavi", "train.feature_transform", "prepare.dataset", "program.runner"):
        Path(sys.modules[name].__file__).resolve().relative_to(workspace)
    device = torch.device(config.get("device", "cuda"))
    set_seed(config["seed"])
    recipe = config["recipe"]
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=False)
    train = DataLoader(IRDropDataset(config["data_root"], config["train_manifest"]),
                       batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=0,
                       generator=torch.Generator().manual_seed(config["seed"]))
    val = DataLoader(IRDropDataset(config["data_root"], config["val_manifest"]),
                     batch_size=config["val_batch_size"], shuffle=False, num_workers=0)
    if not len(train) or not len(val):
        raise ValueError("空DataLoader")

    class Predictor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = build_model()
        def forward(self, feature):
            changed = transform(feature)
            if changed.shape != feature.shape or not torch.isfinite(changed).all():
                raise ValueError("特征变换须保持shape及有限值")
            return self.model(changed)
    model = Predictor().to(device)
    opt = recipe.get("optimizer")
    if opt:
        args = {"lr": recipe["initial_lr"], "weight_decay": opt.get("weight_decay", .01)}
        kind = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}[opt["name"]]
        if opt["name"] == "sgd":
            args["momentum"] = opt.get("momentum", 0.)
        optimizer = kind(model.parameters(), **args)
    else:
        optimizer = build_optimizer(model)
    for group in optimizer.param_groups:
        group["lr"] = recipe["initial_lr"]
    initial = validate(model, val, device)
    print("initial_metrics:", initial, flush=True)
    cuda = device.type == "cuda"
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    iterator = iter(train)
    # Native candidate build_scheduler is honored when changed; B0 keeps official before-update cosine.
    custom_scheduler = None
    if not recipe.get("scheduler") and config.get("custom_scheduler"):
        custom_scheduler = training.build_scheduler(optimizer, config["lr_horizon_steps"])
    with (out / "training.jsonl").open("x", encoding="utf-8") as log:
        for step in range(config["steps"]):
            try:
                feature, target, ids = next(iterator)
            except StopIteration:
                iterator = iter(train)
                feature, target, ids = next(iterator)
            feature, target = feature.to(device), target.to(device)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            scheduler = recipe.get("scheduler", {"name": "cosine", "min_lr": 1e-7})
            if not custom_scheduler:
                if hasattr(training, "research_lr") and not recipe.get("scheduler"):
                    lr = training.research_lr(step, config["lr_horizon_steps"], recipe["initial_lr"])
                else:
                    minimum = scheduler.get("min_lr", 1e-7)
                    lr = (recipe["initial_lr"] if scheduler["name"] == "constant" else
                          minimum + .5 * (recipe["initial_lr"] - minimum) *
                          (math.cos(math.pi * step / config["lr_horizon_steps"]) + 1))
                if not math.isfinite(lr) or lr <= 0:
                    raise ValueError("学习率非正或非有限")
                for group in optimizer.param_groups:
                    group["lr"] = lr
            lr = optimizer.param_groups[0]["lr"]
            prediction = model(feature)
            if prediction.shape != target.shape:
                raise ValueError("预测shape与标签不一致")
            spec = recipe.get("loss")
            if spec:
                loss = {"l1": lambda: F.l1_loss(prediction, target), "mse": lambda: F.mse_loss(prediction, target),
                        "smooth_l1": lambda: F.smooth_l1_loss(prediction, target, beta=spec.get("beta", .01))}[spec["name"]]() * spec.get("scale", 100.)
            else:
                loss = compute_loss(prediction, target)
            if loss.ndim or not torch.isfinite(loss):
                raise ValueError("loss必须为有限标量")
            loss.backward()
            parameters = [p for p in model.parameters() if p.requires_grad]
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                raise ValueError("缺失或非有限梯度")
            optimizer.step()
            if custom_scheduler:
                custom_scheduler.step()
            entry = {"step": step + 1, "lr": lr, "loss": float(loss.detach()), "sample_ids": list(ids)}
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            print(entry, flush=True)
    if cuda:
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) / 1024**2 if cuda else 0.
    final = validate(model, val, device)
    environment = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                   "numpy": np.__version__, "opencv": cv2.__version__, "cuda": torch.version.cuda,
                   "cudnn": torch.backends.cudnn.version(), "device": str(device),
                   "gpu": torch.cuda.get_device_name(device) if cuda else None,
                   "cudnn_deterministic": torch.backends.cudnn.deterministic,
                   "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                   "cudnn_tf32": torch.backends.cudnn.allow_tf32, "torch_num_threads": torch.get_num_threads()}
    result = {**{k: config[k] for k in ("seed", "steps", "batch_size", "val_batch_size", "lr_horizon_steps")},
              "initial_lr": recipe["initial_lr"], "recipe": recipe, "initial_metrics": initial,
              "final_metrics": final, "training_seconds": seconds, "peak_allocated_mib": peak,
              "environment": environment, "optimizer": type(optimizer).__name__,
              "initialized_from": "fresh", "parameter_count": sum(p.numel() for p in model.parameters())}
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
                "steps": config["steps"]}, out / "checkpoint.pt")
    (out / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("final_metrics:", final, flush=True)


if __name__ == "__main__":
    main()
