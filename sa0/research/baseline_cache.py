"""Verified B0 artifacts shared between research sessions; costs remain explicit."""
import json
import math
import time
from pathlib import Path

CORE = ("train/experiment.py", "train/mavi.py", "prepare/dataset.py",
        "prepare/evaluator.py", "program/runner.py", "prepare/split_manifest.py")


def runtime_environment(device="cuda", threads=2):
    import platform
    import torch
    import numpy as np
    import cv2
    from program.runner import set_seed
    set_seed(0)
    torch.set_num_threads(threads)
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": str(torch.__version__),
            "numpy": np.__version__, "opencv": cv2.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "device": device,
            "gpu": torch.cuda.get_device_name(device) if device.startswith("cuda") else None,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32, "torch_num_threads": torch.get_num_threads()}


def contract(c, config, workspace, seed):
    return {"version": 1, "seed": seed, "steps": config["steps"],
            "batch_size": config["batch_size"], "val_batch_size": config["val_batch_size"],
            "lr_horizon_steps": config["lr_horizon_steps"], "initial_lr": config["baseline_initial_lr"],
            "device": config.get("device", "cuda"),
            "core_hashes": {name: c.sha256(Path(workspace) / name) for name in CORE},
            "data_hashes": c.dataset_hashes(config)}


def publish(c, config, seed, attempts, destination, environment):
    """Adapt our native resumable B0 format to research format without retraining."""
    import torch
    from b0.checkpoint import load_checkpoint
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    native = c.read_json(Path(attempts[-1]) / "result.json")
    cfg = c.read_json(Path(attempts[-1]) / "config.json")
    expected = contract(c, config, c.ROOT, seed)
    if native.get("status") != "completed" or not native.get("checkpoint_reload_passed"):
        raise ValueError("B0未完成或checkpoint重载未通过")
    for field in ("seed", "steps", "batch_size", "lr_horizon_steps"):
        if native[field] != expected[field]:
            raise ValueError(f"B0缓存条件不匹配：{field}")
    if config["baseline_initial_lr"] != 2e-4 or config["val_batch_size"] != 1:
        raise ValueError("native B0缓存仅支持冻结官方配方")
    recorded = native["source"]["file_sha256"]
    if any(recorded.get(name) != value for name, value in expected["core_hashes"].items()):
        raise ValueError("B0模型/评价/数据处理代码不匹配")
    # Resolve manifests and every data array from recorded hashes, not just path strings.
    data = native["data"]
    import csv
    for field, split in (("train_manifest", "train"), ("val_manifest", "validation")):
        path = Path(config[field])
        if c.sha256(path) != data[split]["manifest_sha256"]:
            raise ValueError("B0清单不匹配")
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                for name in row:
                    if c.sha256(Path(config["data_root"]) / name) != data["files_sha256"][f"{split}/{name}"]:
                        raise ValueError("B0数组不匹配")
    env = native["environment"]
    mapped = {"cuda_runtime": "cuda", "torch_threads": "torch_num_threads"}
    for field in ("python", "platform", "torch", "numpy", "opencv", "cuda_runtime", "cudnn", "gpu", "torch_threads"):
        if env[field] != environment[mapped.get(field, field)]:
            raise ValueError(f"B0环境不匹配：{field}")
    # Official recipe is frozen in native checkpoint; no weights-only import.
    checkpoint = Path(attempts[-1]) / native["checkpoint"]["path"]
    if c.sha256(checkpoint) != native["checkpoint"]["sha256"]:
        raise ValueError("B0 checkpoint哈希不一致")
    state = load_checkpoint(checkpoint)
    resume_contract = state["resume_contract"]
    if state["step"] != config["steps"] or resume_contract["seed"] != seed or cfg["device"] != expected["device"]:
        raise ValueError("checkpoint步数/seed/设备不一致")
    if any(resume_contract["code"].get(name) != c.sha256(c.ROOT / name) for name in resume_contract["code"]):
        raise ValueError("native B0执行代码已变化")
    from train.experiment import build_model
    model = build_model()
    model.load_state_dict(state["state_dict"], strict=True)
    rows, costs, first_metrics = {}, [], None
    for attempt in attempts:
        folder = Path(attempt)
        try:
            result = c.read_json(folder / "result.json")
        except (FileNotFoundError, json.JSONDecodeError):
            result = None
        if not (folder / "config.json").is_file() or not (folder / "training.jsonl").is_file():
            costs.append({"path": str(folder), "completed": False, "training_seconds": None})
            continue
        start = c.read_json(folder / "config.json").get("resume")
        # Later resumed attempts replace any updates beyond the last saved checkpoint.
        boundary = load_checkpoint(Path(start))["step"] if start else 0
        rows = {step: value for step, value in rows.items() if step <= boundary}
        if first_metrics is None and result and not boundary:
            first_metrics = result["initial_validation"]
        for line in (folder / "training.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if result:
                    raise
                continue  # A power cut can leave a truncated last line in a failed attempt.
            if item["step"] <= config["steps"]:
                rows[item["step"]] = item
        costs.append({"path": str(folder), "completed": bool(result),
                      "training_seconds": result["resources"]["training_seconds"] if result else None})
    if sorted(rows) != list(range(1, config["steps"] + 1)):
        raise ValueError("B0训练谱系缺步，不能发布缓存")
    metrics = native["validation"]
    if not all(type(metrics[k]) in (int, float) and math.isfinite(metrics[k]) for k in metrics):
        raise ValueError("B0指标非有限值")
    result = {**{k: expected[k] for k in ("seed", "steps", "batch_size", "val_batch_size", "lr_horizon_steps")},
              "initial_lr": 2e-4, "recipe": {"initial_lr": 2e-4}, "initial_metrics": first_metrics,
              "final_metrics": metrics, "environment": environment, "initialized_from": "fresh_then_exact_resume" if len(attempts) > 1 else "fresh",
              "training_seconds": sum(x["training_seconds"] or 0 for x in costs),
              "peak_allocated_mib": native["resources"]["peak_allocated_mib"],
              "parameter_count": native["model"]["parameter_count"], "native_result": str(Path(attempts[-1]) / "result.json"),
              "accounting_uncertain": any(x["training_seconds"] is None for x in costs)}
    temp = destination.with_name(destination.name + ".publishing")
    if temp.exists():
        temp.rename(temp.with_name(temp.name + f".interrupted_{time.time_ns()}"))
    temp.mkdir(parents=True)
    artifacts = temp / "artifacts"
    artifacts.mkdir()
    c.write_json(artifacts / "result.json", result)
    torch.save({"model": {"model." + k: v for k, v in state["state_dict"].items()},
                "optimizer": state["optimizer_state_dict"], "steps": config["steps"], "config": cfg}, artifacts / "checkpoint.pt")
    (artifacts / "training.jsonl").write_text("".join(json.dumps(rows[i], allow_nan=False) + "\n" for i in sorted(rows)), encoding="utf-8")
    c.write_json(temp / "receipt.json", {"contract": expected, "environment": environment, "costs": costs,
        "artifact_hashes": {name: c.sha256(artifacts / name) for name in ("result.json", "checkpoint.pt", "training.jsonl")},
        "native_checkpoint_sha256": c.sha256(checkpoint), "source_recipe": cfg["recipe"]})
    c.write_json(temp / "experiment.json", {**{k: config[k] for k in ("data_root", "train_manifest", "val_manifest", "steps", "batch_size", "val_batch_size", "lr_horizon_steps", "device")},
        "seed": seed, "recipe": {"initial_lr": 2e-4}, "native_source": str(Path(attempts[-1])), "baseline_import": True})
    temp.rename(destination)


def read(c, config, workspace, seed, environment):
    folder = Path(config["baseline_cache_dir"]) / f"seed_{seed}"
    if not (folder / "receipt.json").is_file():
        raise RuntimeError(f"共享B0未准备好：{folder}，先运行python -m b0.shared_run")
    receipt = c.read_json(folder / "receipt.json")
    if receipt["contract"] != contract(c, config, workspace, seed) or receipt["environment"] != environment:
        raise ValueError("共享B0的训练条件、源码、数据或环境不匹配，禁止复用")
    for name, value in receipt["artifact_hashes"].items():
        if c.sha256(folder / "artifacts" / name) != value:
            raise ValueError("共享B0产物被修改")
    result = c.read_json(folder / "artifacts/result.json")
    return folder, receipt, result
