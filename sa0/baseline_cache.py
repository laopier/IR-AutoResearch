"""Verified baseline reuse; candidates still train from scratch."""
import ast
import copy
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time


def probe_environment(frozen, seed, timeout):
    # Reuse the exact environment expression from the unchanged training worker.
    tree = ast.parse((frozen / "worker.py").read_text(encoding="utf-8"))
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    expression = next(node.value for node in main.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "environment" for target in node.targets))
    script = (
        "import json, platform, sys\n"
        f"sys.path.insert(0, {str(frozen)!r})\n"
        "import torch, numpy as np, cv2\n"
        "from program.runner import set_seed\n"
        "from train.experiment import build_model, build_optimizer\n"
        f"set_seed({seed!r})\n"
        "device = torch.device('cuda')\n"
        "optimizer = build_optimizer(build_model())\n"
        f"environment = {ast.unparse(expression)}\n"
        "print('SA0_ENVIRONMENT=' + json.dumps(environment, allow_nan=False))\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=frozen,
                            capture_output=True, text=True, encoding="utf-8", timeout=timeout)
    if result.returncode:
        raise RuntimeError("基线复用环境检查失败：" + result.stderr[-3000:])
    rows = [line for line in result.stdout.splitlines() if line.startswith("SA0_ENVIRONMENT=")]
    if len(rows) != 1:
        raise RuntimeError("基线复用环境检查未返回唯一结果")
    return json.loads(rows[0].split("=", 1)[1])


def reuse_baseline(c, protocol, frozen, session, source):
    from sa0.session import session_lock

    source = Path(source).resolve()
    if source == session.resolve():
        raise ValueError("baseline_source_session不能指向当前会话")
    started = time.perf_counter()
    # A source still being written must never be read as a completed cache.
    with session_lock(source):
        old_protocol = c.read_json(source / "protocol.json")
        for key in (*c.FIXED_RESULT_KEYS, "baseline_initial_lr", "data_root", "train_manifest", "val_manifest"):
            if old_protocol.get(key) != protocol[key]:
                raise ValueError(f"复用基线的训练条件不同：{key}")
        if c.read_json(source / "data_hashes.json") != c.dataset_hashes(protocol):
            raise ValueError("复用基线的数据或split哈希不同")
        old_hashes = c.read_json(source / "source_hashes.json")
        if c.source_hashes(source / "frozen") != old_hashes or c.source_hashes(frozen) != old_hashes:
            raise ValueError("复用基线的冻结源码或训练worker不同")
        original = c.read_json(source / "baseline/record.json")
        if original.get("training_status") != "completed":
            raise ValueError("来源基线未成功完成，不能复用")
        baseline = original["result"]
        for key in c.FIXED_RESULT_KEYS:
            if baseline[key] != protocol[key]:
                raise ValueError(f"来源基线结果与协议不一致：{key}")
        if baseline.get("initial_lr") != protocol["baseline_initial_lr"]:
            raise ValueError("来源基线学习率与协议不一致")
        # Also reject non-finite metrics/cost and a baseline outside formal limits.
        c.compare_results(baseline, baseline, protocol)
        cost = original.get("process_seconds")
        if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
            raise ValueError("来源基线进程成本无效")
        artifacts = source / "baseline/artifacts"
        required = ("result.json", "checkpoint.pt", "training.jsonl")
        if any(not (artifacts / name).is_file() for name in required):
            raise ValueError("来源基线缺少结果、checkpoint或训练轨迹")
        if c.read_json(artifacts / "result.json") != baseline:
            raise ValueError("来源基线record与result.json不一致")
        current = probe_environment(frozen, protocol["seed"], protocol["process_timeout_seconds"])
        if not baseline.get("environment") or current != baseline["environment"]:
            raise ValueError("复用基线的运行环境不同")

        hashes = {name: c.sha256(artifacts / name) for name in required}
        temporary = session / "baseline_reuse.tmp"
        if temporary.exists():
            raise RuntimeError("存在未完成的基线复制，请检查baseline_reuse.tmp；不自动覆盖")
        temporary.mkdir()
        shutil.copytree(artifacts, temporary / "artifacts")
        for name in ("experiment.json", "train.log"):
            shutil.copy2(source / "baseline" / name, temporary / name)
        if any(c.sha256(temporary / "artifacts" / name) != value for name, value in hashes.items()):
            raise RuntimeError("基线复制后的文件哈希不同")
        destination = session / "baseline"
        record = copy.deepcopy(original)
        record.update(result_path=str(destination / "artifacts/result.json"),
                      config_path=str(destination / "experiment.json"),
                      log_path=str(destination / "train.log"), baseline_reused=True,
                      actual_reuse_seconds=time.perf_counter() - started,
                      reuse_origin={"session": str(source), "artifact_sha256": hashes,
                                    "environment": current,
                                    "budget_accounting": "original_baseline_cost_charged"})
        c.write_json(temporary / "record.json", record)
        temporary.rename(destination)
        return record
