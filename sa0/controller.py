import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sa0.agent import call_agent
from sa0.decision import evaluate_candidate
from sa0.feedback import write_agent_input
from sa0.proposal import (
    build_candidate_source,
    validate_proposal,
)


ROOT = Path(__file__).resolve().parents[1]

FIXED_RESULT_KEYS = (
    "seed",
    "steps",
    "batch_size",
    "val_batch_size",
    "lr_horizon_steps",
)

METRIC_KEYS = (
    "mae_float",
    "nrms_official",
    "ssim_official",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def source_hashes(workspace: Path) -> dict:
    return {
        path.relative_to(workspace).as_posix(): sha256(path)
        for path in sorted(workspace.rglob("*.py"))
    }


def dataset_hashes(config: dict) -> dict:
    """记录清单以及清单实际引用的数据文件。"""
    data_root = Path(config["data_root"]).resolve()
    hashes = {}

    for key in ("train_manifest", "val_manifest"):
        manifest = Path(config[key]).resolve()
        hashes[str(manifest)] = sha256(manifest)

        with manifest.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            for row in csv.reader(handle):
                if len(row) != 2:
                    raise ValueError("数据清单必须每行两列")

                for relative_path in row:
                    path = (data_root / relative_path).resolve()
                    path.relative_to(data_root)
                    hashes[str(path)] = sha256(path)

    return hashes


def freeze_sources(destination: Path):
    destination.mkdir()

    # 当前模型、数据处理、评价的Python依赖。
    for package in ("train", "prepare", "program"):
        source_dir = ROOT / package

        for source in source_dir.rglob("*.py"):
            target = destination / source.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)

    # 同一会话的所有实验使用同一份训练worker。
    shutil.copy2(
        ROOT / "sa0/experiment.py",
        destination / "worker.py",
    )


def make_training_config(
    protocol: dict,
    workspace: Path,
    out_dir: Path,
    initial_lr: float,
) -> dict:
    keys = (
        "data_root",
        "train_manifest",
        "val_manifest",
        "seed",
        "steps",
        "batch_size",
        "val_batch_size",
        "lr_horizon_steps",
    )

    config = {key: protocol[key] for key in keys}
    config.update({
        "workspace": str(workspace),
        "out_dir": str(out_dir),
        "initial_lr": initial_lr,
    })
    return config


def execute_training(
    protocol: dict,
    frozen: Path,
    folder: Path,
    workspace: Path,
    initial_lr: float,
) -> dict:
    config = make_training_config(
        protocol,
        workspace,
        folder / "artifacts",
        initial_lr,
    )

    config_path = folder / "experiment.json"
    write_json(config_path, config)

    environment = os.environ.copy()
    environment["IR_SA0_CONFIG"] = str(config_path)

    command = [
        sys.executable,
        "-u",
        str(frozen / "worker.py"),
    ]

    before_sources = source_hashes(workspace)
    started = time.perf_counter()

    with (folder / "train.log").open(
        "x", encoding="utf-8"
    ) as log_handle:
        try:
            with subprocess.Popen(
                command, cwd=workspace, env=environment,
                stdout=log_handle, stderr=subprocess.STDOUT,
            ) as completed:
                write_json(folder / "worker_process.json", {
                    "pid": completed.pid, "started_at": time.time(),
                })
                try:
                    completed.wait(timeout=protocol["process_timeout_seconds"])
                except BaseException:
                    completed.kill()
                    completed.wait()
                    raise
        except subprocess.TimeoutExpired:
            record = {
                "training_status": "timed_out",
                "comparison_status": "not_run",
            }
        else:
            record = {
                "training_status": (
                    "completed"
                    if completed.returncode == 0
                    else "failed"
                ),
                "comparison_status": "not_run",
                "returncode": completed.returncode,
            }

    record["process_seconds"] = time.perf_counter() - started
    record["config_path"] = str(config_path)
    record["log_path"] = str(folder / "train.log")

    if source_hashes(workspace) != before_sources:
        record["training_status"] = "invalid"
        record["reason"] = "source_changed_during_execution"

    if record["training_status"] == "completed":
        result_path = folder / "artifacts/result.json"

        if not result_path.is_file():
            record["training_status"] = "invalid"
            record["reason"] = "result_missing"
        else:
            if not (folder / "artifacts/checkpoint.pt").is_file():
                record["training_status"] = "invalid"
                record["reason"] = "checkpoint_missing"
            else:
                record["result"] = read_json(result_path)
                record["result_path"] = str(result_path)

    return record


def compare_results(
    baseline: dict,
    candidate: dict,
    protocol: dict,
) -> dict:
    for key in FIXED_RESULT_KEYS:
        if candidate[key] != baseline[key]:
            raise ValueError(f"对照条件不同：{key}")

    for result in (baseline, candidate):
        for key in METRIC_KEYS:
            value = result["final_metrics"][key]
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"验证指标不是有限值：{key}")

        for key in (
            "training_seconds",
            "peak_allocated_mib",
        ):
            value = result[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"资源记录无效：{key}")

    if baseline.get("environment") != candidate.get("environment"):
        raise ValueError("environment differs between baseline and candidate")
    if protocol["budget_status"] == "formal_limits" and (
        baseline["training_seconds"] > protocol["max_training_seconds"]
        or baseline["peak_allocated_mib"] > protocol["max_peak_allocated_mib"]
    ):
        raise ValueError("基线超过正式资源上限，不能作合规对照")
    return evaluate_candidate(
        candidate,
        baseline,
        protocol["max_training_seconds"],
        protocol["max_peak_allocated_mib"],
        budget_status=protocol["budget_status"],
    )


def build_request(protocol: dict) -> str:
    examples = {
        "learning_rate": {
            "hypothesis": "依据、预期及风险",
            "kind": "learning_rate",
            "change": {"initial_lr": 0.0004},
        },
        "model": {
            "hypothesis": "依据、预期及风险",
            "kind": "model",
            "change": {
                "edits": [{
                    "path": "train/mavi.py",
                    "old": "唯一匹配的原代码片段",
                    "new": "替换后的代码片段",
                }]
            },
        },
    }

    allowed = protocol["allowed_kinds"]

    request = (
        "根据固定基线、源码和实验历史，提出一个新候选。"
        f"本次允许的候选类型只有：{', '.join(allowed)}。"
        "每轮只选择一种修改类型，不重复已有候选。"
        "学习率候选保持基线模型；"
        "模型候选保持基线学习率。"
        "模型修改只能涉及train/mavi.py，"
        "保持MAVI()、init_weights()及输入输出接口可用。"
        "不要改变数据处理、损失、优化器或评价口径。"
        "假设说明依据、预期及风险，"
        "不要把推测写成已经证实的原因。"
        "old片段必须在冻结源码中唯一匹配，"
        "必要时提供整个类或函数作为上下文。"
        "短步数运行仅检查流程，不能据此判断收敛或泛化优势。"
        "只返回一个JSON对象，不使用Markdown代码块。"
        "不调用工具，不修改文件，不启动训练。"
    )

    if "learning_rate" in allowed:
        request += (
            "学习率必须大于1e-7，且不超过"
            f"{protocol['initial_lr_upper_bound']}。"
        )

    for kind in allowed:
        request += (
            f"\n{kind}提案格式：\n"
            + json.dumps(examples[kind], ensure_ascii=False)
        )

    return request

def main():
    from sa0.session import run_session
    run_session(sys.modules[__name__], read_json(ROOT / "sa0/config.json"))


if __name__ == "__main__":
    main()
