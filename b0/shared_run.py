"""Sequential seed012 B0, checkpoint recovery and verified shared cache export."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from sa0 import controller as c
from sa0.session import atomic_json, session_lock
from sa0.research import baseline_cache

ROOT = Path(__file__).resolve().parents[1]


def command(config, seed, output, checkpoint=None):
    args = [sys.executable, "-u", "-m", "b0.run", "--scope", "development_baseline"]
    for field, flag in (("data_root", "--data-root"), ("train_manifest", "--train-manifest"),
                        ("val_manifest", "--validation-manifest"), ("steps", "--steps"),
                        ("batch_size", "--batch-size"), ("lr_horizon_steps", "--lr-horizon-steps"),
                        ("device", "--device"), ("checkpoint_every", "--checkpoint-every")):
        args += [flag, str(config[field])]
    args += ["--seed", str(seed), "--output", str(output), "--num-workers", "0", "--cpu-threads", "2", "--log-every", "10"]
    if checkpoint:
        args += ["--resume", str(checkpoint)]
    return args


def latest_checkpoint(attempts, target):
    from b0.checkpoint import load_checkpoint
    best = None
    for attempt in attempts:
        for path in Path(attempt).glob("checkpoint_*.pt"):
            try:
                step = load_checkpoint(path)["step"]
            except Exception as error:
                print(f"跳过无法读取的checkpoint：{path}，{error}", flush=True)
                continue
            if step == target:
                try:
                    result = c.read_json(path.parent / "result.json")
                    finished = result.get("status") == "completed" and result.get("checkpoint_reload_passed") is True
                except (FileNotFoundError, json.JSONDecodeError):
                    finished = False
                if not finished:
                    print(f"最终验证未归档，改从较早checkpoint恢复：{path}", flush=True)
                    continue
            if 0 < step <= target and (best is None or step > best[0]):
                best = step, path
    return best


def run(config):
    if config["seeds"] != [0, 1, 2] or config["baseline_initial_lr"] != 2e-4 or config["val_batch_size"] != 1:
        raise ValueError("共享B0必须为官方配方seed012及验证batch1")
    if config.get("dataset_report"):
        from prepare.pilot_ready import verify
        verify(config["data_root"], config["dataset_report"], [config])
    root = Path(config["baseline_cache_dir"])
    root.mkdir(parents=True, exist_ok=True)
    with session_lock(root):
        environment = baseline_cache.runtime_environment(config["device"], 2)
        contracts = [baseline_cache.contract(c, config, ROOT, seed) for seed in config["seeds"]]
        execution = {name: c.sha256(ROOT / name) for name in ("b0/run.py", "b0/checkpoint.py", "b0/shared_run.py", "sa0/research/baseline_cache.py")}
        state_path = root / "state.json"
        if state_path.exists():
            state = c.read_json(state_path)
            if state["config"] != config or state["contracts"] != contracts or state["execution"] != execution or state["environment"] != environment:
                raise ValueError("共享B0协议/源码/数据/环境改变，请恢复原条件或使用新缓存目录")
        else:
            state = {"version": 1, "config": config, "contracts": contracts, "execution": execution,
                     "environment": environment, "seeds": {}, "started_at": time.time()}
            atomic_json(state_path, state)
        for seed in config["seeds"]:
            node = state["seeds"].setdefault(str(seed), {"attempts": []})
            entry = root / f"seed_{seed}"
            if (entry / "receipt.json").exists():
                baseline_cache.read(c, config, ROOT, seed, environment)
                node["status"] = "completed"
                atomic_json(state_path, state)
                print(f"seed={seed}已完成，核验并复用", flush=True)
                continue
            if node.get("status") == "running" and node.get("pid"):
                try:
                    os.kill(node["pid"], 0)
                except ProcessLookupError:
                    pass
                else:
                    raise RuntimeError("原B0进程仍在运行，拒绝重复启动")
            bootstrap = config.get("seed0_bootstrap") if seed == 0 and not node["attempts"] else None
            if bootstrap:
                folder = Path(bootstrap)
                if not (folder / "result.json").is_file():
                    raise ValueError("配置的seed0起点结果不存在；确认后设seed0_bootstrap=null可从头训练")
                if c.read_json(folder / "result.json")["seed"] != seed:
                    raise ValueError("起点seed不匹配")
                node["attempts"].append(str(folder))
                atomic_json(state_path, state)
            best = latest_checkpoint(node["attempts"], config["steps"])
            if best and best[0] == config["steps"]:
                # A final checkpoint without a completed result is not publishable.
                final_dir = best[1].parent
                if not (final_dir / "result.json").exists():
                    raise RuntimeError("目标步checkpoint存在但最终验证未归档，先检查；不当成完成")
                if str(final_dir) != node["attempts"][-1]:
                    raise RuntimeError("恢复谱系的最终结果不在最后一次尝试，需人工检查")
            else:
                output = root / "native" / f"seed_{seed}" / f"attempt_{len(node['attempts']) + 1:03d}"
                output.parent.mkdir(parents=True, exist_ok=True)
                log = output.with_suffix(".log")
                node.update(status="reserved", pid=None)
                node["attempts"].append(str(output))
                atomic_json(state_path, state)
                print(f"B0 seed={seed}，目标{config['steps']}步，从{best[0] if best else 0}步恢复；日志：{log}", flush=True)
                try:
                    with log.open("x", encoding="utf-8") as stream:
                        process = subprocess.Popen(command(config, seed, output, best[1] if best else None),
                                                   cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                        node.update(status="running", pid=process.pid)
                        atomic_json(state_path, state)
                        try:
                            code = process.wait()
                        except BaseException:
                            process.terminate()
                            try:
                                process.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                            raise
                    if code:
                        raise RuntimeError(f"B0退出码{code}，查看{log}")
                except BaseException:
                    node.update(status="interrupted_or_failed", pid=None)
                    atomic_json(state_path, state)
                    raise
            baseline_cache.publish(c, config, seed, node["attempts"], entry, environment)
            node.update(status="completed", pid=None)
            atomic_json(state_path, state)
            print(f"seed={seed}缓存发布完成：{entry}", flush=True)
        receipts = [c.read_json(root / f"seed_{seed}" / "receipt.json") for seed in config["seeds"]]
        results = [c.read_json(root / f"seed_{seed}" / "artifacts/result.json") for seed in config["seeds"]]
        import statistics
        metrics = ("mae_float", "nrms_official", "ssim_official")
        atomic_json(root / "summary.json", {"status": "completed", "seeds": config["seeds"], "steps": config["steps"],
            "mean": {k: statistics.mean(r["final_metrics"][k] for r in results) for k in metrics},
            "std": {k: statistics.stdev(r["final_metrics"][k] for r in results) for k in metrics},
            "known_training_seconds": sum(r["training_seconds"] for r in results),
            "accounting_uncertain": any(r["accounting_uncertain"] for r in results),
            "native_attempts": [x for receipt in receipts for x in receipt["costs"]],
            "note": "共享成本须另计；导入不代表零科研成本。中断损耗时间可能未知。"})
        print("共享B0 seed012完成：", root / "summary.json", flush=True)


def main():
    config = json.loads((ROOT / "b0/shared_config.json").read_text(encoding="utf-8"))
    if "--check-only" in sys.argv:
        from prepare.pilot_ready import verify
        from b0.checkpoint import load_checkpoint
        verify(config["data_root"], config["dataset_report"], [config])
        env = baseline_cache.runtime_environment(config["device"], 2)
        bootstrap = config.get("seed0_bootstrap")
        if bootstrap:
            folder = Path(bootstrap)
            result = c.read_json(folder / "result.json")
            checkpoint = folder / result["checkpoint"]["path"]
            if c.sha256(checkpoint) != result["checkpoint"]["sha256"]:
                raise ValueError("起点checkpoint哈希错误")
            state = load_checkpoint(checkpoint)
            saved = state["resume_contract"]
            for name, value in (("seed", 0), ("batch_size", config["batch_size"]), ("lr_horizon_steps", config["lr_horizon_steps"]), ("cpu_threads", 2), ("device", config["device"]), ("scope", "development_baseline")):
                if saved[name] != value:
                    raise ValueError(f"seed0恢复条件不匹配：{name}")
            for name, value in saved["code"].items():
                if c.sha256(ROOT / name) != value:
                    raise ValueError(f"seed0恢复执行代码不同：{name}")
            for name, value in saved["environment"].items():
                mapped = {"cuda_runtime": "cuda", "torch_threads": "torch_num_threads"}.get(name, name)
                if env[mapped] != value:
                    raise ValueError(f"seed0恢复环境不同：{name}")
            if state["step"] >= config["steps"]:
                raise ValueError("seed0起点步数应小于新目标")
        print("检查通过：数据已审计，seed0恢复配方/源码/环境匹配；未启动训练、未创建缓存。", flush=True)
        return
    run(config)


if __name__ == "__main__":
    main()
