"""Random search with frozen MAVI and the shared training/selection interface."""
import os
from pathlib import Path
import shutil
import time

from sa0.controller import (
    ROOT, FIXED_RESULT_KEYS, read_json, write_json, sha256, source_hashes,
    dataset_hashes, freeze_sources, execute_training, compare_results,
)
from sa0.baseline_cache import reuse_baseline
from sa0.session import atomic_json, session_lock, contract_check, history_item
from b1.search import validate_config, generate_plan


def run_session(c, value):
    protocol = validate_config(value)
    if protocol.get("dataset_report"):
        from prepare.pilot_ready import verify
        verify(protocol["data_root"], protocol["dataset_report"], [protocol])
    session = Path(protocol["session_dir"]).resolve()
    session.mkdir(parents=True, exist_ok=True)
    frozen = session / "frozen"
    with session_lock(session):
        state_path = session / "session_state.json"
        if state_path.exists():
            if c.read_json(session / "protocol.json") != protocol:
                raise ValueError("已有B1会话配置不同，恢复原配置或使用新session_dir")
        else:
            if (session / "protocol.json").exists():
                raise RuntimeError("B1初始化未完成，请检查并使用新会话")
            c.write_json(session / "protocol.json", protocol)
            c.freeze_sources(frozen)
            c.write_json(session / "source_hashes.json", c.source_hashes(frozen))
            c.write_json(session / "data_hashes.json", c.dataset_hashes(protocol))
            names = ("b1/controller.py", "b1/search.py", "sa0/controller.py", "sa0/session.py",
                     "sa0/baseline_cache.py", "sa0/decision.py", "prepare/pilot_ready.py")
            c.write_json(session / "harness_hashes.json", {name: c.sha256(c.ROOT / name) for name in names})
            # Plan is fixed before the baseline and all candidate results.
            c.write_json(session / "search_plan.json", generate_plan(protocol))
            atomic_json(state_path, {"process_seconds": 0., "stop_reason": None})
        contract_check(c, session, protocol)
        plan = c.read_json(session / "search_plan.json")
        if plan != generate_plan(protocol):
            raise ValueError("预先生成的随机搜索计划发生变化")
        baseline_folder = session / "baseline"
        if not baseline_folder.exists() and protocol["baseline_source_session"]:
            print("B1核验并复用固定基线：", protocol["baseline_source_session"], flush=True)
            reuse_baseline(c, protocol, frozen, session, protocol["baseline_source_session"])
        if not baseline_folder.exists():
            baseline_folder.mkdir()
            print("B1开始训练固定基线：", baseline_folder / "train.log", flush=True)
            limited = {**protocol, "process_timeout_seconds": min(
                protocol["process_timeout_seconds"], protocol["max_session_process_seconds"] or float("inf"))}
            try:
                baseline_record = c.execute_training(limited, frozen, baseline_folder, frozen,
                                                     protocol["baseline_initial_lr"])
            except KeyboardInterrupt:
                c.write_json(baseline_folder / "record.json", {"training_status": "interrupted"})
                raise
            c.write_json(baseline_folder / "record.json", baseline_record)
        if not (baseline_folder / "record.json").is_file():
            raise RuntimeError("B1基线未归档，不自动重训")
        baseline_record = c.read_json(baseline_folder / "record.json")
        if baseline_record.get("training_status") != "completed":
            raise RuntimeError("B1基线未完成，请检查日志；不自动重训")
        baseline = baseline_record["result"]
        state = {"process_seconds": baseline_record["process_seconds"], "stop_reason": None}
        history = []
        interrupted = False
        failures = 0
        for candidate in plan["trials"]:
            number = candidate["trial_number"]
            folder = session / f"trial_{number:03d}"
            contract_check(c, session, protocol)
            if folder.exists():
                if not (folder / "record.json").is_file():
                    process_file = folder / "worker_process.json"
                    if process_file.is_file():
                        pid = c.read_json(process_file)["pid"]
                        try:
                            os.kill(pid, 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError(f"旧worker进程{pid}仍存在，先检查状态")
                    phase = c.read_json(folder / "phase.json") if (folder / "phase.json").is_file() else {}
                    c.write_json(folder / "record.json", {
                        "training_status": "interrupted", "comparison_status": "not_run",
                        "reason": "未归档尝试不自动重跑，训练成本按保护上限保守计入",
                        "process_seconds": protocol["process_timeout_seconds"] if phase.get("stage") == "training" else 0.,
                        "trial_number": number, "trial_dir": str(folder), "candidate": candidate,
                    })
                record = c.read_json(folder / "record.json")
            else:
                remaining = (protocol["max_session_process_seconds"] - state["process_seconds"]
                             if protocol["max_session_process_seconds"] is not None else float("inf"))
                if remaining <= 0:
                    state["stop_reason"] = "session_process_budget_exhausted"
                    break
                if failures >= protocol["stop_after_failures"]:
                    state["stop_reason"] = "consecutive_failures_limit"
                    break
                folder.mkdir()
                c.write_json(folder / "candidate.json", candidate)
                workspace = folder / "workspace"
                shutil.copytree(frozen, workspace)
                atomic_json(folder / "phase.json", {"stage": "training", "started_at": time.time()})
                print(f"B1第{number}轮：initial_lr={candidate['initial_lr']}，日志：{folder / 'train.log'}", flush=True)
                limited = {**protocol, "process_timeout_seconds": min(remaining, protocol["process_timeout_seconds"])}
                started = time.perf_counter()
                try:
                    record = c.execute_training(limited, frozen, folder, workspace, candidate["initial_lr"])
                    if record["training_status"] == "completed":
                        try:
                            record["decision"] = c.compare_results(baseline, record["result"], protocol)
                        except (KeyError, ValueError, TypeError) as error:
                            record.update(comparison_status="invalid", reason=str(error))
                        else:
                            record["comparison_status"] = "valid"
                    elif (folder / "train.log").is_file():
                        tail = (folder / "train.log").read_text(encoding="utf-8", errors="replace")[-3000:]
                        record["reason"] = tail.strip().splitlines()[-1] if tail.strip() else "training_failed"
                except KeyboardInterrupt:
                    record = {"training_status": "interrupted", "comparison_status": "not_run",
                              "process_seconds": time.perf_counter() - started}
                    interrupted = True
                except Exception as error:
                    record = {"training_status": "not_completed", "comparison_status": "not_run",
                              "reason": f"{type(error).__name__}: {error}",
                              "process_seconds": time.perf_counter() - started}
                record.update(trial_number=number, trial_dir=str(folder), candidate=candidate,
                              selection_method="precommitted_random_search", agent_calls=0)
                try:
                    contract_check(c, session, protocol)
                except RuntimeError as error:
                    record.update(comparison_status="invalid", reason=str(error))
                    state["stop_reason"] = "experiment_contract_changed"
                c.write_json(folder / "record.json", record)
            history.append(history_item(record, folder))
            state["process_seconds"] += record.get("process_seconds", 0.)
            # Same progress semantics as SA0: pending improved candidates count as progress.
            decision = record.get("decision", {})
            failures = 0 if decision.get("metrics_ok") and decision.get("keep") is not False else failures + 1
            atomic_json(state_path, state)
            print({"trial": number, "status": record["training_status"],
                   "decision": record.get("decision"), "reason": record.get("reason")}, flush=True)
            if interrupted:
                state["stop_reason"] = "user_interrupted"
                break
            if state["stop_reason"] == "experiment_contract_changed":
                break
        if state["stop_reason"] is None:
            state["stop_reason"] = "max_trials_reached"
        summary = {"condition": "B1", "search_plan": plan, "baseline": baseline, "history": history,
                   "budget_status": protocol["budget_status"], "agent_calls": 0,
                   "process_seconds": state["process_seconds"], "stop_reason": state["stop_reason"],
                   "baseline_reused": baseline_record.get("baseline_reused", False),
                   "baseline_reuse_origin": baseline_record.get("reuse_origin"),
                   "actual_current_process_seconds": (
                       state["process_seconds"] - baseline_record["process_seconds"]
                       + baseline_record.get("actual_reuse_seconds", 0.)
                       if baseline_record.get("baseline_reused") else state["process_seconds"]),
                   "retained_candidates": [x["record"]["trial_dir"] for x in history
                                           if x["record"].get("decision", {}).get("keep") is True],
                   "promising_candidates": [x["record"]["trial_dir"] for x in history
                                            if x["record"].get("decision", {}).get("metrics_ok")
                                            and x["record"].get("decision", {}).get("keep") is None]}
        atomic_json(session / "summary.json", summary)
        atomic_json(state_path, state)
        print("B1会话归档：", session / "summary.json", flush=True)
        if interrupted:
            raise KeyboardInterrupt


def main():
    import sys
    run_session(sys.modules[__name__], read_json(ROOT / "b1/config.json"))


if __name__ == "__main__":
    main()
