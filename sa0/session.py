"""Session orchestration, history, budgets and recovery between trials."""
import ast
from contextlib import contextmanager
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import time


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def session_lock(session):
    handle = (session / ".lock").open("a+b")
    if handle.seek(0, 2) == 0:
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("本会话已有controller运行，不能并发启动")
    try:
        yield
    finally:
        if os.name == "nt":
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def normalize_protocol(value):
    p = copy.deepcopy(value)
    p.setdefault("history_sessions", [])
    p.setdefault("history_results", [])
    p.setdefault("max_agent_calls", p["max_trials"])
    p.setdefault("max_session_process_seconds", p["process_timeout_seconds"] * (p["max_trials"] + 1))
    p.setdefault("stop_after_failures", p["max_trials"])
    p.setdefault("first_proposal_path", None)
    p.setdefault("baseline_source_session", None)
    if p["baseline_source_session"] is not None and (
        not isinstance(p["baseline_source_session"], str) or not p["baseline_source_session"].strip()
    ):
        raise ValueError("baseline_source_session必须为非空路径或null")
    p.setdefault("agent", {})
    defaults = {
        "codex_path": "/mnt/c/Users/皮/AppData/Local/OpenAI/Codex/bin/8aaf1547b825b104/codex.exe",
        "model": "gpt-6.1-sol", "reasoning_effort": "low",
        "timeout_seconds": 180, "max_prompt_chars": 100000,
    }
    p["agent"] = {**defaults, **p["agent"]}
    allowed = p.get("allowed_kinds")
    if (not isinstance(allowed, list) or not allowed
            or any(kind not in {"model", "learning_rate"} for kind in allowed)):
        raise ValueError("allowed_kinds配置无效")
    for key in ("steps", "batch_size", "val_batch_size", "max_trials",
                "lr_horizon_steps", "max_agent_calls", "stop_after_failures"):
        if type(p[key]) is not int or p[key] <= 0:
            raise ValueError(f"{key}必须是正整数")
    if type(p["seed"]) is not int or p["seed"] < 0:
        raise ValueError("seed必须是非负整数")
    if p["steps"] > p["lr_horizon_steps"]:
        raise ValueError("steps不能超过lr_horizon_steps")
    for key in ("initial_lr_upper_bound", "baseline_initial_lr", "process_timeout_seconds",
                "max_session_process_seconds", "max_training_seconds", "max_peak_allocated_mib"):
        if type(p[key]) not in (int, float) or not math.isfinite(p[key]) or p[key] <= 0:
            raise ValueError(f"{key}必须是正的有限数值")
    if p["baseline_initial_lr"] <= 1e-7 or p["initial_lr_upper_bound"] <= 1e-7:
        raise ValueError("学习率必须大于1e-7")
    if p["budget_status"] not in {"test_limits", "formal_limits"}:
        raise ValueError("未知的预算状态")
    for key in ("timeout_seconds", "max_prompt_chars"):
        x = p["agent"][key]
        if type(x) not in (int, float) or not math.isfinite(x) or x <= 0:
            raise ValueError(f"agent.{key}配置无效")
    for key in ("codex_path", "model", "reasoning_effort"):
        if not isinstance(p["agent"][key], str) or not p["agent"][key].strip():
            raise ValueError(f"agent.{key}配置无效")
    return p


def curve_summary(path):
    if not path.is_file():
        return None
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        return {"steps_logged": 0}
    indices = sorted(set([0, len(rows) - 1] + list(range(0, len(rows), max(1, len(rows) // 20)))))
    finite_losses = [r["loss"] for r in rows
                     if type(r.get("loss")) in (int, float) and math.isfinite(r["loss"])]
    return {"steps_logged": len(rows), "sampled_steps": [rows[i] for i in indices],
            "min_loss": min(finite_losses) if finite_losses else None,
            "last_loss": rows[-1].get("loss"), "full_log": str(path)}


def model_signature(source):
    tree = ast.dump(ast.parse(source), include_attributes=False)
    return hashlib.sha256(tree.encode("utf-8")).hexdigest()


def history_item(record, folder):
    item = {"record": record,
            "training_curve": curve_summary(folder / "artifacts/training.jsonl")}
    log = folder / "train.log"
    if record.get("training_status") != "completed" and log.is_file():
        item["training_error_excerpt"] = log.read_text(encoding="utf-8", errors="replace")[-5000:]
    return item


def proposal_signature(c, proposal, baseline_source):
    if proposal["kind"] == "learning_rate":
        return "learning_rate:" + repr(float(proposal["change"]["initial_lr"]))
    return "model:" + model_signature(c.build_candidate_source(baseline_source, proposal))


def import_history(c, protocol, frozen, data_hashes):
    history = []
    signatures = set()
    current_source = (frozen / "train/mavi.py").read_text(encoding="utf-8")
    current_hashes = c.source_hashes(frozen)
    for directory in protocol["history_sessions"]:
        old = Path(directory)
        summary = c.read_json(old / "summary.json")
        old_protocol = c.read_json(old / "protocol.json")
        comparable = (
            all(old_protocol.get(k) == protocol[k] for k in c.FIXED_RESULT_KEYS)
            and old_protocol.get("baseline_initial_lr") == protocol["baseline_initial_lr"]
            and c.read_json(old / "data_hashes.json") == data_hashes
            and c.read_json(old / "source_hashes.json") == current_hashes
        )
        old_source = (old / "frozen/train/mavi.py").read_text(encoding="utf-8")
        same_model = model_signature(old_source) == model_signature(current_source)
        for item in summary.get("history", []):
            record = item["record"]
            history.append({"source_session": str(old), "context_only": not comparable,
                            "baseline": summary["baseline"], "record": record,
                            "training_curve": item.get("training_curve"),
                            "training_error_excerpt": item.get("training_error_excerpt")})
            proposal = record.get("proposal")
            # Do not block retrying a malformed patch that never reached training.
            if same_model and proposal and record.get("result"):
                try:
                    signatures.add(proposal_signature(c, proposal, current_source))
                except (ValueError, SyntaxError):
                    pass
    for item in protocol["history_results"]:
        baseline = c.read_json(Path(item["baseline_result"]))
        candidate = c.read_json(Path(item["candidate_result"]))
        proposal = c.validate_proposal(item["proposal"])
        history.append({"context_only": True, "source": "legacy_result",
                        "baseline": baseline,
                        "record": {"training_status": "completed", "result": candidate,
                                   "proposal": proposal}})
        if proposal["kind"] == "learning_rate":
            signatures.add(proposal_signature(c, proposal, current_source))
    return history, signatures


def contract_check(c, session, protocol):
    try:
        harness_file = session / "harness_hashes.json"
        if harness_file.exists():
            expected = c.read_json(harness_file)
            actual = {name: c.sha256(c.ROOT / name if "/" in name else c.ROOT / "sa0" / name)
                      for name in expected}
            if actual != expected:
                raise RuntimeError("会话执行代码发生变化，不能沿用旧会话")
        if c.source_hashes(session / "frozen") != c.read_json(session / "source_hashes.json"):
            raise RuntimeError("冻结源码发生变化，停止会话")
        if c.dataset_hashes(protocol) != c.read_json(session / "data_hashes.json"):
            raise RuntimeError("数据或split发生变化，停止会话")
    except (OSError, ValueError) as error:
        raise RuntimeError("冻结文件或数据无法核验，停止会话") from error


def usage_from_log(path):
    if not path.is_file():
        return {"reported_tokens": None}
    text = path.read_text(encoding="utf-8")
    match = re.search(r"tokens used\s*\n([\d,]+)", text)
    model = re.search(r"^model: (.+)$", text, re.MULTILINE)
    return {"reported_tokens": int(match.group(1).replace(",", "")) if match else None,
            "reported_model": model.group(1) if model else None}


def made_progress(record):
    decision = record.get("decision", {})
    return bool(decision.get("metrics_ok")) and decision.get("keep") is not False


def run_session(c, raw_protocol):
    protocol = normalize_protocol(raw_protocol)
    session = Path(protocol["session_dir"]).resolve()
    session.mkdir(parents=True, exist_ok=True)
    with session_lock(session):
        state_path = session / "session_state.json"
        if state_path.exists():
            if c.read_json(session / "protocol.json") != protocol:
                raise ValueError("已有会话协议与配置不同，请恢复原配置或使用新session_dir")
            state = c.read_json(state_path)
        else:
            if (session / "protocol.json").exists():
                raise RuntimeError("旧版或未完成初始化的会话不能直接恢复，请使用新session_dir")
            c.write_json(session / "protocol.json", protocol)
            c.freeze_sources(session / "frozen")
            c.write_json(session / "source_hashes.json", c.source_hashes(session / "frozen"))
            harness_names = ("controller.py", "session.py", "baseline_cache.py", "agent.py", "decision.py", "feedback.py", "proposal.py", "PROMPT.md")
            harness = {
                name: c.sha256(c.ROOT / "sa0" / name) for name in harness_names
            }
            for name in getattr(c, "EXTRA_HARNESS_FILES", ()):
                harness[name] = c.sha256(c.ROOT / name)
            c.write_json(session / "harness_hashes.json", harness)
            data_hashes = c.dataset_hashes(protocol)
            c.write_json(session / "data_hashes.json", data_hashes)
            imported, signatures = import_history(c, protocol, session / "frozen", data_hashes)
            c.write_json(session / "imported_history.json", imported)
            (session / "instructions.md").write_text(
                getattr(c, "PROMPT_PATH", c.ROOT / "sa0/PROMPT.md").read_text(encoding="utf-8"), encoding="utf-8"
            )
            state = {"agent_calls": 0, "process_seconds": 0.0,
                     "seen_proposals": sorted(signatures), "stop_reason": None}
            atomic_json(state_path, state)
        contract_check(c, session, protocol)
        frozen = session / "frozen"
        baseline_folder = session / "baseline"
        if not baseline_folder.exists() and protocol["baseline_source_session"]:
            from sa0.baseline_cache import reuse_baseline
            print("核验并复用固定基线：", protocol["baseline_source_session"], flush=True)
            reuse_baseline(c, protocol, frozen, session, protocol["baseline_source_session"])
        if not baseline_folder.exists():
            baseline_folder.mkdir()
            print("开始训练固定基线，日志：", baseline_folder / "train.log", flush=True)
            try:
                baseline_protocol = {**protocol, "process_timeout_seconds": min(
                    protocol["process_timeout_seconds"], protocol["max_session_process_seconds"])}
                record = c.execute_training(baseline_protocol, frozen, baseline_folder, frozen,
                                            protocol["baseline_initial_lr"])
            except KeyboardInterrupt:
                c.write_json(baseline_folder / "record.json",
                             {"training_status": "interrupted", "comparison_status": "not_run"})
                raise
            c.write_json(baseline_folder / "record.json", record)
        if not (baseline_folder / "record.json").exists():
            raise RuntimeError("基线运行未归档，先确认子进程已停止；请使用新会话，不自动重训")
        baseline_record = c.read_json(baseline_folder / "record.json")
        if baseline_record["training_status"] != "completed":
            raise RuntimeError("基线未完成，请查看日志并使用新会话；不自动重复基线训练")
        baseline = baseline_record["result"]
        state["process_seconds"] = baseline_record["process_seconds"]
        history = []
        for number in range(1, protocol["max_trials"] + 1):
            folder = session / f"trial_{number:03d}"
            if not folder.exists():
                break
            if not (folder / "record.json").exists():
                # Crash recovery must not overlap a surviving worker.
                process_file = folder / "worker_process.json"
                if process_file.exists():
                    pid = c.read_json(process_file)["pid"]
                    try:
                        os.kill(pid, 0)
                    except ProcessLookupError:
                        pass
                    else:
                        raise RuntimeError(f"旧worker进程{pid}仍存在，请先确认其状态")
                phase = c.read_json(folder / "phase.json") if (folder / "phase.json").exists() else {}
                elapsed = protocol["process_timeout_seconds"] if phase.get("stage") == "training" else 0
                c.write_json(folder / "record.json", {
                    "training_status": "interrupted", "comparison_status": "not_run",
                    "reason": "未归档尝试不自动重跑；训练成本按保护上限保守计入",
                    "process_seconds": elapsed,
                    "trial_number": number, "trial_dir": str(folder),
                })
            record = c.read_json(folder / "record.json")
            history.append(history_item(record, folder))
            state["process_seconds"] += record.get("process_seconds", 0.0)
            if record.get("proposal"):
                try:
                    signature = proposal_signature(c, record["proposal"],
                        (frozen / "train/mavi.py").read_text(encoding="utf-8"))
                    if signature not in state["seen_proposals"]:
                        state["seen_proposals"].append(signature)
                except (ValueError, SyntaxError):
                    pass
        atomic_json(state_path, state)
        imported = c.read_json(session / "imported_history.json")
        source = (frozen / "train/mavi.py").read_text(encoding="utf-8")
        instructions = (session / "instructions.md").read_text(encoding="utf-8")
        failures = 0
        for item in reversed(history):
            if made_progress(item["record"]):
                break
            failures += 1
        for number in range(len(history) + 1, protocol["max_trials"] + 1):
            contract_check(c, session, protocol)
            remaining = protocol["max_session_process_seconds"] - state["process_seconds"]
            if remaining <= 0:
                state["stop_reason"] = "session_process_budget_exhausted"
                break
            if failures >= protocol["stop_after_failures"]:
                state["stop_reason"] = "consecutive_failures_limit"
                break
            replay = protocol.get("first_proposal_path") if number == 1 else None
            if not replay and state["agent_calls"] >= protocol["max_agent_calls"]:
                state["stop_reason"] = "agent_call_limit"
                break
            folder = session / f"trial_{number:03d}"
            folder.mkdir()
            feedback = {"protocol": protocol, "baseline": baseline,
                        "baseline_training_curve": curve_summary(baseline_folder / "artifacts/training.jsonl"),
                        "imported_history": imported, "history": history, "model_source": source}
            prompt = c.write_agent_input(instructions, feedback, c.build_request(protocol), folder / "agent_input.md")
            atomic_json(folder / "phase.json", {"stage": "proposal"})
            proposal = None
            record = {}
            training_started = None
            try:
                if replay:
                    path = Path(replay)
                    response = path.read_text(encoding="utf-8")
                    c.write_json(folder / "proposal_origin.json", {"source": "existing_proposal",
                                 "path": str(path), "sha256": c.sha256(path)})
                else:
                    # Reserve a call before invoking: interruption cannot cause a free retry.
                    state["agent_calls"] += 1
                    atomic_json(state_path, state)
                    print(f"第{number}轮：agent选择修改方向", flush=True)
                    response = c.call_agent(prompt.read_text(encoding="utf-8"), folder / "agent_cli.log",
                                            settings=protocol["agent"])
                (folder / "agent_response.txt").write_text(response, encoding="utf-8")
                proposal = c.validate_proposal(json.loads(response))
                c.write_json(folder / "proposal.json", proposal)
                if proposal["kind"] not in protocol["allowed_kinds"]:
                    raise ValueError("提案类型不在允许范围内")
                signature = proposal_signature(c, proposal, source)
                if signature == "model:" + model_signature(source):
                    raise ValueError("候选最终源码没有改变基线模型")
                if signature in state["seen_proposals"] and not replay:
                    raise ValueError("候选在当前或导入历史中已尝试，不重复训练")
                initial_lr = protocol["baseline_initial_lr"]
                if proposal["kind"] == "learning_rate":
                    initial_lr = proposal["change"]["initial_lr"]
                    if initial_lr > protocol["initial_lr_upper_bound"] or initial_lr == protocol["baseline_initial_lr"]:
                        raise ValueError("候选学习率越界或与基线相同")
                workspace = folder / "workspace"
                shutil.copytree(frozen, workspace)
                if proposal["kind"] == "model":
                    (workspace / "train/mavi.py").write_text(
                        c.build_candidate_source(source, proposal), encoding="utf-8"
                    )
                if signature not in state["seen_proposals"]:
                    state["seen_proposals"].append(signature)
                atomic_json(state_path, state)
                atomic_json(folder / "phase.json", {"stage": "training", "started_at": time.time()})
                actual_protocol = {**protocol, "process_timeout_seconds": min(remaining, protocol["process_timeout_seconds"])}
                print(f"第{number}轮：{proposal['kind']}，训练日志：{folder / 'train.log'}", flush=True)
                training_started = time.perf_counter()
                record = c.execute_training(actual_protocol, frozen, folder, workspace, initial_lr)
                if record["training_status"] == "completed":
                    try:
                        record["decision"] = c.compare_results(baseline, record["result"], protocol)
                    except (KeyError, ValueError, TypeError) as error:
                        record.update(comparison_status="invalid", reason=str(error))
                    else:
                        record["comparison_status"] = "valid"
            except KeyboardInterrupt:
                record.update(training_status="interrupted", comparison_status="not_run")
                if training_started is not None:
                    record["process_seconds"] = time.perf_counter() - training_started
                if proposal is not None:
                    record["proposal"] = proposal
                record.update(trial_number=number, trial_dir=str(folder))
                c.write_json(folder / "record.json", record)
                state["stop_reason"] = "user_interrupted"
                atomic_json(state_path, state)
                raise
            except Exception as error:
                record.update(training_status="not_completed", comparison_status="not_run",
                              reason=f"{type(error).__name__}: {error}")
                if training_started is not None:
                    record["process_seconds"] = time.perf_counter() - training_started
            if proposal is not None:
                record["proposal"] = proposal
            record.update(trial_number=number, trial_dir=str(folder),
                          agent_usage=({"reported_tokens": 0, "source": "replayed_proposal"}
                                       if replay else usage_from_log(folder / "agent_cli.log")))
            if hasattr(c, "trial_metadata"):
                record.update(c.trial_metadata(folder))
            try:
                contract_check(c, session, protocol)
            except RuntimeError as error:
                record.update(comparison_status="invalid", reason=str(error))
                record.pop("decision", None)
                state["stop_reason"] = "experiment_contract_changed"
            c.write_json(folder / "record.json", record)
            atomic_json(folder / "phase.json", {"stage": "archived"})
            history.append(history_item(record, folder))
            state["process_seconds"] += record.get("process_seconds", 0.0)
            failures = 0 if made_progress(record) else failures + 1
            atomic_json(state_path, state)
            print({"trial": number, "status": record["training_status"],
                   "decision": record.get("decision"), "reason": record.get("reason")}, flush=True)
            if state["stop_reason"] == "experiment_contract_changed":
                break
        if len(history) >= protocol["max_trials"] and state["stop_reason"] != "experiment_contract_changed":
            state["stop_reason"] = "max_trials_reached"
        retained = [x["record"]["trial_dir"] for x in history
                    if x["record"].get("decision", {}).get("keep") is True]
        promising = [x["record"]["trial_dir"] for x in history
                     if x["record"].get("decision", {}).get("metrics_ok")
                     and x["record"].get("decision", {}).get("keep") is None]
        tokens = [x["record"].get("agent_usage", {}).get("reported_tokens") for x in history]
        atomic_json(session / "summary.json", {
            "baseline": baseline, "history": history, "budget_status": protocol["budget_status"],
            "stop_reason": state["stop_reason"], "agent_calls": state["agent_calls"],
            "process_seconds": state["process_seconds"],
            "baseline_reused": baseline_record.get("baseline_reused", False),
            "baseline_reuse_origin": baseline_record.get("reuse_origin"),
            "actual_current_process_seconds": (
                state["process_seconds"] - baseline_record["process_seconds"]
                + baseline_record.get("actual_reuse_seconds", 0.0)
                if baseline_record.get("baseline_reused") else state["process_seconds"]
            ),
            "reported_tokens_known_total": sum(x for x in tokens if x is not None),
            "unknown_token_records": sum(x is None for x in tokens),
            "retained_candidates": retained, "promising_candidates": promising,
        })
        atomic_json(state_path, state)
        print("会话归档：", session / "summary.json", flush=True)
