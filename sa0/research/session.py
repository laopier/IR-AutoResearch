"""Recoverable hypothesis-driven local workflow; all budgets are attempt counts."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

from sa0.session import atomic_json, session_lock
from sa0.research import protocol, lineage, agent


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def execute_job(c, config, folder, workspace):
    if config.get("device", "cuda") == "cuda":
        gpu_lock = c.ROOT / "results/.research_v2_gpu_lock"
        gpu_lock.mkdir(parents=True, exist_ok=True)
        with session_lock(gpu_lock):
            return _execute_job(c, config, folder, workspace)
    return _execute_job(c, config, folder, workspace)


def _execute_job(c, config, folder, workspace):
    c.write_json(folder / "experiment.json", config)
    env = {**os.environ, "IR_SA0_CONFIG": str(folder / "experiment.json")}
    started = time.perf_counter()
    with (folder / "train.log").open("x", encoding="utf-8") as log:
        try:
            with subprocess.Popen([sys.executable, "-u", str(workspace / "worker.py")], cwd=workspace,
                                  env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                c.write_json(folder / "worker_process.json", {"pid": process.pid})
                try:
                    process.wait(timeout=config["worker_timeout_seconds"])
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
        except subprocess.TimeoutExpired:
            record = {"status": "timed_out", "reason": "单任务保护超时"}
        else:
            record = {"status": "completed" if process.returncode == 0 else "failed", "returncode": process.returncode}
    record["process_seconds"] = time.perf_counter() - started
    record["gpu_accounting"] = "local_worker_wall_proxy"  # no scheduler-allocation claim
    if record["status"] == "completed":
        for name in ("result.json", "checkpoint.pt", "training.jsonl"):
            if not (Path(config["out_dir"]) / name).is_file():
                record.update(status="invalid", reason=f"missing {name}")
                return record
        record["result"] = c.read_json(Path(config["out_dir"]) / "result.json")
    else:
        text = (folder / "train.log").read_text(encoding="utf-8", errors="replace")
        record["error_excerpt"] = text[-5000:]
        record["reason"] = record.get("reason") or (text.strip().splitlines()[-1] if text.strip() else "worker failed")
    return record


class Workflow:
    def __init__(self, c, config, retry_failed_full=False):
        self.c, self.p = c, protocol.validate(config)
        self.retry_failed_full = retry_failed_full
        self.root = Path(self.p["session_dir"]).resolve()
        self.state_path = self.root / "state.json"

    def save(self):
        atomic_json(self.state_path, self.s)

    def event(self, kind, **fields):
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"time": time.time(), "kind": kind, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()

    def initialize(self):
        audit_path = self.p.get("dataset_report")
        if audit_path:
            from prepare.pilot_ready import verify
            verify(self.p["data_root"], audit_path, self.p["stages"].values())
        self.root.mkdir(parents=True, exist_ok=True)
        if self.state_path.exists():
            if self.c.read_json(self.root / "protocol.json") != self.p:
                raise ValueError("v2协议变化，请使用新会话")
            self.s = self.c.read_json(self.state_path)
            self.check_contract()
            for call in self.s["calls"]:
                if call["status"] == "reserved":
                    record = Path(call["folder"]) / "record.json"
                    if record.exists():
                        call["cli"] = self.c.read_json(record)
                        if "seconds" in call["cli"]:
                            call["seconds"] = call["cli"]["seconds"]
                    call.update(status="interrupted", timing_uncertain="seconds" not in call)
            for task in self.s["tasks"]:
                if task["status"] == "running":
                    file = Path(task["folder"]) / "worker_process.json"
                    if file.exists():
                        try:
                            os.kill(self.c.read_json(file)["pid"], 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise RuntimeError("旧训练进程仍存在，拒绝并发恢复")
                    task.update(status="interrupted", reason="未完成任务保留次数，不免费重跑",
                                accounting_uncertain=True)
                    atomic_json(Path(task["folder"]) / "record.json", task)
            self.save()
            return
        if (self.root / "protocol.json").exists():
            raise RuntimeError("初始化未完成或旧版目录，请另开会话")
        self.c.write_json(self.root / "protocol.json", self.p)
        base = self.root / "frozen"
        self.c.freeze_sources(base)
        shutil.copy2(self.c.ROOT / "sa0/research/worker.py", base / "worker.py")
        (base / "train/feature_transform.py").write_text("def transform(feature):\n    return feature\n", encoding="utf-8")
        hashes = {}
        for stage in self.p["stages"].values():
            hashes.update(self.c.dataset_hashes({**self.p, **stage}))
        self.c.write_json(self.root / "data_hashes.json", hashes)
        files = ["sa0/controller.py", "sa0/research/PROMPT.md", "sa1/proposal.py", "sa1/agent.py", "sa0/session.py", "prepare/pilot_ready.py"]
        files += [path.relative_to(self.c.ROOT).as_posix() for path in (self.c.ROOT / "sa0/research").glob("*.py")]
        self.c.write_json(self.root / "harness_hashes.json", {name: self.c.sha256(self.c.ROOT / name) for name in files})
        self.c.write_json(self.root / "source_hashes.json", self.c.source_hashes(base))
        self.instructions = (self.c.ROOT / "sa0/research/PROMPT.md").read_text(encoding="utf-8")
        (self.root / "instructions.md").write_text(self.instructions, encoding="utf-8")
        self.s = {"phase": "planning", "hypotheses": {}, "candidates": {
            "B0": {"candidate_id": "B0", "parent_candidate_ids": [], "hypothesis_ids": [],
                   "recipe": {"initial_lr": self.p["baseline_initial_lr"]}, "workspace": str(base),
                   "source_hashes": self.c.source_hashes(base)}},
            "tasks": [], "calls": [], "interpretations": {}, "action_errors": 0,
            "started_at": time.time(), "stop_reason": None, "finalist": None}
        self.save()

    def load_shared_baselines(self):
        cache = self.p.get("baseline_cache_dir")
        if not cache:
            return
        from sa0.research import baseline_cache
        env = baseline_cache.runtime_environment(self.p.get("device", "cuda"), 2)
        spec = {**self.p, **self.p["stages"]["full"]}
        imported = []
        for seed in self.p["confirmation_seeds"]:
            folder, receipt, result = baseline_cache.read(self.c, spec, Path(self.s["candidates"]["B0"]["workspace"]), seed, env)
            imported.append({"task_id": f"SHARED_B0_SEED{seed}", "candidate_id": "B0", "stage": "full", "seed": seed,
                "phase": "shared", "budget_bucket": "shared", "status": "completed", "scientific_valid": True,
                "signature": self.signature("B0", "full", seed), "folder": str(folder),
                "artifact_hashes": receipt["artifact_hashes"], "result": result,
                "process_seconds": 0., "shared_cost_seconds": result["training_seconds"],
                "receipt_sha256": self.c.sha256(folder / "receipt.json"), "recipe": result["recipe"]})
        previous = self.s.get("external_baselines")
        if previous is not None and previous != imported:
            raise ValueError("已导入共享B0发生变化，禁止继续原会话")
        self.s["external_baselines"] = imported
        self.save()

    def check_contract(self):
        hashes = {}
        for stage in self.p["stages"].values():
            hashes.update(self.c.dataset_hashes({**self.p, **stage}))
        if hashes != self.c.read_json(self.root / "data_hashes.json"):
            raise RuntimeError("数据或阶段清单变化")
        expected = self.c.read_json(self.root / "harness_hashes.json")
        if {name: self.c.sha256(self.c.ROOT / name) for name in expected} != expected:
            raise RuntimeError("执行代码变化，不能续跑v2旧会话")
        for node in self.s["candidates"].values():
            if self.c.source_hashes(Path(node["workspace"])) != node["source_hashes"]:
                raise RuntimeError("候选源码发生未登记修改")
        for baseline in self.s.get("external_baselines", []):
            folder = Path(baseline["folder"])
            if self.c.sha256(folder / "receipt.json") != baseline["receipt_sha256"]:
                raise RuntimeError("共享基线审计发生变化")
            for name, value in baseline["artifact_hashes"].items():
                if self.c.sha256(folder / "artifacts" / name) != value:
                    raise RuntimeError("共享基线产物发生变化")
        lineage.check_tree(self.s["candidates"])

    def ask(self, purpose, request):
        self.check_contract()
        folder = self.root / "calls" / f"call_{len(self.s['calls']) + 1:03d}"
        folder.parent.mkdir(exist_ok=True)
        row = {"call_id": folder.name, "purpose": purpose, "folder": str(folder), "status": "reserved", "started_at": time.time()}
        self.s["calls"].append(row)
        self.save()
        # Hidden results/labels never enter state or this feedback envelope.
        context = {"mode": self.p["mode"], "condition": self.p["condition"],
                   "stage_specs": self.p["stages"], "task_limits": {"total": self.p["max_gpu_tasks"],
                       "exploration": self.p["max_exploration_tasks"], "screening": self.p.get("max_screening_tasks"),
                       "full_reserve": self.p.get("full_task_reserve"), "max_full_candidates": self.p.get("max_full_candidates")},
                   "remaining_exploration_tasks": self.p["max_exploration_tasks"] - self.count("exploration"),
                   "remaining_gpu_tasks": self.p["max_gpu_tasks"] - len(self.s["tasks"]),
                   "phase": self.s["phase"], "eligible_full_candidates": self.eligible_full(),
                   "remaining_screening_tasks": self.screening_remaining(),
                   "full_task_reserve": self.p.get("full_task_reserve"),
                   "hypotheses": self.s["hypotheses"], "candidates": self.s["candidates"],
                   "last_action_error": self.s.get("last_action_error"),
                   "experiments": self.s["tasks"], "shared_baselines": self.s.get("external_baselines", []), "interpretations": self.s["interpretations"],
                   "prior_research": [self.c.read_json(Path(call["folder"]) / "research.json")
                                      for call in self.s["calls"] if (Path(call["folder"]) / "research.json").is_file()],
                   "source": {key: {p: (Path(n["workspace"]) / p).read_text(encoding="utf-8") for p in sorted(lineage.ALLOWED_FILES)}
                              for key, n in self.s["candidates"].items()}}
        text = (self.root / "instructions.md").read_text(encoding="utf-8") + "\n反馈：\n" + json.dumps(context, ensure_ascii=False, allow_nan=False) + "\n请求：\n" + request
        if self.p["condition"] == "SA1":
            text += "\n可自主检索；返回{payload:本轮所需对象,research:{status,summary,sources}}。sources含url/title/used/reason；不得虚构来源。"
        else:
            text += "\n离线，不检索；直接返回本轮所需JSON对象。"
        started = time.perf_counter()
        try:
            answer = agent.call(text, folder, self.p["agent"])
            row["status"] = "completed"
            return answer
        except BaseException as error:
            row.update(status="failed", reason=str(error))
            raise
        finally:
            row["seconds"] = time.perf_counter() - started
            if (folder / "record.json").exists():
                row["cli"] = self.c.read_json(folder / "record.json")
            self.save()

    def count(self, phase):
        return sum(t["phase"] == phase for t in self.s["tasks"])

    def bucket_count(self, bucket):
        return sum(t.get("budget_bucket") == bucket for t in self.s["tasks"])

    def screening_remaining(self):
        if not self.p["full_selection_enabled"]:
            return max(0, self.p["max_exploration_tasks"] - self.count("exploration"))
        return max(0, min(self.p["max_screening_tasks"] - self.bucket_count("screening"),
                          self.p["max_exploration_tasks"] - self.count("exploration") - self.p["full_task_reserve"]))

    def eligible_full(self):
        return [key for key in self.s["candidates"] if key != "B0" and self.can_promote(key)]

    def enter_full_selection(self, reason):
        self.s.update(phase="full_selection", screening_end_reason=reason)
        self.event("screening_closed", reason=reason, eligible=self.eligible_full())
        self.save()

    def signature(self, key, stage, seed):
        node = self.s["candidates"][key]
        spec = {**self.p["stages"][stage], "seed": seed, "batch_size": self.p["batch_size"],
                "val_batch_size": self.p["val_batch_size"], "lr_horizon_steps": self.p["lr_horizon_steps"],
                "device": self.p.get("device", "cuda")}
        return digest({"candidate_id": key, "source": node["source_hashes"], "recipe": node["recipe"], "spec": spec})

    def job(self, key, stage, seed):
        if stage not in protocol.STAGES or key not in self.s["candidates"]:
            raise ValueError("未知阶段或候选")
        if self.p["full_selection_enabled"]:
            if key != "B0" and stage == "full" and self.s["phase"] == "exploration":
                raise ValueError("screening不得直接提交候选full")
            if self.s["phase"] == "full_selection" and (
                stage != "full" or (key != "B0" and key not in self.s.get("full_selection_plan", {}).get("candidate_ids", []))
            ):
                raise ValueError("full-selection仅提交锁定名单的full及匹配基线")
        phase = "finalization" if self.s["phase"] == "finalization" else "exploration"
        bucket = ("confirmation" if phase == "finalization" else "full_selection" if self.s["phase"] == "full_selection"
                  else "baseline_init" if self.s["phase"] == "planning" and key == "B0" and stage == "full" else "screening")
        sig = self.signature(key, stage, seed)
        for old in self.s["tasks"] + self.s.get("external_baselines", []):
            if old["signature"] == sig and old["status"] == "completed":
                for name, value in old["artifact_hashes"].items():
                    if self.c.sha256(Path(old["folder"]) / "artifacts" / name) != value:
                        raise RuntimeError("缓存训练产物被修改")
                self.event("task_reused", task_id=old["task_id"], requested_stage=stage)
                return old
        if len(self.s["tasks"]) >= self.p["max_gpu_tasks"] or (phase == "exploration" and self.count(phase) >= self.p["max_exploration_tasks"]):
            raise ValueError("任务尝试额度不足")
        if self.p["full_selection_enabled"] and bucket == "screening" and self.screening_remaining() <= 0:
            raise ValueError("screening已结束，保留full额度，禁止新low/smoke")
        if self.p["full_selection_enabled"] and bucket == "full_selection" and self.bucket_count(bucket) >= self.p["full_task_reserve"]:
            raise ValueError("full任务储备已用尽")
        node = self.s["candidates"][key]
        folder = self.root / "tasks" / f"T{len(self.s['tasks']) + 1:03d}"
        folder.mkdir(parents=True)
        row = {"task_id": folder.name, "candidate_id": key, "stage": stage, "seed": seed,
               "phase": phase, "budget_bucket": bucket, "signature": sig, "status": "running", "folder": str(folder),
               "recipe": node["recipe"], "started_at": time.time()}
        self.s["tasks"].append(row)
        self.save()  # reservation before process launch: failures always consume one attempt
        workspace = folder / "workspace"
        shutil.copytree(Path(node["workspace"]), workspace)
        changed_scheduler = (workspace / "train/experiment.py").read_bytes() != (self.root / "frozen/train/experiment.py").read_bytes()
        config = {**self.p["stages"][stage], "data_root": self.p["data_root"], "seed": seed,
                  "batch_size": self.p["batch_size"], "val_batch_size": self.p["val_batch_size"],
                  "lr_horizon_steps": self.p["lr_horizon_steps"], "recipe": node["recipe"],
                  "workspace": str(workspace), "out_dir": str(folder / "artifacts"),
                  "worker_timeout_seconds": self.p["worker_timeout_seconds"], "device": self.p.get("device", "cuda"),
                  "custom_scheduler": changed_scheduler and scheduler_changed(self.root / "frozen/train/experiment.py", workspace / "train/experiment.py")}
        before = self.c.source_hashes(workspace)
        print(f"{row['task_id']} {key} {stage} seed={seed}，日志：{folder / 'train.log'}", flush=True)
        started = time.perf_counter()
        try:
            row.update(execute_job(self.c, config, folder, workspace))
            if self.c.source_hashes(workspace) != before:
                row.update(status="invalid", reason="运行期间修改源码")
            if row["status"] == "completed":
                result = row["result"]
                for field in ("steps", "seed", "batch_size", "val_batch_size", "lr_horizon_steps"):
                    if result[field] != config[field]:
                        raise ValueError(f"结果条件不一致：{field}")
                if result.get("initialized_from") != "fresh":
                    raise ValueError("不允许checkpoint续训")
                for value in result["final_metrics"].values():
                    if type(value) not in (int, float) or not math.isfinite(value):
                        raise ValueError("指标非有限值")
                for metric in protocol.METRICS:
                    result["final_metrics"][metric]
                row["artifact_hashes"] = {name: self.c.sha256(folder / "artifacts" / name)
                                          for name in ("result.json", "checkpoint.pt", "training.jsonl")}
                row["scientific_valid"] = stage != "smoke"
                if stage == "smoke":
                    atomic_json(folder / "interpretation.json", {"labels": ["engineering_progress"],
                        "scientific_conclusion": None, "facts": "工程检查通过，不用于指标或机制结论"})
        except KeyboardInterrupt:
            row.update(status="interrupted", scientific_valid=False, reason="人工中断")
            raise
        except Exception as error:
            row.update(status="failed", scientific_valid=False, reason=f"{type(error).__name__}: {error}")
        finally:
            row.setdefault("process_seconds", time.perf_counter() - started)
            row.setdefault("gpu_accounting", "local_worker_wall_proxy")
            atomic_json(folder / "record.json", row)
            self.event("task_finished", task_id=row["task_id"], status=row["status"])
            self.save()
        return row

    def compare(self, candidate, baseline):
        if not candidate.get("scientific_valid") or not baseline.get("scientific_valid"):
            raise ValueError("工程检查或失败任务不能科学比较")
        a, b = candidate["result"], baseline["result"]
        for field in ("steps", "seed", "batch_size", "val_batch_size", "lr_horizon_steps", "environment"):
            if a[field] != b[field]:
                raise ValueError(f"不匹配的对照：{field}")
        if self.p["stages"][candidate["stage"]] != self.p["stages"][baseline["stage"]]:
            raise ValueError("不同fidelity不能直接比较")
        return protocol.improved(b["final_metrics"], a["final_metrics"])

    def mechanistic(self, row):
        test = self.s["candidates"][row["candidate_id"]].get("mechanism_test")
        if not test or not row.get("scientific_valid"):
            return False
        for other in self.s["tasks"]:
            if other["candidate_id"] != test["reference_candidate_id"] or other["stage"] != row["stage"] or other["seed"] != row["seed"] or not other.get("scientific_valid"):
                continue
            self.compare(row, other)  # verifies matching comparison; not necessarily an improvement
            delta = row["result"]["final_metrics"][test["metric"]] - other["result"]["final_metrics"][test["metric"]]
            return delta > test["min_delta"] if test["direction"] == "increase" else delta < -test["min_delta"]
        return False

    def interpret(self, row):
        if row["task_id"] in self.s["interpretations"] or row["candidate_id"] == "B0" or not row.get("scientific_valid"):
            return
        answer = self.ask("interpretation", f"解释实验{row['task_id']}。返回{{facts:文字,inference:文字,caveats:文字,next_action:continue/reconsider/finalize}}。观察与推测分开，不自行宣布晋级。")
        if not isinstance(answer, dict) or set(answer) != {"facts", "inference", "caveats", "next_action"} or any(not isinstance(v, str) or not v.strip() for v in answer.values()):
            raise ValueError("实验解释字段无效")
        if answer["next_action"] not in {"continue", "reconsider", "finalize"}:
            raise ValueError("解释动作无效")
        metric = bool(row.get("metrics_ok"))
        mechanism = self.mechanistic(row)
        labels = (["metric_progress"] if metric else []) + (["mechanistic_progress"] if mechanism else [])
        item = {**answer, "task_id": row["task_id"], "labels": labels or ["no_progress"],
                "observed_metrics": row["result"]["final_metrics"], "evidence_task_ids": [row["task_id"], row["baseline_task_id"]],
                "mechanism_evidence_scope": "predeclared matched test; preliminary support, not causal proof" if mechanism else None}
        if mechanism:
            reference = self.s["candidates"][row["candidate_id"]]["mechanism_test"]["reference_candidate_id"]
            evidence = next(t for t in self.s["tasks"] if t["candidate_id"] == reference and t["stage"] == row["stage"] and t["seed"] == row["seed"] and t.get("scientific_valid"))
            item["evidence_task_ids"].append(evidence["task_id"])
        self.s["interpretations"][row["task_id"]] = item
        atomic_json(Path(row["folder"]) / "interpretation.json", item)
        for h in self.s["candidates"][row["candidate_id"]]["hypothesis_ids"]:
            node = self.s["hypotheses"][h]
            node["no_progress_streak"] = 0 if labels else node["no_progress_streak"] + 1
            if node["no_progress_streak"] >= 3:
                node["status"] = "needs_reconsideration"
        self.save()

    def can_promote(self, key):
        wins = [t for t in self.s["tasks"] if t["candidate_id"] == key and t["stage"] == "low" and t.get("metrics_ok") and t.get("scientific_valid") and t["status"] == "completed"]
        if len({t["seed"] for t in wins}) >= 2:
            return True
        return bool(wins) and any(t.get("scientific_valid") and "mechanistic_progress" in self.s["interpretations"].get(t["task_id"], {}).get("labels", [])
                                  and self.s["candidates"][t["candidate_id"]].get("mechanism_test", {}).get("reference_candidate_id") == key
                                  for t in self.s["tasks"] if self.s["candidates"][t["candidate_id"]].get("mechanism_test"))

    def experiment(self, key, stage, seed):
        if self.p["full_selection_enabled"]:
            if stage == "full" and self.s["phase"] != "full_selection":
                raise ValueError("full须先进入full-selection，不与普通low探索混跑")
            if stage != "full" and self.s["phase"] == "full_selection":
                raise ValueError("full-selection阶段禁止新low/smoke")
        if key == "B0" or stage not in {"smoke", "low", "full"}:
            raise ValueError("探索实验须为候选smoke/low/full")
        allowed = self.p["low_seeds"] if stage != "full" else [0]
        if type(seed) is not int or seed not in allowed:
            raise ValueError("阶段seed未预先授权")
        if key not in self.s["candidates"]:
            raise ValueError("候选不存在")
        if (stage != "full" or not self.p["full_selection_enabled"]) and any(self.s["hypotheses"][h]["status"] != "open" for h in self.s["candidates"][key]["hypothesis_ids"]):
            raise ValueError("方向须先重新评估")
        if stage == "full" and not self.can_promote(key):
            raise ValueError("尚未满足两seed改善或改善加匹配关键消融的晋级证据")
        baseline = None
        if stage != "smoke":
            # Reserve enough slots for both matched baseline and candidate before either launches.
            needed = sum(not any(t["signature"] == self.signature(k, stage, seed) and t["status"] == "completed" for t in self.s["tasks"] + self.s.get("external_baselines", [])) for k in ("B0", key))
            if needed > min(self.p["max_exploration_tasks"] - self.count("exploration"), self.p["max_gpu_tasks"] - len(self.s["tasks"])):
                raise ValueError("匹配对照与候选的剩余额度不足")
            if self.p["full_selection_enabled"] and stage == "low" and needed > self.screening_remaining():
                raise ValueError("screening余量不足以完成匹配基线和候选，须进入full-selection")
            baseline = self.job("B0", stage, seed)
            if baseline["status"] != "completed":
                raise RuntimeError("匹配基线失败，停止本次候选提交")
        row = self.job(key, stage, seed)
        if row["status"] == "completed" and baseline:
            try:
                row.update(metrics_ok=self.compare(row, baseline), baseline_task_id=baseline["task_id"],
                           decision={"keep": None, "budget_status": "workflow_validation"})
            except ValueError as error:
                row.update(scientific_valid=False, comparison_status="invalid", reason=str(error))
                atomic_json(Path(row["folder"]) / "record.json", row)
                self.save()
                raise
            atomic_json(Path(row["folder"]) / "record.json", row)
            self.save()
            self.interpret(row)
        return row

    def action(self, answer):
        if not isinstance(answer, dict):
            raise ValueError("动作必须为对象")
        action = answer.get("action")
        if self.p["full_selection_enabled"] and self.s["phase"] == "full_selection":
            raise ValueError("筛选已锁定，不能继续创建/修改候选或low实验")
        if action == "candidate" and set(answer) == {"action", "candidate"}:
            key = answer["candidate"].get("candidate_id")
            destination = self.root / "candidates" / str(key)
            destination.parent.mkdir(exist_ok=True)
            # Validate the path before using the ID as a folder name.
            if not isinstance(key, str) or not key.startswith("C") or not key[1:].isdigit():
                raise ValueError("candidate_id须为C数字")
            with tempfile.TemporaryDirectory(prefix=".candidate_", dir=destination.parent) as temporary:
                Path(temporary).resolve().relative_to(self.root)
                staged = Path(temporary) / "source"
                node = lineage.materialize(self.c, self.p, self.s, answer["candidate"], staged)
                staged.rename(destination)
                node["workspace"] = str(destination)
            self.s["candidates"][key] = node
            self.c.write_json(destination / "candidate.json", node)
            # Metadata is outside hashed source files (*.py).
        elif action == "experiment" and set(answer) == {"action", "candidate_id", "stage", "seed"}:
            if self.p["full_selection_enabled"] and answer["stage"] == "full":
                if answer["seed"] != 0 or not self.can_promote(answer["candidate_id"]):
                    raise ValueError("提前晋级请求缺少证据或seed错误")
                self.enter_full_selection("agent requested early full selection")
            else:
                if self.p["full_selection_enabled"]:
                    keys = ("B0", answer["candidate_id"]) if answer["stage"] == "low" else (answer["candidate_id"],)
                    required = sum(not any(t["signature"] == self.signature(k, answer["stage"], answer["seed"]) and t["status"] == "completed"
                                           for t in self.s["tasks"]) for k in keys)
                    if required > self.screening_remaining():
                        self.enter_full_selection("next screen cannot fit matched baseline and candidate")
                        return
                self.experiment(answer["candidate_id"], answer["stage"], answer["seed"])
        elif action == "reconsider" and set(answer) == {"action", "hypothesis_id", "decision", "reason", "evidence_task_ids"}:
            h = self.s["hypotheses"].get(answer["hypothesis_id"])
            if not h or answer["decision"] not in {"continue", "close"} or not answer["reason"].strip():
                raise ValueError("重新评估动作无效")
            evidence = answer["evidence_task_ids"]
            if not isinstance(evidence, list) or not evidence or not set(evidence) <= {t["task_id"] for t in self.s["tasks"] if t.get("scientific_valid")}:
                raise ValueError("需引用有效实验证据")
            if h["status"] == "closed" and answer["decision"] == "continue" and not any(
                t["task_id"] in evidence and i >= h.get("closed_after_task_count", len(self.s["tasks"]))
                for i, t in enumerate(self.s["tasks"])
            ):
                raise ValueError("重新打开已关闭方向须引用关闭后出现的新证据")
            h.update(status="open" if answer["decision"] == "continue" else "closed", no_progress_streak=0)
            if answer["decision"] == "close":
                h["closed_after_task_count"] = len(self.s["tasks"])
            h.setdefault("reviews", []).append(answer)
        elif action == "add_hypothesis" and set(answer) == {"action", "hypothesis", "evidence_task_ids", "novelty_reason"}:
            if len(self.s["hypotheses"]) >= 5:
                raise ValueError("当前初始3个加最多2个新方向")
            h = protocol.hypothesis(answer["hypothesis"])
            if h["hypothesis_id"] in self.s["hypotheses"] or not answer["novelty_reason"].strip():
                raise ValueError("新方向须新ID及与原方向不同的理由")
            if any((h["mechanism"].strip().casefold(), h["proposed_change"].strip().casefold()) ==
                   (old["mechanism"].strip().casefold(), old["proposed_change"].strip().casefold())
                   for old in self.s["hypotheses"].values()):
                raise ValueError("新假设不能仅重新命名原机制")
            if not answer["evidence_task_ids"] or not set(answer["evidence_task_ids"]) <= set(self.s["interpretations"]):
                raise ValueError("新假设须引用已分析的科学实验")
            h.update(evidence_task_ids=answer["evidence_task_ids"], novelty_reason=answer["novelty_reason"])
            self.s["hypotheses"][h["hypothesis_id"]] = h
        elif action == "finalize" and set(answer) == {"action", "reason", "untested_hypotheses"}:
            if not isinstance(answer["reason"], str) or not answer["reason"].strip():
                raise ValueError("结束探索须说明理由")
            explanations = answer["untested_hypotheses"]
            exercised = {h for t in self.s["tasks"] if t.get("scientific_valid") and t["candidate_id"] != "B0"
                         for h in self.s["candidates"][t["candidate_id"]]["hypothesis_ids"]}
            untested = set(self.s["hypotheses"]) - exercised
            if not isinstance(explanations, dict) or not untested <= set(explanations) or set(explanations) - set(self.s["hypotheses"]) or any(not isinstance(v, str) or not v.strip() for v in explanations.values()):
                raise ValueError("未开展有效科学实验的假设必须说明暂不研究理由")
            self.s.update(phase="finalization", selection_reason=answer["reason"],
                          untested_hypotheses=answer["untested_hypotheses"])
            if self.p["full_selection_enabled"]:
                self.enter_full_selection("agent requested end of screening")
        else:
            raise ValueError("动作schema未授权")
        self.save()

    def full_selection(self):
        eligible = self.eligible_full()
        if not eligible:
            self.s.update(phase="finalization", full_selection_outcome="no_eligible_candidates")
            self.save()
            return
        if not self.s.get("full_selection_plan"):
            path = self.root / "full_selection_plan.json"
            answer = self.c.read_json(path) if path.exists() else self.ask("full_selection", f"screening已关闭，禁止创建候选或提交low。请从{eligible}选择1～{self.p['max_full_candidates']}个候选。只返回{{candidate_ids:[编号],reason:选择理由}}。无须用满名额。")
            if not isinstance(answer, dict) or set(answer) != {"candidate_ids", "reason"}:
                raise ValueError("full选择字段无效")
            ids = answer["candidate_ids"]
            if not isinstance(ids, list) or not 1 <= len(ids) <= self.p["max_full_candidates"] or len(set(ids)) != len(ids) or any(key not in eligible for key in ids) or not isinstance(answer["reason"], str) or not answer["reason"].strip():
                raise ValueError("full必须选择1～2个不同且可晋级的候选")
            self.s["full_selection_plan"] = answer
            if not path.exists():
                self.c.write_json(path, answer)
            self.save()
        completed = []
        for key in self.s["full_selection_plan"]["candidate_ids"]:
            # Archived failures are not retried implicitly. Resume requires explicit retry permission.
            prior = [t for t in self.s["tasks"] if t["candidate_id"] == key and t["stage"] == "full"]
            if prior and not any(t.get("scientific_valid") and t["status"] == "completed" for t in prior):
                if not self.retry_failed_full:
                    continue
            row = self.experiment(key, "full", 0)
            self.s.setdefault("full_retry_authorized", {}).pop(key, None)
            if row["status"] == "completed" and row.get("scientific_valid"):
                completed.append(key)
            self.save()
        if not completed:
            self.s["full_selection_outcome"] = "engineering_failure_no_valid_full"
            self.save()
            raise RuntimeError("可晋级候选full均未有效完成，暂停，不冒充完成；下次需明确授权重试")
        self.s.update(phase="finalization", full_selection_outcome="completed", completed_full_candidates=completed)
        self.save()

    def finalize(self):
        if self.p["full_selection_enabled"] and self.eligible_full() and not any(
            t["candidate_id"] != "B0" and t["stage"] == "full" and t.get("scientific_valid") and t["status"] == "completed"
            for t in self.s["tasks"]
        ):
            raise ValueError("存在可晋级候选但没有有效full，禁止直接finalize")
        self.s["phase"] = "finalization"
        if "untested_hypotheses" not in self.s:
            exercised = {h for t in self.s["tasks"] if t.get("scientific_valid") and t["candidate_id"] != "B0"
                         for h in self.s["candidates"][t["candidate_id"]]["hypothesis_ids"]}
            self.s["untested_hypotheses"] = {h: f"尚未获得有效科学实验，控制器因{self.s.get('stop_reason') or 'finalization'}结束探索；详见最终分析"
                                            for h in self.s["hypotheses"] if h not in exercised}
        full = [t for t in self.s["tasks"] if t["stage"] == "full" and t.get("metrics_ok") and t["candidate_id"] != "B0"]
        best = min(full, key=lambda t: (t["result"]["final_metrics"]["mae_float"], t["result"]["final_metrics"]["nrms_official"], -t["result"]["final_metrics"]["ssim_official"])) if full else None
        key = best["candidate_id"] if best else "B0"
        self.s["finalist"] = key
        self.save()
        paired = []
        if key != "B0":
            for seed in self.p["confirmation_seeds"]:
                base = self.job("B0", "confirmation", seed)
                if base["status"] != "completed":
                    raise RuntimeError("确认匹配基线失败，不提交候选")
                cand = self.job(key, "confirmation", seed)
                if base["status"] != "completed" or cand["status"] != "completed":
                    raise RuntimeError("finalist确认未完成，不能宣称通过")
                paired.append({"seed": seed, "baseline": base, "candidate": cand, "win": self.compare(cand, base)})
        confirmation = {"candidate_id": key, "paired": [{"seed": x["seed"], "baseline_task_id": x["baseline"]["task_id"],
                        "candidate_task_id": x["candidate"]["task_id"], "win": x["win"],
                        "delta": {k: x["candidate"]["result"]["final_metrics"][k] - x["baseline"]["result"]["final_metrics"][k] for k in protocol.METRICS}} for x in paired]}
        if paired:
            base_mean = protocol.mean_metrics([x["baseline"]["result"] for x in paired])
            cand_mean = protocol.mean_metrics([x["candidate"]["result"] for x in paired])
            confirmation.update(baseline_mean=base_mean, candidate_mean=cand_mean,
                passed=protocol.improved(base_mean, cand_mean), seed_wins=sum(x["win"] for x in paired),
                std={label: {k: statistics.stdev(x[label]["result"]["final_metrics"][k] for x in paired) for k in protocol.METRICS} for label in ("candidate", "baseline")})
            if not confirmation["passed"]:
                key = "B0"
        else:
            confirmation.update(passed=False, reason="无完整改善候选，保留B0")
        atomic_json(self.root / "confirmation.json", confirmation)
        row = self.job(key, "full", 0)
        if row["status"] != "completed":
            raise RuntimeError("冻结来源任务未完成")
        freeze = self.root / "final"
        if not freeze.exists():
            temp = self.root / "final.tmp"
            if temp.exists():
                raise RuntimeError("存在未完成冻结目录，请检查；不覆盖")
            temp.mkdir()
            shutil.copytree(Path(self.s["candidates"][key]["workspace"]), temp / "source")
            shutil.copy2(Path(row["folder"]) / "artifacts/checkpoint.pt", temp / "checkpoint.pt")
            self.c.write_json(temp / "FROZEN.json", {"candidate_id": key, "task_id": row["task_id"],
                "environment": row["result"]["environment"], "task_config": self.c.read_json(Path(row["folder"]) / "experiment.json"),
                "lineage": self.s["candidates"], "data_hashes": self.c.read_json(self.root / "data_hashes.json"),
                "source_hashes": self.c.source_hashes(temp / "source"), "checkpoint_sha256": self.c.sha256(temp / "checkpoint.pt"),
                "mode": self.p["mode"], "formal_keep": None, "selection_rule": "complete-stage metric gate, MAE/NRMS/SSIM ranking, paired confirmation mean",
                "selection_reason": self.s.get("selection_reason", self.s.get("stop_reason")),
                "confirmation": confirmation, "inference": "python -m sa0.research.inference --frozen FINAL --feature NPY --output NPY"})
            temp.rename(freeze)
        frozen = self.c.read_json(freeze / "FROZEN.json")
        if self.c.sha256(freeze / "checkpoint.pt") != frozen["checkpoint_sha256"] or self.c.source_hashes(freeze / "source") != frozen["source_hashes"]:
            raise RuntimeError("冻结内容被修改")
        self.s.update(phase="reporting", frozen_candidate_id=key)
        self.save()
        if not (self.root / "agent_report.json").exists():
            narrative = self.ask("final_report", "只返回{report:中文报告文字}，总结假设、未测试方向、谱系、失败、各阶段证据。隐藏评测未配置，不得编造结果，不宣称formal keep。")
            if not isinstance(narrative, dict) or set(narrative) != {"report"} or not isinstance(narrative["report"], str):
                raise ValueError("报告schema无效")
            self.c.write_json(self.root / "agent_report.json", narrative)
        self.s["phase"] = "completed"
        self.save()

    def report(self):
        calls = self.s["calls"]
        usage = [call.get("cli", {}).get("usage") for call in calls]
        token_totals = {field: sum(u.get(field, 0) for u in usage if isinstance(u, dict) and type(u.get(field, 0)) is int)
                        for field in ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")}
        unknown = sum(call.get("cli", {}).get("reported_tokens") is None for call in calls)
        summary = {"version": 2, "condition": self.p["condition"], "mode": self.p["mode"], "phase": self.s["phase"],
                   "stop_reason": self.s.get("stop_reason"), "gpu_task_attempts": len(self.s["tasks"]),
                   "exploration_attempts": self.count("exploration"), "finalization_attempts": self.count("finalization"),
                   "screening_attempts": self.bucket_count("screening"), "full_selection_attempts": self.bucket_count("full_selection"),
                   "full_selection_plan": self.s.get("full_selection_plan"), "full_selection_outcome": self.s.get("full_selection_outcome"),
                   "agent_calls": len(calls), "agent_active_seconds": sum(c.get("seconds", 0) for c in calls),
                   "unknown_agent_time_calls": sum("seconds" not in c for c in calls),
                   "uncertain_worker_tasks": sum(bool(t.get("accounting_uncertain")) for t in self.s["tasks"]),
                   "token_usage": token_totals, "unknown_usage_calls": unknown,
                   "worker_process_seconds": sum(t.get("process_seconds", 0) for t in self.s["tasks"]),
                   "training_seconds": sum(t.get("result", {}).get("training_seconds", 0) for t in self.s["tasks"]),
                   "external_baselines": self.s.get("external_baselines", []),
                   "shared_baseline_training_seconds": sum(t["shared_cost_seconds"] for t in self.s.get("external_baselines", [])),
                   "shared_cost_note": "共享B0成本单列，不计为本会话新训练；跨组汇总须去重，同预算科研对照仍需分摊",
                   "gpu_allocation_seconds": None, "queue_seconds": None,
                   "wall_seconds": time.time() - self.s["started_at"], "formal_run": False,
                   "hidden_evaluation": "not_configured; requires independent evaluator", "frozen_candidate_id": self.s.get("frozen_candidate_id"),
                   "untested_hypotheses": self.s.get("untested_hypotheses", {}),
                   "hypotheses": self.s["hypotheses"], "tasks": self.s["tasks"], "candidate_lineage": self.s["candidates"]}
        atomic_json(self.root / "summary.json", summary)
        text = "# 本地研究流程报告\n\n" + json.dumps({k: v for k, v in summary.items() if k not in {"tasks", "candidate_lineage", "hypotheses"}}, ensure_ascii=False, indent=2)
        if (self.root / "agent_report.json").exists():
            text += "\n\n## Agent分析（事实以归档为准）\n\n" + self.c.read_json(self.root / "agent_report.json")["report"]
        (self.root / "REPORT.md").write_text(text, encoding="utf-8")

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with session_lock(self.root):
            self.initialize()
            self.load_shared_baselines()
            try:
                if self.s["phase"] == "completed":
                    self.report()
                    return
                if self.s["phase"] == "paused":
                    self.s["phase"] = self.s.pop("resume_phase", "exploration")
                    self.s["action_errors"] = 0
                if not self.s["hypotheses"]:
                    base = self.job("B0", "full", 0)
                    if base["status"] != "completed":
                        raise RuntimeError("起点基线失败，先检查资源")
                    plan_path = self.root / "research_plan.json"
                    value = self.c.read_json(plan_path) if plan_path.exists() else self.ask("research_plan", "先返回{hypotheses:[H1,H2,H3三个对象],primary_hypothesis_id:H编号}。每个对象的全部字段见研究规则。")
                    self.s["hypotheses"] = protocol.plan(value)
                    self.s["primary_hypothesis_id"] = value["primary_hypothesis_id"]
                    if not plan_path.exists():
                        self.c.write_json(plan_path, value)
                    self.s["phase"] = "exploration"
                    self.save()
                # Analyze any completed experiment left pending across interruptions.
                for row in self.s["tasks"]:
                    if row.get("baseline_task_id"):
                        self.interpret(row)
                while self.s["phase"] == "exploration":
                    if self.p["full_selection_enabled"] and self.screening_remaining() <= 0:
                        self.enter_full_selection("screening_attempt_limit; full slots preserved")
                        break
                    if self.count("exploration") >= self.p["max_exploration_tasks"]:
                        self.s.update(phase="finalization", stop_reason="exploration_attempt_limit")
                        break
                    if all(h["status"] == "closed" for h in self.s["hypotheses"].values()):
                        self.s.update(phase="full_selection" if self.p["full_selection_enabled"] else "finalization", stop_reason="all_hypotheses_closed")
                        break
                    answer = self.ask("next_action", ACTION_SCHEMA)
                    same = digest(answer) == self.s.get("last_action_signature")
                    self.s["repeated_action_count"] = self.s.get("repeated_action_count", 0) + 1 if same else 0
                    self.s["last_action_signature"] = digest(answer)
                    if self.s["repeated_action_count"] >= 3:
                        raise RuntimeError("连续重复同一动作，无新证据，暂停检查；不作为假设反证")
                    try:
                        self.action(answer)
                    except (ValueError, KeyError, TypeError) as error:
                        self.s["action_errors"] += 1
                        self.s["last_action_error"] = str(error)
                        self.event("invalid_action", reason=str(error))
                        self.save()
                        if self.s["action_errors"] >= 3:
                            raise RuntimeError("连续三个无效动作，暂停检查，不作为假设反证")
                    else:
                        self.s["action_errors"] = 0
                        self.save()
                    failed = [t for t in self.s["tasks"][-3:] if t["status"] not in {"completed", "running"}]
                    if len(failed) == 3 and len({t.get("reason") for t in failed}) == 1:
                        raise RuntimeError("连续相同工程错误，暂停处理资源或实现问题")
                if self.p["full_selection_enabled"] and self.s["phase"] == "full_selection":
                    self.full_selection()
                self.finalize()
            except BaseException as error:
                self.s.update(resume_phase=self.s["phase"], phase="paused", stop_reason=f"{type(error).__name__}: {error}")
                self.save()
                raise
            finally:
                self.report()
                print("v2归档：", self.root / "summary.json", flush=True)


def scheduler_changed(base, candidate):
    import ast
    def node(path):
        return [ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding="utf-8")).body
                if isinstance(n, ast.FunctionDef) and n.name == "build_scheduler"]
    return node(base) != node(candidate)


ACTION_SCHEMA = '''选择一个动作，仅返回JSON：
1. {action:"candidate",candidate:{candidate_id:"C数字",parent_candidate_ids:["B0或C编号"],hypothesis_ids:["H编号"],change:{config:{initial_lr/ loss/optimizer/scheduler配置},edits:[{path,old,new}],resolved_sources:{}},expected_effect:文字,falsification_condition:文字,estimated_gpu_seconds:数值,mechanism_test:null或{reference_candidate_id:父候选,metric:指标,direction:increase/decrease,min_delta:非负数}}。
配置loss支持{name:l1/mse/smooth_l1,scale,beta}，optimizer支持{name:adamw/adam/sgd,weight_decay,momentum}，scheduler支持{name:cosine/constant,min_lr}。不修改的字段省略。
2. {action:"experiment",candidate_id,stage:"smoke/low/full",seed:0或1}；full仅seed0，须有同候选low两seed改善，或low改善加匹配关键消融证据。
3. {action:"reconsider",hypothesis_id,decision:"continue/close",reason:文字,evidence_task_ids:[有效任务ID]}。
4. {action:"add_hypothesis",hypothesis:全部假设字段,evidence_task_ids:[已分析任务ID],novelty_reason:文字}。
5. {action:"finalize",reason:文字,untested_hypotheses:{H编号:未研究理由}}。
Agent自主选择方向，不强制覆盖；不要反复生成空候选或提交已完成实验。'''
