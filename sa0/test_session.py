"""CPU-only integration tests. All model and GPU processes are replaced."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sa0 import agent, controller as c
from sa0.session import normalize_protocol, run_session


MODEL_SOURCE = "class MAVI:\n    activation = 'relu'\n"
LR = {"hypothesis": "simulation", "kind": "learning_rate", "change": {"initial_lr": .0004}}
MODEL = {"hypothesis": "simulation", "kind": "model", "change": {"edits": [
    {"path": "train/mavi.py", "old": "activation = 'relu'", "new": "activation = 'silu'"}
]}}


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sa0_test_")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        real_root = c.ROOT
        (self.root / "sa0").mkdir()
        for name in ("controller.py", "session.py", "baseline_cache.py", "agent.py", "decision.py", "feedback.py", "proposal.py", "PROMPT.md"):
            (self.root / "sa0" / name).write_text((real_root / "sa0" / name).read_text(encoding="utf-8"), encoding="utf-8")
        (self.root / "sa0/experiment.py").write_text("# mock worker\n")
        for package in ("train", "prepare", "program"):
            (self.root / package).mkdir()
            (self.root / package / "__init__.py").write_text("")
        (self.root / "train/mavi.py").write_text(MODEL_SOURCE)
        data = self.root / "data"
        data.mkdir()
        (data / "x.npy").write_bytes(b"fake data, never loaded")
        (data / "y.npy").write_bytes(b"fake label, never loaded")
        for name in ("train", "val"):
            (self.root / (name + ".csv")).write_text("x.npy,y.npy\n")
        self.config = {
            "session_dir": str(self.root / "session"), "data_root": str(data),
            "train_manifest": str(self.root / "train.csv"), "val_manifest": str(self.root / "val.csv"),
            "seed": 0, "steps": 2, "batch_size": 2, "val_batch_size": 1,
            "lr_horizon_steps": 200000, "baseline_initial_lr": .0002,
            "max_trials": 2, "initial_lr_upper_bound": .001,
            "process_timeout_seconds": 10, "max_session_process_seconds": 30,
            "budget_status": "test_limits", "max_training_seconds": 10,
            "max_peak_allocated_mib": 1000, "allowed_kinds": ["learning_rate", "model"],
        }
        self.train_calls = []
        self.prompts = []

    def run_case(self, responses, config=None, failure_at=None, interrupt_at=None, operations=None, runner=None):
        config = copy.deepcopy(config or self.config)
        operations = operations or c
        replies = iter(responses)
        owner = self

        def fake_agent(prompt, log_path, settings):
            owner.prompts.append(prompt)
            log_path.write_text("model: gpt-6.1-sol\ntokens used\n123\n")
            value = next(replies)
            if callable(value):
                return value(prompt, log_path, settings)
            return value if isinstance(value, str) else json.dumps(value)

        class FakeProcess:
            def __init__(self, command, **kwargs):
                self.pid = 2000000000
                self.returncode = 0
                self.config = c.read_json(Path(kwargs["env"]["IR_SA0_CONFIG"]))
                owner.train_calls.append(self.config)
                self.index = len(owner.train_calls)
                kwargs["stdout"].write("simulation only\n")

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def kill(self):
                self.returncode = -9

            def wait(self, timeout=None):
                if self.returncode == -9:
                    return -9
                if interrupt_at == self.index:
                    raise KeyboardInterrupt
                if failure_at == self.index:
                    self.returncode = 1
                    return 1
                out = Path(self.config["out_dir"])
                out.mkdir()
                result = {key: self.config[key] for key in c.FIXED_RESULT_KEYS}
                baseline = out.parent.name == "baseline"
                result.update(
                    environment={"mock": True},
                    initial_lr=self.config["initial_lr"], training_seconds=2., peak_allocated_mib=100.,
                    final_metrics={"mae_float": .1 if baseline else .09,
                                   "nrms_official": .2 if baseline else .19,
                                   "ssim_official": .6 if baseline else .61},
                )
                (out / "result.json").write_text(json.dumps(result))
                (out / "checkpoint.pt").write_bytes(b"simulation checkpoint")
                (out / "training.jsonl").write_text('{"step":1,"loss":1,"sample_ids":["x"]}\n')
                return 0

        with patch.object(c, "ROOT", self.root), patch.object(operations, "ROOT", self.root), \
                patch.object(operations, "call_agent", side_effect=fake_agent, create=True), \
                patch.object(c.subprocess, "Popen", FakeProcess), patch("builtins.print"):
            (runner or run_session)(operations, config)
        return c.read_json(Path(config["session_dir"]) / "summary.json")

    def test_autonomous_lr_and_model_and_independent_sources(self):
        summary = self.run_case([LR, MODEL])
        self.assertEqual(len(self.train_calls), 3)
        self.assertEqual(self.train_calls[1]["initial_lr"], .0004)
        self.assertEqual(self.train_calls[2]["initial_lr"], .0002)
        self.assertIn("'relu'", (self.root / "session/frozen/train/mavi.py").read_text())
        self.assertIn("'silu'", (self.root / "session/trial_002/workspace/train/mavi.py").read_text())
        self.assertEqual(summary["agent_calls"], 2)
        self.assertEqual(summary["reported_tokens_known_total"], 246)
        self.assertEqual(len(summary["promising_candidates"]), 2)
        self.assertEqual(summary["retained_candidates"], [])
        self.assertIn('"kind": "learning_rate"', self.prompts[1])

    def test_resume_completed_session_does_not_repeat_calls(self):
        self.run_case([LR, MODEL])
        count = len(self.train_calls)
        self.run_case([])
        self.assertEqual(len(self.train_calls), count)

    def test_reuse_baseline_skips_training_and_preserves_budget(self):
        self.run_case([LR, MODEL])
        source = self.root / "session"
        old = c.read_json(source / "baseline/record.json")
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "baseline_source_session": str(source), "max_trials": 1}
        before = len(self.train_calls)
        with patch("sa0.baseline_cache.probe_environment", return_value={"mock": True}):
            summary = self.run_case([LR], config=config)
            self.run_case([], config=config)
        self.assertEqual(len(self.train_calls), before + 1)  # candidate only; resume adds nothing
        self.assertTrue(summary["baseline_reused"])
        record = c.read_json(self.root / "second/baseline/record.json")
        self.assertEqual(record["process_seconds"], old["process_seconds"])
        self.assertEqual(summary["baseline"], old["result"])
        self.assertEqual(record["reuse_origin"]["session"], str(source))
        self.assertTrue((self.root / "second/baseline/artifacts/checkpoint.pt").is_file())

    def test_reuse_rejects_changed_steps_before_training(self):
        self.run_case([LR, MODEL])
        config = {**self.config, "session_dir": str(self.root / "second"), "steps": 3,
                  "baseline_source_session": str(self.root / "session")}
        before = len(self.train_calls)
        with self.assertRaisesRegex(ValueError, "steps"):
            self.run_case([], config=config)
        self.assertEqual(len(self.train_calls), before)

    def test_reuse_rejects_changed_environment(self):
        self.run_case([LR, MODEL])
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "baseline_source_session": str(self.root / "session")}
        with patch("sa0.baseline_cache.probe_environment", return_value={"mock": False}):
            with self.assertRaisesRegex(ValueError, "环境"):
                self.run_case([], config=config)
        self.assertFalse((self.root / "second/baseline").exists())

    def test_reuse_rejects_changed_source(self):
        self.run_case([LR, MODEL])
        (self.root / "train/mavi.py").write_text(MODEL_SOURCE + "# changed\n")
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "baseline_source_session": str(self.root / "session")}
        with self.assertRaisesRegex(ValueError, "源码"):
            self.run_case([], config=config)

    def test_reuse_rejects_changed_data(self):
        self.run_case([LR, MODEL])
        (self.root / "data/x.npy").write_bytes(b"changed")
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "baseline_source_session": str(self.root / "session")}
        with self.assertRaisesRegex(ValueError, "数据"):
            self.run_case([], config=config)

    def test_reuse_rejects_missing_checkpoint(self):
        self.run_case([LR, MODEL])
        (self.root / "session/baseline/artifacts/checkpoint.pt").unlink()
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "baseline_source_session": str(self.root / "session")}
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            self.run_case([], config=config)

    def test_invalid_proposal_then_valid_model(self):
        summary = self.run_case(["not json", MODEL])
        self.assertEqual(len(self.train_calls), 2)
        self.assertEqual(summary["history"][0]["record"]["training_status"], "not_completed")

    def test_failed_training_is_feedback_for_next_round(self):
        summary = self.run_case([LR, MODEL], failure_at=2)
        self.assertEqual(summary["history"][0]["record"]["training_status"], "failed")
        self.assertIn('"training_status": "failed"', self.prompts[1])
        self.assertIn("training_error_excerpt", self.prompts[1])

    def test_formal_decision_rejects_resource_violation(self):
        baseline = {key: self.config[key] for key in c.FIXED_RESULT_KEYS}
        baseline.update(final_metrics={"mae_float": .1, "nrms_official": .2, "ssim_official": .6},
                        training_seconds=2., peak_allocated_mib=100.)
        candidate = {**baseline, "training_seconds": 11.,
                     "final_metrics": {"mae_float": .09, "nrms_official": .19, "ssim_official": .61}}
        decision = c.compare_results(baseline, candidate, {**self.config, "budget_status": "formal_limits"})
        self.assertFalse(decision["keep"])
        self.assertEqual(decision["reason"], "resource_budget_exceeded")

    def test_cancelling_model_edits_are_not_trained(self):
        noop = copy.deepcopy(MODEL)
        noop["change"]["edits"].append({"path": "train/mavi.py", "old": "activation = 'silu'", "new": "activation = 'relu'"})
        self.run_case([noop, MODEL])
        self.assertEqual(len(self.train_calls), 2)

    def test_duplicate_proposal_is_not_retrained(self):
        summary = self.run_case([LR, LR])
        self.assertEqual(len(self.train_calls), 2)
        self.assertEqual(summary["history"][1]["record"]["training_status"], "not_completed")

    def test_resume_after_interruption_keeps_baseline(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_case([LR, MODEL], interrupt_at=2)
        count = len(self.train_calls)
        summary = self.run_case([MODEL])
        self.assertEqual(len(self.train_calls), count + 1)
        self.assertEqual(summary["history"][0]["record"]["training_status"], "interrupted")

    def test_imported_history_is_visible_and_deduplicated(self):
        self.run_case([LR, MODEL])
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "history_sessions": [str(self.root / "session")], "max_trials": 1}
        count = len(self.train_calls)
        self.run_case([LR], config=config)
        self.assertEqual(len(self.train_calls), count + 1)  # only the new baseline
        self.assertIn('"source_session"', self.prompts[-1])

    def test_protocol_cannot_change_during_resume(self):
        self.run_case([LR, MODEL])
        with self.assertRaises(ValueError):
            self.run_case([], config={**self.config, "steps": 3})

    def test_agent_call_limit_stops_without_extra_training(self):
        summary = self.run_case([LR], config={**self.config, "max_agent_calls": 1})
        self.assertEqual(len(self.train_calls), 2)
        self.assertEqual(summary["stop_reason"], "agent_call_limit")

    def test_session_process_budget_stops_without_agent(self):
        with patch.object(c.time, "perf_counter", side_effect=iter(range(100))):
            summary = self.run_case([], config={**self.config, "max_session_process_seconds": 1})
        self.assertEqual(len(self.train_calls), 1)
        self.assertEqual(summary["agent_calls"], 0)
        self.assertEqual(summary["stop_reason"], "session_process_budget_exhausted")

    def test_candidate_type_restriction_is_enforced(self):
        summary = self.run_case([LR, MODEL], config={**self.config, "allowed_kinds": ["model"]})
        self.assertEqual(len(self.train_calls), 2)
        self.assertEqual(summary["history"][0]["record"]["training_status"], "not_completed")

    def test_replay_first_proposal_does_not_call_agent(self):
        proposal = self.root / "existing.json"
        proposal.write_text(json.dumps(MODEL))
        summary = self.run_case([], config={**self.config, "first_proposal_path": str(proposal), "max_trials": 1})
        self.assertEqual(summary["agent_calls"], 0)

    def test_environment_mismatch_invalidates_comparison(self):
        result = {key: self.config[key] for key in c.FIXED_RESULT_KEYS}
        result.update(final_metrics={"mae_float": .1, "nrms_official": .2, "ssim_official": .6},
                      training_seconds=2., peak_allocated_mib=100., environment={"torch": "A"})
        other = {**result, "environment": {"torch": "B"}}
        with self.assertRaises(ValueError):
            c.compare_results(result, other, self.config)

    def test_invalid_numeric_config_rejected(self):
        with self.assertRaises(ValueError):
            normalize_protocol({**self.config, "process_timeout_seconds": float("nan")})

    def test_frozen_data_change_blocks_resume(self):
        self.run_case([LR, MODEL])
        (self.root / "data/x.npy").write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            self.run_case([])

    def test_agent_settings_are_used_and_errors_logged(self):
        settings = normalize_protocol(self.config)["agent"]
        log = self.root / "agent.log"
        completed = SimpleNamespace(returncode=0, stdout="{}", stderr="model: configured")
        with patch.object(agent.subprocess, "run", return_value=completed) as call:
            self.assertEqual(agent.call_agent("proposal", log, settings), "{}")
        self.assertIn(settings["model"], call.call_args.args[0])
        self.assertEqual(call.call_args.kwargs["timeout"], settings["timeout_seconds"])
        self.assertTrue(log.with_suffix(".timing.json").exists())

    def test_agent_timeout_keeps_partial_output(self):
        settings = normalize_protocol(self.config)["agent"]
        log = self.root / "timeout.log"
        error = subprocess.TimeoutExpired("mock", 180, output=b'partial', stderr=b'timeout')
        with patch.object(agent.subprocess, "run", side_effect=error):
            with self.assertRaises(RuntimeError):
                agent.call_agent("proposal", log, settings)
        self.assertEqual(log.read_text(), "timeout")
        self.assertEqual(log.with_suffix(".partial_response.txt").read_text(), "partial")


if __name__ == "__main__":
    unittest.main()
