"""No GPU/model calls: protocol, lineage, recovery and complete workflow checks."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from sa0 import controller as c
from sa0.research import protocol, lineage
from sa0.research.session import Workflow
from sa0.session import session_lock


def initial_plan():
    return {"hypotheses": [{"hypothesis_id": f"H{i}", **{k: f"H{i} explicit evidence and prediction" for k in
        ("observation", "mechanism", "proposed_change", "discriminating_experiments", "expected_metrics", "falsification_condition", "risks", "estimated_cost")}}
        for i in (1, 2, 3)], "primary_hypothesis_id": "H1"}


def candidate(key="C1", parents=None, h=None, lr=.0008):
    return {"candidate_id": key, "parent_candidate_ids": parents or ["B0"], "hypothesis_ids": h or ["H1"],
            "change": {"config": {"initial_lr": lr}, "edits": [], "resolved_sources": {}},
            "expected_effect": "MAE improves", "falsification_condition": "matched metrics worsen",
            "estimated_gpu_seconds": 1., "mechanism_test": None}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="ir_workflow_")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        original = c.ROOT
        for package in ("train", "prepare", "program", "sa0", "sa1"):
            (self.root / package).mkdir()
        for file in ("sa0/controller.py", "sa0/experiment.py", "sa0/session.py", "sa1/agent.py", "sa1/proposal.py", "train/experiment.py"):
            shutil.copy2(original / file, self.root / file)
        shutil.copytree(original / "sa0/research", self.root / "sa0/research", ignore=shutil.ignore_patterns("__pycache__"))
        (self.root / "train/mavi.py").write_text("class MAVI:\n    activation='relu'\n")
        (self.root / "prepare/__init__.py").write_text("")
        (self.root / "program/__init__.py").write_text("")
        (self.root / "data").mkdir()
        (self.root / "data/x.npy").write_bytes(b"feature")
        (self.root / "data/y.npy").write_bytes(b"label")
        (self.root / "train.csv").write_text("x.npy,y.npy\n")
        (self.root / "val.csv").write_text("x.npy,y.npy\n")
        self.p = c.read_json(original / "sa0/research_config.json")
        self.p.update(session_dir=str(self.root / "session"), data_root=str(self.root / "data"), device="cpu")
        for spec in self.p["stages"].values():
            spec.update(train_manifest=str(self.root / "train.csv"), val_manifest=str(self.root / "val.csv"))
        self.configs = []
        self.prompts = []
        self.root_patch = patch.object(c, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def fake_job(self, owner, config, folder, workspace):
        self.configs.append(config)
        out = Path(config["out_dir"])
        out.mkdir()
        metrics = {"mae_float": .1, "nrms_official": .2, "ssim_official": .6}
        is_candidate = config["recipe"]["initial_lr"] != self.p["baseline_initial_lr"]
        if is_candidate:
            metrics = {"mae_float": .08, "nrms_official": .18, "ssim_official": .65}
        result = {**{k: config[k] for k in ("seed", "steps", "batch_size", "val_batch_size", "lr_horizon_steps")},
                  "final_metrics": metrics, "training_seconds": 1., "peak_allocated_mib": 2.,
                  "initialized_from": "fresh", "environment": {"mock": True}, "recipe": config["recipe"]}
        (folder / "experiment.json").write_text(json.dumps(config))
        (folder / "train.log").write_text("mock\n")
        (out / "result.json").write_text(json.dumps(result))
        (out / "checkpoint.pt").write_bytes(b"mock checkpoint")
        (out / "training.jsonl").write_text('{"step":1,"loss":1}\n')
        return {"status": "completed", "result": result, "process_seconds": 1.}

    def fake_agent(self, actions):
        iterator = iter(actions)
        def invoke(prompt, folder, settings):
            self.prompts.append(prompt)
            folder.mkdir()
            (folder / "input.md").write_text(prompt, encoding="utf-8")
            (folder / "record.json").write_text(json.dumps({"status": "completed", "seconds": .1,
                "reported_tokens": 60, "usage": {"input_tokens": 50, "output_tokens": 10, "cached_input_tokens": 20}}))
            if "先返回{hypotheses" in prompt:
                return initial_plan()
            if "解释实验T" in prompt:
                return {"facts": "observed", "inference": "preliminary", "caveats": "not causal proof", "next_action": "continue"}
            if "只返回{report:" in prompt:
                return {"report": "workflow only; hidden evaluation not configured"}
            return next(iterator)
        return invoke

    def run_case(self, actions):
        with patch("sa0.research.session.execute_job", side_effect=self.fake_job), \
             patch("sa0.research.session.agent.call", side_effect=self.fake_agent(actions)), patch("builtins.print"):
            Workflow(c, self.p).run()
        return c.read_json(self.root / "session/summary.json")

    def setup_workflow(self):
        w = Workflow(c, self.p)
        w.initialize()
        w.s["hypotheses"] = protocol.plan(initial_plan())
        w.s["phase"] = "exploration"
        return w

    def full_actions(self):
        return [{"action": "candidate", "candidate": candidate()},
                {"action": "experiment", "candidate_id": "C1", "stage": "low", "seed": 0},
                {"action": "experiment", "candidate_id": "C1", "stage": "low", "seed": 1},
                {"action": "experiment", "candidate_id": "C1", "stage": "full", "seed": 0},
                {"action": "finalize", "reason": "enough evidence", "untested_hypotheses": {"H2": "not prioritized", "H3": "not prioritized"}}]

    def test_complete_workflow_promotion_confirmation_freeze_report(self):
        summary = self.run_case(self.full_actions())
        self.assertEqual(summary["phase"], "completed")
        self.assertEqual(summary["gpu_task_attempts"], 10)
        self.assertEqual(summary["exploration_attempts"], 6)
        self.assertEqual(summary["finalization_attempts"], 4)
        self.assertEqual(summary["frozen_candidate_id"], "C1")
        self.assertEqual(summary["unknown_usage_calls"], 0)
        self.assertEqual(summary["token_usage"]["input_tokens"], summary["agent_calls"] * 50)
        self.assertFalse(summary["formal_run"])
        confirm = c.read_json(self.root / "session/confirmation.json")
        self.assertEqual(confirm["seed_wins"], 3)
        self.assertTrue(confirm["passed"])
        self.assertTrue((self.root / "session/final/FROZEN.json").exists())
        self.assertEqual(len(self.configs), 10)
        before = len(self.prompts)
        self.run_case([])
        self.assertEqual(len(self.prompts), before)
        self.assertEqual(len(self.configs), 10)

    def test_no_winner_freezes_b0_without_extra_training(self):
        summary = self.run_case([{"action": "finalize", "reason": "no evidence", "untested_hypotheses": {"H1": "not run", "H2": "not run", "H3": "not run"}}])
        self.assertEqual(summary["frozen_candidate_id"], "B0")
        self.assertEqual(summary["gpu_task_attempts"], 1)

    def test_plan_requires_three_and_primary(self):
        p = initial_plan()
        p["hypotheses"].pop()
        with self.assertRaises(ValueError):
            protocol.plan(p)
        p = initial_plan()
        p["primary_hypothesis_id"] = "H8"
        with self.assertRaises(ValueError):
            protocol.plan(p)

    def test_lineage_missing_parent_and_cycle(self):
        with self.assertRaises(ValueError):
            lineage.check_tree({"C1": {"parent_candidate_ids": ["missing"]}})
        with self.assertRaises(ValueError):
            lineage.check_tree({"C1": {"parent_candidate_ids": ["C2"]}, "C2": {"parent_candidate_ids": ["C1"]}})

    def test_inheritance_and_first_hypothesis_candidate(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        w.action({"action": "candidate", "candidate": candidate("C2", ["C1"], lr=.001)})
        self.assertEqual(w.s["candidates"]["C2"]["recipe"]["initial_lr"], .001)
        self.assertEqual(w.s["candidates"]["C2"]["parent_candidate_ids"], ["C1"])
        with self.assertRaises(ValueError):
            w.action({"action": "candidate", "candidate": candidate("C3", ["C1"], ["H2"])})

    def test_invalid_edit_is_transactional_and_can_be_corrected(self):
        w = self.setup_workflow()
        value = candidate()
        value["change"]["edits"] = [{"path": "train/mavi.py", "old": "missing", "new": "x"}]
        with self.assertRaises(ValueError):
            w.action({"action": "candidate", "candidate": value})
        self.assertFalse((self.root / "session/candidates/C1").exists())
        w.action({"action": "candidate", "candidate": candidate()})

    def test_full_rejects_single_seed_and_smoke_not_scientific(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        with self.assertRaises(ValueError):
            w.experiment("C1", "full", 0)
        with patch("sa0.research.session.execute_job", side_effect=self.fake_job):
            row = w.job("C1", "smoke", 0)
        self.assertFalse(row["scientific_valid"])
        self.assertEqual(w.count("exploration"), 1)
        self.assertFalse(w.can_promote("C1"))

    def test_cross_stage_comparison_rejected(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        with patch("sa0.research.session.execute_job", side_effect=self.fake_job):
            a = w.job("B0", "low", 0)
            b = w.job("C1", "full", 0)
        with self.assertRaises(ValueError):
            w.compare(b, a)

    def test_task_caps_include_baselines_and_failures(self):
        self.p["max_exploration_tasks"] = 3
        self.p["max_gpu_tasks"] = 6
        summary = self.run_case(self.full_actions()[:2])
        self.assertEqual(summary["gpu_task_attempts"], 3)
        self.assertEqual(summary["frozen_candidate_id"], "B0")
        self.assertEqual(summary["stop_reason"], "exploration_attempt_limit")

    def test_failure_consumes_slot_but_does_not_refute_hypothesis(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        with patch("sa0.research.session.execute_job", return_value={"status": "failed", "reason": "OOM", "process_seconds": 1}):
            w.job("C1", "low", 0)
        self.assertEqual(len(w.s["tasks"]), 1)
        self.assertEqual(w.s["hypotheses"]["H1"]["no_progress_streak"], 0)

    def test_edited_evaluator_and_locked_training_function_rejected(self):
        w = self.setup_workflow()
        value = candidate()
        value["change"]["edits"] = [{"path": "prepare/evaluator.py", "old": "x", "new": "y"}]
        with self.assertRaises(ValueError):
            w.action({"action": "candidate", "candidate": value})
        value["change"]["edits"] = [{"path": "train/experiment.py", "old": "model = MAVI()", "new": "model = None"}]
        with self.assertRaises(ValueError):
            w.action({"action": "candidate", "candidate": value})

    def test_loss_optimizer_scheduler_recipe_and_feature_patch(self):
        w = self.setup_workflow()
        value = candidate()
        value["change"]["config"].update(loss={"name": "smooth_l1", "beta": .01}, optimizer={"name": "sgd", "momentum": .9}, scheduler={"name": "constant"})
        value["change"]["edits"] = [{"path": "train/feature_transform.py", "old": "return feature", "new": "return feature * 0.9"}]
        w.action({"action": "candidate", "candidate": value})
        self.assertEqual(w.s["candidates"]["C1"]["recipe"]["loss"]["name"], "smooth_l1")

    def test_two_wins_one_loss_can_pass_mean_gate(self):
        base = [{"final_metrics": {"mae_float": .1, "nrms_official": .2, "ssim_official": .6}}] * 3
        cand = [{"final_metrics": {"mae_float": a, "nrms_official": n, "ssim_official": s}}
                for a, n, s in ((.08, .18, .64), (.08, .18, .64), (.11, .21, .59))]
        self.assertTrue(protocol.improved(protocol.mean_metrics(base), protocol.mean_metrics(cand)))
        cand[2]["final_metrics"]["mae_float"] = .3
        self.assertFalse(protocol.improved(protocol.mean_metrics(base), protocol.mean_metrics(cand)))

    def test_changed_protocol_or_data_blocks_resume(self):
        self.run_case([{"action": "finalize", "reason": "done", "untested_hypotheses": {"H1": "not tested", "H2": "not tested", "H3": "not tested"}}])
        changed = copy.deepcopy(self.p)
        changed["max_gpu_tasks"] = 21
        with self.assertRaises(ValueError):
            Workflow(c, changed).run()
        (self.root / "data/x.npy").write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            Workflow(c, self.p).run()

    def test_formal_mode_is_not_silently_enabled(self):
        with self.assertRaises(ValueError):
            protocol.validate({**self.p, "mode": "formal"})

    def test_no_hidden_field_in_agent_feedback(self):
        self.run_case([{"action": "finalize", "reason": "done", "untested_hypotheses": {"H1": "not tested", "H2": "not tested", "H3": "not tested"}}])
        for prompt in self.prompts:
            self.assertNotIn('"hidden_labels"', prompt)
            self.assertNotIn('"hidden_metrics"', prompt)

    def test_confirmation_two_wins_one_loss_reports_and_accepts_mean(self):
        original = self.fake_job
        def job(owner, config, folder, workspace):
            row = original(owner, config, folder, workspace)
            if config["seed"] == 2 and config["recipe"]["initial_lr"] != self.p["baseline_initial_lr"]:
                row["result"]["final_metrics"] = {"mae_float": .11, "nrms_official": .21, "ssim_official": .59}
                (Path(config["out_dir"]) / "result.json").write_text(json.dumps(row["result"]))
            return row
        self.fake_job = job
        self.run_case(self.full_actions())
        confirm = c.read_json(self.root / "session/confirmation.json")
        self.assertEqual(confirm["seed_wins"], 2)
        self.assertTrue(confirm["passed"])

    def test_conflicting_combination_requires_resolution_and_evidence(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        w.action({"action": "candidate", "candidate": candidate("C2", h=["H2"], lr=.0006)})
        combined = candidate("C3", ["C1", "C2"], ["H1", "H2"], lr=.0007)
        with self.assertRaises(ValueError):
            w.action({"action": "candidate", "candidate": combined})
        w.s["tasks"] = [{"task_id": "T1", "candidate_id": "C1", "scientific_valid": True, "metrics_ok": True},
                        {"task_id": "T2", "candidate_id": "C2", "scientific_valid": True, "metrics_ok": True}]
        broken = copy.deepcopy(combined)
        broken["change"]["config"] = {}
        with self.assertRaisesRegex(ValueError, "冲突"):
            w.action({"action": "candidate", "candidate": broken})
        w.action({"action": "candidate", "candidate": combined})
        self.assertEqual(w.s["candidates"]["C3"]["parent_candidate_ids"], ["C1", "C2"])

    def test_mechanistic_ablation_progress_can_support_promotion(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        ablation = candidate("C2", ["C1"], lr=.00015)
        ablation["mechanism_test"] = {"reference_candidate_id": "C1", "metric": "mae_float", "direction": "increase", "min_delta": .01}
        w.action({"action": "candidate", "candidate": ablation})
        original = self.fake_job
        def job(owner, config, folder, workspace):
            row = original(owner, config, folder, workspace)
            if config["recipe"]["initial_lr"] == .00015:
                row["result"]["final_metrics"] = {"mae_float": .11, "nrms_official": .21, "ssim_official": .59}
                (Path(config["out_dir"]) / "result.json").write_text(json.dumps(row["result"]))
            return row
        with patch("sa0.research.session.execute_job", side_effect=job), patch("sa0.research.session.agent.call", side_effect=self.fake_agent([])), patch("builtins.print"):
            w.experiment("C1", "low", 0)
            self.assertFalse(w.can_promote("C1"))
            ab = w.experiment("C2", "low", 0)
        self.assertFalse(ab["metrics_ok"])
        self.assertIn("mechanistic_progress", w.s["interpretations"][ab["task_id"]]["labels"])
        self.assertTrue(w.can_promote("C1"))

    def test_three_no_progress_requires_review_not_automatic_close(self):
        w = self.setup_workflow()
        for i, lr in enumerate((.0008, .0009, .001), 1):
            w.action({"action": "candidate", "candidate": candidate(f"C{i}", lr=lr)})
        original = self.fake_job
        def job(owner, config, folder, workspace):
            row = original(owner, config, folder, workspace)
            if config["recipe"]["initial_lr"] != self.p["baseline_initial_lr"]:
                row["result"]["final_metrics"] = {"mae_float": .11, "nrms_official": .21, "ssim_official": .59}
                (Path(config["out_dir"]) / "result.json").write_text(json.dumps(row["result"]))
            return row
        with patch("sa0.research.session.execute_job", side_effect=job), patch("sa0.research.session.agent.call", side_effect=self.fake_agent([])), patch("builtins.print"):
            for i in (1, 2, 3):
                w.experiment(f"C{i}", "low", 0)
        self.assertEqual(w.s["hypotheses"]["H1"]["status"], "needs_reconsideration")
        tasks = list(w.s["interpretations"])
        w.action({"action": "reconsider", "hypothesis_id": "H1", "decision": "continue", "reason": "new test needed", "evidence_task_ids": tasks})
        self.assertEqual(w.s["hypotheses"]["H1"]["status"], "open")
        self.assertEqual(len(w.s["tasks"]), 4)  # shared matched B0 reused

    def test_new_hypotheses_require_evidence_and_maximum_five(self):
        w = self.setup_workflow()
        h = initial_plan()["hypotheses"][0]
        action = {"action": "add_hypothesis", "hypothesis": {**h, "hypothesis_id": "H4", "mechanism": "new feature mechanism"}, "evidence_task_ids": [], "novelty_reason": "new mechanism"}
        with self.assertRaises(ValueError):
            w.action(action)
        w.s["interpretations"]["T1"] = {"labels": ["no_progress"]}
        action["evidence_task_ids"] = ["T1"]
        w.action(action)
        action["hypothesis"] = {**h, "hypothesis_id": "H5", "mechanism": "another loss mechanism"}
        w.action(action)
        action["hypothesis"] = {**h, "hypothesis_id": "H6"}
        with self.assertRaises(ValueError):
            w.action(action)

    def test_interrupted_task_counts_and_resume_does_not_auto_repeat(self):
        w = self.setup_workflow()
        w.action({"action": "candidate", "candidate": candidate()})
        with patch("sa0.research.session.execute_job", side_effect=KeyboardInterrupt), patch("builtins.print"):
            with self.assertRaises(KeyboardInterrupt):
                w.job("C1", "low", 0)
        restored = Workflow(c, self.p)
        restored.initialize()
        self.assertEqual(len(restored.s["tasks"]), 1)
        self.assertEqual(restored.s["tasks"][0]["status"], "interrupted")

    def test_checkpoint_tampering_rejects_cache(self):
        w = self.setup_workflow()
        with patch("sa0.research.session.execute_job", side_effect=self.fake_job), patch("builtins.print"):
            row = w.job("B0", "full", 0)
        (Path(row["folder"]) / "artifacts/checkpoint.pt").write_bytes(b"changed")
        with self.assertRaises(RuntimeError):
            w.job("B0", "confirmation", 0)

    def test_global_gpu_lock_rejects_concurrent_v2_worker(self):
        from sa0.research.session import execute_job
        lock = self.root / "results/.research_v2_gpu_lock"
        lock.mkdir(parents=True)
        with session_lock(lock):
            with patch("sa0.research.session._execute_job") as worker:
                with self.assertRaises(RuntimeError):
                    execute_job(c, {"device": "cuda"}, self.root / "unused", self.root)
                worker.assert_not_called()


if __name__ == "__main__":
    unittest.main()
