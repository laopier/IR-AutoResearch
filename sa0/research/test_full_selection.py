"""No GPU calls: screening reserves, mandatory full selection and restart safety."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from sa0 import controller as c
from sa0.research.session import Workflow
from sa0.research import protocol, test_workflow as fixtures


class FullSelectionTests(unittest.TestCase):
    def setUp(self):
        self.f = fixtures.WorkflowTests("test_complete_workflow_promotion_confirmation_freeze_report")
        self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.p = self.f.p
        self.p.update(full_selection_enabled=True, max_gpu_tasks=24, max_exploration_tasks=18,
                      max_screening_tasks=13, full_task_reserve=4, max_full_candidates=2)
        self.root = self.f.root

    def actions(self):
        return [{"action": "candidate", "candidate": fixtures.candidate()},
                {"action": "experiment", "candidate_id": "C1", "stage": "low", "seed": 0},
                {"action": "experiment", "candidate_id": "C1", "stage": "low", "seed": 1},
                {"action": "finalize", "reason": "enough screening", "untested_hypotheses": {"H2": "not prioritized", "H3": "not prioritized"}}]

    def agent(self, actions, response=None):
        regular = self.f.fake_agent(actions)
        def invoke(prompt, folder, settings):
            request = prompt.rsplit("\n请求：\n", 1)[-1]
            if request.startswith("screening已关闭"):
                self.f.prompts.append(prompt)
                folder.mkdir()
                (folder / "record.json").write_text(json.dumps({"status": "completed", "seconds": .1,
                    "reported_tokens": 60, "usage": {"input_tokens": 50, "output_tokens": 10}}))
                return response or {"candidate_ids": ["C1"], "reason": "paired low evidence"}
            return regular(prompt, folder, settings)
        return invoke

    def run_case(self, actions, response=None, job=None, retry=False):
        with patch("sa0.research.session.execute_job", side_effect=job or self.f.fake_job), \
             patch("sa0.research.session.agent.call", side_effect=self.agent(actions, response)), patch("builtins.print"):
            Workflow(c, self.p, retry_failed_full=retry).run()
        return c.read_json(self.root / "session/summary.json")

    def screened(self):
        w = self.f.setup_workflow()
        with patch("sa0.research.session.execute_job", side_effect=self.f.fake_job), \
             patch("sa0.research.session.agent.call", side_effect=self.agent([])), patch("builtins.print"):
            w.s["phase"] = "planning"
            w.job("B0", "full", 0)
            w.s["phase"] = "exploration"
            self.f.add_candidate(w, fixtures.candidate())
            w.experiment("C1", "low", 0)
            w.experiment("C1", "low", 1)
        return w

    def test_finalize_request_cannot_skip_valid_full(self):
        summary = self.run_case(self.actions())
        self.assertEqual(summary["phase"], "completed")
        self.assertEqual(summary["full_selection_attempts"], 1)
        self.assertEqual(summary["screening_attempts"], 4)
        self.assertEqual(summary["full_selection_plan"]["candidate_ids"], ["C1"])
        self.assertEqual(summary["full_selection_outcome"], "completed")
        self.assertEqual(summary["frozen_candidate_id"], "C1")

    def test_screening_limit_automatically_moves_to_full(self):
        self.p["max_screening_tasks"] = 4
        summary = self.run_case(self.actions()[:3])
        self.assertEqual(summary["full_selection_attempts"], 1)
        state = c.read_json(self.root / "session/state.json")
        self.assertIn("full slots preserved", state["screening_end_reason"])
        self.assertLessEqual(summary["gpu_task_attempts"], 24)

    def test_no_eligible_candidate_reports_b0(self):
        self.p["max_screening_tasks"] = 2
        summary = self.run_case(self.actions()[:2])
        self.assertEqual(summary["full_selection_outcome"], "no_eligible_candidates")
        self.assertEqual(summary["full_selection_attempts"], 0)
        self.assertEqual(summary["frozen_candidate_id"], "B0")

    def test_full_phase_rejects_low_and_candidate_creation(self):
        w = self.screened()
        w.enter_full_selection("end screen")
        with self.assertRaises(ValueError):
            w.experiment("C1", "low", 0)
        with self.assertRaises(ValueError):
            w.job("C1", "low", 0)
        with self.assertRaises(ValueError):
            self.f.add_candidate(w, fixtures.candidate("C2", lr=.0009))

    def test_cannot_finalize_eligible_candidates_before_full(self):
        w = self.screened()
        with self.assertRaises(ValueError):
            w.finalize()
        with self.assertRaises(ValueError):
            w.job("C1", "full", 0)

    def test_invalid_selection_never_trains_selected_full(self):
        with self.assertRaises(ValueError):
            self.run_case(self.actions(), response={"candidate_ids": ["B0"], "reason": "invalid"})
        state = c.read_json(self.root / "session/state.json")
        self.assertFalse(any(t.get("budget_bucket") == "full_selection" for t in state["tasks"]))

    def test_failed_full_pauses_then_explicit_retry_consumes_another_task(self):
        def fail(owner, config, folder, workspace):
            if config["steps"] == self.p["stages"]["full"]["steps"] and config["recipe"]["initial_lr"] != self.p["baseline_initial_lr"]:
                return {"status": "failed", "reason": "OOM", "process_seconds": 1.}
            return self.f.fake_job(owner, config, folder, workspace)
        with self.assertRaises(RuntimeError):
            self.run_case(self.actions(), job=fail)
        before = len(c.read_json(self.root / "session/state.json")["tasks"])
        with self.assertRaises(RuntimeError):
            self.run_case([])  # no implicit re-training
        self.assertEqual(len(c.read_json(self.root / "session/state.json")["tasks"]), before)
        summary = self.run_case([], retry=True)
        self.assertEqual(summary["full_selection_attempts"], 2)
        self.assertEqual(summary["phase"], "completed")

    def test_full_reserve_must_fit_exploration_budget(self):
        with self.assertRaises(ValueError):
            protocol.validate({**self.p, "max_screening_tasks": 18})

    def test_exhausted_screening_does_not_consume_full_reserve(self):
        w = self.screened()
        self.p["max_screening_tasks"] = 4
        w.p = protocol.validate(self.p)
        self.assertEqual(w.screening_remaining(), 0)
        with self.assertRaises(ValueError):
            w.job("C1", "smoke", 0)
        self.assertEqual(w.bucket_count("full_selection"), 0)

    def test_data_not_ready_blocks_before_session_creation(self):
        self.p["dataset_report"] = str(self.root / "not_ready.json")
        (self.root / "not_ready.json").write_text('{"ready":false}')
        with self.assertRaises(RuntimeError):
            Workflow(c, self.p).run()
        self.assertFalse((self.root / "session/protocol.json").exists())

    def test_two_selected_candidates_can_both_run_full(self):
        actions = self.actions()[:3] + [
            {"action": "candidate", "candidate": fixtures.candidate("C2", lr=.0009)},
            {"action": "experiment", "candidate_id": "C2", "stage": "low", "seed": 0},
            {"action": "experiment", "candidate_id": "C2", "stage": "low", "seed": 1},
            self.actions()[-1],
        ]
        summary = self.run_case(actions, response={"candidate_ids": ["C1", "C2"], "reason": "compare distinct candidates"})
        self.assertEqual(summary["full_selection_attempts"], 2)
        self.assertEqual(summary["full_selection_plan"]["candidate_ids"], ["C1", "C2"])

    def test_full_reaches_valid_metrics_but_no_improvement_freezes_b0(self):
        def no_gain(owner, config, folder, workspace):
            row = self.f.fake_job(owner, config, folder, workspace)
            if config["steps"] == self.p["stages"]["full"]["steps"] and config["recipe"]["initial_lr"] != self.p["baseline_initial_lr"]:
                row["result"]["final_metrics"] = {"mae_float": .11, "nrms_official": .21, "ssim_official": .59}
                (Path(config["out_dir"]) / "result.json").write_text(json.dumps(row["result"]))
            return row
        summary = self.run_case(self.actions(), job=no_gain)
        self.assertEqual(summary["full_selection_outcome"], "completed")
        self.assertEqual(summary["frozen_candidate_id"], "B0")


if __name__ == "__main__":
    unittest.main()
