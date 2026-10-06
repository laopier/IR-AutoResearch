"""Random-search protocol and CPU-only mocked integration checks."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sa0 import test_session as fixture_module
from b1 import controller
from b1.search import validate_config, generate_plan


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((controller.ROOT / "b1/config.json").read_text(encoding="utf-8"))

    def test_plan_is_deterministic_unique_and_bounded(self):
        p = validate_config(self.config)
        a, b = generate_plan(p), generate_plan(p)
        self.assertEqual(a, b)
        values = [x["initial_lr"] for x in a["trials"]]
        self.assertEqual(len(set(values)), p["max_trials"])
        self.assertNotIn(p["baseline_initial_lr"], values)
        self.assertTrue(all(1e-7 < x <= .001 for x in values))
        other = generate_plan({**p, "search_seed": 1})
        self.assertNotEqual(a["trials"], other["trials"])
        self.assertEqual(p["seed"], self.config["seed"])

    def test_search_bounds_and_seed_are_validated(self):
        for space in ({"distribution": "uniform", "min": 1e-7, "max": .001},
                      {"distribution": "log_uniform", "min": 1e-7, "max": float("nan")},
                      {"distribution": "log_uniform", "min": .001, "max": .0001}):
            with self.assertRaises(ValueError):
                validate_config({**self.config, "search_space": {"initial_lr": space}})
        with self.assertRaises(ValueError):
            validate_config({**self.config, "search_seed": True})

    def test_no_agent_history_or_extra_factors(self):
        for extra in ({"agent": {}}, {"history_sessions": ["prior"]},
                      {"search_space": {"initial_lr": {}, "model": {}}}):
            with self.assertRaises(ValueError):
                validate_config({**self.config, **extra})

    def test_fixed_training_and_budget_match_sa0(self):
        sa0 = json.loads((controller.ROOT / "sa0/config.json").read_text(encoding="utf-8"))
        for key in (*controller.FIXED_RESULT_KEYS, "baseline_initial_lr", "data_root", "train_manifest",
                    "val_manifest", "max_trials", "max_session_process_seconds", "process_timeout_seconds",
                    "max_training_seconds", "max_peak_allocated_mib", "budget_status"):
            self.assertEqual(self.config[key], sa0[key], key)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixture_module.SessionTests("test_autonomous_lr_and_model_and_independent_sources")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        (self.root / "b1").mkdir()
        for filename in ("controller.py", "search.py"):
            (self.root / "b1" / filename).write_text(
                (controller.ROOT / "b1" / filename).read_text(encoding="utf-8"), encoding="utf-8")
        self.operations = SimpleNamespace(**{name: getattr(controller, name) for name in dir(controller)
                                             if not name.startswith("__")})
        self.config = {**self.fixture.config, "condition": "B1", "search_seed": 0,
                       "search_space": {"initial_lr": {"distribution": "log_uniform", "min": 1e-7, "max": .001}}}

    def run_case(self, **kwargs):
        return self.fixture.run_case([], config=kwargs.pop("config", self.config),
                                     operations=self.operations, runner=controller.run_session, **kwargs)

    def test_full_loop_without_agent_fixed_model_and_resume(self):
        summary = self.run_case()
        self.assertEqual(summary["agent_calls"], 0)
        self.assertEqual(self.fixture.prompts, [])
        self.assertEqual(len(self.fixture.train_calls), 3)
        self.assertEqual(len(summary["history"]), 2)
        values = [x["initial_lr"] for x in summary["search_plan"]["trials"]]
        self.assertEqual([x["initial_lr"] for x in self.fixture.train_calls[1:]], values)
        for number in (1, 2):
            self.assertEqual((self.root / f"session/trial_{number:03d}/workspace/train/mavi.py").read_text(),
                             fixture_module.MODEL_SOURCE)
        self.run_case()
        self.assertEqual(len(self.fixture.train_calls), 3)
        self.assertEqual(summary["stop_reason"], "max_trials_reached")

    def test_failure_keeps_precommitted_second_candidate(self):
        summary = self.run_case(failure_at=2)
        self.assertEqual(summary["history"][0]["record"]["training_status"], "failed")
        second = summary["search_plan"]["trials"][1]["initial_lr"]
        self.assertEqual(self.fixture.train_calls[2]["initial_lr"], second)
        self.assertEqual(summary["history"][1]["record"]["training_status"], "completed")

    def test_interrupted_trial_is_not_retrained(self):
        with self.assertRaises(KeyboardInterrupt):
            self.run_case(interrupt_at=2)
        summary = self.run_case()
        self.assertEqual(len(self.fixture.train_calls), 3)
        self.assertEqual(summary["history"][0]["record"]["training_status"], "interrupted")
        self.assertEqual(summary["stop_reason"], "max_trials_reached")

    def test_plan_tampering_stops_resume(self):
        self.run_case()
        path = self.root / "session/search_plan.json"
        plan = json.loads(path.read_text())
        plan["trials"][0]["initial_lr"] = .0009
        path.write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "计划"):
            self.run_case()

    def test_changed_protocol_stops_resume(self):
        self.run_case()
        with self.assertRaises(ValueError):
            self.run_case(config={**self.config, "search_seed": 1})

    def test_budget_stops_before_candidates(self):
        with patch.object(controller.time, "perf_counter", side_effect=iter(range(100))):
            summary = self.run_case(config={**self.config, "max_session_process_seconds": 1})
        self.assertEqual(len(self.fixture.train_calls), 1)
        self.assertEqual(summary["stop_reason"], "session_process_budget_exhausted")

    def test_verified_baseline_reuse_skips_new_baseline(self):
        self.run_case()
        old = json.loads((self.root / "session/baseline/record.json").read_text())
        before = len(self.fixture.train_calls)
        config = {**self.config, "session_dir": str(self.root / "second"),
                  "baseline_source_session": str(self.root / "session")}
        with patch("sa0.baseline_cache.probe_environment", return_value={"mock": True}):
            summary = self.run_case(config=config)
        self.assertEqual(len(self.fixture.train_calls), before + 2)
        self.assertEqual(summary["baseline"], old["result"])
        self.assertTrue(summary["baseline_reused"])
        self.assertEqual(summary["agent_calls"], 0)


if __name__ == "__main__":
    unittest.main()
