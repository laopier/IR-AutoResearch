"""Offline checks for SA1 permission, search evidence and shared orchestration."""
import copy
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sa0 import test_session as fixture_module
from sa0.session import normalize_protocol
from sa1 import agent, controller
from sa1.proposal import validate_response

LR = fixture_module.LR
MODEL = fixture_module.MODEL


def envelope(status="searched"):
    return {"proposal": copy.deepcopy(LR), "research": {
        "status": status, "summary": "公开资料仅作为假设依据。",
        "sources": ([{"url": "https://example.org/paper", "title": "Example",
                      "used": True, "reason": "学习率机制待实际验证"}] if status == "searched" else [])}}


def stream(value, searched=True):
    events = []
    if searched:
        events.append({"type": "item.completed", "item": {"id": "search1", "type": "web_search",
                                                              "query": "IR drop MAVI"}})
    events.extend([
        {"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(value)}},
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}},
    ])
    return "\n".join(json.dumps(event) for event in events)


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.settings = {"codex_path": "mock", "model": "gpt-6.1-sol", "reasoning_effort": "low",
                         "timeout_seconds": 180, "max_prompt_chars": 10000, "web_search": "live"}

    def test_search_permissions_and_audit(self):
        completed = SimpleNamespace(returncode=0, stderr="", stdout=stream(envelope()))
        with patch.object(agent.subprocess, "run", return_value=completed) as invoke:
            proposal = json.loads(agent.call_agent("input", self.folder / "agent_cli.log", self.settings))
        self.assertEqual(proposal, LR)
        command = invoke.call_args.args[0]
        self.assertIn('web_search="live"', command)
        self.assertIn("features.shell_tool=false", command)
        self.assertIn("features.unified_exec=false", command)
        self.assertIn("features.multi_agent=false", command)
        self.assertIn("--json", command)
        audit = json.loads((self.folder / "search_audit.json").read_text())
        self.assertEqual(audit["observed_search_event_count"], 1)
        self.assertEqual(audit["cli_usage"]["input_tokens"], 100)
        self.assertFalse(audit["source_claims_verified"])

    def test_false_search_claim_rejected_with_raw_reply_preserved(self):
        completed = SimpleNamespace(returncode=0, stderr="", stdout=stream(envelope(), False))
        with patch.object(agent.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "未记录"):
                agent.call_agent("input", self.folder / "agent_cli.log", self.settings)
        self.assertTrue((self.folder / "agent_envelope.txt").is_file())

    def test_search_not_needed_is_allowed(self):
        completed = SimpleNamespace(returncode=0, stderr="", stdout=stream(envelope("not_needed"), False))
        with patch.object(agent.subprocess, "run", return_value=completed):
            self.assertEqual(json.loads(agent.call_agent("input", self.folder / "agent_cli.log", self.settings)), LR)

    def test_unreported_search_rejected(self):
        completed = SimpleNamespace(returncode=0, stderr="", stdout=stream(envelope("not_needed")))
        with patch.object(agent.subprocess, "run", return_value=completed):
            with self.assertRaisesRegex(ValueError, "未如实"):
                agent.call_agent("input", self.folder / "agent_cli.log", self.settings)

    def test_timeout_preserves_events(self):
        error = subprocess.TimeoutExpired("mock", 180, output=b'{"type":"turn.started"}', stderr=b"timeout")
        with patch.object(agent.subprocess, "run", side_effect=error):
            with self.assertRaises(RuntimeError):
                agent.call_agent("input", self.folder / "agent_cli.log", self.settings)
        self.assertTrue((self.folder / "events.jsonl").is_file())

    def test_sources_reject_non_web_urls_and_invalid_types(self):
        value = envelope()
        value["research"]["sources"][0]["url"] = "file:///hidden/labels"
        with self.assertRaises(ValueError):
            validate_response(value)
        value = envelope()
        value["research"]["sources"][0]["used"] = 1
        with self.assertRaises(ValueError):
            validate_response(value)

    def test_live_setting_cannot_be_disabled(self):
        with self.assertRaises(ValueError):
            agent.call_agent("input", self.folder / "agent_cli.log", {**self.settings, "web_search": "disabled"})


class IntegrationTests(unittest.TestCase):
    def test_two_rounds_use_sa1_prompt_freeze_and_feedback(self):
        fixture = fixture_module.SessionTests("test_autonomous_lr_and_model_and_independent_sources")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        root = fixture.root
        (root / "sa1").mkdir()
        for filename in ("controller.py", "agent.py", "proposal.py", "PROMPT.md"):
            (root / "sa1" / filename).write_text(
                (controller.ROOT / "sa1" / filename).read_text(encoding="utf-8"), encoding="utf-8")
        operations = SimpleNamespace(**{name: getattr(controller, name) for name in dir(controller)
                                        if not name.startswith("__")})
        operations.PROMPT_PATH = root / "sa1/PROMPT.md"

        def reply(prompt, log, settings):
            (log.parent / "research.json").write_text(json.dumps(envelope()["research"]))
            (log.parent / "search_audit.json").write_text(
                '{"observed_search_event_count": 1, "cli_usage": {"input_tokens": 100, "output_tokens": 20}}')
            return json.dumps(LR)

        summary = fixture.run_case([reply, MODEL], operations=operations)
        self.assertEqual(len(fixture.train_calls), 3)
        self.assertIn("research", summary["history"][0]["record"])
        self.assertEqual(summary["history"][0]["record"]["agent_usage"]["reported_tokens"], 120)
        self.assertIn("https://example.org/paper", fixture.prompts[1])
        self.assertIn("联网 IR-drop", fixture.prompts[0])
        self.assertNotIn("不调用工具，不修改文件，不启动训练。", fixture.prompts[0])
        hashes = json.loads((root / "session/harness_hashes.json").read_text())
        self.assertIn("sa1/agent.py", hashes)
        fixture.run_case([], operations=operations)
        self.assertEqual(len(fixture.train_calls), 3)
        (root / "sa1/PROMPT.md").write_text("changed")
        with self.assertRaises(RuntimeError):
            fixture.run_case([], operations=operations)

    def test_config_matches_sa0_information_and_training(self):
        sa0 = json.loads((controller.ROOT / "sa0/config.json").read_text(encoding="utf-8"))
        sa1 = json.loads((controller.ROOT / "sa1/config.json").read_text(encoding="utf-8"))
        for key in (*controller.FIXED_RESULT_KEYS, "data_root", "train_manifest", "val_manifest",
                    "baseline_initial_lr", "max_trials", "max_agent_calls", "max_session_process_seconds",
                    "history_sessions", "history_results"):
            self.assertEqual(sa0[key], sa1[key], key)
        for key in ("model", "reasoning_effort", "timeout_seconds", "max_prompt_chars"):
            self.assertEqual(sa0["agent"][key], sa1["agent"][key], key)
        self.assertEqual(normalize_protocol(sa1)["agent"]["web_search"], "live")


if __name__ == "__main__":
    unittest.main()
