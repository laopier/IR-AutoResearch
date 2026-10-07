
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from sa0.research.agent import call


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "call"
        self.settings = {"codex_path": "mock", "model": "gpt-6.1-sol", "reasoning_effort": "low",
                         "web_search": "disabled", "timeout_seconds": 900, "max_prompt_chars": 1000}

    def events(self, value, searched=False):
        events = [{"type": "item.completed", "item": {"type": "web_search", "query": "public paper"}}] if searched else []
        events += [{"type": "item.completed", "item": {"type": "agent_message", "text": json.dumps(value)}},
                   {"type": "turn.completed", "usage": {"input_tokens": 20, "output_tokens": 10}}]
        return "\n".join(json.dumps(event) for event in events)

    def test_offline_arbitrary_planning_payload_and_usage(self):
        output = self.events({"hypotheses": []})
        with patch("sa0.research.agent.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=output, stderr="")) as mock:
            self.assertEqual(call("plan", self.folder, self.settings), {"hypotheses": []})
        self.assertIn('web_search="disabled"', mock.call_args.args[0])
        record = json.loads((self.folder / "record.json").read_text(encoding="utf-8"))
        self.assertEqual(record["reported_tokens"], 30)

    def test_online_payload_and_observed_sources(self):
        value = {"payload": {"action": "finalize"}, "research": {"status": "searched", "summary": "evidence",
            "sources": [{"url": "https://example.org/paper", "title": "paper", "used": True, "reason": "mechanism"}]}}
        with patch("sa0.research.agent.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=self.events(value, True), stderr="")):
            self.assertEqual(call("input", self.folder, {**self.settings, "web_search": "live"}), {"action": "finalize"})
        self.assertTrue((self.folder / "research.json").exists())

    def test_online_claim_without_event_rejected(self):
        value = {"payload": {}, "research": {"status": "searched", "summary": "claim", "sources": [{"url": "https://example.org", "title": "x", "used": False, "reason": "x"}]}}
        with patch("sa0.research.agent.subprocess.run", return_value=SimpleNamespace(returncode=0, stdout=self.events(value), stderr="")):
            with self.assertRaises(ValueError):
                call("input", self.folder, {**self.settings, "web_search": "live"})
        self.assertTrue((self.folder / "response.txt").exists())

    def test_failed_call_preserves_usage(self):
        with patch("sa0.research.agent.subprocess.run", return_value=SimpleNamespace(returncode=1, stdout=self.events({}), stderr="error")):
            with self.assertRaises(RuntimeError):
                call("input", self.folder, self.settings)
        self.assertEqual(json.loads((self.folder / "record.json").read_text(encoding="utf-8"))["reported_tokens"], 30)

    def test_timeout_unknown_usage_not_zero(self):
        with patch("sa0.research.agent.subprocess.run", side_effect=subprocess.TimeoutExpired("mock", 900, output=b'partial', stderr=b'timeout')):
            with self.assertRaises(subprocess.TimeoutExpired):
                call("input", self.folder, self.settings)
        record = json.loads((self.folder / "record.json").read_text(encoding="utf-8"))
        self.assertIsNone(record["reported_tokens"])
        self.assertGreaterEqual(record["seconds"], 0)


if __name__ == "__main__":
    unittest.main()
