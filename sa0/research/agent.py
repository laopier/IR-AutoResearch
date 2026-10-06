"""Arbitrary workflow actions; actual CLI JSON usage, including failed calls."""
import json
import subprocess
import time
from sa1.agent import parse_events


def call(prompt, folder, settings):
    folder.mkdir()
    (folder / "input.md").write_text(prompt, encoding="utf-8")
    mode = settings["web_search"]
    command = [settings["codex_path"], "exec", "--sandbox", "read-only", "--ignore-user-config",
               "--ephemeral", "--color", "never", "--json", "-m", settings["model"],
               "-c", "web_search=" + json.dumps(mode), "-c", "model_reasoning_effort=" + json.dumps(settings["reasoning_effort"]),
               "-c", "features.shell_tool=false", "-c", "features.unified_exec=false", "-c", "features.multi_agent=false", "-"]
    started = time.perf_counter()
    record = {"status": "running", "reported_tokens": None, "usage": None}
    (folder / "record.json").write_text(json.dumps(record), encoding="utf-8")
    stdout, stderr = "", ""
    try:
        if len(prompt) > settings["max_prompt_chars"]:
            raise ValueError("输入超过max_prompt_chars")
        completed = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                   encoding="utf-8", timeout=settings["timeout_seconds"])
        stdout, stderr = completed.stdout, completed.stderr
        record["returncode"] = completed.returncode
        # Preserve partial usage even if no final response or a non-zero CLI exit exists.
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") == "turn.completed":
                usage = event.get("usage")
                record["usage"] = usage
                if isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens")):
                    record["reported_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        if completed.returncode:
            raise RuntimeError(f"CLI退出码{completed.returncode}")
        response, searches, _ = parse_events(stdout)
        (folder / "response.txt").write_text(response, encoding="utf-8")
        value = json.loads(response)
        if mode == "live":
            from sa1.proposal import validate_research
            if not isinstance(value, dict) or set(value) != {"payload", "research"}:
                raise ValueError("SA1必须返回payload及research")
            research = validate_research(value["research"])
            (folder / "research.json").write_text(json.dumps(research, ensure_ascii=False, indent=2), encoding="utf-8")
            if (research["status"] == "searched") != bool(searches):
                raise ValueError("检索声明与CLI搜索事件不一致")
            value = value["payload"]
        elif searches:
            raise ValueError("离线调用意外产生检索事件")
        record["search_events"] = searches
        record["source_claims_verified"] = False
        record["status"] = "completed"
        return value
    except subprocess.TimeoutExpired as error:
        stdout, stderr = error.stdout or "", error.stderr or ""
        record.update(status="timed_out", error="CLI调用超时")
        raise
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for name, content in (("events.jsonl", stdout), ("cli.log", stderr)):
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            (folder / name).write_text(content, encoding="utf-8")
        record["seconds"] = time.perf_counter() - started
        from sa0.session import atomic_json
        atomic_json(folder / "record.json", record)
