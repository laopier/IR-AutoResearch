"""Web-enabled CLI calls with raw JSONL events and source adoption records."""
import json
import subprocess
import time
from sa1.proposal import validate_response
from sa0.research.codex_cli import resolve


def parse_events(text):
    events = [json.loads(line) for line in text.splitlines() if line.strip()]
    if any(not isinstance(event, dict) for event in events):
        raise ValueError("CLI事件必须是JSON对象")
    messages = []
    searches = []
    usage = None
    for event in events:
        item = event.get("item", {})
        if not isinstance(item, dict):
            continue
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            messages.append(item["text"])
        if str(item.get("type", "")).startswith("web_search"):
            searches.append(event)
        if event.get("type") == "turn.completed":
            usage = event.get("usage")
    if not messages:
        raise ValueError("CLI未返回最终agent_message，请查看events.jsonl")
    return messages[-1], searches, usage


def call_agent(prompt, log_path, settings):
    if settings.get("web_search") != "live":
        raise ValueError("SA1 agent必须启用live搜索")
    paths = (log_path, log_path.with_name("events.jsonl"), log_path.with_name("agent_envelope.txt"))
    if any(path.exists() for path in paths):
        raise FileExistsError("调用记录已存在，不覆盖")
    if len(prompt) > settings["max_prompt_chars"]:
        raise ValueError("输入超过max_prompt_chars")
    codex_path = resolve(settings)
    command = [
        codex_path, "exec", "--sandbox", "read-only",
        "--ignore-user-config", "--ephemeral", "--color", "never", "--json",
        "-m", settings["model"], "-c", 'web_search="live"',
        "-c", "model_reasoning_effort=" + json.dumps(settings["reasoning_effort"]),
        "-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
        "-c", "features.multi_agent=false", "-",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(command, input=prompt, capture_output=True, text=True,
                                   encoding="utf-8", timeout=settings["timeout_seconds"])
    except subprocess.TimeoutExpired as error:
        for path, content in ((log_path, error.stderr), (paths[1], error.stdout)):
            content = content or ""
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            path.write_text(content, encoding="utf-8")
        raise RuntimeError("SA1 agent调用超时，已保存部分事件与错误") from error
    finally:
        log_path.with_suffix(".timing.json").write_text(
            json.dumps({"seconds": time.perf_counter() - started}), encoding="utf-8")
    log_path.write_text(completed.stderr, encoding="utf-8")
    paths[1].write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError(f"SA1 agent退出码{completed.returncode}，查看{log_path}和{paths[1]}")
    response, searches, usage = parse_events(completed.stdout)
    paths[2].write_text(response, encoding="utf-8")
    # Save observed tool evidence even when the proposed response is malformed.
    audit = {"web_search_mode": "live", "observed_search_events": searches,
             "observed_search_event_count": len(searches), "cli_usage": usage,
             "source_claims_verified": False, "resolved_codex_path": codex_path,
             "model": settings["model"], "reasoning_effort": settings["reasoning_effort"]}
    log_path.with_name("search_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    proposal, research = validate_response(json.loads(response))
    log_path.with_name("research.json").write_text(
        json.dumps(research, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if research["status"] == "searched" and not searches:
        raise ValueError("回复声称检索，但CLI未记录web_search事件，不能确认联网实验")
    if searches and research["status"] != "searched":
        raise ValueError("CLI记录了搜索事件，但research状态未如实报告")
    return json.dumps(proposal, ensure_ascii=False, allow_nan=False)
