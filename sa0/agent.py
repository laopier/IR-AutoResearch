"""Configured offline Codex proposal calls; never execute returned code here."""
import json
import subprocess
import time
from pathlib import Path


def call_agent(prompt: str, log_path: Path, settings: dict) -> str:
    if log_path.exists():
        raise FileExistsError(log_path)
    if len(prompt) > settings["max_prompt_chars"]:
        raise ValueError("输入超过max_prompt_chars，请缩减历史摘要")
    command = [
        settings["codex_path"], "exec", "--sandbox", "read-only",
        "--ignore-user-config", "--ephemeral", "--color", "never",
        "-m", settings["model"],
        "-c", 'web_search="disabled"',
        "-c", "model_reasoning_effort=" + json.dumps(settings["reasoning_effort"]),
        "-c", "features.shell_tool=false",
        "-c", "features.unified_exec=false",
        "-c", "features.multi_agent=false",
        "-",
    ]
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command, input=prompt, capture_output=True, text=True,
            encoding="utf-8", timeout=settings["timeout_seconds"],
        )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr or ""
        stdout = error.stdout or ""
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        log_path.write_text(stderr, encoding="utf-8")
        log_path.with_suffix(".partial_response.txt").write_text(stdout, encoding="utf-8")
        raise RuntimeError("agent调用超时，已保存可取得的输出") from error
    finally:
        log_path.with_suffix(".timing.json").write_text(
            json.dumps({"seconds": time.perf_counter() - started}), encoding="utf-8"
        )
    log_path.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        log_path.with_suffix(".failed_response.txt").write_text(completed.stdout, encoding="utf-8")
        raise RuntimeError(f"agent退出码{completed.returncode}，请查看{log_path}")
    return completed.stdout
