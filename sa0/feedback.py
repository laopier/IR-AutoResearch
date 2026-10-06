"""Build a reviewable, saved input for a single proposal call."""
import json
from pathlib import Path


def write_agent_input(instructions: str, feedback: dict, request: str,
                      output_path: Path) -> Path:
    text = (instructions + "\n\n## 实验反馈\n\n"
            + json.dumps(feedback, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n\n## 本轮请求\n\n" + request)
    with output_path.open("x", encoding="utf-8") as handle:
        handle.write(text)
    return output_path
