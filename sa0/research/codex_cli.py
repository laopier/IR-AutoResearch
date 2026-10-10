"""Resolve the moving Codex desktop CLI installation and record the choice."""
from pathlib import Path


def resolve(settings):
    configured = settings.get("codex_path")
    if configured and configured != "auto":
        return str(configured)
    root = Path(settings.get("codex_bin_root", ""))
    if not root.is_dir():
        raise FileNotFoundError(f"Codex bin root不存在：{root}")
    candidates = [path for path in root.glob("*/codex.exe") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"Codex bin root下没有codex.exe：{root}")
    return str(max(candidates, key=lambda path: path.stat().st_mtime_ns))
