"""Shared pre-training gate for the audited 128/32 pilot (no GPU imports)."""
import csv
import hashlib
import json
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(data_root, report_path, specs):
    root, report_file = Path(data_root).resolve(), Path(report_path)
    if not report_file.is_file():
        raise RuntimeError("128/32数据仍在下载或处理，DATASET审计未生成，禁止启动训练")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if report.get("ready") is not True or report.get("train_samples") != 128 or report.get("validation_samples") != 32:
        raise RuntimeError("128/32数据审计未通过，禁止启动训练")
    selection_file = report_file.parent / "SELECTION.json"
    if sha(selection_file) != report.get("selection_sha256") or len(report.get("files", {})) != 320:
        raise ValueError("抽样哈希不一致或数组审计不完整")
    samples = json.loads(selection_file.read_text(encoding="utf-8"))["samples"]
    expected = {split: {(r["feature"], r["label"]) for r in rows} for split, rows in samples.items()}
    if len(expected["train"]) != 128 or len(expected["validation"]) != 32:
        raise ValueError("抽样清单数量不正确")
    for spec in specs:
        for field, split in (("train_manifest", "train"), ("val_manifest", "validation")):
            with Path(spec[field]).open(encoding="utf-8", newline="") as f:
                rows = [tuple(row) for row in csv.reader(f)]
            if len(rows) != len(expected[split]) or set(rows) != expected[split]:
                raise ValueError("配置清单与抽样审计不一致")
    for name, info in report["files"].items():
        path = (root / name).resolve()
        path.relative_to(root)
        if sha(path) != info["sha256"]:
            raise ValueError("数据数组与审计哈希不一致")
    return report
