"""Deterministic compact research cards built from the frozen local problem."""
import ast
import csv
from pathlib import Path


def _symbols(path):
    if not Path(path).is_file():
        return []
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    return [node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]


def _manifest_size(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return sum(1 for row in csv.reader(handle) if row)


def build(config, frozen):
    stages = {}
    for name, spec in config["stages"].items():
        stages[name] = {
            "steps": spec["steps"],
            "train_examples": _manifest_size(spec["train_manifest"]),
            "validation_examples": _manifest_size(spec["val_manifest"]),
            "train_manifest": spec["train_manifest"],
            "validation_manifest": spec["val_manifest"],
        }
    return {
        "problem": {
            "task": "24-channel physical-design feature tensor to one IR-drop map",
            "input_contract": "dataset returns [1,24,H,W] per sample; transforms must preserve shape and finite values",
            "target_contract": "one [H,W] float map; labels and manifests are frozen",
        },
        "model": {
            "symbols": _symbols(Path(frozen) / "train/mavi.py"),
            "editable_files": ["train/mavi.py", "train/experiment.py", "train/feature_transform.py"],
            "interface_constraints": "build_model and prediction shape remain compatible; no new dependency or checkpoint initialization",
        },
        "training": {
            "baseline_recipe": {"initial_lr": config["baseline_initial_lr"], "optimizer": "native AdamW", "loss": "native L1", "lr_horizon_steps": config["lr_horizon_steps"]},
            "batch_size": config["batch_size"],
            "stages": stages,
            "initialization": "fresh for every experiment",
        },
        "evaluation": {
            "metrics": ["mae_float", "nrms_official", "ssim_official"],
            "strict_success": "MAE decreases, NRMS does not increase, SSIM does not decrease",
            "selection_checkpoint": "training-end checkpoint only; probes are diagnostic and cannot select weights",
            "symbols": _symbols(Path(frozen) / "prepare/evaluator.py"),
        },
        "research_surface": {
            "workflow": "hypothesis -> localized intervention plan -> source/config candidate -> smoke/low/full -> paired confirmation",
            "forbidden": ["change labels", "change evaluator", "use hidden results", "resume parent weights", "candidate-only best checkpoint selection"],
        },
    }


def markdown(cards):
    lines = ["# Research cards (deterministic)", ""]
    for section, values in cards.items():
        lines += [f"## {section}", ""]
        for key, value in values.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
    return "\n".join(lines)
