"""Explicit source/config inheritance, including conflict-checked combinations."""
import ast
import copy
import math
from pathlib import Path
import shutil

ALLOWED_FILES = {"train/mavi.py", "train/experiment.py", "train/feature_transform.py"}


def check_tree(nodes):
    visited, active = set(), set()
    def visit(key):
        if key not in nodes:
            raise ValueError("父候选不存在")
        if key in active:
            raise ValueError("候选谱系存在环")
        if key in visited:
            return
        active.add(key)
        for parent in nodes[key].get("parent_candidate_ids", []):
            visit(parent)
        active.remove(key)
        visited.add(key)
    for key in nodes:
        visit(key)


def recipe(value, upper):
    if not isinstance(value, dict) or set(value) - {"initial_lr", "loss", "optimizer", "scheduler"}:
        raise ValueError("候选配置只能修改initial_lr/loss/optimizer/scheduler")
    for key, val in value.items():
        if key == "initial_lr":
            if type(val) not in (int, float) or not math.isfinite(val) or not 1e-7 < val <= upper:
                raise ValueError("初始学习率越界")
        else:
            if val is None:
                continue  # explicitly remove an inherited override
            if not isinstance(val, dict):
                raise ValueError(f"{key}必须为对象")
            names = {"loss": {"l1", "mse", "smooth_l1"}, "optimizer": {"adamw", "adam", "sgd"},
                     "scheduler": {"cosine", "constant"}}
            fields = {"loss": {"name", "scale", "beta"}, "optimizer": {"name", "weight_decay", "momentum"},
                      "scheduler": {"name", "min_lr"}}
            if val.get("name") not in names[key] or set(val) - fields[key]:
                raise ValueError(f"{key}类型或参数未授权")
            for field, number in val.items():
                if field != "name" and (type(number) not in (int, float) or not math.isfinite(number) or number < 0):
                    raise ValueError(f"{key}.{field}无效")
            if key == "loss" and (val.get("scale", 100) <= 0 or val.get("beta", .01) <= 0):
                raise ValueError("loss scale/beta必须为正数")
            if key == "optimizer" and val.get("momentum", 0) >= 1:
                raise ValueError("momentum必须小于1")
    return value


def check_code(path, text, base):
    tree = ast.parse(text)
    if path == "train/experiment.py":
        # Loss/optimizer/scheduler only; build_model/train_batch and imports stay frozen.
        mutable = {"compute_loss", "build_optimizer", "build_scheduler", "research_lr"}
        def locked(source):
            return [ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                    if not isinstance(n, ast.FunctionDef) or n.name not in mutable]
        if locked(text) != locked(base):
            raise ValueError("train/experiment.py只允许修改loss、optimizer、scheduler函数")
    return tree


def materialize(c, config, state, value, destination):
    required = {"candidate_id", "parent_candidate_ids", "hypothesis_ids", "change", "expected_effect",
                "falsification_condition", "estimated_gpu_seconds", "mechanism_test"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("候选必须声明谱系、改动、预期、反证与机制检验字段")
    key = value["candidate_id"]
    if not isinstance(key, str) or not key.startswith("C") or not key[1:].isdigit() or key in state["candidates"]:
        raise ValueError("新候选ID须为未使用的C数字")
    parents, hs = value["parent_candidate_ids"], value["hypothesis_ids"]
    if not isinstance(parents, list) or not parents or len(set(parents)) != len(parents):
        raise ValueError("必须显式声明不同父候选")
    if any(p not in state["candidates"] for p in parents):
        raise ValueError("父候选不存在")
    if len(parents) > 1:
        for parent in parents:
            if parent != "B0" and not any(t.get("scientific_valid") and t.get("candidate_id") == parent
                                          and (t.get("metrics_ok") or "mechanistic_progress" in state["interpretations"].get(t["task_id"], {}).get("labels", []))
                                          for t in state["tasks"]):
                raise ValueError("组合的每个父机制须先有独立实验证据")
    if not isinstance(hs, list) or not hs or any(h not in state["hypotheses"] for h in hs):
        raise ValueError("hypothesis_ids必须引用已登记方向")
    if any(state["hypotheses"][h]["status"] != "open" for h in hs):
        raise ValueError("方向已关闭或需要重新评估")
    if any(not any(h in n.get("hypothesis_ids", []) for n in state["candidates"].values()) for h in hs) and parents != ["B0"]:
        raise ValueError("每个新假设的首个候选须从B0分叉")
    for name in ("expected_effect", "falsification_condition"):
        if not isinstance(value[name], str) or not value[name].strip():
            raise ValueError(f"{name}不能为空")
    cost = value["estimated_gpu_seconds"]
    if type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0:
        raise ValueError("预计成本必须为非负有限值")
    change = value["change"]
    if not isinstance(change, dict) or set(change) != {"config", "edits", "resolved_sources"}:
        raise ValueError("change必须含config、edits、resolved_sources")
    own = recipe(change["config"], config["initial_lr_upper_bound"])
    if not isinstance(change["edits"], list) or not isinstance(change["resolved_sources"], dict):
        raise ValueError("edits/resolved_sources类型无效")
    if set(change["resolved_sources"]) - ALLOWED_FILES:
        raise ValueError("组合源码路径未授权")
    nodes = state["candidates"]
    check_tree({**nodes, key: value})
    merged = {}
    for parent in parents:
        for name, val in nodes[parent]["recipe"].items():
            if name in merged and merged[name] != val and name not in own:
                raise ValueError(f"父配置冲突须显式覆盖：{name}")
            merged[name] = val
    merged.update(own)
    merged = {k: v for k, v in merged.items() if v is not None}
    recipe(merged, config["initial_lr_upper_bound"])
    edits = change["edits"]
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"path", "old", "new"} or edit["path"] not in ALLOWED_FILES:
            raise ValueError("源码修改路径或字段未授权")
        if not isinstance(edit["old"], str) or not edit["old"] or not isinstance(edit["new"], str):
            raise ValueError("源码替换需非空old及字符串new")
    base = Path(nodes["B0"]["workspace"])
    shutil.copytree(base, destination)
    for path in ALLOWED_FILES:
        original = (base / path).read_text(encoding="utf-8")
        changes = {(Path(nodes[p]["workspace"]) / path).read_text(encoding="utf-8") for p in parents}
        changes.discard(original)
        if len(changes) > 1 and path not in change["resolved_sources"]:
            raise ValueError(f"父源码冲突须提供resolved_sources：{path}")
        text = next(iter(changes)) if changes else original
        text = change["resolved_sources"].get(path, text)
        for edit in edits:
            if edit["path"] == path:
                if text.count(edit["old"]) != 1:
                    raise ValueError("old须恰好匹配一次")
                text = text.replace(edit["old"], edit["new"], 1)
        check_code(path, text, original)
        (destination / path).write_text(text, encoding="utf-8")
    import ast
    def function(source, name):
        return [ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name]
    old = (base / "train/experiment.py").read_text(encoding="utf-8")
    new = (destination / "train/experiment.py").read_text(encoding="utf-8")
    for key, function_name in (("loss", "compute_loss"), ("optimizer", "build_optimizer"), ("scheduler", "build_scheduler")):
        if key in merged and function(old, function_name) != function(new, function_name):
            raise ValueError(f"{key}配置会覆盖源码修改，须先用null移除配置覆盖")
    if merged == nodes[parents[0]]["recipe"] and c.source_hashes(destination) == nodes[parents[0]]["source_hashes"]:
        raise ValueError("候选没有实际改动")
    def semantic(workspace):
        return {path: ast.dump(ast.parse((Path(workspace) / path).read_text(encoding="utf-8")), include_attributes=False)
                for path in ALLOWED_FILES}
    if value["mechanism_test"] is None and any(merged == old["recipe"] and semantic(destination) == semantic(old["workspace"]) for old in nodes.values()):
        raise ValueError("候选与已有配置/模型等价；消融须明确登记机制检验")
    test = value["mechanism_test"]
    if test is not None:
        if (not isinstance(test, dict) or set(test) != {"reference_candidate_id", "metric", "direction", "min_delta"}
                or test["reference_candidate_id"] not in parents or test["metric"] not in {"mae_float", "nrms_official", "ssim_official"}
                or test["direction"] not in {"increase", "decrease"}
                or type(test["min_delta"]) not in (int, float) or not math.isfinite(test["min_delta"]) or test["min_delta"] < 0):
            raise ValueError("机制检验须提前登记父对照、指标、方向和最小差值")
    return {**copy.deepcopy(value), "recipe": merged, "workspace": str(destination),
            "source_hashes": c.source_hashes(destination)}
