import copy
import math

METRICS = ("mae_float", "nrms_official", "ssim_official")
STAGES = ("smoke", "low", "full", "confirmation")


def validate(config):
    p = copy.deepcopy(config)
    p.setdefault("full_selection_enabled", False)
    if type(p["full_selection_enabled"]) is not bool:
        raise ValueError("full_selection_enabled须为布尔值")
    if p.get("version") != 2 or p.get("condition") not in {"SA0", "SA1"}:
        raise ValueError("需要version=2及SA0/SA1条件")
    if p.get("mode") != "workflow_validation":
        raise ValueError("当前版本只开放本地流程验证；正式数据、资源预算与隐藏入口尚未冻结")
    for name in ("max_gpu_tasks", "max_exploration_tasks", "batch_size", "val_batch_size", "lr_horizon_steps"):
        if type(p.get(name)) is not int or p[name] <= 0:
            raise ValueError(f"{name}必须为正整数")
    if p["max_exploration_tasks"] >= p["max_gpu_tasks"]:
        raise ValueError("必须保留确认任务额度")
    if p["full_selection_enabled"]:
        for key in ("max_screening_tasks", "full_task_reserve", "max_full_candidates"):
            if type(p.get(key)) is not int or p[key] <= 0:
                raise ValueError(f"{key}须为正整数")
        if p["max_full_candidates"] > 2 or p["full_task_reserve"] < p["max_full_candidates"]:
            raise ValueError("full最多选择2个候选，预留任务须覆盖选择数量")
        if 1 + p["max_screening_tasks"] + p["full_task_reserve"] > p["max_exploration_tasks"]:
            raise ValueError("探索额度必须容纳初始化B0、screening及full储备")
        if p["max_gpu_tasks"] - p["max_exploration_tasks"] < 4:
            raise ValueError("确认至少预留4次新训练（seed0匹配结果复用）")
    if p.get("confirmation_seeds") != [0, 1, 2] or p.get("low_seeds") != [0, 1]:
        raise ValueError("当前协议使用low seed0/1及确认seed0/1/2")
    for stage in STAGES:
        s = p["stages"][stage]
        if type(s["steps"]) is not int or not 0 < s["steps"] <= p["lr_horizon_steps"]:
            raise ValueError(f"{stage}步数无效")
        for key in ("train_manifest", "val_manifest"):
            if not isinstance(s.get(key), str) or not s[key]:
                raise ValueError(f"{stage}缺少{key}")
    if p["stages"]["confirmation"] != p["stages"]["full"]:
        raise ValueError("确认必须与模拟完整训练条件一致")
    for name in ("baseline_initial_lr", "initial_lr_upper_bound", "worker_timeout_seconds"):
        if type(p[name]) not in (int, float) or not math.isfinite(p[name]) or p[name] <= 0:
            raise ValueError(f"{name}无效")
    expected = "live" if p["condition"] == "SA1" else "disabled"
    if p["agent"].get("web_search") != expected:
        raise ValueError("联网权限与研究条件不一致")
    return p


def improved(base, cand):
    return cand["mae_float"] < base["mae_float"] and cand["nrms_official"] <= base["nrms_official"] and cand["ssim_official"] >= base["ssim_official"]


def mean_metrics(results):
    return {key: sum(r["final_metrics"][key] for r in results) / len(results) for key in METRICS}


def hypothesis(value, expected_id=None):
    required = {"hypothesis_id", "observation", "mechanism", "proposed_change", "discriminating_experiments",
                "expected_metrics", "falsification_condition", "risks", "estimated_cost"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("假设字段不完整或包含额外字段")
    if expected_id and value["hypothesis_id"] != expected_id:
        raise ValueError("初始假设必须为H1/H2/H3")
    for key in required:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"假设{key}必须为非空文字")
    return {**value, "status": "open", "no_progress_streak": 0}


def plan(value):
    if not isinstance(value, dict) or set(value) != {"hypotheses", "primary_hypothesis_id"}:
        raise ValueError("计划必须包含hypotheses和primary_hypothesis_id")
    hs = value["hypotheses"]
    if not isinstance(hs, list) or len(hs) != 3:
        raise ValueError("开始时恰好三个假设")
    result = {f"H{i}": hypothesis(h, f"H{i}") for i, h in enumerate(hs, 1)}
    if len({(h["mechanism"].strip().casefold(), h["proposed_change"].strip().casefold()) for h in result.values()}) != 3:
        raise ValueError("初始假设不能仅换ID重复同一机制及改动")
    if value["primary_hypothesis_id"] not in result:
        raise ValueError("主方向不存在")
    return result
