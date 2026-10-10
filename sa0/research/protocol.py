import copy
import math

METRICS = ("mae_float", "nrms_official", "ssim_official")
STAGES = ("smoke", "low", "full", "confirmation")


def validate(config):
    p = copy.deepcopy(config)
    if p.get("baseline_cache_dir") is not None and (not isinstance(p["baseline_cache_dir"], str) or not p["baseline_cache_dir"]):
        raise ValueError("baseline_cache_dir须为空或非空路径字符串")
    p.setdefault("full_selection_enabled", False)
    if type(p["full_selection_enabled"]) is not bool:
        raise ValueError("full_selection_enabled须为布尔值")
    if p.get("version") != 4 or p.get("condition") not in {"SA0", "SA1"}:
        raise ValueError("需要version=4及SA0/SA1条件")
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
    required = p["agent"].get("required_search_purposes", [])
    if not isinstance(required, list) or any(x not in {"research_plan"} for x in required):
        raise ValueError("required_search_purposes当前仅允许research_plan")
    if p["condition"] == "SA0" and required:
        raise ValueError("SA0禁止要求联网检索")
    if p["condition"] == "SA1" and required != ["research_plan"]:
        raise ValueError("SA1 V4必须在research_plan阶段真实检索")
    gate = p.get("low_gate")
    if not isinstance(gate, dict) or set(gate) != {
        "seed0_max_regression", "promotion_mean_tolerance", "promotion_per_seed_max_regression"
    }:
        raise ValueError("low_gate字段不完整")
    expected_metrics = set(METRICS)
    for name in ("seed0_max_regression", "promotion_per_seed_max_regression"):
        values = gate[name]
        if not isinstance(values, dict) or set(values) != expected_metrics:
            raise ValueError(f"{name}必须覆盖三项指标")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f"{name}必须为非负有限数")
    tolerances = gate["promotion_mean_tolerance"]
    if not isinstance(tolerances, dict) or set(tolerances) != {"nrms_official", "ssim_official"}:
        raise ValueError("promotion_mean_tolerance须覆盖NRMS和SSIM")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in tolerances.values()):
        raise ValueError("promotion_mean_tolerance必须为非负有限数")
    portfolio = p.get("portfolio")
    if not isinstance(portfolio, dict) or set(portfolio) != {
        "initial_items", "min_seed0_before_seed1", "max_seed1_candidates",
        "stagnant_seed0_limit", "score_weights"
    }:
        raise ValueError("portfolio字段不完整")
    for key in ("initial_items", "min_seed0_before_seed1", "max_seed1_candidates", "stagnant_seed0_limit"):
        if type(portfolio[key]) is not int or portfolio[key] <= 0:
            raise ValueError(f"portfolio.{key}须为正整数")
    if not 4 <= portfolio["initial_items"] <= 6:
        raise ValueError("初始干预组合须含4～6项")
    if portfolio["min_seed0_before_seed1"] > portfolio["initial_items"]:
        raise ValueError("seed1前置seed0数量不能超过初始组合大小")
    weights = portfolio["score_weights"]
    if not isinstance(weights, dict) or set(weights) != set(METRICS):
        raise ValueError("score_weights须覆盖三项指标")
    if any(type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in weights.values()) or sum(weights.values()) <= 0:
        raise ValueError("score_weights须为非负有限数且总和大于0")
    fractions = p.get("probe_fractions")
    if (not isinstance(fractions, list) or not fractions or any(
        type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1 for value in fractions
    ) or fractions != sorted(set(fractions)) or fractions[-1] != 1):
        raise ValueError("probe_fractions须为递增唯一的(0,1]列表且包含1")
    return p


def improved(base, cand):
    return cand["mae_float"] < base["mae_float"] and cand["nrms_official"] <= base["nrms_official"] and cand["ssim_official"] >= base["ssim_official"]


def metric_delta(base, cand):
    return {key: cand[key] - base[key] for key in METRICS}


def within_regression_caps(delta, caps):
    return (delta["mae_float"] <= caps["mae_float"]
            and delta["nrms_official"] <= caps["nrms_official"]
            and delta["ssim_official"] >= -caps["ssim_official"])


def promising(base, cand, gate):
    return within_regression_caps(metric_delta(base, cand), gate["seed0_max_regression"])


def screening_score(base, cand, weights):
    """Higher is better; zero is the matched B0 reference."""
    delta = metric_delta(base, cand)
    scale = {key: max(abs(base[key]), 1e-12) for key in METRICS}
    terms = {
        "mae_float": -delta["mae_float"] / scale["mae_float"],
        "nrms_official": -delta["nrms_official"] / scale["nrms_official"],
        "ssim_official": delta["ssim_official"] / scale["ssim_official"],
    }
    total = sum(weights.values())
    return sum(weights[key] * terms[key] for key in METRICS) / total


def aggregate_low_gate(pairs, gate):
    if len(pairs) != 2 or {seed for seed, _, _ in pairs} != {0, 1}:
        return False
    deltas = [metric_delta(base, cand) for _, base, cand in pairs]
    if not all(within_regression_caps(delta, gate["promotion_per_seed_max_regression"]) for delta in deltas):
        return False
    mean = {key: sum(delta[key] for delta in deltas) / len(deltas) for key in METRICS}
    tolerance = gate["promotion_mean_tolerance"]
    return (mean["mae_float"] < 0
            and mean["nrms_official"] <= tolerance["nrms_official"]
            and mean["ssim_official"] >= -tolerance["ssim_official"])


def mean_metrics(results):
    return {key: sum(r["final_metrics"][key] for r in results) / len(results) for key in METRICS}


def hypothesis(value, expected_id=None):
    required = {"hypothesis_id", "observation", "mechanism", "proposed_change", "discriminating_experiments",
                "expected_metrics", "falsification_condition", "risks", "estimated_cost"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("假设字段不完整或包含额外字段")
    if expected_id and value["hypothesis_id"] != expected_id:
        raise ValueError("初始假设ID必须从H1连续编号")
    for key in required:
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"假设{key}必须为非空文字")
    return {**value, "status": "open", "no_progress_streak": 0}


def plan(value):
    if not isinstance(value, dict) or set(value) != {"hypotheses", "primary_hypothesis_id"}:
        raise ValueError("计划必须包含hypotheses和primary_hypothesis_id")
    hs = value["hypotheses"]
    if not isinstance(hs, list) or not 3 <= len(hs) <= 5:
        raise ValueError("开始时须登记3～5个假设")
    result = {f"H{i}": hypothesis(h, f"H{i}") for i, h in enumerate(hs, 1)}
    if len({(h["mechanism"].strip().casefold(), h["proposed_change"].strip().casefold()) for h in result.values()}) != len(result):
        raise ValueError("初始假设不能仅换ID重复同一机制及改动")
    if value["primary_hypothesis_id"] not in result:
        raise ValueError("主方向不存在")
    return result


PLAN_ITEM_FIELDS = {
    "plan_item_id", "hypothesis_id", "parent_candidate_id", "objective", "mechanism",
    "delta", "pre_checks", "run_spec", "probes", "post_condition", "rollback_condition"
}


def plan_item(value, hypotheses, candidates, expected_id=None):
    if not isinstance(value, dict) or set(value) != PLAN_ITEM_FIELDS:
        raise ValueError("干预项字段不完整或包含额外字段")
    key = value["plan_item_id"]
    if (not isinstance(key, str) or not key.startswith("P") or not key[1:].isdigit()
            or (expected_id is not None and key != expected_id)):
        raise ValueError("plan_item_id须为连续P数字")
    if value["hypothesis_id"] not in hypotheses:
        raise ValueError("干预项须引用已登记假设")
    if value["parent_candidate_id"] not in candidates:
        raise ValueError("干预项父候选不存在")
    for field in ("objective", "mechanism", "delta", "run_spec", "post_condition", "rollback_condition"):
        if not isinstance(value[field], str) or not value[field].strip():
            raise ValueError(f"干预项{field}须为非空文字")
    for field in ("pre_checks", "probes"):
        if (not isinstance(value[field], list) or not value[field]
                or any(not isinstance(item, str) or not item.strip() for item in value[field])):
            raise ValueError(f"干预项{field}须为非空文字列表")
    return copy.deepcopy(value)


def intervention_portfolio(value, hypotheses, candidates, expected_size):
    if not isinstance(value, dict) or set(value) != {"items", "selection_rationale"}:
        raise ValueError("干预组合须包含items和selection_rationale")
    if not isinstance(value["selection_rationale"], str) or not value["selection_rationale"].strip():
        raise ValueError("干预组合须说明选择理由")
    items = value["items"]
    if not isinstance(items, list) or len(items) != expected_size:
        raise ValueError(f"初始干预组合须恰好包含{expected_size}项")
    result = {}
    for index, item in enumerate(items, 1):
        parsed = plan_item(item, hypotheses, candidates, f"P{index}")
        if parsed["parent_candidate_id"] != "B0":
            raise ValueError("初始干预组合必须从B0分叉")
        result[parsed["plan_item_id"]] = parsed
    if len({(x["hypothesis_id"], x["delta"].strip().casefold()) for x in result.values()}) != len(result):
        raise ValueError("初始干预组合不能重复同一假设和delta")
    return result
