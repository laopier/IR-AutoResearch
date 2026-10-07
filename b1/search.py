"""Generate the complete search plan before observing any candidate results."""
import math
import random

from sa0.session import normalize_protocol


def validate_config(value):
    if value.get("condition") != "B1":
        raise ValueError("B1必须声明condition=B1")
    if type(value.get("search_seed")) is not int or value["search_seed"] < 0:
        raise ValueError("search_seed必须为非负整数")
    space = value.get("search_space")
    if not isinstance(space, dict) or set(space) != {"initial_lr"}:
        raise ValueError("当前B1只允许预先定义initial_lr搜索空间")
    lr = space["initial_lr"]
    if not isinstance(lr, dict) or set(lr) != {"distribution", "min", "max"}:
        raise ValueError("学习率搜索空间必须包含distribution、min、max")
    if lr["distribution"] != "log_uniform":
        raise ValueError("当前仅支持log_uniform")
    low, high = lr["min"], lr["max"]
    if any(type(x) not in (int, float) or not math.isfinite(x) for x in (low, high)):
        raise ValueError("搜索边界必须为有限数值")
    if not 1e-7 <= low < high:
        raise ValueError("搜索范围必须满足1e-7 <= min < max")
    if "agent" in value or value.get("first_proposal_path"):
        raise ValueError("B1不接受agent配置或人工首轮提案")
    if value.get("history_sessions") or value.get("history_results"):
        raise ValueError("B1随机搜索不读取研究历史选择候选")
    # Validate the same fixed training/budget fields, without exposing agent settings.
    limits = dict(value)
    unlimited = "max_session_process_seconds" in value and value["max_session_process_seconds"] is None
    if unlimited:
        limits["max_session_process_seconds"] = value["process_timeout_seconds"] * (value["max_trials"] + 1)
    normalized = normalize_protocol({**limits, "allowed_kinds": ["learning_rate"], "max_agent_calls": 1,
                                     "initial_lr_upper_bound": high})
    if unlimited:
        normalized["max_session_process_seconds"] = None
    for name in ("agent", "max_agent_calls", "first_proposal_path", "history_sessions", "history_results"):
        normalized.pop(name, None)
    return normalized


def generate_plan(protocol):
    rng = random.Random(protocol["search_seed"])
    bounds = protocol["search_space"]["initial_lr"]
    seen = {protocol["baseline_initial_lr"]}
    trials = []
    attempts = 0
    while len(trials) < protocol["max_trials"]:
        attempts += 1
        if attempts > 100 * protocol["max_trials"]:
            raise ValueError("搜索空间无法生成足够不同的候选")
        lr = math.exp(rng.uniform(math.log(bounds["min"]), math.log(bounds["max"])))
        if not bounds["min"] <= lr <= bounds["max"] or lr <= 1e-7 or lr in seen:
            continue
        seen.add(lr)
        trials.append({"trial_number": len(trials) + 1, "initial_lr": lr})
    return {"method": "random_search", "search_seed": protocol["search_seed"],
            "search_space": protocol["search_space"], "trials": trials}
