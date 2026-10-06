import json
from pathlib import Path
def metrics_improved(baseline: dict, candidate: dict) -> bool:
    if candidate["mae_float"]<baseline["mae_float"] and candidate["nrms_official"]<=baseline["nrms_official"] and candidate["ssim_official"]>=baseline["ssim_official"]:
        return True
    else :return False

def within_budget(
    candidate:dict,
    max_training_seconds:float,
    max_peak_allocated_mib:float
)->bool:
    return candidate["training_seconds"]<=max_training_seconds and candidate["peak_allocated_mib"]<=max_peak_allocated_mib
def should_keep(
    candidate:dict,
    baseline:dict,
    max_training_seconds:float,
    max_peak_allocated_mib:float
)-> bool:
    return (metrics_improved(baseline["final_metrics"],candidate["final_metrics"]) and within_budget(candidate,max_training_seconds,max_peak_allocated_mib))
def evaluate_candidate(
    candidate:dict,
    baseline:dict,
    max_training_seconds:float,
    max_peak_allocated_mib:float,
    budget_status: str = "test_limits",
)-> dict:
    if candidate["steps"] != baseline["steps"]:
        raise ValueError("训练步数不同，不能直接比较")
    if budget_status not in {"test_limits", "formal_limits"}:
        raise ValueError("未知的预算状态")
    metrics_ok = metrics_improved(baseline["final_metrics"], candidate["final_metrics"])
    budget_ok = within_budget(candidate,max_training_seconds,max_peak_allocated_mib)
    if not metrics_ok:
        keep = False
        reason = "validation_metrics_not_improved"
    elif budget_status=="test_limits":
        keep = None
        reason = "metrics_improved_budget_pending"
    elif not budget_ok:
        keep = False
        reason = "resource_budget_exceeded"
    else:
        keep = True
        reason = "keep"
    decision = {
        "metrics_ok":metrics_ok,
        "budget_ok":budget_ok,
        "keep":keep,
        "reason":reason,
        "budget_status":budget_status
    }
    return decision

if __name__ == "__main__":
    baseline_path = Path("/mnt/d/Project/IR-AutoResearch/results/lr_baseline_100_learning_retry1/result.json")
    candidate_path = Path("/mnt/d/Project/IR-AutoResearch/results/sa0_agent_trial_002/result.json")
    with baseline_path.open("r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    with candidate_path.open("r", encoding="utf-8") as handle:
        candidate = json.load(handle)
    decision = evaluate_candidate(candidate, baseline, 1300, 6500)
    
    decision["budget_status"] = "test_limits"
    decision["budget_limit"] = {
        "max_training_seconds":1300,
        "max_peak_allocated_mib":6500
    }
    decision["baseline_result"] = str(baseline_path)
    decision["candidate_result"] = str(candidate_path)
    print(decision)
    decision_path = candidate_path.parent / "decision_learning_v2.json"
    with decision_path.open("x",encoding="utf-8") as handle:
        json.dump(decision,handle,indent=2,allow_nan=False)