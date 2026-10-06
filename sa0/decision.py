"""Frozen selection rule: all three metrics plus declared resource status."""


def evaluate_candidate(candidate, baseline, max_training_seconds,
                       max_peak_allocated_mib, budget_status="test_limits"):
    if budget_status not in {"test_limits", "formal_limits"}:
        raise ValueError("未知的预算状态")
    metrics_ok = (
        candidate["final_metrics"]["mae_float"] < baseline["final_metrics"]["mae_float"]
        and candidate["final_metrics"]["nrms_official"] <= baseline["final_metrics"]["nrms_official"]
        and candidate["final_metrics"]["ssim_official"] >= baseline["final_metrics"]["ssim_official"]
    )
    budget_ok = (
        candidate["training_seconds"] <= max_training_seconds
        and candidate["peak_allocated_mib"] <= max_peak_allocated_mib
    )
    if not metrics_ok:
        keep, reason = False, "validation_metrics_not_improved"
    elif budget_status == "test_limits":
        keep, reason = None, "metrics_improved_budget_pending"
    elif not budget_ok:
        keep, reason = False, "resource_budget_exceeded"
    else:
        keep, reason = True, "keep"
    return dict(metrics_ok=metrics_ok, budget_ok=budget_ok, keep=keep,
                reason=reason, budget_status=budget_status)
