import os
import json
from pathlib import Path
import subprocess
import sys
import math
from sa0.decision_learning import evaluate_candidate
def load_result(path:Path)->dict:
    with path.open("r",encoding="utf-8") as handle:
        return json.load(handle)
def run_training(module_name:str,config_path:Path,log_path:Path)->int:
    child_env = os.environ.copy()
    child_env["IR_SA0_CONFIG"] = str(config_path)
    command = [
        sys.executable,
        "-u",
        "-m",
        module_name,
    ]
    try:
        with log_path.open("x", encoding="utf-8") as log_handle:
            complete = subprocess.run(
                command,
                env=child_env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
    )
            
        return complete.returncode
    except KeyboardInterrupt:
        print("用户手动中断训练")
        raise
def build_feedback(
    baseline: dict,
    candidate: dict,
    trial_record: dict,
) -> dict:
    
    result = {
        "baseline":baseline,
        "candidate":candidate,
        "trial_record":trial_record
    }
    return result
def validate_proposal(proposal: dict) -> dict:
    if not isinstance(proposal, dict):
            raise ValueError("提案必须是JSON对象")
    if not set(proposal)=={"hypothesis", "initial_lr"}:
        raise ValueError("提案必须且只能包含 hypothesis 和 initial_lr")
    if not isinstance(proposal["hypothesis"],str) or not proposal["hypothesis"].strip():
        raise ValueError("hypothesis必须为字符串且不为空")
    initial_lr = proposal["initial_lr"]
    if not (type(initial_lr)==int or type(initial_lr)==float):
        raise ValueError("initial_lr必须为int或float类型")
    if not math.isfinite(initial_lr):
        raise ValueError("initial_lr必须为有限数值")
    if initial_lr<=1e-7:
        raise ValueError("initial_lr必须大于1e-7")
    return proposal
def run_trial(
    experiment_config: dict,
    baseline_path: Path,
    proposal_path: Path,
) -> dict:
    proposal = validate_proposal(load_result(proposal_path))
    experiment_config = experiment_config.copy()
    experiment_config["initial_lr"] = proposal["initial_lr"]
    
    trial_dir = Path(experiment_config["out_dir"])
    candidate_path = trial_dir / "result.json"
    log_path = trial_dir.parent / f"{trial_dir.name}_train.log"
    config_path = trial_dir.parent / f"{trial_dir.name}_config.json"
    with config_path.open("x",encoding="utf-8") as handle:
        json.dump(experiment_config,handle,indent=2,allow_nan=False)
    baseline_result = load_result(baseline_path)
    returncode = run_training("b0.lr_experiment_learning",config_path,log_path)
    if returncode == 0 :
        tmp_result = load_result(candidate_path)
        try:
            decision = evaluate_candidate(tmp_result, baseline_result, 1300, 6500)
        except ValueError as error:
            trial_record = {
                "training_status": "completed",
                "comparison_status": "invalid",
                "reason": str(error),
                "candidate_result": str(candidate_path)
            }
        else:
            trial_record = {
                "training_status": "completed",
                "comparison_status": "valid",
                "decision": decision,
                "candidate_result": str(candidate_path),
            }
        
    else:
        trial_record = {
            "training_status": "failed",
            "comparison_status": "not_run",
            "returncode": returncode,
            "log_path": str(log_path),
}
    trial_record["config_path"] = str(config_path)
    trial_record["baseline_result"] = str(baseline_path)
    trial_record["log_path"] = str(log_path)
    trial_record["budget_status"] = "test_limits"
    trial_record["budget_limits"] = {
        "max_training_seconds": 1300,
        "max_peak_allocated_mib": 6500,
    }
    trial_record["proposal_path"] = str(proposal_path)
    trial_record["hypothesis"] = proposal["hypothesis"]
    print(trial_record)
    record_path = trial_dir.parent / f"{trial_dir.name}_record.json"
    with record_path.open("x",encoding="utf-8") as handle:
        json.dump(trial_record,handle ,allow_nan=False,indent=2)
    return trial_record
if __name__ == "__main__":
    proposal_path = Path(
        "/mnt/d/Project/IR-AutoResearch/results/"
        "sa0_agent_feedback_003_response.md"
    )
    baseline_path = Path(
        "/mnt/d/Project/IR-AutoResearch/results/"
        "lr_baseline_100_learning_retry1/result.json"
    )

    experiment_config = {
        "seed": 0,
        "steps": 100,
        "batch_size": 2,
        "val_batch_size": 1,
        "lr_horizon_steps": 200000,
        "lr_name": "candidate",
        "out_dir": (
            "/mnt/d/Project/IR-AutoResearch/results/"
            "sa0_agent_trial_003"
        ),
    }

    run_trial(experiment_config, baseline_path, proposal_path)