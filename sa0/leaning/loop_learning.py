from pathlib import Path
from sa0.feedback_learning import write_agent_input
from sa0.agent_learning import call_agent,request_proposal
from sa0.controller_learning import run_trial,load_result

if __name__ == "__main__":
    root = Path("/mnt/d/Project/IR-AutoResearch")

    baseline_path = (
        root / "results/lr_baseline_100_learning_retry1/result.json"
    )
    baseline = load_result(baseline_path)

    instructions = (
        root / "sa0/PROMPT_LEARNING_V1.md"
    ).read_text(encoding="utf-8")

    fixed_config = {
        "seed": 0,
        "steps": 100,
        "batch_size": 2,
        "val_batch_size": 1,
        "lr_horizon_steps": 200000,
        "lr_name": "candidate",
    }
    max_trials = 2
    history = [
        {
            "initial_lr": 1e-4,
            "candidate_result": load_result(
                root / "results/lr_candidate_100_learning/result.json"
            ),
            "conclusion": "100步下，三个验证指标均劣于固定基线。",
        },
        {
            "initial_lr": 3e-4,
            "candidate_result": load_result(
                root / "results/sa0_agent_trial_002/result.json"
            ),
            "trial_record": load_result(
                root / "results/sa0_agent_trial_002_record.json"
            ),
            "conclusion": (
                "100步下，三个验证指标均优于固定基线。"
                "旧记录中的资源淘汰使用临时测试阈值，"
                "正式保留决定仍待预算确定。"
            ),
        },
    ]