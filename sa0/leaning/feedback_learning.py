from pathlib import Path
import json
from sa0.controller_learning import load_result, build_feedback
def write_agent_input(
    instructions:str,
    feedback:dict,
    request:str,
    output_path:Path
)->Path:
    feedback_text = json.dumps(feedback,ensure_ascii=False,indent=2,allow_nan=False)
    agent_input = (
        instructions
        + "\n\n## 本轮实验反馈\n\n"
        + feedback_text
        + "\n\n## 本轮请求\n\n"
        + request
    ) 
    agent_input_path = output_path
    with agent_input_path.open("x",encoding="utf-8") as handle:
        handle.write(agent_input)
    return output_path
if __name__ == "__main__":
    baseline = load_result(Path("/mnt/d/Project/IR-AutoResearch/results/lr_baseline_100_learning_retry1/result.json",))
    candidate = load_result(Path("/mnt/d/Project/IR-AutoResearch/results/sa0_agent_trial_002/result.json",))
    previous_candidate = load_result(Path(
    "/mnt/d/Project/IR-AutoResearch/results/lr_candidate_100_learning/result.json"
))
    trial = load_result(Path("/mnt/d/Project/IR-AutoResearch/results/sa0_agent_trial_002_record.json",))
    prompt_path = Path("/mnt/d/Project/IR-AutoResearch/sa0/PROMPT_LEARNING_V1.md")
    instructions = prompt_path.read_text(encoding="utf-8")
    feedback = build_feedback(baseline,candidate,trial)
    feedback["history"] = [
    {
        "candidate_result": previous_candidate,
        "conclusion": "相同100步预算下，MAE和NRMS升高、SSIM降低，指标不满足保留条件。",
        
    }
]
    request = (
        "本轮3e-4候选与固定基线均训练100步，三个验证指标均改善。"
        "1300秒和6500MiB属于临时测试阈值，正式资源判定仍待确定。"
        "请结合本轮结果及history，提出一个新的初始学习率候选。"
        "不重复已评估的1e-4、3e-4，也不将基线2e-4作为新候选。"
        "只返回一个JSON对象，恰好包含hypothesis和initial_lr两个字段。"
        "hypothesis为非空中文字符串，说明依据、预期和风险；"
        "initial_lr为有限数值且大于1e-7。"
        "不要使用Markdown代码块，不添加JSON之外的文字。"
        "不调用工具，不修改代码，不启动训练。"
    )
    
    output_path = Path(
        "/mnt/d/Project/IR-AutoResearch/results/"
        "sa0_learning_proposal_003_input.md"
    )
    write_agent_input(
        instructions,
        feedback,
        request,
        output_path,
    )