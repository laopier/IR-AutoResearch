import subprocess
from pathlib import Path
import json
from sa0.controller_learning import validate_proposal

codex_path = Path(
    "/mnt/c/Users/皮/AppData/Local/OpenAI/Codex/bin/8aaf1547b825b104/codex.exe"
)
def call_agent(prompt: str,log_path: Path) -> str:
    command = [
    str(codex_path),
    "exec",
    "--sandbox", "read-only",
    "-c", 'web_search="disabled"',
    "--ignore-user-config",
    "--ephemeral",
    "-c", "features.shell_tool=false",
    "-c", "features.unified_exec=false",
    "-c", "features.multi_agent=false",
    "-",
]
    completed = subprocess.run(
    command,
    input=prompt,
    capture_output=True,
    text=True,
    encoding="utf-8",
    timeout=180,
)
    print(completed.stderr)
    with log_path.open("x", encoding="utf-8") as handle:
        handle.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)
    else:
        return completed.stdout
def request_proposal(
    prompt_path:Path,
    log_path:Path,
    response_path:Path
)->dict:
    if(log_path.exists() or response_path.exists()):
        raise ValueError("调用记录已存在，使用新的调用编号")
    prompt = prompt_path.read_text(encoding="utf-8")
    response = call_agent(prompt,log_path)
    with response_path.open("x",encoding="utf-8") as handle:
        handle.write(response)
    proposal = json.loads(response)
    return validate_proposal(proposal)
if __name__ =="__main__":
    agent_input_path = Path("/mnt/d/Project/IR-AutoResearch/results/sa0_learning_proposal_002_input.md")
    log_path = agent_input_path.parent / "sa0_agent_feedback_004_cli.log"
    response_path = agent_input_path.parent / "sa0_agent_feedback_004_response.md"
    proposal = request_proposal(agent_input_path,log_path,response_path)
    print(proposal)
