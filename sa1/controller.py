"""Independent SA1 entry point using the same training and decision harness."""
import sys

from sa0.controller import (
    ROOT, FIXED_RESULT_KEYS, METRIC_KEYS, read_json, write_json, sha256,
    source_hashes, dataset_hashes, freeze_sources, make_training_config,
    execute_training, compare_results, validate_proposal, build_candidate_source,
    write_agent_input,
)
from sa0.session import run_session
from sa1.agent import call_agent

PROMPT_PATH = ROOT / "sa1/PROMPT.md"
EXTRA_HARNESS_FILES = ("sa1/controller.py", "sa1/agent.py", "sa1/proposal.py", "sa1/PROMPT.md")


def build_request(protocol):
    # Shared scientific constraints, with only tool permission/output envelope changed.
    from sa0.controller import build_request as offline_request
    request = offline_request(protocol).replace(
        "不调用工具，不修改文件，不启动训练。",
        "允许使用内置web_search检索公开资料；不使用shell，不修改文件，不启动训练。",
    )
    return request + (
        "\n以上learning_rate/model格式是proposal字段的格式。最终JSON恰好包含proposal和research。"
        "research恰好包含status、summary、sources；status为searched、not_needed或unavailable。"
        "sources为数组，每项恰好包含url、title、used、reason；url为实际查阅的HTTP(S)链接，"
        "title和reason为非空字符串，used为布尔值。reason说明采用或不采用的依据。"
        "实际检索后status为searched，并提供来源；未检索须如实说明原因，不虚构来源。"
        "模型/指标/步数等固定约束不因检索而改变。"
    )


def trial_metadata(folder):
    metadata = {}
    for filename, key in (("research.json", "research"), ("search_audit.json", "search_audit")):
        path = folder / filename
        if path.is_file():
            metadata[key] = read_json(path)
    usage = metadata.get("search_audit", {}).get("cli_usage")
    if isinstance(usage, dict):
        values = [usage.get(key) for key in ("input_tokens", "output_tokens")]
        metadata["agent_usage"] = {
            "reported_tokens": (sum(values) if all(type(v) is int and v >= 0 for v in values) else None),
            "source": "cli_json_usage", "details": usage,
        }
    return metadata


def main():
    if "--legacy" not in sys.argv:
        from sa0.research.session import Workflow
        Workflow(sys.modules[__name__], read_json(ROOT / "sa1/research_config.json"),
                 retry_failed_full="--retry-full-once" in sys.argv).run()
        return
    protocol = read_json(ROOT / "sa1/config.json")
    if protocol.get("condition") != "SA1" or protocol["agent"].get("web_search") != "live":
        raise ValueError("SA1必须声明condition=SA1及agent.web_search=live")
    run_session(sys.modules[__name__], protocol)


if __name__ == "__main__":
    main()
