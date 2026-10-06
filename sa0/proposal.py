import ast
import math


def validate_proposal(proposal: dict) -> dict:
    if not isinstance(proposal, dict):
        raise ValueError("提案必须是JSON对象")

    if set(proposal) != {"hypothesis", "kind", "change"}:
        raise ValueError(
            "提案必须且只能包含 hypothesis、kind、change"
        )

    hypothesis = proposal["hypothesis"]
    if not isinstance(hypothesis, str) or not hypothesis.strip():
        raise ValueError("hypothesis必须是非空字符串")

    change = proposal["change"]
    if not isinstance(change, dict):
        raise ValueError("change必须是JSON对象")

    kind = proposal["kind"]

    if kind == "learning_rate":
        if set(change) != {"initial_lr"}:
            raise ValueError("学习率候选只能修改initial_lr")

        initial_lr = change["initial_lr"]
        if type(initial_lr) not in (int, float):
            raise ValueError("initial_lr必须是数值")

        if not math.isfinite(initial_lr) or initial_lr <= 1e-7:
            raise ValueError("initial_lr必须是大于1e-7的有限数值")

    elif kind == "model":
        if set(change) != {"edits"}:
            raise ValueError("模型候选必须提供edits")

        edits = change["edits"]
        if not isinstance(edits, list) or not edits:
            raise ValueError("edits必须是非空列表")

        for edit in edits:
            if not isinstance(edit, dict):
                raise ValueError("每项修改必须是JSON对象")

            if set(edit) != {"path", "old", "new"}:
                raise ValueError("每项修改必须包含path、old、new")

            if edit["path"] != "train/mavi.py":
                raise ValueError("当前只允许修改train/mavi.py")

            if not isinstance(edit["old"], str) or not edit["old"]:
                raise ValueError("old必须是非空字符串")

            if not isinstance(edit["new"], str):
                raise ValueError("new必须是字符串")

            if edit["old"] == edit["new"]:
                raise ValueError("替换前后的代码不能相同")

    else:
        raise ValueError("未知的候选类型")

    return proposal


def build_candidate_source(
    baseline_source: str,
    proposal: dict,
) -> str:
    """从冻结基线生成候选源码，不修改磁盘文件。"""
    proposal = validate_proposal(proposal)

    if proposal["kind"] != "model":
        raise ValueError("只有模型提案需要生成候选源码")

    candidate_source = baseline_source

    for edit in proposal["change"]["edits"]:
        old = edit["old"]

        if candidate_source.count(old) != 1:
            raise ValueError(
                "old必须在当前候选源码中恰好匹配一次"
            )

        candidate_source = candidate_source.replace(
            old,
            edit["new"],
            1,
        )

    # 检查Python语法；模型运行检查由训练子进程负责。
    ast.parse(candidate_source)

    return candidate_source