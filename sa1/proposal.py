"""Validate research metadata separately from the unchanged candidate schema."""
from urllib.parse import urlsplit
from sa0.proposal import validate_proposal


def validate_response(value):
    if not isinstance(value, dict) or set(value) != {"proposal", "research"}:
        raise ValueError("SA1回复必须且只能包含proposal和research")
    proposal = validate_proposal(value["proposal"])
    research = value["research"]
    if not isinstance(research, dict) or set(research) != {"status", "summary", "sources"}:
        raise ValueError("research必须包含status、summary、sources")
    if research["status"] not in {"searched", "not_needed", "unavailable"}:
        raise ValueError("未知的检索状态")
    if not isinstance(research["summary"], str) or not research["summary"].strip():
        raise ValueError("检索总结不能为空")
    if not isinstance(research["sources"], list):
        raise ValueError("sources必须为数组")
    for source in research["sources"]:
        if not isinstance(source, dict) or set(source) != {"url", "title", "used", "reason"}:
            raise ValueError("来源必须包含url、title、used、reason")
        for key in ("url", "title", "reason"):
            if not isinstance(source[key], str) or not source[key].strip():
                raise ValueError(f"来源{key}不能为空")
        url = urlsplit(source["url"])
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("来源必须是HTTP(S)链接")
        if type(source["used"]) is not bool:
            raise ValueError("来源used必须为布尔值")
    if research["status"] == "searched" and not research["sources"]:
        raise ValueError("已检索必须记录来源")
    if research["status"] != "searched" and research["sources"]:
        raise ValueError("未检索不能声称有本轮检索来源")
    return proposal, research
