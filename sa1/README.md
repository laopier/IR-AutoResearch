# SA1 联网自主研究

默认入口已升级到共享假设驱动V4，读取sa1/research_config.json，详见[../sa0/RESEARCH_V4.md](../sa0/RESEARCH_V4.md)。旧V3/V2说明仅用于历史结果解释。
下文是旧版配置/结果说明，使用`--legacy`访问旧入口。
当前V4的128/32预跑使用low200/full1000、四方向seed0横向竞赛和强制full-selection；旧预跑配置保留在[历史说明](../PILOT_RUN_20261007.md)。

入口：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m sa1.controller
```

配置在sa1/config.json，固定指令在sa1/PROMPT.md。默认新会话sa1_autonomous_001，
两轮候选、100步、seed=0、batch=2，复用sa0_autonomous_001的已核验基线。
基线预算扣除、候选初始化、数据、训练和三指标判定均复用SA0实现。
SA1不复制第二套训练器。sa0/session.py只新增可选的prompt/执行代码/记录扩展接口；
SA0继续使用原离线调用。旧归档保留，执行代码更新后不沿用旧会话继续研究。

agent只开放内置web_search=live，shell/文件修改/多agent仍关闭。
首个research_plan调用必须实际检索；后续轮次在本地证据足够时可记录not_needed。联网权限本身不能替代对真实搜索事件的审计。
最终回复包含proposal与research；控制器保存检索信息，再把proposal交给相同训练接口。
每轮额外保存events.jsonl（CLI原始事件）、agent_envelope.txt（原始最终回复）、
research.json（来源及采用理由）、search_audit.json（实际工具事件和CLI usage）。
这些信息进入record和下一轮反馈。声称检索却没有搜索事件会拒绝训练。
URL格式与事件存在会核验，不能据此保证来源内容或采用理由正确；需研究者审阅原始轨迹。
不声称严格搜索次数/token/费用上限；沿用调用次数、超时和计算预算。
CLI token usage原样保存在search_audit；存在合法input_tokens/output_tokens时把两者之和
汇总到record和summary，缓存token字段保留供回查。缺失usage时标未知，不把未知当零，
不把token数当费用，也不把工具等待时间当累计GPU计算。

SA1默认与当前SA0保持同一初始历史，不单独导入SA0新发现。正式对照前应冻结双方协议、
模型/推理设置、信息起点和预算。允许联网是主要研究因素，不保证产生改善。
本地内存不足的问题不会因联网而解决，仍需在稳定资源条件下完成训练。

模拟检查，不调用真实模型或GPU：

```bash
python -m unittest sa0.test_session sa1.test_sa1 -v
```

联网配置和JSONL事件依据官方文档：
https://developers.openai.com/codex/config-reference
https://developers.openai.com/codex/noninteractive
