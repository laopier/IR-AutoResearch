# SA0 设计依据与第一版模型接入

直接协议来源：学长的 IR_predictor_AutoResearch_研究方案_20260928.docx。
SA0 是单 agent、离线、统一起点；正式比较需要按设计隔离、独立研究会话与完整成本账本。
这里“离线”指禁止外部检索，不是禁止远程模型推理。

相关方法来源：[AuDoPEDA, §3](https://arxiv.org/html/2601.06268v2)。

| 论文阶段 | 本项目第一版对应 | 尚未实现 |
| --- | --- | --- |
| S0 代码图与文档 | 人工整理的接口卡片和冻结源码作为模型输入 | tree-sitter 图、Docmaker、检索索引 |
| S1 文献支撑的规划 | 同一个模型从给定源码和基线反馈生成假设 | DSPy、文献语料和检索；本轮不提供外部文献 |
| S2 定位和详细计划 | 结构化 JSON plan 和精确 old/new edits | 图邻域定位、跨文件复杂计划 |
| S3 执行与反馈 | 控制器应用修改、训练评估、筛选、将结果交回模型 | 多候选多轮搜索、自动修复、正式预算和隐藏测试 |

阶段借鉴不等于复现 AuDoPEDA，也不是自研 harness 增益的证据。
本轮模型不能使用 shell、检索、MCP 或文件修改；控制器应用验证后的片段替换。
这与完整 coding agent 自主使用多种工具的实验条件不同，正式比较必须明确区别。

## 两个调用、一条研究历史

`agent_loop.py` 在 Windows 调用已登录的 Codex CLI，GPU 训练在 WSL。
第一个调用接收当前训练源码、接口说明、冻结预算、基线指标，返回假设和修改。
控制器提前登记假设、检查路径/AST/精确匹配、提交候选并训练。
第二个调用携带第一轮完整 proposal、基线和候选结果、决定，返回分析和下一假设。
两个独立 CLI 进程属于同一条显式传递历史的研究轨迹；并未并行启动多个 agent。
这版只有一个候选，下一假设仅记录，不能绕过 max_trials=2。

模型调用依赖已有 ChatGPT CLI 登录，不读取或复制认证文件，不修改全局登录设置。
官方依据：[非交互调用](https://developers.openai.com/codex/noninteractive)、
[登录方式](https://developers.openai.com/codex/auth)、[配置项](https://developers.openai.com/codex/config-reference)。
运行时忽略用户配置，禁用检索、shell、协作、hooks 和插件；使用 CLI 默认模型，不指定模型覆盖。
每次调用最多 180 秒，最多两次，没有自动重试。保存 prompts、schemas、JSONL 事件和输出。
若 CLI 返回 usage 则记录 token；金额未知。时间限制不是硬 token 上限，这一点尚未补齐。
AST 检查和 Python audit 仍不是防恶意代码的完整沙箱。

运行：

```powershell
python -m sa0.agent_loop --session D:\Project\sa0_runs\20261001_agent_loop_01 --data-root D:\WSL\Project\CircuitNet-main\training_set_full_mini\IR_drop
```

若调用、格式、修改或训练失败，记录失败并停止，不自动覆盖结果或重新开始。
