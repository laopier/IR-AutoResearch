# 假设驱动研究流程 V3

V3针对128/32 pilot中“每个候选都跑两个seed但没有候选晋级”的问题调整研究预算，同时保持B0、数据清单和最终评价口径不变。

## 核心变化

- Agent固定使用`gpt-5.6-sol`与`high`推理强度；Codex桌面版CLI从安装目录自动选择最新可执行文件，不再绑定会变化的哈希路径。
- low由100步增加到200步，full与confirmation保持1000步，学习率周期仍为200000步，因此既有三seed B0 full缓存仍可复用。
- 使用两级漏斗：所有候选先跑low seed0；只有相对匹配B0没有超过宽松回退上限的候选，才允许继续跑seed1。
- seed0宽松门仅用于预算路由，不代表科学改善。候选进入full的主路径要求双seed平均MAE严格改善、平均NRMS/SSIM不越过容差，同时每个seed都不得超过最大回退上限。
- full和三seed confirmation继续采用严格门：MAE严格降低、NRMS不升、SSIM不降。
- SA1的`research_plan`调用必须真实发生web search；后续解释或决策在本地证据充分时可以记录`not_needed`。
- 中断恢复时，旧`stop_reason`移入`pause_history`并清空，防止已恢复会话仍被误报为因旧错误结束。

当前pilot仍比较训练结束时checkpoint。没有独立selection split、且B0未按相同规则重跑前，不对候选单方面启用best checkpoint，以避免选择偏差。

## 运行顺序

SA0和SA1共用单GPU锁，必须串行运行。两者使用不同的新会话目录：

- `results/sa0_pilot128_v3_200_1000_001`
- `results/sa1_pilot128_v3_200_1000_001`

旧V2目录不会被覆盖，也不能用V3协议续跑。
