# 假设驱动研究流程 v2

当前默认128/32预跑配置及screening→full-selection调度以[PILOT_RUN_20261007.md](../PILOT_RUN_20261007.md)
为准：low100/full500、总24次、筛选13次、full储备4次。下文保留最初v2设计说明。

当前目标是本地workflow_validation，不是正式研究结果。B0、B1及旧实验文件保持不变。
SA0和SA1共享相同调度、候选权限、训练与评价，区别仅在检索权限和检索轨迹。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m sa0.controller
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m sa1.controller
```

分别读取sa0/research_config.json和sa1/research_config.json，必须串行运行以避免本地OOM。
v2训练提交另有仓库级GPU锁，防止两个v2会话同时启动GPU worker；旧版/B1不使用此锁，仍须手动串行。
新目录为sa0_research_v2_001/sa1_research_v2_001。旧的config.json、PROMPT.md及实现仍保留，
`python -m sa0.controller --legacy` / `python -m sa1.controller --legacy`可访问旧入口。
旧会话只读回查，不直接升级或绕过hash续跑。B1继续使用其现有入口和配置。

## 研究协议

1. 冻结B0代码/数据，建立本地完整阶段seed0匹配基线，然后生成3个初始假设H1/H2/H3和主方向。
2. Agent自主管理方向，不强制覆盖；不研究的方向在finalize和报告中说明原因。
3. 每个方向无候选/实验次数上限。最多新增2个不同机制的新假设，引用已分析的证据及新颖性理由。
4. 显式声明多父候选。继承源码/配置，不继承权重；新方向首个候选从B0分叉。
5. 组合需各父机制先有独立证据；配置冲突明确覆盖，源码冲突用resolved_sources提供完整解决版本。
6. 每次参数更新都从头初始化；训练seed和阶段固定。候选不得自行更改manifest、步数、数据或评价器。
7. 三次有效科学实验没有指标或初步机制证据时必须reconsider，但不自动关闭。
   Agent可继续或关闭；关闭后重开须引用后续新证据。工程失败不作假设反证。
8. 不使用micro。low是科学流程筛选；smoke仅工程检查，不支持机制/指标结论。

| 阶段 | 本地步数 | seed | 用途 |
| --- | ---: | --- | --- |
| smoke | 2 | 0或1 | 导入/shape/有限loss/梯度检查 |
| low | 20 | 0、1 | 初步比较、配对复验和关键消融 |
| full | 100 | 0 | 模拟完整训练；不称正式full-fidelity |
| confirmation | 100 | 0、1、2 | finalist设计锁定后的配对确认 |

低成本/full比较必须有匹配阶段、seed、数据、环境、步数的B0。
full晋级：同候选low两个seed三指标改善；或low改善加事先预测、匹配对照的关键消融。
消融mechanism_test指定reference_candidate_id、metric、direction、min_delta。
只有存在匹配参考实验且方向/差值达到预期，才允许mechanistic_progress；这是初步证据，不能排除所有混杂因素。
interpretation同时保存事实、推断、局限、证据ID和下一步。多个进展标签可同时出现。

## 可修改边界

允许train/mavi.py、train/feature_transform.py；train/experiment.py只允许loss/optimizer/scheduler函数。
模型接口和依赖保持兼容。特征变换在训练、验证及推理中统一应用，不访问标签且保持输入shape。
受控config支持initial_lr、loss、optimizer、scheduler；原始B0为100×L1、AdamW及官方before-update cosine。
配置覆盖优先于候选源码函数；需用null移除继承的loss/optimizer/scheduler配置才能启用对应源码改动。
禁止改prepare、program、清单、控制器、结果和账本。这里只是逻辑/工作区隔离，不是OS安全沙箱。

## 次数与成本

最多20次任务尝试，探索最多10次，其余留给确认；没有会话累计时间/token/agent调用次数上限。
GPU smoke、B0匹配基线、候选、换seed、失败、中断均消耗1次；命中匹配结果缓存不额外消耗。
没有候选改善时保留B0，不要求用满额度。单worker超时3600秒是进程保护；CLI超时900秒也是保护。
连续3个无效动作、重复相同动作或同类工程故障会暂停，不作为假设反证。修复后人工重新运行。
无GPU任务的正常规划可继续，但不能通过无限重复动作或免费重试绕过故障记录。

CLI调用前持久化预留记录。计数包含规划、提案、解释、失败及最终报告。
报告包含agent_calls、agent_active_seconds（客户端调用实际耗时）、各类token、未知usage/时间次数、
training_seconds、worker_process_seconds、wall_seconds。GPU allocation/queue在本地无调度器时为null，
不把worker墙钟冒充真实GPU分配成本。未测区间标未知，不计为已知零耗时。

新worker与旧worker的接口不同，因此v2建立独立匹配基线缓存，不直接复用旧版基线。
同一会话中完整阶段seed0会复用于confirmation seed0，前提为全部训练条件相同。
同一目录重启不重复完成任务；中断任务保留已用次数，是否重试需显式再次提交。
缓存产物、数据、候选源码、执行代码都核验哈希；更改协议请新开session_dir。

## 选择、确认和冻结

完整阶段三指标同时改善的候选中，按MAE最低、NRMS最低、SSIM最高排序选择finalist。
无合格候选则B0。三个配对seed的均值满足同一三指标规则才通过确认，报告逐seed差值、胜负和标准差。
2赢1输可通过均值门槛，但不能称逐seed一致胜出；确认未通过则冻结B0，不边确认边调整设计。
这不替代3次独立AutoResearch会话，也不证明统计显著性。

最终目录保存checkpoint、源代码、协议/环境、清单hash、完整谱系、确认结果和推理命令。

```bash
python -m sa0.research.inference --frozen SESSION/final --feature FEATURE.npy --output PREDICTION.npy
```

hidden evaluator当前未配置，summary明确标not_configured。不读取隐藏标签，不向agent反馈隐藏指标。
正式hidden评测需独立服务/权限/挂载隔离，不应在此agent进程内开放标签。当前程序拒绝formal模式，
正式B0成本、分设计数据、资源预算、热点/跨设计评价和独立隐藏入口尚需服务器阶段实现。
最终formal_keep始终为null，不能把100步模拟完整结果解释为正式科研保留。

## 文件职责与检查

- research/protocol.py：配置、假设字段与指标规则。
- research/lineage.py：多父谱系、冲突、候选权限和独立源代码。
- research/session.py：动作状态机、任务预约/恢复、晋级、确认、冻结和报告。
- research/worker.py：共享训练器，真实执行候选loss、optimizer、scheduler、特征变换。
- research/agent.py：通用JSON动作、离线/联网权限、调用与usage归档。
- research/inference.py：核验冻结模型后预测，不接受标签。
- research/PROMPT.md：共同研究协议，不是写死的候选序列。

```bash
python -m unittest sa0.test_session sa1.test_sa1 b1.test_b1 sa0.research.test_workflow sa0.research.test_agent sa0.research.test_worker b0.test_resume -q
```

状态机使用模拟agent/任务；worker测试仅使用tiny CPU模型，不启动真实GPU训练或模型调用。
