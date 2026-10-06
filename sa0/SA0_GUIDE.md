# SA0 自主实验流程

从 WSL 项目根目录，在 circuitnet 环境启动：

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m sa0.controller
```

`config.json` 是唯一的启动配置，`PROMPT.md` 是研究规则。旧的 `*_learning.py`
保留供学习和回查，当前入口不再依赖它们。

## 当前配置

- 新会话 `results/sa0_autonomous_003`，seed=0，每次100步，训练batch=2、验证batch=1。
- 基线初始学习率2e-4，余弦周期200000步，最低学习率1e-7。
- agent每轮自主选择 `learning_rate` 或 `model`，不强制轮换；最多两次提案调用和两轮候选。
- 学习率候选保持基线模型；模型候选保持基线学习率。候选均从冻结源码重新初始化，
  不叠加此前候选，不复用旧checkpoint继续训练。
- 模型提案只修改 `train/mavi.py`。patch必须唯一匹配且通过Python语法检查。
  真实输入输出和梯度检查由统一训练接口执行；失败记入历史，不自动重试。
- 历史导入1e-4、3e-4、4e-4、6e-4以及GroupNorm实验，保存其所属基线，避免丢失失败证据。
  缺少相同源码/worker契约的历史标为 `context_only`，不直接当作本会话对照结果。
- Codex可执行文件、模型、推理强度、调用超时都在 `agent` 配置内。当前使用
  `gpt-6.1-sol` / `low`；本机CLI调用已明确拒绝none。CLI版本和实际模型从调用日志回查。

## 结果与预算

每轮保存输入、原始响应、提案、候选源码、逐步loss和样本编号、环境、验证指标、资源、
checkpoint和record。下一轮接收已有历史及约20个采样训练点；完整训练日志仍全部归档。

相对本会话固定基线，MAE严格降低、NRMS不升、SSIM不降才是指标改善。
`test_limits` 下改善候选的 `keep` 为null，位于summary的 `promising_candidates`；
预先确定正式预算且为 `formal_limits` 时才允许 `keep=true`，位于 `retained_candidates`。
两个列表均引用完整候选目录，不以MAE单独对存在指标权衡的候选宣布最优。

- `training_seconds` 与 `peak_allocated_mib` 只统计训练段，评价及保存不计入。
- `process_seconds` 包含worker初始化、训练、验证和保存。
- `process_timeout_seconds` 是单worker保护超时。
- `max_session_process_seconds` 限制所有新worker进程累计时间，包含基线和失败尝试；
  不包含agent等待、controller分析、历史实验成本，也不是GPU内核活跃时间。
- `max_agent_calls` 在调用前预留，超时/无效提案同样计数；没有隐式重试。
- token总数来自CLI报告。缺失报告明确标未知，不把未知记成零，不声称严格token上限。
- 连续失败达到 `stop_after_failures` 或资源/调用/轮数上限时停止。
- 当前1300秒/6500MiB仍是历史测试阈值；1800秒进程保护和5400秒会话保护也是本地工程设置，
  不作为论文训练预算的依据。资源判定使用PyTorch allocated峰值，不是整张GPU占用。

## 恢复与冻结

新会话可设置 `baseline_source_session` 为已有会话目录，复用其中成功完成的基线；
设为 `null` 则重新训练。当前配置从 `sa0_autonomous_001` 复用基线，保留001和002的失败记录。
复用前核验训练配置、实际数据与split哈希、冻结源码（含worker）、当前环境和完整产物。
环境检查只初始化库和CPU模型，不训练。任何不一致会停止，绝不自动回退为重新训练。
复制结果、checkpoint及训练轨迹到新会话，记录来源、文件哈希和原始配置；候选仍从头训练。
复用仅导入基线，不自动导入来源会话的候选历史；历史仍由history_sessions/history_results控制。

为了不因缓存而增加搜索额度，`process_seconds` 仍按原基线成本扣除预算。
`actual_current_process_seconds` 单独报告本会话实际训练进程时间加复用检查/复制耗时，
不再次把旧基线训练算作新计算开销。共享基线在跨会话总实际成本统计中只计一次，
按summary中的baseline_reuse_origin追溯。正式实验是否允许共享基线应提前确定；
需要独立基线训练时设为null。改变seed/步数/数据/源码/环境不能复用同一基线。

配置不变时再次运行同一命令，会读取已归档记录，保留完成的基线，从下一未执行轮继续。
已完成会话再次运行不会重新调用agent或训练。协议、数据、源码或会话执行代码变化会拒绝恢复，
请开新会话；旧版会话只作为历史输入，不直接升级后续跑。

中断的候选记为消耗过的一轮，不从中断checkpoint精确续训。异常结束且worker尚存活时
拒绝继续；未归档训练的成本按单次保护上限保守计入，避免零成本重试。基线未完成时不自动重训，
需检查日志并另开会话。会话锁防止同一会话的两个controller同时启动。

每个候选独立复制冻结基线，因此拒绝候选不用改回项目源文件。冻结内容和数据按hash核验。
这些是科研流程边界检查，不是执行生成代码的操作系统安全沙箱。

## 验证

```bash
python -m unittest sa0.test_session -v
```

测试使用模拟agent与训练子进程，不消耗真实模型调用或GPU时间。
新统一版本仍需一次真实运行验证。科学结论还需要预注册正式预算、独立运行波动和更有代表性数据，
不能以一个小数据会话宣称正式科研对照已完成。
