# 假设驱动研究流程 V4

V4把AuDoPEDA式“理解—规划—定位—实现—快速筛选—完整验证”思想移植到当前IR-drop pilot，同时保留本项目原有的数据冻结、配对B0、严格full和三seed confirmation边界。它仍是`workflow_validation`，不是正式科研结论。

## 相比V3的核心变化

1. **确定性研究卡片**：会话初始化生成`research_cards.json`与`RESEARCH_CARDS.md`，明确问题、模型接口、训练条件、指标和可编辑边界。
2. **规划与实现分离**：先提出3～5个假设，再登记4个定位干预项P1～P4；候选必须引用一个P编号。实验后细化也要先`localize`，再写候选补丁。
3. **横向组合竞赛**：先完成至少4个不同干预计划的low seed0，先按三指标Pareto前沿、再按归一化三指标score破平排序；只有宽松安全门内的前2名可获得seed1预算。
4. **双重对照**：所有low候选比较匹配B0；派生候选还必须比较同阶段同seed主父候选，分别记录全局和局部delta。
5. **过程诊断而非挑权重**：在训练25%/50%/75%/100%处记录三指标、近期loss与梯度范数。探针只帮助解释和设计下一实验，最终仍使用训练结束checkpoint。
6. **停滞停止**：连续4个新seed0候选没有超过当前incumbent时停止普通筛选，避免一直生成低价值实验。

## 预算

- B0 full：复用共享1000步结果。
- screening：最多13个任务尝试，包含匹配B0、smoke、low、失败和中断。
- full-selection：预留4个任务尝试，最多选择2个可晋级候选。
- confirmation：保留seed0/1/2配对确认所需总任务额度；可复用同签名已完成训练。
- low：200步；full/confirmation：1000步；每次训练均从头初始化。

score只用于路由预算，不是论文指标，也不等于晋级。晋级仍要求V3的双seed聚合门，或一次严格low胜利加预登记匹配关键消融。full与confirmation仍要求MAE降低、NRMS不升、SSIM不降。

## 新会话目录

- `results/sa0_pilot128_v4_200_1000_001`
- `results/sa1_pilot128_v4_200_1000_001`

V3目录不会被覆盖，V4也不会续跑V3状态。

## 启动

确认共享B0完整、Codex CLI可登录后，在独立tmux窗口运行：

```bash
cd /mnt/d/Project/IR-AutoResearch
source /home/laopier/miniconda3/etc/profile.d/conda.sh
conda activate circuitnet
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
set -o pipefail
python -u -m sa0.controller 2>&1 | tee "results/launch_logs/sa0_v4_$(date +%Y%m%d_%H%M%S).log"
```

SA1把模块名改为`sa1.controller`。建议先完整跑SA0并检查研究卡片、干预组合、排行榜及探针，再启动SA1，避免同时争用单GPU锁。

## 重点归档

- `research_cards.json`：冻结问题理解卡片。
- `research_plan.json`：3～5个假设。
- `intervention_portfolio.json`：初始4个定位干预项。
- `state.json`中的`screening_leaderboard`等价信息可由任务重建；最终汇总写入`summary.json`。
- 每个任务的`artifacts/validation_probes.jsonl`：固定步数诊断。
- `record.json`：相对B0的`metric_delta`、相对父候选的`local_metric_delta`与筛选score。
- `confirmation.json`：最终三seed配对确认。

## checkpoint原则

V4没有引入`best.pt`。当前验证集同时参与研究选择，没有独立selection split；只给候选挑最佳checkpoint会形成不对称选择偏差。等正式服务器版具备独立selection split后，可以预登记“B0和所有候选完全相同的best规则”，再把best checkpoint纳入公平比较。
