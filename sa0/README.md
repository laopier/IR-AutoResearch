# SA0 mini rehearsal

程序化模型接入见 [DESIGN_SOURCES.md](DESIGN_SOURCES.md) 和 `agent_loop.py`。
下方命令及 TASK.md 描述的是先前由当前对话逐步驱动的控制器演练；模型接入版会自动生成候选并接收结果。

任务与限制见 [TASK.md](TASK.md)。所有代码和结果在新建的 session 中，不能复用已有输出目录。
本例在 WSL 的 circuitnet Python 3.11/CUDA 环境运行。以下命令从项目根目录执行；`python` 需替换为该环境解释器。

```bash
python -m sa0.control init --session /mnt/d/Project/sa0_runs/20261001_mini_sa0_01 --data-root /mnt/d/WSL/Project/CircuitNet-main/training_set_full_mini/IR_drop --steps 5
python -m sa0.control run --session /mnt/d/Project/sa0_runs/20261001_mini_sa0_01 --name baseline --role baseline
python -m sa0.control propose --session /mnt/d/Project/sa0_runs/20261001_mini_sa0_01 --hypothesis sa0/hypothesis_smooth_l1.json
# 仅修改 session/workspace/train/experiment.py 中登记过的损失函数。
python -m sa0.control commit --session /mnt/d/Project/sa0_runs/20261001_mini_sa0_01
python -m sa0.control run --session /mnt/d/Project/sa0_runs/20261001_mini_sa0_01 --name smooth_l1 --role candidate
python -m sa0.control decide --session /mnt/d/Project/sa0_runs/20261001_mini_sa0_01
```

baseline 必须先完成，假设必须先登记，候选必须提交干净的 Git commit 才能运行。
预算耗尽或出现无账本的中断试验时，控制器拒绝新试验；并不自动重试。并发运行同一 session 不受支持。
这是单候选演练，控制器不负责启动 LLM、生成候选或多轮自主搜索。

`protocol.json` 是固定预算；`session.json` 记录来源和哈希；`events.jsonl` 是操作时间线；
`trials/*/ledger.json` 包含成功/失败试验消耗；`decision.json` 是保留或回退记录。
回退会恢复两个可修改文件并新增 Git commit，候选历史和训练结果仍保留。
这里的 single_gpu_wall_hours 是占用训练/评估流程的墙钟小时估算，不是 GPU 内核活跃时间。

这轮两个新试验的预算相同，不能拿 5 步候选与此前 batch size 2、100 步的 B0 结果直接比较。
候选改变了损失函数，因此两个试验的训练 loss 数值也不能直接比较；最终使用统一的验证指标。
验证样本参与选型，不能称为测试集。三张训练图和一张验证图无法证明泛化能力。
