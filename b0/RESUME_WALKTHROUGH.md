# 从 2 步恢复到第 4 步

先读 `checkpoint.py` 的 `capture_rng()`，再看 `run.py` 的 `save_training_state()`，最后读恢复分支。这三个位置对应“保存什么、什么时候保存、如何继续”。

## 保存状态不是保存模型对象

`model.state_dict()` 保存参数和 buffer，例如 BatchNorm 的 running_mean；不包含模型类的实现。恢复时先按同一代码建立模型，再用 `load_state_dict()` 填回状态。

`optimizer.state_dict()` 保存 AdamW 的每个参数对应的 step、exp_avg 和 exp_avg_sq，以及参数组配置。只恢复权重、重新创建空 AdamW，会丢失之前的更新统计。

checkpoint 的 `step=2` 表示已经完成两次 `optimizer.step()`。恢复后的 Python 循环从 `range(2, 4)` 开始：零起始索引 2、3，对应日志的第 3、4 步。学习率仍调用 `official_lr(step, 200000)`，不会重置为第 0 步。

## 随机数为什么也要保存

固定 seed 只定义随机序列的起点，不能表达“已经使用到序列的哪个位置”。保存 RNG state 才能表达当前进度。Python、NumPy、PyTorch CPU 和 CUDA 各有自己的状态。

恢复流程会先构建模型、验证和重建数据迭代器。这些准备动作可能消耗随机数，所以最后才调用 `restore_rng()`，随后执行下一次训练更新。

## 样本顺序为什么单独保存

训练 DataLoader 使用自己的 `torch.Generator`，它与全局 Torch RNG 分开。我们保存：

- 本轮 `iter(loader)` 之前的 generator 状态。
- 本轮已经读取的完整 batch 数。
- 保存时 generator 当前状态。

恢复时回到本轮起点，重建同一打乱顺序，跳过已完成的 batch，并检查 generator 状态一致。跳过只读数据，不执行反向传播或 optimizer.step。若断点正好在一轮末尾，下一次读取会进入 StopIteration 分支，按同样状态开始下一轮。

这个实现优先易读与正确，限制 worker=0；恢复游标需要重复读取已完成部分，数据很大时会有开销。以后可以把预生成的 permutation 和游标保存为专门 sampler，优化恢复速度，再重新做等价性测试。

## 怎样证明恢复正确

不只看最终 MAE。测试会比较连续 4 步与 2+2 步的：每步样本 ID、loss、学习率，最终模型参数和 buffer，优化器内部状态，以及 RNG 和采样状态。实际 MAVI 验证用三个独立进程运行，确保不是沿用了旧进程的内存。

本地验证通过并不等于所有 GPU 环境都逐位一致。Kaggle 换卡或更新环境后，应在新环境再做一次短程等价性验证。

## 旧权重怎么办

2026-10-02 首次 Kaggle 100 步结果只有模型权重。无法从权重反推 AdamW 的历史统计和 RNG，因此程序会拒绝将其作为完整续训 checkpoint。它仍能用于预测和评价。

每个恢复分段输出到新目录，并记录父 checkpoint 路径与 SHA256。保留完整链条才能知道结果来自哪次运行；所有分段和失败运行的资源开销都要计入实验成本。
