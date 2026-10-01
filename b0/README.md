# B0 小规模验证与服务器训练交接

本目录提供人工监督的 B0 固定配方训练入口。目标是把现有 MAVI 的数据读取、训练、验证、权重保存和重新加载验证走通，再在服务器上建立开发集 baseline。它独立于 `program/policy.json` 的 agent 搜索规则，不启动 SA0，不读取隐藏测试。

## 已核对的配方

| 项目 | 设置 |
| --- | --- |
| 模型 | MAVI，in_channels=1，out_channels=4，bilinear=False |
| 初始化 | 随机初始化；不使用官方预训练权重 |
| 优化器 | AdamW，lr=2e-4，betas=(0.9,0.999)，weight_decay=1e-2 |
| 损失 | 100 × 平均 L1 |
| 学习率 | 官方单周期余弦，每次更新前按零起始步数计算；最低 1e-7 |
| 默认训练预算 | 200000 次参数更新，batch_size=2 |
| 数据增强 | 无 |
| 数值与取样 | float32；训练 shuffle=True、drop_last=True；验证 batch_size=1 |
| 随机种子 | 0，记录 Python、NumPy、PyTorch 环境 |

`--steps` 是本次执行的更新次数，`--lr-horizon-steps` 是学习率计划长度。小规模执行 100 步时仍保留 200000 步计划，避免把学习率在第 100 步就压到最低。训练样本数不能代替更新步数。

原始官方参考来自 CircuitNet Git 仓库提交 `41ade1d7e913a418b4495b6b13630b26b0ff7279`。`b0/reference/` 保存用于核对的模型、训练器、配方和 loss 原文；MMCV 初始化参考来自 v1.6.0。原始代码中的可选 MMCV 导入在核对脚本中通过提取其上游初始化函数适配，模型主体不改。

参考链接：[CircuitNet](https://github.com/circuitnet/CircuitNet/tree/41ade1d7e913a418b4495b6b13630b26b0ff7279/routability_ir_drop_prediction)、[MMCV 初始化函数](https://github.com/open-mmlab/mmcv/blob/v1.6.0/mmcv/cnn/utils/weight_init.py)。许可证见根目录 CIRCUITNET_LICENSE 与 `b0/reference/MMCV_LICENSE`。

固定 seed、worker 数和关闭 TF32 是本项目的复现控制；worker 数为 0，官方原训练器为 16。当前使用自建开发集划分，不能称为官方划分上的完整复现成绩，也不保证不同 PyTorch、硬件的结果逐位一致。

## 服务器环境

本机验证环境为 WSL Linux、Python 3.11.15、PyTorch 2.11.0+cu128、NumPy 2.4.6、OpenCV 5.0.0.93、RTX 3070 Laptop 8GB。应优先在服务器使用相同软件版本；CUDA-enabled PyTorch 的安装需与服务器驱动匹配，由服务器环境负责人确认。当前代码需要 Python >=3.10，依赖清单中的 NumPy 建议 Python 3.11。

先安装匹配服务器驱动的 PyTorch，再执行：

交接 zip 不包含 `.git` 历史。先解压到团队现有 Git 仓库，或在解压目录初始化仓库、按团队真实身份提交全部交接代码，确认 `git status --short` 无代码改动。原有 runner 的测试需要可用 Git commit；正式实验也需要冻结代码版本。不要将伪造作者信息写入配置。

```bash
git init
# 如果尚未配置身份，先按团队实际身份配置本仓库的 user.name 和 user.email。
git add .
git commit -m "Prepare fixed-recipe B0 server baseline"
```

```bash
python -m pip install -r b0/requirements-validated.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m unittest discover -s tests -v
python -m unittest b0.test_preflight -v
python -m b0.verify_official --output /tmp/b0_parity_server.json
```

输出文件应使用新路径，核对脚本和训练入口都不覆盖已有实验。完整复现需要冻结实际环境；如果服务器版本不同，保留新的 environment.json 并重新通过测试。

## 数据准备与预检查

完整 IR-drop 数据应具有下列布局；压缩包不包含数据，需要服务器自行准备允许使用的数据。

```text
DATA_ROOT/
  feature/<manifest中的文件名>.npy
  label/<manifest中的文件名>.npy
```

现有 train_N28_seed0.csv 为 6370 个样本，validation_N28_seed0.csv 为 708 个样本，均属于 RISCY。这是同设计族开发划分；不是完整设计及衍生版本隔离的 zero-shot 协议。后续跨设计实验需要另行冻结 train、validation、隐藏 test 清单，不能由本入口的结果证明跨设计泛化。

先检查数据文件和数组头信息，不启动训练：

```bash
bash b0/launch_server.sh /path/to/IR_drop /path/to/results/b0_seed0_run01 --preflight-only
```

入口会检查每个文件是否存在、路径边界、数值类型、形状、重复路径和 train/validation 文件重叠。默认不在预检查中扫描全体数组值，实际取样时 Dataset 会检查 NaN/Inf；需要全量数值扫描时附加 `--check-values`，它会读取全部训练和验证数组。

标签单位、24 项特征含义、有效区域与归一化约定仍需数据说明或生成代码确认。数据读取没有额外归一化，浮点 MAE 不能擅自解释为 mV。兼容 NRMS/SSIM 使用现有评价器的 [0,1] 截断和 uint8 转换口径。

## 推荐执行顺序

先在服务器做 100 步诊断，验证显存与环境：

```bash
bash b0/launch_server.sh /path/to/IR_drop /path/to/results/b0_server_probe_seed0 --steps 100 --log-every 10
```

服务器确认数据来源、划分用途和训练资源后，固定配置执行完整开发集 baseline：

```bash
bash b0/launch_server.sh /path/to/IR_drop /path/to/results/b0_development_seed0_200k
```

可设置 `B0_PYTHON=/path/to/env/bin/python`。默认单 GPU，可通过 `CUDA_VISIBLE_DEVICES=0` 指定。若 batch=2 显存不足，先停止并检查环境；batch_size=1 会改变配方，应另列实验且后续候选统一采用同样预算，不自动回退或悄悄开启混合精度。

程序默认前后各验证一次、结束后重新载入权重再验证。前后验证可用于工程诊断，mini 数据不用于性能结论。没有验证集早停、挑选 checkpoint 或隐藏 test 入口。

## 输出与恢复边界

每次输出到全新目录，包含 config.json、environment.json、source.json、source_snapshot.zip、数据清单副本与哈希、每步 loss/lr/sample_ids、result.json、checkpoint_final.pt；每 10000 步另存阶段 checkpoint。失败时保留日志和 failure.json。

checkpoint 包含模型权重、步数、配置与来源，支持推理和独立评价；不包含 optimizer、采样进度和 RNG 恢复状态，因此当前不支持无缝断点续训。遇到中断，应保留失败记录，在新目录重跑，不能把分段重启视为连续完整训练。

source.json 如实记录 Git commit 和 dirty 状态，并保存实际源码快照与 SHA256。今日 mini 运行是工程验证，未把工作区强行清理或提交；正式可比较实验应先由项目负责人冻结版本、确认 clean Git，再执行。

最终 result.json 的 peak_allocated_mib 包括验证和 checkpoint 重新加载阶段，不能只解释为训练峰值；training_seconds 是含数据读取的训练阶段墙钟时间，single_gpu_training_hours 以此计算。该入口没有报告热点、按设计等权 MAE、跨设计 test 或 SA0 会话成本，这些是后续研究协议的工作。

## 本机 mini 复跑命令

```bash
python -m b0.run \
  --data-root /path/to/training_set_full_mini/IR_drop \
  --train-manifest prepare/manifests/smoke_train.csv \
  --validation-manifest prepare/manifests/smoke_validation.csv \
  --output /path/to/results/b0_mini_new_run \
  --scope mini --steps 100 --lr-horizon-steps 200000 \
  --batch-size 2 --log-every 10 --check-values
```

mini 有 3 个训练、1 个验证样本。drop_last=True 且 batch_size=2 时，每次遍历只取一个完整 batch，之后重新洗牌；100 步不是 100 个独立样本。默认全部使用原始 256×256 分辨率。
