# IR-AutoResearch

IR-drop 预测器与自主研究流程的学习、复现和实验管理仓库。
当前实现 B0 固定训练、B1 随机学习率搜索、SA0 离线自主研究和 SA1 联网自主研究。
已完成部分真实小数据闭环与模拟测试，尚未完成正式科研对照、独立重复与跨设计隐藏评测。

## 当前进度

- MAVI 数据入口、模型、训练接口及统一 MAE/NRMS/SSIM 评价器。
- B0 固定配方：3 个训练样本、1 个验证样本，100 步训练、权重保存与重新加载验证。
- SA0 控制器：冻结起点、提前登记假设、权限/预算检查、独立评估、保留/回退、版本及成本记录。
- 程序化模型接入：通过已有 Codex CLI 登录提出候选，控制器训练验证，再把结果交回模型。
- B1：固定MAVI，提前生成随机学习率计划，不调用agent。
- SA1：使用同一训练/判定接口，开放web_search并记录搜索事件、来源及采用理由。
- 研究轨迹和学习导读。正式跨设计划分、完整训练、多轮自主搜索与硬 token 预算仍需推进。

## 从哪里开始阅读

| 位置 | 用途 |
| --- | --- |
| [项目导读](PROJECT_WALKTHROUGH_20261001.md) | 调用链、文件逐一说明、服务器连接与 SA1 计划 |
| [prepare](prepare) | 数据清单、Dataset 和评价器 |
| [train](train) | MAVI 与单步训练接口 |
| [B0 说明](b0/README.md) | 官方配方核对、Linux 服务器训练 |
| [B0 交接](B0_HANDOFF_20261001.md) | 已完成的验证和服务器交接步骤 |
| [SA0 入口](sa0/controller.py) / [完整说明](sa0/SA0_GUIDE.md) | 离线多轮研究、冻结源码、基线复用、预算和恢复 |
| [B1 入口](b1/controller.py) / [说明](b1/README.md) | 固定模型的随机学习率搜索 |
| [SA1 入口](sa1/controller.py) / [说明](sa1/README.md) | 联网提案及可回查的检索记录 |
| [实验规则](program/README.md) | 基础权限、评测与实验原则 |
| 本地 results 目录 | 新运行的结果、prompt、日志和成本；不随代码同步 |

## 运行与环境

本机已验证 WSL Linux、Python 3.11、CUDA-enabled PyTorch 2.11、RTX 3070 Laptop。
GPU 数据不随 Git 分发。先按 [B0 README](b0/README.md) 配置环境、数据与预检查。
服务器启动器：

```bash
bash b0/launch_server.sh DATA_ROOT NEW_OUTPUT_DIR --preflight-only
```

模型接入版目前从 Windows 调用 Codex CLI，在本机 WSL 训练；纯 Linux/SSH 执行后端还未实现。
模型调用需要已有登录。新的实验必须使用新的输出目录，不能覆盖已有记录。
SA0禁用研究检索，SA1开放内置web_search；远程模型调用所需网络不等于研究检索权限。



## 版本与产物

仓库保留原有开发历史，`main` 管理当前代码和文档。
`results/b0_mini_20261001_seed0_steps100` 保存 B0 小规模结果。
`results/sa0_mini_20261001` 与 `results/sa0_agent_loop_20261001` 保存两次不同的演练记录。
早期SA0导出包含隔离工作区的Git bundle，可以恢复对应代码历史；不复制训练数据。
大型 `.pt` 权重放在本 private 仓库的 Release 附件，位置与 SHA256 见 [产物索引](ARTIFACTS_20261001.md)。

原始 `.npy` 数据、认证文件、环境目录和 Python 缓存不进入 Git。
本仓库中的本机路径是实验来源记录；另一台机器运行时需要配置自己的数据与解释器路径。

## 第三方来源

MAVI 与参考训练代码来自 [CircuitNet](https://github.com/circuitnet/CircuitNet)，许可证见 [CIRCUITNET_LICENSE](CIRCUITNET_LICENSE)。
MMCV 初始化参考许可证见 [MMCV_LICENSE](b0/reference/MMCV_LICENSE)。
本项目借鉴 [AuDoPEDA](https://arxiv.org/abs/2601.06268) 的阶段划分，没有复现其完整系统。
