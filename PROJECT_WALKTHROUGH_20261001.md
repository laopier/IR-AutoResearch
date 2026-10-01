# IR-AutoResearch：操作链条与文件导读

扫描日期：2026-10-01。项目根目录为 `D:\Project\IR-AutoResearch`。
当前实现为 B0 固定配方训练、SA0 单候选控制器及模型接入演练；没有已实现的 SA1 入口。
下文分别描述实际代码与未来 SA1 的设计，不能把计划当作已经完成的功能。

## 1. 校外操作服务器：先解决连接，再解决训练执行位置

在深大不影响技术上远程操作科大服务器，但要具备允许的访问路径和服务器账户。
三种可选路径：

1. 获得适用的学校/超算 VPN 权限，从你的电脑连接 VPN 后 SSH 登录服务器。
2. 你与服务器都安装 Tailscale，配置允许的节点访问，用服务器的 Tailscale 地址连接已有 SSH 服务。
3. 校内服务器不能安装软件时，远程控制一台有权使用、常开、能 SSH 到该服务器的校内电脑，再从那台电脑登录服务器。

UU 等远程桌面属于第 3 种路径，操作画面和键鼠。Tailscale 提供设备之间的网络连通，SSH 提供命令行登录。
只在你自己电脑安装 Tailscale，不会自动获得校园网所有地址的访问能力。
如果 Tailscale 只安装在校内电脑上，服务器没有加入 tailnet，可以在那台电脑上执行 SSH；
也可以由管理员配置受限的 SSH 跳板或 subnet router。不要默认开放整个校园子网。

Tailscale 会尝试直连，不能直连时可能经 relay，速度需实测。校园网络的访问限制仍可能阻止连接。
官方依据：[防火墙与中继](https://tailscale.com/docs/integrations/firewalls)、
[Subnet routers](https://tailscale.com/docs/features/subnet-routers)。

科大超算中心提供针对其用户的校外 VPN，不用于访问其它校内 IP；普通校园 OpenVPN 页面说明仅面向教师。
因此需要先确认学长说的是超算中心机器还是实验室机器，以及你的账户可使用哪条路径。
官方依据：[超算 VPN](https://scc.ustc.edu.cn/portal/vpn)、[校园 OpenVPN](https://openvpn.ustc.edu.cn/)。
UU 的实际被控系统支持和连接效果需用网易官方客户端核对，这里没有确认其 Linux 支持或下载地址。

远程连接与实验执行是两件事：

```text
你的电脑 --远程连接--> 科大电脑或服务器 --SSH/作业提交--> GPU 训练
```

当前 `sa0/agent_loop.py` 从 Windows 启动本机 WSL，并没有 SSH 远程执行后端。
连接学长服务器后仍需迁移：要么把流程改成在服务器本地运行，要么保留本地模型调用、
通过 SSH/作业系统提交远端训练并取回日志。集群应使用管理员规定的提交方式。
大数据和 checkpoint 尽量留在服务器，只回传代码变化、日志和指标；长任务由持久会话或作业系统管理。

连接 GPU 的网络与 SA1 的联网权限不同。SA0 也可远程使用 GPU 和调用云端模型，
它禁止的是 agent 为研究获取额外外部资料。

## 2. 当前 SA0 模型接入：一次完整操作链条

### 第一步：启动并检查环境

运行 `python -m sa0.agent_loop ...`。
`main()` 解析 session 目录、数据根目录和 WSL Python 路径，交给 `run()`。
它要求 Windows、Codex CLI 在 PATH 中、已有登录状态、输出目录尚不存在。
这里只检查登录状态，不读取、打印或复制认证文件。

### 第二步：冻结起点和数据

`run()` 通过 WSL 执行 `sa0.control init`。
`control.init()` 校验 smoke 清单的 SHA256 与步数预算，确认文件存在和训练/验证不重叠。
然后复制 prepare/program/train/tests 到 session/workspace，建立独立 Git 仓库，提交初始版本。
3 张训练图及标签复制到 data_train，1 张验证图及标签复制到 data_validation。
最后保存协议、起点 commit、源码和数据哈希。
独立副本的意义是可追溯地探索，不把当前项目的基线代码直接改成候选。

### 第三步：先跑同预算基线

`control run --role baseline` 校验代码、数据、干净 Git 状态和剩余预算。
它分别启动两个 worker：train worker 训练 5 步并保存权重；evaluate worker 重新加载权重并统一评估。
训练使用 `IRDropDataset -> DataLoader -> build_model/build_optimizer/build_scheduler -> train_batch`。
验证使用 `model.eval() -> torch.no_grad() -> evaluate_batch`。
两阶段是独立进程，训练 worker 的 Python 文件白名单不含验证数据目录。
这些护栏不是完整的操作系统安全沙箱。

### 第四步：把资料和实测基线交给模型

`agent_loop.model_call()` 把以下内容写入 prompt：人工整理的接口卡片、当前训练源码、
固定协议、基线训练轨迹和验证指标。
通过已登录 Codex CLI 发起非交互调用。该调用禁用外部检索、shell、协作和插件，
要求模型输出符合 JSON Schema 的假设、风险、控制变量、详细计划与精确 old/new 修改片段。
模型不在这次调用中直接编辑工作区或自行启动训练。

### 第五步：检查并应用模型建议

`validate_edits()` 检查字段、精确匹配、Python AST、函数签名和修改范围。
第一版只允许修改 `train/experiment.py` 中一个函数：`compute_loss` 或 `build_optimizer`。
虽然通用 control.py 允许 mavi.py 与 experiment.py，模型接入版的范围更窄。
先用 `control propose` 登记假设，再写入片段替换，再用 `control commit` 提交候选。
这能证明假设在修改前已记录，避免事后把结果包装成原先的预期。
AST 检查不等于能证明候选满足全部科研语义或阻止所有恶意代码。

### 第六步：跑候选并统一筛选

`control run --role candidate` 在相同清单、步数、batch size、seed 下训练和评估。
代码 diff、commit、checkpoint、指标和资源消耗都保存到独立 trial 目录。
`control decide` 按提前冻结的规则保留或回退；回退新增 commit，保留失败候选的历史。
本轮规则、5 步预算和 1.1 倍显存阈值只是演练参数，不是学长方案或论文提供的正式阈值。
不同损失函数的训练 loss 数值不宜直接比较，选择使用同一验证器的指标。

### 第七步：把结果交回模型

第二次 `model_call()` 携带第一轮完整 proposal、基线/候选结果和控制器决定。
模型输出证据解释、局限性、下一假设和下一验证方法。
两个调用靠显式传递完整研究历史连接，不依赖模型自动记住上个 CLI 进程。
这是同一个研究角色的连续步骤，不是 planner/executor 两个独立 agent。
下一假设只记录，因为本轮预算只有 baseline 加一个候选。

### 第八步：独立核对产物

`verify_session.py` 核对源代码/数据/checkpoint 哈希、原项目未改动、Git 状态、采样序列、
学习率序列、调用账本与资源上限。只有未改优化器的候选才适用其“学习率轨迹相同”检查；
若以后允许学习率本身作为候选变量，需要按登记的控制变量调整核对逻辑。

## 3. SA1：未来多出的链条

学长定义：SA1 = SA0 加外部检索权限，固定其他实验条件，用于测量检索的收益和成本。
目前没有 sa1/ 目录、检索工具、检索日志或已运行的 SA1 实验。

```text
冻结起点/预算
    -> agent 理解代码与当前结果
    -> agent 提出检索问题
    -> 检索允许的外部资料，记录来源和检索成本
    -> agent 解释采用哪条证据、提出假设、定位修改
    -> 控制器检查并应用修改
    -> 同预算训练、统一验证、保留/回退
    -> 结果反馈 -> 下一研究轮次
```

SA1 应保留与 SA0 相同的起点、基础模型/版本、数据和隐藏测试隔离、编辑权限、训练资源预算、
接受规则和独立研究轮次数。额外记录查询词、URL、时间、资料快照/摘要、采用理由和检索 token/时间/费用。
文献检索结果应作为证据输入，不能把网页内容当作新的执行权限或项目指令。
不得通过联网获取含隐藏测试知识的权重或标签。

不能只把 `web_search="disabled"` 改成启用：当前 prompt 禁止工具活动，model_call() 还会拒绝检索事件。
需要显式设计检索接口、允许的工具、日志与成本，并保持实验控制器的训练约束。
正式多轮循环、失败反馈/修复、硬 token 预算和可靠沙箱也尚未完成。

## 4. 项目根目录的文件逐一说明

以下路径都相对于 `D:\Project\IR-AutoResearch`，同名初始化文件只做 Python 包标记。

### prepare：固定的数据入口和评价器

| 文件 | 作用 |
| --- | --- |
| prepare/__init__.py | 使 prepare 成为 Python 包，无训练逻辑。 |
| prepare/dataset.py | 读清单和 .npy，检查路径、数值和形状，转 float32 tensor；feature 从 H,W,24 变成 1,24,H,W，label 变成 H,W。不负责从原始仿真生成数据或重新归一化。 |
| prepare/evaluator.py | 统一计算浮点 MAE，以及官方 8-bit 口径的 NRMS、SSIM；候选不能修改评分标准。 |
| prepare/split_manifest.py | 读取清单、识别设计族、生成划分、检查不重叠。分组工具并不证明现有清单满足正式跨设计隔离。 |
| prepare/manifests/smoke_train.csv | 3 个训练样本的 feature/label 相对路径，用于演练。 |
| prepare/manifests/smoke_validation.csv | 1 个验证样本，用于演练中的选型，不是隐藏测试。 |
| prepare/manifests/train_N28_seed0.csv | 6370 个 RISCY 开发训练样本的清单，不是数据本体。 |
| prepare/manifests/validation_N28_seed0.csv | 708 个 RISCY 开发验证样本的清单；与训练均属 RISCY，不能据此宣称跨设计泛化。 |

### train：待研究的预测器与训练配方

| 文件 | 作用 |
| --- | --- |
| train/mavi.py | MAVI 网络结构及初始化，决定张量怎样经过 3D/2D 层得到 IR 图像。 |
| train/experiment.py | build_model、AdamW、余弦 scheduler、compute_loss 与 train_batch；连接模型和一个参数更新步骤。 |

### program：早期 runner 与规则

| 文件 | 作用 |
| --- | --- |
| program/__init__.py | 包标记。 |
| program/README.md | 人可读的 AutoResearch 目标、权限、数据隔离与改动原则。 |
| program/policy.json | 机器可读的 smoke 预算、清单哈希和路径权限；正式 search/confirmation/final_test 当前禁用。 |
| program/runner.py | 早期固定步数的训练/验证与结果写出入口；不是模型调度器，也未完整执行 policy 的所有要求。当前 SA0 走 control.py/worker.py，不能把三个入口理解成串行都要调用。 |

### b0：固定官方配方的训练与交接

| 文件 | 作用 |
| --- | --- |
| b0/__init__.py | 包标记。 |
| b0/run.py | 固定 B0 配方、预检查、训练、验证、保存 checkpoint、重新加载验证和保存来源记录；不自动搜索候选。 |
| b0/launch_server.sh | Linux 服务器命令包装，传入数据根和新输出目录，调用 b0.run；不负责网络登录或申请 GPU。 |
| b0/requirements-validated.txt | 已验证的 NumPy/OpenCV 依赖版本；PyTorch 需根据服务器驱动安装。 |
| b0/verify_official.py | 固定初始化与输入，对照官方参考核验权重、前向输出、损失、一次优化更新和学习率。 |
| b0/test_preflight.py | 数据预检查与学习率配方相关测试。 |
| b0/README.md | 配方、环境、数据布局、服务器运行及限制说明。 |
| b0/reference/circuitnet_mavi.py | 官方 MAVI 参考源码，用于核对，不是候选编辑入口。 |
| b0/reference/circuitnet_train.py | 官方训练器参考。 |
| b0/reference/circuitnet_configs.py | 官方训练配置参考。 |
| b0/reference/circuitnet_losses.py | 官方 loss 参考。 |
| b0/reference/mmcv_weight_init_v1_6_0.py | 上游初始化函数参考，用于没有完整 MMCV 的核对环境。 |
| b0/reference/MMCV_LICENSE | MMCV 参考文件的许可证。 |

### sa0：控制器与模型接入

| 文件 | 作用 |
| --- | --- |
| sa0/__init__.py | 包标记。 |
| sa0/agent_loop.py | Windows 外层流程：CLI 模型调用、结构化输出、验证/应用 edits、驱动 WSL 控制器、将实测结果交回模型。当前只有一次候选，不能直接部署到纯 Linux 或远程 SSH 后端。 |
| sa0/control.py | 可信实验控制器：init、propose、commit、run、decide；冻结起点、限制修改和预算、记录版本、训练评估、保留/回退。 |
| sa0/worker.py | 实际 GPU 工作进程；train 模式更新参数并存权重，evaluate 模式严格加载权重并调用统一评价器。 |
| sa0/verify_session.py | 只核对既有产物，不新增训练；对应当前 agent_candidate 目录命名和第一轮未改学习率的实验。 |
| sa0/test_control.py | 权限、预算、日志不覆盖、提前登记、Python audit 和真实 Git 回退的测试。 |
| sa0/test_agent_loop.py | 路径越界、无效片段、部分应用及不允许改动范围的测试。 |
| sa0/hypothesis_smooth_l1.json | 第一轮由当前对话提出的 beta=0.1 示例假设；不是接入版模型自动提出的 beta=0.01，也不是每轮固定候选。 |
| sa0/TASK.md | 第一轮人工监督控制器演练的任务与限制。 |
| sa0/README.md | 两种演练入口和手工控制器命令。 |
| sa0/DESIGN_SOURCES.md | 学长方案与 AuDoPEDA 的对应关系，哪些借鉴、哪些未实现、模型接入限制。 |

### tests：保护基础功能的测试

| 文件 | 作用 |
| --- | --- |
| tests/test_dataset.py | 数据数值、形状、路径边界与批次契约。 |
| tests/test_split_manifest.py | 清单格式、重复样本、划分确定性和不重叠。 |
| tests/test_evaluator.py | 指标与官方参考一致、完美预测、常量图像及非法输入。 |
| tests/test_mavi_contract.py | 模型前向形状、有限数值和反向梯度。 |
| tests/test_train_experiment.py | 训练构造接口、损失形状和有限性、参数确实更新。 |
| tests/test_data_model_integration.py | 数据读入后能真实经过 MAVI，保护数据/模型接口。 |
| tests/test_runner.py | 早期 runner 预算与结果记录。 |

### results 与根目录

| 文件 | 作用 |
| --- | --- |
| results/smoke_mavi_seed0_step1.json | 早期 1 步 smoke 结果记录。 |
| results/b0_mini_20261001_seed0_steps100/config.json | 当次 B0 的训练设置。 |
| results/b0_mini_20261001_seed0_steps100/environment.json | 软件、CUDA、GPU 环境。 |
| results/b0_mini_20261001_seed0_steps100/data_report.json | 清单、文件和形状的预检查记录。 |
| results/b0_mini_20261001_seed0_steps100/source.json | Git、源码来源、哈希与快照信息。 |
| results/b0_mini_20261001_seed0_steps100/source_snapshot.zip | 当次源码快照，避免当前代码变化后丢失实验起点。 |
| results/b0_mini_20261001_seed0_steps100/train_manifest.csv | 当次固定的训练清单副本。 |
| results/b0_mini_20261001_seed0_steps100/validation_manifest.csv | 当次固定的验证清单副本。 |
| results/b0_mini_20261001_seed0_steps100/training.jsonl | 每步训练 loss、学习率等轨迹；一行一条 JSON。 |
| results/b0_mini_20261001_seed0_steps100/checkpoint_final.pt | 最终模型权重；没有完整优化器状态，不等于可精确续训的 checkpoint。 |
| results/b0_mini_20261001_seed0_steps100/result.json | 汇总指标、训练资源和重新加载验证结果。 |
| CIRCUITNET_LICENSE | CircuitNet 代码许可证。 |
| B0_HANDOFF_20261001.md | 给服务器执行者的交接说明。 |
| .gitignore | 忽略 Python 缓存与编辑器配置。 |
| PROJECT_WALKTHROUGH_20261001.md | 本导读：连接方案、当前调用链、SA1 计划与文件用途。 |

`.git/` 是 Git 管理的版本对象、索引、配置和日志，不属于研究实现；不要逐个修改内部文件。
`__pycache__/` 和 `.pyc` 是 Python 自动生成的字节码缓存，不是新的训练源码或实验结果。

## 5. 实验目录：保存一次研究历史

本轮位于 `D:\Project\sa0_runs\20261001_agent_loop_01`，它在项目源码目录之外。
先前 `20261001_mini_sa0_01` 是人工监督控制器演练，二者是不同实验，不能混用候选或结果。

### session 根目录与数据

| 文件/目录 | 作用 |
| --- | --- |
| protocol.json | 训练/评估预算、随机种子、步数、筛选和权限约定。 |
| session.json | 初始 Git commit、来源、数据/源码哈希。 |
| proposal.json | 改码前登记的候选假设。 |
| proposal_receipt.json | 登记假设的哈希与登记时的代码 commit。 |
| events.jsonl | session 初始化、假设登记、候选提交、试验完成和决定的时间线。 |
| decision.json | 实测筛选检查、保留/回退决定和选中 commit。 |
| workspace/ | 独立 Git 仓库，包含 prepare/program/train/tests 的副本；文件角色与前文相同。 |
| data_train/manifest.csv | 本次训练清单。 |
| data_train/feature/*.npy | 3 个训练特征数组，每个与同名 label 配对。 |
| data_train/label/*.npy | 3 个预处理后的 IR-drop 标签数组。 |
| data_validation/manifest.csv | 本次验证清单。 |
| data_validation/feature/*.npy | 1 个验证特征数组。 |
| data_validation/label/*.npy | 对应验证标签；参与选型，不是隐藏测试。 |

### 每次 trial 的全部文件

`trials/baseline/` 与 `trials/agent_candidate/` 的同名文件作用一致。

| 文件 | 作用 |
| --- | --- |
| hypothesis.json | 当次假设，基线则说明未修改代码。 |
| source.json | 运行的 commit、代码哈希、改动路径和 baseline/candidate 角色。 |
| code.diff | 相对冻结起点的修改；基线为空 diff。 |
| training.jsonl | 每步 loss、学习率、抽到的样本；用来核对公平比较。 |
| train.log | 训练进程 stdout/stderr，失败时先查看。 |
| train.json | 完成步数、参数量、训练时间、峰值显存与运行环境。 |
| checkpoint.pt | 当次训练后的模型 state_dict，不含完整优化器续训状态。 |
| evaluate.log | 独立评估进程 stdout/stderr。 |
| evaluate.json | 验证 MAE/NRMS/SSIM、样本数、评估时间和峰值显存。 |
| result.json | 合并协议、版本、训练、评估与 checkpoint 哈希的结构化结果。 |
| ledger.json | 成功/失败、计入预算的墙钟时间与错误；失败也消耗次数和预算。 |

### agent 目录与两次模型调用

| 文件/目录 | 作用 |
| --- | --- |
| agent/protocol.json | 模型调用次数和超时、编辑范围、启动器哈希与演练范围。 |
| agent/context/ | 本轮为空，是 CLI 的工作目录；实际代码作为 prompt 提供，不是模型自行扫描这里。 |
| agent/hypothesis.json | 从模型 proposal 提取的五个假设字段，提交给 control propose。 |
| agent/summary.json | 第一轮 proposal、控制器决定和模型反馈的合并记录。 |
| agent/loop_ledger.json | 外层闭环总耗时及成功/失败，不等于 GPU 内核活跃时间。 |
| agent/verification.json | 事后哈希、轨迹、预算与 token 核对结果。 |
| agent/REPORT.md | 人可读实验报告。 |

`agent/01_proposal/` 是提出候选的调用，`agent/02_feedback/` 是分析实测结果的调用；各有：

| 文件 | 作用 |
| --- | --- |
| prompt.txt | 真正发给模型的上下文和要求；看它可判断模型知道什么。 |
| schema.json | 模型最终 JSON 应有哪些字段和类型。 |
| request.json | CLI 命令、超时、登录方式和检索设置；不是认证密钥。 |
| events.jsonl | CLI 原始事件流、模型输出及 turn.completed usage；与 session 根事件流不同。 |
| response.json | 模型结构化输出：01 包含假设/plan/edits，02 包含解释/局限/下一假设。 |
| stderr.log | CLI 诊断、警告或调用错误。 |
| ledger.json | 当次模型调用耗时、CLI 返回的 token、金额未知字段与错误。 |

## 6. 适合当前学习进度的阅读顺序

已经理解 dataset 和模型后，下一步看 `agent_loop.py` 的 `run()`，先沿调用链找输入和输出。
然后看 `model_call()`：subprocess 启动 CLI、stdin 传 prompt、JSON Schema 控制输出、JSONL 提取 usage。
再看 `validate_edits()` 与 `control.py` 的 propose/commit/run/decide，理解如何把模型建议变成可审核实验。
最后从真实 `01_proposal/response.json -> code.diff -> evaluate.json -> 02_feedback/response.json` 对照一遍。
理解现有链条之后，再增加 SA1 的检索步骤，避免一次引入太多不可解释的变化。
