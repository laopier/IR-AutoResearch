# SA0 离线单 agent 自主研究

默认入口已升级到假设驱动V3，读取research_config.json，详见[RESEARCH_V3.md](RESEARCH_V3.md)。
当前128/32预跑与独立full-selection说明见[预跑配置](../PILOT_RUN_20261007.md)。
下文的config.json/SA0_GUIDE.md是旧版流程说明，使用`--legacy`访问。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m sa0.controller
```

配置在config.json，研究规则在PROMPT.md，完整说明见[SA0_GUIDE.md](SA0_GUIDE.md)。
controller提供训练和比较接口，session负责冻结起点、历史、预算、多轮研究与恢复。
agent每轮自主选择学习率或模型修改，proposal验证结构并生成独立候选源码。
所有候选从头初始化，对固定基线使用MAE严格降低、NRMS不升、SSIM不降规则。
test_limits下改善候选仍为keep=null，不视为正式资源合规。

支持核验后复用旧会话基线；新实验使用新的session_dir，不覆盖原始记录。
同会话配置与执行代码不变时可恢复，已归档轮次不重跑。
仅工作区/hash层面的实验隔离，不是生成代码的操作系统安全沙箱。

旧学习脚本保留用于理解逐步实现；当前入口不依赖它们。
早期control/agent_loop演示入口已由统一controller替代，旧版本可从Git历史查看。
本地结果和数据不随当前代码更新分发；跨设备运行需要配置自己的路径和环境。
