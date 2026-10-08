# B1 固定 MAVI 随机学习率搜索

2026-10-08默认更新为1000步、新会话b1_pilot128_random1000_001。B1暂不导入共享B0，仍自行建立匹配基线；旧500步说明为历史预跑记录。

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python -m b1.controller
```

配置为b1/config.json，新会话默认results/b1_pilot128_random_001。
固定MAVI、优化器、loss、数据、评价口径和训练seed；只改变初始学习率。
当前为128训练/32验证、500步、训练batch=2、验证batch=1、余弦周期200000步，最多两轮候选。
数据审计未通过拒绝启动；新数据重新建立匹配基线，候选均从头初始化。
max_session_process_seconds=null关闭累计时间上限，14400秒单任务保护仍保留。
具体配置及B1尚未接入三seed确认的边界见[预跑说明](../PILOT_RUN_20261007.md)。
不调用agent、不联网、不改模型源码，不读取历史结果来选择候选。

search_seed控制候选抽样；seed控制模型初始化和训练样本顺序，二者职责不同。
当前initial_lr在(1e-7, 1e-3)间按log_uniform抽样，排除基线值及重复候选。
对数均匀指log(learning rate)上均匀抽样，不是学习率本身均匀抽样。
会话开始前一次生成并保存search_plan.json，失败后不改变剩余候选，不按结果调搜索范围。
此范围是本地工程试跑配置，并非论文已预注册搜索空间，正式实验前双方共同冻结范围。

复用SA0训练器、指标判定、数据/源码hash核验、基线缓存和成本口径。
每轮保存candidate.json、完整源码workspace、训练日志、逐步loss/sample_ids、环境、
指标、checkpoint和record；summary保留全部失败结果与符合三指标规则的候选列表。
test_limits下改善仍是keep=null，不能正式保留。MAE降低但其他指标变差不能胜出。
暂不额外按MAE挑一个“最优”模型，避免悄悄改变共同选择规则。

同一配置和session_dir再次启动不重复已归档训练；中断轮次不自动免费重试。
更换配置、搜索计划、数据、源码、执行代码会拒绝恢复，需新session_dir。
缓存基线仍扣除原基线预算成本，actual_current_process_seconds报告实际新增进程/复用开销。
程序上限包括失败和中止训练。它不是严格GPU-hours账本；正式研究仍需固定统一GPU预算。

这个版本比较的是随机学习率搜索与agent研究。B1固定模型且搜索空间较窄，
正式结论应披露这种差异，不能声称已代表全面调参或BO。扩大超参数空间需先冻结规则，
并重新训练不满足基线复用条件的基线。本地内存不足不会由随机搜索自动解决。

```bash
python -m unittest sa0.test_session sa1.test_sa1 b1.test_b1 -v
```

测试模拟agent和训练进程，不消耗真实模型调用或GPU训练。
