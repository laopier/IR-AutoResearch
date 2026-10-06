# 本地交接包验证记录

执行日期：2026-10-02。未在 Kaggle 上运行。

- 源码冻结版本：`60a4ab02c409be1966c9d7ae7e84417f100ee5a1`。
- 输入 zip：14,393,986 bytes，SHA256 `0dde525cbded373d32b04e0e730e2b97b2b79ccda4c8228211a199b713d46391`。
- 在 WSL 中执行生成 Notebook 的所有代码单元格，仅替换 input/working 路径并将 STEPS 从 100 改为 2。
- 4 对数据与源码 bundle 哈希检查通过；Git clone 后 HEAD 校验通过。
- B0 batch=2、seed=0、lr horizon=200000；2 次更新完成。
- checkpoint 保存后重新加载验证通过，结果 zip 导出成功。
- RTX 3070 Laptop 8GB，训练 8.87 秒，峰值 allocated 显存 6363 MiB。该速度不代表 Kaggle GPU 速度。

本地日志：`D:/Project/kaggle_mini_local_validation/b0_mini_20261002T034017820258Z/result/result.json`。

下一步：确认 Kaggle GPU 权限，将输入 zip 添加为 Private Dataset，导入 Notebook 做 100 步云端验证。当前没有云端训练完成记录。
