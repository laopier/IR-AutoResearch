# Kaggle B0 小规模验证

先确认 Kaggle Notebook Settings 能选择 GPU。账号已有不代表 GPU 权限已启用；按网站提示完成所需验证。此包不需要 Kaggle API key 或 GitHub token。

## 生成交接材料

用 Python 3.10+ 在仓库运行；输出目录必须不存在。代码取当前已提交 HEAD，数据只读取 smoke 清单中的 4 对文件。

```powershell
python kaggle/build_mini.py --data-root D:/WSL/Project/CircuitNet-main/training_set_full_mini/IR_drop --output D:/Project/IR-AutoResearch_kaggle_mini_20261002
```

输出包括 `B0_mini.ipynb`、`IR_AutoResearch_kaggle_mini.zip`、`BUILD.json`。zip 保存 Git bundle、8 个数据文件及 SHA256；不会包含 GitHub 登录凭据或旧模型权重。

## 上传与运行

1. 将 zip 上传到自己的 **Private Dataset**，核对可见性。
2. 导入 `B0_mini.ipynb` 为 **Private Notebook**，添加刚才的 Dataset。
3. Settings 中启用 GPU。Internet 可以关闭，所有代码与数据已在 zip 内。
4. 顺序运行所有单元格。沿用 B0 配方，100 次更新、batch=2、seed=0、学习率周期 200000。只使用一块 GPU。
5. 完成后 Save Version 保存输出，并下载 `_outputs.zip`。结果里的环境、速度、显存与 checkpoint reload 验证用于确认迁移情况。

不要把小样本成绩当成完整 B0。2026-10-02 首次 Kaggle 包冻结于 `60a4ab0`，只保存模型权重，不支持精确续训。后续 B0 已增加 `--resume`，见 `b0/README.md`；要使用新功能，必须用提交后的新代码重新打包。旧输入包不会随本地修改自动更新。Notebook 暂不自动发现父 checkpoint，恢复命令需显式设置 `--resume` 和新的累计目标。

未自动安装固定依赖：先使用平台预装的 torch/numpy/cv2 并记录版本。环境差异可能导致数值不同；若导入或执行失败，保存错误后再适配，避免未经检查替换 CUDA PyTorch。

本地执行 notebook 的短程测试只验证打包与代码流程，不代表已经在 Kaggle GPU 上完成运行。
