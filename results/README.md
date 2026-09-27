# results/（实验结果）

每次运行

```bash
python scripts/run_all.py --config configs/<配置名>.yaml
```

都会在这里生成一个文件夹 `results/<日期-时间>_<实验名>/`。里面只有小文件，**应该上传到 GitHub**，方便以后对比、也方便 Claude 读取和分析：

| 文件 | 内容 |
|---|---|
| `REPORT.md` | 自动生成的实验报告：数据集概要、蒸馏配置、准确率表格、对比图 |
| `config.yaml` | 这次实验实际使用的全部参数 |
| `eval_results.json` | 测试集准确率（test accuracy）：`kd` 与 `kd_topk` 两个学生 |
| `teacher_metrics.json`、`student_*_metrics.json` | 教师 / 学生的最佳验证集准确率（best validation accuracy） |
| `student_ablation.png` | 两个学生的准确率对比图 |
| `run_info.json` | 代码版本（git commit）、是否有未提交的改动、设备、耗时、Python / PyTorch 版本 |

模型权重（checkpoint，`.pt`）和缓存数据在 `artifacts/<同名文件夹>/`，文件大，**不上传**。

小提示：`run_info.json` 里 `uncommitted_changes` 为 `true` 时，说明跑实验时代码有没提交的改动，
这个结果以后可能无法完全复现。正式实验前先把代码提交（commit）。
