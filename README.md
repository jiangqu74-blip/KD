# KD：超图知识蒸馏（Hypergraph Knowledge Distillation）

[![CI](https://github.com/jiangqu74-blip/KD/actions/workflows/ci.yml/badge.svg)](https://github.com/jiangqu74-blip/KD/actions/workflows/ci.yml)

**一句话**：先训练一个能看到超图结构的**教师模型**（teacher），再把它的知识**蒸馏**（knowledge distillation）给一个只看节点特征的**学生模型**（student，0-hop MLP），比较两种蒸馏方式：**KD** 和 **KD+TopK**。
方法细节见 [`docs/method.md`](docs/method.md)。

> **注意**：当前的合成数据太简单，只看节点特征就能 100% 分类，所以还比较不出两种方法的差别。详见文末「已知问题」。

## 目录结构（Directory layout）

```text
KD/
├── src/                     核心代码（source code）
│   ├── data.py                生成合成超图数据
│   ├── models.py              教师 TeacherHGNN、学生 StudentMLP
│   ├── rgvq.py                RGVQ-lite 残差软量化
│   ├── distill.py             导出蒸馏包（Top-K 角色）
│   ├── train_teacher.py       训练教师
│   ├── train_student.py       训练学生（kd / kd_topk）
│   ├── eval.py                测试集评估
│   ├── plots.py               画图
│   └── main.py                命令行入口：python -m src.main --mode ...
├── configs/                 实验配置（每个参数都有中文注释）
│   ├── quick.yaml             冒烟测试（smoke test）：只检查流程能不能跑通
│   └── default.yaml           默认实验
├── scripts/
│   ├── run_all.py             ★ 一键跑完整实验
│   └── update_report.py       生成实验报告
├── results/                 实验结果汇总（小文件，上传 GitHub）
├── artifacts/               模型权重、缓存数据（大文件，不上传；运行后自动出现）
├── data/                    真实数据集放这里（不上传）
├── notebooks/
│   └── colab_quickstart.ipynb 在 Google Colab 上用免费 GPU 跑
├── docs/method.md           方法说明
├── tests/                   自动测试（tests）
├── .github/workflows/ci.yml GitHub 上的自动检查（CI）
├── requirements.txt         需要安装的 Python 包
└── Makefile                 常用命令的快捷方式
```

## 在哪里跑实验？（Where to run experiments）

| 地方 | 硬件 | 适合做什么 | 怎么用 |
|---|---|---|---|
| **Claude 云端会话**（cloud session） | 4 核 CPU、15 GB 内存、**没有 GPU**；临时机器，闲置后会被回收 | 写代码、调试、小规模实验、分析结果 | 在会话里直接说「跑一下 default 实验」 |
| **Google Colab** | 免费 GPU（有使用时长限制；需要能访问 Google） | 需要 GPU 的实验 | 打开 [`notebooks/colab_quickstart.ipynb`](https://colab.research.google.com/github/jiangqu74-blip/KD/blob/main/notebooks/colab_quickstart.ipynb)，从上到下依次运行 |
| **自己的电脑 / 实验室服务器 / GPU 租用平台（如 AutoDL）** | 看机器 | 正式实验、长时间训练、大数据集 | 按下面「快速开始」的命令 |

**按现在的规模，云端 CPU 就够了**：在云端会话里，`configs/default.yaml` 的完整流程约 20 多秒就能跑完。
等换成真实的大数据集，或者把模型加大，再用 GPU。

## 快速开始（Quick start）

```bash
git clone https://github.com/jiangqu74-blip/KD.git
cd KD
pip install -r requirements.txt

python scripts/run_all.py --config configs/quick.yaml     # 检查环境是否正常
python scripts/run_all.py --config configs/default.yaml   # 默认实验
python scripts/run_all.py --config configs/default.yaml --device cuda   # 有 NVIDIA GPU 时
```

- 需要 Python 3.10 或更新版本。Windows 上在 PowerShell 或 Anaconda Prompt 里运行同样的命令即可。
- 也可以用 conda：`conda env create -f environment.yml`，然后 `conda activate msrd4-hypergraph-rgvq-distill`。

## 做一个新实验（Run a new experiment）

1. 复制 `configs/default.yaml`，比如存成 `configs/bigger_teacher.yaml`，修改里面的参数和 `name`。
2. 运行：`python scripts/run_all.py --config configs/bigger_teacher.yaml`
3. 结果出现在 `results/<日期-时间>_bigger_teacher/`：报告 `REPORT.md`、测试准确率 `eval_results.json`、
   对比图、实际使用的参数 `config.yaml`、代码版本和耗时 `run_info.json`（说明见 [`results/README.md`](results/README.md)）。
4. 把新的结果文件夹提交（commit）到 GitHub，方便以后对比，也方便让 Claude 分析。

程序开始前会先检查配置文件：参数名写错（例如把 `epochs` 写成 `epoch`）会立刻报错，不会白跑一遍。

## 日常工作流程（Workflow with Claude）

```text
① 你在云端会话里说要做什么
② Claude 改代码、跑测试和小实验 → 推送（push）到一个 claude/… 分支（branch）
③ 你在 GitHub 上查看改动 → Create PR（创建合并请求）→ Merge（合并）进 main
④ 需要 GPU 时：在 Colab / 服务器上拉取（pull）最新的 main，跑实验
⑤ 把 results/ 里的新结果传回 GitHub → Claude 读取并分析
```

`main` 是主分支，只通过合并请求（Pull Request）更新。每次推送后 GitHub 会自动检查（CI：代码风格 + 测试 + 冒烟实验），
显示绿色 ✓ 才说明没问题。

## 常用命令（Makefile，Linux / macOS）

| 命令 | 作用 |
|---|---|
| `make setup` | 安装依赖 |
| `make test` | 运行测试 |
| `make lint` / `make format` | 检查 / 自动修正代码风格 |
| `make quick` | 冒烟测试 |
| `make run CONFIG=configs/default.yaml` | 跑一个完整实验 |

只想单独跑某一步时，可以直接用命令行入口，例如 `python -m src.main --help` 查看全部参数。

## 已知问题（Known issues）

- **合成数据太简单**：一个只看节点特征的线性分类器（不用超图、不用教师）就能在测试集上达到 100%，
  所以默认配置下教师和两个学生全是 100%，**无法比较 KD 和 KD+TopK**。需要先调整数据生成的难度。
- 其他需要注意的研究设定（学生只在训练节点上蒸馏、没有报告教师的测试准确率、只用一个随机种子等）
  见 [`docs/method.md`](docs/method.md) 第 5 节。
- `--use_infonce` / `--use_ib` 只是预留开关，尚未实现。
