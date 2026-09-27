# 方法说明 | MSRD4 Hypergraph RGVQ Distillation

> 本文件只讲**方法**（method）。怎么运行实验、目录结构，请看仓库根目录的 `README.md`。
> 下面的描述是按照 `src/` 里的代码实际行为整理的；如果和你的研究设想不一致，以你的设想为准，再改代码。

## 1. 任务是什么

- 数据是一张**超图**（hypergraph）：普通图的一条边只连 2 个节点，超图的一条**超边**（hyperedge）可以连任意多个节点。
  用关联矩阵（incidence matrix）`H ∈ {0,1}^{|V|×|E|}` 表示：`H[v, e] = 1` 表示节点 v 属于超边 e。
- 任务是**节点分类**（node classification）：给每个节点预测类别。
- 思路是**知识蒸馏**（knowledge distillation, KD）：
  - **教师**（teacher）`TeacherHGNN`：能看到超图结构 `H`，效果好但推理要用到整张图。
  - **学生**（student）`StudentMLP`：**0-hop**，只看节点自己的特征 `X`，推理时完全不用 `H`，部署快。
  - 目标：让学生通过模仿教师，学到超图结构里的信息。

## 2. 教师模型 TeacherHGNN（`src/models.py`）

输入特征先线性投影，然后重复 `L` 个 stage，每个 stage 依次做：

1. **N2E**（node → edge）：把超边里所有节点的表示取平均，得到超边表示（与节点顺序无关，permutation-invariant）。
2. **Tap-E（RGVQ-lite）**：对超边表示做多层残差软量化（residual soft vector quantisation，`src/rgvq.py`），
   得到每条超边的**角色分布**（role distribution，即在码本 codebook 上的概率）。
3. **E2N**（edge → node）：把节点所在超边的表示取平均，传回节点。
4. **Tap-N（RGVQ-lite）**：对节点表示同样做量化，得到**节点参与角色**（node participation roles）。

最后接分类层输出 logits。全程直接在超图关联矩阵上计算，不把超图转成普通图（no 2-section conversion）。

**RGVQ-lite**：每一层（level）用可学习码本计算相似度 logits → 带温度（temperature）的 softmax 得到角色概率 →
用概率加权码本重建残差 → 残差减去重建值后进入下一层。温度被限制在 `[0.1, 5.0]` 之间。

## 3. 训练流程（`scripts/run_all.py` 按顺序调用）

| 步骤 | 代码 | 做什么 |
|---|---|---|
| 1. 训练教师 | `src/train_teacher.py` | 生成合成超图并缓存；用训练集节点的交叉熵（cross-entropy）训练，按验证集准确率保存最好的模型 |
| 2. 导出蒸馏包 | `src/main.py --mode export_distill`、`src/distill.py` | 教师对全部节点前向一次，导出：类别软标签（soft labels）、每个节点的 Top-K 节点角色、以及把超边角色平均到节点上的 Top-K "edge@node" 角色 |
| 3. 训练学生（KD） | `src/train_student.py`，`--ablation kd` | 损失 = 学生与教师软标签之间的 KL 散度（KL divergence） |
| 4. 训练学生（KD+TopK） | 同上，`--ablation kd_topk` | 在 KD 损失上再加 `topk_weight ×` Top-K 掩码 KL（masked KL）：让学生的角色预测头去拟合教师的 Top-K 角色 |
| 5. 评估 | `src/eval.py` | 在测试集节点上计算两个学生的准确率 |
| 6. 报告 | `scripts/update_report.py` | 生成 `REPORT.md` 和对比图 |

蒸馏包只存每个节点的 Top-K 切片，大小是 `O(|V|)`，不需要存整张图。

## 4. 合成数据（`src/data.py`）

- 节点特征：每个类别一个高斯簇（Gaussian cluster）。
- 超边大小：截断的帕累托分布（truncated Pareto），即长尾（long-tailed）——大部分超边小，少数很大。
- 超边成员：以概率 `p_mixed` 随机混合各类别；否则偏向同一类别（同类节点的抽样权重合计 85%）。
- 划分：60% 训练 / 20% 验证 / 20% 测试。
- 全部由 `seed` 控制，同一个 seed 生成的数据完全相同。

## 5. 目前需要注意的地方

- **合成数据目前太简单，还不能用来比较方法。** 只看节点特征的线性分类器（logistic regression，不用超图、不用教师）
  在测试集上就能达到 100%（seed 1、2、3 都是）：类中心之间最近的距离约 18–23，而类内平均散布只有约 4–5。
  所以用 `configs/default.yaml` 跑出来教师、两个学生全是 100%，**KD 和 KD+TopK 看不出差别**。
  需要一个控制难度的参数（例如缩小类中心距离、加大类内噪声），让"只看特征"明显做不到 100%，超图结构才有用武之地。
- 学生训练只在**训练集节点**上使用教师软标签，没有使用真实标签的交叉熵，也没有用到验证/测试节点上的教师软标签。
  很多图蒸馏工作（例如 GLNN）会在所有节点上用教师软标签，并加上真实标签的交叉熵。是否要改，取决于你的实验设定。
- 报告里没有教师的**测试集**准确率，只有验证集准确率；做蒸馏对比时，通常需要把教师的测试准确率作为参照。
- `--use_infonce` 和 `--use_ib`（InfoNCE / 信息瓶颈 information bottleneck）只是预留的开关，打开会直接报错（尚未实现）。
- 单次运行只用一个随机种子；正式结论需要多个种子（例如 5 个）取平均值和标准差。
