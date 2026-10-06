# 探索式影响力最大化：协议侦察与环境落地

**目的**：在写任何策略代码之前，先把要复现的协议从**原始论文 + 官方代码**里钉死，避免照着二手描述自己发明一个环境。
**状态**：协议已钉死；环境已实现并通过 17 条闸门；冒烟测试已暴露两个必须写进报告的结构性事实。尚未训练任何策略。

---

## 1. 目标工作（已定位到一手来源）

| 工作 | 身份 | 对我们的角色 |
|---|---|---|
| **Geometric-DQN** | Kamarthi, Vijayan, Wilder, Ravindran, Tambe. *Influence maximization in unknown social networks: Learning Policies for Effective Graph Sampling.* AAMAS 2020. [arXiv:1907.11625](https://ar5iv.labs.arxiv.org/html/1907.11625) · [AAMAS PDF](https://www.ifaamas.org/Proceedings/aamas2020/pdfs/p575.pdf) | 最直接的框架级对手 |
| **官方代码** | [`kage08/graph_sample_rl`](https://github.com/kage08/graph_sample_rl)（29 stars，最后推送 2022-07）。含 `data/`、`icm.py`、`expts/net_env.py`、`expts/influence.py`、`expts/change_baseline.py`、`train.py`、`rl_alg/dqn.py`、`diffpool/`、`ge/` | 可复现的环境与超参来源 |
| **CHANGE** | Wilder et al. 2018，官方仓库内 `expts/change_baseline.py` | 论文自己保留的唯一基线（他们测过 SNOWBALL / RECOMMEND / RANDOM-GREEDY，分别差 42.1% / 41.7% / 3.9%，因此只留 CHANGE） |
| **CLAIM** | UAI 2021，课程学习 + hindsight replay 改善同一任务的 RL 样本效率。[PMLR PDF](http://proceedings.mlr.press/v161/li21b/li21b.pdf) · 声称的[复现仓库](https://github.com/HughLee1994/CLAIM-Curriculum-Learning-Policy-for-Influence-Maximization-in-Unknown-Social-Networks) | ⚠ **该仓库只有一个 README，没有任何代码**；已核实的只有论文本身。**不得写成「已有可直接运行的基线」**。要对照 CLAIM 须自行实现课程学习 + hindsight replay，属额外工作量，需单独评估 |
| **IM-META** | 用节点元数据推断连接再决定调查谁。[arXiv:2106.02926](https://ar5iv.labs.arxiv.org/html/2106.02926) · [IEEE](https://ieeexplore.ieee.org/abstract/document/10423221) | 一旦用属性信息就必须对齐信息权限 |

**因此**：「首次把 GNN＋RL 用于未知网络 IM」不成立，Geometric-DQN 已经做了。

---

## 2. 从官方代码读出的精确协议

来源：`train.py` 的 argparse 默认值 + `expts/net_env.py` + `expts/influence.py` + `icm.py` + `utils.greedy`。

| 项 | 值 | 出处 |
|---|---|---|
| 免费初始种子 | **5 个随机节点**，其邻域直接揭示 | `--extra-seeds 5`；`NetworkEnv.__init__` 对每个 seed 调 `enlarge_graph` |
| 调查预算 | **T = 5** | `--sample-budget 5`；`for stps in range(budget)` |
| 可调查对象 | 已发现但未调查的**前沿**节点；重复调查已调查节点是错误 | `net_env.step` 的 `possible_actions` / `active` 分支 |
| 一次调查的收益 | 揭示该节点在**真图**中的所有邻居 | `enlarge_graph` |
| 扩散模型 | 标准 IC，**p = 0.1** | `--prop-prob 0.1` |
| 影响估计 | **100 条 live-edge 样本** | `--samples 100`，`expts/influence.SAMPLES` |
| 种子选择 | 在**已发现图**上跑**懒贪心 (CELF)**，k = **10** | `utils.greedy`；`--infl-budget 10`；`expts/influence.BUDGET` |
| 收益 | 把该种子集拿到**完整图**上评估的 IC 影响力（**含种子自身**） | `influence()` 先 `greedy(discovered)` 再 `f_set1(full_graph)(S)` |
| 目标函数语义 | 节点被影响 ⟺ 与某个种子处于同一 live-edge 连通分量；分量权重 = 分量内目标节点数 | `icm.live_edge_to_adjlist` 的 `ws[i] = |target ∩ cc[i]|` |
| CHANGE 的预算 | `Change(g, budget = budget*2)` → 5 个全局随机节点 + 各 1 个随机邻居 = 最多 10 个揭示节点 | `train.py` 第 255 行 + `change_baseline.py` |
| 训练规模 | 10000 episodes，batch 100，lr 1e-4，γ=0.99，ε 0.1 衰减 0.999 | `train.py` |
| 表示 | DeepWalk（walk_len 10、num_walks 80、win 5、iter 50、dim 60）+ DiffPool 图池化 | `--actiondim 60`、`diffpool/`、`ge/` |

**两处代码与论文不一致，以代码为准并记录**：
1. 论文式 (1) 用 `(I − CHANGE)/(OPT − CHANGE)` 归一化奖励；代码里 `opt_reward=0`、`norm_reward=0`，所以训练信号是**原始影响力**，不是式 (1)。
2. `greedy(list(range(len(graph))), ...)` 的候选集是**全部 n 个节点**（因为 `NetworkEnv` 把整图节点都 `add_nodes_from` 进已发现图）。未发现节点在 live-edge 样本里是孤立点、边际恰为 1，所以实际仍从已发现节点里选；本复现**故意保留**这一行为，因为「修正」它会改变基线数值。

---

## 3. 已下载并核验的数据

官方 `data/` 下 22 张图，全部可用 networkx 3.6.1 直接 `pickle.load`，且**节点标签本来就是 0..n−1**（官方代码把 `rg.choice(len(g))` 的结果直接当节点用，因此这一点必须成立）：

| 家族 | 图 | n 范围 |
|---|---|---|
| `rt/`（转发网络，星形为主） | damascus, israel, obama, occupy, tlot, assad, copen, voteonedirection | 761 – 3698 |
| `mammal/` | bhp, kcs, plj, rob | 1218 – 1686 |
| `netscience/` | netscience | 1589 |
| `interact/` | ia-crime-moreno, ia-enron-only, ia-infect-dublin, ia-infect-hyper, rt-twitter-copen, soc-wiki-Vote | 113 – 889 |
| `rand_data/` | rand_100, rand_200, rand_500 | 91 – 457 |

按图划分训练/测试在数据上完全可行（同家族多张图）。

---

## 4. 已实现的环境与闸门

`ccim/xplore/icm.py` —— IC 的 live-edge 估计与懒贪心选择器；`ccim/xplore/env.py` —— 发现环境与已发表策略。
与官方实现的差异只有一处，且已说明：live edge 用 `Binomial(m, p)` 抽数量再均匀选边，与逐边 `random.random() < p` **分布相同**但快得多；有单测比对分量数分布。

`tests/test_xplore.py`，**17 条全过**（全套 95 条通过）：

- **估计器正确性**：live-edge 估计与**显式 IC 模拟**在 p=0.1/0.3、多组种子下一致（相对误差 < 6%）；p=0 时影响力恰等于种子数；空集影响力为 0；单调性；分量数分布与逐边伯努利抽样一致。
- **选择器正确性**：贪心返回值等于其自身目标值；落在 (1−1/e) 近似保证内；不劣于任何单节点。
- **环境正确性**：一次调查**恰好**揭示真邻居、不多不少；初始种子**不消耗**预算；只能调查前沿，重复调查被拒绝；预算被严格遵守。
- 收益在**完整图**上度量（已发现子图不可能支撑更大的影响力）。

---

## 5. 两个结构性事实（已按 20 组配对基线收紧）

> **本节结论已由 `CCIM_XPLORE_基线曲线.md` 的 20 组配对结果修正。** 原先的措辞比证据强，下面是收紧后的版本。

**事实 A：CHANGE 是「权限对照」，不是公平策略对照。**
官方 `Change` 用 `random.sample(range(len(graph)))` 在**全部 n 个节点**上均匀调查；前沿策略只能调查已经听说过的节点。冒烟测试里 `damascus` 5 次调查后：前沿策略发现 53 个节点，CHANGE 发现 781 个。这个差距**同时混合了访问权限与调查策略**，本复现既不把它全部归因于权限，也不写成 RL 的劣势。要分解需要额外的受控实验（给 CHANGE 同样只走前沿，或给前沿策略全局权限），本轮不做。

**事实 B（收紧）：能支持的是「在这个初始观测下可见度数无法区分前沿候选」，不是「整个调查过程没有学习空间」。**

`damascus` 的一个初始观测里，5 个种子真实度数是 `[41, 1, 1, 2, 1]`——一个 hub 加四片叶子；hub 作为种子已被揭示，前沿恰是它的邻居，观测度数全为 1，于是三种策略在该步完全重合。但必须区分两种情况：

- **度数相同、观测邻域不同**：连到哪个已调查节点、邻居之间是否相连等仍然携带信号。这需要除度数以外的合法特征，不能据此说没有学习空间。
- **可见结构与合法特征都对称**（同一个 hub 下、无其他可见差异的叶子）：仅依赖这些输入的**置换等变** GNN 确实无法区分它们。此时也**不能靠节点编号打破对称再声称学到了可迁移结构**。

后续调查会打破对称（queried 一个叶子会揭示它的其他邻居）。因此「第一步可见度数退化」**不等于**「整个过程无学习空间」。20 组配对结果也确认：度数的可区分性依图而变——`interact__soc-wiki-Vote` 上 degree_max 的并列比例只有 0.066、前沿不同度数值有 4.62 个，度数在那里**是有区分度的**，而它并没有因此战胜随机。

同理，`mammal/` 度分布更平缓**不自动意味着更适合学习**；选它是为了覆盖不同图结构，不是预先认定它会赢。

**事实 C：全局播种 ≠ 全局调查，必须分开写。**
官方代码把整图节点都加进已发现图，`greedy(range(len(graph)))` 因此在全部 n 个节点上选种子。协议的正确表述是：**全部节点身份已知；未调查节点的关系未知；最终种子候选允许覆盖全部节点。** 实测 T=5 时被选中的 10 个种子里平均有 **1.40 个当时尚未被发现**（`damascus` 上 3.5–3.7 个）。本复现所有方法统一采用该协议，不一边写「忠实复现」一边悄悄改成只在已发现节点里播种。

---

## 6. 下一步（本轮已完成第 1–2 项，见 `CCIM_XPLORE_基线曲线.md`）

1. ~~跑出基线的「调查次数—完整图影响力」曲线~~ **已完成**：3 张开发图 × 20 组配对 × 9 个预算。
2. ~~评估口径~~ **已完成**：选择用 100 样本（独立流、跨方法配对），最终评估用 1000 样本（跨方法跨预算共享），并报告评估标准误。
3. 下一步才实现学习型调查策略（Geometric-DQN 的核心思路），在**本轮选定的开发设置**上做直接对比；训练/推理时间单列。
4. 报告必须区分：**影响力提升**（真目标）与**发现节点数**（诊断量）——`interact` 上 degree_max 在 T=20 发现了 276 个节点而 random 只有 197 个，影响力却略低（171.59 vs 173.68），两者明显不是一回事。
