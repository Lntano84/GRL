# 官方架构低预算改编版：状态与两处修正

> **统一名称**：本项目自此称 **「官方架构低预算改编版」**（adapted version）。两项主比较相应称为 **「改编版 − random」** 与 **「改编版 − degree_min」**。不得简称为「官方版」或「忠实复现」。

**状态**：作者的五个组件全部组装并验证可运行（14 处机械性兼容补丁）；训练已启动，预计 **7.2 小时**（5 种子 × 2000 episode × 2.58 s；训练循环本身，检查点评估与最终开发评估另计）。检查点按种子落盘，中断也不丢。

---

## 1. 作者代码：5/5 组件可运行

`scripts/official_assembly.py` 从已下载源码重建 `_official/` 包树并运行探针（结果见 `results/official_assembly.json`）：

| 组件 | 角色 | 探针 |
|---|---|---|
| `diffpool/encoders.py::GcnEncoderGraph` | 图表示（DiffPool） | ✅ batched `(1,n,feat)`/`(1,n,n)` forward 通过 |
| `rl_alg/dqn.py::DQNTrainer` | Q 网络与训练步 | ✅ 构造 + `get_values2` / `gradient_update_sarsa` / `td_compute` |
| `expts/net_env.py::NetworkEnv` | 调查环境 | ✅ 运行 |
| `expts/influence.py` + `icm.py` | 奖励 | ✅ 采样通过 |
| `ge/walker.py::RandomWalker` | DeepWalk 的**游走阶段** | ✅ 96,000 token / 0.04 s |

**14 处兼容补丁全部是机械性的**（`numba` 缺失→空装饰器；`np.int`、`nx.to_numpy_matrix`、`init.xavier_uniform`、`init.constant` 改名；**Python ≥3.11 移除了对 set 调用 `random.sample`**；绘图置空）。**无一处改动算法行为。**

## 2. 唯一的库缺口被压缩到最小

`gensim` 不可安装（pip 无外网）。作者 `ge/models/deepwalk.py` 只有一行依赖它。因此：

- **游走用作者的 `ge/walker.py`**（未替换）；
- **只替换词向量训练器**为 `ccim/xplore/deepwalk_torch.py` 的 batched skip-gram——保留 `sg / window / iter / size / min_count`，**把 `hs=1` 换成负采样**。⚠ 这是**算法差异**，属复现偏差。

## 3. 两处必须记录的修正

### 修正一（对上一轮报告）：官方训练奖励**是**被归一化的

我此前写「代码里 `opt_reward=0`/`norm_reward=0`，论文式 (1) 的归一化未启用」。这只对 `env.reward` 成立。`train.py` 第 442 行在终止步做 `r1 = r1 / opt`，其中 `opt = influence(g, g)[0]`（**全知贪心**的影响力）。所以**有效训练信号确实被归一化，归一化基准是 OPT，不是 CHANGE**。已按代码保留。

### 修正二：真正的瓶颈不是 DeepWalk，是作者的 `influence` 实现

| 项 | 实测 |
|---|---|
| 作者 `influence(discovered, full)` 单次 | **47.2 s** |
| 作者 `times_mean_env=5` → 每 episode 奖励 | **≈ 236 s** |
| 于是批准的 2000×5 预算 | **≈ 655 h** |

我此前估的「奖励 ≈1.2 s/episode」是按**我自己**的向量化估计器算的；作者的实现慢约 40 倍（逐样本 NetworkX 连通分量 + 对 `range(n)` 的多线性贪心）。

**处置**：奖励改用 **同目标的替代实现**（`influence_equivalent()`）——**同一个目标量**（已发现图上 100 条 live-edge 样本贪心选 k=10，再在完整图上用 100 条新样本评估），只是实现换成 `ccim.xplore.icm.LiveEdgeObjective`。

> ⚠ **函数名不是等价证明。** 已有的 1.85% 交叉核验（4000 样本）支持的是**在固定种子集上的传播估计兼容性**；它**不能**证明「100 样本上贪心选种子，再独立评估」这一**整个随机奖励流程**一致——有限样本误差可能改变贪心选中的种子。本轮不追加强验证，证据与此限制一起报告。

同时发现作者 `parallel_influence` 的两处算术问题：`times=1` 时索引算术**一个 worker 都不启动**并对方括号空列表取均值；`times=5` 时**只跑 4 个进程**。

**重复次数：本轮实际执行 5 次，与作者代码实际的 4 次不同。** 若每次是独立同分布重复，4 次与 5 次均值的**期望相同，但方差与成本不同**，因此这是**明确的行为修正，不只是兼容补丁**。实际重复次数记入产物（`reward_repeats_executed`）。

**`r / opt` 的读法**：这是**相对全知贪心参照**的归一化；`opt` 是 `influence(g, g)` 给出的贪心解影响力，**不是已证明的全局最优值**。

## 4. 偏差清单（启动前落盘于 `results/xplore_official_plan.json`）

1. 嵌入训练器：gensim `Word2Vec(hs=1)` → skip-gram 负采样（**算法差异**）
2. episode 预算：2000 vs 作者 10000
3. 嵌入轮数：10 vs 作者 50（2、3 由你选定，启动前写死）
4. **奖励重复次数：实际 5 次 vs 作者代码实际的 4 次**（**行为修正**，不是兼容补丁）
5. 奖励重复的并行方式：串行，不用 `multiprocessing.Manager`
6. **奖励实现**：`LiveEdgeObjective` 替代作者的 `influence`（**同目标的替代实现**，非等价证明）
7. 绘图 / tensorboard 移除

**因此本轮的正式名称是「官方架构低预算改编版」，不得称为「官方版」或「完全忠实复现」。**

## 5. 代码里保留、论文未写的两处事实

- 训练奖励除以 **OPT**（非论文式 (1) 的 CHANGE）；
- 默认 `--const_features 1` 时，喂给图编码器的是**全 1 的节点特征矩阵**（`make_const_attrs`），DeepWalk 嵌入只从**动作表示**进入。

## 6. 交付（训练完成后）

**表一**：各训练种子的 epoch 0 / 选中 / 最终检查点，在训练图与验证图上的传播收益 + 训练成本。
**表二**：开发评估（`interact__soc-wiki-Vote`，20 组固定观测）——五个模型逐个及平均的收益、相对 **random** 与 **degree_min** 的配对差与区间、调查次数、推理时间。
主比较已预先固定为 `官方版 − random` 与 `官方版 − degree_min`；`degree_max` 与简化版仅作辅助。

**不做**：不把官方版优于简化版归因于任何单一组件（结构、训练量、奖励采样同时变化）。
