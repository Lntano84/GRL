# 忠实复现：组件就绪，但作者训练预算在本环境不可达

**进度**：官方完整版的**全部五个组件**已能运行；唯一不可执行的库是 gensim，且已用**只替换嵌入训练器**的方式补上。
**阻塞点**：作者配置（10000 episode × 逐 step 重算 DeepWalk × 5 种子）在本机需要约 **146 小时**。训练预算必须由你定，我不自行缩减后再称其为「官方版」。

---

## 1. 作者代码的组装与探针结果

`scripts/official_assembly.py` 从已下载的官方源码重建 `_official/` 包树，并运行探针：

| 组件 | 论文/代码中的角色 | 探针结果 |
|---|---|---|
| `diffpool/encoders.py::GcnEncoderGraph` | **图表示（DiffPool）** | ✅ forward 通过（batched `(1,n,feat)`/`(1,n,n)`，输出 `(1,1)`） |
| `rl_alg/dqn.py::DQNTrainer` | **Q 网络与训练步**（`get_node_embeddings` / `get_values` / `gradient_update_sarsa`） | ✅ 构造通过 |
| `expts/net_env.py::NetworkEnv` | **调查环境** | ✅ 运行一步，前沿 10 |
| `expts/influence.py` + `icm.py` | **奖励（IC 影响力 + 贪心选种）** | ✅ 采样通过 |
| `ge/walker.py::RandomWalker` | **DeepWalk 的随机游走阶段**（p=q=1） | ✅ 120 节点 9,600 条游走 / 96,000 token，**0.04 s** |

**兼容性补丁共 14 处，全部是机械性的**（逐条记录在 `results/official_assembly.json`）：
`numba` 未安装→空装饰器；`np.int`、`nx.to_numpy_matrix`、`init.xavier_uniform`、`init.constant` 在新版库中改名/移除；**Python ≥3.11 移除了对 set 调用 `random.sample`**（作者在 `net_env.py` / `change_baseline.py` 中这样用）；matplotlib 绘图语句置空（无头环境）。
**没有任何一处改动算法行为。**

## 2. 唯一的库缺口，与替换范围

`gensim` **无法安装**：本机 pip 无外网（`pypi.tuna.tsinghua.edu.cn` 无任何可用版本，连 `six` 都取不到）。作者 `ge/models/deepwalk.py` 唯一不可执行的一行就是 `from gensim.models import Word2Vec`。

因此替换范围被压到最小：

| DeepWalk 的组成 | 本复现使用 |
|---|---|
| 随机游走生成（num_walks=80, walk_length=10, p=q=1） | **作者的 `ge/walker.py`**（未替换） |
| 词向量训练（`sg=1, hs=1, window=5, iter=50, size=60, min_count=0`） | **替换**为 `ccim/xplore/deepwalk_torch.py` 的 batched skip-gram（保留 sg/window/iter/size/min_count，**把 hierarchical softmax 换成负采样**） |

⚠ **这是算法行为差异，属复现偏差**，因此本轮结果**不得称为完全忠实复现**。

## 3. 成本实测与投影（这就是阻塞点）

在真实观测子图上实测（`interact__ia-crime-moreno`，5 个初始种子，观测子图沿 episode 从 16 增长到 30 个节点）：

| 项 | 实测 |
|---|---|
| 作者 `get_embeds`（iter=50）单次 | **1.55 s** |
| `train.py` 每 episode 调用 `get_embeds` 次数 | **6**（开局 1 次 + 每步 1 次，`s_embs = get_embeds(env.sub)`） |
| → 嵌入部分 / episode | **≈ 9.3 s** |
| 奖励 `parallel_influence(times=5)` / episode | ≈ 1.2 s（串行；作者用 `Manager`+`Process`，Windows 上更慢） |
| **合计** | **≈ 10.5 s / episode** |

**作者配置投影**：10,000 episode × 5 种子 = 50,000 episode → **≈ 146 小时**。

（对比：我的简化版用端到端 GCN 而不是逐步重算 DeepWalk，因此是 0.239 s/episode，二者相差约 44 倍——**这个差距几乎全部来自「每步重算 DeepWalk」**，不是来自 DiffPool 或 DQN。）

## 4. 需要你定的取舍

作者入口配置（`train.py`：`--eps 10000 --emb_iters 50`）不可达，必须放弃其中一项或两项。三条可行路径：

| 方案 | 偏差 | 5 种子预计耗时 |
|---|---|---|
| **A** 保 episode=10000，`emb_iters` 50→10 | 只改嵌入训练轮数 | **≈ 29 h** |
| **B** 保 `emb_iters`=50，episode 10000→2000 | 只改训练预算 | **≈ 29 h** |
| **C** episode 2000 且 `emb_iters` 10 | 两项都改 | **≈ 5.8 h** |
| （D）作者全配置，只跑 1 个种子 | 无偏差，但只有 1 种子 | ≈ 29 h |

无论选哪条，**训练预算与 `emb_iters` 都会在启动前写死并记入 `deviations_from_paper`**，不会看曲线临时延长。
