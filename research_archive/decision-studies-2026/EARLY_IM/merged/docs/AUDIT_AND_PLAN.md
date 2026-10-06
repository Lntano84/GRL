# 仓库审计 + 实施计划（阶段一交付）

> 日期：2026-09-17
> 范围：`github.com/Lntano84/GRL`，本地 `main` = `11d59a8`，工作区 clean，**81 tests passed**
> 结论摘要：**架构比预期完整，可以复用；但有一个必须先解决的建模阻塞项（见 §6）。**

---

## 0. 一句话结论

核心 pipeline（learned predictor → shortlist → audit → progressive verification → fallback）
**已经在 `src/grl/algorithms/sequential_im.py` 里实现了**，而且是通过统一的
`oracle.score(seeds, candidates, step)` 接口组织的——**换掉 oracle 就能把整套机制搬到
overexposure 设定上**。真正缺的不是算法，是三样东西：

1. **一个与 overexposure 一致的 MC oracle**（现有 oracle 是 live-edge/IC 专用，不适用）；
2. **一个论文级 experiment runner**（现在是 59 个脚本、13,481 行，没有统一入口）；
3. **退化性（degeneracy）的明确定义**——这是阻塞项，见 §6。

---

## 1. 代码规模与结构（实测）

| 区域 | 文件 | 行数 |
| --- | --- | --- |
| `src/grl/` | 40 | 4,346 |
| `scripts/` | 59 | 13,481 |
| `tests/` | 21 | 1,257 |

```
src/grl/
├── algorithms/sequential_im.py   192  ← 核心顺序决策框架（A/B/C 的答案）
├── oracle/marginal.py            129  ← MC oracle + learned oracle 包装
├── diffusion/
│   ├── independent_cascade.py    172
│   ├── overexposure.py           407  ← 阈值窗口模型（已实现）
│   └── __init__.py                51  ← 命名空间隔离（有回归测试）
├── baselines/  classic_im / degree / degree_discount
├── evaluation/ spread / sequential / ranking / gnn_metrics
├── models/     marginal_gain / marginal_gain_overlap / gnn
├── training/   marginal_dataset / marginal_trainer / gnn_trainer / comparison
├── diagnostics/ oracle / candidate_benchmark / retrieval_reranking
└── features/ experiments/ utils/
```

---

## 2. A. 当前已经存在什么

### 2.1 论文机制：**全部已在位**

`src/grl/algorithms/sequential_im.py` 提供四个策略，全部走同一接口：

| 函数 | 机制 | 对应论文组件 |
| --- | --- | --- |
| `full_oracle_greedy` | 每步对所有候选跑精确 oracle | ground-truth reference |
| `learned_greedy` | 只用 learned score | Learned-only control |
| `selective_greedy` | learned 排序取 Top-M → 精确验证 | shortlist + verification（固定 M） |
| **`adaptive_selective_greedy`** | 自适应 top-M → **residual envelope audit** → 扩展 → fallback | **audit + progressive verification + fallback** |

`adaptive_selective_greedy` 的 audit 逻辑是真实存在的，不是命名：

```python
residuals     = [verified[v] - learned[v] for v in verified]      # 实际残差
residual_max  = max(residuals)                                    # 经验上界
residual_std  = sqrt(var(residuals))
outsider_upper = learned[outsider] + residual_max + beta * residual_std   # 未验证者上界
stable   = winner 在最后 min_rounds 轮不变
certified = stable and verified[winner] >= outsider_upper
# 无法认证 → target += batch_m → 直到 cap；cap = 全部候选时为 fallback
```

**这正是"不盲信 learned prediction + 按风险增加 oracle 计算 + 必要时回退"的实现。**
文档里也明确标注了它的理论地位：*"an operational first certification baseline, not yet a
formal probabilistic guarantee"* —— 这个自我限定是诚实的，应当保留。

### 2.2 扩散模型

`overexposure.py`（407 行）实现了论文的阈值窗口模型：
窗口从 2D simplex 均匀采样；`δ(v,t) = Σ_{u∈A^in_t(v)} ω_uv`（A^in 为**曾经**正激活过的入邻居）；
状态机 inactive → positive → negative 单向；`P(positive|δ) = 2δ(1−δ)`（Lemma 1）；
`DETERMINISTIC`/`STOCHASTIC` 两种模式；`overexposure_free=True` 把 τ 钳到 1。

配套 `tests/test_overexposure.py`（357 行，24 个测试）+ `test_diffusion_namespace.py`（防命名冲突）。

### 2.3 Predictor 确实 condition on state

`LearnedMarginalOracle.score` 传入 `mask[seeds]=1`，即模型输入是
`(embeddings, norm_degrees, seed_mask, candidate)` —— **是条件边际增益，不是静态节点分**。
这一点符合论文要求，无需重做。

### 2.4 其它已就位

- `estimate_overexposure_spread_over_configs`：**配对** MC 估计器（同 trial 共享窗口）；
- `estimate_spread_over_configs`（IC）：同样配对，共享 live graph；
- Degree / Degree Discount / Random / Max_Degree / IMRank / PageRank / CELF / IGA / UB 九种 baseline；
- 图加载器支持 4 种边表格式 + 注释行 + header 歧义（有 133 行回归测试）。

---

## 3. B. 哪些可以复用（不需要重写）

| 组件 | 复用方式 |
| --- | --- |
| `adaptive_selective_greedy` 及另外 3 个策略 | **直接用**，只换 oracle |
| `LearnedMarginalOracle` | **直接用**（state-conditioned 已满足） |
| `overexposure.run_overexposure` / `sample_threshold_windows` | **直接用** |
| `estimate_overexposure_spread_over_configs` | **直接用**作 paired 估计 |
| 九种 baseline | **直接用** |
| `marginal_dataset` / `marginal_trainer` | 需改标签生成（现依赖 IC live-edge） |
| `graph_loader`、`evaluation/*`、`utils/config` | **直接用** |

---

## 4. C. 哪些缺失

| 缺口 | 现状 | 影响 |
| --- | --- | --- |
| **1. Overexposure MC oracle** | `BatchedMonteCarloMarginalOracle` 是 **live-edge/IC 专用**：采 live graph 再算 reachability（`oracle/marginal.py:48-92`）。**对 overexposure 不成立** | **阻塞级**：没有它就没有 ground truth，第四阶段无法开始 |
| **2. Oracle cost 计量** | `OracleStats` 有 `candidate_evaluations` / `mc_candidate_samples` / `live_edge_samples`，但 `sequential_im` 的 `steps` 只记 `verified=len(shortlist)`，**没有把 MC 模拟次数作为一等量记录** | 论文的核心指标（quality vs oracle cost）目前无法直接产出 |
| **3. 统一 experiment runner** | 59 个脚本、13,481 行，**没有** `scripts/experiments/run_overexposure_experiments.py` | 第十一阶段要求不满足 |
| **4. CSV / figure 管线** | 结果都是 JSON，**没有任何绘图代码** | 第九阶段的 4 图 3 表无法产出 |
| **5. Corruption harness** | `evaluate_robustness_stress.py` 存在，但**没有**把它接到新 oracle 上；没有统一的 corruption 注入接口 | 第七阶段无法开始 |
| **6. Degeneracy 定义** | `overexposure_free` 存在，但**实测不退化到 IC**（§6） | **阻塞级科学问题** |
| **7. Predictor 训练标签** | `marginal_dataset` 用 IC live-edge 生成标签 | 需改为 overexposure 配对标签 |
| **8. 多图覆盖** | 主实验只用 NetHEPT；我们已有 9 张图可用 | 第八阶段要求"不要假装有多个"——现在确实可以真的有多个 |

---

## 5. D. 当前实验是否可复现

**部分可复现，但有两个具体问题。**

### 5.1 可复现的部分

- 81 个测试全过，扩散与 oracle 有单测；
- MC 估计器接受显式 `random_seed`，同 seed 同图**确实可复现**（我实测过配对估计器逐 trial 值完全一致）；
- `test_reproducibility.py` 存在。

### 5.2 问题一：`scripts/` 里的实验脚本没有统一 seed 协议

`scripts/experiments/` 下 43 个脚本各自决定 seed 派生方式（我写的那些用 `base_seed + 13*size`
这类约定，早期脚本用别的）。**跨脚本结果不能直接拼进同一张表。** 第十阶段要求"统一 seed protocol"，
这需要 runner 层面强制执行。

### 5.3 问题二：README 记录的首轮结果是负面的，且未复跑

README 第 191 行记录 `Spearman = -0.377`、`Pairwise accuracy = 0.357`、`Top-1 recall = 0.000`。
README 自己说明是"2 epoch、极少样本"的链路验证。**这个数字不能进论文，但也绝对不能删**——
它记录了 predictor 在最弱配置下的表现。建议在 runner 里作为 smoke 输出保留。

### 5.4 我这次审计中的一处误报（记录在案）

我一度判定 `run_independent_cascade` 有"方向 bug"（种子 {3} 在链 0→1→2→3 上传播出 2 个节点）。
**该判定是错的**：`range(4)` 建的是 5 节点链，`{3}→{3,4}` 完全正确。
我用单边、链、反向边三组无歧义用例复核后确认 **IC 实现正确**。
根因：我把自己注释里的错误预期值当成了 ground truth，没有核对。

### 5.5 一处方法学不一致（真实，但影响有限）

`estimate_spread_over_configs`（IC）**正确共享 live graph**（配对），
而 `run_independent_cascade` 逐边消耗 RNG，因此直接对它与 OE 做逐 trial 配对**并不严格**。
我在 `scripts/audit/check_degeneracy.py` 里给 IC 用了 `random.Random(seed+t)`，
严格说配对不严 —— 该脚本中 IC 侧的数字只能当**量级参考**，不能作为精确对比。
正式实验必须统一走 `estimate_*_over_configs`。

---

## 6. ⚠️ 退化性：**v2 重大更正 —— 原先的"阻塞项"判定是错的**

### 6.1 PI 决策（2026-09-17）：**方案 A，并明确"不要回避比较"**

PI 的要求：*"A，我觉得可以，但是尽量比它好对吧"*，随后**确认论文定位**为：

> 在 RR 不适用的设定下，用 **learning + audit + verification + fallback**，
> 以**更少的 oracle 调用**逼近 oracle，并对 predictor 腐蚀鲁棒。
> **不声称在 spread 数值上打败 IC/IMM。**

IC/IMM 进入论文的身份是 **"not directly applicable"**（RR 集为空），不是被打败的对象。

### 6.2 我在 v1 里报的"10 倍差距"是**采样选择的产物**

v1 用**单个 `max(out_degree)` 节点**（out_degree = 210）做种子，IC 从它出发几乎不扩散，
于是得到 ic 43.3 / oe 366.7。**该数字来自一个刻意挑出的极端种子，不能推广。**

### 6.3 统一协议下的真实图景

`scripts/audit/find_comparable_regime.py`（全图均匀种子，5 组独立种子集，100 trials，
IC 与 OE 共享 trial index）：

| `\|S\|` | 覆盖率 | IC | OE | OE/IC |
| --- | --- | --- | --- | --- |
| 1 | 0.2% | 8.83 | 204.36 | 23.1 |
| 2 | 0.4% | 15.90 | 308.93 | 19.4 |
| 3 | 0.6% | 18.23 | 313.93 | 17.2 |
| 5 | 1.1% | 30.59 | 364.17 | 11.9 |
| 8 | 1.7% | 60.48 | 366.83 | 6.07 |
| 12 | 2.5% | 71.56 | 367.19 | 5.13 |
| 20 | 4.2% | 103.69 | 367.29 | 3.54 |
| 40 | 8.4% | 159.35 | 365.48 | 2.29 |
| 80 | 16.8% | 226.16 | 365.33 | 1.62 |

**读法**：OE 在 **5 个种子**时就激活 364/475 ≈ 77% 的图；IC 要到约 **80 个种子**才追上。
小预算区间两者相差 **12~23 倍**。

### 6.4 机制（已定位）

- **IC** 要求路径上**每条边逐个成功**（Congress 每条约 0.0034 概率）；
- **阈值窗口模型**只要求**入邻居权重和越过 κ**（κ~U[0,1]，稠密归一化图上 κ 常很小）。

所以 `oe_free`（τ=1）是**"带随机 κ 的线性阈值过程"**。仓库文档写
*"equivalent to the linear threshold model"* **是准确的**——是我 v1 里对"退化到 IC"的期待错了。

### 6.5 结论：这是一个**建模选择**，不是 bug，也不再是阻塞项

**降 κ_hi 不能提高可比性**（实测：κ_hi ≤ 0.5 时 OE 反而更高，~375）。
唯一接近持平的区间是 `|S| = 80`（16.8% 覆盖，比值 1.62），但那里两者都饱和，
方法差异被压缩，不能用来排序方法。

**因此**：论文不做跨模型 spread 比较。质量轴用**各模型自己的 relative gap to oracle**，
成本轴用 **MC cascade 次数**。这正是 PI 确认的定位。

### 6.6 处置结果

| 项 | 状态 |
| --- | --- |
| ~~退化性阻塞项~~ | **已解除**。退化的正确含义是"τ=1 时不可能出现负激活"（已加测试），**不是**"退化成 IC" |
| IC 基线 | 保留，身份为 "not directly applicable"，附 RR 集为空的实证 |
| 跨模型 spread 表 | **不做** |
| `scripts/audit/compare_ic_vs_oe.py` | **已废弃**：把两个不同协议的测量拼在一起，结论不可信。保留文件并标注 |

---

## 6bis. 退化性的正确契约（已写入测试）

`tests/test_overexposure_contract.py` 现在固定了三条可验证的退化性质：

1. `overexposure_free=True` 时 **`negative` 集合恒为空**（τ=1 ⇒ δ 不可能超过 τ）；
2. 该极限下 **`|S|` 增大 ⇒ 平均 spread 单调增**（移除过度暴露后恢复常规趋势）；
3. 负激活是**永久**的（单向状态机）。

**不**声称、也**不**测试"退化成 IC"——因为实测不成立，且那不是这个模型的语义。

---

## 7. 实施计划（阶段二 ~ 十一）

按依赖排序，每个阶段都标注"是否需要 PI 决策"。

### 阶段 2：Overexposure diffusion 加固（1 天，无需决策）

**现状：模型已实现，我只需补齐任务书要求的形式保证。**

- [ ] 新增 `tests/test_overexposure_contract.py`：
  - 空 seed set / `k=0` / isolated nodes；
  - 所有概率严格 ∈ [0,1]（对 κ,τ,δ 做网格扫描）；
  - 同 graph+seed+params 复现性；
  - 参数全部从 config 读取，无 hard-code。
- [ ] 在 `configs/` 增加 `overexposure:` 段（`activation_mode`、`overexposure_free`、
  `window_lo`、`seed`），并在 `run_overexposure` 上游接一个 config 解析层。
- [ ] **不重写** `overexposure.py`（已 407 行 + 24 测试，工作正常）。

### 阶段 3：State-Tracking MC Oracle（2 天，无需决策）

- [ ] 新增 `src/grl/oracle/overexposure_mc.py`：
  ```python
  class OverexposureMonteCarloOracle:
      def __init__(self, graph, mc_runs, random_seed, overexposure_free=False,
                   activation_mode=DETERMINISTIC, window_lo=None)
      def score(self, seeds, candidates, step=0) -> dict[int, float]   # Δ(v|S)
      def spread(self, seeds) -> dict[str, float]
      # stats: mc_cascades 作为一等计量
  ```
- [ ] **复用** `estimate_overexposure_spread_over_configs` 实现配对 `Δ(v|S)`；
- [ ] **新增 `OracleStats.mc_cascades`**（现有 stats 没有这个一等量，而这正是论文核心指标）；
- [ ] `MCGreedyOracle` = `full_oracle_greedy(pool, k, OverexposureMonteCarloOracle)`；
- [ ] `tests/test_overexposure_oracle.py`：与暴力枚举 MC 对比、成本计数正确、seed 复现。

### 阶段 4：问题成立性实验（2 天，**依赖 §6 的决策**）

复用现有 `scripts/audit/check_degeneracy.py`（已写好），扩充为：

- [ ] 4 张图 × 3 个 budget × {IC, OE, OE-free, degree, degree discount, MC-greedy};
- [ ] 输出：degree vs 真实边际的 ρ、node ranking 是否被 overexposure 改变、
  marginal 是否依赖当前 S、IC 与 OE 的优化行为差异；
- [ ] **所有 baseline 同图、同 k、同扩散参数、同评测 MC 预算**（写进 runner 强制）。

### 阶段 5：接入现有 GRL（3 天，无需决策）

- [ ] 新增 `scripts/experiments/run_overexposure_experiments.py`（阶段 11 的 runner）：
  组装 `OverexposureMonteCarloOracle` + `LearnedMarginalOracle` → 调
  `adaptive_selective_greedy` 等四个策略；
- [ ] 逐项验证任务书列的 6 个检查点（predictor 是否 condition on S、是否 marginal、
  audit 是否真检查可靠性、verification 是否真省 oracle、fallback 是否真存在、
  corrupted predictor 是否触发 fallback）；
- [ ] 训练标签改用 overexposure 配对标签（改 `marginal_dataset`，加 `--diffusion overexposure`）。

### 阶段 6：Baseline / ablation 矩阵（2 天）

A–J 十项中，**A/B/C/D 已有**，E–J 由阶段 5 的四个策略组合覆盖。
需要新增的只有 corruption 注入（阶段 7）。**明确标注不适用项**：
`RR/RIS/IMM/OPIM` 在 overexposure 下 `not directly applicable`
（Gate 1d 已证明 RR 集为空），**不伪造比较**。

### 阶段 7：Robustness（2 天）

- [ ] 统一 corruption 接口：`CorruptedOracle(base, mode, strength)`
  mode ∈ {clean, noise_gaussian, shuffle_top, random, adversarial_sign_flip}；
- [ ] 报告 5 个量与不变量：influence quality、oracle MC calls、runtime、
  fallback frequency、verification frequency；
- [ ] **不报 predictor accuracy 作为主指标**。

### 阶段 8–10：正式实验矩阵 + 协议统一（3 天）

- [ ] runner 强制：同图/k/扩散参数/评测 MC 预算/seed 协议；
- [ ] config 驱动，`--seeds 5`、`--budgets 5 10 20`、`--graphs ...`；
- [ ] **最终 quality 一律用独立的 evaluation oracle 评**（不能复用 predictor 的偏差）。

### 阶段 11：Runner + CSV + 图（3 天）

- [ ] `run_overexposure_experiments.py`：config → checkpoint/resume → CSV → summary table → figures；
- [ ] 只产出任务书指定的 4 图 3 表，每张图存 CSV + PNG + 参数 JSON + git commit hash。

---

## 8. 我的评估：三个必须先说的问题

### 8.1 最大的科学风险

**不是技术风险，是"IC 基线不可比"（§6）。** 论文主线是 quality/cost tradeoff，
需要一个 cost 轴和一个 quality 轴。quality 轴如果只能在同一模型内部比较，
那么"relative gap to MC oracle"就是唯一可用的质量指标——这**可行**，
但意味着**论文不能再声称"我们的方法比 classical IM 好"**，
只能声称"在 RR 不适用的设定下，我们用学习+验证+回退逼近 oracle"。

**我认为这个定位是可发表的，而且是诚实的**，但需要 PI 确认这是否是想要的 claim。

### 8.2 第二个风险：predictor 的历史表现是负的

README 记录 `Spearman = −0.377`。如果 predictor 在 overexposure 下也不比 degree 好，
那么 audit/verification/fallback 的价值主张会变成"我们证明了学习没用"——
这与我自己在 Gate 3 得到的结果一致（学习型排序器打不过解析基线）。

**但注意**：Gate 3 的结论是"**打不过解析基线**"，不是"打不过 degree"。
论文的卖点是 **cost 轴**（省 oracle 调用），而不是 accuracy 轴——
所以即使 predictor 只是"中等准确"，只要 audit 能识别它的不可靠并回退，
**cost/quality tradeoff 的主张仍然成立**。这是我建议的定位。

### 8.3 第三：仓库里有 13,481 行脚本，需要治理

59 个脚本里至少两条线：主线（static IC + audit，~14 个）和我这两天的 overexposure 线（~13 个），
其余是历史探索。**不建议删除**（任务书禁止删失败实验），但建议：

- 新增 `scripts/experiments/README.md` 标注每个脚本的状态（main / superseded / historical）；
- runner 只调用明确标注为 main 的模块。

---

## 9. 需要 PI 决策的三个问题

1. **§6 的退化性**:选 A（放弃跨模型比较）／B（改比 relative gap）／C（重新定义退化轴）？
   我推荐 **A**，理由：论证最干净，且与"RR 不适用"这一最强立项理由一致。
2. **§8.1 的 claim 定位**:是否接受"论文不声称优于 classical IM，只声称在 RR 失效设定下
   用学习+验证+回退逼近 oracle"？
3. **阶段顺序**:是否同意先做阶段 2–3（加固 + oracle，共 3 天，不依赖任何决策），
   同时等 §6 的决策再进阶段 4？我推荐这样并行。

---

## 10. 本轮实际改动的文件

### 审计脚本（阶段一）

| 文件 | 性质 | 说明 |
| --- | --- | --- |
| `scripts/audit/check_degeneracy.py` | 新增 | 四配置退化实测（**IC 侧配对不严，仅作量级参考**） |
| `scripts/audit/diagnose_spread_scale.py` | 新增 | 定位量级差异机制 |
| `scripts/audit/calibrate_oe_to_ic.py` | 新增 | κ_hi 扫描（结论：降 κ_hi 不能提高可比性） |
| `scripts/audit/find_comparable_regime.py` | 新增 | **统一协议的种子规模扫描（本文档 §6.3 的数据来源）** |
| `scripts/audit/compare_ic_vs_oe.py` | **已废弃** | 拼接了两个不同协议的测量，结论不可信；保留并标注 |

### 阶段二：扩散契约（已完成）

| 文件 | 性质 | 说明 |
| --- | --- | --- |
| `src/grl/diffusion/params.py` | **新增** | config 驱动的参数解析 + 校验（`OverexposureParams`） |
| `src/grl/diffusion/overexposure.py` | **修改** | `sample_threshold_windows` 增加 `window_lo` 参数（显式实验旋钮） |
| `tests/test_overexposure_contract.py` | **新增** | 41 个契约测试：边界情况、概率界、复现性、config 校验、退化契约 |

### 阶段三：Overexposure MC Oracle（已完成）

| 文件 | 性质 | 说明 |
| --- | --- | --- |
| `src/grl/oracle/overexposure_mc.py` | **新增** | 状态追踪 MC oracle；`OracleStats` **新增 `mc_cascades` 一等计量** |
| `src/grl/oracle/__init__.py` | **修改** | 导出新 oracle |
| `tests/test_overexposure_oracle.py` | **新增** | 17 个测试：配对性、成本计数、与手写估计器一致、负边际可复现 |

**测试总数：81 → 139 passed。**

### 仍未做

阶段 4（问题成立性，依赖已解除，可以开始）、阶段 5–11。

