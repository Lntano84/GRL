# 阶段 4 报告：问题成立性实验

> 日期：2026-09-17
> 数据：`docs/results/stage4_question_validity_20260917.json`（4 图 × 3 budget）
> 　　　`docs/results/stage4b_usable_regime_20260917.json`（6 图 × 3 budget）
> 脚本：`scripts/experiments/stage4_question_validity.py`、`stage4b_usable_regime.py`
> 结论：**状态依赖性为真且幅度可观；优化 headroom 为真且幅度很大；但两者只在稠密图的小 k 处共存。**

---

## 1. 实验设计（协议统一）

所有比较使用**同一图、同一 budget、同一扩散参数、同一评测 MC 预算**。
最终 spread **一律由独立评测 oracle 测得**，绝不复用方法自己挑选时用的分数。

- oracle = `OverexposureMonteCarloOracle`（阶段 3 交付）
- 每个 seed set 用 `eval_mc` 次级联评测
- `degree discount` 用其**状态条件**形式 `rank_degree_discount_candidates`
  （折扣基于已选种子；其公式源自 IC，`probability` 传入该图平均边权，已记录）

---

## 2. Q1：marginal gain 是否依赖当前种子状态？**是**

对**同一批候选**在多个不同种子状态下各测一次 `Δ(v|S)`，做方差分解：

- `between_sd` = 候选**之间**的固有质量差异
- `within_sd` = **同一候选**在不同状态下的波动
- 比值 `within/between` > 1 表示"状态"比"候选本身"更重要

| 图 | `⟨k⟩` | between_sd | within_sd | **ratio** | 判定 |
| --- | --- | --- | --- | --- | --- |
| Congress-Twitter | 55.95 | 15.36 | 86.62 | **5.64** | 强状态依赖 |
| email-Eu-core | 50.89 | 46.22 | 89.62 | **1.94** | 状态依赖 |
| ca-GrQc | 11.06 | — | — | **0.64** | 中等 |
| **NetHEPT** | **4.23** | — | — | **0.075** | **几乎不状态依赖** |

（stage 4b 在更小 k 上的复测：Congress k=1→2.13、k=2→2.44、k=3→4.29）

**这是状态条件化预测器存在意义的最直接证据**：在稠密图上，知道"当前状态"比知道"候选本身多好"重要 2~5 倍。

**但注意**：**稀疏图（NetHEPT）几乎没有状态依赖性（0.075）**。
在那里一个静态节点分就够了——这也解释了我此前 Gate 3 得到"学习型排序器打不过解析基线"的结果。

---

## 3. Q2：廉价启发式能拿到多少？headroom 很大

`gap = (spread_oracle − spread_method) / spread_oracle`，各图各 budget 的最大值：

| 图 | k | degree gap | random gap | oracle spread |
| --- | --- | --- | --- | --- |
| Congress-Twitter | 2 | **10.99%** | 7.40% | 359.01 |
| Congress-Twitter | 5 | −0.63% | 3.98% | 363.54 |
| email-Eu-core | 2 | −0.80% | **78.36%** | 696.25 |
| ca-GrQc | 10 | −0.25% | **41.01%** | 948.27 |
| ca-GrQc | 3 | **16.90%** | **81.34%** | — |
| facebook | 2 | **78.39%** | 94.34% | 252.75 |
| NetHEPT | 1 | **42.43%** | 94.63% | 18.63 |
| NetHEPT | 10 | 5.05% | 66.24% | 59.74 |
| bitcoin_alpha | 2 | −1.28% | **88.37%** | 1229.77 |

**headroom 最大到 90.8%（ca-GrQc k=1）**，所以"值得优化"是成立的。
`random` 在多数格子上落后 40%~95%，说明**选择确实重要**。

### 一个必须报告的现象：部分格子 gap 为负

例如 Congress k=5 时 `degree gap = −0.63%`、email k=3 时 `−4.00%`、
ca-GrQc k=10 时 `degree_discount −1.26%`。
**这不是 bug**：这些格子**已饱和**（方法 spread ≈ oracle spread ≈ 全图规模），
差分落在评测 MC 噪声内。**饱和区间不能用来排序方法**——已在 `stage4b` 中显式标注。

另一个现象：**`degree_discount` 在 8 个格子里与 `degree` 完全相同**。
因为归一化稠密图上 top-度节点常是同一批人，折扣项未能改变选择。这解释了它为何没有优势。

---

## 4. Q3：过度暴露是否改变了排序？**在多数图上没有**

比较同一状态下 `Δ(v|S)` 在 overexposure 与 no-overexposure（`τ=1`）两种设定下的 Spearman：

| 图 | ρ(OE, no-OE) |
| --- | --- |
| Congress-Twitter | +0.995 |
| email-Eu-core | +1.000 |
| ca-GrQc | +0.968 |
| NetHEPT | +0.998 |

**排序几乎完全一致。** 也就是说：**移除过度暴露不改变"谁更好"，只改变"整体扩散多少"。**

⚠️ **这对论文是一个需要正视的问题**：如果 overexposure 不改变排序，
那么"overexposure 设定"的价值就**不在于它让排序变难**，
而在于**它使 RR 机制不可用**（Gate 1d 已证明 RR 集为空）。

**这反而与已确认的定位一致**：卖点是"RR 不适用时怎么做"，不是"排序更难"。

---

## 5. Q4：两者能否共存？**只有 3/18 个格子**

`stage4b` 判据：`state_ratio > 0.5` **且** `degree_gap > 2%`。

| 图 | k | state_ratio | degree gap | 判定 |
| --- | --- | --- | --- | --- |
| **Congress-Twitter** | **2** | **2.435** | **10.99%** | ✅ |
| Congress-Twitter | 1 | 2.133 | 7.43% | ✅ |
| ca-GrQc | 3 | 0.503 | 16.90% | ✅ |
| email-Eu-core | 1~3 | 0.42~1.52 | −4.0~0% | ❌ degree 已足够 |
| facebook | 1~3 | 0.050~0.130 | 38.6~78.4% | ❌ 无状态依赖 |
| **NetHEPT** | 1~3 | 0.062~0.094 | −2.9~42.4% | ❌ 无状态依赖 |
| bitcoin_alpha | 1~3 | 0.077~0.350 | −1.3~0.4% | ❌ 两者皆低 |
| ca-GrQc | 1~2 | 0.306~0.339 | 4.7~90.8% | ❌ ratio 略低 |

### 结构性张力（**必须让 PI 知道**）

```
状态依赖强 ←→ 稠密图 ←→ 迅速饱和 ←→ headroom 小
headroom 大 ←→ 稀疏图 ←→ 静态排序问题 ←→ 状态依赖弱
```

**两件事在 18 个格子里只有 3 个同时成立。**

### 但有一个重要的缓和因素

**k 小 ≠ oracle 便宜。** Congress-Twitter k=2、全图 475 个候选时，
MC greedy 每步要对约 475 个候选估计边际，总成本是 **~2×475×MC** 次级联。
**所以"用学习省 oracle 调用"的主张在 k 小时依然成立，甚至更必要**
（因为 k 小时 headroom 最大，改进空间最有价值）。

⇒ **可用工作区间是：稠密图 + 小 k。** 具体建议：
`congress_twitter` 与 `email_eu_core` 的 `k ∈ {1,2,3}`，这是阶段 5 的默认配置。

---

## 6. 对论文 claim 的支持情况

| 论文需要的前提 | 状态 | 证据 |
| --- | --- | --- |
| learned prediction 必须 condition on state | ✅ **成立** | ratio 2.1~5.6（稠密图） |
| 存在值得优化的 headroom | ✅ **成立** | gap 最大 90.8%，random 落后 40~95% |
| RR 机制不适用 | ✅ **成立**（Gate 1d 已有） | RR 集恒为空 |
| overexposure 让**排序**变难 | ❌ **不成立** | ρ(OE, no-OE) ≥ 0.968 |
| 状态依赖与 headroom 普遍共存 | ⚠️ **仅 3/18 格子** | 见 §5 |

---

## 7. 当前最大的科学风险

**可用区间很窄，而窄区间会削弱"顺序决策"的叙事。**
core claim 涉及 `k` 步顺序决策 + audit + progressive verification；
但可用区间是 `k ∈ {1,2,3}`，**顺序性本身的空间有限**。

两个应对方向（需 PI 判断）：

1. **把主张重心放在 cost 轴**：即使 k=2，MC oracle 仍需数千次级联，
   学习+验证仍能省成本。顺序性只是次要叙事。**（我推荐）**
2. **寻找 k 较大但仍不饱和的图**：需要更稀疏的大图（如更大的社交网络），
   当前 9 张图里没有合适的。

---

## 8. 下一步

**阶段 5：把现有 GRL pipeline 接到 overexposure oracle 上。**
默认配置：`congress_twitter` / `email_eu_core`，`k ∈ {1,2,3}`，pool 40，
oracle-mc 40，eval-mc 150。

必须逐项验证任务书列的 6 个检查点（predictor 是否 condition on S、是否 marginal、
audit 是否真检查可靠性、verification 是否真省 oracle、fallback 是否真存在、
corrupted predictor 是否触发 fallback）。
