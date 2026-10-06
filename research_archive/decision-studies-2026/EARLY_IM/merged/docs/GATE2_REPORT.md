# Gate 2 结果：零噪声策略与停止规则

> 日期：2026-09-17
> 前置：`docs/GATE1_REPORT.md`（Gate 1c v3 / 1d）、`docs/EXPERIMENTS_20260916.md`（Gate 2 主表）
> 结论：**Gate 2 通过。解析排序在稠密图饱和区间显著优于度数（8/8 稳健），但优势是图相关的。**

---

## 一、Gate 2b：`delta2` 是好的**排序器**、坏的**校准器**

`exposure_scores_delta` 用**已实现的**曝光状态评分（40 个候选，每个 60 次配对试验）：

| `|S|/n` | 真实边际均值 | 解析估计均值 | 比值 | Spearman |
| --- | --- | --- | --- | --- |
| 0% | **+193.586** | +2.852 | **0.01** | 0.902 |
| 5% | −0.661 | −0.251 | 0.38 | 0.667 |
| 10% | **+0.167** | −0.095 | **−0.57** | 0.263 |
| 20% | **+0.120** | −0.192 | **−1.60** | 0.427 |

**两条结论**：

1. **绝对尺度错约 100 倍**（未饱和时 2.85 vs 193.6）；
2. **饱和区间符号相反**（真实 +0.12，解析 −0.19）。

因此**解析分的零交叉不能当停止判据**。Gate 2b 的第一版策略就因此在
**1 个种子**后停止（spread 347 vs 度数基线 364）。

但 **Spearman 全程为正**（0.90 / 0.67 / 0.26 / 0.43）⇒ 它排序可用。

> **教训：排序与停止是两个不同的问题，需要不同的信息。**

---

## 二、Gate 2c：拆开两个职责——解析排序 + 观测 spread 停止

`docs/results/overexposure_plateau_stop_20260916.json`
`scripts/experiments/evaluate_overexposure_plateau_stop.py`

- **排序**用解析分（零噪声闭式）；
- **停止**用**观测到的 spread**（阈值窗口固定后，级联是种子集的确定性函数
  —— Gate 1d —— 所以观测值**无估计噪声**，不像 MC 贪心要用 `MC=25` 的噪声估计）。

### Congress-Twitter（budget = 10，pool = 60，200 次配对试验）

| `\|S\|/n` | 度数 top-10 | **`plateau@0`（纯解析排序）** | `plateau@2`（耐心 2） | plateau@2 实际用种 |
| --- | --- | --- | --- | --- |
| 0% | +362.15 ±2.13 | **+366.55 ±1.61** | +365.44 ±2.62 | 3 |
| 5% | **−4.59** ±0.63 | **−0.50** ±0.61 | −1.42 ±0.61 | 6 |
| 10% | **−1.97** ±0.71 | **+2.23** ±0.60 | +0.98 ±0.48 | 5 |
| 20% | **−5.06** ±0.61 | **+1.00** ±0.46 | +0.17 ±0.38 | 3 |

**解析排序把度数的伤害从 −5 变成 +1**，且 `plateau@2` 只用 3~6 个种子就接近满预算效果。

---

## 三、Gate 2d：对 selection draw 稳健吗？**8/8 全胜**

`docs/results/overexposure_selection_robustness_20260916.json`
`scripts/experiments/evaluate_overexposure_selection_robustness.py`

单一 selection draw 可能是偶然。用 **8 次独立 selection draw** 重做
"选种 → 评测"全过程（每次 100 次配对评测，评测窗口跨策略共享）：

| `\|S\|/n` | 度数 | `analytic` 均值 [min, max] | 胜度数 | `plateau@2` 均值 | 胜度数 |
| --- | --- | --- | --- | --- | --- |
| 0% | +362.72 | **+366.30** [364.29, 368.05] | **8/8** | +365.02 | 8/8 |
| 10% | **−2.51** | **+1.19** [0.48, 2.10] | **8/8** | +0.99 | 8/8 |
| 20% | **−5.55** | **+0.23** [0.09, 0.42] | **8/8** | −0.74 | 8/8 |

**没有任何一次抽样出现反例。** 度数基线本身也很稳定（8 次全部在 +5.07~+5.97），
说明 10% 那格度数确实退化，解析排序的优势不是抽样噪声。

---

## 四、⚠️ 但优势是**图相关的**：NetHEPT 上度数常常胜出

`docs/results/overexposure_plateau_stop_nethept_20260916.json`

| `\|S\|/n` | 度数 | **`plateau@0`** | 差 | 胜者 |
| --- | --- | --- | --- | --- |
| 0% | +123.45 ±5.53 | **+126.21** ±5.58 | +2.8 | 解析（微弱） |
| 10% | +41.80 ±2.07 | **+49.05** ±2.00 | **+7.3** | 解析 |
| **20%** | **+15.70** ±0.94 | +11.05 ±0.92 | **−4.6** | **度数** |
| 40% | +7.04 ±0.40 | **+9.30** ±0.65 | +2.3 | 解析 |

**NetHEPT 上度数全程为正**（+7.04 ~ +41.80），且在 `|S|/n = 20%` 时**反超解析排序**。

### 对比总结

| 图 | `⟨k⟩` | 度数在饱和区间是否有害 | 解析排序是否更好 |
| --- | --- | --- | --- |
| **Congress-Twitter**（475 节点） | **55.95** | ✅ 有害（−1.97 ~ −5.55） | ✅ 是（8/8） |
| **NetHEPT**（15,233 节点） | **4.23** | ❌ 无害（全为正） | ⚠️ 多数区间是，20% 时否 |

⇒ **可主张的说法**：
> "在**稠密网络**的饱和区间，度数排序会变成负信号，而状态感知的解析排序
> 能在**零仿真预算**下避免这一伤害。"

⇒ **不可主张**："状态感知排序普遍优于度数"。

### 机制解释（为什么稠密图上度数会反号）

入权归一化后 `δ(v) ≤ 1`，且 `δ(v)` 由**入度**驱动、种子价值由**出度**驱动。
Congress-Twitter 的 `ρ(入度, 出度) = +0.491` 但 `⟨k⟩ = 55.95`：
高度数节点入邻居极多 ⇒ 其邻域**已被自身推入过度暴露区**，
再加它就会让更多邻居越过 `θ^τ`。
NetHEPT 的 `⟨k⟩ = 4.23`，单个种子的曝光贡献有限，饱和主要靠**种子数量**而非度数，
所以度数不反号。

---

## 五、对 DASFAA 的意义

### 现在可以主张的完整链条

1. **RR 结构性不可用**（Gate 1d）—— 最强立项理由，可证明；
2. **度数在稠密图饱和区间反号**（Gate 1c v3 + Gate 2 主表），
   `ρ_degree` 三图一致由正转负，Congress 上实测 −5.55；
3. **精确 MC 贪心在饱和区间也失效**（Gate 2 主表：+0.42 vs 解析 +2.26）
   —— 因为真边际接近零时噪声主导决策；
4. **解析排序 + 观测停止** 在稠密图上零预算击败两者（8/8 稳健）。

### ⚠️ 仍然存在的风险（必须诚实面对）

**第 4 条的主角仍是那个无训练的解析公式。** GRL 若要立项，必须回答：

> **"GNN 相对 `delta2` 的增量在哪里？"**

目前的证据只能支持：`delta2` 在**稠密图饱和区间**优于度数与 MC 贪心。
**一个学习型模型必须在同一区间再超过 `delta2`**，否则论文卖点不成立。

### 建议的 Gate 3

| 优先级 | 动作 | 判据 |
| --- | --- | --- |
| **P0** | 训练一个**状态条件化排序器**，目标为真实边际增益 | 在 Congress `\|S\|/n ≥ 10%` 上 Spearman 或 top-5 精度**超过 `delta2`** |
| **P0** | 把 Gate 2 主表复现到 Wiki-Vote / Trust Bitcoin-Alpha | 确认"稠密 ⇒ 度数反号"这一图级判据 |
| **P1** | 用 `\|S\|/n` 而非绝对 `\|S\|` 作为实验轴 | 已确立为必须 |
| **P1** | 量化 `λ(S) − σ(S)`（对 `σ^κ`/`σ^τ` 跑真 RR） | 把 Gate 1d 做成定量对比 |

---

## 六、复现命令

```powershell
$M = "C:\Users\windows\Desktop\_grl_merge\merged"; cd $M; $env:PYTHONPATH="$M\src"

# Gate 2c：解析排序 + 耐心停止（Congress）
python scripts/experiments/evaluate_overexposure_plateau_stop.py `
  --graph congress_twitter --fractions 0.0 0.05 0.10 0.20 `
  --budget 10 --pool-size 60 --trials 200 --patience 0 1 2 3 `
  --output docs/results/overexposure_plateau_stop_20260916.json

# Gate 2d：跨 selection draw 稳健性（8/8 全胜就在这里）
python scripts/experiments/evaluate_overexposure_selection_robustness.py `
  --graph congress_twitter --fractions 0.0 0.10 0.20 `
  --budget 10 --pool-size 60 --selection-draws 8 --eval-trials 100 --patience 2 3 `
  --output docs/results/overexposure_selection_robustness_20260916.json

# 跨图对照（NetHEPT 上度数胜出）
python scripts/experiments/evaluate_overexposure_plateau_stop.py `
  --graph nethept --fractions 0.0 0.10 0.20 0.40 `
  --budget 10 --pool-size 120 --trials 80 --patience 0 2 `
  --output docs/results/overexposure_plateau_stop_nethept_20260916.json
```
