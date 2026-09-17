# 2026-09-16 实验汇总（CCIM / overexposure 方向）

> 本文件汇总今日全部实验、结论、被推翻的中间结论，以及可复现命令。
> 详细判定见 `docs/GATE1_REPORT.md`；论文数据集核对见 `docs/PAPER_DATASETS_AND_BENCHMARK_PLAN.md`。

---

## 一、今日最重要的三个结论

### 1. RR 在该模型上**结构性不可用**（最强立项理由）

`docs/results/overexposure_monotonicity_rr_20260916.json`

```
nodes whose window includes 0    : 0  / 15233
RR-set size（300 次反向 BFS）     : min=0 median=0 mean=0.0 max=0
TIM 估计  n · Pr[S ∩ RR ≠ ∅]     : 0.0
真实 Monte-Carlo σ(S), |S|=50     : 111.9
```

两层原因：

- **没有锚点**：正激活要求 `δ ∈ [θ^κ, θ^τ]`，而窗口均匀采自 2D simplex
  `{(κ,τ): 0 ≤ κ ≤ τ ≤ 1}`，所以 `θ^κ = 0` 是零测集（实测 15233 节点中 **0 个**满足 `κ ≤ 0`）。
  于是 `δ = 0` 的节点正激活概率为 **0**，反向可达没有起点。
- **没有可采样的随机结构**：`δ(v,t) = Σ_{u∈A^in_t(v)} ω_uv`，其中 `A^in_t(v)` 是
  **"曾经正激活过"的入邻居集合**（论文原文明确包含 later-negatively-activated 的节点）。
  这是集合函数；**给定窗口后 δ 完全由 S 决定**，全过程唯一随机性是窗口采样。
  因此既不能采 live-edge 也不能采 trigger set —— **RR 恒等式的定义域不覆盖本模型**。

**交叉印证**：论文 Theorem 5 只给 **`γ/k`** 近似比（非 `1−1/e`），
且用 `σ^κ`/`σ^τ` 两个单调次模 LT 函数夹逼（Lemma 2 / Theorem 6，`λ(S) ≥ σ(S)`），
**其求解路径本身绕开了非单调目标**。

### 2. 度数在**饱和区间**确实反号（用户最初假设成立）

`docs/results/overexposure_paper_graphs_20260916.json`

控制变量是**种子比例 `|S|/n`**，不是绝对 `|S|`：

| 图 | n | `\|S\|/n` | ρ_degree | ρ_delta2 | 负边际占比 |
| --- | --- | --- | --- | --- | --- |
| **Congress-Twitter** | 475 | 0% | +0.472 | +0.833 | 0% |
| | | 5.1% | −0.324 | +0.323 | 49.2% |
| | | **10.1%** | −0.130 | +0.062 | **50.8%** |
| | | 40% | **−0.397** | +0.476 | 34.2% |
| **Wiki-Vote** | 7115 | 0% | +0.792 | +0.850 | 0% |
| | | 40% | **−0.229** | +0.250 | 10.8% |
| **NetHEPT** | 15233 | 0% | +0.912 | +0.962 | 0% |
| | | 40% | **−0.141** | +0.578 | 4.2% |

- `ρ_degree` **三图一致由正转负**；
- `delta2` 是**唯一全程保持正相关**的评分器；
- Congress-Twitter 的 `σ(S)` **非单调**（`monotone: False`），
  完全复现论文 §7.3(i) 对 Occupywallstnyc 的描述。

### 3. 但 GRL 的对手是 `delta2`，不是度数（Gate 2 风险）

`delta2` 是**无训练的两跳解析公式**，却在所有区间优于度数：

```
delta2 − degree：Congress +0.3003 / NetHEPT +0.2685 / Wiki-Vote +0.1332  (Spearman)
                 Congress +0.2995 / NetHEPT +0.2165 / Wiki-Vote +0.2161  (gain_capture)
```

⇒ **"用 GNN 学更准的评分器"必须超过 `delta2`。**

---

## 二、今日被推翻的中间结论（必须记录，避免重犯）

| # | 中间结论 | 为何错 | 更正 |
| --- | --- | --- | --- |
| 1 | "度数在过度暴露下不失效"（Gate 1a/1a'） | 候选池是**全图均匀抽样**，度分布重尾 ⇒ 池内几乎全是低度节点，测的是"高度节点增益大"这条全局规律；且 `\|S\|/n ≈ 0`，未进入饱和区间 | 度数在 `\|S\|/n ≥ 20%` 时反号 |
| 2 | "非单调性在标准设定下不存在"（Gate 1c v2） | 只在 NetHEPT 上测，而 `\|S\|=200` 对 15233 个节点仅 **1.3%** | 按**比例**扫描后，三图均出现负边际 |
| 3 | "换非归一化权重会让过度暴露更剧烈" | 实测**方向相反**：均匀 `p=0.01` 使 `δ` 上限仅 ~0.15，级联几乎不发生（平均边际 1.046），负边际 0/120 | 该模型与均匀稀疏权重**本质不兼容** |
| 4 | "`nethept_uniform.yaml` 提供均匀权重" | `_parse_graph_file` 只在行内**无第 3 列**时才用 `default_probability`，而 NetHEPT.txt 自带权重列 ⇒ 配置从未生效 | 生成 `data/NetHEPT_uniform.txt` |
| 5 | "Gate 1b 证明自适应 +58%" | 后续在饱和区间重测，`nonadaptive` 与 `adaptive` 几乎相同 | 需区分"顺序"与"状态条件化" |

### 一条方法学教训

> **本项目最容易翻车的地方是"区间"。** 同一个假设在 `|S|/n ≈ 0` 不成立、
> 在 `|S|/n ≥ 20%` 成立。**所有后续实验必须同时报告 `|S|` 与 `|S|/n`。**

---

## 三、代码与数据变更

### 修复的真实 bug

**`src/grl/data/graph_loader.py`** — `_parse_graph_file` 无法解析 NetworkX 边表格式：

```
congress.edgelist 内容:  0 4 {'weight': 0.002105263157894737}
旧代码按空白切分后 float(parts[2]) -> ValueError: could not convert string to float: "{'weight':"
```

**修复**：新增 `_parse_weight`（正则提取 dict 字面量里的权重，
无权重键时回落默认概率）+ `_find_header_end`（header 判定需同时满足
"两列非负整数"、"`m ≥ n`"、"`n` 至少为全图最大节点 ID 的 1/10"，
避免两列图的**首个边行被当成 header 吞掉**）。

**四种格式已验证**：

| 文件 | 格式 | 结果 |
| --- | --- | --- |
| `congress.edgelist` | `u v {'weight': w}` | 475 节点 / 13,289 边 ✅ |
| `wiki-Vote.txt` | `u v`（两列） | 7,115 节点 / 103,689 边 ✅ |
| `NetHEPT.txt` | `n m` header + `u v w` | 15,233 节点 / 32,235 边 ✅ |
| `NetHEPT_uniform.txt` | `n m` header + `u v` | 15,233 节点 / 32,235 边 ✅ |

**测试**：`tests/test_graph_loader_formats.py`（新增）
+ `tests/test_diffusion_namespace.py` ⇒ **全套 81 passed**。

### 新增数据

| 路径 | 内容 |
| --- | --- |
| `data/paper/wiki-Vote.txt` | SNAP Wiki-Vote（7,115 / 103,689） |
| `data/paper/congress/congress_network/congress.edgelist` | Congress-Twitter（475 / 13,289，带权重） |
| `data/paper/bitcoin-alpha.csv.gz` | Trust Bitcoin-Alpha（待解析） |
| `data/NetHEPT_uniform.txt` | **真**均匀权重 NetHEPT（仅用于反证，非对标用图） |

⚠️ **Occupywallstnyc 不在 SNAP 上**（多个候选 URL 均 404），需另找来源。

### 新增配置

`configs/congress_twitter.yaml`、`configs/wikivote.yaml`（`configs/nethept_uniform.yaml` 已修正指向）。

---

## 四、脚本清单（今日新增）

| 脚本 | 用途 | 关键结论 |
| --- | --- | --- |
| `evaluate_overexposure_pool_ranking.py` | 度数检索池 vs 均匀池对照 + bootstrap CI | 池子构成是 1a/1a' 失败的根因 |
| `evaluate_overexposure_monotonicity_rr.py` | 饱和曲线、权重压力、入权归一化、反向可达 | **Gate 1d：RR 集为空** |
| **`evaluate_overexposure_paper_graphs.py`** | **论文三图 × 种子比例扫描** | **Gate 1c v3：度数反号** |
| `evaluate_overexposure_cost_quality.py` | 成本-质量前沿框架 | Gate 2 摊销定位工具 |
| `evaluate_overexposure_gate2_headroom.py` | `degree`/`delta2`/`exact greedy` 三方对比 + capture 比 | Gate 2 |
| `evaluate_overexposure_sequential_saturated.py` | 饱和区间的静态/非自适应/自适应/状态感知对比 | Gate 2b |

---

## 五、可复现命令

```powershell
$M = "C:\Users\windows\Desktop\_grl_merge\merged"; cd $M; $env:PYTHONPATH="$M\src"

# Gate 1c v3：论文三图 × 种子比例（最重要）
python scripts/experiments/evaluate_overexposure_paper_graphs.py `
  --graphs congress_twitter wiki_vote nethept `
  --fractions 0.0 0.01 0.02 0.05 0.10 0.20 0.40 `
  --candidates 120 --mc-runs 20 --top-k 10 `
  --output docs/results/overexposure_paper_graphs_20260916.json

# Gate 1d：反向可达 + NetHEPT 饱和曲线
python scripts/experiments/evaluate_overexposure_monotonicity_rr.py `
  --sizes 0 10 30 60 120 250 500 1000 2000 4000 `
  --weight-scales 1.0 3.0 6.0 10.0 --stress-seed-sizes 0 200 800 `
  --output docs/results/overexposure_monotonicity_rr_20260916.json

# Gate 1a''：检索池条件排序 + bootstrap CI
python scripts/experiments/evaluate_overexposure_pool_ranking.py `
  --budgets 10 20 --contexts 6 --mc-runs 30 --pool-size 400 `
  --output docs/results/overexposure_pool_ranking_20260916.json

# Gate 2：成本-质量前沿
python scripts/experiments/evaluate_overexposure_gate2_headroom.py `
  --graph congress_twitter --fractions 0.0 0.02 0.05 0.10 0.20 `
  --pool-size 80 --mc-greedy 60 --mc-eval 150 `
  --output docs/results/overexposure_gate2_congress_20260916.json

# Gate 2b：饱和区间顺序策略
python scripts/experiments/evaluate_overexposure_sequential_saturated.py `
  --graph congress_twitter --budgets 10 25 50 --trials 8 --pool-size 120 `
  --output docs/results/overexposure_sequential_saturated_20260916.json

# 测试
python -m pytest tests/ -q
```

---

## 六、下一步（按优先级）

| 优先级 | 动作 | 理由 |
| --- | --- | --- |
| **P0** | **Gate 2：让 GNN 超过 `delta2`** | `delta2` 无训练却全局占优；只超过 `degree` 不足以支撑 DASFAA |
| **P0** | 把 1a'' 重跑到 `\|S\|/n` 均匀网格（三图） | 现有结论均在低覆盖率区间 |
| **P1** | 修 `delta2` 的零点偏差（高分时的停止判据） | 实测它在 `\|S\|` 大时系统性为负，不能直接当停止条件 |
| **P1** | 找 Occupywallstnyc 数据源；解析 Trust Bitcoin-Alpha | 补齐论文四数据集 |
| **P1** | 对 `σ^κ`/`σ^τ` 跑真 RR，量化 `λ(S) − σ(S)` | 把"RR 只能解上界"做成定量实验 |
| **P2** | 复现 UB / IGA / Max_Degree / IMRank / PageRank / CELF | 论文 baseline |

### 待决策

- 导师同步：**暂缓**（等 Gate 2 有初步结果）。
- GRL 定位：**摊销/重复查询成本**（门控 2 的核心卖点）。
