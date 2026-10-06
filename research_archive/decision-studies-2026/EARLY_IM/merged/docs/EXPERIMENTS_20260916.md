# 2026-09-16 / 17 实验汇总（CCIM / overexposure 方向）

> 本文件汇总全部实验、结论、被推翻的中间结论、**已知故障**，以及可复现命令。
> 详细判定见 `docs/GATE1_REPORT.md`；论文数据集核对见 `docs/PAPER_DATASETS_AND_BENCHMARK_PLAN.md`。

---

## 一、Gate 2 通过：解析基线击败精确贪心，GRL 有了明确对手（最新，最强）

`docs/results/overexposure_gate2_congress_paired_20260916.json`

Congress-Twitter，`budget = 10`（每个策略可加 10 个种子），pool = 60，
**400 次配对试验、逐 trial 共享同一组阈值窗口**（标准误已列出）：

| `\|S\|` | `\|S\|/n` | 度数 top-10 | **`delta2` top-10** | 精确 MC 贪心 top-10 | 胜者 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0% | +362.51 ±0.61 | +361.64 ±0.59 | **+367.28 ±0.58** | 贪心 |
| 24 | 5.1% | **−4.39** ±0.48 | **+0.49** ±0.39 | −0.01 ±0.45 | **`delta2`** |
| 48 | 10.1% | **−1.61** ±0.47 | **+2.24** ±0.32 | +0.28 ±0.46 | **`delta2`** |
| 95 | 20.0% | **−5.17** ±0.40 | **+2.26** ±0.24 | +0.42 ±0.31 | **`delta2`** |

三条结论：

1. **未饱和时（`|S|/n = 0`）三者几乎持平**（362~367），度数已经够好。
   真正的竞争区间是 `|S|/n ≥ 5%`。
2. **饱和区间度数变成有害信号**：加入度数最高的 10 个节点使 spread **下降** 1.6~5.2
   （`|S|=95` 时 −5.17 ± 0.40 = 13 倍标准误）。
3. **`delta2` 不仅打败度数，还打败了精确 MC 贪心**（`|S|=95`：+2.26 vs +0.42）。
   原因：贪心用 `MC=25` 的**噪声**估计找边际，而饱和区间真边际接近零 ⇒ **噪声主导决策**；
   `delta2` 是解析的、零噪声。

⇒ **Gate 2 的缺口被精确定义了**：不是"排序更准"，而是
**"在近乎零边际的饱和区间，用零噪声的方式判断该不该加、加哪个"**。
这是一个明确、可度量、且当前无解的目标。

---

## 二、RR 在该模型上**结构性不可用**（最强立项理由）

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

---

## 三、度数在**饱和区间**确实反号（用户最初假设成立）

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

### 补充：NetHEPT 在 `|S|/n = 10%` 仍为正

用逐 trial 共享窗口的 120 次配对（`|S| = 1523`）：

| 配置 | spread |
| --- | --- |
| `seeds`（1523 个） | 2739.77（std 59.3） |
| `seeds + 度数 top-10` | **2780.74**（std 58.4） |
| **配对边际** | **+40.97 ± 1.62，0/120 为负** |

⇒ **塌缩是图相关的**：Congress-Twitter（`⟨k⟩=55.95`）会塌缩，NetHEPT（`⟨k⟩=4.23`）不会。

---

## 四、已澄清的虚警：配对估计器**没有**故障（自纠）

> **记录此节是为了防止以后重复排查。** 我在本轮一度判定
> `estimate_overexposure_spread_over_configs` 在高 `|S|` 时不可信，**该判定是错的**。

**虚警的由来**：我在复现脚本里比较了
`spread([seeds]) = 2736.49` 与 `spread([by_degree]) = 140.71`，
把后者当成了"基准 + 10 个种子"的结果。但 `[by_degree]` 是**只含 10 个种子的配置**，
它与"1523 个种子 + 这 10 个"是**完全不同的集合**。140.71 只是 10 个种子单独作用时的
绝对 spread，不是任何边际增益。**我把不可比的量当成了差值。**

**决定性验证**（`cfg = seeds + deg10`，1533 个节点）：

| 方法 | 100 次试验均值 |
| --- | --- |
| 手写循环（`random.Random(rs+off)` → 采样窗口 → `run_overexposure`） | **2777.14** |
| `estimate_overexposure_spread_over_configs([cfg], 100, rs)` | **2777.14** |

逐 trial 值逐个相同（2815 / 2806 / 2766 / 2662 / 2765 / 2745 …），窗口哈希一致。
另确认估计器**不会**修改图的边权。

### 一条方法学教训（比故障本身更重要）

> **做"边际增益"对比时必须让两次测量共享同一组随机数，并显式打印两个配置的基数。**
> 这次的错误不是代码 bug，而是**比较了两个基数不同的集合**，
> 且输出里没有打印 `|seeds|` 和 `|variant|`。

---

## 五、今日被推翻的中间结论（必须记录，避免重犯）

| # | 中间结论 | 为何错 | 更正 |
| --- | --- | --- | --- |
| 1 | "度数在过度暴露下不失效"（Gate 1a/1a'） | 候选池是**全图均匀抽样**，度分布重尾 ⇒ 池内几乎全是低度节点，测的是"高度节点增益大"这条全局规律；且 `\|S\|/n ≈ 0`，未进入饱和区间 | 度数在 `\|S\|/n ≥ 20%` 时反号 |
| 2 | "非单调性在标准设定下不存在"（Gate 1c v2） | 只在 NetHEPT 上测，而 `\|S\|=200` 对 15233 个节点仅 **1.3%** | 按**比例**扫描后，三图均出现负边际 |
| 3 | "换非归一化权重会让过度暴露更剧烈" | 实测**方向相反**：均匀 `p=0.01` 使 `δ` 上限仅 ~0.15，级联几乎不发生（平均边际 1.046），负边际 0/120 | 该模型与均匀稀疏权重**本质不兼容** |
| 4 | "`nethept_uniform.yaml` 提供均匀权重" | `_parse_graph_file` 只在行内**无第 3 列**时才用 `default_probability`，而 NetHEPT.txt 自带权重列 ⇒ 配置从未生效 | 生成 `data/NetHEPT_uniform.txt` |
| 5 | "Gate 1b 证明自适应 +58%" | 后续在饱和区间重测，`nonadaptive` 与 `adaptive` 几乎相同 | 需区分"顺序"与"状态条件化" |
| 6 | "配对估计器在高 `\|S\|` 时坏了" | 比较了两个基数不同的集合（见第四节） | 估计器正确 |

### 一条方法学教训

> **本项目最容易翻车的地方是"区间"。** 同一个假设在 `|S|/n ≈ 0` 不成立、
> 在 `|S|/n ≥ 20%` 成立。**所有后续实验必须同时报告 `|S|` 与 `|S|/n`。**

---

## 六、代码与数据变更

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

## 七、脚本清单（今日新增）

| 脚本 | 用途 | 关键结论 |
| --- | --- | --- |
| `evaluate_overexposure_pool_ranking.py` | 度数检索池 vs 均匀池对照 + bootstrap CI | 池子构成是 1a/1a' 失败的根因 |
| `evaluate_overexposure_monotonicity_rr.py` | 饱和曲线、权重压力、入权归一化、反向可达 | **Gate 1d：RR 集为空** |
| **`evaluate_overexposure_paper_graphs.py`** | **论文三图 × 种子比例扫描** | **Gate 1c v3：度数反号** |
| `evaluate_overexposure_cost_quality.py` | 成本-质量前沿框架 | Gate 2 摊销定位工具 |
| **`evaluate_overexposure_gate2_headroom.py`** | `degree`/`delta2`/`exact greedy` 三方对比 | **Gate 2：饱和区间 capture 比率不稳定，改报绝对边际** |
| `evaluate_overexposure_sequential_saturated.py` | 饱和区间的静态/非自适应/自适应/状态感知对比 | Gate 2b |

---

## 八、可复现命令

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

# Gate 2：成本-质量前沿（注意 capture 列在饱和区间不稳定）
python scripts/experiments/evaluate_overexposure_gate2_headroom.py `
  --graph congress_twitter --fractions 0.0 0.05 0.10 0.20 `
  --budget 10 --pool-size 60 --mc-greedy 25 --mc-eval 150 `
  --output docs/results/overexposure_gate2_congress_20260916.json

# 测试
python -m pytest tests/ -q
```

### ⚠️ 复现 Gate 2 主表时的手写配对模板

脚本内的 `estimate_overexposure_spread_over_configs` 在同一次调用内共享窗口，
口径正确但**在饱和区间噪声占比高**。主表数字用下面这种**逐 trial 显式共享窗口**的写法得到，
建议后续统一用这个模板：

```python
for t in range(T):
    r = random.Random(base_seed + 7 + t)
    ws = sample_threshold_windows(nodes, r)
    base = run_overexposure(g, seeds, ws, r).spread
    variant = run_overexposure(g, seeds + extra, ws, r).spread
    marginal.append(variant - base)
```

---

## 九、下一步（按优先级）

| 优先级 | 动作 | 理由 |
| --- | --- | --- |
| **P0** | **在饱和区间做"零噪声 + 会停止"的策略**，与 `delta2` 和贪心比 | Gate 2 缺口已精确：`\|S\|/n ≥ 5%` 时 delta2（+2.26）> 贪心（+0.42）> 度数（−5.17） |
| **P0** | 把 1a'' 重跑到 `\|S\|/n` 均匀网格（三图） | 现有结论均在低覆盖率区间 |
| **P1** | 量化 `delta2` 的零点偏差并修正 | 高分时 `g'(δ)→−2`，其零交叉不是真实停止点 |
| **P1** | 在 Wiki-Vote / NetHEPT 上复现 Gate 2 主表 | 验证"塌缩是图相关的"这一判断 |
| **P1** | 找 Occupywallstnyc 数据源；解析 Trust Bitcoin-Alpha | 补齐论文四数据集 |
| **P1** | 对 `σ^κ`/`σ^τ` 跑真 RR，量化 `λ(S) − σ(S)` | 把"RR 只能解上界"做成定量实验 |
| **P2** | 复现 UB / IGA / Max_Degree / IMRank / PageRank / CELF | 论文 baseline |

### 待决策

- 导师同步：**暂缓**（用户 2026-09-16 指示：等 Gate 2 有初步结果；**现在已有结果，可汇报**）。
- GRL 定位：**摊销/重复查询成本**（用户选定）。
