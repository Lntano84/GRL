# 仓库合并说明（GaoYucen/GRL ↔ Lntano84/GRL）

> 合并日期：2026-09-15
> 合并结果位置：`C:\Users\windows\Desktop\_grl_merge\merged`
> 目标仓库：`Lntano84/GRL`（**尚未 push，等你确认**）

---

## 一、合并前的实际状况（重要）

两个仓库**同源**（README 哈希完全相同），但从 2026-08-19 起**分裂成两条并行线**：

| | **GaoYucen/GRL** | **Lntano84/GRL** |
| --- | --- | --- |
| HEAD | `1a56731` | `55816fe` |
| 提交数 | **45** | 8 |
| 最后提交 | 2026-09-05 | 2026-08-31 |
| 文件数 | **157** | 111 |
| 独有内容 | ⭐ **ICLR 2027 完整论文草稿**（`paper/iclr2027/`）<br>研究状态四件套（`RESEARCH_STATE` / `NEXT_STEPS` / `DECISIONS` / `EXPERIMENT_LOG`）<br>**14 个实验脚本**（`scripts/experiments/`）<br>14 个结果 JSON（`docs/results/`）<br>RR screening + audited residual + trust gate 全链路 | 候选检索基准（S1）<br>检索重排联合实验（L3）<br>**overlap 特征**（S4）<br>`diagnostics/candidate_benchmark`、`diagnostics/retrieval_reranking`、`experiments/overlap`、`features/overlap`、`models/marginal_gain_overlap`、`training/comparison` |

⚠️ **两边都不包含对方**（`gaoyucen` 里没有 `55816fe/71e1569/bcc6686` 三个提交，反之亦然）。所以这次合并**双方都有真实贡献**，不是单向覆盖。

---

## 二、合并策略

**以 GaoYucen 为骨架**（因为它有论文脚手架 + 更完整的研究记录），**再把 Lntano 的独有模块移植进来**。

```
base   = GaoYucen/GRL @ 1a56731        （157 文件）
+ 28   = Lntano 独有文件                （B 侧：检索/overlap/诊断）
+ 11   = 冲突文件采用 Lntano 版本        （B 侧更新）
+ 1    = 修一处合并回归
```

### 2.1 ⚠️ 关键决策：11 个冲突文件采用了 **Lntano（B 侧）** 版本

这是合并中最需要判断的地方。逐个核对提交日期后发现：

| 文件 | GaoYucen | Lntano | **采用** |
| --- | --- | --- | --- |
| `diffusion/independent_cascade.py` | 08-19 (2890B) | **08-21 (4577B)** | **Lntano** |
| `models/marginal_gain.py` | 08-19 (2875B) | **08-21 (3480B)** | **Lntano** |
| `training/marginal_dataset.py` | 08-19 (2888B) | **08-21 (6173B)** | **Lntano** |
| `baselines/degree_discount.py` | 07-15 (911B) | **08-31 (2972B)** | **Lntano** |
| `baselines/__init__.py` | 07-15 | **08-31** | **Lntano** |
| `diagnostics/__init__.py` | 07-15 | **08-31** | **Lntano** |
| `models/__init__.py` | 08-19 | **08-31** | **Lntano** |
| `training/__init__.py` | 08-19 | **08-21** | **Lntano** |
| `diffusion/__init__.py` | 08-19 | **08-21** | **Lntano** |
| `tests/test_diffusion.py` | 07-15 | **08-21** | **Lntano** |
| `tests/test_marginal_dataset.py` | 08-19 | **08-21** | **Lntano** |

**理由**：Lntano 侧新增了 A 侧没有的能力——
- `estimate_marginal_gains`（**多候选配对 MC**，同一批 live-edge 采样，避免负边际增益标签）
- `_sample_live_graph`、`_reachable_nodes` 等辅助函数
- 动态 Degree Discount（每步重算）

> ⚠️ **如果这里选错，会丢掉 A 的核心引擎改进。** 这是本次合并最需要你复核的一点。

### 2.2 修掉的一处合并回归

`src/grl/diffusion/__init__.py`（B 侧版本）**只导出了 3 个函数**，漏了 `run_independent_cascade`。
但 `scripts/run_first_smoke.py` 与 `tests/test_diffusion.py` 依赖它 ⇒ **导入会失败**。

已补上导出，现在 `grl.diffusion` 导出 4 个：`run_independent_cascade`、`estimate_spread`、`estimate_marginal_gain`、`estimate_marginal_gains`。

---

## 三、验证结果 ✅

| 检查项 | 结果 |
| --- | --- |
| **pytest 全量** | ✅ **36 passed**（10.52s） |
| **17 个核心模块 import** | ✅ 全部通过（A 侧 9 个 + B 侧 8 个） |
| 两个 track 的脚本共存 | ✅ `scripts/experiments/`（A）与 `scripts/run_*.py`（B）无冲突 |
| `grl.diffusion` API | ✅ 4 个函数齐备 |

**A 侧模块**（paper/audit track）：`models.marginal_gain`、`diffusion`、`training.marginal_dataset`、`training.gnn_trainer`、`training.marginal_trainer`、`evaluation.{spread,sequential,ranking}`、`diagnostics.oracle`

**B 侧模块**（retrieval/overlap track）：`diagnostics.candidate_benchmark`、`diagnostics.retrieval_reranking`、`experiments.overlap`、`features.overlap`、`models.marginal_gain_overlap`、`training.comparison`、`baselines.{degree,degree_discount}`

---

## 四、合并后的目录结构

```
merged/
├─ AGENTS.md                    ← 会话恢复入口
├─ README.md
├─ configs/                     （A 侧配置 + B 侧 4 个新配置）
├─ data/                        （NetHEPT / Epinions / Twitter / BigTestData）
├─ docs/
│   ├─ RESEARCH_STATE.md        ⭐ 研究状态（含 P0 paper-value risk 记录）
│   ├─ NEXT_STEPS.md            ⭐ 下一步（含 ICLR 路线六步）
│   ├─ DECISIONS.md             ⭐ 决策日志
│   ├─ EXPERIMENT_LOG.md        ⭐ 实验日志
│   ├─ results/                 （14 个结果 JSON + OPIM-C 对比 md）
│   ├─ experiments/             （2026-09-02 marginal gain 验证）
│   ├─ CODEBASE_GUIDE.md / DEVELOPMENT_PLAN.md / EXPERIMENT_PROTOCOL.md
│   ├─ PAPER_CODE_MAPPING.md
│   └─ COLLABORATOR_HANDOFF.md / OVERLAP_GENERALIZATION.md / RETRIEVAL_RERANKING.md   （B 侧）
├─ paper/iclr2027/              ⭐ 论文草稿（src/ + notes/ + build/）
├─ scripts/
│   ├─ experiments/             （A 侧 14 个实验脚本）
│   ├─ run_*.py                 （B 侧 4 个入口 + 通用入口）
│   └─ debug/
├─ src/grl/
│   ├─ algorithms/sequential_im.py
│   ├─ oracle/marginal.py
│   ├─ diagnostics/{oracle, candidate_benchmark, retrieval_reranking}
│   ├─ experiments/overlap.py
│   ├─ features/overlap.py
│   ├─ models/{gnn, marginal_gain, marginal_gain_overlap}
│   ├─ training/{gnn_trainer, marginal_trainer, marginal_dataset, comparison}
│   ├─ diffusion/independent_cascade.py
│   ├─ baselines/{degree, degree_discount}
│   └─ evaluation/{spread, sequential, ranking, gnn_metrics}
└─ tests/                       （36 个测试）
```

---

## 五、⚠️ 仍需你确认/处理

### 5.1 合并方向与 push

你要求"整合到 Lntano 里面"。当前合并结果的 **git 历史是重新初始化的**（因为两个仓库历史独立，无法直接 merge）。

**两个选项**：

| 选项 | 做法 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **A（推荐）** | 以合并结果为准，**force push 到 `Lntano84/GRL`** | 干净、单一历史 | 会覆盖 Lntano 现有 8 个提交（但内容已保留在合并树里） |
| **B（保守）** | 把合并结果作为**新分支** `merge/gaoyucen` push 到 Lntano | 不破坏现有 main | 历史仍分裂 |
| **C** | 把合并结果 push 到 **GaoYucen/GRL** | 保留 45 个提交历史 | 违背你"整合到 Lntano"的要求 |

⚠️ **我倾向 A 或 B，但这是覆盖式操作，必须你点头我才做。**

### 5.2 一处需要你复核的技术决策

**§2.1 的 11 个冲突文件我选了 Lntano 版本。** 请在合并结果里跑一次：

```powershell
cd C:\Users\windows\Desktop\_grl_merge\merged
$env:PYTHONPATH='src'
python -m pytest -q
python scripts/run_first_smoke.py --config configs/smoke/network_science_first_round.yaml
```

**如果 A 侧某些论文结果无法复现**，说明该文件的 A 侧版本有 Lntano 没有的改动 ⇒ 需要逐个 review 而不是整体替换。

### 5.3 数据文件

`data/twitter-d.txt` 有 **37MB**，两个仓库都提交了。合并后仍在。如果 Lntano 仓库有体积限制，建议改为 Git LFS 或移出。

---

## 六、复现命令

```powershell
cd C:\Users\windows\Desktop\_grl_merge\merged
$env:PYTHONPATH='src'
python -m pytest -q                                    # 36 passed
python scripts/inspect_dataset.py --config configs/nethept.yaml

# A 侧（paper/audit track）
python scripts/experiments/evaluate_fullgraph_rr_screening_multiseed.py --help
python scripts/experiments/evaluate_audited_residual_gate.py --help

# B 侧（retrieval/overlap track）
python scripts/run_candidate_benchmark.py --config configs/smoke/candidate_benchmark_nethept.yaml
python scripts/run_retrieval_reranking.py --config configs/smoke/retrieval_reranking_nethept.yaml
python scripts/run_overlap_experiment.py --config configs/smoke/overlap_nethept.yaml
```

---

## 附：合并的 git 记录

```
1eaafea  fix: export run_independent_cascade from grl.diffusion (merge regression)
b880e86  merge: adopt B-side newer engine layer (diffusion/models/training/baselines)
325c9e7  merge: port B-side retrieval/overlap/diagnostics modules from Lntano84/GRL @ 55816fe
65d806b  base: GaoYucen/GRL @ 1a56731 (paper + audit track)
```
