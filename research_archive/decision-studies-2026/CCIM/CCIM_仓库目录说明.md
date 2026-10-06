# CCIM 仓库目录说明

**根目录**：`C:\Users\windows\Desktop\im算法与基准阅读包\06_CCIM`
**ASCII 入口（推荐）**：`C:\Users\windows\Desktop\ccim_work` —— 是指向同一目录的 junction。
> 真实路径含中文，某些控制台会按 GBK 错显成 `im绠楁硶涓庡熀鍑嗛槄璇诲寘`；8.3 短名是 `IM算法~1`，**仍含中文**。所以用 `ccim_work`。

**不是 git 仓库**，无任何内容被推送。

---

## 顶层

| 目录/文件 | 内容 |
|---|---|
| `ccim/` | **线 A（CCIM，图已知）的库** |
| `ccim/xplore/` | **线 B（探索式 IM，图未知）的库** |
| `scripts/` | 全部可执行脚本（47 个） |
| `tests/` | 7 个测试文件，含合法性闸门 |
| `_data/xplore/` | **22 张官方图**（`pickle`，标签 0..n−1） |
| `_official/` | **从作者仓库重建的包树**（由脚本生成，可重建） |
| `_ref/` | 作者原始源码快照 `gsrl_*.py` 与论文 |
| `_extracted/` | 抽取出的论文正文（如 `0614.txt`） |
| `_notes/` | 临时笔记 |
| `results/` | **全部产物**（44 json / 22 txt / 9 log / 2 pt / 3 个子目录） |
| `CCIM_*.md` | 35 份报告（见下） |

---

## 库

### `ccim/` —— 线 A：图已知时的种子选择

| 文件 | 作用 |
|---|---|
| `model.py` | `load_graph` / `cascade` / `sigma` / `normalized`；**文档记录了两个 K-core 反例与历史上的 worklist bug** |
| `baselines.py` | `Ledger`（含 `timer=` 与五桶计时）、`Result`、degree / greedy / random+ 基线 |
| `baselines.py` | **`Ledger` 的预算单位是「发起的查询数」**，缓存命中仍计数 |
| `search.py` | 带 trace 的传统局部搜索 |
| `scan_order.py` | random / degree / threshold 三种**扫描顺序**（已收口） |
| `lazy_candidates.py` | **惰性 Fisher–Yates**（`LazyShuffle` / `ExplicitShuffle` / `decode`）—— 已验收的工程优化 |
| `swap_features.py` | 换点的 8 维特征 |
| `batched_search.py` | 批处理 32 的搜索框架 + 六桶计时 |
| `learned_rank.py` | 冻结的换点排序模型 + **按 `S` 分组**的数据集构建 |
| `dqn.py` / `shaping.py` | 已暂停的 DQN 分支（保留作历史） |

### `ccim/xplore/` —— 线 B：图未知时的调查决策

| 文件 | 作用 |
|---|---|
| `icm.py` | **IC 的 live-edge 估计 + 懒贪心**；与作者 `icm.py` 数值兼容（1.85% @4000 样本） |
| `env.py` | 调查环境（初始种子 / 前沿访问 / 完整图收益）与已发表策略 |
| `learner.py` | 简化 Geometric-DQN（GCN 编码器 + DQN）；含「观测看不到隐藏网络」的闸门 |
| `deepwalk_torch.py` | **唯一被替换的组件**：作者游走 + torch skip-gram（替代 gensim `hs=1`） |
| `dqn_opt.py` | 共享嵌入版更新（与作者差 4.0e-07，非逐位） |
| `dqn_batched.py` | padding+mask 批量版（**已停在诊断**） |
| `progress.py` | **逐位精确的原子断点续跑**（网络/优化器/replay+优先级/全部随机流） |

---

## 脚本（按用途）

**线 B 主链（按时间顺序）**

| 脚本 | 用途 |
|---|---|
| `official_assembly.py` | 从 `_ref/` 重建 `_official/` 并探针验证（14 处机械补丁逐条记录） |
| `xcheck_official_icm.py` | **直接执行作者 `icm.py`**，与本实现做数值交叉核验 |
| `xplore_baseline.py` | 3 张开发图 × 20 配对 × 9 预算的基线曲线（`--plan-only` 先冻结计划） |
| `xplore_learn.py` | 简化 Geometric-DQN 训练（`--variant` 支持 fast/batched） |
| `xplore_diag.py` | 训练贡献诊断（epoch 0 / 早期 / 选中 / 末期） |
| `xplore_official.py` | **官方架构改编版训练**，支持 `--variant` / `--episodes` / `--resume` |
| `xplore_official_eval.py` | 开发评估：五模型 + random/degree_max/degree_min，含**启发式逐位复现闸门** |
| `shorthorizon.py` / `shorthorizon_table.py` | 批量版短程对照（3 种子 × 200 episode）与汇总 |
| `survey_value_probe.py` | **调查价值诊断**（特权挑选 / 独立确认双批次） |
| `legal_ranker.py` | **合法信息排序器**（9 特征、单一 L2 pairwise logistic） |
| `legal_ranker_audit.py` | 结果审计：从**单一逐状态表**重生所有数字 |
| `frozen_validation.py` | **100 新状态冻结验证**（四臂、两主比较、Holm） |

**性能与等价性**

| 脚本 | 用途 |
|---|---|
| `bench_update.py` | 按线程数测更新耗时（1/2/4/16） |
| `bench_profiler.py` | `torch.profiler` 算子级分析 |
| `bench_variants.py` / `bench_batched.py` / `batched_check.py` | 变体计时与批量等价性 |
| `equiv_check.py` | Q/损失/梯度/参数/动作逐项对照 |
| `first_step.py` | 第一步参数差的定位（Adam 符号步） |
| `resume_check.py` | 断点续跑两阶段验证 |
| `attribute_diffusion_cost.py` | 扩散成本归属（四个假设逐一检验） |

**线 A**

`measure_ca_grqc.py`、`search_probe_ca_grqc.py`、`optimise_candidate_generation.py`、`list_candidate_instances.py`、`audit_search_trace.py`、`audit_search_logs.py`、`audit_query_accounting.py`、`diagnose_scan_order.py`、`check_cascade_correctness.py`、`sanity_check_setting.py`、`check_greedy_vs_paper.py`、`bound_ceiling.py`、`probe_hyperparameters.py`、`collect_swap_records.py`、`train_swap_ranker.py`、`audit_checkpoint_selection.py`、`learned_ranking_pilot.py`、`wallclock_comparison.py`、`inspect_swap_records.py`

---

## 测试（`tests/`，共 7 个文件）

| 文件 | 关键闸门 |
|---|---|
| `test_ccim_diffuser.py` | K-core 反例回归、worklist bug 回归 |
| `test_lazy_shuffle.py` | 惰性 Fisher–Yates 与显式实现逐位一致 |
| `test_batched_search.py` | **`batch_size=1` 与已验收顺序扫描逐位等价** |
| `test_learned_rank.py` | 状态 id 跨轨迹碰撞回归、numpy/torch 排序一致 |
| `test_xplore.py` | 估计器对显式 IC 模拟、调查只揭示真邻居 |
| `test_xplore_learner.py` | **观测看不到隐藏网络** |
| `test_ranker_legality.py` | 孪生图（只改隐藏边）下特征/分数/选择全一致 |

---

## 报告（`CCIM_*.md`）

**总览（先读这两份）**
- `CCIM_最新实验报告.md` —— 结论、负面结果、开放问题
- `CCIM_仓库目录说明.md` —— 本文件

**线 B 主线**
`CCIM_XPLORE_协议侦察.md`（官方协议逐条）→ `CCIM_XPLORE_基线曲线.md` → `CCIM_XPLORE_GEODQN.md`（简化版）→ `CCIM_XPLORE_训练贡献诊断.md` → `CCIM_XPLORE_官方版状态.md` / `_官方版就绪与预算阻塞.md` → `CCIM_XPLORE_fast500试验.md` / `_fast500结果.md`

**线 B 调查价值（最新、最重要）**
`CCIM_XPLORE_调查价值诊断.md`（**正向空间**）→ `CCIM_XPLORE_合法信息排序器.md` / `_结果.md` → `CCIM_XPLORE_结果审计.md`（**发现比较 bug**）→ `CCIM_XPLORE_冻结验证.md`（**判决：停止扩展**）

**性能/工程**
`CCIM_XPLORE_梯度更新耗时定位.md`、`_消除重复计算.md`、`_批量化前提.md`、`_批量化结果.md`、`_短程结果.md`、`_第一步定位与短程.md`、`_断点续跑.md`、`_短程进度.md`

**线 A**
`CCIM_GATE1_多步决策价值.md`、`CCIM_GATE2_奖励重分配.md`、`CCIM_SEARCH_LOG_AUDIT.md`、`CCIM_SEARCH_TRACE.md`、`CCIM_SCAN_ORDER_PILOT.md`、`CCIM_CANDIDATE_INSTANCES.md`、`CCIM_CA_GRQC_PROBE.md`、`CCIM_CA_GRQC_SEARCH_PROBE.md`、`CCIM_CANDIDATE_GENERATION_OPT.md`、`CCIM_LEARNED_RANKING_PILOT.md`、`CCIM_WALLCLOCK_COMPARISON.md`

**其他**
`CCIM_论文信息.md`、`CCIM_阅读笔记_预读框架.md`、`CCIM_交接记录_关机.md`

---

## `results/` 关键产物

| 文件/目录 | 内容 |
|---|---|
| `xplore_curve.json` | 基线配对曲线 2,160 条记录 |
| `official_assembly.json` | 14 处兼容补丁逐条 + 5 个探针结果 |
| `xcheck_official_icm.json` | 与作者估计器的交叉核验 |
| `official_checkpoints/` | 改编版 5 种子的 ep0/100/250/500 快照 + `_progress.pt` 续跑点 |
| `frozen_ranker.json` | **冻结的排序器**（权重、标准化参数、配置、哈希） |
| `frozen_validation.json` | 100 新状态的验证结果与逐状态配对差 |
| `legal_ranker_audit.json` | 含逐状态表、候选 ID、确认值的审计数据 |
| `survey_value_probe.json` | 调查价值诊断 |
| `shorthorizon_table.json` | 批量版短程对照汇总 |
| `bench_threads_*.json` | 各线程数的更新耗时 |

---

## 复现入口（最短路径）

```powershell
cd C:\Users\windows\Desktop\ccim_work

# 测试
python -m pytest tests/ -q

# 线 B：调查价值诊断（约 90 秒，无需训练）
python -u scripts/survey_value_probe.py

# 线 B：冻结验证（约 10 分钟，无需训练）
python -u scripts/frozen_validation.py

# 线 B：从断点继续训练官方架构改编版
$env:PYTHONUNBUFFERED='1'; python -u scripts/xplore_official.py --resume

# 重建作者代码包树并探针
python scripts/official_assembly.py

# 与作者估计器交叉核验
python scripts/xcheck_official_icm.py
```

**注意（踩过的坑）**
- 长任务用 `python -u`（否则 `Tee-Object` 缓冲会让崩溃时的输出全丢）
- 不要用 PowerShell 的 `Set-Content` 往返 UTF-8 文档（会按 GBK 破坏）
- 启动长训练前先确认 `replay > batch_size`，否则一次梯度更新都不会触发
- `pip` 需加 `--index-url https://pypi.org/simple`（默认镜像 `pypi.tuna.tsinghua.edu.cn` 取不到任何包）
