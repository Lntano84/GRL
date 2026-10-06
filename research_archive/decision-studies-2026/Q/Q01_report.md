# Q01 —— 超时重试空间与强规则筛查（第一步：数据与反馈资格）

> # ⚠️ 机制部分已被 Q01R_report.md 取代
>
> 已撤回：「取消当完成」的错记窗口（我的见证引入了代码里不存在的截止值）、
> 「不存在重试臂」（存在隐式重跑路径）、「9 处同类违规」（需按是否超出该次执行所能揭示的信息重分类）。
> 保留：候选淘汰泄漏、数值矩阵可用、以及在固定预算比较里必须去掉特权停止条件。
> 保留本文件作记录。


## 结论

**数据资格通过；反馈权限不通过。**

| 项 | 结果 |
|---|---|
| 四张官方矩阵可直接回放 | **通过**（无需 Dropbox，无需数据库，无需 EXPLAIN 计划）|
| 数据完整性 | **通过**（无缺失、无无穷、无零、无负值；行标签唯一）|
| 合法反馈接口 | **不通过**：官方实现有两处分支读取**隐藏真值**来决定记账与候选集 |
| 重试臂 | **不存在**：`timeout_m` 从不被用于再次选择候选 |
| 因此 | 官方实现上的回放结果**不能直接当作可部署成绩**；本轮不据此评价论文结论 |

---

## 1. 数据资格（半天预算内完成）

**入口。** 你给的仓库地址可用：`zixy17/LimeQO`（默认分支 `main`）。
README 说明 `dataset/` 只含部分数据、完整数据需从 Dropbox 下载，
**但四张矩阵 CSV 和四个初始掩码都已随仓库提供**，因此第一步不需要 Dropbox。
EXPLAIN 计划（`dataset/*.zip`）只对 LimeQO+ 训练需要，本步不需要。

| 工作负载 | 行（查询） | 列（hint） | 行标签唯一 | 取值范围（秒）| 单元格中位（秒）| 缺失 | 无穷 | 零 | 负 |
|---|---:|---:|:--:|---|---:|---:|---:|---:|---:|
| ceb | 3,133 | 49 | True | 0.008108 – 4368 | 2.109 | 0 | 0 | 0 | 0 |
| job | 113 | 49 | True | 0.002916 – 2238 | 1.323 | 0 | 0 | 0 | 0 |
| stack | 6,191 | 49 | True | 0.003534 – 317 | 2.541 | 0 | 0 | 0 | 0 |
| dsb | 964 | 49 | True | 0.002545 – 969.9 | 6.119 | 0 | 0 | 0 | 0 |

（数值单位由 README 明确为**秒**。）

### 1.1 已核实的回放语义

| 要素 | 结论 | 依据 |
|---|---|---|
| 矩阵含义 | 行 = 查询，列 = hint，单元格 = 该 hint 下的**真实运行时间（秒）** | README + `dataset.py` |
| 默认计划 | **第 0 列**。`default_time = sum(matrix[:,0])`；它等于该行最优的比例为 ceb 758/3133、job 43/113、stack 502/6191、dsb 146/964 | `dataset.py:34-35` |
| 初始掩码 | `init_<name>_mask.npy`，形状与矩阵一致，标记起跑前**已观测**的单元格 | `dataset.py:31` |
| 重复计划对应关系 | 由**并查集**给出：`get_same_hints(q,h)` 返回与 h 等价的全部 hint；观测一个即等价观测整组 | `dataset.py:188-203`，`utils/union_find.py` |
| 矩阵中的数值重复 | 每个数据集都有若干**完全相同**的列：ceb/job/dsb 各 7 对、stack 15 对；例如列 (5,6)、(12,13)、(19,20) 在四张表里都成对相同 | `q01_data_profile.json` |
| 注意 | 并查集来自 EXPLAIN 计划（需 Dropbox）；**矩阵 CSV 本身不含等价类信息**。若不做等价合并，回放会低估“一次观测揭示多条 hint”的效果 |

### 1.2 可改进空间（诊断量）

| 工作负载 | 严格优于本行默认计划的单元格 | 占比 | 相对默认计划的中位加速 | 最大加速 |
|---|---:|---:|---:|---:|
| ceb | 32,198 | 21.0% | 1.15x | 284.4x |
| job | 911 | 16.5% | 1.26x | 51.0x |
| stack | 62,738 | 20.7% | 1.16x | 113.8x |
| dsb | 10,945 | 23.2% | 1.56x | 262.1x |

默认计划只在少数查询上是最优的，因此**提升空间真实存在**；这不是新发现，
只是确认这份数据能支撑后续的成本—质量比较。

---

## 2. 反馈权限：两处分支读取隐藏真值（已坐实）

### 2.1 负载分支

```python
dataset.py:30   self.matrix = self.matrix_df.to_numpy()        # 真实运行时间，全部隐藏
limeqo.py:89    if dataset.matrix[select, hint] >= min_observed[select]:
limeqo.py:90        explored_m[select, same_hints] = 1
limeqo.py:92    if dataset.matrix[select, hint] >= timeout_tolerance:
limeqo.py:93        timeout_m[select, same_hints] = timeout_tolerance
limeqo.py:97    mask[select, same_hints] = 1
limeqo.py:98    explored_m[select, same_hints] = 1
```

### 2.2 最小见证

固定上下文：当前最好值 `b = 10`，候选在 `t = 2` 被取消，`timeout_tolerance = 10`。
**合法的唯一反馈是「运行时间 ≥ 2」。** 只改变隐藏真实运行时间 x：

| 隐藏 x | x ≥ t（合法可见）| x ≥ b | x ≥ tol | 写入 `mask` | 写入 `timeout_m` | 记账结果 |
|---:|:--:|:--:|:--:|:--:|:--:|---|
| 0.5 | False | False | False | 是 | 否 | booked as a MEASUREMENT of the runtime |
| 1.9 | False | False | False | 是 | 否 | booked as a MEASUREMENT of the runtime |
| 2.0 | True | False | False | 是 | 否 | booked as a MEASUREMENT of the runtime |
| 4.0 | True | False | False | 是 | 否 | booked as a MEASUREMENT of the runtime |
| 9.9 | True | False | False | 是 | 否 | booked as a MEASUREMENT of the runtime |
| 10.0 | True | True | True | 否 | 是 | recorded as a censored lower bound |
| 25.0 | True | True | True | 否 | 是 | recorded as a censored lower bound |

**同一个合法观察「运行时间 ≥ 2」导致两种互斥的记账：**

- `x < tol`：被取消的运行被写进 `mask`，即**当作一次完整的运行时间测量**；
- `x ≥ tol`：被写进 `timeout_m`，即当作**删失下界**。

两者的差别不只是记账：写进 `mask` 的那一支还会 `cnt += 1`（计入 `new_observe_size`）
并把该单元格从后续候选中移除。所以**隐藏真值同时决定了记账和候选集**。

**需要精确说明的一点。** 在本例中 `alpha*b = tol`，所以若截止值取 `t = alpha*b`，
则 `t = tol`，两支的分界恰好落在 t 上。真正会被错记的窗口是
**`t ≤ x < tol`**（运行被截断、但仍落在容忍度内）：此时代码把一次被取消的运行
当作测量，而合法反馈无法把它与「运行完成且耗时 x」区分开。

### 2.3 同类分支遍布全部策略

源码扫描发现**未加掩码**读取真实矩阵的位置共 **9 处**：

| 文件 | 行 | 代码 |
|---|---:|---|
| `strategies/limeqo.py` | 89 | `if dataset.matrix[select, hint] >= min_observed[select]:` |
| `strategies/limeqo.py` | 92 | `if dataset.matrix[select, hint] >= timeout_tolerance:` |
| `strategies/limeqo.py` | 114 | `if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:` |
| `strategies/limeqo_plus.py` | 114 | `if dataset.matrix[select, hint] >= min_observed[select]:` |
| `strategies/limeqo_plus.py` | 117 | `if dataset.matrix[select, hint] >= timeout_tolerance:` |
| `strategies/limeqo_plus.py` | 139 | `if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:` |
| `strategies/greedy.py` | 74 | `if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:` |
| `strategies/random.py` | 60 | `if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:` |
| `strategies/qo_advisor.py` | 68 | `if dataset.matrix[select, hint] >= min_observed[select]:` |

因此这不是 LimeQO 独有的一行疏漏，而是**整组策略共用的实现约定**：
`random`、`greedy`、`qo_advisor`、`limeqo`、`limeqo_plus` 都依赖它。

---

## 3. 不存在重试臂（对 Q01 主问题直接相关）

`timeout_m` 的全部出现位置：

| 行 | 类型 | 代码 |
|---:|---|---|
| 28 | 初始化 | `timeout_m = np.zeros_like(dataset.matrix)` |
| 41 | **唯一读取** | `log_timeout_m = np.log1p(timeout_m)` |
| 44 | 传入 ML | `pred_m = censored_als(log_m, mask, log_timeout_m, ...)` |
| 93 | 写入 | `timeout_m[select, same_hints] = timeout_tolerance` |
| 117 | 写入 | `timeout_m[file_i, same_hints] = min_observed[file_i]` |

唯一一次读取是把删失指示喂给矩阵分解。**没有任何代码用 `timeout_m` 决定是否以更长截止值重跑**，
候选过滤器（82–85 行）也不看它。

**这对选题的含义：** 官方实现里**没有「重试旧超时候选」这件事**。
所以「何时重试」既不是它已经解决的问题，也不是它已经排除的问题——
它是一个**尚未被这套代码表达**的决策，这既不是优势也不是缺口，需要独立检验。

---

## 4. 本轮不做什么、以及现在的证据层次

本轮**没有**：训练任何模型、运行完整实验、下载 Dropbox 数据、重建数据库、调窗口或阈值。

| 层次 | 当前判断 |
|---|---|
| 存在优化空间 | **有**：16.5–23.2% 的单元格优于本行默认计划 |
| 合法观测可辨识 | **未测**：接口要先修，本轮只证伪了现成实现 |
| 学习能够捕捉 | 未测 |
| 收益超过成本 | 未测 |
| 相对已有方法有新意 | 未建立；BayesQO / LimeQO 仍是对手 |

### 4.1 对下一步的直接影响

按你的计划，第 2 步是「建立最小反馈接口」。本轮结果表明该接口**不能建立在官方策略代码之上**，
需要：

1. 把 `matrix` 从策略可见对象中彻底移除，只暴露 `observe(q,h) -> (value, kind)`，
   其中 `kind ∈ {measured, cancelled_at_t}`；
2. 记账规则改为**只看观测类型**：`cancelled_at_t` 一律进 `timeout_m`，
   不论隐藏真值是否落在容忍度内；
3. 用本报告 §2.2 的两世界例子作为**回归测试**：两个世界必须给出相同的策略状态；
4. 若要复用官方策略做对照，必须把 89/92（以及其它 7 处）替换为只读观测的版本，
   并**明确标注这是修改版**，不能称为官方成绩。

---

## 5. 产物与哈希

### 5.1 冻结的代码版本

| 文件 | sha256 |
|---|---|
| `src\data\dataset.py` | `D9A656DD27EA2B6330F40F61654C0D5178D43B6F49BE96E05E26360F80463C42` |
| `src\models\matrix_factorization.py` | `C716E6DBA81BBDCA1608F84C9996400F0E070746C55DA2AFEC05A6632091E2FF` |
| `src\run_experiment.py` | `10C8124BBDEE40DF6E46BBC0D7DE8325073E35739BE55034603A25DB9260DE78` |
| `src\strategies\base.py` | `BA5943D070E4C5A7B47D8CD1EDDF9D60DAFC741D95119197522969E71BB6DF96` |
| `src\strategies\greedy.py` | `69D8F99475B33B9D60B078D31FA12DF6DDD73D66CF8741856489B00688536F19` |
| `src\strategies\limeqo.py` | `B2880921F5EBDEA424ABCA1539E919E95C9D7B8C1B7A40155E919DBC74E19F9B` |
| `src\strategies\limeqo_plus.py` | `BE3930D4960E6AECCAF4612B77AD46813DF74C62BBC3A9CB7E9D81673CCF7F9E` |
| `src\strategies\oracle.py` | `F36AEBD447C834A1B0148D25C7928DA4E5D6EA39436604F67C3E6C4C5B7AF395` |
| `src\strategies\qo_advisor.py` | `141468518338AADEAE5D53D99F8767699AA2E65FB7E4882699E4D5A69BB58D39` |
| `src\strategies\random.py` | `D3218F24F18B666825DDFAF59BC88BC856770D45C8A10C046F6203FBEDA6107E` |
| `src\utils\union_find.py` | `ACE9BCF249C9A31E63DFD9EF4D27932B107AA72B20181B7D6386D0CF0913FA5E` |

### 5.2 数据文件

| 文件 | 字节 | sha256 |
|---|---:|---|
| `ceb-matrix.csv` | 2,935,346 | `DC6CD87809277B43583CE62658C002C5C32F1AD4B07F6B261A17EBE129622920` |
| `dsb-matrix.csv` | 903,005 | `188D37DF36CE8D8ACBDC5C29A989905DD0B930B2A5F3EE4AFE478DB4A7406E60` |
| `init_ceb_mask.npy` | 1,228,264 | `2648BA3863831E2FB7405B9B46440EA0BBF2AD29C187F74467584C94316011C0` |
| `init_dsb_mask.npy` | 378,016 | `17CCD069A6C377B8AC5D3059DD913D2DB1C73137F039C3395FC892D5F81F559C` |
| `init_job_mask.npy` | 44,424 | `982041E30FC7EDFEA3CFF974D9C4482528DEA8D32199FF4D3E90BA55A3AA98F0` |
| `init_stack_mask.npy` | 2,427,000 | `BF19494A6FC65B6D1067FFA6EC1820C72CE8EBD6787763D3C8818F0B1D359120` |
| `job-matrix.csv` | 105,524 | `D2CA376FC4692149630362E187C17E557AE6FAA00FFEF00B3341E4F006BDA645` |
| `stack-matrix.csv` | 5,991,453 | `AA1AE2402DF66C3A36FEB62799ADB98F8D877D6943E31C07A54152A01DC1181D` |

### 5.3 脚本

| 脚本 | 作用 |
|---|---|
| `Q01_work/q01_fetch.py` | 按仓库路径抓取文件（>1MB 走 download_url）|
| `Q01_work/q01_profile.py` | 四张矩阵的数据资格剖析 |
| `Q01_work/q01_two_world.py` | 用官方策略跑两世界，记录真值读取 |
| `Q01_work/q01_attribution.py` | 把每次真值读取归因到调用行 |
| `Q01_work/q01_branch_audit.py` | 分支真值表 + 未掩码读取源码扫描 + 暴露量扫描 |
| `Q01_work/q01_minimal_witness.py` | **最小见证**（决策类型 vs 隐藏真值）|
| `Q01_work/q01_exposure.py` | 错记窗口与「无重试臂」确认 |
