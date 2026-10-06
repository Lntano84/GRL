# HANDOFF — Stage 08（连续搜索轨迹诊断：EMPTY vs FULL，何时产出可用改善）

目录：`08_LOT_REOPT/stage06_neighborhood_content/`

## 冻结的问题

> EMPTY 的改善通常何时出现？FULL 是否存在更早、且能合法传给 EMPTY 的改善解？

## 答案

**EMPTY 的改善出现在后段，很晚；FULL 没有更早优势，其解也不能合法传给 EMPTY。**

| 方法 | ρ | 首次可用改善达到 | 平均时间 | 20 s 检查点改善数 | 40 s 检查点改善数 |
|---|---|---|---|---|---|
| EMPTY | 0.75 | **16/16** | **29.63 s** | **0/16** | 16/16 |
| EMPTY | 1.10 | **16/16** | **17.07 s** | **12/16** | 16/16 |
| FULL | 0.75 | **2/16** | 39.78 s | **0/16** | 1/16 |
| FULL | 1.10 | **16/16** | **31.73 s**（比 EMPTY 更晚） | **0/16** | 15/16 |

- **EMPTY 的 60 个改善事件**：≤5 s 有 **1** 个、≤10 s 有 1 个、≤20 s 有 12 个、≤40 s 有 58 个；**中位数 27.86 s**。
- **首次达到 2% 的时间**：EMPTY rho0.75 平均 29.99 s、rho1.10 平均 18.70 s；总计 50/64 达到（未达标者保持"尚未达到"，未填 40 秒）。
- **迁移判定**：40 s 上 FULL 优于同期 EMPTY 的 **11 例中 0 例可迁移**（短期 Y 违反 3–4 个）→ **全部 BLOCKED**。且这 11 例**全在 40 s**；5/10/20 s 检查点上 FULL **从未**优于 EMPTY。

## 裁决（对照预登记预案）

- 预案①"EMPTY 改善主要在后段、FULL 无更早优势 → **保留连续 EMPTY，不再尝试颠倒顺序**"：**成立**。
- 预案②"FULL 有早期优势且部分解可迁移 → 才设计先宽后窄"：**不成立**（无早期优势，且 0 例可迁移）。
- 预案③"FULL 早期解更好但违反固定条件 → 记录为迁移障碍"：**部分命中**——违反属实，但"早期更好"不成立。
- 预案④"两者大多无改善 → 查日志时间去向"：不成立（58/64 有改善），但时间去向已查明（见下）。
- 预案⑤"不同状态时间偏好稳定交叉 → 才验证信息能否预测"：**未观察到稳定交叉**。

**⇒ 保留连续 EMPTY；不设计"先宽后窄"；无启动 GRL 训练的依据。**

## 机制（来自求解器日志）

`large_rho0.75_s2|D1_m0_2p` s1 的 HiGHS 日志：**前 20 秒 primal 一直等于修复成本**，对偶界只推进约 0.9%；primal 首次下降在 **25.0 s**。时间构成 `Solve 39.86 s`（`MIP 33.50`、`subMIP 6.36`），**LP iterations 113932**（separation 68899 / strong br. 18737 / heuristics 24091），而**节点数只有 2**。

⇒ **瓶颈是根节点的对偶界收紧与分离/强分支**，不是策略模块不足。这直接解释 Stage 07：它切掉的 5 秒正是 EMPTY 完全不产出的时段。

## 关键数据

- 64 次运行 = 16 冻结 large 状态 × {EMPTY, FULL} × 2 种子；每次**连续 40 秒**，不中断、不重启、不切分；wall 2594 s。
- 起点、κ、固定参考均为冻结 `S_r`；模型/求解器/线程同 Stage 07。
- `stage08_savecheck.py`：**64 个交付方案 + 158 个事件向量全部重验**，最大约束违反 **0**，`usable` 标记与实际重验逐条一致（158/158）；事件时间非降、目标不劣化；**64/64** 含提交起点事件且未被计为改善；MIP start 采用 64/64；日志 64/64 保留。
- 与 Stage 07 一致性：**20 秒前缀 vs Stage 07 的 E20 交付成本 28/32 相同**（差异为独立批次的求解器噪声）。

## 边界（勿过度解读）

- 检查点是**同一条 40 秒轨迹的前缀**，不是分别设置 5/10/20/40 秒截止的独立运行。
- 验解**离线**：检查点数值是"到该时刻找到的解的质量"，**不是严格实时交付成绩**；模型准备、提交起点、回调开销均未从端到端墙钟扣除。
- **未证明统一时间阈值**，且**受 40 秒窗口右截断**（58/60 事件落在窗内）："5 秒太短"确定，"至少需要几秒"未测。
- 存在一次异常早事件（`large_rho1.10_s4|D2_all_1p` s1，0.12 s），经验解确认可行且优于修复，单例不改变分布结论。
- rho0.75 上 FULL 仅 2/16 改善的原因**未确定**。
- 40 秒是诊断窗口，**不代表部署预算已从 20 秒改为 40 秒**。

## 交付物

- `stage08_report.md` — 报告
- `stage08_runs.json` — 完整逐运行记录（每次改善的时间、标量、完整变量向量、离线验解结果）
- `stage08_results.csv` — 精简逐运行结果
- `stage08_logs/` — **64 个 HiGHS 求解日志**
- `stage08_savecheck.py` / `stage08_analyse.py` / `stage08_analysis.json`
- `stage08_run.py`（逐运行 checkpoint + `--resume`）/ `stage08_preflight.py`（回调接口预检）

## 复现命令

```powershell
cd "C:\Users\windows\Desktop\im算法与基准阅读包\08_LOT_REOPT\stage06_neighborhood_content"
.\.venv-hs\Scripts\python.exe stage08_preflight.py
.\.venv-hs\Scripts\python.exe stage08_savecheck.py
.\.venv-hs\Scripts\python.exe stage08_analyse.py
# 重跑（约 43 分钟）：
.\.venv-hs\Scripts\python.exe stage08_run.py --resume
```

## 接口事实（highspy 1.15.1，已由预检验证）

- 订阅：`h.cbMipImprovingSolution.subscribe(fn)`，`fn` 收到 `HighsCallbackEvent`。
- 取值：`event.val(np.arange(n_cols))` 读取 incumbent 向量；`data_out.mip_solution` 为其底层数组。
- 字段：`data_out` 提供 `objective_function_value`、`mip_primal_bound`、`mip_dual_bound`、`mip_gap`、`mip_node_count`、`running_time`。
- **提交的起点本身会触发一次事件**（目标值等于 `S_r`，时间约 0.05 s）——**不可计为改善**；分析中按成本等于 `J_r` 排除。

## 环境

- `.venv-hs\` 是指向 `stage05_mipstart\.venv-hs` 的目录联接：highspy 1.15.1 / numpy 2.5.3 / scipy 1.18.1。
- 系统 Python 3.13.14 **没有** highspy；清华 pip 镜像不含 highspy，需 `--index-url https://pypi.org/simple`。
