# HANDOFF — Stage 06（邻域内容 vs 匹配随机；是否值得部署）

目录：`08_LOT_REOPT/stage06_neighborhood_content/`

> **本文件已按配对口径重写。** 初版的实例级聚合有缺陷（见 C12），导致初版裁决数字全部失真。

## 本轮冻结的研究问题

> 在新的独立名义实例上，同样释放 24 个短期 Y，利用故障、生产延续和容量竞争信息选变量，是否比匹配数量与位置的随机选择更有效，并能改善简单 FULL／EMPTY 策略？

## 裁决：必须分两层陈述

**内容层：PASSED**（C1–C4 全通过）

| 标准 | 取值 | 判定 |
|---|---|---|
| C1 ≥4 个名义实例 G ≥ 2% | 7 | PASS |
| C2 ≥2 个 scale-rho 组合 ≥ 2% | 2（medium rho0.75 +7.37%、medium rho1.10 +4.44%） | PASS |
| C3 两个**求解器种子**方向一致 | seed0 +2.58%、seed1 +1.88%；medium 两种子均正、large 两种子均负 | PASS |
| C4 实例均值 ≥ 2% | **+2.23%** | PASS |

**部署层：NOT SUPPORTED**（不属于 C1–C4，另行陈述）

- **没有任何**固定规则优于冻结简单策略（medium→FULL, large→EMPTY）；8 条规则平均 G 全为负。
- 最好的 always DEPENDENCY-24 = **−2.03%**。
- 逐实例事后选优（oracle）不可部署，已从结果中移除。

**两层不矛盾**：内容层过线是因为 medium 的正效应（+5.90%）压过 large 的轻微负效应（−1.44%）；部署层里 DEPENDENCY-24 在两个尺度上都是负的。

## 关键数据

- 512 次运行 = 32 状态（16 名义实例 × 2 故障）× 8 配置 × 2 求解器种子；20 s；highspy 1.15.1。
- 16 个名义实例用独立新种子 2–5（medium/large × rho 0.75/1.10 × 4）。
- 全部 512 次热启动自**同一**修复解 `x0`，MIP start 采用 **512/512**。
- 配对指标 `G_n(a,b) = (1/4)·Σ_d Σ_s (J_{n,d,s,b} − J_{n,d,s,a})/max(1,|J_r(n,d)|)`；**两个故障的 J_r 在 16/16 个实例上都不同**（相对差 18%–132%）。
- `stage06_savecheck.py` 重新验证 **1024 个方案**，最大约束违反 9.09e-12；选择错误 0；输出劣于修复解 0 次。
- 匹配随机与 DEPENDENCY-24 平均重合 **44–47%**（min 25%，max 67%），剖面按 0/1 类别 **192/192** 保持。
- DEPENDENCY-24 构造：large 平均依赖扩张 15.75 / medium 15.94（非种子位共 16），按短缺回填 ≈0。

## 本轮修正（全部只在分析/诊断层，原始数据未改动）

| 编号 | 内容 |
|---|---|
| **C10** | 按 `split("_")[2]` 取 rho；索引 2 是**实例种子**、索引 1 才是 rho → `by_scale_rho` 把 rho 平均掉了 |
| **C11** | 诊断脚本把"短缺分数为正的格数"实现成 `len(rel)`，恒为 24 |
| **C12** | **影响裁决**：`inst[name][cfg]` 与 `inst_jr[name]` 在遍历**状态**时被反复覆盖，后一个故障覆盖前一个 → "实例均值"从未平均两个故障，归一化基准取的是最后访问的故障值。**初版 C4=FAIL 是聚合缺陷造成的假象** |
| **C13** | C3 初版检查的是"对三个随机配置总体平均为正"，**并未检查两个求解器种子方向** |
| **C14** | `_profile` 用**原始浮点**修复 Y 作分层键；单状态有 7 个浮点值但只有 2 个类别 → 分层被碎片化，"190/192 剖面保持"与"MR2 分层池不足"都是假象。按 0/1 类别重算 **192/192** |

修正实现在 `stage06_aggregate.py`（配对聚合的唯一定义，analyse / verdict / tables 共用）与 `stage06_profile_check.py`。`stage06_runs.json` / `stage06_results.csv` **未被修改**。

## 可行性迁移证书（large 部分失败的直接解释）

`F_EMPTY ⊆ F_DEPENDENCY ⊆ F_FULL`，已在 **32 个 (状态 × 种子)** 上用落盘释放集合验证（违例 0）。在此基础上：

- EMPTY 交付解严格优于 DEPENDENCY-24 的配对共 **10 次，覆盖 6 个故障状态**；
- 这 10 次的 EMPTY 解**全部**满足 DEPENDENCY-24 的固定条件、短期翻转数**全部为 0**、经独立验解器复核可行且成本一致；
- ⇒ 更好的解本来就在 `F_DEPENDENCY` 内，DEPENDENCY-24 **没在 20 秒内找到它**：是**搜索失败**，不是邻域排除了它。
- **边界**：不解释 DEP 与随机的全部差距（随机可能用到 DEP 未释放的变量），也不适用于 medium。

## 交付物

- `stage06_report.md` — 报告（配对口径，含修正块与两层裁决）
- `stage06_per_run.csv` — **逐运行结果 512 行**
- `stage06_results.csv` — 运行器直接输出
- `stage06_runs.json` — 完整记录，**含每个运行的完整释放集合与选中/原始方案**
- `stage06_states.json` — 32 状态的修复解与元数据
- `stage06_analysis.json` / `stage06_verdict.json` / `stage06_tables.md` — 汇总、两层判定、机器生成表格
- `stage06_aggregate.py` — **配对聚合唯一定义**
- `stage06_aggregate_fix.py` — 聚合缺陷对照复核（缺陷口径 vs 配对口径）
- `stage06_profile_check.py` — 剖面与重合率的修正诊断
- `stage06_migration_certificate.py` — 可行性迁移证书
- `stage06_savecheck.py` — 抽样完整性复核（1024 个方案）

## 复现命令

```powershell
cd "C:\Users\windows\Desktop\im算法与基准阅读包\08_LOT_REOPT\stage06_neighborhood_content"
.\.venv-hs\Scripts\python.exe stage06_savecheck.py             # 抽样完整性复核
.\.venv-hs\Scripts\python.exe stage06_aggregate_fix.py         # 聚合口径对照
.\.venv-hs\Scripts\python.exe stage06_profile_check.py         # 剖面/重合率修正诊断
.\.venv-hs\Scripts\python.exe stage06_migration_certificate.py # 迁移证书
.\.venv-hs\Scripts\python.exe stage06_analyse.py               # Q1/Q2（配对口径）
.\.venv-hs\Scripts\python.exe stage06_verdict.py               # 两层判定
.\.venv-hs\Scripts\python.exe stage06_tables.py                # 报告中表格
```

## 环境

- `.venv-hs\` 是指向 `stage05_mipstart\.venv-hs` 的目录联接（junction）：highspy 1.15.1 / numpy 2.5.3 / scipy 1.18.1。
- 系统 Python 3.13.14 **没有** highspy；清华 pip 镜像不含 highspy，需 `--index-url https://pypi.org/simple`。
