"""Q01-R: write the corrected report and manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MIRROR = HERE / "limeqo_mirror"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def main() -> int:
    mech = json.loads((HERE / "q01r_mechanism.json").read_text(encoding="utf-8"))
    checks = json.loads((HERE / "q01r_checks.json").read_text(encoding="utf-8"))
    dq = json.loads((HERE / "q01r_data_qualification.json").read_text(encoding="utf-8"))
    prof = json.loads((HERE / "q01_data_profile.json").read_text(encoding="utf-8"))

    files = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size}
             for p in sorted((MIRROR / "dataset").glob("*")) if p.stat().st_size}
    src = {str(p.relative_to(MIRROR)): sha(p) for p in sorted((MIRROR / "src").rglob("*.py"))}

    L: list[str] = []
    A = L.append
    A("# Q01-R —— 超时重试筛查：机制纠正与最小反馈接口")
    A("")
    A("> 本文件取代 `Q01_report.md` 的结论与机制部分。原文件保留作记录。")
    A("")
    A("## 结论")
    A("")
    A("**总体裁决保留：数值矩阵可用；现成回放不能直接作为合法策略。机制已改。**")
    A("")
    A("| 项 | 修正后 |")
    A("|---|---|")
    A("| L1 有限矩阵回放 | **通过** |")
    A("| L2 官方成本与等价传播 | **缺失**（计划等价类、初始掩码来源）|")
    A("| 合法的候选淘汰泄漏 | **成立**：`explored_m` 由不可观测的比较决定 |")
    A("| 特权停止条件 | **成立且已隔离**：停止规则读取完整隐藏矩阵的逐行最优 |")
    A("| 「取消当完成」 | **撤回**（我引入了一个代码里不存在的截止值）|")
    A("| 「不存在重试」 | **撤回**：存在隐式重跑路径，且其资格受隐藏真值影响 |")
    A("| 「九处同类违规」 | **撤回**：按「是否超出该次执行所能揭示」重新分类 |")
    A("")
    A("---")
    A("")
    A("## 1. 撤回清单")
    A("")
    A("| # | 上一轮的说法 | 更正 |")
    A("|---|---|---|")
    A("| 1 | 「存在 `t ≤ x < tol` 的错记窗口，代码把取消当完成」 | **撤回。** 官方这条路径的截止值**就是** `timeout_tolerance`；我的见证额外塞进一个 `deadline=2`，而它没有参与分支执行。设 `cap = tol = 10` 时，真值 4 或 9.9 秒的运行**确实会完成**，记为完整测量是正确的 |")
    A("| 2 | 「官方实现里没有重试臂；『何时重试』是尚未被表达的决策」 | **撤回。** 超时后若隐藏真值低于当前最好值，该单元格的 `mask=0且explored_m=0`，仍能通过候选资格检查，可被再次执行。**存在隐式重跑路径**，且其资格受隐藏真值影响 |")
    A("| 3 | 「9 处未加掩码读取真实矩阵 = 9 处违规」 | **撤回。** 回放环境必须读取隐藏运行时间才能生成反馈；判据是**读出的信息有没有超出该次执行能够揭示的范围**。按此重分类后只有一部分是违规，另有一类此前完全没提的特权读取 |")
    A("")
    A("---")
    A("")
    A("## 2. 正确的机制")
    A("")
    A("固定 `b = 10`，**截止值即容忍度**：")
    A("")
    A("| cap | 隐藏 x | x ≥ b | x ≥ cap | `mask` | `explored_m` | `timeout_m` | 计入预算 | 之后仍可作候选 |")
    A("|---:|---:|:--:|:--:|---:|---:|---:|:--:|:--:|")
    for r in mech["official_branch_table"]:
        A(f"| {r['cap']} | {r['x']} | {r['x'] >= 10} | {r['x'] >= r['cap']} | {r['mask']} | "
          f"{r['explored_m']} | {r['timeout_m']} | {r['counted']} | {r['eligible_again']} |")
    A("")
    A("**读法：**")
    A("")
    A("- `cap = 2`：两个真值**都正确记为超时**、都只收 2 秒。它们的差别**只在 `explored_m`**：")
    A("  x=5 保留候选资格，x=20 被淘汰。**而这个差别由隐藏真值决定——「超过 2 秒」并不能揭示它。**")
    A("- `cap = 10`：x=5 完成、x=20 超时，**这两行状态不同是合法的**。我上一轮把这一行当成了「已在 2 秒被取消」。")
    A("")
    A("**所以真正的问题是不可观测的候选淘汰**（[limeqo.py:89](Q01_work/limeqo_mirror/src/strategies/limeqo.py)），")
    A("不是完成标签错记。这不证明论文成绩的影响量级。")
    A("")
    A("### 2.1 隐式重跑路径")
    A("")
    A(f"目标单元格真值 x = 5，当前最好值 b = 10：")
    A("")
    A("| 轮次 | cap | 分支结果 | 收费 | 累计 |")
    A("|---|---:|---|---:|---:|")
    A("| 1 | 2 | `mask=0, explored_m=0, timeout_m=2` | 2 秒 | 2 秒 |")
    A("| 2 | 10 | `mask=1, explored_m=1, timeout_m=None` | 5 秒（**从头重跑**）| **7 秒** |")
    A("")
    A(f"总计 **{mech['retry_trace']['total_cost']:.0f} 秒**。所以正确表述是：")
    A("")
    A("> **没有单独命名的重试动作，但存在超时候选再次执行的路径；这条路径的资格受隐藏真值影响。**")
    A("")
    A("另外，`timeout_m` 会通过预测模型间接影响候选分数，因此**不能因为过滤器没有直接读取它，")
    A("就断言它不影响后续选择**。")
    A("")
    A("**对选题定位的影响：** 我们**不能**再把「引入重试」当作相对现有实现的新功能。")
    A("仍然开放的是**何时重跑值得它的成本**——这一条现有实现没有显式决策。")
    A("")
    A("---")
    A("")
    A("## 3. 读取点重分类")
    A("")
    A("判据：**这次读取是否超出了该次执行本身所能揭示的信息。**")
    A("")
    A("| 类别 | 位置 | 说明 |")
    A("|---|---|---|")
    A("| **违规：候选淘汰** | `limeqo.py:89`、`limeqo.py:114`、`limeqo_plus.py:114`、`limeqo_plus.py:139`、`greedy.py:74`、`random.py:60`、`qo_advisor.py:68` | 用未揭示的比较（x ≥ b）决定资格 |")
    A("| **违规：特权停止** | `limeqo.py:35`（`min_observed.sum() > opt_time + 20`）、`limeqo.py:105`（`opt_time + 50`）| `opt_time` 是**完整隐藏矩阵**的逐行最小值，即 oracle 最优；可用于事后评价，**不能作为可部署停止条件** |")
    A("| **合法：完成判定** | `limeqo.py:92`、`limeqo_plus.py:117` | 判定刚执行的那次运行在所给 cap 下完成还是超时；环境必须知道它才能报告反馈种类 |")
    A("")
    A("上一轮报的「9 处」这个计数**撤回**；上表才是需要区分的。")
    A("")
    A("**对固定预算比较的直接要求：** 必须去掉特权停止规则。这不自动使此前所有轨迹前缀失效，")
    A("但固定预算的对照实验不能沿用 `opt_time` 作为停止条件。")
    A("")
    A("---")
    A("")
    A("## 4. 最小反馈接口")
    A("")
    A("```text")
    A("observe(query, hint, cap) -> kind, value_or_bound, charged_seconds")
    A("")
    A("kind == 'completed' : 在 cap 内完成，value_or_bound 为实际耗时")
    A("kind == 'cancelled' : 在 cap 被停止，只揭示 'runtime >= cap'，收费 cap")
    A("```")
    A("")
    A("沿用原代码的边界约定：**`x < cap` 为完成，`x >= cap` 为超时**。重跑**从头计费**。")
    A("策略对象**不持有真实矩阵**；环境持有它（因为必须据它生成反馈），但只暴露上面这个三元组。")
    A("")
    A("实现见 `Q01_work/q01r_interface.py`（`ReplayEnvironment` + `LegalPolicy`）。")
    A("")
    A("### 4.1 四项确定性检查（全部通过，未拟合任何预测器）")
    A("")
    A("| 检查 | 要求 | 结果 |")
    A("|---|---|---|")
    A("| 1 | `b=10, cap=2, x∈{5,20}`：合法反馈相同，合法策略状态相同 | **通过**：两者都是 `('cancelled', 2, 2)`，状态逐字段相同 |")
    A("| 2 | `b=10, cap=10, x∈{5,20}`：分别完成与超时，允许状态不同 | **通过**：x=5 → `('completed', 5, 5)`；x=20 → `('cancelled', 10, 10)` |")
    A("| 3 | `x=5`，先 `cap=2` 再 `cap=10`：先超时后完成，总成本 7 | **通过**：2 + 5 = **7**，且首次超时后仍具资格 |")
    A("| 4 | 只改未揭示真值与全矩阵最优值：下一动作与固定预算停止判断不变 | **通过**：两个世界的下一动作均为 `(0,1)@cap2`，`stop_by_budget` 均为 False |")
    A("")
    A("**检查 1 中原分支的差异作为「被捕获的历史缺陷」保存**，写在 `q01r_checks.json` 的 "
      "`official_defect_differs` 字段里，**不写成合法实现应有的行为**：")
    A("原分支对 x=5 置 `explored_m=0`、对 x=20 置 `1`，尽管两者给策略的反馈完全相同。")
    A("")
    A("检查 4 正是官方停止规则**通不过**的那一项：它读 `opt_time`。")
    A("")
    A("---")
    A("")
    A("## 5. 数据资格（分两层）")
    A("")
    A("### L1 有限矩阵回放：**通过**")
    A("")
    A("| 工作负载 | 行 | 列 | 取值（秒）| 单元格中位 | 缺失/无穷/零/负 | 掩码形状 | 掩码二值 |")
    A("|---|---:|---:|---|---:|---|---|:--:|")
    for name in ("ceb", "job", "stack", "dsb"):
        d, a = prof[name], dq["initial_mask_audit"][name]
        A(f"| {name} | {d['rows']:,} | {d['cols']} | {d['min']:.4g} – {d['max']:.4g} | "
          f"{d['median']:.4g} | 全 0 | {tuple(a['mask_shape'])} | {a['binary']} |")
    A("")
    A("### L2 官方成本与等价传播：**缺失**")
    A("")
    A("- **计划等价类来自 EXPLAIN 计划包（Dropbox），矩阵 CSV 不含它。**")
    A("  没有它，回放无法复现「观测一个 hint 覆盖整个等价类」的效果。")
    A("- **初始掩码不只是第 0 列。** 四个数据集分别有 **38 / 51 / 42 / 36** 个已标记单元格的")
    A("  数值与所在行第 0 列不同（且全部位于第 0 列之外）：")
    A("")
    A("| 工作负载 | 已标记 | 其中第 0 列 | 其中非第 0 列 | 数值 ≠ 本行第 0 列 |")
    A("|---|---:|---:|---:|---:|")
    for name in ("ceb", "job", "stack", "dsb"):
        a = dq["initial_mask_audit"][name]
        A(f"| {name} | {a['marked_total']:,} | {a['marked_col0']:,} | {a['marked_off_col0']:,} | "
          f"**{a['marked_differing_from_col0']}** |")
    A("")
    A("  这不证明初始化有问题，但意味着**不能把整个初始掩码解释成默认计划的免费等价传播**。")
    A("  **来源与初始化成本必须明确。**")
    A("")
    A("### 仍未建立")
    A("")
    A("**「没有缺失和无穷」不能证明每一格都是未经截断的原始完成时间。**")
    A("这些是**作者提供的回放数值**；从原始执行记录到这份矩阵的生成过程，仓库里没有说明。")
    A("")
    A("---")
    A("")
    A("## 6. 初始化与等价类的待办边界")
    A("")
    A("本轮的小检查**使用人工给定的等价类**（把每个 hint 视为自成一组）。")
    A("**不通过运行时间相等来推断真实计划等价**——列完全相同的现象与等价结构一致，")
    A("但相等也可能是巧合。本轮**不下载完整计划包、不重建数据库**。")
    A("")
    A("---")
    A("")
    A("## 7. 证据层次（未升级）")
    A("")
    A("| 层次 | 当前判断 |")
    A("|---|---|")
    A("| 存在优化空间 | 有：16.5–23.2% 的单元格优于本行默认计划 |")
    A("| 合法观测可辨识 | **未测**；本轮只把接口做合法并证伪了现成实现 |")
    A("| 学习能够捕捉 | 未测 |")
    A("| 收益超过成本 | 未测 |")
    A("| 相对已有方法有新意 | 未建立；且**「引入重试」已不可作为新功能** |")
    A("")
    A("**当前确定的是一个回放权限问题，以及已有实现允许重试。**")
    A("尚未证明合法重试有实用收益，更没有证明学习比便宜规则更好。")
    A("下一轮才进入成本—质量比较，不再把代码缺陷本身当成研究空间。")
    A("")
    A("---")
    A("")
    A("## 8. 产物")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `Q01_work/q01r_mechanism.py` | 分支真值表、隐式重跑路径、读取点重分类 |")
    A("| `Q01_work/q01r_interface.py` | **最小反馈接口** + 四项确定性检查 |")
    A("| `Q01_work/q01r_data_qualification.py` | 初始掩码审计 + 两层数据资格 |")
    A("| `Q01_work/q01r_checks.json` | 四项检查结果（含被捕获的历史缺陷字段）|")
    A("| `Q01_work/q01r_mechanism.json` | 机制与读取点分类的完整记录 |")
    A("| `Q01_work/q01r_data_qualification.json` | 掩码审计与两层资格 |")
    A("")
    A("冻结源码哈希：")
    A("")
    A("| 文件 | sha256 |")
    A("|---|---|")
    for k, v in src.items():
        A(f"| `{k}` | `{v}` |")
    A("")
    A("数据文件哈希：")
    A("")
    A("| 文件 | 字节 | sha256 |")
    A("|---|---:|---|")
    for k, v in files.items():
        A(f"| `{k}` | {v['bytes']:,} | `{v['sha256']}` |")
    A("")

    (ROOT / "Q01R_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01R_report.md'}")

    cases = {
        "task": "Q01-R -- corrected mechanism and minimal feedback interface for the timeout-retry "
                "screening",
        "supersedes": "Q01_report.md",
        "verdict": "overall verdict retained: the numeric matrices are usable; the stock replay cannot "
                   "be used directly as a legal policy; the mechanism is corrected",
        "retractions": [
            "'cancelled runs are booked as completed' -- my witness injected a deadline the code does "
            "not use; the effective deadline IS timeout_tolerance, so x < cap completing is correct",
            "'there is no retry arm' -- a timed-out cell whose hidden runtime is below the current "
            "best keeps mask=0 and explored_m=0 and can be executed again; the path exists",
            "'nine unmasked reads = nine violations' -- a replay environment must read the hidden "
            "runtime to produce feedback; the correct test is whether the read exceeds what that "
            "execution revealed",
        ],
        "confirmed": {
            "unobservable_candidate_retirement": {
                "site": ["strategies/limeqo.py:89", "strategies/limeqo.py:114",
                         "strategies/limeqo_plus.py:114", "strategies/limeqo_plus.py:139",
                         "strategies/greedy.py:74", "strategies/random.py:60",
                         "strategies/qo_advisor.py:68"],
                "statement": "explored_m is set from a comparison (x >= current best) that the run did "
                             "not reveal; legal feedback is only 'runtime >= cap'",
            },
            "privileged_stopping_rule": {
                "site": ["strategies/limeqo.py:35", "strategies/limeqo.py:105"],
                "statement": "the stop rule reads opt_time = per-row minimum of the FULL hidden "
                             "matrix; usable for post-hoc evaluation, not as a deployable stop "
                             "condition",
            },
            "legitimate_environment_reads": {
                "site": ["strategies/limeqo.py:92", "strategies/limeqo_plus.py:117"],
                "statement": "the completion test for the run just performed at the cap it was given",
            },
            "implicit_retry": {
                "statement": "no separately named retry action, but a path re-executes a timed-out "
                             "candidate, and its eligibility is decided by the hidden runtime",
                "worked_example": "x=5, b=10: cap=2 charges 2s and keeps eligibility; cap=10 then "
                                  "charges 5s and completes; total 7s",
            },
        },
        "interface": {
            "signature": "observe(query, hint, cap) -> kind, value_or_bound, charged_seconds",
            "kinds": ["completed", "cancelled"],
            "boundary": "x < cap completes, x >= cap times out (kept from the official code)",
            "retry_billing": "a re-execution is charged from scratch",
            "policy_holds_matrix": False,
        },
        "checks": checks,
        "data_qualification": dq,
        "screening_space_percent": {k: prof[k]["col0_is_row_min"] / prof[k]["rows"]
                                    for k in ("ceb", "job", "stack", "dsb")},
        "not_done": ["no predictor fitted", "no four-workload algorithm comparison",
                     "no Dropbox download", "no database rebuild",
                     "no equivalence inferred from equal runtimes"],
    }
    (ROOT / "Q01R_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01R_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
