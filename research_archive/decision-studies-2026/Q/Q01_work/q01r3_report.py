"""Q01-R3: write the witness-correction note (folded into the next step, not a standalone round)."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> int:
    w = json.loads((HERE / "q01r3_witness_corrections.json").read_text(encoding="utf-8"))
    ineq = w["inequality"]
    ws = w["valid_stop_witness"]
    prev = w["previous_check4_actual"]

    L: list[str] = []
    A = L.append
    A("# Q01-R3 —— 两个见证的更正（并入下一步，不单独成轮）")
    A("")
    A("> 更正 `Q01R2_report.md` 的两处见证。核心规则、权限机制排查、读取点分类均不变。")
    A("")
    A("## 结论")
    A("")
    A("| 项 | 处置 |")
    A("|---|---|")
    A("| `opt_time = 260` / `min_observed.sum() = 200` 的对照 | **撤回**，违反 `Σ min_h W ≤ Σ b`；换用有效两世界 |")
    A("| 原检查 4 的「策略已知 b = 10」场景 | **撤回**，该次运行实际超时，策略从未收到该测量 |")
    A("| 预算耗尽时的动作 | 明确为 **STOP**；「若预算允许的下一候选」另列 |")
    A("")
    A("---")
    A("")
    A("## 1. 为什么 260 不可能")
    A("")
    A("任何已观察的最好值 `b_q` 都是该行**子集**上的最小值，因此不可能低于整行最小值：")
    A("")
    A("$$\\sum_q \\min_h W_{qh} \\;\\le\\; \\sum_q b_q$$")
    A("")
    A("在四张真实矩阵上核验：")
    A("")
    A("| 工作负载 | `Σ min_h W`（完整矩阵最优） | `Σ b`（默认计划和） | 不等式成立 |")
    A("|---|---:|---:|:--:|")
    for k, v in ineq.items():
        A(f"| {k} | {v['full_optimum']:,.2f} | {v['default_sum']:,.2f} | **{v['holds']}** |")
    A("")
    A("所以「已观察最好值之和 200、完整矩阵最优 260」是**手填标量**、不可能出现。该对照撤回。")
    A("")
    A("---")
    A("")
    A("## 2. 有效的特权停止见证")
    A("")
    A("$$W_A=\\begin{bmatrix}100&50\\\\100&50\\end{bmatrix},\\qquad")
    A("W_B=\\begin{bmatrix}100&99\\\\100&99\\end{bmatrix}$$")
    A("")
    A("两个世界都**只执行第 0 列，截止值 101**（`x=100 < 101`，因此确实完成）：")
    A("")
    A("| 量 | 世界 A | 世界 B |")
    A("|---|---:|---:|")
    A(f"| 已完成观测 | 100、100 | 100、100 |")
    A(f"| 已观察最好值之和 | {ws['A']['min_observed_sum']:.0f} | {ws['B']['min_observed_sum']:.0f} |")
    A(f"| 从**完整矩阵**算出的最优值 | {ws['A']['opt_time']:.0f} | {ws['B']['opt_time']:.0f} |")
    A(f"| 原规则 `Σb > opt_time + 20` | **{ws['A']['continue']}** | **{ws['B']['continue']}** |")
    A(f"| 合法策略下一候选 | `(0,1)@cap2` | `(0,1)@cap2` |")
    A("")
    A(f"- 揭示的历史完全相同：**{w['witness_checks']['identical_history']}**")
    A(f"- 特权停止判断不同：**{w['witness_checks']['stop_differs']}**")
    A(f"- 合法策略的下一候选相同：**{w['witness_checks']['legal_next_identical']}**")
    A("")
    A("**这才是「同样的合法历史、不同的特权停止判断」。** 停止规则确实有问题，但原来的 260 不能作为证据。")
    A("")
    A("---")
    A("")
    A("## 3. 原检查 4 的实际产出")
    A("")
    A("原代码实际执行的是 `x = 10, cap = 10`。沿用冻结的 `x < cap` 才算完成规则，")
    A("`10 < 10` 为假，**该次运行超时**：")
    A("")
    A("| 执行 | 反馈 | 收费 |")
    A("|---|---|---:|")
    for c in prev["calls"]:
        A(f"| `({c['q']},{c['h']})` cap={c['cap']:.0f} | `{c['kind']}` | {c['charged']:.0f} |")
    A("")
    A(f"- 策略的 `best_get(0)` = **{prev['policy_best_get_0']}**（仍是未知，不是 10）")
    A(f"- 累计成本 = **{prev['spent']:.0f}**")
    A("")
    A("环境的 `best` 虽然预设了 10，**策略却没有收到这份已完成测量**。")
    A("因此「状态相同」成立，但**不是**报告暗示的「策略已知当前最好值为 10」的场景。该场景撤回。")
    A("")
    A("### 3.1 预算纪律")
    A("")
    A(f"- 预算 12，已花 {prev['spent']:.0f}，剩余 0 → **动作是 STOP**")
    A(f"- 【另列，不作为实际动作】若预算允许，下一候选会是 `{prev['next_if_budget_remained']}` "
      f"@ cap {prev['next_cap_if_budget_remained']:.0f}")
    A("")
    A("`STOP` 是实际动作；「若预算允许」只是诊断信息，两者不得混写。")
    A("")
    A("---")
    A("")
    A("## 4. 本轮验收后的状态")
    A("")
    A("| 项 | 状态 |")
    A("|---|---|")
    A("| 读取点分类 | **接受** |")
    A("| 快照包含数值、下界保持、合法剪枝 | **实现接受** |")
    A("| 特权停止与检查 4 的见证 | **已替换为有效例子** |")
    A("| 可用于成本比较 | **尚未**：还需接好初始化、等价传播与预算调度；现有 `runner` 只是按给定动作列表执行，不是预算受控的搜索循环 |")
    A("")
    A("---")
    A("")
    A("## 5. 产物")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `Q01_work/q01r3_witnesses.py` | 不等式核验、有效两世界见证、原检查 4 的实际产出与预算纪律 |")
    A("| `Q01_work/q01r3_witness_corrections.json` | 上述全部结果 |")
    A("")

    (ROOT / "Q01R3_witnesses.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01R3_witnesses.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
