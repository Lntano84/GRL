"""Q01-R2: write the correction report."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> int:
    c = json.loads((HERE / "q01r2_corrections.json").read_text(encoding="utf-8"))
    sites = c["read_sites"]
    legal = [s for s in sites if s["verdict"] == "LEGAL"]
    viol = [s for s in sites if s["verdict"].startswith("VIOLATION")]

    L: list[str] = []
    A = L.append
    A("# Q01-R2 —— 四处修正")
    A("")
    A("> 本文件修正 `Q01R_report.md` 的读取点分类与验收测试；总体裁决不变。")
    A("> 不新增工作负载、不拟合模型、数据数值资格不重验。")
    A("")
    A("## 结论")
    A("")
    A("四处修正全部通过。**最重要的修正是第 1 条：我把表达式形状当成了违规。**")
    A("")
    A("| # | 修正 | 结果 |")
    A("|---|---|---|")
    A("| 1 | 读取点分类：同一个 `x >= b` 在不同截止值下权限不同 | **违规清单收窄**：5 处移出，保留 2 处 + 特权停止另列 |")
    A("| 2 | 状态快照必须带数值，不只带键 | 通过（键快照相等、值快照不等）|")
    A("| 3 | 检查 4 接入真实环境与 runner | 通过（含特权停止的真实对照对）|")
    A("| 4 | 下界取最大值，并验证 `L ≥ b` 可合法剪枝 | 通过 |")
    A("")
    A("---")
    A("")
    A("## 1. 读取点分类修正")
    A("")
    A("**判据：** 在 `cap` 处超时，环境合法揭示的是「`x ≥ cap`」。")
    A("要用它淘汰「`x ≥ b`」的候选，必须**揭示的下界本身已经蕴含** `x ≥ b`。")
    A("")
    A("| 实际截止值 | 超时后是否知道 `x ≥ b` | 据此淘汰是否合法 |")
    A("|---|---|---|")
    A("| `cap < b` | 不知道 | **不合法** |")
    A("| `cap = b` | 知道 | **合法** |")
    A("")
    A("### 1.1 逐个位置")
    A("")
    A("| 位置 | 实际截止值 | 揭示 | 蕴含 `x ≥ b` | 判定 |")
    A("|---|---|---|---|---|")
    for s in sites:
        A(f"| `{s['site']}` | {s['cap']} | {s['revealed']} | {s['implies_x_ge_b']} | "
          f"{'**违规**（cap<b 时）' if s['verdict'].startswith('VIOLATION') else '**合法**'} |")
    A("")
    A("**移出违规清单（我上一轮判错）：**")
    A("")
    for s in legal:
        A(f"- `{s['site']}` —— 截止值**就是**当前最好值 b，超时本身即揭示 `x ≥ b`")
    A("")
    A("这五处是随机补足/贪心/随机策略路径，它们**读取真值来产生反馈是合法的环境模拟**。")
    A("")
    A("**确认违规（收窄后）：**")
    A("")
    A("- `limeqo.py:89`、`limeqo_plus.py:114` —— **预测驱动路径**的容忍度 `min(α·b, β·pred)`")
    A("  可能**低于**当前最好值 b；此时用隐藏真值淘汰候选，超出了该次执行揭示的信息。")
    A("- `limeqo.py:35`、`limeqo.py:105` —— **特权停止规则**，另列（见 §2）。")
    A("")
    A("**因此问题应限定为：**")
    A("")
    A("> **LimeQO / LimeQO+ 的预测驱动路径在 `cap < b` 时，额外利用隐藏真值决定候选资格。**")
    A("")
    A("**不得再按表达式形状扩张违规清单。**")
    A("")
    A("---")
    A("")
    A("## 2. 特权停止规则（仍单列）")
    A("")
    A("本轮不再只是「声明它会失败」，而是给出**一个它确实会给出不同判断的真实对照对**：")
    A("")
    A("规则：`while min_observed.sum() > opt_time + 20`（`limeqo.py:35`）。")
    A("`min_observed` 由**掩码后**的矩阵计算，因此两个世界相同；")
    A("`opt_time` 是**完整隐藏矩阵**的逐行最小值，因此可以不同。")
    A("")
    A("| 世界 | `min_observed.sum()` | `opt_time` | 继续探索？ |")
    A("|---|---:|---:|:--:|")
    for r in c["check4"]["privileged_stop_demo"]:
        A(f"| {r['world']} | {r['min_observed_sum']:.0f} | {r['opt_time']:.0f} | "
          f"**{r['continue']}** |")
    A("")
    A("两行合法反馈完全相同，停止判断却相反 —— 这就是特权停止条件被暴露的方式。")
    A("")
    A("---")
    A("")
    A("## 3. 状态快照修正")
    A("")
    A("旧实现只保存字典的**键**：")
    A("")
    A("```python")
    A('"mask": sorted(self.mask),      # 只有键'),
    A('"bounds": sorted(self.bounds),  # 只有键'),
    A("```")
    A("")
    A("这正是你举的反例：一个策略保存完成值 5、下界 2，另一个保存完成值 9、下界 8，")
    A("旧快照仍判定相等。修正后：")
    A("")
    A("```python")
    A('"mask": sorted((list(k), v) for k, v in self.mask.items()),'),
    A('"bounds": sorted((list(k), v) for k, v in self.bounds.items()),'),
    A("```")
    A("")
    A("并且**验收快照中的成本不再四舍五入**（`spent` 保持原值）。")
    A("")
    A("实测：")
    A("")
    A(f"- 键快照相等：**{c['snapshot']['keys_only_equal']}**（旧缺陷）")
    A(f"- 值快照相等：**{c['snapshot']['value_snapshot_equal']}**")
    A(f"- 修正后的快照能抓住键快照漏掉的差异：**{c['snapshot']['pass']}**")
    A("")
    A("---")
    A("")
    A("## 4. 检查 4 接入真实环境")
    A("")
    A("旧版本只比较了两份手写常量。现在两份**完整小矩阵**经过**同一个 runner**：")
    A("")
    A("```")
    A("runner(truth, best, plan):  env = ReplayEnvironment(...); pol = LegalPolicy()")
    A("                            for (q,h,cap) in plan: observe -> record")
    A("```")
    A("")
    A("两份矩阵只差一个未揭示的单元格 `(0,2)`：世界 A 为 7.0，世界 B 为 900.0。")
    A("")
    A("| 量 | 结果 |")
    A("|---|---|")
    A(f"| 揭示的反馈逐条相同 | **{c['check4']['check4']['feedback_identical']}**（两条都是 `(0,0)@cap10` 超时、`(0,1)@cap2` 超时）|")
    A(f"| 下一动作 | A={c['check4']['check4']['next_action_A']}，B={c['check4']['check4']['next_action_B']} → **相同** |")
    A(f"| 固定预算停止判断 | A={c['check4']['check4']['stop_A']}，B={c['check4']['check4']['stop_B']} → **相同** |")
    A(f"| 策略状态 | 相同 = **{c['check4']['check4']['state_identical']}** |")
    A("")
    A("这才证明了原要求：**从两份隐藏数据经过实际回放入口运行，策略动作与预算停止判断保持不变。**")
    A("")
    A("原先那两份手写常量只证明了「给简单策略相同的手写输入会得到相同动作」——这是不够的。")
    A("")
    A("---")
    A("")
    A("## 5. 下界取最大值与合法剪枝")
    A("")
    A("旧实现每次超时直接覆盖：`self.bounds[(q,h)] = val`，")
    A("所以「先截止 10 秒、再截止 2 秒」会把下界从 10 降成 2。修正为：")
    A("")
    A("$$L_{qh} \\leftarrow \\max(L_{qh},\\ cap)$$")
    A("")
    A("实测：")
    A("")
    A(f"- 先 `cap=10`（收费 10）再 `cap=2`（收费 2），存储下界保持 **{c['bounds']['bound_after_caps_10_then_2']}**")
    A(f"- 下界保持最大值：**{c['bounds']['bound_kept_max']}**")
    A("")
    A("**并且补充了一条我上一轮漏掉的合法剪枝。**")
    A("「超时不能判断是否超过当前最好值」**只适用于 `cap < b`**：")
    A("若已有下界 `L_{qh} ≥ b_q`，该候选**已不能直接改善当前最好值**，可以合法剪枝。")
    A("")
    A("| 情形 | 剪枝 |")
    A("|---|---|")
    A(f"| `(0,0)` 揭示 b=10；`(0,1)` 下界 L=10 ≥ b=10 | **合法剪枝** = {c['bounds']['prune_legal_when_L_ge_b']} |")
    A(f"| `(0,1)` 仅在 `cap=2` 超时，L=2 < b=10 | **拒绝剪枝** = {c['bounds']['prune_refused_when_L_below_b']} |")
    A("")
    A("**这一条对后续基线很重要：** 后续基线必须利用这个合法剪枝。")
    A("否则我们可能人为制造大量无效重试，再把「避免它们」包装成学习优势。")
    A("")
    A("若将来要研究「执行更差计划以获取跨查询信息」，那应作为**另一种明确动作与目标**，")
    A("不能混在这里。")
    A("")
    A("---")
    A("")
    A("## 6. 研究定位再收窄")
    A("")
    A("「何时重跑」**不是**原实现完全没有决策：它已有候选资格、预测排序、截止值和随机补足")
    A("共同形成的**隐式规则**，只是其中包含信息权限问题。")
    A("")
    A("所以问题应写成：")
    A("")
    A("> **在仅使用合法反馈的条件下，重试与探索新候选如何分配预算，")
    A("> 能否超过已有预测规则和廉价重试规则？**")
    A("")
    A("**不能写成「现有方法没有考虑重试」。**")
    A("")
    A("---")
    A("")
    A("## 7. 状态")
    A("")
    A("| 项 | 状态 |")
    A("|---|---|")
    A("| 数据数值资格（L1） | 通过，**本轮不重验** |")
    A("| 权限缺陷 | **无需重新发现**；本轮只收窄了清单 |")
    A("| 接口 | 已可用于成本—质量比较（下界保持 + 合法剪枝已补齐）|")
    A("| 选题证据 | **没有升级**：仍无证据支持或否定，也没有启动 GRL 的依据 |")
    A("")
    A("---")
    A("")
    A("## 8. 产物")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `Q01_work/q01r2_corrections.py` | 四处修正的全部代码与验收 |")
    A("| `Q01_work/q01r2_corrections.json` | 分类表、快照对照、真实 runner 检查、下界与剪枝结果 |")
    A("| `Q01R_report.md` | 上一版（读取点分类已被本文件 §1 取代）|")
    A("")

    (ROOT / "Q01R2_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01R2_report.md'}")

    cases = {
        "task": "Q01-R2 -- four corrections to the read-site classification and the acceptance tests",
        "supersedes_in_part": "Q01R_report.md (its read-site classification)",
        "verdict_unchanged": True,
        "corrections": {
            "1_read_sites": {
                "rule": "a timeout at cap reveals only 'x >= cap'; retiring a cell because 'x >= b' "
                        "is legal only when the revealed bound already implies it",
                "moved_out_of_violations": [s["site"] for s in legal],
                "reason_moved": "these paths use cap = current best, so the timeout itself reveals "
                                "x >= b",
                "confirmed_violations": ["strategies/limeqo.py:89",
                                         "strategies/limeqo_plus.py:114"],
                "confirmed_violation_statement": "the prediction-driven path of LimeQO / LimeQO+ "
                                                 "uses a tolerance that may sit below the current "
                                                 "best, and then retires the cell using the hidden "
                                                 "runtime",
                "listed_separately": ["strategies/limeqo.py:35", "strategies/limeqo.py:105"],
                "do_not_grow_from_expression_shape": True,
            },
            "2_state_snapshot": c["snapshot"],
            "3_check4_real_runner": c["check4"],
            "4_lower_bound": c["bounds"],
        },
        "research_positioning": "under legal feedback only, how should budget be split between "
                                "retrying and exploring new candidates, and can that beat both the "
                                "existing prediction rule and cheap retry rules?  It must NOT be "
                                "phrased as 'existing methods do not consider retry'",
        "not_redone": ["data numeric qualification (L1)", "rediscovery of the permission defect"],
        "evidence_level": "unchanged: no new evidence for or against the topic, and no basis to "
                          "start GRL",
    }
    (ROOT / "Q01R2_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'Q01R2_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
