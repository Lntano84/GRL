"""Q02: write the report and frozen artifacts."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest().upper()


def main() -> int:
    r = json.loads((HERE / "q02_results.json").read_text(encoding="utf-8"))
    why = json.loads((HERE / "q02_retry_why.json").read_text(encoding="utf-8"))
    rows, D, seeds, arms, fracs = r["rows"], r["D"], r["seeds"], r["arms"], r["budget_fracs"]

    def g(frac, arm, seed, key):
        return [x for x in rows if x["budget_frac"] == frac and x["arm"] == arm
                and x["seed"] == seed][0][key]

    mb = [g(1.0, "MF-BEST", s, "P") for s in seeds]
    mr = [g(1.0, "MF-RETRY", s, "P") for s in seeds]
    rel = [(mb[i] - mr[i]) / mb[i] for i in range(len(seeds))]
    mean_rel = sum(rel) / len(rel)

    L: list[str] = []
    A = L.append
    A("# Q02 —— JOB 上的四臂成本—质量筛查")
    A("")
    A("## 结论")
    A("")
    A("**预先固定的门槛未通过：三个种子中只有 2 个更好，等权平均 +1.79%，低于 2%。**")
    A("")
    A("| 门槛条件 | 结果 |")
    A("|---|---|")
    A("| MF-RETRY 在三个种子均更好 | **否**（seed 0 为 +0.00%）|")
    A("| 平均相对 MF-BEST 降低 ≥ 2% | **否**（**+1.79%**）|")
    A("| 优于两个廉价对照 | **是**（三个种子都优于 RANDOM-BEST 与 GREEDY-BEST）|")
    A("| **合计** | **未通过** |")
    A("")
    A("按固定判读表，本轮落在**「方向混合或差距小：记为不确定，不改倍率、不追加种子」**。")
    A("")
    A("**并且有一个决定性的事实必须写在结论旁边：**")
    A("在这份冻结配置下，重试分支**几乎完全没有被触发**——")
    A("MF-RETRY 在 239 次探索动作中只做了 **1 次**重试。所以这轮**实际上没有测到「重试」这个机制**，")
    A("它测到的是「提前截止但不重试」。我不把它写成「场景不够难」，也不写成方向被否定。")
    A("")
    A("---")
    A("")
    A("## 1. 冻结配置与两处必须说明的执行偏差")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 工作负载 | JOB：113 查询 × 49 hint |")
    A("| 等价类 | 来自 EXPLAIN 计划的 `hint_list`；**113/113 行标签全部匹配**，0 冲突，1,393/1,393 个多 hint 类数值一致 |")
    A(f"| D | Σ W[q,0] = **{D:,.4f}** |")
    A(f"| 预算 | 0.25D / 0.5D / D = {', '.join(f'{f*D:,.2f}' for f in fracs)} |")
    A(f"| 种子 | {seeds}；四臂 × 三预算 × 三种子 = **{len(rows)} 次回放** |")
    A(f"| 预测器 | `censored_als(rank={r['predictor']['rank']}, lambda={r['predictor']['lambda']}, iters={r['predictor']['iters']})`，未扫参 |")
    A(f"| 截止值 | 新候选 `{r['caps']['new']}`；重试 `{r['caps']['retry']}` |")
    A("")
    A("### 1.1 执行偏差（必须明说）")
    A("")
    A("**第一，初始测量成本单列，不占用探索预算。**")
    A(f"第 0 列基线本身花费 **恰好 D = {D:,.4f}**。若把它记入预算，则 `B = 0.25D` 会被初始化一次耗尽，")
    A("四臂退化成同一条轨迹（我第一版实现正是如此，四臂 P 完全相同——这是你指出的「接口还没接好」的直接后果）。")
    A("因此本轮把 `init_cost` 作为**所有方法相同的独立行项**单独计费，预算 B 只用于其上的探索。")
    A(f"每次回放的合计模拟耗时 = init_cost({D:,.4f}) + B。")
    A("")
    A("**第二，预算的货币是模拟执行秒数，策略计算耗时（wall-clock）单列。**")
    A("把 wall-clock 秒加进模拟秒会混合单位，因此没有相加，而是并列报告（§5）。")
    A("")
    A("**其它均为冻结配置：** 初始化只用第 0 列及其等价类（不使用来源未解释的 `init_mask`）；")
    A("同一等价类内一次观测传播给整类、只收一次执行成本；所有方法排除已完成候选与 `L ≥ b_q` 的候选；")
    A("非正/非有限预测不得产生零成本动作，一律进入统一随机回退并计数；")
    A("派发前检查剩余预算，末次截止值不超过剩余预算，**不使用真实耗时预判「这次来得及完成」**。")
    A("")
    A("---")
    A("")
    A("## 2. 主比较（满预算 B = D）")
    A("")
    A("| 种子 | MF-BEST | MF-RETRY | 相对变化 | RANDOM-BEST | GREEDY-BEST |")
    A("|---:|---:|---:|---:|---:|---:|")
    for i, s in enumerate(seeds):
        A(f"| {s} | {mb[i]:.4f} | {mr[i]:.4f} | **{rel[i]:+.2%}** | "
          f"{g(1.0,'RANDOM-BEST',s,'P'):.4f} | {g(1.0,'GREEDY-BEST',s,'P'):.4f} |")
    A(f"| **均值** | **{sum(mb)/3:.4f}** | **{sum(mr)/3:.4f}** | **{mean_rel:+.2%}** | "
      f"{sum(g(1.0,'RANDOM-BEST',s,'P') for s in seeds)/3:.4f} | "
      f"{sum(g(1.0,'GREEDY-BEST',s,'P') for s in seeds)/3:.4f} |")
    A("")
    A("四臂相对 MF-BEST 的相对变化（正值表示更好）：")
    A("")
    A("| 臂 | seed 0 | seed 1 | seed 2 | 均值 |")
    A("|---|---:|---:|---:|---:|")
    for arm in arms:
        v = [g(1.0, arm, s, "P") for s in seeds]
        rr = [(mb[i] - v[i]) / mb[i] for i in range(3)]
        A(f"| {arm} | {rr[0]:+.2%} | {rr[1]:+.2%} | {rr[2]:+.2%} | **{sum(rr)/3:+.2%}** |")
    A("")
    A("**MF-RETRY 与 MF-BEST 在 seed 0 完全相同**，这正是「重试几乎没被触发」的直接表现。")
    A("")
    A("---")
    A("")
    A("## 3. 成本曲线 P(B)/D（三种子均值）")
    A("")
    A("| 预算 | RANDOM-BEST | GREEDY-BEST | MF-BEST | MF-RETRY |")
    A("|---|---:|---:|---:|---:|")
    for frac in fracs:
        cells = []
        for arm in arms:
            v = [g(frac, arm, s, "P_over_D") for s in seeds]
            cells.append(f"{sum(v)/3:.4%}")
        A(f"| {frac:.2f}D | " + " | ".join(cells) + " |")
    A("")
    A("两档较低预算只是成本曲线诊断；主比较是满预算那一档。")
    A("可以看到**收益递减**：预算翻倍带来的 P 下降在 0.5D→1.0D 段明显小于 0.25D→0.5D 段。")
    A("")
    A("---")
    A("")
    A("## 4. 为什么重试没有触发（本轮最关键的事实）")
    A("")
    A("| 量（B = D，三种子合计） | MF-RETRY |")
    A("|---|---:|")
    A("| 探索动作 | 239 |")
    A("| **其中重试动作** | **1** |")
    A("| 发生超时 | 144 |")
    A("| 随机回退次数 | 3 |")
    A("")
    A("重试并非不可行：三种子合计有 **185 个（类别, 决策）对**是「重试可行」的。")
    A("但按该臂**自己的打分**，这些类别排得极低：")
    A("")
    A("| 种子 | 决策数 | 含重试可行类的决策 | 由重试类居首的决策 |")
    A("|---:|---:|---:|---:|")
    for s in seeds:
        ds = why[f"MF-RETRY_{s}"]
        A(f"| {s} | {len(ds)} | {sum(1 for d in ds if d['n_retry_feasible'] > 0)} | "
          f"{sum(1 for d in ds if d['top_is_retry'])} |")
    A("")
    A("典型情形（seed 1，决策 3）：1,788 个候选中有 1 个重试可行类，它排在第 **63** 位。")
    A("")
    A("### 4.1 根因")
    A("")
    A("超时产生的下界不可能超过该次的截止值，而截止值至多为 `b_q`。因此：")
    A("")
    A("> 一个超时类要保持 `L < b_q`（即重试可行），只有当它的截止值**严格小于** `b_q`。")
    A("")
    A("而冻结的截止规则是 `cap_new = min(b_q, 15 x̂)`。在 JOB 上，第 0 列基线把多数 `b_q` 压得较小，")
    A("于是 `15 x̂ ≥ b_q` 经常成立、`cap_new` 就等于 `b_q`；这类超时一发生就落在 `L ≥ b_q`，")
    A("按既定规则被**合法剪枝**，永远不再成为候选。")
    A("")
    f"实测（B = D，seed 0）：25 次探索性超时中，**23 次**的下界达到了 `b_q`（合法剪枝），"
    A("只有 2 次留下重试可行的下界。")
    A("")
    A("**换句话说：在这份冻结配置下，「提前截止」这一半在起作用，而「重试」这一半被 α = 15 的截止规则")
    A("本身挤掉了。** 这是配置的后果，不是方法的评价。")
    A("")
    A("---")
    A("")
    A("## 5. 预算核算")
    A("")
    A(f"`init_cost` 四臂相同：**{rows[0]['init_cost']:.4f}**（独立行项，不计入 B）")
    A("")
    A("| 预算 | 臂 | 平均模拟执行 | 预算 | 超支 | 平均策略墙钟 | 平均总墙钟 | 回退 |")
    A("|---|---|---:|---:|:--:|---:|---:|---:|")
    for frac in fracs:
        for arm in arms:
            sub = [x for x in rows if x["budget_frac"] == frac and x["arm"] == arm]
            over = sum(1 for x in sub if x["exec_seconds"] > sub[0]["B"] + 1e-9)
            A(f"| {frac:.2f}D | {arm} | "
              f"{sum(x['exec_seconds'] for x in sub)/len(sub):.4f} | {sub[0]['B']:.4f} | "
              f"{'**是**' if over else '否'} | "
              f"{sum(x['policy_seconds'] for x in sub)/len(sub):.2f}s | "
              f"{sum(x['wall_seconds'] for x in sub)/len(sub):.2f}s | "
              f"{sum(x['fallbacks'] for x in sub)} |")
    A("")
    A("所有回放都**没有超支**，且非正/非有限预测确实进入了随机回退并计数（MF 两臂各 3 次）。")
    A("")
    A("---")
    A("")
    A("## 6. 判读")
    A("")
    A("| 预先固定的档位 | 是否命中 |")
    A("|---|---|")
    A("| 三种子均更好 + 平均 ≥2% + 优于两个廉价对照 → 进入第二工作负载确认 | **否** |")
    A("| MF-BEST 三种子均不差，或廉价对照稳定更好 → 不扩展该方案 | **否**：廉价对照并未更好（三个种子都更差）|")
    A("| **方向混合或差距小 → 记为不确定，不改倍率、不追加种子** | **本轮落在这里** |")
    A("| 只有隐藏真值诊断发现改善空间 → 不算通过 | 未使用该档 |")
    A("")
    A("### 6.1 本轮真正测到了什么")
    A("")
    A("**必须和门槛结果分开写：**")
    A("")
    A("1. 在 JOB、这份初始化与这三档预算下，**全局经验中位数基线无法容纳**「提前截止＋重试」组合的净收益")
    A("   —— 但**主要原因是重试分支被截止规则挤掉**，不是因为发现了重试无价值。")
    A("2. 「提前截止」那一半确实在工作：MF-BEST 在三个种子都优于 GREEDY-BEST，")
    A("   说明按预测改善/执行成本排序比「优先最大 b_q」更省预算。")
    A("3. 这**不是**对方向的否定：我**没有**得到「重试无用」的证据，因为重试只发生了 1 次。")
    A("")
    A("### 6.2 我明确不做的")
    A("")
    A("按你的固定判读，**不改倍率、不追加种子**。因此我**不**去调 α = 15 来「让重试发生」——")
    A("那正是你警告过的「不断修改场景」。若要让重试可被检验，应当**重新预注册**一个不同的截止规则，")
    A("并把「重试发生率」当作**设计参数**在跑之前固定。")
    A("")
    A("---")
    A("")
    A("## 7. 限制")
    A("")
    A("1. **单个工作负载（JOB）**，三种子**不构成跨工作负载泛化证据**，也不构成统计显著性。")
    A("2. **2% 是筛查门槛**，不是显著性或论文标准。")
    A("3. **重试发生率 0.4%**，所以主比较对「重试」这一半几乎没有检验力（§4）。")
    A("4. **初始化作为独立行项**，总模拟耗时 = init_cost + B（§1.1）。")
    A("5. **预算货币是模拟秒**，策略墙钟单列，未相加。")
    A("6. **删失低秩预测器是拟合过的轻量模型**；本轮报告应表述为「拟合了固定的删失低秩预测器」，")
    A("   **不得**写成「没有训练任何模型」。")
    A("7. 所有方法共用一个预测器与一个排序规则，因此本轮不分解「重试的单独因果贡献」。")
    A("")
    A("---")
    A("")
    A("## 8. 产物")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `Q02_work/q02_build_mapping.py` | 从 EXPLAIN 计划构建 JOB 等价映射（按 CSV 行标签连接）|")
    A("| `Q02_work/q02_experiment.py` | 预算受控搜索循环 + 四臂 + 三预算 × 三种子 |")
    A("| `Q02_work/q02_diagnose.py` | 重试发生率诊断 |")
    A("| `Q02_work/q02_retry_why.py` | 重试为何不触发的逐步归因 |")
    A("| `Q02_work/q02_results.json` | 36 次回放的全部数值 |")
    A("| `Q02_work/q02_retry_why.json` | 每次决策的候选数、重试可行数与排名 |")
    A("| `Q02_work/job_equivalence.json` | 冻结的等价映射 |")
    A("")
    A("逐次动作（查询、类别成员、截止值、反馈、收费、累计成本）保存在每次 `Replay.actions` 中，")
    A("可通过 `q02_experiment.py` 的 `Replay` 重放得到。")
    A("")
    A("哈希：")
    A("")
    A("| 文件 | sha256 |")
    A("|---|---|")
    for p in (HERE / "job_equivalence.json", HERE / "q02_results.json",
              ROOT / "Q01_work" / "limeqo_mirror" / "dataset" / "job-matrix.csv"):
        if p.exists():
            A(f"| `{p.name}` | `{sha(p)}` |")
    A("")

    (ROOT / "Q02_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'Q02_report.md'}")

    cases = {
        "task": "Q02 -- one four-arm cost-quality screening on JOB",
        "question": "can prediction-driven early cutoff plus retry beat running straight to the "
                    "current best, once all re-run costs are charged?",
        "gate": r["gate"],
        "gate_not_met": "2 of 3 seeds better; mean +1.79% < 2%",
        "reading": "mixed direction / small difference -> record as undetermined; do not change the "
                   "multiplier, do not add seeds",
        "critical_fact": "the retry branch fired 1 time in 239 exploration actions, so this round did "
                         "not test retry; it tested early cutoff without retry",
        "why_retry_barely_fires": {
            "mechanism": "a timeout cannot reveal a bound above its cap, and the cap is at most b_q; "
                         "so a timed-out class stays retry-feasible only if its cap was strictly "
                         "below b_q",
            "frozen_cap": "cap_new = min(b_q, 15 * x_hat), so on JOB the cap frequently equals b_q "
                          "and the timeout is then legally pruned",
            "measured": "at B = D, seed 0: of 25 exploratory timeouts, 23 reached b_q and were pruned "
                        "legally, 2 left a retry-feasible bound",
            "retry_feasible_pairs": 185,
            "topped_by_a_retry": 1,
            "example_rank": "seed 1 decision 3: 1 retry-feasible class among 1788 candidates, ranked 63",
        },
        "execution_deviations": {
            "init_cost_separate": f"the column-0 baseline costs exactly D = {D:.4f}; charging it to B "
                                  f"would exhaust the 0.25D budget and make all four arms identical "
                                  f"(the first implementation did exactly that).  It is therefore "
                                  f"charged as a separate line item identical across arms, and the "
                                  f"budget B applies to exploration on top of it.  Aggregate "
                                  f"simulated spend = init_cost + B.",
            "budget_currency": "simulated execution seconds; wall-clock policy time is measured and "
                               "reported separately because adding it would mix units",
        },
        "protocol": {
            "workload": "JOB", "queries": 113, "hints": 49, "D": D,
            "equivalence": "from EXPLAIN hint_list, joined on CSV row labels; 113/113 matched, "
                           "0 conflicts, 1393/1393 multi-hint classes have identical matrix values",
            "init": "column 0 and its equivalence class only; the unexplained init_mask is unused",
            "budgets": [f * D for f in fracs], "budget_fracs": fracs, "seeds": seeds, "arms": arms,
            "predictor": r["predictor"], "caps": r["caps"],
            "replays": len(rows),
            "prune": "completed candidates and L >= b_q are excluded by every arm",
            "fallback": "a non-positive or non-finite prediction may not produce a zero-cost action; "
                        "the arm draws a uniformly random feasible candidate and the count is logged",
        },
        "cost_curve_P_over_D_mean": {f"{f}": {arm: sum(g(f, arm, s, "P_over_D") for s in seeds) / 3
                                              for arm in arms} for f in fracs},
        "relative_vs_MF_BEST_full_budget": {
            arm: sum((mb[i] - g(1.0, arm, s, "P")) / mb[i] for i, s in enumerate(seeds)) / 3
            for arm in arms},
        "not_concluded": ["that retry is useless (it fired once)",
                          "any cross-workload generalisation",
                          "statistical significance",
                          "that the direction is invalid"],
        "declined": "the multiplier was not changed and no seeds were added, per the pre-registered "
                    "reading; making retry testable would require a NEW pre-registration with the "
                    "retry-incidence rate fixed as a design parameter before running",
    }
    (ROOT / "Q02_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'Q02_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
