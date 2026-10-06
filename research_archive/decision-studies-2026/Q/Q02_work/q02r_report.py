"""Q02R: write the corrected report and artifacts."""
from __future__ import annotations

import hashlib
import json
import statistics
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
    r = json.loads((HERE / "q02r_results.json").read_text(encoding="utf-8"))
    rows, D, seeds, arms, fracs = r["rows"], r["D"], r["seeds"], r["arms"], r["budget_fracs"]

    def g(frac, arm, seed, key):
        return [x for x in rows if x["budget_frac"] == frac and x["arm"] == arm
                and x["seed"] == seed][0][key]

    mb = [g(1.0, "MF-BEST", s, "P") for s in seeds]
    mr = [g(1.0, "MF-RETRY", s, "P") for s in seeds]
    rel = [(mb[i] - mr[i]) / mb[i] for i in range(len(seeds))]
    mean_rel = statistics.fmean(rel)
    total_retry = sum(x["retry_actions"] for x in rows)

    L: list[str] = []
    A = L.append
    A("# Q02R —— 协议修正后的四臂成本—质量筛查")
    A("")
    A("> 取代 `Q02_report.md` 的主比较。旧结果保留为**实现缺陷版**；数据与等价映射继续复用。")
    A("> 未改数据、未改预测器、未改倍率、未加种子、未进入第二工作负载。")
    A("")
    A("## 结论")
    A("")
    A(f"**门槛仍未通过：三个种子中 2 个更好，等权平均 +3.73%，但 seed 2 为 −0.33%。**")
    A("")
    A(f"**并且修正协议后，重试次数是 {total_retry}。** 不是「几乎没触发」，而是**一次都没有触发**。")
    A("")
    A("| 门槛条件 | 结果 |")
    A("|---|---|")
    A(f"| MF-RETRY 在三个种子均更好 | **否**（seed 2 = {rel[2]:+.2%}）|")
    A(f"| 平均相对 MF-BEST 降低 ≥ 2% | **是**（**{mean_rel:+.2%}**）|")
    A("| 优于两个廉价对照 | **是** |")
    A("| **合计** | **未通过** |")
    A("")
    A("按固定判读表：**方向混合 → 记为不确定，不改倍率、不追加种子。**")
    A("按你要求的收口方式：**如实接受整个策略的成绩。**")
    A("")
    A("---")
    A("")
    A("## 1. 撤回清单（Q02 的实现缺陷）")
    A("")
    A("| # | 上一轮的说法 | 更正 |")
    A("|---|---|---|")
    A("| 1 | 「两臂同一个排序规则」 | **撤回。** MF-BEST 用 `min(b_q, x̂)` 作分母，MF-RETRY 用受剩余预算影响的 `cap`。**即使零次重试，两臂也会选到不同的新候选** |")
    A("| 2 | 「本轮只测到了提前截止但不重试」 | **撤回。** 两臂**都**在提前截止：`cap_for()` 对新候选一律返回 `min(b_q, 15x̂)`，所以 MF-BEST 也被传入了 x̂。准确表述是「同时改变了候选资格和排序规则，且两臂均提前截止」|")
    A("| 3 | 把「239 次探索、1 次重试」当作主预算数据 | **撤回。** 239/1 是**全部三档预算合计**。主预算（B = D）单独是 158 次探索、1 次重试 |")
    A("| 4 | 「23/25 次超时运行到了 b_q」 | **撤回。** 那是用**运行结束时**的下界与最好值算的，只能说明这些候选**最终**满足剪枝条件，不能证明每次超时**当时**就达到了 b_q |")
    A("| 5 | 把初始化成本单列称为「执行偏差」 | **更正：那是你上一轮明确写出的协议，不是偏差。** 第一版把它算进探索预算是我的实现错误 |")
    A("| 6 | 「模拟秒与墙钟秒相加会混合单位」 | **撤回。** 两者单位都是秒。约定的是「数据库执行时间回放值 ＋ 本机控制策略耗时」这一成本模型 |")
    A("| 7 | 把 0.02–0.19 秒称为全部策略开销 | **撤回。** 当时只计时了 ALS，未计入候选生成与排序 |")
    A("| 8 | `fallbacks` 计数 | **撤回。** 它统计的是「出现过无效预测的决策次数」，不是实际执行随机回退的次数 |")
    A("")
    A("### 1.1 关于「重试率低」的研究逻辑更正")
    A("")
    A("接受你的纠正：")
    A("")
    A("- **评价整个冻结策略是否有用，不要求它频繁重试**；很少选择重试本身就是策略的表现。")
    A("- **估计重试动作的单独贡献**，才需要额外的对照设计。")
    A("- **重试发生率由数据、反馈和策略共同决定，是结果**，不能直接冻结成外生设计参数。")
    A("")
    A("所以上一轮「应把重试发生率当作设计参数重新预注册」的说法**撤回**。")
    A("本轮无法裁决的主要原因是**对照实现错误**，不是重试率低。")
    A("")
    A("---")
    A("")
    A("## 2. 两项保护性检查（主比较的前置条件，均通过）")
    A("")
    A("| 检查 | 要求 | 结果 |")
    A("|---|---|---|")
    A(f"| 1 | `b = 10, x̂ = 0.1`：MF-BEST 计划截止值 = 10，MF-RETRY = 1.5 | **通过**：{r['guards']['guard1']['mf_best_cap']} 与 {r['guards']['guard1']['mf_retry_cap']} |")
    A("| 2 | 两 MF 臂拿到相同候选状态时，评分与排序一致 | **通过**：三个候选的评分与次序逐项相同 |")
    A("")
    A("检查 2 的实测排序：`(1,3) 79.0 → (0,1) 19.0 → (0,2) 0.4286`，两臂完全一致。")
    A("")
    A("---")
    A("")
    A("## 3. 冻结的协议")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 截止值（按臂显式分派） | RANDOM-BEST / GREEDY-BEST / MF-BEST → **`b_q`**；MF-RETRY → 新候选 `min(b_q, 15x̂)`、重试 `min(b_q, 2L)` |")
    A("| 评分（两 MF 臂共用） | `S = (b_q − x̂) / max(1e-9, min(b_q, x̂))`，**分母不再调用 `cap_for()`** |")
    A("| 成本模型 | **回放执行秒数 ＋ 本机策略秒数**，两者都计入预算；分项记录 |")
    A("| 初始化 | 第 0 列及其等价类；`init_cost` 单列，**不计入 B**（协议）|")
    A("| 派发前截断 | 计划截止值再按**扣掉本次策略成本后**的剩余预算截断 |")
    A("| 预测器 | `censored_als(rank=5, lambda=0.2, iters=50)`，未扫参 |")
    A("| 回放次数 | 四臂 × 三预算 × 三种子 = **36** |")
    A("")
    A("---")
    A("")
    A("## 4. 主比较（满预算 B = D）")
    A("")
    A("| 种子 | MF-BEST | MF-RETRY | 相对变化 | RANDOM-BEST | GREEDY-BEST | 重试 |")
    A("|---:|---:|---:|---:|---:|---:|---:|")
    for i, s in enumerate(seeds):
        A(f"| {s} | {mb[i]:.4f} | {mr[i]:.4f} | **{rel[i]:+.2%}** | "
          f"{g(1.0,'RANDOM-BEST',s,'P'):.4f} | {g(1.0,'GREEDY-BEST',s,'P'):.4f} | "
          f"{g(1.0,'MF-RETRY',s,'retry_actions')} |")
    A(f"| **均值** | **{statistics.fmean(mb):.4f}** | **{statistics.fmean(mr):.4f}** | "
      f"**{mean_rel:+.2%}** | "
      f"{statistics.fmean([g(1.0,'RANDOM-BEST',s,'P') for s in seeds]):.4f} | "
      f"{statistics.fmean([g(1.0,'GREEDY-BEST',s,'P') for s in seeds]):.4f} | |")
    A("")
    A(f"**seed 2 = {rel[2]:+.2%}** 是门槛未过的唯一原因；平均改善 {mean_rel:+.2%} 已越过 2%。")
    A("")
    A("---")
    A("")
    A("## 5. 成本曲线 P(B)/D（三种子均值）")
    A("")
    A("| 预算 | RANDOM-BEST | GREEDY-BEST | MF-BEST | MF-RETRY |")
    A("|---|---:|---:|---:|---:|")
    for frac in fracs:
        cells = [f"{statistics.fmean([g(frac, arm, s, 'P_over_D') for s in seeds]):.4%}"
                 for arm in arms]
        A(f"| {frac:.2f}D | " + " | ".join(cells) + " |")
    A("")
    A("---")
    A("")
    A("## 6. 重试活动：修正协议后为零")
    A("")
    A("| 预算 | 臂 | 探索动作 | 重试 | 实际回退 | 无效预测出现 |")
    A("|---|---|---:|---:|---:|---:|")
    for frac in fracs:
        for arm in arms:
            sub = [x for x in rows if x["budget_frac"] == frac and x["arm"] == arm]
            A(f"| {frac:.2f}D | {arm} | {sum(x['explore_actions'] for x in sub)} | "
              f"{sum(x['retry_actions'] for x in sub)} | "
              f"{sum(x['fallbacks_taken'] for x in sub)} | "
              f"{sum(x['fallbacks_available'] for x in sub)} |")
    A("")
    A(f"**36 次回放合计重试 {total_retry} 次。** 两臂的差异**全部来自新候选的截止值不同**")
    A("（MF-RETRY 用 `min(b_q, 15x̂)`、MF-BEST 直接用 `b_q`），**与「重试」无关**。")
    A("")
    A("> 因此本轮的准确表述是：**这是一个「提前截止对不提前截止」的比较结果。**")
    A("> 它不能用来评价重试。按你的收口要求，我如实接受整个策略的成绩。")
    A("")
    A("「实际回退」与「无效预测出现」已分成两栏；此前混用同一计数的问题已修正。")
    A("")
    A("---")
    A("")
    A("## 7. 成本核算（策略成本已恢复完整）")
    A("")
    A(f"`init_cost` 四臂相同：**{rows[0]['init_cost']:.4f}**（单列，不计入 B）")
    A("")
    A("| 预算 | 臂 | 平均执行 | 平均策略 | 平均合计 | 预算 | 超支 |")
    A("|---|---|---:|---:|---:|---:|:--:|")
    for frac in fracs:
        for arm in arms:
            sub = [x for x in rows if x["budget_frac"] == frac and x["arm"] == arm]
            me = statistics.fmean([x["exec_cost"] for x in sub])
            mp = statistics.fmean([x["policy_cost"] for x in sub])
            ov = sum(1 for x in sub if x["spent"] > x["B"] + 1e-9)
            A(f"| {frac:.2f}D | {arm} | {me:.4f} | {mp:.4f} | {me+mp:.4f} | {sub[0]['B']:.4f} | "
              f"{'**是**' if ov else '否'} |")
    A("")
    A("策略成本现在包含候选生成、ALS 预测、排序与状态更新，量级 0.09–2.3 秒，**不再是 0.02–0.19 秒**。")
    A("所有 36 次回放**总支出都不超过预算**（首版曾有 GREEDY-BEST @0.5D 超支 0.0065 秒，")
    A("原因是截止值在本次决策的策略成本累加**之前**确定；现已在派发前用扣掉策略成本后的剩余预算重新截断，")
    A("并加断言保护）。")
    A("")
    A("---")
    A("")
    A("## 8. 判读")
    A("")
    A("| 预先固定的档位 | 是否命中 |")
    A("|---|---|")
    A("| 三种子均更好 + 平均 ≥2% + 优于两个廉价对照 → 进入第二工作负载确认 | **否**（种子方向混合）|")
    A("| MF-BEST 三种子均不差，或廉价对照稳定更好 → 不扩展该方案 | **否** |")
    A("| **方向混合或差距小 → 记为不确定，不改倍率、不追加种子** | **本轮落在这里** |")
    A("| 只有隐藏真值诊断发现改善空间 → 不算通过 | 未使用该档 |")
    A("")
    A("### 8.1 本轮真正测到了什么")
    A("")
    A("1. **「提前截止」相对「直接运行到 b_q」在 JOB 上平均改善 3.73%，但方向不稳定**（seed 2 略差）。")
    A("2. 两个 MF 臂都优于两个廉价对照，说明预测驱动的候选排序本身有价值。")
    A("3. **关于重试：本轮没有证据。** 因为修正协议后重试一次都没发生，这与「重试无用」是不同的陈述。")
    A("")
    A("### 8.2 明确不做")
    A("")
    A("不改倍率、不追加种子、不设计更容易触发重试的场景、不进入第二工作负载。")
    A("")
    A("---")
    A("")
    A("## 9. 限制")
    A("")
    A("1. **单个工作负载（JOB）**；三种子不构成跨工作负载泛化证据，也不构成统计显著性。")
    A("2. **2% 是筛查门槛**，不是显著性或论文标准。")
    A("3. **重试为 0**，所以本轮对「重试」这一半没有检验力（§6）。这不是缺陷，是结果。")
    A("4. **成本模型含本机策略耗时**，硬件可迁移性有限，这一点明确采用而非回避。")
    A("5. **预测器是拟合的轻量模型**：应表述为「拟合了固定的删失低秩预测器」，不得写成「没有训练任何模型」。")
    A("6. 所有方法共用同一预测器；本轮不分解各机制单独的因果贡献。")
    A("")
    A("---")
    A("")
    A("## 10. 产物")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `Q02_work/q02r_experiment.py` | 协议修正后的实现：按臂分派截止值、共用评分、完整策略成本、断言保护 |")
    A("| `Q02_work/q02r_results.json` | 36 次回放的汇总、守卫检查、协议定义、门槛判定 |")
    A("| `Q02_work/q02r_actions.json` | **逐条动作日志（1,671 条）**，含 `b_before`、`L_before`、`is_retry`、`prediction`、`score`、`planned_cap`、`cap_reason`、`actual_cap`、反馈、成本分项 |")
    A("| `Q02_work/q02_experiment.py` | 旧实现（缺陷版），保留 |")
    A("")
    A("哈希：")
    A("")
    A("| 文件 | sha256 |")
    A("|---|---|")
    for p in (HERE / "job_equivalence.json", HERE / "q02r_results.json",
              HERE / "q02r_actions.json"):
        if p.exists():
            A(f"| `{p.name}` | `{sha(p)}` |")
    A("")

    (ROOT / "Q02R_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'Q02R_report.md'}")

    cases = {
        "task": "Q02R -- protocol-corrected four-arm cost-quality screening on JOB",
        "supersedes": "Q02_report.md main comparison",
        "verdict": "GATE NOT MET: 2 of 3 seeds better, mean +3.73% (>= 2%), but seed 2 is -0.33%",
        "reading": "mixed direction -> undetermined; multiplier unchanged, no seeds added; the "
                   "strategy's overall result is accepted as it stands",
        "retractions": [
            "'both arms used the same ranking rule' -- MF-BEST used min(b_q, x_hat) as denominator, "
            "MF-RETRY used the budget-dependent cap, so the arms differed even with zero retries",
            "'this round only tested early cutoff without retry' -- BOTH arms were cutting early, "
            "because cap_for() returned min(b_q, 15*x_hat) for a fresh candidate in both MF arms; "
            "the main contrast was mis-defined",
            "'239 exploration actions and 1 retry at the main budget' -- 239/1 is the total across "
            "all three budgets; the main budget alone was 158/1",
            "'23 of 25 timeouts ran to b_q' -- computed from end-of-run bounds, not from the state "
            "at each timeout",
            "'init_cost is an execution deviation' -- it is the protocol the auditor specified; "
            "charging it to the exploration budget was the implementation error",
            "'simulated and wall-clock seconds cannot be added' -- both are seconds; the agreed cost "
            "model is replay execution time plus local policy time",
            "'0.02-0.19 s is the whole strategy cost' -- only the ALS call was timed",
            "'fallbacks counts random fallbacks' -- it counted decisions with an invalid prediction",
            "'the retry incidence should be fixed as a design parameter' -- the incidence is an "
            "outcome determined by data, feedback and policy, not an exogenous parameter",
        ],
        "guards": r["guards"],
        "protocol": r["protocol"],
        "gate": r["gate"],
        "total_retry_actions_all_36_replays": total_retry,
        "key_finding": "with the protocol corrected, retry never fires: the two MF arms differ only "
                       "in the cap given to FRESH candidates (min(b_q,15*x_hat) vs b_q), so this is "
                       "an early-cutoff-versus-no-cutoff comparison and says nothing about retry",
        "relative_vs_MF_BEST": {f"seed_{s}": rel[i] for i, s in enumerate(seeds)},
        "mean_relative": mean_rel,
        "cost_model": "replay execution seconds + local policy seconds, both charged to B; "
                      "init_cost separate and identical; every replay stayed within budget after the "
                      "last-action truncation fix",
        "action_log": "Q02_work/q02r_actions.json, 1671 records, fields include b_before, L_before, "
                      "is_retry, prediction, score, planned_cap, cap_reason, actual_cap, feedback "
                      "and per-item costs",
        "not_done": ["no data change", "no predictor change", "no multiplier change",
                     "no extra seeds", "no second workload",
                     "no attempt to make retry fire more often"],
    }
    (ROOT / "Q02R_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'Q02R_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
