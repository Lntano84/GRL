"""RT-02R: write the corrected report and artifacts."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DAY = 86400.0


def main() -> int:
    r = json.loads((HERE / "rt02r_results.json").read_text(encoding="utf-8"))
    dg = r["diagnostics"]
    sc = r["scores"]
    dec = r["decomposition"]
    fracs = list(sc.keys())

    L: list[str] = []
    A = L.append
    A("# RT-02R —— 修正后的受控完成反馈实验")
    A("")
    A("## 结论")
    A("")
    A("**修正四成协议偏差后，KM 仍然没有超过 HISTORY，而且差距的性质现在可以准确定位：**")
    A("**KM 与 HISTORY 的差距几乎全部来自回退样本，KM 自身（组级 KM 生效的那部分）贡献约等于零。**")
    A("")
    A("| 窗口 | KM − HISTORY 总差 | 其中来自 KM 回退样本 | 其中来自组级 KM 生效样本 |")
    A("|---|---:|---:|---:|")
    for f in fracs:
        d = dec[f]
        A(f"| {float(f)*100:.0f}% | {d['total']:+.6f} | **{d['fallback_contribution']:+.6f}** | "
          f"{d['group_km_contribution']:+.6f} |")
    A("")
    A("（正数表示 KM 更差。）")
    A("")
    A("按事先约定的收口条件：**KM 在 0/3 个窗口超过 HISTORY，`KM vs min(HISTORY, RECENT)` "
      f"三个窗口分别 {sc[fracs[0]]['km_vs_min_relative']:+.2%} / "
      f"{sc[fracs[1]]['km_vs_min_relative']:+.2%} / {sc[fracs[2]]['km_vs_min_relative']:+.2%}"
      "，等权平均 "
      f"{sum(sc[f]['km_vs_min_relative'] for f in fracs)/len(fracs):+.2%}。**")
    A("")
    A("**因此：在这份数据上停止扩展「七天近期分组 KM」方案。**")
    A("不靠提高删失率维持它，也不宣称在线完成反馈问题整体无价值。")
    A("")
    A("---")
    A("")
    A("## 1. 本轮修正了四处协议偏差")
    A("")
    A("| # | 上一轮的问题 | 本轮修正 |")
    A("|---|---|---|")
    A("| 1 | 「KM 中位数不可识别时回退到 HISTORY」被实现成了回退到**历史全局中位数** `gl_hist`。"
      "两种回退规则被合并成 `if v is None or len(pairs) < MIN_GROUP: v = gl_hist` | "
      "两条规则**分开实现并分别计数**：组内近期样本 <30 → 该方法的全局估计（KM 用**全局 KM**）；"
      "中位数不可识别 → **该作业的 HISTORY 预测** |")
    A(f"| 2 | 时间跨度没有沿用旧冻结值（代码在筛选 A 类后重算了 `t0`） | 本轮改用**原冻结时钟** "
      f"`t0={r['frozen_clock']['t0']:,.0f}`、`t1={r['frozen_clock']['t1']:,.0f}`（跨度 "
      f"{(r['frozen_clock']['t1']-r['frozen_clock']['t0'])/86400:.2f} 天）|")
    A("| 3 | 隐藏标签扰动检查只做了第一个窗口，且只比较 `(z, δ)`，没有重算预测 | "
      "**三个窗口都做**，并且**重跑全部四种方法**，比对输入与预测；不一致即中止 |")
    A("| 4 | 把整体结果强行归入原判读表第 4 类 | **不再归类**，改为分开报告四件事（见 §4）|")
    A("")
    A("### 1.1 关于冻结时钟的一处必须说明的事实")
    A("")
    A(f"改用原冻结 `t0 = {r['frozen_clock']['t0']:,.0f}` 之后，三个更新点确实移回了预定位置。")
    A(f"但有一点必须写明：**A 类作业的最早开始时间是 `{r['earliest_classA_start']:,.0f}`**，")
    A(f"比原冻结 `t0` 晚约 {(r['earliest_classA_start']-r['frozen_clock']['t0'])/DAY:.2f} 天。")
    A("因此原冻结窗口的**最前面约 0.56 天没有任何 A 类作业**，前 40% 窗口的历史信息因此更少。")
    A("这不是本轮的选择，而是「只用品类 A」与「沿用全轨迹冻结窗口」两条规则叠在一起的结果；")
    A("按你的要求，本轮**不做窗口搜索**，所以保留原时钟并如实记录这一后果。")
    A("")
    A("---")
    A("")
    A("## 2. 预检")
    A("")
    A("| 预检 | 结果 |")
    A("|---|---|")
    A("| 三方法入选作业 ID 完全一致 | **通过**（RECENT / KM / FULL-LABEL 由同一近期队列构造）|")
    A("| FULL-LABEL 覆盖率 100% | **通过**（三个窗口均 100.0000%）|")
    A("| 隐藏标签扰动：输入与预测均不变 | **三个窗口全部通过**（分别扰动 "
      f"{dg[fracs[0]]['censored_recent']:,} / {dg[fracs[1]]['censored_recent']:,} / "
      f"{dg[fracs[2]]['censored_recent']:,} 条隐藏时长 ×1000）|")
    A("| KM 手算校验 | **通过**（沿用 RT-02 的 4 个用例 + 置换不变性）|")
    A("")
    A("### 2.1 修正后的回退确实生效了")
    A("")
    A("这是第 1 条修正的直接证据——**全局 KM 与历史全局中位数现在是不同的数**：")
    A("")
    A("| 窗口 | 全局 HISTORY | 全局 RECENT（已完成） | **全局 KM** | 全局 FULL | 全局 KM ≠ 全局 HISTORY |")
    A("|---|---:|---:|---:|---:|:--:|")
    for f in fracs:
        d = dg[f]
        kmv = "None" if d["gl_km"] is None else f"{d['gl_km']:,.0f}"
        A(f"| {float(f)*100:.0f}% | {d['gl_hist']:,.0f} | {d['gl_rec']:,.0f} | **{kmv}** | "
          f"{d['gl_full']:,.0f} | **{d['km_returns_global_km_distinct_from_gl_hist']}** |")
    A("")
    A("---")
    A("")
    A("## 3. 成绩与逐方法回退归因")
    A("")
    A("| 窗口 | HISTORY | RECENT | KM | FULL-LABEL | KM vs min(H,R) |")
    A("|---|---:|---:|---:|---:|---:|")
    for f in fracs:
        s = sc[f]["scores"]
        A(f"| {float(f)*100:.0f}% | {s['HISTORY']:.5f} | {s['RECENT']:.5f} | {s['KM']:.5f} | "
          f"{s['FULL_LABEL']:.5f} | {sc[f]['km_vs_min_relative']:+.2%} |")
    A("")
    A("### 3.1 每个方法自己的回退构成")
    A("")
    A("| 窗口 | 方法 | 组内估计 | 回退到全局 | 回退到该作业 HISTORY |")
    A("|---|---|---:|---:|---:|")
    for f in fracs:
        rows = r["per_job"][f]
        n = len(rows)
        for m in ("HISTORY", "RECENT", "KM", "FULL_LABEL"):
            cnt = {}
            for x in rows:
                cnt[x[f"{m}_reason"]] = cnt.get(x[f"{m}_reason"], 0) + 1
            grp = sum(v for k, v in cnt.items()
                      if k.startswith("group"))
            glob = sum(v for k, v in cnt.items() if "global" in k)
            histb = cnt.get("median_unidentifiable_uses_history", 0)
            if m != "KM":
                A(f"| {float(f)*100:.0f}% | {m} | {grp:,} | {glob:,} | — |")
            else:
                A(f"| {float(f)*100:.0f}% | {m} | {grp:,} | {glob:,} | {histb:,} |")
    A("")
    A("---")
    A("")
    A("## 4. 分开报告（不再强行归入原判读表）")
    A("")
    A("| # | 事实 | 依据 |")
    A("|---|---|---|")
    A(f"| 1 | **当前实现没有通过继续门槛**：KM 在 0/3 个窗口超过 HISTORY，"
      f"等权平均 {sum(sc[f]['km_vs_min_relative'] for f in fracs)/len(fracs):+.2%} | §3 |")
    A(f"| 2 | **RECENT 与 FULL-LABEL 的整体差异很小**：三个窗口分别 "
      f"{abs(sc[fracs[0]]['scores']['RECENT']-sc[fracs[0]]['scores']['FULL_LABEL']):.5f} / "
      f"{abs(sc[fracs[1]]['scores']['RECENT']-sc[fracs[1]]['scores']['FULL_LABEL']):.5f} / "
      f"{abs(sc[fracs[2]]['scores']['RECENT']-sc[fracs[2]]['scores']['FULL_LABEL']):.5f} |")
    A("| 3 | **KM 与 HISTORY 的大差距主要位于回退样本**，组级 KM 生效部分贡献约等于零 | §结论表 |")
    A("| 4 | **预定方法尚有协议偏差**：见 §1，本轮已修正三处（时钟、回退、盲检），"
      "但第 4 条（判读表不穷尽）属于设计层面，需要下一轮重新预注册 | §1.4 |")
    A("")
    A("### 4.1 我上一轮的错误归因")
    A("")
    A("上一轮我把结果解释成「设计前提不成立：删失几乎被抹掉，所以结构上无法回答」。")
    A("**这个解释被本轮否证了两点：**")
    A("")
    A(f"1. 删失并没有被抹掉：三个窗口仍有 "
      f"{dg[fracs[0]]['censored_recent']:,} / {dg[fracs[1]]['censored_recent']:,} / "
      f"{dg[fracs[2]]['censored_recent']:,} 条删失记录（"
      f"{dg[fracs[0]]['censored_share']:.2%} / {dg[fracs[1]]['censored_share']:.2%} / "
      f"{dg[fracs[2]]['censored_share']:.2%}）。RECENT 与 FULL-LABEL 的标签集合**不同**，")
    A("   不是「同一份数据」。当时那个说法**撤回**。")
    A("2. 当时的 KM−HISTORY 差距**几乎全部来自回退实现，而不是删失效应**。")
    A("   修正回退后差距依旧，但性质说清楚了：它是**估计量有效样本不足**造成的，")
    A("   与删失比例高低不是同一件事。")
    A("")
    A("---")
    A("")
    A("## 5. 限制")
    A("")
    A("1. **回退主导。** 组级估计只在 "
      f"{sum(1 for x in r['per_job'][fracs[0]] if x['KM_reason']=='group_km'):,}–"
      f"{sum(1 for x in r['per_job'][fracs[2]] if x['KM_reason']=='group_km'):,} 个测试作业上生效，")
    A("   其余走全局估计。所以「KM 的成绩」主要是回退后的成绩。")
    A("2. **30 条阈值未调整**（按你的要求），因此无法判断放宽阈值是否会改变结论。")
    A("3. **条件化回放。** 只选品类 A 使用了存档信息，不能宣称部署时知道哪些作业会成功。")
    A("4. **只测执行跨度**，未测排队延迟或响应时间。")
    A("5. **不接调度器**，未测任何调度收益，也未测排序变化。")
    A(f"6. **前 {float(r['earliest_classA_start']-r['frozen_clock']['t0'])/86400:.2f} 天没有 A 类作业**，")
    A("   这是沿用原冻结时钟的直接后果（§1.1），本轮不做窗口搜索。")
    A("7. **未训练神经网络**；只用了分组经验中位数与 Kaplan-Meier。")
    A("")
    A("---")
    A("")
    A("## 6. 停止扩展的理由（明确不是「场景不够难」）")
    A("")
    A("按你要求的收口条件，本轮判定为**停止扩展这套数据上的「七天近期分组 KM」方案**。")
    A("理由是可复算的三条：")
    A("")
    A("1. KM 在 0/3 窗口超过 HISTORY，方向一致，不是「差异小或方向不一致」。")
    A("2. 差距集中在回退样本；**组级 KM 生效部分的贡献约等于零**，")
    A("   说明问题不在删失纠正本身，而在**近期窗口内没有足够的组级样本**。")
    A("3. 这是一个**统计样本量**问题，不是「场景难度」问题；")
    A("   提高删失率不会增加组级样本，反而会更少，所以**不靠调比例维持它**。")
    A("")
    A("同时明确不宣称：在线完成反馈问题整体无价值；也不宣称任何信息价值上界。")
    A("")
    A("---")
    A("")
    A("## 7. 复现出口")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `RT_work/rt02r_fixed.py` | 本轮全部代码：冻结时钟、两条回退规则分离、三窗口盲检、误差分解 |")
    A("| `RT_work/rt02r_results.json` | 冻结结果：**逐作业预测、回退原因、误差**、分解、诊断 |")
    A("| `RT_work/rt02_experiment.py` | 上一轮（回退实现有误），保留作记录 |")
    A("| `RT_work/r3_estimator_lib.py` | 估计器工具与 17 项测试（含开关一致性 guard）|")
    A("")

    (ROOT / "RT02R_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'RT02R_report.md'}")

    cases = {
        "task": "RT-02R -- corrected controlled completion-feedback experiment",
        "supersedes": "RT02_report.md",
        "verdict": "STOP expanding the '7-day recent group-wise KM' scheme on this dataset",
        "verdict_reason": "KM does not beat HISTORY in 0/3 windows, the direction is consistent, and "
                          "the gap is concentrated on fallback rows while the group-level KM rows "
                          "contribute ~0; this is a group-sample-size issue, not a scenario-difficulty "
                          "issue, so raising the censoring share would not help",
        "protocol_fixes": {
            "frozen_clock": {"t0": r["frozen_clock"]["t0"], "t1": r["frozen_clock"]["t1"],
                             "note": "the superseded run recomputed t0 after the class-A filter; "
                                     "earliest class-A start is "
                                     f"{r['earliest_classA_start']:.0f}, i.e. the first "
                                     f"{(r['earliest_classA_start']-r['frozen_clock']['t0'])/86400:.2f} "
                                     "days of the frozen window contain no class-A job"},
            "fallback_rules_separated": {
                "sparse_group": "use that METHOD's global estimate (KM -> global KM)",
                "median_unidentifiable": "use that JOB's HISTORY prediction",
                "superseded_bug": "both cases fell through to the history global median",
            },
            "blindness_check": "all three windows, inputs AND predictions, abort on mismatch",
        },
        "diagnostics": dg,
        "scores": sc,
        "decomposition": dec,
        "retracted_from_RT02": [
            "'structural inability to answer' / 'no power' -- censoring was NOT zero (881/895/781 "
            "records) and RECENT and FULL-LABEL do not see identical labels",
            "'the small censoring share explains the poor result' -- the gap is almost entirely a "
            "fallback artefact, not a censoring effect",
            "forcing the result into interpretation category 4 ('differences small or inconsistent') "
            "-- HISTORY beats KM in all three windows, consistently",
        ],
        "not_concluded": ["that the online completion-feedback problem is worthless",
                          "any information-value upper bound",
                          "any scheduling benefit"],
        "per_job_outputs": "in rt02r_results.json under 'per_job': predictions, fallback reason and "
                           "error for every test job, for all four methods",
    }
    (ROOT / "RT02R_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'RT02R_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
