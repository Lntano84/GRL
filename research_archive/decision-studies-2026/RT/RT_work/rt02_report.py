"""RT-02: write the report and frozen artifacts."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> int:
    r = json.loads((HERE / "rt02_results.json").read_text(encoding="utf-8"))
    res = r["results"]
    fracs = [f for f in ("0.4", "0.6", "0.8") if f in res and "scores" in res[f]]

    L: list[str] = []
    A = L.append
    A("# RT-02 —— 一次受控的完成反馈实验")
    A("")
    A("## 结论")
    A("")
    A("**门槛未通过。而且失败的原因不是方法优劣，而是设计的前提不成立：**")
    A("**在只用品类 A 的队列里，删失几乎不存在（0.7–0.9%）。**")
    A("")
    A("四项预检全部通过。三个窗口的相对变化**没有一个为正**，等权平均 **−17.86%**，")
    A("预先固定的门槛（三窗口均为正且平均 ≥1%）**未达到**。")
    A("")
    A("按预先固定的判读表，本轮落在**第 4 类（差异小或方向不一致）**：")
    A("记录为这个条件化队列上的**弱证据**；不写成整个方向无效，更不写成 oracle 上界。")
    A("")
    A("---")
    A("")
    A("## 1. 冻结的设置")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 记录 | 仅品类 A：全部任务有结束时间、存档状态 Terminated、时间有限、时长为正 |")
    A("| 标签 | `p_j = max_t e_jt − min_t s_jt` |")
    A("| 回放起点 | `min_t s_jt` |")
    A("| 研究对象 | **执行跨度的完成反馈**，不称为累计服务量 |")
    A(f"| 入选记录 | **{r['n_classA']:,}** |")
    A(f"| 时间跨度 | `{r['t0']:,.0f}` .. `{r['t1']:,.0f}`（{(r['t1']-r['t0'])/86400:.1f} 天）|")
    A(f"| 更新点 | 跨度的 **40% / 60% / 80%**，不重选 |")
    A(f"| 近期训练队列 | 自 T 起**前 7 天**内开始的作业 |")
    A(f"| 测试队列 | 自 T 起**后 2 天**内开始的作业 |")
    A("| 近期队列可见通道 | `z_j = min(p_j, T−s_j)`，`δ_j = 1[e_j ≤ T]` |")
    A("| 目标 | 作业时长中位数 |")
    A("| 指标 | 平均绝对对数误差 `mean |log(1+p̂) − log(1+p)|` |")
    A("")
    A("**条件化披露。** 只选品类 A 使用了存档后的信息（我们知道这些作业最终成功）。")
    A("因此这是**成功作业队列的条件化回放**，不是生产复现：部署时并不知道哪些在跑作业会成功。")
    A("")
    A("---")
    A("")
    A("## 2. 四项预检（全部通过）")
    A("")
    A("| 预检 | 要求 | 结果 |")
    A("|---|---|---|")
    A("| 1 | 三个近期方法入选作业 ID 完全一致，只有标签权限不同 | **通过**：三个窗口的近期队列分别 "
      f"{res['0.4']['recent_ids_count']:,} / {res['0.6']['recent_ids_count']:,} / "
      f"{res['0.8']['recent_ids_count']:,} 个作业，**0 个重复 ID**，RECENT/KM/FULL-LABEL 由同一集合构造 |")
    A("| 2 | FULL-LABEL 覆盖率 100%，不允许静默丢记录 | **通过**：三个窗口均 **100.0000%** |")
    A("| 3 | 改动被遮蔽的最终时长（保持可见开始时间、删失状态、年龄不变）后，RECENT/KM 的输入不变 | "
      "**通过**：把隐藏侧的最终时长乘 1000 后，可见通道 `(z, δ)` 对整个近期队列**逐项不变** |")
    A("| 4 | 手算样本验证 KM：无删失、有并列与删失、曲线未达中位数 | **通过**（4 个用例 + 1 个置换不变性） |")
    A("")
    A("### 2.1 KM 手算校验明细")
    A("")
    A("| 用例 | 期望 | 实测 |")
    A("|---|---|---|")
    A("| 无删失，时刻 1..5 | 中位 3 | **3** |")
    A("| 并列 + 删失：2 事件@1（5 在险），1 事件@2（3 在险） | S: 0.6 → 0.4，中位 2 | **2** |")
    A("| 全部删失 | 未达 0.5 | **None** |")
    A("| 6 条中 1 个事件 | 未达 0.5 | **None** |")
    A("| 置换 (时刻, 事件) 对 | 结果不变 | **仅 1 个取值** |")
    A("")
    A("并列时刻按「同一时刻的人一起处理」实现，因此风险集在同一时刻内相同，**并列顺序不可能影响结果**。")
    A("本轮没有自行实现分位数检验。")
    A("")
    A("---")
    A("")
    A("## 3. 结果")
    A("")
    A("| 窗口 | 近期队列 | 其中删失 | 测试队列 | 历史 |")
    A("|---|---:|---:|---:|---:|")
    for f in fracs:
        d = res[f]
        A(f"| T/{float(f)*100:.0f}%（第 {float(f)*(r['t1']-r['t0'])/86400:.1f} 天） | "
          f"{d['n_recent']:,} | **{d['censored_share_recent']:.2%}** | {d['n_test']:,} | "
          f"{d['n_history']:,} |")
    A("")
    A("### 3.1 平均绝对对数误差（越小越好）")
    A("")
    A("| 窗口 | HISTORY | RECENT | KM | FULL-LABEL（不可部署） | KM 相对 min(H,R) |")
    A("|---|---:|---:|---:|---:|---:|")
    for f in fracs:
        s = res[f]["scores"]
        h, rec, km, fl = (s["HISTORY"]["mean_abs_log_error"], s["RECENT"]["mean_abs_log_error"],
                          s["KM"]["mean_abs_log_error"], s["FULL_LABEL"]["mean_abs_log_error"])
        rel = (min(h, rec) - km) / min(h, rec)
        A(f"| {float(f)*100:.0f}% | {h:.5f} | {rec:.5f} | {km:.5f} | {fl:.5f} | "
          f"**{rel:+.2%}** |")
    A("")
    A("### 3.2 回退比例（必须与成绩一起读）")
    A("")
    A("| 窗口 | KM 回退到 HISTORY | HISTORY 组样本不足 | RECENT 组样本不足 |")
    A("|---|---:|---:|---:|")
    for f in fracs:
        d = res[f]
        n = d["n_test"]
        A(f"| {float(f)*100:.0f}% | {d['km_fallback']:,}/{n:,} ({d['km_fallback']/n:.1%}) | "
          f"{d['history_group_fallback']/n:.1%} | {d['recent_group_fallback']/n:.1%} |")
    A("")
    A("**回退比例极高（KM 87–91%），所以「KM 的成绩」主要是回退后的成绩，")
    A("而不是 KM 自身的估计。** 这一点必须写在结论旁边。")
    A("")
    A("---")
    A("")
    A("## 4. 为什么门槛没过：设计前提不成立")
    A("")
    A("本轮预先固定的判读是这样写的：若传统删失估计已经足够，就不再开发复杂学习器。")
    A("实际发生的是更前置的一件事：**这个设计里几乎没有删失可纠正。**")
    A("")
    A("| 事实 | 数值 |")
    A("|---|---|")
    A("| 近期队列的删失比例 | **0.7% – 0.9%** |")
    A("| FULL-LABEL 相对 RECENT 的改善 | 三个窗口分别 "
      f"{res['0.4']['scores']['RECENT']['mean_abs_log_error']-res['0.4']['scores']['FULL_LABEL']['mean_abs_log_error']:+.5f} / "
      f"{res['0.6']['scores']['RECENT']['mean_abs_log_error']-res['0.6']['scores']['FULL_LABEL']['mean_abs_log_error']:+.5f} / "
      f"{res['0.8']['scores']['RECENT']['mean_abs_log_error']-res['0.8']['scores']['FULL_LABEL']['mean_abs_log_error']:+.5f}"
      "（近乎为零） |")
    A("")
    A("**含义：** 只用品类 A 之后，RECENT / KM / FULL-LABEL 拿到的基本是**同一份数据**。")
    A("三者的预测几乎相同，所以这个实验**结构上无法**回答「删失估计是否有用」。")
    A("")
    A("这与质检轮（RT-01R）的标签资格结论完全一致：")
    A("**真正缺标签的那 23% 记录不在品类 A 里**，而把它们纳入又需要先把「在跑 / 失败 / 没记上」")
    A("区分开——那正是当前数据做不到的事。")
    A("")
    A("### 4.1 另一个必须报告的事实")
    A("")
    A("HISTORY 在两个窗口上**明显优于** RECENT 与 KM（1.37 vs 1.69、1.19 vs 1.50）。")
    A("按你预先固定的判读，这**不属于**第 3 类（「只有相对 RECENT 改善但不能超过 HISTORY」），")
    A("而是相反方向：长期历史更强。这条也要如实记录。")
    A("")
    A("---")
    A("")
    A("## 5. 判读")
    A("")
    A("| 预先固定的档位 | 是否命中 |")
    A("|---|---|")
    A("| KM 相对 HISTORY 和 RECENT 都稳定改善 → 进入固定调度器验证 | **否** |")
    A("| FULL-LABEL 改善而 KM 没改善 → 提示估计方法未用好信息 | **否**：FULL-LABEL 本身几乎没改善 |")
    A("| 只有相对 RECENT 改善，不能超过 HISTORY → 不扩展 | **否**：方向相反，HISTORY 更强 |")
    A("| 各方法差异很小或方向不一致 → 弱证据，不写成方向无效 | **本轮实际落在这里**（且原因已定位） |")
    A("")
    A("**因此：不进入调度器验证，也不开发复杂学习器。但这不是对方向的否定。**")
    A("")
    A("本轮的可用产出是**一个结论清晰的负结果加一个已定位的原因**：")
    A("条件化成成功队列会把删失抹掉，所以这个设计测不到删失纠正的价值。")
    A("")
    A("---")
    A("")
    A("## 6. 限制")
    A("")
    A("1. **条件化回放。** 只选品类 A 用了存档信息，不能宣称部署时知道哪些作业会成功。")
    A("2. **删失比例过低（<1%），因此本轮的比较没有检验力**，不能据此判断 KM 是否有用。")
    A("3. **回退主导。** KM 与 RECENT 有 87–91% 的测试作业走了回退路径，")
    A("   所以成绩主要反映回退规则，而非各自的估计。")
    A("4. **分组稀疏。** 217,982 个 group 中只有 5,955 个（覆盖 53.3% 作业）有 ≥30 条记录，")
    A("   而近期 7 天队列更小，导致回退率极高。")
    A("5. **只测执行跨度。** 未测排队延迟与响应时间（数据不含可观测的等待时间，见 RT-01R §5）。")
    A("6. **不接调度器。** 未测任何调度收益；也未测排序变化。")
    A("7. **未训练神经网络**，只用了分组经验中位数与 Kaplan-Meier。")
    A("")
    A("---")
    A("")
    A("## 7. 下一步若要重获检验力（本轮不做）")
    A("")
    A("要让「删失纠正是否有用」可被检验，训练队列必须**真的有相当比例的删失**。")
    A("可选路径：")
    A("")
    A("1. **把队列定义放宽到品类 A 之外**，但前提是先能给「在跑 / 失败 / 静默缺失」一个可判定的事件定义。")
    A("   这需要生命周期记录，当前数据没有。")
    A("2. **缩短近期窗口**，使 T 前 7 天内开始的作业有更大比例尚未完成。")
    A("   但这属于改设计，必须**重新预注册**，不能在本轮事后调参。")
    A("3. **换一个带完整标签且删失可构造的数据集**，用人为截断（遮住 T 之后的完成时间）")
    A("   制造已知比例的删失。这样 oracle 覆盖率已知，信息价值才可谈量级。")
    A("")
    A("第 2 条最省，但必须先把「删失比例」当作设计参数而不是结果，并在选窗口前固定。")
    A("")
    A("---")
    A("")
    A("## 8. 复现出口")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `RT_work/rt02_experiment.py` | 本轮全部代码：KM 校验、四项预检、四方法、门槛判读 |")
    A("| `RT_work/rt02_results.json` | 冻结结果（每窗口的分数、回退率、删失比例、预检结论）|")
    A("| `RT_work/r3_estimator_lib.py` | 修正后的估计器工具与 17 项测试（含开关一致性 guard）|")
    A("| `RT_work/r7_rescue_eligibility.py` | 956 → 895 的资格复核 |")
    A("| `RT01R_report.md` | 质检轮修正版（含第二轮修正 §5R）|")
    A("")

    (ROOT / "RT02_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT / 'RT02_report.md'}")

    cases = {
        "task": "RT-02 -- one controlled completion-feedback experiment on the ATLAS/PAI trace",
        "status": "pre-registered gate NOT met; the design's premise failed, so the round has no "
                  "power to judge whether censoring correction helps",
        "design": {
            "records": "class A only (all tasks ended, status Terminated, finite times, positive "
                       "duration)",
            "label": "p_j = max_t e_jt - min_t s_jt",
            "replay_start": "min_t s_jt",
            "target": "per-job duration median",
            "metric": "mean absolute log error",
            "update_points": [0.40, 0.60, 0.80],
            "recent_window_days": 7,
            "test_window_days": 2,
            "visible_channel": "z_j = min(p_j, T - s_j), delta_j = 1[e_j <= T]",
            "fallbacks": {"min_group": 30,
                          "km_median_unreached": "fall back to HISTORY and record the share",
                          "history_insufficient": "fall back to the historical global median"},
            "conditioning_disclosure": "selecting class A uses archive information; this is a "
                                       "conditional replay of a successful-job queue, not a "
                                       "production reproduction",
        },
        "prechecks": {
            "1_identical_recent_ids": True,
            "2_full_label_coverage": 1.0,
            "3_blindness_to_hidden_labels": True,
            "4_km_hand_computed_cases": 5,
        },
        "results": res,
        "gate": {"positive_windows": 0, "equal_weight_mean_relative_change": -0.1786,
                 "threshold": {"all_three_positive": True, "mean_at_least": 0.01},
                 "met": False},
        "why_not_met": "the recent queue is only 0.7-0.9% censored once restricted to class A, so "
                       "RECENT, KM and FULL-LABEL receive essentially the same data; the experiment "
                       "cannot answer whether censoring correction helps",
        "secondary_observation": "HISTORY beats RECENT and KM in two of three windows (1.37 vs 1.69 "
                                 "and 1.19 vs 1.50), which is the opposite of the pre-registered "
                                 "'only better than RECENT' case",
        "not_concluded": ["that the direction is invalid",
                          "any oracle bound on information value",
                          "any scheduling benefit"],
    }
    (ROOT / "RT02_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"wrote {ROOT / 'RT02_cases.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
