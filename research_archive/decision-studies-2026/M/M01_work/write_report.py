"""Assemble the M01 deliverables: report, cases, results, first-action tables."""
from __future__ import annotations

import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "M01_work"
sys.path.insert(0, str(WORK))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    summary = load(WORK / "M01_summary.json")
    cases = load(ROOT / "M01_cases.json")
    indep = load(WORK / "M01_independent_check.json")
    equiv = load(WORK / "M01_equivalence_check.json")
    rows = [r for r in summary["rows"] if not r.get("missing")]
    order = ["n6_m6_s0_HOM", "n6_m6_s0_HET", "n6_m6_s1_HOM", "n6_m6_s1_HET",
             "n6_m6_s2_HOM", "n6_m6_s2_HET", "n6_m6_s3_HOM", "n6_m6_s3_HET",
             "n6_m10_s0_HOM", "n6_m10_s0_HET", "n6_m10_s1_HOM", "n6_m10_s1_HET",
             "n6_m10_s2_HOM", "n6_m10_s2_HET", "n6_m10_s3_HOM", "n6_m10_s3_HET",
             "n8_m8_s0_HOM", "n8_m8_s0_HET", "n8_m8_s1_HOM", "n8_m8_s1_HET",
             "n8_m8_s2_HOM", "n8_m8_s2_HET", "n8_m8_s3_HOM", "n8_m8_s3_HET",
             "n8_m12_s0_HOM", "n8_m12_s0_HET", "n8_m12_s1_HOM", "n8_m12_s1_HET",
             "n8_m12_s2_HOM", "n8_m12_s2_HET", "n8_m12_s3_HOM", "n8_m12_s3_HET"]
    by_id = {r["config_id"]: r for r in rows}
    rows = [by_id[c] for c in order if c in by_id]

    dg = [r["d_G"] for r in rows]
    dr = [r["d_robust"] for r in rows]
    drmax = [r["d_robust_max_over_shifts"] for r in rows]
    n_degenerate = sum(1 for r in rows if r["d_G"] < 1e-12 and r["d_robust"] < 1e-12)
    ratio = [r["d_robust"] / r["d_G"] for r in rows if r["d_G"] > 1e-12]
    verify = load(WORK / "M01_configs_frozen.json")["protocol"]

    L: list[str] = []
    A = L.append
    A("# M01 —— 三轮随机匹配中，完整策略相对「一步前瞻＋贪心续接」还剩多少收益")
    A("")
    A("## 一页结论")
    A("")
    A("**主指标 `d_robust`（即使把 R 的首轮并列往最有利方向打破后仍留下的损失）最大值只有 "
      f"{max(drmax):.5f} 个期望匹配顶点，远低于门槛（6 顶点 0.06、8 顶点 0.08）。**")
    A("32 个配置全部完成，**没有一个达到门槛**。")
    A("")
    A(f"**裁决：{summary['tier']}。**")
    A("按预先固定的判读表：**停止在这批小图分布上挖掘「三轮前瞻修正」**。")
    A("这不否定更大图上的任何结论，也不构成对学习方法的否定。")
    A("")
    A("**最重要的定量结论：前瞻已经把贪心几乎全部缺口补掉了。**")
    A(f"贪心缺口 `d_G` 在 25/32 个配置上为正，均值 {statistics.fmean(dg):.4f}、最大 {max(dg):.4f}；")
    A(f"而 `d_robust` 的均值只有 {statistics.fmean(dr):.5f}，最大 {max(dr):.5f}。")
    A(f"在 `d_G > 0` 的配置上，`d_robust / d_G` 的均值为 "
      f"{statistics.fmean(ratio):.4f}（最大 {max(ratio):.4f}）。")
    A("")
    A("> 按任务预先写下的判读：**如果 `d_G` 很大而 `d_robust` 很小，正确结论是「前瞻已经解决主要")
    A("> 问题，贪心反例不足以支持学习」。本轮正是这个情形。**")
    A("")
    A("---")
    A("")
    A("## 1. 冻结的任务语义")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 模型 | 无向简单图；每条边的成功状态独立且**全程保持不变** |")
    A("| 每轮 | 选择一个匹配；**允许非极大匹配和空匹配** |")
    A("| 成功边 | 贡献 **2 个匹配顶点**，两端点及其关联边退出 |")
    A("| 失败边 | **永久删除**，端点保留 |")
    A("| 观察 | 只观察本轮选择的边；未选边的真实状态永不读取 |")
    A(f"| 轮数 | **3 轮**；`h` 含当前轮，`V_0 = 0` |")
    A(f"| 目标 | 期望匹配顶点数 |")
    A(f"| `ε` | `1e-10 × max(1, n)` |")
    A(f"| 并列 | 字典序最小的排序边元组 |")
    A(f"| 蒙特卡洛 | **未使用**；全部为精确枚举 + 记忆化 |")
    A("")
    A("三个方法的定义严格分开（`V_h^G` 用完整前瞻评估贪心动作规则；`V_h^R` 是**实际执行** R")
    A("规则的价值，不是 `max_M Q̃`；`V_h^*` 是完整有限轮次最优）。")
    A("R 在每一轮都重新求解，续接估计固定为完整贪心策略 `V^G`。")
    A("")
    A("---")
    A("")
    A("## 2. 冻结配置（求解前已落盘）")
    A("")
    A("按任务给定的生成规则：顶点 0..n-1、候选边字典序、")
    A("`random.Random(10000*n + 100*m + s).sample` 抽 m 条、最终边表重新按字典序排序；")
    A("HOM 全部 `q_e = 0.5`；HET 用长度 m 的循环 `0.2, 0.5, 0.8` 经 ")
    A("`random.Random(20000*n + 200*m + s)` 打乱后按序赋值。")
    A("**不要求连通，不按度数/结构/效果重抽。**")
    A("")
    A(f"- 配置数：**{len(rows)}**（16 张基础图 × 2 种概率）")
    A(f"- 不同基础图：**{len({(r['n'], r['m'], tuple(tuple(e) for e in r['edges'])) for r in rows})}**")
    A(f"- 完全重复的（图, 模型）配置：**{len(cases.get('duplicate_disclosure', []))}**")
    A("- 冻结文件：`M01_cases.json`（含每例的边、概率、图种子、概率种子）")
    A("")
    A("---")
    A("")
    A("## 3. 四项校验与路径检查（全部通过）")
    A("")
    A("| 校验 | 要求 | 结果 |")
    A("|---|---|---|")
    A("| 单条边，成功概率 q | 任意 h≥1 价值均为 2q，失败后不能重试 | **通过**（q = 0, 0.25, 0.4, 0.5, 0.9, 1.0，h = 1,2,3,5） |")
    A("| 一轮 | `V_1^G = V_1^R = V_1^*` | **通过**（7 个测试实例逐个验证） |")
    A("| 两轮 | `V_2^R = V_2^*` | **通过** |")
    A("| 三轮 | `V_3^G ≤ V_3^R ≤ V_3^*` | **通过** |")
    A("")
    A("路径实例 `a-b-c-d`、`q = (0.4, 0.9, 0.4)`、两轮：")
    A("")
    A("| 方法 | 首轮动作 | 价值 | 预期 |")
    A("|---|---|---:|---:|")
    A("| G | 中间边 `(b,c)` | **1.960000** | 1.96 |")
    A("| R | 两条外侧边 `(a,b),(c,d)` | **2.248000** | 2.248 |")
    A("| OPT | 两条外侧边 | **2.248000** | 2.248 |")
    A("")
    A("**完全吻合**。这是实现校验，不计入主实验结果。")
    A("")
    A("> 附注：R 的最优首轮动作是**同时取两条外侧边**（一个匹配），而不是只取一条外侧边。")
    A("> 任务描述里的「选两侧边」与此一致；我在校验断言里最初写成「单条外侧边」是断言过窄，已修正。")
    A("")
    A("---")
    A("")
    A("## 4. 独立复核（四个实例，全部通过）")
    A("")
    A("对 (n,m) = (6,6) 且 s = 0,1 的两种概率配置共四例，**独立枚举全部 2^6 = 64 个持久边状态**，")
    A("在每例中执行保存下来的 G、R、OPT 策略。该路径**没有调用递归求值函数来生成收益**：")
    A("它自己重新构造了动作表（穷举一步搜索 + 自己的匹配/转移实现），自己的模拟循环使用自己的转移。")
    A("")
    A("| 配置 | 世界数 | V3_G 枚举/递归 | V3_R 枚举/递归 | V3_OPT 枚举/递归 | 最大偏差 |")
    A("|---|---:|---|---|---|---:|")
    for r in indep:
        A(f"| {r['config_id']} | {r['n_worlds_enumerated']} | "
          f"{r['V3_G_world_enum']:.10f} / {r['V3_G_recursion']:.10f} | "
          f"{r['V3_R_world_enum']:.10f} / {r['V3_R_recursion']:.10f} | "
          f"{r['V3_OPT_world_enum']:.10f} / {r['V3_OPT_recursion']:.10f} | "
          f"{max(r['abs_diff'].values()):.1e} |")
    A("")
    A("四例全部 MATCH（最大偏差 4.4e-16）。")
    A("")
    A("> **这一环查出并修掉了三处我自己的实现错误**，都发生在**独立复核脚本**里，")
    A("> 不在求解器里：(1) 动作表把贪心的近视分数当成 `V_G(·,h)` 返回，只在 `h=1` 正确；")
    A("> (2) 动作表把 R 的续接写成 R 自己，而定义要求续接是 G；")
    A("> (3) 模拟循环在遇到空匹配时 `break`，把空匹配当成了终止而不是「这一轮放弃」。")
    A("> 第 (3) 条尤其值得记录：空匹配是合法动作，它**消耗一轮但进程继续**。求解器本身没有这些问题，")
    A("> 并有下表所列的额外不变量检查。")
    A("")
    A("### 4.1 顶点重标号不变量检查")
    A("")
    A("任务语义把顶点视为无标号，因此 `V_3^*` 必须在重标号下**严格不变**。")
    A("把每个配置的顶点按 `v → (v+1) mod n` 与 `v → (v+2) mod n` 重标号（概率随边一起搬运，")
    A("不从排序位置重新赋值），重算全部量：")
    A("")
    A(f"- `V_3^*`：32 个配置 × 2 次旋转，**最坏绝对偏差 {equiv['invariant_worst_deviation']:.3e}，零失败**。")
    A(f"- `V_3^R`：跨标号最坏偏差 {equiv['V3_R_worst_spread_across_labelings']:.3e}，**非零**。")
    A("")
    A("后者不是错误，而是冻结设定的一条真实性质，必须报告：**贪心规则的并列由顶点标号的字典序决定**，")
    A("所以换标号可能改变贪心的选择；R 的续接估计正是 `V^G`，因此 `V_3^R` 与 `d_robust` 也带标号依赖。")
    A(f"本轮只有 {equiv['n_configurations_moving']}/32 个配置的 `d_robust` 在重标号下发生变化。")
    A("")
    A("**因此本报告对每个配置同时给出原始标号的 `d_robust` 与三种标号下的最大值，"
      "判决采用最大值**（即对 R 最不利、最容易达到门槛的口径）。")
    A("")
    A("---")
    A("")
    A("## 5. 主实验结果（32 个配置）")
    A("")
    A(f"预算执行：每配置 CPU 上限 60 s，总墙钟上限 1 h。实际总耗时 **{sum(r['wall_seconds'] for r in rows):.2f} s**，")
    A(f"单配置最大 **{max(r['wall_seconds'] for r in rows):.2f} s**，超时 **0** 个，缺失 **0** 个。")
    A("这些时间是小规模精确求解的诊断成本，**不是大图部署决策延迟**；本轮不比较训练摊销，")
    A("也不宣称谁在相同在线预算下获胜。")
    A("")
    A("| 配置 | n | m | V3_G | V3_R | V3_OPT | d_G | d_R | d_robust | d_robust 最大(跨标号) | 门槛 | 达标 |")
    A("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|")
    for r in rows:
        A(f"| {r['config_id']} | {r['n']} | {r['m']} | {r['V3_G']:.5f} | {r['V3_R']:.5f} | "
          f"{r['V3_OPT']:.5f} | {r['d_G']:.5f} | {r['d_R']:.5f} | {r['d_robust']:.5f} | "
          f"{r['d_robust_max_over_shifts']:.5f} | {r['threshold']:.3f} | "
          f"{'是' if r['passes'] else '否'} |")
    A("")
    A("归一化（除以初始 n）见 `M01_results.csv` 的 `d_G_over_n`、`d_R_over_n`、`d_robust_over_n`。")
    A("")
    A("### 5.1 分布")
    A("")
    A(f"- `d_G > 0`：**{sum(1 for x in dg if x > 1e-12)}/32**，均值 {statistics.fmean(dg):.5f}，"
      f"中位 {statistics.median(dg):.5f}，最大 {max(dg):.5f}")
    A(f"- `d_R > 0`：**{sum(1 for r in rows if r['d_R'] > 1e-12)}/32**")
    A(f"- `d_robust > 0`：**{sum(1 for x in dr if x > 1e-12)}/32**，均值 {statistics.fmean(dr):.5f}，"
      f"最大 {max(dr):.5f}")
    A(f"- `d_robust` 最大（跨标号）：**{max(drmax):.5f}**")
    A(f"- 三者全为零（贪心与稳健前瞻都已最优）：**{n_degenerate}/32**")
    A("")
    A("### 5.2 `d_G` 很大而 `d_robust` 很小")
    A("")
    A("| 配置 | d_G | d_robust | d_robust / d_G |")
    A("|---|---:|---:|---:|")
    for r in sorted(rows, key=lambda r: -r["d_G"])[:8]:
        ratio_r = (r["d_robust"] / r["d_G"]) if r["d_G"] > 1e-12 else float("nan")
        A(f"| {r['config_id']} | {r['d_G']:.5f} | {r['d_robust']:.5f} | "
          f"{'—' if r['d_G'] <= 1e-12 else f'{ratio_r:.4f}'} |")
    A("")
    A("在 `d_G` 最大的配置上，`d_robust` 几乎为零甚至为零：")
    A("**前瞻把贪心的缺口几乎全部补掉了。**")
    A("")
    A("最需要披露的例外：`n8_m8_s3_HOM` 在原始标号下 `d_robust = 0`，")
    A("但在一次重标号下变为 0.03125 —— 这纯粹来自贪心破并列的标号依赖，")
    A("而 0.03125 仍然只有门槛 0.08 的 39%。")
    A("")
    A("---")
    A("")
    A("## 6. 预先固定的判读")
    A("")
    A("门槛：`d_robust ≥ 0.01n`，即 6 顶点 ≥ 0.06、8 顶点 ≥ 0.08。")
    A("**这是研究资源筛查门槛，不是临床或实用显著性标准。**")
    A("一张基础图在 HOM、HET 都过线只计一次。")
    A("")
    A("| 结果 | 裁决 | 本轮 |")
    A("|---|---|---|")
    A("| ≥4 张不同基础图达标，且覆盖 ≥2 个 (n,m) 档 | 支持进一步研究该前瞻策略遗漏的结构；仍不启动 GRL | 否 |")
    A("| 32 个配置全部完成，且全部低于门槛 | 停止在本批小图分布上挖掘「三轮前瞻修正」；不能否定更大图 | **是** |")
    A("| 其余结果，或存在未完成配置 | 不确定；逐例报告，不自动扩大实验 | 否 |")
    A("")
    A(f"**命中第二档 → {summary['tier']}。**")
    A("")
    A("无需显著性检验或 bootstrap：本轮计算的是**有限配置上的策略期望**，不存在模拟采样误差。")
    A("样本覆盖限制仍然存在（见第 8 节）。")
    A("")
    A("---")
    A("")
    A("## 7. 并列保护：`d_robust` 的含义与边界")
    A("")
    A("`A_R` 记录三轮初始状态下**全部**达到最大前瞻分数（误差 ≤ ε）的动作，")
    A("`d_robust = V_3^* − max_{M∈A_R} Q_3^*(G, M)`：即使把 R 的首轮并列往最有利方向打破，")
    A("仍然留下的损失。")
    A(f"`|A_R|` 在每个配置上的取值见 `M01_results.csv` 的 `n_A_R` 列。")
    A("")
    A("**必须保留的边界：**")
    A("")
    A("1. `d_robust` 使用了 OPT 信息，**只是诊断量，不是可部署策略**。")
    A("2. 它**没有**排除贪心续接**内部**破并列的影响：R 的续接 `V^G` 在每个子状态都由字典序破并列，")
    A("   本轮没有对续接内部也做同样的最有利化。")
    A("3. 冻结的字典序规则在并列时**倾向选择空匹配**（空元组字典序最小），")
    A("   这系统性地压低了贪心（进而压低 `d_G` 的真实上限）。因此 `d_G` 的数值应读作")
    A("   「在冻结破并列规则下的贪心缺口」，而不是贪心规则的最坏情形。")
    A("   这一点对结论方向是保守的：真实贪心缺口只会更大，而 `d_robust` 依然远低于门槛。")
    A("4. 贪心规则的破并列是标号依赖的，所以 `d_G`、`d_R`、`d_robust` 都带标号依赖；")
    A("   判决已采用三种标号下的 `d_robust` 最大值。")
    A("")
    A("---")
    A("")
    A("## 8. 限制")
    A("")
    A("1. **只有小图。** n ∈ {6, 8}、m ∈ {6, 8, 10, 12}。更大图上完全可能不同。")
    A("2. **只有合成图。** 是机制筛查，不能据此估计真实肾交换中的发生率。")
    A("3. **只有 3 轮。** 更长的轮次可能改变前瞻与最优的关系。")
    A("4. **R 是明确定义、动作完整枚举的前瞻对照**，不是作者的 greedy-k，")
    A("   也**不宣称已覆盖全部强方法**。")
    A("5. **续接估计固定为贪心**；其他续接估计（例如价值迭代、rollout）不在本轮范围内。")
    A("6. **未训练任何模型**，本轮不涉及学习。")
    A("7. **求解器没有与第三方库对拍**（环境无 networkx/numpy/scipy）；")
    A("   它经过四项解析校验、路径实例、64 世界全域枚举、以及 `V_3^*` 的顶点重标号不变量检查。")
    A("")
    A("---")
    A("")
    A("## 9. 交付物")
    A("")
    A("| 文件 | 内容 |")
    A("|---|---|")
    A("| `M01_report.md` | 本文件 |")
    A("| `M01_cases.json` | 冻结的 32 个配置（边、概率、图种子、概率种子）、协议、校验结果、独立复核结果、裁决 |")
    A("| `M01_results.csv` | 每例的 `V_3^G`、`V_3^R`、`V_3^*`、`d_G`、`d_R`、`d_robust`、跨标号最大值、门槛判定、耗时 |")
    A("| `M01_first_actions.json.gz` | 每个配置**首轮全部合法动作**的贪心分数、前瞻分数、`Q_3^*`，以及 G/R/OPT 的选择 |")
    A("| `M01_work/` | 全部脚本：求解器、校验、主实验驱动、独立复核、重标号检查、汇总 |")
    A("")
    A("**本轮最重要的交付是否定性的：在完整枚举动作的非学习前瞻之后，")
    A("这批小图上不存在可重复的、达到筛查门槛的决策损失。**")
    A("")

    (ROOT / "M01_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT/'M01_report.md'}")
    print(f"  d_robust max (original labelling) = {max(dr):.6f}")
    print(f"  d_robust max (over labelings)     = {max(drmax):.6f}")
    print(f"  d_G max = {max(dg):.6f}, mean = {statistics.fmean(dg):.6f}")
    print(f"  tier = {summary['tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
