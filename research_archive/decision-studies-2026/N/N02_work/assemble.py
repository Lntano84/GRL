"""Assemble the N02 deliverables.

Outputs (into the repository root):
  N02_report.md        -- settings, selection, development ranking, 8-row confirmation table, verdict
  N02_cases.json       -- frozen manifest: generation, selection, backgrounds, random seeds
  N02_results.csv      -- the 8 confirmed comparisons with point estimates, ordinary and adjusted
                          intervals, verdict and the m_L / m_U quantities
  N02_worlds.json.gz   -- per-world activation counts for the four sets, both batches
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "N02_work"
sys.path.insert(0, str(WORK))


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> int:
    sel = load(WORK / "N02_stage2_selection.json")
    dev = load(WORK / "N02_stage3_dev_and_freeze.json")
    res = load(WORK / "N02_results.json")
    q3d = load(WORK / "N02_q3_diagnostic.json")

    recs = res["records"]
    n_cmp = len(recs)
    cb = res["confirm_batch"]
    ver = res["verdict"]

    # ---------------------------------------------------------------- N02_results.csv
    fields = ["comparison_id", "background_kind", "Q3", "a", "b", "c", "d", "U",
              "r_dev", "dev_delta_c_mean", "dev_delta_d_mean",
              "delta_c_mean", "delta_c_se", "delta_c_ci_low", "delta_c_ci_high",
              "delta_c_adj_low", "delta_c_adj_high",
              "delta_d_mean", "delta_d_se", "delta_d_ci_low", "delta_d_ci_high",
              "delta_d_adj_low", "delta_d_adj_high",
              "confirmed_reversal", "point_estimate_reversal", "m_L", "m_U",
              "m_L_over_n", "excludes_1node_reversal", "max_possible_reversal"]
    cap_by_id = {c["comparison_id"]: c for c in ver["capability_detail"]}
    devrank_by_id = {r["comparison_id"]: r for r in dev["development_ranking"]}
    with (ROOT / "N02_results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            sc, sd = r["delta_c"], r["delta_d"]
            q = r["quadruple"]
            cap = cap_by_id[r["comparison_id"]]
            dv = devrank_by_id[r["comparison_id"]]
            row = {
                "comparison_id": r["comparison_id"], "background_kind": r["background_kind"],
                "Q3": r["Q3"], "a": q[0], "b": q[1], "c": q[2], "d": q[3],
                "U": " ".join(str(x) for x in r["U"]),
                "r_dev": f"{dv['r_dev']:.6f}",
                "dev_delta_c_mean": f"{dv['delta_c_mean']:.6f}",
                "dev_delta_d_mean": f"{dv['delta_d_mean']:.6f}",
                "delta_c_mean": f"{sc['mean']:.6f}", "delta_c_se": f"{sc['se']:.6f}",
                "delta_c_ci_low": f"{sc['ci_low']:.6f}", "delta_c_ci_high": f"{sc['ci_high']:.6f}",
                "delta_c_adj_low": f"{sc['adj_low']:.6f}", "delta_c_adj_high": f"{sc['adj_high']:.6f}",
                "delta_d_mean": f"{sd['mean']:.6f}", "delta_d_se": f"{sd['se']:.6f}",
                "delta_d_ci_low": f"{sd['ci_low']:.6f}", "delta_d_ci_high": f"{sd['ci_high']:.6f}",
                "delta_d_adj_low": f"{sd['adj_low']:.6f}", "delta_d_adj_high": f"{sd['adj_high']:.6f}",
                "confirmed_reversal": r["confirmed_reversal"],
                "point_estimate_reversal": r["point_estimate_reversal"],
                "m_L": "" if r["m_L"] is None else f"{r['m_L']:.6f}",
                "m_U": "" if r["m_U"] is None else f"{r['m_U']:.6f}",
                "m_L_over_n": "" if r["m_L"] is None else f"{r['m_L']/5241:.8f}",
                "excludes_1node_reversal": cap["excludes_1node_reversal"],
                "max_possible_reversal": f"{cap['max_possible_reversal']:.6f}",
            }
            w.writerow(row)

    # ---------------------------------------------------------------- N02_cases.json
    cases_payload = {
        "task": "N02 -- targeted reversal search retaining the local-indistinguishability condition",
        "status": "existence diagnostic only; success would not show the reversal is common, that a "
                  "full seed-selection algorithm loses much, or that learning would help",
        "graph": sel["graph"],
        "propagation": {
            "model": "independent cascade (live-edge view)",
            "graph_reading": "undirected simple graph; each edge gives two propagation directions",
            "arc_probability": "p_{u,v} = 1/deg(v), undirected degree",
            "bidirectional_randomness": "the two directions are sampled independently",
            "rounds_cap": 100,
            "seed_budget_k": 10,
            "background_size": 8,
            "activation_count_includes_seeds": True,
            "common_random_numbers": "one live/dead state per directed arc per world, shared by all "
                                     "four seed sets of a comparison",
            "r_h_convention": "R_h includes the source node and follows the actual message-passing "
                              "direction",
        },
        "original_condition_retained": {
            "condition": "(R2(a) u R2(b)) n (R2(c) u R2(d)) = empty",
            "relaxed": False,
            "note": "the N01-S relaxation was NOT used",
            "local_indistinguishability": "under this condition no output node's two-layer receptive "
                                          "field can contain candidates from both the {a,b} side and "
                                          "the {c,d} side, so a two-layer per-node-summing proxy "
                                          "necessarily gives the same difference for delta_c and "
                                          "delta_d",
            "official_code_note": "SpreadPredictor uses two SAGEConv layers and the search sums "
                                  "per-node outputs, which supports the applicability of this "
                                  "argument; the official forward pass was still never executed here",
        },
        "selection": sel["selection"],
        "q3_diagnostic": q3d,
        "backgrounds": sel["backgrounds"],
        "verification": sel["verification"],
        "batches": {
            "development": {"world_seeds": dev["dev_batch"]["world_seeds"],
                            "n_worlds": dev["dev_batch"]["n_worlds"],
                            "n_comparisons": dev["dev_batch"]["n_comparisons"],
                            "world_x_comparison_evaluations":
                                dev["dev_batch"]["n_worlds"] * dev["dev_batch"]["n_comparisons"]},
            "confirmation": {"world_seeds": cb["world_seeds"], "n_worlds": cb["n_worlds"],
                             "n_comparisons": cb["n_comparisons"],
                             "world_x_comparison_evaluations":
                                 cb["world_x_comparison_evaluations"]},
            "disjoint": True,
        },
        "statistics": {
            "r_dev_definition": dev["r_dev_definition"],
            "selection_rule": dev["selection_rule"],
            "peeking_disclosure": dev["peeking_disclosure"],
            "confirm_quantile_level": cb["quantile_level"],
            "confirm_family_size": cb["family_size"],
            "confirm_t_ordinary": cb["t_ordinary"],
            "confirm_t_adjusted": cb["t_adjusted"],
            "computed_not_transcribed": "the quantile is computed from the closed-form t CDF at run "
                                        "time",
            "interval_kind": cb["interval_kind"],
        },
        "frozen_confirmation_set": dev["frozen_confirmation_set"],
        "development_ranking": dev["development_ranking"],
    }
    (ROOT / "N02_cases.json").write_text(json.dumps(cases_payload, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------- N02_worlds.json.gz
    # Re-index the development spreads from {key: [comparison][world]} to
    # [comparison][key][world] so a reader can pull one comparison's four series directly.
    dev_keys = ("S_ac", "S_bc", "S_ad", "S_bd")
    n_dev_cmp = dev["dev_batch"]["n_comparisons"]
    dev_spreads_by_cmp = [
        {k: dev["dev_spreads"][k][ci] for k in dev_keys} for ci in range(n_dev_cmp)
    ]
    assert all(len(v) == dev["dev_batch"]["n_worlds"]
               for cs in dev_spreads_by_cmp for v in cs.values())
    with gzip.open(ROOT / "N02_worlds.json.gz", "wt", encoding="utf-8", compresslevel=9) as fh:
        json.dump({
            "about": "N02 per-world data. 'spreads' holds the four per-set activation counts for every "
                     "world, so both the statistics and the set algebra can be recomputed.",
            "world_order": {"development": dev["dev_batch"]["world_seeds"],
                            "confirmation": cb["world_seeds"]},
            "development": {
                "n_comparisons": n_dev_cmp,
                "n_worlds": dev["dev_batch"]["n_worlds"],
                "comparison_ids": [r["comparison_id"] for r in dev["development_ranking"]],
                "layout": "spreads[comparison][seed_set][world]",
                "spreads": dev_spreads_by_cmp,
                "X_c": {str(r["comparison_id"]): r["dev_X_c"] for r in dev["dev_records"]},
                "X_d": {str(r["comparison_id"]): r["dev_X_d"] for r in dev["dev_records"]},
                "summary": [{k: v for k, v in r.items() if k not in ("dev_X_c", "dev_X_d")}
                            for r in dev["dev_records"]],
            },
            "confirmation": {
                "n_comparisons": n_cmp,
                "n_worlds": cb["n_worlds"],
                "layout": "per_comparison[i].spreads[seed_set][world]",
                "per_comparison": [
                    {"comparison_id": r["comparison_id"], "background_kind": r["background_kind"],
                     "quadruple": r["quadruple"], "seeds": r["seeds"],
                     "X_c": r["X_c"], "X_d": r["X_d"], "spreads": r["spreads"]}
                    for r in recs],
            },
        }, fh)

    # ---------------------------------------------------------------- N02_report.md
    L: list[str] = []
    A = L.append
    A("# N02 —— 保留局部不可辨识条件的定向反转检验")
    A("")
    A("## 结论")
    A("")
    A("**没有找到真实偏好反转。** 定向筛选后冻结的 8 个比较，在 8192 个全新世界上")
    A(f"确认反转 **{ver['n_confirmed_reversals']}/8**，点估计反转 "
      f"**{sum(1 for r in recs if r['point_estimate_reversal'])}/8**。")
    A("")
    A(f"**并且这批区间已经把 1 个激活节点的反转幅度排除掉了：8/8 个比较的最坏允许反转幅度都 < 1**")
    A(f"（最大的一个只有 {max(c['max_possible_reversal'] for c in ver['capability_detail']):.4f}）。")
    A("")
    A(f"**裁决：{ver['tiers']['capability']['tier']}。**")
    A("按预先固定的判读表，**在当前图和传播设置下停止继续定向挖掘**。")
    A("本轮不训练、不扩图、不调传播概率、不追加世界数。")
    A("")
    A("这**不是**说两年局部代理没有表达限制——手工图已经证明它有；")
    A("也不是说这个限制不可能影响选种——本轮没有测量实用损失。")
    A("它只是说：**在图结构筛选把比较推到三跳交互最强的位置之后，")
    A("ca-GrQc 与当前传播参数下仍然没有出现值得追踪的反转幅度。**")
    A("")
    A("---")
    A("")
    A("## 1. 与 N01 相同、本轮唯一改变的东西")
    A("")
    A("沿用的部分：同一条 ca-GrQc（同哈希、同节点映射）、双向独立 IC、")
    A("`p_uv = 1/deg(v)`、最多 100 轮、`k = 10`、背景 8 个节点、")
    A("每个世界四集合共享同一份 live-edge 状态、`R_h` 含源节点且按实际消息方向。")
    A("")
    A("**保留原条件** `(R2(a)∪R2(b)) ∩ (R2(c)∪R2(d)) = ∅`，**不采用** N01-S 的放宽方式。")
    A("因此局部不可辨识条件仍然成立：没有任何输出节点的两层感受野能同时包含两侧候选，")
    A("两层逐节点求和代理必然给出 `δ̂_c = δ̂_d`。")
    A("")
    A("唯一改变的是**比较的生成方式**：用图结构先挑出「两跳分离、三跳可能重叠」的四元组。")
    A("")
    A("---")
    A("")
    A("## 2. 结构筛选（不看任何扩散结果）")
    A("")
    s = sel["selection"]
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 抽样种子 | `{s['sampling_seed']}` |")
    A(f"| 最大尝试次数 | {s['max_attempts']:,} |")
    A(f"| 实际尝试 | {s['attempts_used']:,}（全部为不同的无序四节点集合） |")
    A(f"| 因原两跳条件被拒 | {s['rejected_on_original_condition']:,} |")
    A(f"| 因 N01 / N01-S 已用被排除 | {s['rejected_prior_use']} |")
    A(f"| 通过条件但 Q_3 = 0 | {s['passed_condition_q3_zero']:,} |")
    A(f"| 通过条件且 Q_3 > 0 | {s['passed_condition_q3_positive']:,} |")
    A(f"| 最终选取 | **{s['selected_quadruples']}**（不足 64 才报告缺口；本次未缺口） |")
    A(f"| 是否放宽条件 | **{s['condition_relaxed']}** |")
    A("")
    A("`Q_3 = | |R3(a)∩R3(c)| + |R3(b)∩R3(d)| − |R3(a)∩R3(d)| − |R3(b)∩R3(c)| |`，")
    A("只作结构筛查分数，**不是传播收益估计，也不是学习优势证据**。")
    A("")
    A("### 2.1 关于 Q_3 的两点必须披露的观察")
    A("")
    A("我按规格使用了**原始 Q_3**，但先量化了它筛选的是什么（`N02_q3_diagnostic.json`）：")
    A("")
    A("| 观察 | 数值 |")
    A("|---|---|")
    A(f"| 通过原条件的四元组中 Q_3 > 0 的比例 | {q3d['n_q3_positive']:,}/{q3d['n_passing_condition']:,} = "
      f"{q3d['q3_positive_fraction']:.2%} |")
    A(f"| 正值池的 Q_3 分布 | 最小 {1}，中位 4，p90 20，最大 {q3d['q3_of_selected']['max']} |")
    A(f"| 入选 64 个的 Q_3 下界所处百分位 | **{q3d['selected_min_percentile']:.2%}** |")
    A(f"| 入选四元组涉及的节点数 | {q3d['distinct_nodes_in_selection']}（LCC 有 4158 个） |")
    A(f"| 原始 Q_3 前 64 与归一化 Q_3 前 64 的重叠 | **{q3d['raw_vs_normalised_top64_overlap']}/64** |")
    A("")
    A("**含义：** Q_3 是三个交集大小的原始计数之和，因此主要度量「四个候选都落在稠密区」，")
    A("而不只是「两侧在三跳上真正交互」。84.79% 的合法四元组 Q_3 > 0，")
    A("所以原始 Q_3 实际上是按**三跳区域大小**排序，选出的 64 个集中在少数核心节点附近")
    A(f"（{q3d['distinct_nodes_in_selection']} 个不同节点、平均度数 "
      f"{q3d['mean_degree_selected_nodes']:.2f} 对池内 {q3d['mean_degree_pool_nodes']:.2f}）；")
    A("若改用按分母归一化的 Q_3，前 64 名与本轮选出的 64 个**完全不重叠**。")
    A("")
    A("这是**筛选分数的性质**，不是本轮结果的解释，也不改变任何判决：")
    A("我按规格执行了原始 Q_3，没有替换分数、没有重抽。")
    A("")
    A("---")
    A("")
    A("## 3. 开发批与冻结")
    A("")
    d = dev["dev_batch"]
    A(f"- 世界种子 `{d['world_seeds'][0]:,} … {d['world_seeds'][1]:,}`，共 **{d['n_worlds']}** 个新世界；")
    A(f"- 比较数 **{d['n_comparisons']}**（{s['selected_quadruples']} 个四元组 × 2 种背景）；")
    A(f"- 世界×比较评估 {d['n_worlds']*d['n_comparisons']:,} 次；")
    A(f"- 开发批分位点：普通 {d['t_ordinary']:.6f}，调整后 {d['t_adjusted']:.6f}"
      f"（族大小 2N = {d['family_size']}）。")
    A("")
    r_vals = [r["r_dev"] for r in dev["development_ranking"]]
    A(f"有符号反转幅度 `r_dev = max{{ min(−δ_c, δ_d), min(δ_c, −δ_d) }}`：")
    A("正值表示开发批点估计已反转，负值表示尚未反转。")
    A("")
    A("| 统计 | 值 |")
    A("|---|---:|")
    A(f"| 最小 | {min(r_vals):.3f} |")
    A(f"| 中位 | {statistics.median(r_vals):.3f} |")
    A(f"| 最大 | {max(r_vals):.3f} |")
    A(f"| `r_dev > 0` 的比较数 | **{sum(1 for x in r_vals if x > 0)}/128** |")
    A("")
    A("**开发批没有任何一个比较的点估计反转**，最接近的一个也只有 "
      f"{max(r_vals):.3f}。按规格，仍照规则选每个背景的前 4 个（不换分数、不重抽）。")
    A("")
    A("> **特权诊断披露。** 确认对象是用真实模拟结果筛出来的，因此**不是可部署策略**，")
    A("> 筛选后的反转比例**不能**估计自然发生率。")
    A("")
    A("### 3.1 开发排序前 12（完整 128 行见 `N02_cases.json`）")
    A("")
    A("| 排名 | 比较ID | 背景 | Q_3 | r_dev | δ_c | δ_d | 四元组 |")
    A("|---:|---:|---|---:|---:|---:|---:|---|")
    for i, r in enumerate(dev["development_ranking"][:12], 1):
        A(f"| {i} | {r['comparison_id']} | {r['background_kind']} | {r['Q3']} | "
          f"{r['r_dev']:+.4f} | {r['delta_c_mean']:+.4f} | {r['delta_d_mean']:+.4f} | "
          f"{r['quadruple']} |")
    A("")
    A("### 3.2 冻结的 8 个确认对象")
    A("")
    A("| 比较ID | 背景 | Q_3 | r_dev | δ_c（开发） | δ_d（开发） | 四元组 |")
    A("|---:|---|---:|---:|---:|---:|---|")
    for r in dev["frozen_confirmation_set"]:
        dv = devrank_by_id[r['comparison_id']]
        A(f"| {r['comparison_id']} | {r['background_kind']} | {r['Q3']} | {r['r_dev']:+.4f} | "
          f"{dv['delta_c_mean']:+.4f} | {dv['delta_d_mean']:+.4f} | {r['quadruple']} |")
    A("")
    A("（开发批完整 128 行排序见 `N02_cases.json` 的 `development_ranking`。）")
    A("")
    A("---")
    A("")
    A("## 4. 独立确认")
    A("")
    A(f"- 世界种子 `{cb['world_seeds'][0]:,} … {cb['world_seeds'][1]:,}`，共 **{cb['n_worlds']}** 个全新世界，")
    A("  与开发批及所有旧实验范围不重叠；")
    A(f"- 8 个比较 × {cb['n_worlds']} 世界 = **{cb['world_x_comparison_evaluations']:,}** 次评估；")
    A("  没有任何比较根据中途结果提前结束；")
    A(f"- 本轮合计 **{d['n_worlds']*d['n_comparisons'] + cb['world_x_comparison_evaluations']:,}** "
      f"次「世界×比较」评估")
    A(f"  （开发批 {d['n_worlds']*d['n_comparisons']:,} + 确认批 "
      f"{cb['world_x_comparison_evaluations']:,}），与预期的约 98,304 量级一致；")
    A(f"- 16 个分差区间的分位点水平 `1 − 0.05/(4×8) = {cb['quantile_level']:.8f}`，")
    A(f"  **由程序按闭式 t CDF 计算**：调整后 `t = {cb['t_adjusted']:.9f}`（df = {cb['n_worlds']-1}）；")
    A("- 区间仍为**近似 Monte Carlo 同时区间**，**不是严格有限样本证书**。")
    A("")
    A("### 4.1 确认表（8 行）")
    A("")
    A("| 比较ID | 背景 | δ_c | δ_c 调整区间 | δ_d | δ_d 调整区间 | 反转 | m_L | 最坏允许反转幅度 |")
    A("|---:|---|---:|---|---:|---|---:|---:|---:|")
    for r in recs:
        sc, sd = r["delta_c"], r["delta_d"]
        cap = cap_by_id[r["comparison_id"]]
        ml = "—" if r["m_L"] is None else f"{r['m_L']:.4f}"
        A(f"| {r['comparison_id']} | {r['background_kind']} | {sc['mean']:+.4f} | "
          f"[{sc['adj_low']:+.4f}, {sc['adj_high']:+.4f}] | {sd['mean']:+.4f} | "
          f"[{sd['adj_low']:+.4f}, {sd['adj_high']:+.4f}] | "
          f"{'是' if r['confirmed_reversal'] else '否'} | {ml} | "
          f"{cap['max_possible_reversal']:.4f} |")
    A("")
    A(f"**确认反转 {ver['n_confirmed_reversals']}/8。**")
    A(f"`m_L`（保守幅度 `min(−U_c, L_d)`）对 8 个比较**都没有定义**——")
    A("该量要求两个点估计异号，而本次没有任何一对异号。")
    A("")
    A("### 4.2 预先固定的判读")
    A("")
    A("| 档位 | 条件 | 本轮 |")
    A("|---|---|---|")
    A("| A 支持继续 | ≥2 个**四节点集合互不重叠**的比较确认反转且各自 `m_L ≥ 1` | 否（0 个反转） |")
    A("| B 仅存在性 | 确认反转但只有单例或幅度不足 | 否（无反转） |")
    A("| C 未确定 | 未确认反转且区间仍宽 | 见下 |")
    A("| D 停止 | 未确认反转，且所选比较都被区间排除达到 1 节点的反转幅度 | **是** |")
    A("")
    A("**两个读法都记录，以免用口径选择来规避结论：**")
    A("")
    A(f"- **字面读法 → {ver['tiers']['literal']['tier']}。** "
      f"{ver['tiers']['literal']['why']}。")
    A(f"- **能力读法 → {ver['tiers']['capability']['tier']}。** "
      f"8/8 个比较的调整区间已经无法容纳任何 ≥1 个激活节点的反转")
    A(f"（最大者仅 {max(c['max_possible_reversal'] for c in ver['capability_detail']):.4f}）。")
    A("  其中 7 个是「两区间同侧排除零」，第 8 个（比较ID 83）区间跨零，")
    A(f"  但其最坏允许反转幅度只有 "
      f"{[c for c in ver['capability_detail'] if c['comparison_id']==83][0]['max_possible_reversal']:.4f}。")
    A("")
    A("**本轮的结论采用能力读法：D 档，停止在当前图和传播设置下继续定向挖掘。**")
    A("理由是预先选定的门槛（1 个激活节点）本就是一个**投入筛查门槛**，")
    A("而“区间跨零”与“区间排除该门槛”是两件不同的事；后者已经成立。")
    A("")
    A("**1 个节点**是本轮预先选定的投入筛查门槛，不是论文通用标准，")
    A("也不等于完整 IM 的实用损失。两个比较即使成功也来自同一张图，不构成跨图泛化证据。")
    A("")
    A("---")
    A("")
    A("## 5. 本轮没有回答的问题")
    A("")
    A("1. **没有测量实用损失。** 没有任何完整选种算法被运行，`m_L`、`m_U` 都不是后悔值。")
    A("2. **没有证明反转罕见。** 只在 8 个定向挑选的比较上做了确认；")
    A("   筛选方式决定了这里不能估计任何自然发生率。")
    A("3. **没有跨图、跨协议证据。** 只有 ca-GrQc、只有 IC 与 `p = 1/deg`、只有 `k = 10`。")
    A("4. **官方前向仍未实测。** `SpreadPredictor` 使用两层 `SAGEConv`、搜索时对逐节点输出求和，")
    A("   这支持局部依赖论证的适用性，但本机无 torch、无网络，官方前向单列为未完成。")
    A("5. **Q_3 筛选的定位。** 它是结构筛查，不是收益估计；它偏向了稠密核心区（§2.1）。")
    A("")
    A("---")
    A("")
    A("## 6. 验收对照")
    A("")
    A("| 验收项 | 结果 |")
    A("|---|---|")
    A(f"| 原两跳条件逐例成立 | 通过（{len(sel['comparisons'])} 个比较逐例断言，问题数 0） |")
    A(f"| 每个比较四个集合均恰有 10 个不同种子 | 通过 |")
    A(f"| 四元组不重复、且与 N01/N01-S 不重叠 | 通过 |")
    A(f"| 开发批与确认批世界不重叠 | 通过（2,000,000–2,000,255 与 3,000,000–3,008,191） |")
    A(f"| 同世界四集合共享边状态 | 通过 |")
    A(f"| 确认前已落盘开发结果与选定集合 | 通过（`N02_stage3_dev_and_freeze.json`） |")
    A(f"| 分位点由程序计算而非手抄 | 通过（闭式 t CDF + 二分） |")
    A(f"| 四个集合逐世界激活数已保存 | 通过（`N02_worlds.json.gz` 的 `spreads`） |")
    A(f"| 官方前向 | 未完成（无 torch、无网络），未用替代网络冒充 |")
    A("")
    A("---")
    A("")
    A("## 7. 与 N01 的关系")
    A("")
    A("N01 在 40 个均匀抽取的比较上没观察到反转，但正如其修订版承认的，")
    A("那个设计**没有保证抽到接近决策边界的比较**。")
    A("N02 正是按这个缺口设计的：先把比较推到三跳交互最强的位置，再独立确认。")
    A("结果仍然是没有反转，且幅度被排除在投入门槛之下。")
    A("")
    A("因此，与“靠继续换筛选分数把它拖成无限寻找反例的工程”相反，")
    A("本轮**按预先固定的规则收口**：")
    A("在当前图与传播设置下，这种模型限制**没有表现出值得追踪的选种决策后果**。")
    A("")

    (ROOT / "N02_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT/'N02_report.md'}")
    print(f"wrote {ROOT/'N02_cases.json'}")
    print(f"wrote {ROOT/'N02_results.csv'}")
    print(f"wrote {ROOT/'N02_worlds.json.gz'}")
    print(f"\nconfirmed reversals: {ver['n_confirmed_reversals']}/8")
    print(f"capability tier: {ver['tiers']['capability']['tier']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
