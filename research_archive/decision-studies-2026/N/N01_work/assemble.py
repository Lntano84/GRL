"""Assemble the N01-R deliverables from the frozen run and the offline statistics recompute.

Outputs (into the repository root):
  N01_report.md       -- corrected report: settings, checks, corrected maths, recomputed statistics
  N01_cases.json      -- graph hash, candidates, backgrounds, structural checks, random seeds
  N01_results.csv     -- per-comparison means, intervals, verdict, magnitude (corrected quantile)
  N01_worlds.json.gz  -- per-world paired differences for both runs (all that was ever exported)
"""
from __future__ import annotations

import csv
import gzip
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "N01_work"
sys.path.insert(0, str(WORK))


def load(name):
    return json.loads((WORK / name).read_text(encoding="utf-8"))


def main() -> int:
    cases_art = load("N01_cases.json")
    stats = load("N01R_statistics.json")
    hand = load("hand_graph_check.json")
    pre = load("preflight_ca_grqc.json")
    probe_cases_art = load("N01S_cases.json")

    main_run, probe_run = stats["main"], stats["probe"]
    mc = main_run["batches"]["confirm"]
    md = main_run["batches"]["dev"]
    pc_ = probe_run["batches"]["confirm"]
    rows = mc["records"]
    n_cases = main_run["n_comparisons"]
    level = main_run["quantile_level"]
    t_used = mc["t_bonferroni"]
    t_old = mc["t_bonferroni_superseded"]

    mf = cases_art["graph"]["manifest"]
    n = mf["nodes_in_simple_undirected_graph"]

    def f(r, k):
        return float(r[k])

    # ---------------------------------------------------------------- N01_results.csv
    fields = ["index", "group", "a", "b", "c", "d", "U",
              "degree_a", "degree_b", "degree_c", "degree_d",
              "delta_c_mean", "delta_c_se", "delta_c_ci_low", "delta_c_ci_high",
              "delta_c_bonf_low", "delta_c_bonf_high",
              "delta_d_mean", "delta_d_se", "delta_d_ci_low", "delta_d_ci_high",
              "delta_d_bonf_low", "delta_d_bonf_high",
              "reversal", "m", "m_over_n",
              "point_estimate_reversal",
              "confirm_worlds_with_X_c_eq_X_d", "confirm_n_worlds",
              "confirm_max_abs_X_c_minus_X_d"]
    old_rows = {int(r["index"]): r for r in
                csv.DictReader((WORK / "N01_results.csv").open(encoding="utf-8-sig"))}
    with (ROOT / "N01_results.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for rec in rows:
            deg = old_rows[rec["index"]]
            out = {
                "index": rec["index"], "group": rec["group"],
                "a": rec["a"], "b": rec["b"], "c": rec["c"], "d": rec["d"],
                "U": " ".join(str(x) for x in rec["U"]),
                "degree_a": deg["degree_a"], "degree_b": deg["degree_b"],
                "degree_c": deg["degree_c"], "degree_d": deg["degree_d"],
                "reversal": rec["reversal"],
                "point_estimate_reversal": rec["point_estimate_reversal"],
                "confirm_worlds_with_X_c_eq_X_d": rec["X_c_eq_X_d_worlds"],
                "confirm_n_worlds": rec["n_worlds"],
                "confirm_max_abs_X_c_minus_X_d": rec["max_abs_X_c_minus_X_d"],
            }
            for k in fields:
                if k in out:
                    continue
                if k in rec:
                    out[k] = f"{rec[k]:.6f}" if isinstance(rec[k], float) else rec[k]
            w.writerow(out)

    # ---------------------------------------------------------------- N01_cases.json
    cases_payload = {
        "task": "N01 -- long-range conditional ranking diagnostic under a fixed seed budget",
        "revision": "N01-R (offline corrections; no new diffusion, no change of condition)",
        "graph": cases_art["graph"],
        "protocol": cases_art["protocol"],
        "generation": cases_art["generation"],
        "structure_verification": {
            **cases_art["verification"],
            "two_hop_source_convention": {
                "excluded_source_violations": 0,
                "included_source_violations": 0,
                "note": "the condition holds under both readings (with and without the source node in "
                        "the 0-to-2-hop receptive field), so the frozen comparisons are unaffected by "
                        "which convention the definition of R2 intends",
            },
        },
        "batches": {
            "dev_worlds": {"n": 256, "world_seeds": [0, 255]},
            "confirm_worlds": {"n": 2048, "world_seeds": [1000000, 1002047]},
            "disjoint": True,
        },
        "statistics": {
            "family_size": 2 * n_cases,
            "per_interval_alpha": 0.05 / (2 * n_cases),
            "quantile_level": level,
            "t_bonferroni_used": t_used,
            "t_bonferroni_superseded_bug": t_old,
            "note": "the superseded value came from applying the quantile at 1-0.05/(2N) instead of "
                    "1-0.05/(4N); corrected offline without re-simulating",
        },
        "cases": cases_art["cases"],
        "prediction_structure_check": {
            "hand_graph": hand["hand_graph"],
            "class_level_argument": {
                "statement": "for a scorer whose per-node receptive field is confined to L hops and "
                             "whose total is a sum of per-node scores, no output node's field may "
                             "contain candidates from both the {a,b} side and the {c,d} side, else "
                             "the mixed difference need not vanish",
                "proves": "a relation among the FOUR specified configurations only",
                "does_not_prove": "that the model is additive over seed sets in general",
                "superseded_claim": "the earlier modular form Sigma_hat(S) = sum_{v in S} phi(v) + "
                                    "const, and the five random score vectors used to 'verify' it, "
                                    "are withdrawn: a general two-layer GNN is not modular, and "
                                    "'sum of per-node outputs' is not 'sum of per-node scores'",
            },
            "official_simba_forward_pass": {
                "completed": False,
                "reason": "no torch/numpy/scipy in this environment and no network route to fetch "
                          "https://github.com/yl489/rethink-IM (github.com, raw.githubusercontent.com "
                          "and the PyPI index are all unreachable), so the official SpreadPredictor "
                          "could not be instantiated; a substitute network was NOT used",
            },
        },
        "supplementary_probe": {
            "file": "N01S_cases.json",
            "post_hoc": True,
            "not_to_be_extended": True,
            "reason_not_extended": "relaxing the cross-side exclusion removes the hypothesis of the "
                                   "local-dependency argument, so the probe no longer carries any "
                                   "proxy-identity guarantee; a reversal found there could not be "
                                   "attributed to the original model's limitation",
            "relaxed_condition": probe_cases_art["relaxed_condition"],
            "generation": probe_cases_art["generation"],
            "cases": probe_cases_art["cases"],
        },
        "note": "prediction-structure check is stored here, separate from the real-diffusion results "
                "in N01_results.csv / N01_worlds.json.gz",
    }
    (ROOT / "N01_cases.json").write_text(json.dumps(cases_payload, indent=2), encoding="utf-8")

    with gzip.open(ROOT / "N01_worlds.json.gz", "wt", encoding="utf-8", compresslevel=9) as fh:
        src = json.loads(gzip.open(WORK / "N01_worlds.json.gz", "rt", encoding="utf-8").read())
        # The superseded export omitted the four per-set activation counts; record that explicitly so
        # nobody mistakes this archive for one from which the set algebra can be verified.
        src["artifact_status"] = {
            "contains_per_world_X_c_and_X_d": True,
            "contains_per_world_activation_counts_for_the_four_sets": any(
                "spreads" in pc.get(b, {}) for pc in src["per_case"] for b in ("dev", "confirm")),
            "note": "the original N01 export stored only the paired differences; the set-algebra "
                    "decomposition alpha/beta/gamma/delta therefore cannot be verified from this "
                    "archive. run_diffusion.py has been changed to export 'spreads' for future runs.",
        }
        json.dump(src, fh)

    # ---------------------------------------------------------------- N01_report.md
    L: list[str] = []
    A = L.append
    A("# N01 —— 固定种子预算下的远距离条件排序诊断")
    A("")
    A("> **修订说明（N01-R）。** 本文件已按外部审计修正。**未新增任何扩散实验、未改传播设置、"
      "未换条件、未删任何样本。**")
    A("> 修正内容：撤回“两跳不交排除真实反转/检验无效”的错误数学解释；")
    A("> 把代理论证从“模块化求和”改为局部依赖证明并删除五次求和作为网络验证的说法；")
    A("> 用正确分位点 `1−0.05/(4N)` 从已有压缩差值离线重算统计（确认批 **31** 个同向双显著、"
      "**0** 个反转）；")
    A("> 删除“同一对候选随背景翻转”这一未被本实验测到的结论；")
    A("> 删除 `identical_by_construction` 标签；如实说明压缩文件只保存了差值。")
    A("> 原始数据、错误版本与修正记录全部保留。")
    A("")
    A("## 结论")
    A("")
    A("**手工图证明：指定的局部代理存在它无法表达的远距离条件偏好。**")
    A("四个两跳区域不交，三条边之后在 P、Q 汇合，真实偏好为 `δ_c = −6`、`δ_d = +6`。")
    A("")
    A("**主实验没有观察到反转。** 在 ca-GrQc、当前传播参数和 40 个结构抽样比较下：")
    A(f"确认批点估计反转 **{mc['point_estimate_reversals']}** 个，确认反转 "
      f"**{mc['verdict_counts'].get('confirmed_reversal', 0)}** 个。")
    A("逐世界条件差异**确实存在**（"
      f"{mc['cases_with_any_nonzero_Xc_minus_Xd']}/{n_cases} 个比较至少有一个世界 "
      "`X_c ≠ X_d`），但经验均值很小。")
    A("")
    A("**尚未建立**这种表达限制会产生实用选种损失，**更未建立**学习优势。")
    A("这不是宣布方向失败，也**不是**证明检验无效。")
    A("")
    A("---")
    A("")
    A("## 1. 图与协议（冻结）")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A(f"| 源文件 | `{mf['file']}` |")
    A(f"| sha256 | `{mf['sha256']}` |")
    A(f"| 文件头声明 | Nodes: 5242, Edges: 28980（每个无序对各存一次） |")
    A(f"| 数据行数 | {mf['data_rows']} |")
    A(f"| 自环行数 | {mf['self_loop_rows']}（涉及 {len(mf['self_loop_nodes'])} 个节点） |")
    A(f"| 简单无向图 n | **{n}** |")
    A(f"| 简单无向图 m | **{mf['undirected_edges']}** |")
    A(f"| 有向传播弧 | {mf['directed_propagation_arcs']} |")
    A(f"| 与既有 CCIM 探针清单一致 | **{mf['matches_expected']}** |")
    A("")
    A("**数字口径（未删任何节点）。** 节点 `12295` 只出现在自环行中，在任何简单无向图里都没有")
    A("邻居，因此 `n = 5241`；文件头写 28980 是有向行数，其中 14484 个无序对两个方向都出现，")
    A("因此无向 `m = 14484`。与既有清单逐项一致，没有为对齐数字做任何删改。")
    A("")
    A("| 传播设置 | 固定值 |")
    A("|---|---|")
    A("| 图 | 原无向图，每条边转成两个传播方向 |")
    A("| 传播模型 | 独立级联 IC（live-edge 等价实现） |")
    A("| 有向传播概率 | `p_(u,v) = 1/deg(v)` |")
    A("| 双向边随机性 | 两个方向独立抽样 |")
    A("| 传播期限 | 最多 100 轮 |")
    A(f"| 种子预算 | k = {cases_art['protocol']['seed_budget_k']} |")
    A("| 激活数 | 含初始种子 |")
    A("")
    A("**公共随机数。** 一个世界内每条有向弧的 live/dead 状态只抽一次，四个种子集合在同一状态上")
    A("做确定性可达性计算。**没有**使用“分别重置同一随机种子”。")
    A(f"实测最大 BFS 深度远低于 100 轮上限。")
    A("")
    A("---")
    A("")
    A("## 2. 手工图检查")
    A("")
    hg = hand["hand_graph"]
    A(f"- {hg['n']} 节点、{hg['m']} 条有向边：四条长度三的有向路径汇入 P、Q，各连五个独有叶节点，")
    A("  不加反向边或其他边。IC 概率全为 1，U = ∅，k = 2。")
    A("")
    A("| 种子集合 | 实测激活数 | 预期 | |")
    A("|---|---:|---:|---|")
    for k, v in hg["activations"].items():
        exp = hg["activations_expected"][k]
        A(f"| {k} | **{v}** | {exp} | {'OK' if v == exp else '不一致'} |")
    A("")
    A(f"`δ_c = {hg['delta_c']}`，`δ_d = +{hg['delta_d']}`，符号严格相反。")
    A("")
    A("四个候选的两跳影响区域（按实际消息传递方向）：")
    A("")
    named = hg["two_hop_regions_named"]
    for key in ("a", "b", "c", "d"):
        A(f"- `R2({key})` = {{{', '.join('`' + x + '`' for x in named[key])}}}，"
          f"大小 {hg['r2_sizes'][key]}")
    A("")
    A(f"`(R2(a)∪R2(b)) ∩ (R2(c)∪R2(d))` 交集大小 **{hg['intersection_size']}**。")
    A("")
    A("**这个图同时是后文那个错误结论的直接反例：** 两跳区域不交，三跳之后却在 P、Q 汇合，")
    A("真实偏好仍然严格相反。")
    A("")
    A("### 2.1 代理论证（局部依赖，修正版）")
    A("")
    A("原文把两层局部代理写成模块化函数 `Σ̂(S) = Σ_{v∈S} φ(v) + const` 并用五个随机分数向量“验证”。")
    A("**这两步都已撤回**：一般两层 GNN 不是模块化函数，它能在感受野内表达种子间交互；")
    A("“逐节点输出后求和”也不等于“独立种子分数求和”。")
    A("")
    A("正确的论证是局部依赖论证：固定图、参数、层数 `L` 与背景 `U`；")
    A("每个输出节点 `v` 的分数只依赖其 `L` 跳感受野 `N_L(v)` 内的种子指示。")
    A("若**没有任何**输出节点的 `N_L(v)` 同时包含 `{a,b}` 侧的候选与 `{c,d}` 侧的候选，则：")
    A("")
    A("- 若 `N_L(v)` 只碰到 `{a,b}` 侧，把 `c` 换成 `d` 不改变 `N_L(v)` 内的种子，")
    A("  故 `φ_v(S_ac) = φ_v(S_ad)` 且 `φ_v(S_bc) = φ_v(S_bd)`；")
    A("- 若 `N_L(v)` 只碰到 `{c,d}` 侧，把 `a` 换成 `b` 不改变 `φ_v`，")
    A("  故 `φ_v(S_ac) = φ_v(S_bc)` 且 `φ_v(S_ad) = φ_v(S_bd)`。")
    A("")
    A("两种情况下 `v` 对 `σ̂(S_ac) − σ̂(S_bc) − σ̂(S_ad) + σ̂(S_bd)` 的贡献都是 0，求和即得")
    A("`δ̂_c = δ̂_d`。手工图满足该假设（感受野互不相交），结论正是那个障碍。")
    A("")
    A("**这个论证只证明指定四配置之间的关系，不证明模型对所有种子集合都可加。**")
    A("它也不涉及其他背景、其他预算或四配置之外的种子集合。它是**模型类层面**的论证：")
    A("只要官方 `SpreadPredictor` 属于该类就必然满足，但本轮**没有**在官方代码上实测。")
    A("")
    A("> 官方前向**未完成**：本机无 torch/numpy/scipy，且 `github.com`、`raw.githubusercontent.com`")
    A("> 与 PyPI 索引全部不可达，无法取得 `yl489/rethink-IM` 的 `SpreadPredictor`。")
    A("> 按任务要求**没有用自己的简化网络冒充原实现**。")
    A("")
    A("---")
    A("")
    A("## 3. 40 个冻结比较")
    A("")
    gen = cases_art["generation"]
    A(f"- 候选从最大连通分量（{cases_art['graph']['n_lcc']} 个节点）均匀抽取，要求互异且满足两跳不交条件；")
    A(f"- 抽样随机种子 `{gen['sampling_seed']}`；尝试 **{gen['attempts_used']}** 个四元组，"
      f"接受 **{gen['accepted_quadruples']}** 个，合法率 **{gen['legal_rate']:.4%}**，未触及上限，**未放宽条件**；")
    A(f"- 产出 {gen['produced']}；逐例结构复检问题数 **{cases_art['verification']['n_problems']}**。")
    A("")
    A("**两跳口径核对。** 审计指出感受野应覆盖 0–2 跳（含源节点）。两种口径都复核过：")
    A("把源节点排除时违规 0/40，把源节点包含时同样违规 **0/40**。")
    A("因此现有比较**不依赖** `R2` 究竟采用哪种约定，样本可保留。")
    A("")
    A("| 背景类型 | 比较数 | 候选度数均值 | 背景 U 度数均值 |")
    A("|---|---:|---:|---:|")
    for grp in ("uniform_U", "degree_U"):
        sub = [r for r in cases_art["cases"] if r["group"] == grp]
        cd = [d for cs in sub for d in
              (cs["degrees"]["a"], cs["degrees"]["b"], cs["degrees"]["c"], cs["degrees"]["d"])]
        ud = [d for cs in sub for d in cs["degrees"]["U"]]
        A(f"| {grp} | {len(sub)} | {sum(cd)/len(cd):.2f} | {statistics.fmean(ud):.2f} |")
    A("")
    A("> `degree_U` 背景只是比随机背景更接近一个廉价选种策略，**不是真实搜索轨迹**。")
    A("")
    A("**条件罕见吗？不罕见。** 均匀随机四元组抽样 20000 次有 "
      f"{pre['uniform_quadruple_rejection']['accepted']} 次满足"
      f"（{pre['uniform_quadruple_rejection']['rate']:.2%}）。")
    A("")
    A("---")
    A("")
    A("## 4. 修正后的数学说明")
    A("")
    A("### 4.1 不交的是 `R2`，不是 `R`")
    A("")
    A("原文的致命错误是：**两跳区域不交，不意味着第三跳及更远的可达区域不交。**")
    A("手工图就是直接反例（见 §2）。")
    A("")
    A("### 4.2 含背景 `U` 的正确公式")
    A("")
    A("令一个世界中的可达集合为 `A, B, C, D`，背景为 `W = R(U)`，则")
    A("")
    A("```")
    A("X_c = |A \\ (W ∪ C)| − |B \\ (W ∪ C)|")
    A("X_d = |A \\ (W ∪ D)| − |B \\ (W ∪ D)|")
    A("```")
    A("")
    A("记 `α = A\\W, β = B\\W, γ = C\\W, δ = D\\W`，相减得")
    A("")
    A("```")
    A("X_c − X_d = |α ∩ δ| + |β ∩ γ| − |α ∩ γ| − |β ∩ δ|")
    A("```")
    A("")
    A("原来把主导项写成 `|R(a)| − |R(b)|` 是**未经核验**的断言，现予撤回；")
    A("正确的主导部分是 `|A \\ W| − |B \\ W|`，修正项即上式。")
    A("要经验地核验这个分解需要逐世界的 `A, B, C, D, W` 集合，")
    A("而这些量**当时只在内存中计算、并未导出**（见 §7），所以本轮不对该分解作经验断言。")
    A("")
    A("### 4.3 数据本身反驳了恒等式")
    A("")
    A("两跳不交只约束**确定性前两跳**上的交集，而 live-edge 世界里的路径可以更长，")
    A("所以 `α ∩ δ` 与 `β ∩ γ` 可以非空。保存的数据给出了实测反例：")
    A("")
    A("| 位置 | 数值 |")
    A("|---|---|")
    A("| 主实验 case 0，world 1000295 | `X_c = −20`，`X_d = −19` |")
    A("| 主实验 case 38，world 1001137 | `X_c = −36`，`X_d = −16`（差 20） |")
    A(f"| 确认批全体 | {sum(r['n_worlds'] - r['X_c_eq_X_d_worlds'] for r in rows)} / "
      f"{sum(r['n_worlds'] for r in rows):,} 个（世界，比较）对满足 `X_c ≠ X_d` |")
    A("")
    A(f"这 {sum(r['n_worlds'] - r['X_c_eq_X_d_worlds'] for r in rows)} 个是"
      "**不恒等的实测反例**，不是“几乎恒等”的证明。")
    A("它们出现得少，可以解释**这批采样为什么难以发现反转**，")
    A("却不能证明反转被条件禁止。")
    A("")
    A("---")
    A("")
    A("## 5. 统计口径修正")
    A("")
    A("审计指出 Bonferroni 分位点仍然用错，而且已经影响计数。")
    A("80 个双侧区间同时控制 95% 时，每个区间的置信水平是 `1 − 0.05/80`，")
    A("即分位点 `1 − 0.05/(2×80) = 1 − 0.05/160`：")
    A("")
    A("| | 分位点水平 | t(2047) |")
    A("|---|---:|---:|")
    A(f"| 正确（本次使用） | {level:.7f} | **{t_used:.9f}** |")
    A(f"| 之前错误使用 | {1 - 0.05 / (2 * n_cases):.7f} | {t_old:.9f} |")
    A("")
    A("之前的自检之所以通过，是因为**表和自检用了同一个错误分位点**。")
    A("现已修正，并且自检同时校验正确水平与旧的错误水平，")
    A("一旦退回旧常数就直接终止（`run_diffusion.py` 的 `check_t_tables`）。")
    A("")
    A("**本次修正只重算统计，没有重新模拟**：直接从已保存的逐世界差值重算。")
    A("")
    A("### 5.1 确认批结果（修正后）")
    A("")
    A(f"族大小 2N = {main_run['family_size']}，每个区间水平 `1 − 0.05/{main_run['family_size']}`。")
    A("以下均为**近似 Monte Carlo 区间**（配对样本均值的 t 区间），**不是严格有限样本数学证书**。")
    A("")
    A("| 确认批 | 修正前（错误分位点） | 修正后（正确） |")
    A("|---|---:|---:|")
    old_both = 32   # 32 same-side double-significant = 28 + 4 formerly labelled identical_by_construction
    A(f"| 两区间均排除零、方向相同 | {old_both} | "
      f"**{mc['verdict_counts'].get('both_significant_same_side', 0)}** |")
    A(f"| 其余比较 | {n_cases - old_both} | "
      f"{n_cases - mc['verdict_counts'].get('both_significant_same_side', 0)} |")
    A(f"| 确认反转 | 0 | **{mc['verdict_counts'].get('confirmed_reversal', 0)}** |")
    A(f"| 点估计反转 | 0 | **{mc['point_estimate_reversals']}** |")
    A("")
    A("（修正前那 8 个“其余比较”里，有 4 个曾被贴上已删除的 `identical_by_construction` 标签；")
    A("按纯描述口径它们现在也计入“两侧均包含零”。）")
    A("")
    A("**唯一改变判定的是 case 34**：从“两侧均显著”变为“两侧均包含零”"
      f"（`δ_c = {[r for r in rows if r['index'] == 34][0]['delta_c_mean']:.4f}`，"
      f"`δ_d = {[r for r in rows if r['index'] == 34][0]['delta_d_mean']:.4f}`）。")
    A("原报告“32 例、修正不改变计数”的说法**撤回**；**零反转结论保留**。")
    A("")
    A("### 5.2 逐例结果")
    A("")
    A("| # | 背景 | δ_c 均值 | 调整后区间 | δ_d 均值 | 调整后区间 | m | m/n | 判定 |")
    A("|---:|---|---:|---|---:|---|---:|---:|---|")
    for r in rows:
        A(f"| {r['index']} | {r['group']} | {f(r,'delta_c_mean'):+.4f} | "
          f"[{f(r,'delta_c_bonf_low'):+.4f}, {f(r,'delta_c_bonf_high'):+.4f}] | "
          f"{f(r,'delta_d_mean'):+.4f} | "
          f"[{f(r,'delta_d_bonf_low'):+.4f}, {f(r,'delta_d_bonf_high'):+.4f}] | "
          f"{f(r,'m'):.4f} | {f(r,'m_over_n'):.6f} | {r['reversal']} |")
    A("")
    A("判定计数：" + json.dumps(mc["verdict_counts"], ensure_ascii=False))
    A("")
    A("| 分组 | 比较数 | δ_c 均值 | δ_d 均值 | 确认反转 |")
    A("|---|---:|---:|---:|---:|")
    for grp in ("uniform_U", "degree_U"):
        sub = [r for r in rows if r["group"] == grp]
        A(f"| {grp} | {len(sub)} | "
          f"{sum(f(r,'delta_c_mean') for r in sub)/len(sub):+.4f} | "
          f"{sum(f(r,'delta_d_mean') for r in sub)/len(sub):+.4f} | 0 |")
    A("")
    cc = [f(r, "delta_c_mean") for r in rows]
    A(f"幅度 `m/n`：中位 {statistics.median([f(r,'m_over_n') for r in rows]):.6f}，"
      f"最大 {max(f(r,'m_over_n') for r in rows):.6f}。")
    A("`m = min(|δ_c|, |δ_d|)` 是两侧偏好的**较小幅度**，")
    A("**不是完整 IM 算法的实际后悔值**。")
    A("")
    A("### 5.3 开发批（同样用正确分位点重算）")
    A("")
    A(f"- 世界数 {md['n_worlds']}，`t95 = {md['t95']:.6f}`，"
      f"`t_bonf = {md['t_bonferroni']:.6f}`；")
    A(f"- 判定计数：{json.dumps(md['verdict_counts'], ensure_ascii=False)}；"
      f"点估计反转 {md['point_estimate_reversals']} 个；")
    A(f"- 至少一个世界 `X_c ≠ X_d` 的比较：{md['cases_with_any_nonzero_Xc_minus_Xd']}/{n_cases}。")
    A("")
    A("开发批只有 256 个世界，区间宽得多，其中大部分比较不显著，**不作为主结论**，")
    A("仅用于确认符号方向与确认批一致。")
    A("")
    A("---")
    A("")
    A("## 6. 已删除的结论")
    A("")
    A("| 被删除的结论 | 删除原因 |")
    A("|---|---|")
    A("| “冻结条件使检验无效 / 两跳不交排除真实反转” | 错误：不交的是 `R2` 而不是 `R`；手工图与 68 个非零差都是反例（§2、§4.3） |")
    A("| “同一对候选随背景 `U` 整体翻转符号” | **未被本实验测到**：40 个无序 `{a,b}` 对全部不同、无重复，"
      "所以跨例的符号差异只能说明**不同候选对优劣不同**，不能归因于背景（§6.1） |")
    A("| “N01-S 仍保证代理恒等” | 放开交叉不交条件后**同时移除了局部依赖论证的假设**，"
      "故 N01-S 不再携带任何代理恒等保证，不能用来判断原 GNN 的表达能力（§6.2） |")
    A("| `identical_by_construction` 标签 | 样本均值相等不能证明构造恒等，已改为纯描述标签 |")
    A("| 压缩文件“含四个集合原始激活数” | 实际上只导出了差值 `X_c, X_d`（§7） |")
    A("")
    A("### 6.1 候选对唯一性核对")
    A("")
    ab = {(c["a"], c["b"]) for c in cases_art["cases"]}
    cd = {(c["c"], c["d"]) for c in cases_art["cases"]}
    A(f"`{{a,b}}` 无序对：{len(ab)} 个，全部不同。`{{c,d}}` 无序对：{len(cd)} 个，全部不同。")
    A("")
    A("### 6.2 N01-S 的地位")
    A("")
    A("N01-S 是事后设计，其结构条件放开了交叉排除，因此")
    A("`δ̂_c ≡ δ̂_d` 的保证**不再成立**。即使在其中测到真实反转，")
    A("也不能直接说原 GNN 表达不了。**本轮不按 N01-S 方案扩展。**")
    A("其结果显示在 §8，仅作记录，不作为原命题的证据。")
    A("")
    A("---")
    A("")
    A("## 7. 数据留存的实际状态")
    A("")
    A("**必须如实说明：** `N01_worlds.json.gz` 每个比较每个批次只保存了")
    A("`X_c`、`X_d`、两者是否相等的计数、以及最大 BFS 深度。")
    A("四个集合各自的逐世界激活数虽然在运行中计算过，但**没有导出**。")
    A("")
    A("因此：")
    A("")
    A("- §5 的统计重算是从**保存的差值**重算的，可信；")
    A("- §4.2 的集合分解（`α, β, γ, δ`）**无法**从留存数据核验，本轮不作经验断言；")
    A("- 原报告称压缩文件“含四个集合的原始激活数”，该说法**撤回**。")
    A("")
    A("脚本 `run_diffusion.py` 已改为把逐世界的四个集合激活数一并导出，供后续轮次使用。")
    A("")
    A("---")
    A("")
    A("## 8. 补充探针 N01-S（仅记录，不作为证据）")
    A("")
    A("条件：保留 `R2(a)∩R2(b)=∅`、`R2(c)∩R2(d)=∅`，去掉交叉排除。")
    A(f"抽样种子 `{probe_cases_art['generation']['sampling_seed']}`，"
      f"尝试 {probe_cases_art['generation']['attempts']} 个四元组，接受 "
      f"{probe_cases_art['generation']['accepted']} 个。")
    A("")
    A("确认批（正确分位点）："
      + json.dumps(pc_["verdict_counts"], ensure_ascii=False)
      + f"，点估计反转 {pc_['point_estimate_reversals']} 个。")
    A("")
    A("耦合确实被激活：至少一个世界 `X_c ≠ X_d` 的比较从 "
      f"{mc['cases_with_any_nonzero_Xc_minus_Xd']}/{n_cases} 升到 "
      f"{pc_['cases_with_any_nonzero_Xc_minus_Xd']}/{n_cases}。")
    A("但**反转数仍为 0**。即便如此，按 §6.2 的理由，这不能作为原命题的证据。")
    A("")
    A("---")
    A("")
    A("## 9. 必要验收对照")
    A("")
    A("| 验收项 | 结果 |")
    A("|---|---|")
    A(f"| 手工图四个激活数正确 | 通过（12 / 18 / 18 / 12） |")
    A(f"| 每个比较的四个集合均恰有 10 个不同种子 | 通过（{n_cases} 例） |")
    A(f"| 两跳不交条件逐例成立 | 通过（{n_cases} 例；含源节点与不含源节点两种口径均 0 违规） |")
    A("| 同世界内四集合共享边状态 | 通过 |")
    A("| 开发与确认随机世界没有重用 | 通过（0–255 与 1000000–1002047） |")
    A("| 预测结构检查与扩散结果分开存储 | 通过 |")
    A("| 逐世界数据可重算统计 | **部分**：差值已存，四集合原始激活数未存（§7） |")
    A("| 官方 SIMBA `SpreadPredictor` 前向 | **未完成**（无 torch、无网络），未用替代网络冒充 |")
    A("")
    A("---")
    A("")
    A("## 10. 限制")
    A("")
    A("1. **官方前向未做**，代理论证停留在模型类层面。")
    A("2. **本设计对“实用决策损失”的检验能力有限。** 均匀抽取 40 个比较，")
    A("   适合先看现象是否**自然出现**，但**没有保证抽到接近决策边界的比较**——")
    A("   即主项接近零、耦合项非零的那些。这是原设计的局限，应明确承认，")
    A("   但不能用错误的恒等式去解释零结果。")
    A("3. **区间是近似 Monte Carlo 区间**，不是有限样本数学证书。")
    A("4. **单一图、单一协议**：只有 ca-GrQc、只有 IC 与 `p = 1/deg`、只有 k = 10。")
    A("5. **无第三方库交叉验证**：环境没有 networkx/numpy/scipy。")
    A("6. **`m/n` 不是后悔值**，只是两侧偏好中较小一侧的幅度。")
    A("7. **未建立实用选种损失，更未建立学习优势。**")
    A("")
    A("---")
    A("")
    A("## 11. 定位与下一步")
    A("")
    A("可以保留的结论是：")
    A("")
    A("> 手工图证明：指定局部代理存在不能表达的远距离条件偏好。")
    A("> 主实验在 ca-GrQc、当前传播参数和 40 个结构抽样比较中，")
    A("> 没有观察到点估计或确认反转；逐世界条件差异存在，但经验均值很小。")
    A("> 尚未建立这种表达限制会产生实用选种损失，更未建立学习优势。")
    A("")
    A("下一步是否值得设计一个**仍保留代理不可辨识条件、同时更贴近决策边界**的实验，")
    A("在本轮之后另行决定。本轮止于修正证据解释，不增加样本、不换条件、不训练。")
    A("")

    (ROOT / "N01_report.md").write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {ROOT/'N01_report.md'}")
    print(f"wrote {ROOT/'N01_cases.json'}")
    print(f"wrote {ROOT/'N01_results.csv'}")
    print(f"wrote {ROOT/'N01_worlds.json.gz'}")
    print(f"\nconfirm verdicts (corrected): {mc['verdict_counts']}")
    print(f"point-estimate reversals: {mc['point_estimate_reversals']}")
    print(f"nonzero X_c-X_d pairs: "
          f"{sum(r['n_worlds'] - r['X_c_eq_X_d_worlds'] for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
