#!/usr/bin/env python
"""
Emit the markdown tables for stage06_report.md, on the CORRECTED paired aggregation.

    .venv-hs/Scripts/python.exe stage06_tables.py > stage06_tables.md

Every number is machine-derived from `stage06_runs.json`; nothing is hand-transcribed.
See `stage06_aggregate.py` for the paired definition

    G_n(a,b) = (1/4) * sum_d sum_s (J_{n,d,s,b} - J_{n,d,s,a}) / max(1, |J_r(n,d)|)
"""

from __future__ import annotations

import io
import sys
from collections import defaultdict

from stage06_aggregate import (CONFIGS, RANDOMS, THRESHOLD, describe, instance_g,
                               instance_g_random_mean, instance_g_simple, load_runs,
                               pair_cells, scale_of)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

SHORT = {"FULL": "FULL", "EMPTY": "EMPTY", "SHORTAGE-24": "SHORT24", "WINDOW-24": "WIN24",
         "DEPENDENCY-24": "DEP24", "MATCHED-RANDOM-1": "MR1", "MATCHED-RANDOM-2": "MR2",
         "MATCHED-RANDOM-3": "MR3"}
SEEDS = (0, 1)


def fmt(x: float) -> str:
    return "+0.0000" if abs(x) < 5e-5 else "%+.4f" % x


def main() -> None:
    runs = load_runs()
    cells = pair_cells(runs)
    names = sorted({k[0] for k in cells})
    dis_names = sorted({k[1] for k in cells})

    # ---------------- T1
    print("### T1 每个名义实例的分故障修复成本 J_r（归一化基准，故障相关）\n")
    print("| 实例 | %s | %s | J_r 两故障相对差 |" % (dis_names[0], dis_names[1]))
    print("|---|---|---|---|")
    for n in names:
        a = next(c["FULL"]["repair"] for k, c in cells.items()
                 if k[0] == n and k[1] == dis_names[0])
        b = next(c["FULL"]["repair"] for k, c in cells.items()
                 if k[0] == n and k[1] == dis_names[1])
        print("| %s | %.6g | %.6g | %.3f |" % (n, a, b, abs(a - b) / max(1.0, abs(a))))
    print()

    # ---------------- T2
    print("### T2 每个名义实例的平均选中 J（2 个故障 × 2 个求解器种子）\n")
    print("| 实例 | " + " | ".join(SHORT[c] for c in CONFIGS) + " |")
    print("|---|" + "---|" * len(CONFIGS))
    for n in names:
        ks = [k for k in cells if k[0] == n]
        row = "| %s | " % n
        row += " | ".join("%.6g" % (sum(cells[k][c]["cost"] for k in ks) / len(ks))
                          for c in CONFIGS)
        row += " |"
        print(row)
    print()

    # ---------------- T3
    print("### T3 Q1 主比较：DEP24 vs 三个匹配随机的均值（配对后的 G_n）\n")
    v = instance_g_random_mean(cells, "DEPENDENCY-24")
    print("| 实例 | G_n |")
    print("|---|---|")
    for n in names:
        print("| %s | %s |" % (n, fmt(v[n])))
    s = describe(list(v.values()))
    print("| **均值** | **%s** |" % fmt(s["mean"]))
    print()
    print("汇总：n=%d 均值=%s 中位=%s min=%s max=%s 正=%d 负=%d 零=%d ≥2%%=%d\n"
          % (s["n"], fmt(s["mean"]), fmt(s["median"]), fmt(s["min"]), fmt(s["max"]),
             s["pos"], s["neg"], s["zero"], s["ge2pct"]))

    # ---------------- T4
    print("### T4 Q1 按 scale / scale-rho 分组\n")
    print("| 比较 | 分组 | n | 平均 G |")
    print("|---|---|---|---|")
    for tag, a, b in (("DEP24 vs 随机均值", None, None),
                      ("DEP24 vs SHORT24", "DEPENDENCY-24", "SHORTAGE-24"),
                      ("DEP24 vs WIN24", "DEPENDENCY-24", "WINDOW-24"),
                      ("WIN24 vs SHORT24", "WINDOW-24", "SHORTAGE-24"),
                      ("DEP24 vs FULL", "DEPENDENCY-24", "FULL"),
                      ("DEP24 vs EMPTY", "DEPENDENCY-24", "EMPTY")):
        vv = instance_g_random_mean(cells, "DEPENDENCY-24") if a is None \
            else instance_g(cells, a, b)
        for key, sel in (("scale", scale_of),
                         ("scale-rho", lambda n: "%s %s" % (n.split("_")[0],
                                                            n.split("_")[1].replace("rho", "")))):
            g: dict = defaultdict(list)
            for n, x in vv.items():
                g[sel(n)].append(x)
            for k in sorted(g):
                print("| %s | %s %s | %d | %s |"
                      % (tag, key, k, len(g[k]), fmt(sum(g[k]) / len(g[k]))))
    print()

    # ---------------- T5
    print("### T5 Q2 可部署比较：固定规则 vs 冻结简单策略（medium→FULL, large→EMPTY）\n")
    print("| 固定规则 | 平均 G | 中位 | min | max | 正/负/零 | ≥2% | ≤−2% |")
    print("|---|---|---|---|---|---|---|---|")
    for c in CONFIGS:
        vv = instance_g_simple(cells, c)
        s = describe(list(vv.values()))
        print("| always %s | %s | %s | %s | %s | %d/%d/%d | %d | %d |"
              % (SHORT[c], fmt(s["mean"]), fmt(s["median"]), fmt(s["min"]), fmt(s["max"]),
                 s["pos"], s["neg"], s["zero"], s["ge2pct"], s["le_neg2pct"]))
    print()

    # ---------------- T6
    print("### T6 按尺度看，各固定规则相对简单策略的平均 G\n")
    print("| scale | 排序（越靠前越好） |")
    print("|---|---|")
    for sc in ("medium", "large"):
        rows = []
        for c in CONFIGS:
            vv = instance_g_simple(cells, c)
            xs = [x for n, x in vv.items() if scale_of(n) == sc]
            rows.append((sum(xs) / len(xs), SHORT[c]))
        rows.sort(reverse=True)
        print("| %s | %s |" % (sc, " · ".join("%s:%s" % (k, fmt(m)) for m, k in rows)))
    print()

    # ---------------- T7
    print("### T7 分求解器种子的方向稳定性（DEP24 vs 随机均值）\n")
    print("| 分组 | seed0 平均 G | seed1 平均 G | 方向一致 |")
    print("|---|---|---|---|")
    per = {}
    for sd in SEEDS:
        sub = {k: x for k, x in cells.items() if k[2] == sd}
        per[sd] = instance_g_random_mean(sub, "DEPENDENCY-24")
    for key, sel in (("全部", lambda n: "all"), ("medium", scale_of), ("large", scale_of)):
        if key == "全部":
            a0 = list(per[0].values())
            a1 = list(per[1].values())
        else:
            a0 = [x for n, x in per[0].items() if sel(n) == key]
            a1 = [x for n, x in per[1].items() if sel(n) == key]
        m0, m1 = sum(a0) / len(a0), sum(a1) / len(a1)
        print("| %s | %s | %s | %s |"
              % (key, fmt(m0), fmt(m1), "是" if (m0 > 0) == (m1 > 0) else "否"))
    print()

    # ---------------- T8
    print("### T8 匹配随机与 DEP24 的重合率（|R∩D|/24，来自已跑集合）\n")
    print("| 配置 | n | 平均重合 | min | max | 0/1 类别剖面保持 |")
    print("|---|---|---|---|---|---|")
    from stage06_profile_check import profile_class
    from lsp_gen import apply_disruption, build_instance, disruptions_for
    from stage06_run import solution_from_record
    import json
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)
    rels = {(r["state"], r["config"]): [tuple(u) for u in r["release"]] for r in runs}
    ctx = {}
    for cfg in RANDOMS:
        vals, ok, n = [], 0, 0
        for r in runs:
            if r["config"] != cfg:
                continue
            st = r["state"]
            if st not in ctx:
                rec = ST[st]
                inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                         rec["meta"]["seed"])
                dis = next(d for d in disruptions_for(inst)
                           if d.name == rec["disruption"]["name"])
                pert = apply_disruption(inst, dis)
                ctx[st] = solution_from_record(pert, rec["repair"])
            rep = ctx[st]
            D = set(rels[(st, "DEPENDENCY-24")])
            R = set(rels[(st, cfg)])
            vals.append(len(R & D) / len(D))
            ok += int(profile_class(list(R), rep) == profile_class(list(D), rep))
            n += 1
        sv = sorted(vals)
        med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
        print("| %s | %d | %.4f | %.4f | %.4f | %d/%d |"
              % (cfg, n, sum(vals) / n, sv[0], sv[-1], ok, n))
    print()

    # ---------------- T9
    print("### T9 未产生任何改善的运行数（选中成本 == 修复成本）\n")
    print("| 配置 | flat/64 | medium | large |")
    print("|---|---|---|---|")
    for c in CONFIGS:
        sub = [r for r in runs if r["config"] == c]
        flat = [r for r in sub if abs(float(r["selected_cost"]) - float(r["repair_cost"])) <= 1e-9]
        fm = sum(1 for r in flat if r["scale"] == "medium")
        print("| %s | %d/64 | %d | %d |" % (SHORT[c], len(flat), fm, len(flat) - fm))
    print()


if __name__ == "__main__":
    main()
