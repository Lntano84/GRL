"""Independent accounting of GRID26 source vs whole-controller savings.

Uses only completed, already-audited development runs; no AC calls or fits.
Accounting terms are not a causal attribution of non-search variation.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid26"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


manifest = read(OUT / "manifest.json")
audit = read(OUT / "audit.json")
comparison = read(OUT / "comparison.json")
assert audit["passed"] and len(manifest) == audit["runs"] == 48
index = {(s["repeat"], s["week"], s["method"]): s for s in manifest}
assert len(index) == 48
methods = sorted({s["method"] for s in manifest})
weeks = sorted({s["week"] for s in manifest})
records = []
source_pairs = []
for method in methods:
    group = [s for s in manifest if s["method"] == method]
    full = [index[(s["repeat"], s["week"], "FULL")] for s in group]
    denom = sum(s["controller_s"] for s in full)
    source_saving = sum(f["n1_source_s"] - s["n1_source_s"] for s, f in zip(group, full)) / denom
    other_saving = sum((f["controller_s"] - f["n1_source_s"]) - (s["controller_s"] - s["n1_source_s"]) for s, f in zip(group, full)) / denom
    total_saving = 1 - sum(s["controller_s"] for s in group) / denom
    assert abs(total_saving - source_saving - other_saving) < 1e-12
    official = next(g for g in comparison["groups"] if g["method"] == method)
    assert abs(total_saving - official["pooled_controller_reduction"]) < 1e-12
    assert all(s["cost"] == f["cost"] and s["steps"] == f["steps"] and s["complete"] == f["complete"] for s, f in zip(group, full))
    repeats = []
    for repeat in range(3):
        rr = [s for s in group if s["repeat"] == repeat]
        ff = [index[(repeat, s["week"], "FULL")] for s in rr]
        repeats.append({"repeat": repeat,
                        "controller_reduction": 1 - sum(s["controller_s"] for s in rr) / sum(s["controller_s"] for s in ff),
                        "search_reduction": 1 - sum(s["n1_source_s"] for s in rr) / sum(s["n1_source_s"] for s in ff)})
    records.append({"method": method, "runs": len(group), "controller_s": sum(s["controller_s"] for s in group),
                    "search_s": sum(s["n1_source_s"] for s in group), "controller_reduction": total_saving,
                    "search_reduction": official["pooled_search_reduction"],
                    "source_accounting_contribution": source_saving, "other_accounting_contribution": other_saving,
                    "query_reduction": official["pooled_query_reduction"],
                    "cpu_reduction": 1 - sum(s["controller_cpu_s"] for s in group) / sum(s["controller_cpu_s"] for s in full),
                    "prior_s": sum(s["prior_s"] for s in group), "feature_s": sum(s["feature_s"] for s in group),
                    "forward_rank_s": sum(s["forward_rank_s"] for s in group),
                    "controller_plus_physical_reduction": 1 - sum(s["controller_s"] + s["physical_s"] for s in group) / sum(s["controller_s"] + s["physical_s"] for s in full),
                    "completion_dates": sum(index[(0, w, method)]["complete"] for w in weeks), "dates": len(weeks),
                    "repeats": repeats})
    if method in ["FULL", "OLDNN128_GUARD"]:
        continue
    for s in group:
        p = index[(s["repeat"], s["week"], "OLDNN128_GUARD")]
        own_dir = OUT / "runs" / f"r{s['repeat']}__{s['week']}__{method}"
        ppo_dir = OUT / "runs" / f"r{s['repeat']}__{s['week']}__OLDNN128_GUARD"
        for path in sorted(own_dir.glob("call[0-9][0-9][0-9].json")):
            own, old = read(path), read(ppo_dir / path.name)
            source_pairs.append({"repeat": s["repeat"], "week": s["week"], "method": method, "call": path.name,
                                 "same_chosen_id": own["chosen_pool_id"] == old["chosen_pool_id"],
                                 "same_query_count": own["public_queries"] == old["public_queries"],
                                 "same_feedback_id_set": {r["pool_id"] for r in own["outcomes"]} == {r["pool_id"] for r in old["outcomes"]},
                                 "same_dispatch_order": [r["pool_id"] for r in own["outcomes"]] == [r["pool_id"] for r in old["outcomes"]]})

assert len(weeks) == 4
assert sum(s["steps"] for s in manifest) == audit["physical_rows"] == 95424
assert sum(s["public_forecasts"] for s in manifest) == 379368
result = {"passed": True, "records": records, "source_pairs": source_pairs,
          "source_pair_checks": len(source_pairs),
          "same_chosen_id": sum(r["same_chosen_id"] for r in source_pairs),
          "same_query_count": sum(r["same_query_count"] for r in source_pairs),
          "same_feedback_id_set": sum(r["same_feedback_id_set"] for r in source_pairs),
          "same_dispatch_order": sum(r["same_dispatch_order"] for r in source_pairs),
          "full_search_fraction": comparison["groups"][0]["pooled_full_search_fraction"],
          "maximum_accounting_residual": max(abs(r["controller_reduction"] - r["source_accounting_contribution"] - r["other_accounting_contribution"]) for r in records),
          "new_forecasts": 0, "new_fits": 0, "reserved_family_evaluations": 0,
          "scope": "Four exposed dates and three timing repetitions, not an independent confirmation. Same-query comparisons do not prove equality of all learned rankings. Decomposition is accounting, not causal identification."}
(OUT / "contribution.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
fields = [k for k in records[0] if k != "repeats"]
with (OUT / "contribution.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows({k: r[k] for k in fields} for r in records)
lines = ["# GRID26：实际提速贡献核对", "", "48条冻结串行回放及保存数据审计完成。模型、K128、rho_safe=0.9与恢复周期均未改变。", "",
         "| 方法 | 搜索时间减少 | 总控制时间减少 | 搜索查询减少 | 完成日期 |", "|---|---:|---:|---:|---:|"]
for r in records:
    lines.append(f"| {r['method']} | {100*r['search_reduction']:+.2f}% | {100*r['controller_reduction']:+.2f}% | {100*r['query_reduction']:.2f}% | {r['completion_dates']}/4 |")
lines += ["", "全部动作、观测、成本与对应FULL逐步相同；六月均在同一步失败。失败周的低累计成本不作为收益。",
          f"FULL中目标搜索占总控制时间{100*result['full_search_fraction']:.2f}%。即使完全消除这部分，也不能据此推导同幅整体提速。", "",
          "## 时间分解（百分点，算术分解，不是因果归因）", "", "| 方法 | 搜索贡献 | 其余控制贡献 | 总控制减少 |", "|---|---:|---:|---:|"]
for r in records[1:]:
    lines.append(f"| {r['method']} | {100*r['source_accounting_contribution']:+.3f} | {100*r['other_accounting_contribution']:+.3f} | {100*r['controller_reduction']:+.3f} |")
lines += ["", "## 重复稳定性", "", "| 方法 | 重复0总控制减少 | 重复1 | 重复2 |", "|---|---:|---:|---:|"]
for r in records[1:]:
    v = [100*x['controller_reduction'] for x in r['repeats']]
    lines.append(f"| {r['method']} | {v[0]:+.2f}% | {v[1]:+.2f}% | {v[2]:+.2f}% |")
lines += ["", f"与PPO逐调用对照：{result['same_chosen_id']}/{result['source_pair_checks']}选择相同，{result['same_query_count']}/{result['source_pair_checks']}查询数相同；两GNN的图结构独有查询收益未建立。执行顺序并不全相同，不能把所有排序预测说成相同。",
          "PPO总控制较慢的汇总读数主要同时含有较大的非搜索耗时变化；不能把这项变化解释成GNN造成的加速。",
          "", "## 可以写与不能写", "", "可以写：在这四个开发日期上，保护搜索复现FULL的质量路径，少查34.09%，搜索墙钟减少约17%–20%。",
          "总控制减少0.88%–1.40%是完整重复的本机测量点估计；重复0反向，因此尚不能写成稳定、显著的端到端提速。",
          "不能写：节省34%整体延迟、GNN稳定超过PPO、已建立独立泛化或RL增量。",
          "新训练0、封存日期评价0。软件测量、共同断言和轻量快照仍有开销；不是生产服务延迟保证。", "",
          "下一步保留质量保护与当前权重，先对排名中重复的静态动作元数据构造做等价性与开销检查；同时给予PPO同样优化，不用工程缓存冒充图模型创新。"]
(OUT / "GRID26_contribution_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETED:48/48 serial rollouts; saved-data audit, paired timing review and contribution arithmetic passed.95424 physical steps,379368 public forecasts.No pending GRID26 process.No new fits/RL/reserved evaluations.\n", encoding="utf-8")
print(json.dumps({"passed": True, "records": records, "source_pair_checks": result["source_pair_checks"],
                  "same_chosen_id": result["same_chosen_id"], "same_query_count": result["same_query_count"],
                  "same_feedback_id_set": result["same_feedback_id_set"], "same_dispatch_order": result["same_dispatch_order"]}, indent=2))
