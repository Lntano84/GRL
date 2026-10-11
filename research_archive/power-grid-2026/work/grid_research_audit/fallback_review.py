"""Offline coverage diagnosis, restricted to GRID26 repeat 0 development calls.

No new forecasts, rollouts, model fits, or reserved-family outcomes are read.
Full-pool outcomes are privileged diagnostics, never policy inputs.
"""
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid26_review"
OUT.mkdir(parents=True, exist_ok=True)
assert not (OUT / "fallback_review.json").exists(), "Preserve completed diagnostics"
RUNS = ROOT / "outputs/grid26/runs"
manifest = json.loads((ROOT / "outputs/grid26/manifest.json").read_text())
weeks = sorted({r["week"] for r in manifest if r["repeat"] == 0})
methods = ["GNN0_128_GUARD", "GNN1_128_GUARD", "OLDNN128_GUARD"]
rows, inputs = [], {}


def load(path):
    blob = path.read_bytes()
    inputs[str(path.relative_to(ROOT))] = hashlib.sha256(blob).hexdigest()
    return json.loads(blob)


def best(outcomes, ids=None):
    candidates = [r for r in outcomes if r["strict_admissible"]
                  and (ids is None or r["pool_id"] in ids)]
    # Native rewards in these saved logs are exactly float32 values.
    return max(candidates, key=lambda r: (r["reward"], -r["pool_id"])) if candidates else None


for week in weeks:
    full_dir = RUNS / f"r0__{week}__FULL"
    for path in sorted(full_dir.glob("call[0-9][0-9][0-9].json")):
        full = load(path)
        ff = {r["pool_id"]: r for r in full["outcomes"]}
        assert len(ff) == len(full["representatives"]) == full["public_queries"]
        full_best = best(full["outcomes"])
        assert (None if full_best is None else full_best["pool_id"]) == full["chosen_pool_id"]
        valid = [r for r in full["outcomes"] if r["strict_admissible"]]
        for method in methods:
            own = load(RUNS / f"r0__{week}__{method}" / path.name)
            assert own["state_hash"] == full["state_hash"]
            assert own["rho_limit"] == full["rho_limit"]
            assert own["representatives"] == full["representatives"]
            assert own["chosen_pool_id"] == full["chosen_pool_id"]
            shortlist = set(own["shortlist"])
            prefix = best(full["outcomes"], shortlist)
            prefix_id = None if prefix is None else prefix["pool_id"]
            assert prefix_id == own["shortlist_choice"]
            assert len(own["outcomes"]) == own["public_queries"]
            for result in own["outcomes"]:
                ref = ff[result["pool_id"]]
                for field in ["rho", "reward", "done", "illegal", "ambiguous", "exceptions", "strict_admissible"]:
                    assert result[field] == ref[field], (week, path.name, field)
            safe = own["guard_rho"]
            assert safe == .9
            # Check guard logic independently from the saved fallback flag.
            predicted_fallback = len(shortlist) < len(ff) and (prefix is None or prefix["rho"] >= safe)
            assert predicted_fallback == own["fallback"]
            rows.append({"week": week, "call": path.name, "method": method,
                         "fallback": own["fallback"], "guard_reason": own["guard_reason"],
                         "pool_size": len(ff), "shortlist_size": len(shortlist),
                         "public_queries": own["public_queries"],
                         "complement_queries": sum(r["phase"] == "fallback" for r in own["outcomes"]),
                         "prefix_choice": prefix_id, "full_choice": full["chosen_pool_id"],
                         "same_prefix_choice": prefix_id == full["chosen_pool_id"],
                         "full_winner_in_shortlist": full_best is not None and full_best["pool_id"] in shortlist,
                         "full_winner_rank": None if full_best is None else own["order"].index(full_best["pool_id"]) + 1,
                         "full_best_rho": None if full_best is None else full_best["rho"],
                         "min_valid_rho": min((r["rho"] for r in valid), default=None),
                         "full_has_below_safe_candidate": any(r["rho"] < safe for r in valid),
                         "full_has_admissible_candidate": bool(valid)})

assert len(weeks) == 4 and len(rows) == 78
summary = []
for method in methods:
    group = [r for r in rows if r["method"] == method]
    fallback = [r for r in group if r["fallback"]]
    total = sum(r["public_queries"] for r in group)
    complement = sum(r["complement_queries"] for r in group)
    summary.append({"method": method, "calls": len(group), "fallback_calls": len(fallback),
                    "public_queries": total,
                    "fallback_call_queries": sum(r["public_queries"] for r in fallback),
                    "fallback_call_query_fraction": sum(r["public_queries"] for r in fallback) / total,
                    "complement_queries": complement, "complement_query_fraction": complement / total,
                    "fallback_same_prefix_choice": sum(r["same_prefix_choice"] for r in fallback),
                    "fallback_full_winner_in_shortlist": sum(r["full_winner_in_shortlist"] for r in fallback),
                    "fallback_full_no_admissible": sum(not r["full_has_admissible_candidate"] for r in fallback),
                    "fallback_full_no_below_safe_candidate": sum(not r["full_has_below_safe_candidate"] for r in fallback),
                    "fallback_has_below_safe_but_prefix_missed": sum(r["full_has_below_safe_candidate"] for r in fallback),
                    "guard_reasons": dict(collections.Counter(r["guard_reason"] for r in group))})

result = {"passed": True, "scope": "GRID26 repeat 0, four exposed development dates, 26 distinct search states",
          "summary": summary, "rows": rows, "input_sha256": inputs,
          "new_forecasts": 0, "new_physical_steps": 0, "new_fits": 0, "reserved_evaluations": 0,
          "limitations": ["Full-pool coverage is privileged, not an online stopping certificate.",
                          "Same saved-state action does not authorize removing the safety guard.",
                          "Three rankers evaluated on the same states are not independent state samples.",
                          "No counterfactual wall-clock saving or closed-loop quality claim is derived."]}
(OUT / "fallback_review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
with (OUT / "fallback_per_call.csv").open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
lines = ["# GRID26 补查瓶颈离线诊断", "", "只读重复0的四个已暴露开发日期；26个不同搜索状态。无新增预测、轨迹、拟合或封存日期评价。", "",
         "| 排序器 | 全池补查调用 | 补查调用占查询 | 前128已等同FULL选择 | 前128含FULL最优动作 | 全池也无rho<0.9候选 |",
         "|---|---:|---:|---:|---:|---:|"]
for s in summary:
    lines.append(f"| {s['method']} | {s['fallback_calls']}/{s['calls']} | {100*s['fallback_call_query_fraction']:.2f}% | {s['fallback_same_prefix_choice']}/{s['fallback_calls']} | {s['fallback_full_winner_in_shortlist']}/{s['fallback_calls']} | {s['fallback_full_no_below_safe_candidate']}/{s['fallback_calls']} |")
lines += ["", "等同FULL选择包括两次双方都无合法动作的状态。16次补查中，14次FULL有合法最优动作；两GNN的前128分别包含其中8次、7次，PPO包含5次。漏掉次数分别6、7、9；这是26个开发状态上的事后诊断，不能当作图模型增量确认。", "", "## 可辨识的改进方向", "",
          "若全池也没有低于作者安全阈值的合法候选，更准确的排序不能让当前阈值守卫提前结束；这是守卫与搜索策略问题。",
          "这26个保存状态上，保持K128及现有守卫，理想排序也不能进一步降低查询数量：16个状态必补查全池，其余10个已经只查128。此结论仅覆盖当前调用状态与当前搜索规则。",
          "若全池有安全候选而前128漏掉，才有针对安全候选覆盖率改进排序的直接空间。",
          "前128已选中FULL动作仍触发补查，只说明这些保存状态上补查没有改变动作；它不构成可部署的免查证明，不能用隐藏FULL结果在线决定停止。",
          "当前模型与作者PPO接受相同诊断；不能将共有的补查现象解释为图模型独有优势。", "",
          "保留K128、rho_safe=0.9、当前权重及质量基线。本轮不撤销守卫、不扫阈值、不启动RL。",
          "若继续改进，先设计用合法观测筛选后续查询的质量保护方案，并给予PPO、廉价规则相同信息和预算；不能先降低质量要求来获取提速。",
          "本次诊断事后提出，属于开发分析；没有独立泛化、停止保证或实际新提速结果。"]
(OUT / "FALLBACK_REVIEW.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"passed": True, "summary": summary}, indent=2))
