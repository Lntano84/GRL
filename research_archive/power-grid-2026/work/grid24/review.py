"""Quality-first development review; sealed historical references, no fits."""
import argparse
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid24"
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["pilot", "all"], default="pilot")
args = parser.parse_args()
design = json.loads((OUT / "design.json").read_text())
audit = json.loads((OUT / ("pilot_audit.json" if args.phase == "pilot" else "audit.json")).read_text())
assert audit["passed"]
current = json.loads((OUT / "manifest.json").read_text())
if args.phase == "pilot":
    keys = {(x["week"], x["method"]) for x in design["execution_order"] if x["phase"] == "pilot"}
    current = [x for x in current if (x["week"], x["method"]) in keys]
old = {(x["week"], x["method"]): x for x in json.loads((ROOT / "outputs/grid23/manifest.json").read_text())}


def rows(directory):
    with gzip.open(directory / "steps.jsonl.gz", "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


preserved = json.loads((OUT / "preservation_hashes.json").read_text())
for name, sha in preserved.items():
    assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
records = []
for s in current:
    week, method = s["week"], s["method"]
    ref = old[(week, "FULL")]
    unguarded = old[(week, method.removesuffix("_GUARD"))]
    directory = OUT / "runs" / (week + "__" + method)
    trace = rows(directory)
    reference_trace = rows(ROOT / "outputs/grid23/runs" / (week + "__FULL"))
    calls = [json.loads(p.read_text()) for p in sorted(directory.glob("call[0-9][0-9][0-9].json"))]
    difference = next((a["step"] for a, b in zip(trace, reference_trace)
                       if any(a[k] != b[k] for k in ["before_hash", "action_hash", "after_hash", "raw_cost"])), None)
    joint = s["complete"] and ref["complete"]
    regret = s["cost"] / ref["cost"] - 1. if joint else None
    records.append({"week": week, "method": method, "complete": s["complete"], "steps": s["steps"],
        "full_complete": ref["complete"], "full_steps": ref["steps"], "cost": s["cost"], "full_cost": ref["cost"],
        "cost_regret_joint_complete": regret,
        "lost_full_complete": bool(ref["complete"] and not s["complete"]),
        "earlier_failure": bool(not s["complete"] and s["steps"] < ref["steps"]),
        "first_physical_difference_vs_full": difference,
        "exact_full_trajectory": difference is None and len(trace) == len(reference_trace),
        "n1_calls": s["n1_calls"], "expansions": s["fallbacks"],
        "unsafe_expansions": sum(c["fallback"] and c["guard_reason"] == "shortlist_above_author_safe" for c in calls),
        "empty_expansions": sum(c["fallback"] and c["guard_reason"] == "no_admissible_shortlist" for c in calls),
        "n1_queries": s["n1_forecasts"], "full_n1_queries": ref["n1_forecasts"],
        "n1_query_reduction": 1. - s["n1_forecasts"] / ref["n1_forecasts"],
        "controller_s": s["controller_s"], "full_controller_s": ref["controller_s"],
        "controller_reduction_descriptive": 1. - s["controller_s"] / ref["controller_s"],
        "forecast_s": sum(c["forecast_s"] for c in calls),
        "old_unguarded_cost": unguarded["cost"], "old_unguarded_complete": unguarded["complete"],
        "cost_gain_vs_unguarded_joint_complete": (1. - s["cost"] / unguarded["cost"]) if s["complete"] and unguarded["complete"] else None,
        "decision_p95_s": sorted(r["decision_s"] for r in trace)[math.ceil(.95 * len(trace))-1],
        "timing_scope": "Single new instrumented execution versus sealed historical timing; not a controlled speed benchmark."})
groups = []
for method in sorted({r["method"] for r in records}):
    rr = [r for r in records if r["method"] == method]
    completed = [r["cost_regret_joint_complete"] for r in rr if r["cost_regret_joint_complete"] is not None]
    group = {"method": method, "weeks": len(rr), "complete_weeks": sum(r["complete"] for r in rr),
        "joint_complete": len(completed), "mean_cost_regret": statistics.mean(completed) if completed else None,
        "worst_cost_regret": max(completed) if completed else None,
        "lost_completed": sum(r["lost_full_complete"] for r in rr),
        "earlier_failure": sum(r["earlier_failure"] for r in rr),
        "mean_n1_query_reduction": statistics.mean(r["n1_query_reduction"] for r in rr),
        "pooled_n1_query_reduction": 1. - sum(r["n1_queries"] for r in rr) / sum(r["full_n1_queries"] for r in rr),
        "mean_controller_reduction_descriptive": statistics.mean(r["controller_reduction_descriptive"] for r in rr),
        "exact_full_paths": sum(r["exact_full_trajectory"] for r in rr)}
    group["development_quality_gate"] = bool(completed and group["lost_completed"] == 0 and
        group["earlier_failure"] == 0 and group["mean_cost_regret"] <= .005 and group["worst_cost_regret"] <= .01)
    groups.append(group)
pilot_pass = len(records) == 2 and all(r["complete"] and r["cost_regret_joint_complete"] is not None and
                                     r["cost_regret_joint_complete"] <= .01 and r["n1_query_reduction"] > 0 for r in records)
result = {"phase": args.phase, "pilot_gate_passed": pilot_pass if args.phase == "pilot" else None,
          "records": records, "groups": groups, "preservation_checks": len(preserved),
          "new_fits": 0, "reserved_family_evaluations": 0, "scope": "Development-only quality-first heuristic revision, no GNN-specific or production safety claim."}
(OUT / ("pilot_review.json" if args.phase == "pilot" else "comparison.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
lines = ["# GRID24：原安全阈值触发的缓存完整搜索", "",
         "优先修复质量。仅使用原 LJN rho_safe=0.9；不按时间切换，不扫描阈值，不改旧权重。", "",
         "| 方法 | 周 | 完成 | 对 FULL 成本差 | 搜索查询减少 | 完整搜索回退 | 物理路径与 FULL 相同 |",
         "|---|---|---:|---:|---:|---:|---:|"]
for r in records:
    cost = "不作成本比较" if r["cost_regret_joint_complete"] is None else f'{100*r["cost_regret_joint_complete"]:+.4f}%'
    lines.append(f'| {r["method"]} | {r["week"]} | {r["complete"]} | {cost} | {100*r["n1_query_reduction"]:.2f}% | {r["expansions"]}/{r["n1_calls"]} | {r["exact_full_trajectory"]} |')
lines += ["", "这是已暴露开发周上的质量修订；相同路径是本批事实，不构成跨场景安全保证。",
          "时间是带记录的新运行与历史运行的描述性对照，不能直接认证提速。查询减少与真实耗时分别报告。",
          "GNN 与原 PPO 享有同一回退规则。当前模块为辅助搜索，不是新 RL 训练。",
          f"旧文件/权重保存哈希检查 {len(preserved)} 项通过。保留 GRID22/23 及所有旧权重。",
          "", "技术依据：已有 Verify+Greedy 混合控制与原 LJN 安全阈值；它们是前作，不是本轮创新主张。",
          "https://arxiv.org/html/2407.19865v1"]
(OUT / ("GRID24_pilot_report.md" if args.phase == "pilot" else "GRID24_report.md")).write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text(("PILOT AUDITED. Remaining frozen runs not started. " if args.phase == "pilot" else "COMPLETE. All frozen runs audited. ") +
                                 "No new fits/RL or reserved-family evaluation.\n", encoding="utf-8")
print(json.dumps({"phase": args.phase, "pilot_gate_passed": result["pilot_gate_passed"], "groups": groups}, indent=2))
