"""Paired serial timing and quality. No selection of winning repeats."""
import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid25"
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["pilot", "all"], default="pilot")
phase = parser.parse_args().phase
audit = json.loads((OUT / ("pilot_audit.json" if phase == "pilot" else "audit.json")).read_text())
assert audit["passed"]
summaries = json.loads((OUT / "manifest.json").read_text())
if phase == "pilot": summaries = [s for s in summaries if s["repeat"] == 0 and s["week"] == "2035-03-05_0"]
assert len(summaries) == (4 if phase == "pilot" else 48)
index = {(s["repeat"], s["week"], s["method"]): s for s in summaries}
preserved = json.loads((OUT / "preservation_hashes.json").read_text())
for name, expected in preserved.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name


def quantile(values, q):
    x = sorted(values)
    return x[math.ceil(len(x)*q)-1] if x else None


records = []
for s in summaries:
    full = index[(s["repeat"], s["week"], "FULL")]
    path = OUT / "runs" / f"r{s['repeat']}__{s['week']}__{s['method']}"
    with gzip.open(path / "steps.jsonl.gz", "rt", encoding="utf-8") as f: rows = [json.loads(x) for x in f]
    calls = [json.loads(p.read_text()) for p in sorted(path.glob("call[0-9][0-9][0-9].json"))]
    assert s["exact_full_trajectory"] and s["complete"] == full["complete"] and s["steps"] == full["steps"]
    assert s["cost"] == full["cost"]
    record = {"week": s["week"], "repeat": s["repeat"], "method": s["method"],
        "complete": s["complete"], "steps": s["steps"], "cost": s["cost"], "exact_full_trajectory": True,
        "controller_s": s["controller_s"], "full_controller_s": full["controller_s"],
        "controller_reduction": 1.-s["controller_s"]/full["controller_s"],
        "controller_cpu_s": s["controller_cpu_s"], "full_controller_cpu_s": full["controller_cpu_s"],
        "search_s": s["n1_source_s"], "full_search_s": full["n1_source_s"],
        "search_reduction": 1.-s["n1_source_s"]/full["n1_source_s"],
        "n1_forecast_s": s["n1_forecast_s"], "all_forecast_s": s["all_forecast_s"],
        "all_forecast_fraction": s["all_forecast_s"]/s["controller_s"],
        "full_search_fraction": full["n1_source_s"]/full["controller_s"],
        "other_control_s": s["controller_s"]-s["n1_source_s"],
        "full_other_control_s": full["controller_s"]-full["n1_source_s"],
        "physical_s": s["physical_s"], "setup_s": s["setup_s"], "bookkeeping_s": s["bookkeeping_s"],
        "controller_plus_physical_reduction": 1.-(s["controller_s"]+s["physical_s"])/(full["controller_s"]+full["physical_s"]),
        "init_inclusive_reduction": 1.-(s["controller_s"]+s["physical_s"]+s["setup_s"]+s["preloop_import_s"])/(full["controller_s"]+full["physical_s"]+full["setup_s"]+full["preloop_import_s"]),
        "decision_p95_s": quantile([r["decision_s"] for r in rows], .95),
        "decision_p99_s": quantile([r["decision_s"] for r in rows], .99),
        "decision_max_s": max(r["decision_s"] for r in rows),
        "search_p95_s": quantile([c["source_s"] for c in calls], .95),
        "n1_queries": s["n1_forecasts"], "full_n1_queries": full["n1_forecasts"],
        "query_reduction": 1.-s["n1_forecasts"]/full["n1_forecasts"],
        "public_forecasts": s["public_forecasts"], "full_public_forecasts": full["public_forecasts"],
        "prior_s": s["prior_s"], "feature_s": s["feature_s"], "forward_rank_s": s["forward_rank_s"]}
    records.append(record)
groups = []
for method in sorted({r["method"] for r in records}):
    rr = [r for r in records if r["method"] == method]
    by_week = []
    for week in sorted({r["week"] for r in rr}):
        w = [r for r in rr if r["week"] == week]
        by_week.append({"week": week, "mean_controller_reduction": statistics.mean(r["controller_reduction"] for r in w),
            "min_controller_reduction": min(r["controller_reduction"] for r in w),
            "max_controller_reduction": max(r["controller_reduction"] for r in w),
            "controller_faster_repeats": sum(r["controller_reduction"] > 0 for r in w),
            "mean_search_reduction": statistics.mean(r["search_reduction"] for r in w), "repeats": len(w)})
    by_repeat = [{"repeat": rep, "pooled_controller_reduction": 1.-sum(r["controller_s"] for r in rr if r["repeat"]==rep)/sum(r["full_controller_s"] for r in rr if r["repeat"]==rep)} for rep in sorted({r["repeat"] for r in rr})]
    groups.append({"method": method, "runs": len(rr), "weeks": by_week, "repeat_pooled": by_repeat,
        "pooled_controller_reduction": 1.-sum(r["controller_s"] for r in rr)/sum(r["full_controller_s"] for r in rr),
        "week_macro_controller_reduction": statistics.mean(w["mean_controller_reduction"] for w in by_week),
        "pooled_search_reduction": 1.-sum(r["search_s"] for r in rr)/sum(r["full_search_s"] for r in rr),
        "pooled_query_reduction": 1.-sum(r["n1_queries"] for r in rr)/sum(r["full_n1_queries"] for r in rr),
        "pooled_full_search_fraction": sum(r["full_search_s"] for r in rr)/sum(r["full_controller_s"] for r in rr),
        "pooled_other_control_change_s": sum(r["other_control_s"]-r["full_other_control_s"] for r in rr),
        "total_controller_faster_runs": sum(r["controller_reduction"]>0 for r in rr),
        "all_quality_paths_equal": True, "families": len(by_week)})
result = {"phase": phase, "groups": groups, "records": records, "preserved_file_checks": len(preserved),
    "pilot_quality_passed": all(r["exact_full_trajectory"] for r in records) if phase=="pilot" else None,
    "scope": "Instrumented single-worker serial control wall time. CPU-thread, hardware and method order frozen. Development only; no production deployment or GNN-specific advantage guarantee.",
    "repeat_scope": "Three timing repetitions are not three independent dates. No significant or equivalence claim from step counts.",
    "new_model_fits": 0, "reserved_family_evaluations": 0}
name = "pilot_review" if phase=="pilot" else "comparison"
(OUT / (name + ".json")).write_text(json.dumps(result, indent=2), encoding="utf-8")
with (OUT / (name+".csv")).open("w", newline="", encoding="utf-8-sig") as f:
    w=csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)
lines=["# GRID25：统一串行计时", "", "计时包含完整控制调用，候选文件保存和离线验解在计时外；所有方法相同设置。", "",
    "| 方法 | 运行数 | 总控制时间减少 | 搜索时间减少 | 搜索查询减少 | 质量路径相同 |", "|---|---:|---:|---:|---:|---|"]
for g in groups: lines.append(f"| {g['method']} | {g['runs']} | {100*g['pooled_controller_reduction']:+.2f}% | {100*g['pooled_search_reduction']:+.2f}% | {100*g['pooled_query_reduction']:.2f}% | True |")
lines += ["", "本报告使用全部预定重复，不选最快批次。总控制与搜索分开；少查候选不等同于端到端同幅提速。",
    "所有轨迹与对应 FULL 的物理动作/观测/成本逐步相同，包括六月同一步失败。失败周低累计成本不作为收益。",
    "轻量模拟计时、共同控制器断言与候选输入快照仍有开销；这是本机串行控制调用基准，不是生产服务的延迟保证。",
    "四周已用于开发，三次计时不是独立日期样本。未打开任何预留家族；图模型独有增量须对同规则 PPO 另行评估。",
    "", "逐周、逐重复、尾延迟、CPU、初始化与物理步时间详见对应 JSON/CSV。"]
(OUT / ("GRID25_pilot_report.md" if phase=="pilot" else "GRID25_report.md")).write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps({"phase": phase, "groups": groups},indent=2))
