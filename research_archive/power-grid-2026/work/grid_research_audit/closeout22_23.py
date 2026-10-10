"""Reconcile two sealed deployment configurations; do not alter their results."""
import gzip
import hashlib
import json
import math
import runpy
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid_research_audit"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def trace(stage, week, method):
    path = ROOT / f"outputs/grid{stage}/runs" / (week + "__" + method) / "steps.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    state = load(ROOT / "outputs/grid_pipeline/continue22_23/state.json")
    assert state["status"] == "COMPLETE"
    batches, checks = {}, 0
    for stage, runs in [(22, 28), (23, 24)]:
        path = ROOT / f"outputs/grid{stage}"
        audit = load(path / "audit.json")
        assert audit["passed"] and audit["runs"] == runs
        for name, sha in load(path / "delivery_manifest.json").items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
            checks += 1
        f, d, c = (load(path / p) for p in ["finished.json", "design.json", "comparison.json"])
        assert len(f["summaries"]) == runs and f["counts"]["model_fits"] == 0
        by = {(s["week"], s["method"]): s for s in f["summaries"]}
        assert len(by) == runs
        for row in c["versus_full"]:
            eligible = [(by[(w, row["method"])], by[(w, "FULL")]) for w in d["weeks"]
                if by[(w, row["method"])]["complete"] and by[(w, "FULL")]["complete"]]
            assert len(eligible) == row["joint_complete_weeks"]
            independent_gain = statistics.mean((b["cost"]-a["cost"])/abs(b["cost"]) for a,b in eligible)
            assert abs(independent_gain-row["mean_cost_gain"]) < 1e-12
            assert abs(max((a["cost"]-b["cost"])/abs(b["cost"]) for a,b in eligible)-row["worst_cost_regret"]) < 1e-12
        batches[stage] = {"finished": f, "design": d, "comparison": c, "audit": audit}
    assert batches[22]["design"]["weeks"] == batches[23]["design"]["weeks"]
    groups = []
    for stage, b in batches.items():
        by = {(s["week"],s["method"]):s for s in b["finished"]["summaries"]}
        for row in b["comparison"]["versus_full"]:
            if row["method"].startswith("GNN"):
                groups.append({"stage":stage, "method":row["method"],
                    "completed_weeks":sum(by[(w,row["method"])]["complete"] for w in b["design"]["weeks"]),
                    **row})
    first_differences, timing = [], []
    physical_keys = ["step","before_hash","action_hash","after_hash","raw_cost","done","complete"]
    for stage,b in batches.items():
        by = {(s["week"],s["method"]):s for s in b["finished"]["summaries"]}
        for week in b["design"]["weeks"]:
            full = by[(week,"FULL")]
            full_rows = trace(stage,week,"FULL")
            for method in [s["method"] for s in b["design"]["methods"] if s["method"].startswith("GNN")]:
                a = by[(week,method)]
                rows = trace(stage,week,method)
                first = next((i for i,(x,y) in enumerate(zip(rows,full_rows),1) if any(x[k]!=y[k] for k in physical_keys)), None)
                first_differences.append({"stage":stage,"week":week,"method":method,
                    "first_physical_difference_step":first,
                    "same_full_physical_trace":first is None and len(rows)==len(full_rows),
                    "method_steps":len(rows),"full_steps":len(full_rows)})
                if a["complete"] and full["complete"]:
                    timing.append({"stage":stage,"week":week,"method":method,
                        "full_controller_s":full["controller_s"],"method_controller_s":a["controller_s"],
                        "full_search_s":full["n1_source_s"],"method_search_s":a["n1_source_s"],
                        "full_other_controller_s":full["controller_s"]-full["n1_source_s"],
                        "method_other_controller_s":a["controller_s"]-a["n1_source_s"],
                        "same_full_physical_trace":first is None and len(rows)==len(full_rows)})
    result = {"passed":True,"sealed_file_hash_checks":checks,"completed_rollouts":52,
        "physical_steps":sum(b["finished"]["counts"]["physical_steps"] for b in batches.values()),
        "public_forecasts":sum(b["finished"]["counts"]["public_forecasts"] for b in batches.values()),
        "gnn_groups":groups,"first_physical_differences":first_differences,"timing_components":timing,
        "new_fits":0,"new_RL":False,"final_holdout_opened":False,
        "scope":"Four exposed development weeks. Reconciles saved artifacts and outcome arithmetic; no independent AC rerun. Two GNN fitting seeds are not independent dates. The deployment revision changed both exact tie handling and budget. Instrumented wall timing is not a causal attribution of all speedup to GNN. Aborted GRID22 work is separately disclosed and is not zero-cost."}
    (OUT / "GRID22_23_reconciliation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    runpy.run_path(str(ROOT / "work/grid22/model_registry.py"),run_name="__main__")
    registry = load(OUT / "power_model_registry.json")
    for version in registry["versions"]:
        for fit in version["fits"]:
            for field in ["checkpoint","initial_checkpoint","curve"]:
                p = ROOT / fit[field]
                assert p.is_file()
                fit[field+"_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    registry["closed_loop"] = {"stages":[22,23],"status":"COMPLETE_AUDITED_SEALED", "runs":52,
        "model_selection":"Same V1 weights; both GNN seeds retained. K32 first-order tie and common native-tie K128 reported separately. No new model or RL training.",
        "results":{str(k):v["comparison"] for k,v in batches.items()}}
    (OUT / "power_model_registry.json").write_text(json.dumps(registry,indent=2),encoding="utf-8")
    text = (OUT / "POWER_MODEL_CHANGELOG.md").read_text(encoding="utf-8")
    text = text.replace("运行中，不能提前宣告运营收益", "已完成审计；K32一GNN种子丢失九月完整周，闭环筛查未过")
    needle = next(line for line in text.splitlines() if line.startswith("| GRID22 |"))
    text = text.replace(needle, needle + "\n| GRID23 | 同一V1权重；共同精确破并列＋K128，24条整周比较 | 两GNN种子完成3/4周，与FULL一致；三月成本仍高8.073%，未过辅助质量门槛；同预算图增量未建立 | 配置修订，不是新模型拟合；仍是开发数据 |")
    (OUT / "POWER_MODEL_CHANGELOG.md").write_text(text,encoding="utf-8")
    lines = ["# GRID22–23 完整收尾（2026-10-10）", "",
        "52条预定闭环轨迹及两次保存数据审计全部完成。旧模型、失败种子、中断片段与修订配置均保留。当前配置仍未通过预先固定的闭环质量与图增量门槛。", "",
        "| 配置 | GNN种子 | 完整周 | 丢失FULL完整周 | 共同完成周平均成本改善 | 最坏成本增加 | 搜索查询平均减少 | 控制时间平均减少 |", "|---|---|---:|---|---:|---:|---:|---:|"]
    for r in groups:
        lines.append(f"| GRID{r['stage']} | {r['method']} | {r['completed_weeks']}/4 | {','.join(r['lost_complete_weeks']) or '无'} | {r['mean_cost_gain']:.3%} | {r['worst_cost_regret']:.3%} | {r['mean_n1_query_reduction']:.3%} | {r['mean_controller_time_reduction']:.3%} |")
    lines += ["", "K128修订消除了本批九月种子1的提前失败，两GNN均在六月与FULL同样于1901步失败。三月所有K128短名单方法成本都为3,203,408.50，高于FULL的2,964,114.38；所以扩大候选并没有解决闭环成本。", "",
        "三月首个物理路径分歧发生在1733步。保存输入匹配诊断中，FULL选309，GNN均选774；FULL候选在两个GNN排序中分别为417、431位，未进入128候选。预测最大ρ仅差0.0005015；近优ρ保留指标不能保证整周运营成本。这个观察不证明全部差距由训练软标签温度、网络深度或单个动作造成。", "",
        "实际任务仍是辅助候选搜索。当前模型预测下一步拓扑风险；连续优化器随后可能改调度、弃能和储能，最终还评价长期成本与存活。训练目标与完整交付目标尚未对齐，这是待检验机制，不是已证明的因果解释。", "",
        "补充保存教师状态诊断使用原LJN固定安全阈值0.9：K128未达到安全阈值时扩展完整搜索，五种排名都保留26/26个FULL选择，查询约减少34%。这是条件诊断，不是实际闭环成绩，也不能作为GNN独有优势；没有在当前配置上追加阈值扫描、全周回放或训练。", "",
        "查询数是实际公开预测调用计数。控制墙钟含记录开销且每配置只有一次顺序测量；非搜索控制耗时也有波动，包括相同物理轨迹的运行，不能把全部墙钟下降归因于GNN。另存组件分解，后续速度确认需控制或重复时序测量。", "",
        f"总计{result['physical_steps']:,}个成功交付轨迹物理步、{result['public_forecasts']:,}次公开预测。GRID22中断另含305个已保存步和至少4,253次公开预测，不能记为零成本；细节保留在GRID22报告及resume_evidence.json。", "",
        "没有重新拟合GNN、MLP或偏置，也没有启动新RL。本轮改的是候选预算和各对照共用的精确破并列规则。V0/V1的初始权重、最终权重与训练曲线在模型注册表中重新核对SHA-256。", "",
        "这四个周已用于开发，两个日期家族与训练变体重叠；预留日期家族的最终评价尚未开始。没有论文成功率或录用保证，也没有把当前配置的失败外推为整个电网学习控制无效。", "",
        "下一项模型改动应先检验合法输入是否充分表示候选所移动的具体发电机、负荷与线路，以及训练偏好是否对应最终的控制后果；继续扩大短名单或叠加网络层数没有由本轮建立依据。这个建议没有作为已启动的训练记录。", "",
        "详细交付：outputs/grid22/GRID22_report.md、outputs/grid23/GRID23_report.md、各自comparison.json/audit.json/delivery_manifest.json；补充核对：GRID22_23_reconciliation.json、GRID23_shared_choice_coverage.json和POWER_MODEL_CHANGELOG.md。"]
    (OUT / "GRID22_23_closeout.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"passed":True,"completed_rollouts":52,"sealed_file_hash_checks":checks,"gnn_groups":groups}))


if __name__ == "__main__":
    main()
