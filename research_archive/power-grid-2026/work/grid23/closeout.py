"""Complete-first comparisons, full timing disclosures and archived checkpoints."""
import hashlib
import json
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid23"
D = json.loads((OUT / "design.json").read_text()); F = json.loads((OUT / "finished.json").read_text())
A = json.loads((OUT / "audit.json").read_text()); assert A["passed"]
S = {(r["week"], r["method"]): r for r in F["summaries"]}
def mean(v): return statistics.mean(v) if v else None


def compare(method, reference):
    per = []; lost = []; earlier = []; gained = []
    for week in D["weeks"]:
        a, b = S[(week, method)], S[(week, reference)]
        joint = a["complete"] and b["complete"]
        if b["complete"] and not a["complete"]: lost.append(week)
        if not b["complete"] and not a["complete"] and a["steps"]<b["steps"]-1: earlier.append(week)
        if not b["complete"] and a["complete"]: gained.append(week)
        per.append({"week": week, "method": method, "reference": reference, "complete": a["complete"], "reference_complete": b["complete"],
                    "steps": a["steps"], "reference_steps": b["steps"], "joint_cost_eligible": joint,
                    "cost_gain": (b["cost"]-a["cost"])/abs(b["cost"]) if joint else None,
                    "controller_time_reduction": 1-a["controller_s"]/b["controller_s"] if joint else None,
                    "n1_query_reduction": 1-a["n1_forecasts"]/b["n1_forecasts"] if joint and b["n1_forecasts"] else None})
    eligible = [r for r in per if r["joint_cost_eligible"]]
    group = {"method": method, "reference": reference, "joint_complete_weeks": len(eligible), "lost_complete_weeks": lost,
             "earlier_failures": earlier, "gained_complete_weeks": gained,
             "mean_cost_gain": mean([r["cost_gain"] for r in eligible]),
             "mean_controller_time_reduction": mean([r["controller_time_reduction"] for r in eligible]),
             "mean_n1_query_reduction": mean([r["n1_query_reduction"] for r in eligible if r["n1_query_reduction"] is not None]),
             "worst_cost_regret": max([-r["cost_gain"] for r in eligible], default=None)}
    if reference == "FULL":
        group["auxiliary_screen"] = bool(eligible) and not lost and not earlier and group["worst_cost_regret"]<=.01 and group["mean_controller_time_reduction"]>=.1 and (group["mean_n1_query_reduction"] or 0)>=.3
    return group, per


groups = []; per_week = []
for item in D["methods"]:
    if item["method"] == "FULL": continue
    group, per = compare(item["method"], "FULL"); groups.append(group); per_week += per
extra = []
for reference in ["OLDNN128", "MLP0_128", "BIAS0_128"]:
    gains = []; lost = []
    for week in D["weeks"]:
        a, b, r = S[(week, "GNN0_128")], S[(week, "GNN1_128")], S[(week, reference)]
        if r["complete"] and (not a["complete"] or not b["complete"]): lost.append(week)
        if r["complete"] and a["complete"] and b["complete"]:
            gains.append({"week": week, "gain": (r["cost"]-(a["cost"]+b["cost"])/2)/abs(r["cost"])})
    extra.append({"reference": reference, "joint_complete_weeks": len(gains), "mean_cost_gain": mean([r["gain"] for r in gains]),
                  "lost_complete_weeks": lost, "per_week": gains,
                  "graph_increment_screen": bool(gains) and not lost and mean([r["gain"] for r in gains])>=.005})
result = {"versus_full": groups, "per_week": per_week, "mean_two_GNN_seeds_vs_refs": extra,
          "both_gnn_auxiliary_pass": all(g["auxiliary_screen"] for g in groups if g["method"].startswith("GNN")),
          "graph_increment_pass": all(g["graph_increment_screen"] for g in extra),
          "scope": "Four previously exposed developmental weeks; two training seeds are not independent weeks. Single instrumented walltiming, all failures retained. No new RL or final-test claim."}
(OUT / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
lines = ["# GRID23：共同精确破并列与128候选的闭环修订", "", "冻结24条完整轨迹，四个原开发周、相同环境种子。原PPO其他模块、连续控制和30分钟恢复节奏保留。只改变当前断线时的候选搜索。没有新增训练或RL。", "",
         "所有方法共享端点处理、冷却约束、完整动作向量去重、严格合法性与同状态完整回退。完整搜索实际路径逐步复现GRID19。GNN的两个训练种子均报告，MLP／固定偏置预先固定用种子0。", "",
         "| 方法 | 完成周 | 对完整搜索共同完成周 | 成本平均改善 | 控制时间平均减少 | 搜索查询平均减少 | 丢失完整周 | 辅助角色筛查 |",
         "|---|---:|---:|---:|---:|---:|---|---|"]
for g in groups:
    completed = sum(S[(w,g["method"])]["complete"] for w in D["weeks"])
    def percent(v): return "N/A" if v is None else f"{100*v:.3f}%"
    lines.append(f"| {g['method']} | {completed}/4 | {g['joint_complete_weeks']} | {percent(g['mean_cost_gain'])} | {percent(g['mean_controller_time_reduction'])} | {percent(g['mean_n1_query_reduction'])} | {','.join(g['lost_complete_weeks']) or '无'} | {g['auxiliary_screen']} |")
lines += ["", "成本只在共同完成的整周计算；早失败的低累计成本没有作为优势。逐周步数、存活和异常仍完整记录于comparison.json与各运行summary.json。", "",
          "| 两GNN种子平均对照 | 共同完成周 | 平均成本改善 | 图增量筛查 |", "|---|---:|---:|---|"]
for g in extra: lines.append(f"| {g['reference']} | {g['joint_complete_weeks']} | {percent(g['mean_cost_gain'])} | {g['graph_increment_screen']} |")
lines += ["", f"两个GNN均通过辅助角色筛查：{result['both_gnn_auxiliary_pass']}；对同为128候选的PPO、MLP及固定偏置的图增量筛查：{result['graph_increment_pass']}。这是固定的开发投资门槛，不是显著性、等价性或论文录用保证。",
          "", f"累计{A['physical_rows']:,}物理步、{F['counts']['public_forecasts']:,}公共预测、{A['source_calls']}次真实断线搜索；运行墙钟{F['wall_s']/60:.2f}分钟。独立保存数据审计通过，费用最大残差{A['max_independent_cost_residual']:.4g}。",
          "", "控制时间含特征、先验、模型、排序、仿真以及逐调用记录开销；完整候选日志更大，故这不是无记录生产速度基准。各部分另存，可查看是否为预测查询减少，不能只引用拟合／推理毫秒数。模型加载与物理步计时另列。",
          "", "没有独立AC再运行或重新拟合模型。模型决策拓扑与实际交付拓扑逐项检查，但仿真规则不等于生产安全认证。作者模块N1是断线场景标签，不代表全体N−1安全校验。",
          "", "V0与V1源代码／初始和最终权重／曲线全部封存。现有四周已用于开发且两个日期家族与训练变体重叠；这些结果不能宣称封存泛化确认。后续评价必须按日期家族分离，未证明强化学习的额外作用。"]
if F.get("resumed"):
    recovery = json.loads((OUT / "resume_evidence.json").read_text())
    lines += ["", f"运行曾中断：保留前{recovery['completed_episodes_preserved']}条完整轨迹，归档九月FULL的未完成首次尝试后，仅重跑该未完成轨迹并续跑剩余配置。已完成文件与原冻结代码均经哈希核验未变。中断原因未确定。",
              f"未完成尝试可读到{recovery['partial_readable_rows']}步、至少{recovery['partial_logged_public_forecasts_lower_bound']}次公共预测、至少{recovery['partial_logged_controller_s_lower_bound']:.3f}秒控制计算；gzip未正常关闭，未落盘的在途工作无法准确重建。这部分不能算作零成本。上文运行墙钟是成功交付轨迹之和，不含中断工作与停机间隔；完整记录见resume_evidence.json。"]
lines += ["", "此次同时更换共同的精确破并列规则与候选预算；这是修订后配置的成绩，不是两个因素的独立因果消融。原GRID22与其模型保持封存。"]
(OUT / "GRID23_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE.24 canonical-tie K128 development runs audited and sealed. No pending jobs, new fits or RL in stage23.\n", encoding="utf-8")
files = list(OUT.rglob("*"))+list((ROOT / "work/grid23").rglob("*"))
seal = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and "__pycache__" not in p.parts and p.name not in ["delivery_manifest.json", "closeout.log"]}
(OUT / "delivery_manifest.json").write_text(json.dumps(seal, indent=2), encoding="utf-8")
print(json.dumps({"both_auxiliary_pass": result["both_gnn_auxiliary_pass"], "graph_increment_pass": result["graph_increment_pass"], "groups": groups, "extra": extra, "sealed_files": len(seal)}))
