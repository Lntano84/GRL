"""Week-level descriptive screen; no confidence or online-policy claims."""
import hashlib
import json
import math
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid20"
D = json.loads((OUT / "design.json").read_text()); F = json.loads((OUT / "finished.json").read_text())
A = json.loads((OUT / "audit.json").read_text()); assert A["passed"]
P = json.loads((OUT / "per_state.json").read_text())
meta = {p.stem:json.loads(p.read_text()) for p in (OUT / "inputs/development").glob("*.json")}
methods = D["models"]; per_week = []; macro = []


def avg(values): return statistics.mean(values) if values else None


for method in methods:
    for budget in D["budgets"]:
        group = []
        for week in D["development_weeks"]:
            rows = [r for r in P if r["method"] == method and r["budget"] == budget and r["week"] == week]
            available = [r for r in rows if r["full_has_valid"]]
            total_full = math.fsum(meta[r["state_id"]]["canonical_count"] for r in rows)
            total_queries = math.fsum(r["expected_unique_public_queries"] for r in rows)
            r = {"method":method,"budget":budget,"week":week,"states":len(rows),"full_has_valid_states":len(available),
                 "near_best":avg([float(p["near_best_raw"]) for p in available]),
                 "raw_available":avg([float(p["shortlist_has_valid"]) for p in available]),
                 "fallback":avg([float(p["fallback"]) for p in rows]),
                 "delivered_gap":avg([p["delivered_rho_gap"] for p in available]),
                 "query_reduction":None if not total_full else 1-total_queries/total_full,
                 "mean_queries":avg([p["expected_unique_public_queries"] for p in rows]),
                 "mean_prediction_ranking_s":avg([p["prediction_and_ranking_s"] for p in rows])}
            per_week.append(r); group.append(r)
        combined = {"method":method,"budget":budget,"informative_weeks":sum(r["near_best"] is not None for r in group),"states":sum(r["states"] for r in group)}
        for key in ["near_best","raw_available","fallback","delivered_gap","query_reduction","mean_queries","mean_prediction_ranking_s"]:
            combined[key] = avg([r[key] for r in group if r[key] is not None])
        macro.append(combined)
lookup = {(r["method"],r["budget"]):r for r in macro}


def neural_average(kind,budget):
    peers = [lookup[(f"{kind}_seed{s}",budget)] for s in D["seeds"]]
    return {k:avg([p[k] for p in peers]) for k in ["near_best","fallback","query_reduction","delivered_gap"]}


g = neural_average("GNN",32); mlp = neural_average("MLP",32); old = lookup[("OLDNN",128)]
others = {name:lookup[(name,32)]["near_best"] for name in ["FREQUENCY","RIDGE","OLDNN","LOCAL","PREFIX","RANDOM"]}
others["MLP_AVERAGE"] = mlp["near_best"]
nonempty = {name:v for name,v in others.items() if v is not None}
best_name = max(nonempty,key=lambda k:(nonempty[k],k)) if nonempty else None
week_advantage = []
for week in D["development_weeks"]:
    def retention(names):
        return avg([r["near_best"] for r in per_week if r["week"]==week and r["budget"]==32 and r["method"] in names and r["near_best"] is not None])
    ga = retention([f"GNN_seed{s}" for s in D["seeds"]])
    names = [f"MLP_seed{s}" for s in D["seeds"]] if best_name == "MLP_AVERAGE" else [best_name]
    ba = retention(names)
    week_advantage.append({"week":week,"gnn":ga,"reference":ba,"gain":None if ga is None or ba is None else ga-ba})
valid_metrics = all(v is not None for v in [g["near_best"],g["fallback"],g["query_reduction"],old["near_best"]])
priority = bool(valid_metrics and g["near_best"]>=.90 and g["fallback"]<=.10 and g["query_reduction"]>=.65 and g["near_best"]>=old["near_best"]-.05)
signal = bool(best_name is not None and g["near_best"] is not None and g["near_best"]-nonempty[best_name]>=.05 and sum(r["gain"] is not None and r["gain"]>0 for r in week_advantage)>=3)
family_overlap = [w for w in D["development_weeks"] if w[:10] in {t[:10] for t in D["training_weeks"]}]
result = {"per_week":per_week,"macro":macro,"gnn32":g,"mlp32":mlp,"best_equal32_reference":best_name,
          "week_advantage":week_advantage,"closed_loop_priority":priority,"additional_graph_signal":signal,
          "development_shared_date_families":family_overlap,
          "interpretation":"Development screen only. Missing/rare decisions and shared-date families limit independence. A failed screen warrants fit/role diagnosis and at most one archived revision, not universal failure."}
(OUT / "comparison.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
lines = ["# GRID20：GNN监督拓扑候选排序V0", "", "固定设置：20个训练周、4个开发周；GNN与同信息MLP各两个拟合种子、固定64轮，外部PPO、LOCAL、频率、岭回归、PREFIX和哈希随机作对照。原模型不覆盖；没有RL或学习策略物理回放。", "",
         "主指标为各开发周内的近最佳候选保留率，再按周等权平均；候选最佳ρ差≤0.01。没有有效候选的状态保留在回退/查询账目中。拟合种子和同周状态都不当作独立样本。", "",
         "| 方法 | K | 近最佳保留 | 回退 | 公共查询减少 | 交付ρ差 |", "|---|---:|---:|---:|---:|---:|"]
for r in macro:
    def number(k,percent=False): return "NA" if r[k] is None else f"{r[k]*(100 if percent else 1):.4f}{'%' if percent else ''}"
    lines.append(f"| {r['method']} | {r['budget']} | {number('near_best',True)} | {number('fallback',True)} | {number('query_reduction',True)} | {number('delivered_gap')} |")
lines += ["", f"预设闭环优先筛查：{priority}。同K图表示额外信号：{signal}，参照是全段最强方法{best_name}，不是逐状态oracle。", "",
          f"训练状态数{json.loads((OUT / 'training_finished.json').read_text())['rows']}；开发状态{F['development_states']}。四次神经拟合总计{sum(r['training_s'] for r in F['fits']):.2f}秒，岭回归{F['ridge_fit_s']:.2f}秒；阶段墙钟{F['wall_s']:.2f}秒。教师采集费用在GRID18/19单独公开，不能视为免费标签。",
          "", f"独立记录审计通过：{A['input_states']}个输入、{A['candidate_descriptors']:,}个候选描述、{A['prediction_replays']}次神经复预测、{A['delivery_rows']}行查询/交付重算。训练侧标准化和岭回归方程已复核。没有独立神经重拟合或另一个AC求解器的核验。",
          "", "计时是存储状态下的特征构造及前向/排序；没有执行学习短名单，查询减少是按已存完整反馈计算的预期唯一调用量。不能直接宣称在线加速、成本下降、周存活或N−1安全。缓存回退复用本次状态已查询结果，所有方法共享合法端点修复与完全相同的重复动作消除。",
          "", f"开发周与训练周共享日期家族：{', '.join(family_overlap) if family_overlap else '无'}。不同后缀仍可能相关，本轮不是独立最终确认；后续确认应按日期家族隔离，不能按开发结果换周。", "",
          "复用原两层图编码和同信息MLP，增加候选本地母线编码、已知动作描述及逐线路公开特征。这是预拟合实现完善，保留早期原型。软标签监督与断线端点处理已见于2025论文，不构成单独新意；GNN+RL也已有电网领域工作。",
          "", "来源：[2025软标签GNN及合法端点处理](https://arxiv.org/html/2503.15190v2)、[2026物理先验GNN+RL](https://arxiv.org/html/2604.01830v1)。这些是方法依据，当前代码是本地简化改编，不是完整复现两篇论文。"]
(OUT / "GRID20_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. V0 fits and audit sealed; developmental ranking only, no RL/learned physical rollout.\n",encoding="utf-8")
paths = list(OUT.rglob("*"))+list((ROOT / "work/grid20").rglob("*"))
seal = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and "__pycache__" not in p.parts and p.name not in ["delivery_manifest.json","closeout.log"]}
(OUT / "delivery_manifest.json").write_text(json.dumps(seal,indent=2),encoding="utf-8")
print(json.dumps({"priority":priority,"additional_graph_signal":signal,"gnn32":g,"best_reference":best_name,"sealed_files":len(seal)}))
