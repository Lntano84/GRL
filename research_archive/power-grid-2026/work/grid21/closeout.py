"""Aggregate the single prior revision without overwriting V0."""
import hashlib
import json
import math
import statistics
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid21"
D = json.loads((OUT / "design.json").read_text()); F = json.loads((OUT / "finished.json").read_text())
A = json.loads((OUT / "audit.json").read_text()); assert A["passed"]
P = json.loads((OUT / "per_state.json").read_text()); rows = []; macro = []
meta = {p.stem:json.loads(p.read_text()) for p in (OUT / "inputs/development").glob("*.json")}
def avg(v): return statistics.mean(v) if v else None
for name in D["models"]:
    for k in D["budgets"]:
        group = []
        for week in D["development_weeks"]:
            here = [r for r in P if r["method"]==name and r["budget"]==k and r["week"]==week]
            valid = [r for r in here if r["full_has_valid"]]
            full = sum(meta[r["state_id"]]["canonical_count"] for r in here)
            row = {"method":name,"budget":k,"week":week,"near_best":avg([float(r["near_best_raw"]) for r in valid]),
                   "fallback":avg([float(r["fallback"]) for r in here]),"gap":avg([r["delivered_rho_gap"] for r in valid]),
                   "query_reduction":None if not full else 1-sum(r["expected_unique_public_queries"] for r in here)/full}
            group.append(row); rows.append(row)
        macro.append({"method":name,"budget":k,**{key:avg([r[key] for r in group if r[key] is not None]) for key in ["near_best","fallback","gap","query_reduction"]}})
lookup = {(r["method"],r["budget"]):r for r in macro}
def mean_kind(kind,k): return {key:avg([lookup[(f"{kind}_seed{s}",k)][key] for s in D["seeds"]]) for key in ["near_best","fallback","gap","query_reduction"]}
g = mean_kind("GNN",32); b = mean_kind("BIAS",32); m = mean_kind("MLP",32); old = lookup[("OLDNN",32)]
preserved = g["near_best"]>=old["near_best"]-.02 and g["fallback"]<=old["fallback"]+.02
improved = g["near_best"]>=old["near_best"]+.03 or g["gap"]<=old["gap"]-.001
new = {"per_week":rows,"macro":macro,"gnn32":g,"bias32":b,"mlp32":m,"old32":old,"preserves_base_screen":preserved,"extra_vs_old_screen":improved,
       "scope":"Single documented development revision, same old26 dev states, no independent final confirmation or learned physical rollout. Matched-prior bias and MLP control the extra prior information."}
(OUT / "comparison.json").write_text(json.dumps(new,indent=2),encoding="utf-8")
lines = ["# GRID21：保留PPO先验的有界残差修订V1", "", "V0完整封存，没有覆盖其失败。V1仅增加基于原PPO＋LOCAL完整排名的公开先验，候选头零初始化，修正限制在±2，并以0.1 KL正则保留先验。没有加入另行记录的对象／预测特征假设。", "",
         "GNN、MLP和909参数固定动作偏置均共享该先验、训练数据和目标；两种子、64轮固定结束。PRIOR_ONLY严格复现原OLDNN的顺序。FREQUENCY/RIDGE额外参照未添加先验，不能当作先验感知校准方法。", "",
         "| 方法 | K | 近最佳保留 | 回退 | 唯一公共查询减少 | 交付ρ差 |", "|---|---:|---:|---:|---:|---:|"]
for r in macro: lines.append(f"| {r['method']} | {r['budget']} | {100*r['near_best']:.3f}% | {100*r['fallback']:.3f}% | {100*r['query_reduction']:.3f}% | {r['gap']:.5f} |")
lines += ["",f"保留原优势筛查：{preserved}；对原排序额外收益筛查：{improved}。还须检查与同先验MLP、固定偏置相比是否有独特图表示作用。",
          "",f"新拟合{F['model_fits']}次；神经／偏置训练共{sum(r['training_s'] for r in F['fits']):.2f}秒，阶段墙钟{F['wall_s']:.2f}秒。没有新增公共预测或物理步；原PPO参数不更新。",
          "",f"审计{A['input_states']}个输入、{A['candidate_descriptors']:,}个动作描述、{A['prediction_replays']}次复预测、{A['delivery_rows']}行交付账目；全部开发PRIOR_ONLY与V0OLDNN顺序逐条相同，原历史封存预检通过。没有独立重新拟合或独立AC复验。",
          "", "仍在已读开发集上修订，不是独立确认。两开发日期与训练家族重叠，最终评价须分离。当前只是短名单反馈回放，不能宣称整周成本、安全或真实推理加速。先验构建开销包含在特征构建记录，模型和排名另计。",
          "", "V0的3/4严格正向条件没有用于此轮：原排序在两个开发周已经100%近最佳保留，造成该条件的天花板；V0原裁决保持不变。新条件在拟合前固定，也不是统计显著性或等价检验。",
          "", "方法依据：[GNN代理＋策略先验融合的电网RL工作](https://arxiv.org/html/2604.01830v1)。当前监督有界残差只是本地改编，不复现该论文，也不把先验融合本身当新意。"]
(OUT / "GRID21_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. V1 prior-preserving revision sealed, no RL or physical learned rollout.\n",encoding="utf-8")
paths = list(OUT.rglob("*"))+list((ROOT / "work/grid21").rglob("*"))
seal = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and "__pycache__" not in p.parts and p.name not in ["delivery_manifest.json","closeout.log"]}
(OUT / "delivery_manifest.json").write_text(json.dumps(seal,indent=2),encoding="utf-8")
print(json.dumps({"preserved":preserved,"improved":improved,"gnn32":g,"bias32":b,"mlp32":m,"sealed_files":len(seal)}))
