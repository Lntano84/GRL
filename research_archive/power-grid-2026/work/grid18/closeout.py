"""Seal the borrowed baseline enhancement and all teacher feedback."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid18"
A = json.loads((OUT / "audit.json").read_text()); F = json.loads((OUT / "finished.json").read_text()); assert A["passed"]
lines = ["# GRID18：完整搜索加入已发表的端点修复", "", "2026-10-10。四个已见训练周，种子0；只改断线搜索候选的母线指令，保留原NN20、连续控制和ALWAYS_RESTORE。未拟合新模型。", "",
         "## 闭环结果", "", "| 周 | 原/修复后步数 | 原/修复后完整 | 共同完整周成本增益 | 完全相同的物理前缀 |", "|---|---|---|---:|---:|"]
for r in A["comparison"]:
    gain = "不排名失败累计成本" if r["cost_gain"] is None else f"{100*r['cost_gain']:.4f}%"
    lines.append(f"| {r['week']} | {r['full_steps']}/{r['masked_steps']} | {r['full_complete']}/{r['masked_complete']} | {gain} | {r['identical_physical_prefix']} |")
lines += ["", "一月仍在1719步失败。四月与七月动作、观测、成本完全相同；十月从第101步出现物理分支差异，完整周成本下降7.6862%。这是补齐成熟方法后的真实开发结果，不是GNN/RL的成绩，也不能泛化为平均生产收益。", "",
          "## 修复及审计", "", "仅清除当前仍断开、且基础动作未意图重连的线路端点母线指令；保留基础重连，其他字段不变。原库、模型和控制器文件没有修改。只在断线完整搜索模块启用，没有改原NN20或拓扑回归模块。", "",
          f"原来631个非法候选对应的端点指令得到处理。实际回放{A['physical_rows']:,}步，26次搜索、{A['candidate_labels']:,}个公共候选反馈；候选非法记录为{A['illegal_candidate_labels']}。公共预测总计{F['counts']['public_forecasts']:,}，阶段约{F['wall_s']/60:.2f}分钟。",
          "", "独立审计重建全部候选端点修改与源选优，重算实际费用和固定控制轨迹；未用另一个AC求解器重跑物理结果。控制器插桩与历史批不同，不据此声称耗时加速。合法不等于降低风险：仍有预测终止、异常或不能改善当前ρ的候选。", "",
          "## 对后续训练的含义", "", "所有后续候选排序对照应共同拥有这项修复。原PPO128与LOCAL128在GRID16的优点保留为历史成绩，不直接挪作新基线的成绩。现有26个相关搜索状态不足以支撑可靠训练评价，下一阶段固定新增训练/开发周，并在新模型拟合前封存其身份与算法设置。", "",
          "GNN将先承担监督候选排序，复用原图编码结构；是否需要RL由后续附加作用决定。不能把已发表的软标签候选学习配方当作原创，也不能从修复收益推出学习优势。", "",
          "来源：[2025软标签GNN论文§4.4.2](https://arxiv.org/html/2503.15190v2#S4.SS4.SSS2)。"]
(OUT / "GRID18_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. Known full-search baseline repair verified. Original model intact; no new fits. Teacher feedback sealed.\n", encoding="utf-8")
files = list(OUT.rglob("*")) + list((ROOT / "work/grid18").rglob("*"))
sealed = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and "__pycache__" not in p.parts and p.name != "delivery_manifest.json"}
(OUT / "delivery_manifest.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print(json.dumps({"sealed_files": len(sealed), "candidate_labels": A["candidate_labels"]}))
