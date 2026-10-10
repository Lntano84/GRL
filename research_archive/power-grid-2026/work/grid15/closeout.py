"""Record the frozen qualification verdict and preserve all GRID15 artifacts."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid15"
A = json.loads((OUT / "analysis.json").read_text())
F = json.loads((OUT / "finished.json").read_text())
assert json.loads((OUT / "audit.json").read_text())["passed"]
for manifest in [ROOT / "outputs/grid14/delivery_manifest.json", OUT / "code_freeze.json"]:
    for name, sha in json.loads(manifest.read_text()).items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == sha, name
lines = ["# GRID15：断线拓扑搜索的辅助任务资格", "", "2026-10-10。4 个已见训练周，环境种子 0；仅留存原控制器已经产生的公共预测，没有训练。", "",
         "## 冻结裁决", "", "训练优先级门槛未通过：仅七月的 N1 公共预测时间超过整周控制决策时间的 20%，不是至少两周。各 <=128 候选规则均未达到 95% 近最优保留率，存在局部质量差异，但尚不能把它写成学习价值。此门槛不作事后改判。", "",
         "| 规则 | 平均候选 | 与完整池最优ρ差≤0.01 | 丢失全部合格候选的状态 | 减少源搜索调用 |", "|---|---:|---:|---:|---:|"]
for r in A["rules"]:
    lines.append(f"| {r['rule']} | {r['mean_queries']:.1f} | {100*r['near_best_retention']:.1f}% | {r['lost_valid_calls']} | {100*r['source_query_reduction']:.1f}% |")
lines += ["", "26 次真实 N1 搜索中，23 次完整池有合格改进候选；保留率的分母是这 23 次，不是所有候选或独立实例。状态嵌套在 4 周中，不作显著性与泛化声明。ZONE 平均 336.5 次调用，不是等预算 128 候选对照。", "",
          "| 训练周 | N1 预测秒 | 全控制决策秒（含记录器） | 比例 |", "|---|---:|---:|---:|"]
for r in A["time"]:
    lines.append(f"| {r['week']} | {r['n1_public_forecast_s']:.2f} | {r['controller_s_instrumented']:.2f} | {100*r['n1_forecast_fraction_of_instrumented_controller_s']:.1f}% |")
lines += ["", "预测耗时只包围原 simulate 调用；控制耗时包含留存记录开销。它是单次带记录器测量，不是未插桩的公平加速成绩；没有根据保存的预测时间求和冒充闭环耗时。", "",
          "## 接下来真正未回答的事", "", "局部最优ρ损失不等于完整周失败或成本损失。下一阶段只验证有完整搜索回退的便宜短名单在闭环中的质量—费用—实际耗时权衡；这不是把 GRID15 门槛改成通过，也不自动启动 GNN 或 RL。保留原 NN20、危险 QP 与 ALWAYS_RESTORE。", "",
          "## 文献边界", "", "[LJN 作者代码](https://github.com/lajavaness/l2rpn-2023-ljn-agent)已有神经 Top-K 和区域搜索。", "",
          "[2025 软标签 GNN 工作](https://arxiv.org/html/2503.15190v2)用候选公共仿真结果构造软标签，并比较网络与安全筛查；我们不能把相同配方当作原创。", "",
          "[2026 Gibbs 先验预印本](https://arxiv.org/html/2604.01830v1)已经直接结合 GNN 风险代理、候选 Top-K 与 PPO。其 case118 表中也体现质量与速度的权衡，LJN 对照为 topology-only；没有据此声称我们的混合连续控制版本已胜过该方法。训练代码/该方法复现尚未核验。", "",
          "## 审计与资源", "", f"4 条原轨迹、{F['counts']['physical_steps']:,} 物理步、{F['counts']['public_forecasts']:,} 公共预测、23,264 条候选反馈、26 次合法信息重建核验。控制轨迹逐物理步与 GRID14 AUTHOR 的动作、前后观测、费用完全相同。模拟阶段约 {F['wall_s']/60:.2f} 分钟。",
          "", "全部原搜索动作池、状态、候选反馈和源代码封存；分析独立重建区域/距离排序并复算源选择条件，但没有独立 AC 潮流求解，也未重复神经网络拟合。封存测试未读取。"]
(OUT / "GRID15_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. Frozen supervised-priority gate not passed. No training or outstanding job. Closed-loop cheap-shortlist qualification is a separate next question.\n", encoding="utf-8")
paths = list(OUT.rglob("*")) + list((ROOT / "work/grid15").rglob("*"))
sealed = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
          if p.is_file() and "__pycache__" not in p.parts and p.name != "delivery_manifest.json"}
(OUT / "delivery_manifest.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print(json.dumps({"sealed_files": len(sealed), "pilot_gate": A["supervised_pilot_qualification"], "resources": F["counts"]}))
