"""Preserve the closed-loop comparison and its original investment boundaries."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid16"
F = json.loads((OUT / "finished.json").read_text()); A = json.loads((OUT / "audit.json").read_text())
C = json.loads((OUT / "comparison.json").read_text()); assert A["passed"]
lines = ["# GRID16：带完整搜索回退的闭环短名单", "", "2026-10-10。四个已见训练周、环境种子0；未拟合新模型。原有 NN20、连续控制与 ALWAYS_RESTORE 保留。", "",
         "## 完整周主比较", "", "失败周不按累计成本排名。下表只对各方法与 FULL 同时完成的周取等权平均；成功率单列。正成本增益表示相对 FULL 的模拟运行成本下降。", "",
         "| 方法 | 共同完整周 | 成本增益 | N1预测减少 | 全控制决策时间减少 | 失去FULL完整周 | 冻结筛查 |", "|---|---:|---:|---:|---:|---:|---|"]
for r in C["aggregate"]:
    lines.append(f"| {r['rule']} | {r['joint_weeks']} | {100*r['mean_cost_gain_joint']:.3f}% | {100*r['mean_n1_query_reduction_joint']:.1f}% | {100*r['mean_controller_time_reduction_joint']:.1f}% | {len(r['lost_complete_weeks'])} | {'通过' if r['simple_qualified'] else '未通过'} |")
lines += ["", "筛查同时要求：不丢完整周、失败周不提前超过一步、完整周平均成本增加不超过1%、N1预测减少至少30%、全控制时间减少至少10%。它不是显著性或等价检验，不能推广为生产服务性能。", "",
          "| 周 | 方法 | 运行步数 | 完整 | 模拟成本 | N1预测 | 完整池回退 | 控制秒 |", "|---|---|---:|---|---:|---:|---:|---:|"]
for r in F["summaries"]:
    lines.append(f"| {r['week']} | {r['rule']} | {r['steps']} | {r['complete']} | {r['cost']:.2f} | {r['n1_forecasts']} | {r['fallbacks']} | {r['controller_s']:.2f} |")
lines += ["", "## 公平性与作用边界", "", "LOCAL128 使用当前观测的局部距离和线路负载；ZONE 使用原作者区域划分，候选数可变，因此不是与128个候选等预算的对照。OLDNN128 使用原有 rho 输入 PPO，对909个池动作中重合的280个评分，其余按 LOCAL128 补足；没有改动或训练作者模型。", "",
          "短名单找不到原选择条件认可的动作才回退到完整池。回退只在同一个当前观测、基础动作和预测步长下复用已有候选反馈；不跨步缓存、不重复支付预测。候选排序不读取完整池反事实成绩。", "",
          "本阶段保留原 Greedy 的可行性准则，未悄悄实施新的动作修复。若后续修复断线端点的 set-bus 行为，应单独保存变体并比较；这也是避免把基线工程缺陷算成模型贡献的原因。", "",
          "这里的N1沿用作者模块名称，表示当前已有线路断开的搜索分支；没有新增逐线路故障枚举，不能当作完整N−1安全分析或安全保证。", "",
          "执行顺序逐周轮换，但仅每格一次计时。控制时间包括候选排序、推理、插桩记录等本轮开销；未报告置信区间，也没有借失败导致的较短运行宣称加速。", "",
          "## 审计与资源", "",
          f"16条轨迹，共 {F['counts']['physical_steps']:,} 物理步、{F['counts']['public_forecasts']:,} 公共预测，回放阶段约 {F['wall_s']/60:.2f} 分钟。审计重算 {A['physical_rows']:,} 步成本与轨迹、{A['source_calls']} 次搜索排序和 {A['cache_reuses_checked']} 次候选缓存复用。FULL逐步重现GRID15。审计没有独立AC求解缓存结果，没有新增物理步或预测。",
          "", "## 研究裁决", "",
          "通过筛查的便宜规则应成为后续强基线，不能记为GNN/RL的收益。未通过的规则也不能据此否定所有图学习。GRID15训练优先级门槛保持未通过；GRID16回答的是不同的闭环质量与耗时问题，不回溯改判。",
          "", "[LJN原代码](https://github.com/lajavaness/l2rpn-2023-ljn-agent)已经包括神经Top-K和区域搜索。[软标签图模型工作](https://arxiv.org/html/2503.15190v2)已经研究候选排序与断线端点可行性修复；[2026预印本](https://arxiv.org/html/2604.01830v1)已有GNN风险代理与PPO结合。因此不能以模块组合本身声称新意。"]
(OUT / "GRID16_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. No outstanding rollout. No new model fits. Closed-loop cheap-baseline comparison preserved; GRID15 priority verdict unchanged.\n", encoding="utf-8")
paths = list(OUT.rglob("*")) + list((ROOT / "work/grid16").rglob("*"))
sealed = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and "__pycache__" not in p.parts and p.name != "delivery_manifest.json"}
(OUT / "delivery_manifest.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print(json.dumps({"sealed_files": len(sealed), "qualifying_rules": [r["rule"] for r in C["aggregate"] if r["simple_qualified"]]}))
