"""Report every frozen week and seal audited teacher data."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid19"
F = json.loads((OUT / "finished.json").read_text()); A = json.loads((OUT / "audit.json").read_text())
D = json.loads((OUT / "design.json").read_text()); assert A["passed"] and F["passed_engineering"]
assert [r["week"] for r in F["summaries"]] == D["weeks"]
lines = ["# GRID19：新训练／开发周的完整教师采集", "", "2026-10-10。先用周名称按季度和固定哈希划分，不读取成绩选周；16个新训练周、4个开发周，旧4个训练周另保留。", "",
         "使用GRID18修复后的完整搜索控制器，保留原PPO、连续控制和恢复规则。这里只采集真实发生的断线搜索调用，不制造触发，不替换早失败周。作者模块N1指断线场景，不代表完整N−1安全校验。", "",
         "| 周 | 用途 | 物理步数 | 完整 | 搜索状态 | 候选反馈 |", "|---|---|---:|---|---:|---:|"]
for r in F["summaries"]:
    split = "训练" if r["week"] in D["new_training_weeks"] else "开发"
    lines.append(f"| {r['week']} | {split} | {r['steps']} | {r['complete']} | {len(r['calls'])} | {r['n1_forecasts']} |")
lines += ["", f"累计{A['physical_rows']:,}物理步、{A['source_calls']:,}个实际搜索状态、{A['candidate_labels']:,}条候选反馈；公共预测总计{F['counts']['public_forecasts']:,}。采集墙钟{F['wall_s']/60:.2f}分钟，没有新模型拟合。",
          "", f"独立保存数据审计通过：完整修改字段、候选选择、物理费用与公共调用账目重算；费用最大残差{A['max_independent_cost_residual']:.3g}。候选非法记录{A['illegal_candidate_labels']}。这不是另一个AC求解器的重复物理验证。",
          "", "失败轨迹仅用于保留分布和学习状态，不以失败累计成本排名。开发轨迹和训练轨迹互斥，开发标签在GRID20全部拟合保存后才读取；这里的开发周不是最终独立确认集。外部PPO已有官方训练经验，不能声称它未见整个官方分布。",
          "", "监督辅助将使用这些反馈排序拓扑候选；不把教师搜索成本隐藏为免费数据，也不从采集完成推出GNN优势或RL必要性。原模型和历史记录未改动。"]
(OUT / "GRID19_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. Frozen20-week teacher collection and independent saved-data audit passed. No fits in this stage.\n", encoding="utf-8")
paths = list(OUT.rglob("*"))+list((ROOT / "work/grid19").rglob("*"))
sealed = {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.is_file() and "__pycache__" not in p.parts and p.name not in ["delivery_manifest.json","closeout.log"]}
(OUT / "delivery_manifest.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print(json.dumps({"sealed_files":len(sealed),"search_states":A["source_calls"],"fits":0}))
