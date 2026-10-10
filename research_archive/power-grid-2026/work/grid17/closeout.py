"""Preserve the small engineering probe without extrapolating its effects."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "outputs/grid17"
A = json.loads((OUT / "audit.json").read_text()); F = json.loads((OUT / "finished.json").read_text()); Q = json.loads((OUT / "qualification.json").read_text())
assert A["passed"]
lines = ["# GRID17：已发表的断线端点修复在本环境适用", "", "2026-10-10。先做零预测的动作语义检查，再做12组配对公共预测。未训练。", "",
         f"既有GRID15的{Q['recorded_illegal_candidates']}个非法候选，其线路冷却冲突都能在动作拓扑影响检查中清除；这只是静态条件，不证明{Q['recorded_illegal_candidates']}个候选的潮流可行性。随后固定每个已见训练周最早符合条件的搜索状态、按原动作ID取前三个，共12对。",
         "", f"12/12原非法动作在修复后通过严格的非法、歧义、异常、终止检查；其中{A['illegal_to_valid_improving']}/12同时使最大负载率低于当时状态。不能将此工程选择的命中率当作自然发生率，也不能将低于当前ρ当作优于完整池最佳动作。", "",
         "| 周 | 状态步 | 动作ID | 当前ρ | 修复预测ρ | 严格有效 | 改善当前ρ |", "|---|---:|---:|---:|---:|---|---|"]
for r in A["details"]:
    lines.append(f"| {r['week']} | {r['step']} | {r['pool_id']} | {r['current_rho']:.5f} | {r['masked_rho']:.5f} | {r['masked_strictly_valid']} | {r['masked_improving']} |")
lines += ["", "## 权限与复算", "", "仅清除当前仍断开且基础动作没有主动重连意图的线路端点set-bus赋值。保留基础动作已有的显式重连、状态切换与母线重连；候选和基础动作均不原地修改。", "",
          "按保存的实际动作重放1277个前缀物理步，每步观测与原轨迹逐字节相同。24次公共预测，原动作反馈重新匹配旧记录；修复动作不实际执行，没有整周控制成绩。预测观测单独保存实际向量布局，未按物理观测布局误解码。独立审计从原始字段重建端点清除，未重复AC求解。",
          "", f"预测探针约{F['wall_s']:.2f}秒，无模型拟合。", "",
          "## 下一步", "", "该修复应先加入非学习完整搜索基线，保存其全部实际搜索反馈，再观察真实闭环表现。即使有收益，也属于借用成熟方法补齐工程底座，不能算作GNN/RL或原创贡献。", "",
          "来源：[2025软标签GNN论文§4.4.2](https://arxiv.org/html/2503.15190v2#S4.SS4.SSS2)。其环境与动作库不同，本轮不称全文复现。"]
(OUT / "GRID17_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
(OUT / "RUN_STATE.md").write_text("COMPLETE. Known endpoint repair applicable in selected engineering cases; whole-policy benefit not yet established. No fits or outstanding probes.\n", encoding="utf-8")
files = list(OUT.rglob("*")) + list((ROOT / "work/grid17").rglob("*"))
sealed = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files if p.is_file() and "__pycache__" not in p.parts and p.name != "delivery_manifest.json"}
(OUT / "delivery_manifest.json").write_text(json.dumps(sealed, indent=2), encoding="utf-8")
print(json.dumps({"sealed_files": len(sealed), "strictly_valid": A["illegal_to_strictly_valid"], "improving": A["illegal_to_valid_improving"]}))
