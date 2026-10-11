"""Preserve GRID26 offline fallback review without changing prior archives."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "work/GRL-results-integration"
ARCH = REPO / "research_archive/power-grid-2026"
TARGET = ARCH / "grid26_review_archive_manifest.json"
assert not TARGET.exists()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


preserved = []
for name in ["archive_manifest.json", "grid24_archive_manifest.json", "grid26_archive_manifest.json"]:
    for r in json.loads((ARCH / name).read_text())["files"]:
        p = ARCH / r["path"]
        assert sha(p) == r["sha256"], p
        preserved.append((p, r["sha256"]))
review = json.loads((ROOT / "outputs/grid26_review/fallback_review.json").read_text())
assert review["passed"]
assert json.loads((ROOT / "outputs/grid26_review/review_audit.json").read_text())["passed"]
for relative, digest in review["input_sha256"].items():
    assert sha(ROOT / relative) == digest
records = []
files = list(sorted((ROOT / "outputs/grid26_review").glob("*")))
files += [Path(__file__).resolve(), ROOT / "work/grid_research_audit/fallback_review.py"]
for p in files:
    rel = p.relative_to(ROOT)
    dest = ARCH / rel
    assert not dest.exists()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, dest)
    assert sha(dest) == sha(p)
    records.append({"path": rel.as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)})
TARGET.write_text(json.dumps({"files": records, "preserved_checks": len(preserved),
    "scope": "Offline GRID26 repeat-0 fallback diagnosis. No reruns, fits, guard modifications, or reserved evaluation."}, indent=2), encoding="utf-8")
docs = REPO / "docs"
p = docs / "RESEARCH_STATE.md"
b = p.read_text(encoding="utf-8")
b = b.replace("下一步先核对静态动作元数据缓存，不用通用缓存冒充图模型贡献。", "补查离线诊断：26个开发调用中16次全池补查消耗91.41%查询，且全池也没有rho<0.9的合法候选；保持K128/守卫时，更准排序不能进一步省查询。下一步优先研究合法的分批补查/停止判据；静态缓存为共有工程优化，不作图模型贡献。")
b = b.replace("# Research state — 2026-10-10", "# Research state — 2026-10-11")
p.write_text(b, encoding="utf-8")
(docs / "NEXT_STEPS.md").write_text("""# Next steps — 2026-10-11 / after GRID26 fallback review

用户授权电网辅助搜索继续自主迭代，原模型与失败结果必须保留。GRID26及保存结果补查诊断已完成，没有未结回放。

1. 阅读[实际提速分解](../research_archive/power-grid-2026/outputs/grid26/GRID26_contribution_report.md)。搜索减少17%–20%，总控制仅减少0.88%–1.40%，重复0反向；不把少查34%写成整体提速。
2. [补查瓶颈](../research_archive/power-grid-2026/outputs/grid26_review/FALLBACK_REVIEW.md)：重复0的26个不同开发状态，16次全池补查消耗91.41%查询，16次全池均无rho<0.9合法候选。保持K128和当前守卫，单纯提高排序准确率无法在这些状态上进一步减少查询。先设计合法的分批补查/停止判据与廉价强对照，不立即加层或新训练。
3. 保留当前权重、K128与rho_safe=0.9作为质量基线。新的停止规则必须单独冻结、记录和评价；隐藏FULL结果只用于诊断，不可作为部署输入。不能将前缀覆盖当作停止证书，不以降低质量要求换提速。
4. GNN的前128在14次有合法FULL选择的补查中覆盖8/7次，PPO5次；这是事后开发线索，尚无独有查询/闭环增量。后续必须给予PPO与廉价规则相同信息及预算。静态动作元数据缓存可做共同等价优化，其成本占比很小，不冒充图模型创新。
5. 预留8日期家族尚未打开；部署实现和公平计时协议固定后再进行封存评价。四周为开发，两个训练种子与计时重复不是独立日期样本。无需再次计时证明相同配置是GNN贡献。

LG01自动跟进仍暂停，历史方向不自动重启。完整大轨迹保留本地，Git为选择性归档。投稿时间与分类待临近工作单独按官方来源核验；不按日期强行投稿。
""", encoding="utf-8")
for name, addition in [
    ("EXPERIMENT_LOG.md", """\n## 2026-10-11 — GRID26 fallback review, no new simulation

Read104 saved call logs from repeat0/four exposed dates;26 distinct states and78 ranker/state rows.16/26 calls use full complement,accounting for91.41% of ranker queries;complement alone77.65%.All16 full pools lack admissiblerho<0.9 candidates.Thus fixedK128/guard cannot benefit in query count from a more accurate ranking on these states.14 full pools contain an admissible winner;shortlist coverageGNN0/GNN1/PPO=8/7/5,not an independent graph-increment confirmation.No new forecasts,rollouts,fits or hidden-family outcomes.Independent prefix-phase arithmetic agrees.[Review](../research_archive/power-grid-2026/outputs/grid26_review/FALLBACK_REVIEW.md).\n"""),
    ("DECISIONS.md", """\n## 2026-10-11 — Improve the costly fallback decision before network capacity

The objective is useful same-quality acceleration,not to force GNN attribution.Whole-controller gains remain small and repeated timing reverses;PPO shares present savings.Offline diagnosis identifies16 fallback states with no safe candidate even in the full pool;under unchangedK128/guard better ranking cannot reduce their queries.Prioritize a separately evaluated legal staged-search/stopping design rather than blind depth/GPU/RL changes.Preserve protected baseline and allweights;coverage of privilegedFULL winners is not an online certificate.No guard was removed or threshold changed in this review.\n""")]:
    p = docs / name
    p.write_text(p.read_text(encoding="utf-8") + addition, encoding="utf-8")
p = ARCH / "README.md"
p.write_text("""# 最新诊断：GRID26 补查瓶颈（2026-10-11）

[离线补查报告](outputs/grid26_review/FALLBACK_REVIEW.md) · [逐调用表](outputs/grid26_review/fallback_per_call.csv) · [独立复核](outputs/grid26_review/review_audit.json)

26个已暴露开发调用中16次全池补查消耗91.41%查询；这16个状态全池也没有rho<0.9合法候选。保持K128与当前守卫时，单靠更准排序不能进一步省查询。前128在14次有合法FULL最优动作的补查中，GNN0/GNN1/PPO覆盖8/7/5；属于事后开发诊断，不是图模块查询增量确认。没有新预测、训练、守卫修改或封存日期评价。下一步优先研究合法分批补查/停止，而非加层。

""" + p.read_text(encoding="utf-8"), encoding="utf-8")
for p, digest in preserved:
    assert sha(p) == digest
print(json.dumps({"archived_files": len(records), "preserved_checks": len(preserved), "passed": True}))
