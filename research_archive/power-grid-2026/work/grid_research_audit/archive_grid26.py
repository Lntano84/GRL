"""Archive completed GRID26 and preserve superseded GRID25, without raw traces."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "work/GRL-results-integration"
ARCH = REPO / "research_archive/power-grid-2026"
TARGET = ARCH / "grid26_archive_manifest.json"
assert not TARGET.exists(), "Do not overwrite an archived stage"
assert json.loads((ROOT / "outputs/grid26/audit.json").read_text())["passed"]
assert json.loads((ROOT / "outputs/grid26/contribution.json").read_text())["passed"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


preserved = []
for name in ["archive_manifest.json", "grid24_archive_manifest.json"]:
    for r in json.loads((ARCH / name).read_text())["files"]:
        path = ARCH / r["path"]
        assert sha(path) == r["sha256"], path
        preserved.append((path, r["sha256"]))
proto = ARCH / "protocols/GRID26_before_runs"
for r in json.loads((proto / "manifest.json").read_text())["files"]:
    path = proto / r["path"]
    assert sha(path) == r["sha256"], path
    preserved.append((path, r["sha256"]))

files = []
for stage in ["grid25", "grid26"]:
    files.extend(sorted((ROOT / "work" / stage).glob("*.py")))
    files.extend(p for p in sorted((ROOT / "outputs" / stage).iterdir()) if p.is_file() and p.suffix in [".json", ".csv", ".md"])
    files.extend(sorted((ROOT / "outputs" / stage / "runs").glob("*/summary.json")))
    files.extend(sorted((ROOT / "outputs" / stage / "runs").glob("*/metadata.json")))
files.extend([Path(__file__).resolve(), ROOT / "work/grid_research_audit/grid26_contribution.py"])
records = []
for path in files:
    rel = path.relative_to(ROOT)
    dest = ARCH / rel
    assert not dest.exists(), dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dest)
    assert sha(dest) == sha(path)
    records.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
TARGET.write_text(json.dumps({"files": records, "preserved_checks": len(preserved),
    "scope": "Own source, root summaries, individual summary/metadata; raw vectors/forecast logs remain local. GRID25 is instrumentation diagnostic only. GRID26 is completed serial development timing."}, indent=2), encoding="utf-8")

readme = ARCH / "README.md"
body = readme.read_text(encoding="utf-8")
body = body.replace("[当前 GRID26 提速核验协议（尚无正式结果）]", "[GRID26 回放前封存协议]")
head = """# 最新：GRID26 实际提速核验完成

[贡献核对](outputs/grid26/GRID26_contribution_report.md) · [48条串行结果](outputs/grid26/comparison.csv) · [保存数据审计](outputs/grid26/audit.json) · [增量归档清单](grid26_archive_manifest.json)

48/48条开发回放，95,424物理步与379,368公开预测；所有动作、观测和成本路径与FULL相同，均完成3/4日期，六月同一步失败。两GNN搜索时间减少19.55%/17.14%，总控制仅减少1.40%/0.88%，查询少34.09%。重复0总控制反向；不宣称稳定显著端到端提速。FULL搜索仅占总控制9.80%。原PPO同样省搜索18.31%；156/156 GNN–PPO逐调用选择与查询数量相同，图模块独有增量未建立，排序/查询集合并不都相同。

GRID25四条试点作为测量修订历史保留，不能拿其较有利时间替代正式矩阵。新拟合/RL/预留家族评价均为0。旧模型、旧失败和回放前协议不变。

## 既有记录

"""
readme.write_text(head + body, encoding="utf-8")
docs = REPO / "docs"
state = docs / "RESEARCH_STATE.md"
body = state.read_text(encoding="utf-8")
paragraph = next(line for line in body.splitlines() if line.startswith("- **当前：GRID26"))
body = body.replace(paragraph, "- **当前：GRID26 实际提速核验完成**。48条串行开发回放及审计完成；质量路径逐步等同FULL，3/4日期完成、六月同一步失败。两GNN搜索减少19.55%/17.14%，总控制减少1.40%/0.88%，查询少34.09%。搜索仅占FULL总控制9.80%；重复0总控制反向，尚无稳定显著端到端提速。原PPO也省搜索18.31%；图模型独有增量未建立。95,424物理步、379,368公开预测；新拟合/RL/预留评价0；无未结GRID26作业。下一步先核对静态动作元数据缓存，不用通用缓存冒充图模型贡献。")
state.write_text(body, encoding="utf-8")
(docs / "NEXT_STEPS.md").write_text("""# Next steps — 2026-10-10 / after GRID26

用户授权电网辅助搜索继续自主迭代，原模型与失败结果必须保留。GRID26已完成全部48条串行回放与保存数据审计，没有未结回放。

1. 阅读[实际提速分解](../research_archive/power-grid-2026/outputs/grid26/GRID26_contribution_report.md)。搜索减少17%–20%，总控制仅减少0.88%–1.40%，重复0反向；不把少查34%写成整体提速。
2. 保留权重、K128、作者rho_safe=0.9及完整质量保护。下一项先做排名的静态动作元数据缓存等价性与开销检查，优化同时给予PPO；不增加K、不扫阈值、不立即新训练。
3. GNN–PPO的156次逐调用选择和查询数量全部相同；排名/查询集合不全相同，尚无图模块的实用独有增量。后续训练必须针对已验证的剩余不足，加入同信息强对照。
4. 预留8日期家族尚未打开；部署实现和公平计时协议固定后再进行封存评价。现有四周均为开发，两训练种子和计时重复都不是独立日期样本。
5. ICAPS2027全文日期2026-12-14、ECAI2027官网全文日期2027-04-14；ECML2027八月为举办时间，ADS截止待正式CFP。投稿需贡献与质量、成本证据，不按日期强行投稿。

LG01自动跟进仍暂停，历史方向不自动重启。完整大轨迹保留本地，Git为选择性归档。
""", encoding="utf-8")
with (docs / "EXPERIMENT_LOG.md").open("a", encoding="utf-8") as f:
    f.write("\n## 2026-10-10 — GRID26 actual serial control timing completed\n\n48/48 runs,95424 physical steps,379368 public forecasts; saved-data audit and independent timing accounting pass. Same delivered action/observation/cost paths as sealed FULL; completion3/4 dates including same June failure. GNN0/GNN1 search savings19.55%/17.14%; total-control savings1.40%/0.88%; pooled query saving34.09%. FULL source fraction9.80%; repeat0 total slowdown. PPO search saving18.31%, total slowdown2.82% with large non-search variation; not a GNN-caused speed attribution.156 GNN/PPO call comparisons have equal choice and query count, but only96 identical queried sets and0 identical dispatch sequences. New fits/RL/reserved evaluations0. [Contribution report](../research_archive/power-grid-2026/outputs/grid26/GRID26_contribution_report.md).\n")
with (docs / "DECISIONS.md").open("a", encoding="utf-8") as f:
    f.write("\n## 2026-10-10 — Separate source acceleration from whole control and graph increment\n\nKeep all48 GRID26 repetitions rather than the favorable old GRID25 pilot. Exact quality preservation and17%–20% source acceleration are developmental evidence;0.88%–1.40% whole-controller point estimates with a reversed repeat do not establish stable significant total acceleration. Non-search time decomposition is accounting, not identified causality. PPO shares the same query reductions; do not label its non-search variation a graph advantage. Next check static action metadata caching with exact rank equivalence and the same optimization for PPO. Preserve checkpoints, guard and hidden families; no new training is implied by engineering optimization.\n")
for path, expected in preserved:
    assert sha(path) == expected, path
print(json.dumps({"archived_files": len(records), "preserved_checks": len(preserved), "bytes": sum(r["bytes"] for r in records)}))
