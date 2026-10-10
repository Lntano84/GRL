"""Close GRID24, preserve history, and copy selected artifacts to the GRL archive."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/grid24"
REPO = ROOT / "work/GRL-results-integration"
DEST = REPO / "research_archive/power-grid-2026"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    assert not (OUT / "reconciliation.json").exists(), "Already finalized"
    audit = read(OUT / "audit.json")
    finished = read(OUT / "finished.json")
    comparison = read(OUT / "comparison.json")
    assert audit["passed"] and finished["passed_engineering"]
    assert audit["runs"] == len(comparison["records"]) == 12
    assert all(g["development_quality_gate"] for g in comparison["groups"])
    assert all(r["exact_full_trajectory"] for r in comparison["records"])
    checks = {}
    for name in ["preservation_hashes.json", "code_freeze.json", "parallel_code_freeze.json"]:
        entries = read(OUT / name)
        for file, expected in entries.items():
            assert sha(ROOT / file) == expected, file
        checks[name] = len(entries)
    assert finished["counts"]["physical_steps"] == audit["physical_rows"]
    assert finished["counts"]["model_fits"] == audit["new_model_fits"] == 0
    records = comparison["records"]
    with (OUT / "per_run_comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    groups = comparison["groups"]
    report = OUT / "GRID24_report.md"
    text = report.read_text(encoding="utf-8")
    text += "\n## 完整裁决与计量\n\n"
    for group in groups:
        text += (f"- {group['method']}：完成 {group['complete_weeks']}/{group['weeks']} 周；"
                 f"共同完成周成本平均差 {100*group['mean_cost_regret']:.4f}%；"
                 f"汇总搜索查询减少 {100*group['pooled_n1_query_reduction']:.4f}%；"
                 f"逐周平均减少 {100*group['mean_n1_query_reduction']:.4f}%。\n")
    text += (f"\n共 {audit['physical_rows']:,} 个交付物理步、{finished['counts']['public_forecasts']:,} 次公开预测；"
             f"独立保存数据审计重算 {audit['source_calls']} 次搜索调用、{audit['candidate_feedback']:,} 项候选反馈。"
             "没有新增标签训练、模型拟合或 RL；没有打开预留日期家族。\n\n"
             "全部 12 条动作、观测与成本轨迹逐步等同于对应历史 FULL，包括六月同一步失败；"
             "失败周不计算累计成本优势。此结论覆盖四个已暴露开发周，两个训练种子不是独立周。\n\n"
             "试点串行，剩余十条以两个独立进程并行。计时包含记录，且历史 FULL 来自不同批次；"
             "这些时间不能认证端到端提速，也不用于选择方法。搜索查询减少是有效计量，但不是全部控制费用减少。\n\n"
             "三个排序器在本批产生同样的搜索查询量与物理轨迹：保护规则有用，但没有建立 GNN 的独有增量。"
             "原 82% 左右查询节省伴随质量损失；本轮是质量与查询量的另一运行点，不能混合引用两版最优数字。\n\n"
             "审计范围为保存数据的独立算术/合法性/固定输入/候选次序与权重预测复核，没有独立重跑全部 AC 物理过程。"
             "rho_safe=0.9 是作者既有工程阈值，不构成形式安全证书。\n\n"
             "下一步先在相同串行计时条件下验证实际控制耗时，并比较使用同一保护规则的原 PPO；"
             "协议定稿后再打开保留日期家族。当前不扩大 K、不扫安全阈值、不改模型目标或启动新训练。\n")
    report.write_text(text, encoding="utf-8")
    changelog = ROOT / "outputs/grid_research_audit/POWER_MODEL_CHANGELOG.md"
    old = changelog.read_text(encoding="utf-8")
    assert "| GRID24 |" not in old
    marker = "\nV1训练目标包含KL正则"
    row = ("\n| GRID24 | 同一 V1/PPO 权重；K128 后以作者既有 rho_safe=0.9 触发缓存完整搜索 | "
           f"12 条开发轨迹全部逐步重现 FULL；两个 GNN 与 PPO 完成 3/4 周，共同完成成本差 0，"
           f"汇总搜索查询减少 {100*groups[0]['pooled_n1_query_reduction']:.4f}% | "
           "通用保护规则，未建立图模型增量；并行单次时间非正式速度证据；新拟合/RL/封存评价均为0 |\n")
    assert marker in old
    changelog.write_text(old.replace(marker, row + marker), encoding="utf-8")
    shutil.copy2(changelog, OUT / "POWER_MODEL_CHANGELOG_after_GRID24.md")
    reconciliation = {"passed": True, "runs": 12, "checks": checks,
        "counts": finished["counts"], "all_exact_full_paths": True,
        "groups": groups, "new_model_fits": 0, "reserved_family_evaluations": 0,
        "speed_certified": False, "gnn_specific_advantage_established": False,
        "scope": "Frozen development quality screen, preserved original models and common rule/PPO comparator. Saved-data audit, no independent AC replay."}
    (OUT / "reconciliation.json").write_text(json.dumps(reconciliation, indent=2), encoding="utf-8")
    (OUT / "RUN_STATE.md").write_text("COMPLETE. 12/12 quality runs and saved-data audit passed. No active job, no new fits/RL, no reserved-family evaluation. Controlled speed measurement remains future work.\n", encoding="utf-8")
    delivery = []
    for base in [OUT, ROOT / "work/grid24"]:
        for path in sorted(base.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.name != "delivery_manifest.json":
                delivery.append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    (OUT / "delivery_manifest.json").write_text(json.dumps({"files": delivery, "file_count": len(delivery),
        "scope": "Local complete GRID24 delivery, including large raw traces; not all files are in the selected Git archive."}, indent=2), encoding="utf-8")
    historical_archive = read(DEST / "archive_manifest.json")
    for entry in historical_archive["files"]:
        assert sha(DEST / entry["path"]) == entry["sha256"], entry["path"]
    archive = []
    sources = [p for p in OUT.iterdir() if p.is_file() and p.suffix in [".md", ".json", ".csv", ".patch"] and p.stat().st_size <= 4_000_000]
    sources += list((ROOT / "work/grid24").glob("*.py"))
    sources += [ROOT / "outputs/grid_venue_review/CCF_B_GRID_VENUE_REVIEW_2026-10-10.md"]
    for path in sorted(sources):
        rel = path.relative_to(ROOT)
        target = DEST / rel
        assert target.resolve().is_relative_to(DEST.resolve()) and not target.exists(), target
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        assert sha(target) == sha(path)
        archive.append({"path": rel.as_posix(), "bytes": path.stat().st_size, "sha256": sha(path)})
    (DEST / "grid24_archive_manifest.json").write_text(json.dumps({"files": archive,
        "historical_archive_verified_files": len(historical_archive["files"]),
        "included_bytes": sum(r["bytes"] for r in archive),
        "scope": "Incremental selected GRID24 source/reports and venue review. Historical manifest/files unchanged. Raw traces, third-party data and models remain local."}, indent=2), encoding="utf-8")
    archive_readme = DEST / "README.md"
    body = archive_readme.read_text(encoding="utf-8")
    prefix = ("# 最新：GRID24 质量补强（2026-10-10）\n\n"
        "[完整报告](outputs/grid24/GRID24_report.md) · [逐运行结果](outputs/grid24/per_run_comparison.csv) · "
        "[审计](outputs/grid24/audit.json) · [更改记录](outputs/grid24/POWER_MODEL_CHANGELOG_after_GRID24.md) · "
        "[增量归档清单](grid24_archive_manifest.json)\n\n"
        f"沿用原 rho_safe=0.9，在高风险时缓存补查剩余候选。12/12 开发轨迹逐步重现 FULL，"
        f"成本差为0且未丢失完整周；搜索查询汇总减少 {100*groups[0]['pooled_n1_query_reduction']:.4f}%。"
        "此通用补救对两 GNN 和原 PPO 都有效，尚未证明图模型增量或正式端到端提速。"
        "旧权重、失败结果、GRID00–23 原归档文件与清单保持不变。未新训练或打开预留评价。\n\n"
        "[会议与前作核查](outputs/grid_venue_review/CCF_B_GRID_VENUE_REVIEW_2026-10-10.md) · "
        "[更早投稿时间](outputs/grid24/METHOD_AND_VENUE_NOTE.md)\n\n## GRID00–23 历史记录\n\n")
    archive_readme.write_text(prefix + body, encoding="utf-8")
    docs = REPO / "docs"
    for name in ["RESEARCH_STATE", "NEXT_STEPS"]:
        before = docs / f"history/{name}_before_GRID24_2026-10-10.md"
        assert not before.exists()
        shutil.copy2(docs / (name + ".md"), before)
    state = (docs / "RESEARCH_STATE.md").read_text(encoding="utf-8")
    insertion = ("\n- **GRID24 当前质量修订已完成**：12/12 条开发轨迹、23,856 个物理步、89,064 次公开预测完成且审计通过。"
        f"两 GNN 与原 PPO 的全部轨迹逐步等同对应 FULL，均完成3/4周；共同完成周成本差0，"
        f"汇总搜索查询减少 {100*groups[0]['pooled_n1_query_reduction']:.4f}%。"
        "守卫沿用作者既有rho_safe=0.9，补查时复用缓存。新拟合/RL/预留家族评价均为0；没有未结运行。\n"
        "- 当前未建立 GNN 相对相同守卫的原 PPO 的增量；并行含记录计时不认证实际提速。"
        "GRID23的约82%查询节省和质量差距作为未保护历史消融保留，不能与GRID24的0成本差混合为同一成绩。\n")
    heading, rest = state.split("\n", 1)
    (docs / "RESEARCH_STATE.md").write_text(heading + "\n" + insertion + "\n以下 GRID22/23 为已保留历史状态：\n" + rest, encoding="utf-8")
    (docs / "NEXT_STEPS.md").write_text("# Next steps — 2026-10-10 / after GRID24\n\n"
        "用户已授权电网辅助搜索自主迭代，原模型与失败结果必须保留。GRID24质量补强完成，没有未结作业。\n\n"
        "1. 先读[GRID24报告](../research_archive/power-grid-2026/outputs/grid24/GRID24_report.md)及其审计：开发轨迹质量已保住，但正式端到端提速未测。\n"
        "2. 下一项验证应使用一致串行硬件、相同记录成本与所有控制组件，比较FULL和有同一保护规则的GNN/PPO，报告总控制时间、搜索时间、尾延迟及查询量；不得用并行开发计时认证速度。\n"
        "3. 图模型增量尚未建立；若修订目标/输入，保留原版本并加入同信息非图对照。不凭GNN+RL组合宣称新意。当前不增加K、扫阈值或立即新训练。\n"
        "4. 方法与计时协议冻结后，才打开预留8日期家族的全部变体；四周结果仍是开发证据，两训练种子不是独立周。\n"
        "5. ICAPS2027全文日期2026-12-14、ECAI2027官网全文日期2027-04-14；ECML2027八月是举办时间，详细ADS截止待正式CFP。投稿需贡献、质量与成本证据齐备，不按日期强行投稿。\n\n"
        "[原优先级快照](history/NEXT_STEPS_before_GRID24_2026-10-10.md)。LG01自动跟进仍暂停；历史方向不自动重启。完整原始大轨迹保留本地，仓库为选择性归档。\n", encoding="utf-8")
    experiment = ("\n## 2026-10-10 — GRID24 quality-first safeguarded auxiliary search\n\n"
        "[Report and archived sources](../research_archive/power-grid-2026/outputs/grid24/GRID24_report.md). "
        "Keep V1/PPO weights and K128, use author's existing rho_safe=0.9 to trigger cached full complement evaluation. "
        "12 runs on four exposed development weeks: 23,856 physical steps, 89,064 public forecasts. "
        f"All delivered action/observation/cost paths equal sealed GRID23 FULL; cost regret0 on joint-complete weeks, completion3/4 per method. Pooled search-query reduction{100*groups[0]['pooled_n1_query_reduction']:.4f}%. "
        "All three rankers, including PPO, share the result. Preserved79 old-file hashes and625 historical archive files. "
        "Saved-data independent mask/alias/admission/choice/cost and model rank replay passed; no independent AC rerun. "
        "No new fits/RL/reserved-family evaluation. Two remaining workers ran concurrently; time is descriptive, not certified speed.\n")
    decision = ("\n## 2026-10-10 — Retain quality protection before speed claims\n\n"
        "Human goal: speed with adequate control quality. GRID24 uses published verify/fallback logic and unchanged author threshold; it is not a new threshold sweep or new network. "
        "Quality is recovered on all four exposed weeks with fewer public searches. Keep it as the current protected operating point, preserve unguarded models/results as ablations. "
        "Do not combine unguarded82% query reduction with guarded0 cost regret. "
        "PPO achieves the same result, so graph-specific novelty remains unestablished. Next evidence priority is controlled total-control timing, then untouched date-family evaluation under a frozen protocol. "
        "No new training, large K sweep, or final holdout opened this stage.\n")
    for name, addition in [("EXPERIMENT_LOG.md", experiment), ("DECISIONS.md", decision)]:
        with (docs / name).open("a", encoding="utf-8") as f: f.write(addition)
    print(json.dumps({"passed": True, "local_delivery_files": len(delivery), "new_archive_files": len(archive),
        "historical_archive_preserved_files": len(historical_archive["files"]), "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
