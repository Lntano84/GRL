"""Build a second manuscript only from finished, audited GRID62 receipts."""
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / 'outputs/grid_paper_v1'
OUT = ROOT / 'outputs/grid_paper_v2'
read = lambda p: json.loads(p.read_text(encoding='utf-8'))
mean = lambda a: math.fsum(a) / len(a) if a else None
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert not OUT.exists(), 'Preserve completed manuscript versions.'
    r = read(ROOT / 'outputs/grid62/review.json')
    assert r['passed_saved_data'] and r['runs'] == 24
    d = read(ROOT / 'outputs/grid62/design.json')
    timing = read(ROOT / 'outputs/grid62_timing_review/review.json')
    assert len(d['weeks']) == 3
    OUT.mkdir()
    sources = read(V1 / 'source_receipts.json')
    for record in sources:
        assert sha(ROOT / record['path']) == record['sha256'], record['path']
    for p in ('outputs/grid62/review.json', 'outputs/grid62/per_family.csv', 'outputs/grid62/per_run.csv',
              'outputs/grid62/payload_manifest.json', 'outputs/grid62/finished.json'):
        sources.append({'path': p, 'sha256': sha(ROOT / p), 'bytes': (ROOT / p).stat().st_size})
    rows = r['family_rows']
    fmt = lambda x: 'not comparable' if x is None else f'{100*x:+.3f}%'
    lines = ['The final frozen screen produced **' + r['verdict'] + '**. All three remaining fresh dates and all 24 runs are retained. No date, model, stopping rule, or threshold was changed after the freeze.', '',
        '| Policy | Completed families | Quality-screen families | Controller-time reduction | Query reduction | Worst complete-week cost change |',
        '|---|---:|---:|---:|---:|---:|']
    for x in r['summaries']:
        lines.append(f"| {x['method']} | {x['completed_families']}/3 | {x['quality_families']}/3 | {fmt(x['macro_control_reduction_full'])} | {fmt(x['macro_query_reduction_full'])} | {fmt(x['worst_complete_cost_delta_full'])} |")
    lines += ['', '| Date / primary variant | FRAME completion / steps | FRAME cost change | FRAME controller-time reduction | FRAME query reduction |', '|---|---|---:|---:|---:|']
    for x in rows:
        if x['method'] == 'FRAME':
            lines.append(f"| {x['week']} | {x['complete']} / {x['steps']} | {fmt(x['cost_delta_full'])} | {fmt(x['control_reduction_full'])} | {fmt(x['query_reduction_full'])} |")
    frame = next(x for x in r['summaries'] if x['method'] == 'FRAME')
    lines += ['', f"FRAME has a positive controller-time direction on {r['positive_control_families']}/3 date families and {r['completed_comparable_families']}/3 complete comparable horizons. The first-verified-safe reference meets its own quality-and-time competitiveness rule: {r['first_safe_competitive']}.", '',
        f"The saved-data audit checks {r['physical_rows']:,} physical rows, {r['matched_feedback']:,} matched simulation-feedback items, and {r['exact_repeat_paths']}/12 identical repeated physical paths. It does not independently re-solve AC power flow.", '',
        'The three date families are not a balanced seasonal sample or independent grids. Timing repeats are averaged within date before ratios and do not enlarge the evidence unit. The 0.5% quality screen and 5% time threshold are predeclared investment criteria, not equivalence or significance certification.']
    no = {x['week']: x for x in rows if x['method'] == 'NO_LOCAL'}
    gains = []
    for x in rows:
        if x['method'] == 'FRAME' and x['cost_delta_full'] is not None and no[x['week']]['cost_delta_full'] is not None:
            gain = no[x['week']]['cost_delta_full'] - x['cost_delta_full']
            gains.append({'week': x['week'], 'cost_gain_of_full': gain})
    if gains:
        lines += ['', 'Completion relative to NO_LOCAL, in FULL-normalized complete-week cost: ' + '; '.join(x['week'] + ' ' + fmt(x['cost_gain_of_full']) for x in gains) + '. Zero or negative effects remain reported; the development March mechanism is not assumed to recur on every fresh date.']
    active = timing['active_only_descriptive']
    lines += ['', 'Timing qualification and competing interpretation: the May episode contains zero topology-search requests for every policy. Its differences in controller time cannot be attributed to an uncalled search. On the two dates where search is actually invoked, FRAME reduces controller time by ' + fmt(active['FRAME']['macro_control_reduction_full']) + ', while FIRST_SAFE_BEST reduces it by ' + fmt(active['FIRST_SAFE_BEST']['macro_control_reduction_full']) + '. These are post-run descriptive subsets, not a replacement of the frozen primary result. The first-safe reference also passes the quality screen on all three fresh dates. Consequently this confirmation does not establish consistent framework superiority over that reference.', '',
        'Repeated physical paths are identical, but controller time can differ substantially: FRAME is 76.28 versus 44.73 seconds in April; FIRST_SAFE_BEST is 13.43 versus 6.84 seconds in the inactive May episode; NO_LOCAL is 88.31 versus 62.92 seconds in November. Both process-CPU and continuous-optimization/compilation times change. No power events were observed, but load and clock causes were not independently measured. The complete timing data remain reported; no unfavorable repeat is dropped.']
    text = (V1 / 'manuscript.md').read_text(encoding='utf-8')
    text = text.replace('**Working manuscript, 13 October 2026.** This draft presents the framework as the main subject. The primary fresh-date confirmation is frozen in GRID62 and is not yet available in this version.',
        '**Working manuscript V2, 13 October 2026.** This draft presents the framework as the main subject. The primary fresh-date confirmation in GRID62 is completed and audited.')
    pending = '**PENDING IN V1.** Insert GRID62\'s audited per-date cost, survival, controller time, query counts, and frozen verdict here. The table must retain all three selected dates, all four policies, both timing repeats, and failures. No winner-selecting revision of the method is permitted using this result. A negative or mixed result changes the strength of the conclusion rather than disappearing from the manuscript.'
    assert text.count(pending) == 1
    text = text.replace(pending, '\n'.join(lines))
    fresh_abstract = (f" On the separately frozen three-date confirmation, FRAME's actual controller-time reduction is {fmt(frame['macro_control_reduction_full'])}, its topology-query reduction is {fmt(frame['macro_query_reduction_full'])}, and it passes the quality screen on {frame['quality_families']}/3 date families. The frozen verdict is {r['verdict']}.")
    text = text.replace('A final evaluation on all three remaining unaccessed date families is frozen separately.', fresh_abstract.strip())
    text = text.replace('The method requires no newly trained graph model', 'The first-verified-safe reference also satisfies the fresh-date quality screen, and timing variation limits claims of superiority over it. The method requires no newly trained graph model')
    text = text.replace('The final conclusion and headline numbers will be revised only after the separately frozen three-date confirmation is audited; any failed or mixed outcome will remain part of the report.',
        f"The separately frozen three-date confirmation is audited, with verdict {r['verdict']}. This limited evidence supports only the corresponding finite-matrix claim; all failures and mixed outcomes remain visible. No additional model or parameter is selected from these outcomes.")
    (OUT / 'manuscript.md').write_text(text, encoding='utf-8')
    for name in ('framework_diagram.md', 'related_work_difference.md', 'software_environment.json'):
        shutil.copy2(V1 / name, OUT / name)
    with (OUT / 'main_results.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, list(r['summaries'][0]))
        writer.writeheader()
        writer.writerows(r['summaries'])
    shutil.copy2(ROOT / 'outputs/grid62/per_family.csv', OUT / 'per_family.csv')
    (OUT / 'source_receipts.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    outcome = {'verdict': r['verdict'], 'frame': frame, 'completion_gains': gains,
        'sources_checked': len(sources), 'v1_retained': True, 'new_fits': 0,
        'timing_review': timing, 'stable_superiority_over_first_safe_established': False,
        'warning': 'Draft, not submission-ready. Author/venue metadata and final literature positioning remain. Do not broaden small fresh-date evidence or rename adopted policies as full published reproductions.'}
    (OUT / 'paper_status.json').write_text(json.dumps(outcome, indent=2), encoding='utf-8')
    zh = ['# 框架论文当前状态', '', '最终冻结裁决：' + r['verdict'] + '。', '',
        f"FRAME实际控制时间宏平均减少{fmt(frame['macro_control_reduction_full'])}，查询减少{fmt(frame['macro_query_reduction_full'])}；质量{frame['quality_families']}/3日期，完成{frame['completed_families']}/3日期。", '',
        '已形成完整英文V2初稿，包含主表、逐日期结果、伪代码、条件性质、机制、不利结果和复现入口；V1原稿保留。', '',
        '论文核心为渐进拓扑搜索与变电站补全的质量—计算权衡。GNN只作可替换辅助，共有缓存不算独有创新。', '',
        '三个日期是全部剩余未用池，两季度分布不均；一次有限确认不认证跨电网、生产收益或录用。', '',
        '关键边界：FIRST_SAFE_BEST在新日期也质量3/3；只看有实际搜索的两日期，FRAME控制少8.25%，FIRST少11.92%。五月没有搜索，不能把它的计时差当搜索效果。重复计算耗时变异较大；论文不写稳定全面胜过成熟停止策略。', '',
        '投稿前的人类事项：作者与署名、目标会议/track、篇幅与格式。技术事项：按目标整理文献新意差别、压缩主表和正文、检查复现包下载入口。不要再扩模型/基线或调封存日期上的参数。']
    (OUT / '论文进度与剩余事项.md').write_text('\n'.join(zh) + '\n', encoding='utf-8')
    print(json.dumps(outcome))


if __name__ == '__main__':
    main()
