"""Audit diagnostic counts, preserve corrections, and publish the read-only investigation."""
import collections, json, math, shutil
from pathlib import Path
import run as G

def main():
    G.verify()
    O, R, REPO = G.O, G.R, G.REPO
    assert G.read(O / 'review.json')['passed']
    fit = G.rows_at(G.B.O / 'fit_rows.json.gz')
    cal = G.rows_at(G.B.O / 'calibration_rows.json.gz')
    # Audit found defaultdict access had inserted calls without eligible prefixes.
    # Preserve the first output and frozen producer rather than editing either silently.
    original = G.read(O / 'family_summary.json')
    fixes = []
    for s in original:
        rows = fit if s['partition'] == 'fit' else cal
        actual = len({r['meta']['call_id'] for r in rows})
        if s['calls_with_eligible_prefixes'] != actual:
            fixes.append(dict(model=s['model'], partition=s['partition'], field='calls_with_eligible_prefixes',
                              before=s['calls_with_eligible_prefixes'], after=actual))
    if fixes:
        d = O / 'pre_audit_snapshots/counter_01'
        d.mkdir(parents=True, exist_ok=False)
        shutil.copy2(O / 'family_summary.json', d / 'family_summary.json')
        shutil.copy2(G.W / 'run.py', d / 'run.py')
        G.save('counter_correction.json', dict(cause='defaultdict access during iteration inserted empty groups;len(grouped) counted all calls.',
            fixes=fixes, scope='Only descriptive eligible-call counts. Score intervals,losses,queries,source data and audit replay results unchanged.',
            preserved={p.relative_to(O).as_posix(): G.sha(p) for p in d.iterdir()}))
        for s in original:
            for f in fixes:
                if s['model'] == f['model'] and s['partition'] == f['partition']: s[f['field']] = f['after']
        G.save('family_summary.json', original)
    summaries = G.read(O / 'family_summary.json')
    calls = G.read(O / 'all_call_threshold_intervals.json')
    trans = G.read(O / 'all_nonmonotone_transitions.json')
    curves = G.read(O / 'group_loss_diagnostic_curves.json')
    for s in summaries:
        cc = [c for c in calls if c['model'] == s['model'] and c['partition'] == s['partition']]
        assert s['calls'] == len(cc)
        assert s['calls_with_eligible_prefixes'] == len({r['meta']['call_id'] for r in (fit if s['partition'] == 'fit' else cal)})
        for key, field in [('action_loss_nonmonotone_calls', 'action_loss_increases'), ('joint_loss_nonmonotone_calls', 'joint_loss_increases'), ('query_nonmonotone_calls', 'query_count_decreases')]:
            assert s[key] == len({v['call_id'] for v in trans if v['model'] == s['model'] and v['partition'] == s['partition'] and v[field]})
    # Support for any future depth-specific task, without fitting or selecting a task.
    support = []
    for part, rows in [('fit', fit), ('calibration', cal)]:
        for n in sorted({r['meta']['visible_count'] for r in rows}):
            rr = [r for r in rows if r['meta']['visible_count'] == n]
            neg = [r for r in rr if not r['target']['stop_label']]
            support.append(dict(partition=part, visible_count=n, prefixes=len(rr), continue_prefixes=len(neg),
                continue_calls=len({r['meta']['call_id'] for r in neg}), continue_dates=sorted({r['meta']['date'] for r in neg}),
                stop_prefixes=len(rr)-len(neg)))
    G.save('risk_support_by_depth.json', support)
    assert sum(x['prefixes'] for x in support) == 584
    assert sum(x['continue_prefixes'] for x in support) == 100
    hh = G.read(O / 'existing_calibration_threshold_checks.json')
    finals = [next(h for h in reversed(hh) if h['model'] == model) for model in G.MODELS]
    assert all(h['action_differences'] == h['future_sup_action_failures'] == h['future_sup_joint_failures'] == 0 for h in finals)
    # Existing calibration histories are also recomputed by a separate scalar interval lookup.
    curve_checks = 0
    for curve in curves:
        cc = [c for c in calls if c['model'] == curve['model'] and c['partition'] == curve['partition']]
        for p in curve['curve']:
            losses = []
            for c in cc:
                idx = sum(s['lo_inclusive'] <= p['threshold'] for s in c['intervals']) - 1
                # Equal lower endpoints can occur at tau=1; the final fallback wins.
                losses.append((c['date'], c['intervals'][idx]['action_loss']))
            assert sum(v for d, v in losses) == p['mismatching_calls']
            dates = sorted({d for d, v in losses})
            macro = sum(sum(v for d, v in losses if d == day) / sum(d == day for d, v in losses) for day in dates) / len(dates)
            assert abs(macro - p['date_macro_mismatch']) < 1e-12
            curve_checks += 1
    sample = G.read(O / 'sample_size_illustration.json')
    for x in sample['minimum_zero_failure_dates_95pct']:
        r, n = x['illustrative_upper'], x['n']
        assert (1-r)**n <= .05 and (1-r)**(n-1) > .05
    G.verify()
    review = G.read(O / 'review.json')
    review.update(eligible_call_count_audited=True, preserved_counter_correction=bool(fixes),
                  independently_checked_group_curve_points=curve_checks, depth_support_audited=True,
                  final_calibration_cutoffs_no_later_errors_in_saved_family=True)
    G.save('review.json', review)
    notes = '''# 三篇原文与适用条件

1. Schuster等，Confident Adaptive Language Modeling，NeurIPS2022，§4.1–4.3。用局部退出分数节省生成计算，以完整序列损失校准阈值；性能不必随阈值单调，采用LTT固定顺序检验。假设校准和新提示iid。我们借鉴完整输出约束，不能照搬其语言模型实验结果。\n原文：https://papers.nips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf

2. Angelopoulos等，Conformal Risk Control，ICLR2024，定理1、§2.3。基本结论针对可交换、关于阈值非增、右连续的有界随机损失函数，控制校准与新样本联合随机性下的期望风险。非单调损失可用未来上确界转成单调上界，但可能更保守。不能把期望风险保证写成每个状态正确或高概率条件风险保证。\n原文：https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf

3. Angelopoulos等，Learn then Test，arXiv:2110.01052v5，2022修订版，§1.1、§2.1–2.3。预训练模型固定，用有效p值与FWER控制筛选风险合格参数；允许非单调风险。基础设定校准样本iid；固定顺序须不依赖同一检验数据，或另分数据学习顺序。多阈值检验之间允许相关，不代表同轨迹前缀可当iid观测。\n原文：https://arxiv.org/pdf/2110.01052v5

逐篇打开原文核对，未用二手博客作技术依据。阅读日期2026-10-24。只摘述与本次可用性判断有关的内容；不复制论文图表或实验结论。
'''
    (O / '论文阅读与适用条件.md').write_text(notes, encoding='utf8')
    example = next(t for t in trans if t['model'] == 'RF140' and t['partition'] == 'calibration' and t['action_loss_increases'])
    before, after = example['before'], example['after']
    lines = ['# GRID145：风险校准能否解决高分误停', '',
        '结论：风险控制值得借鉴，但不能靠换一种校准公式解决目前的收益问题。应先改善高分端的错误区分能力。现有阈值只排除了已有校准搜索的错误；它不等于未来风险保证。本轮不训练、不更换模型或阈值。', '',
        '## 从论文中学到什么', '',
        '[CALM](https://papers.nips.cc/paper_files/paper/2022/file/6fac9e316a4ae75ea244ddcef1982c71-Paper-Conference.pdf)将局部退出决策和完整序列的一致性联系起来。[Learn then Test](https://arxiv.org/pdf/2110.01052v5)用多重检验处理参数选择，允许非单调风险。对我们最有用的是：评价一次完整搜索的最终动作，不能用局部高分替代最终结果。', '',
        '[Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)的基本公式需要损失随阈值非增；其§2.3也给出上界单调化。它控制的是期望风险，与LTT的高概率风险控制不同。两者都需要合适的数据抽样条件，不能把同一日期的许多前缀当成独立样本。三篇原文位置和限制另见“论文阅读与适用条件.md”。', '',
        '## 提高阈值后，确实可能先对后错', '',
        f"校准记录{example['call_id']}：RF140在阈值区间[0.5,{example['threshold_boundary']:.9f})于{before['global_queries']}处停止，再补查{before['local_queries']}个，共{before['queries']}次查询，选中{before['selected']}，与原FRAME一致。阈值升至[{after['lo_inclusive']:.9f},{after['hi_exclusive']:.9f})后，改在{after['global_queries']}处停止，补查{after['local_queries']}个，共{after['queries']}次，却选中{after['selected']}，原动作仍为{after['original_selected']}。", '',
        '多查一批改变了当前最好候选，也改变了同站补查的位置。候选结果一直缓存，但不同停止深度执行的补查集合并不必然包含前一深度的补查集合，因此最终动作一致性和总查询数都不必单调。这个结论来自同一保存状态的假设搜索，不是新物理闭环实验。', '',
        '| 固定模型 | 分区 | 搜索数 | 有前缀可提前停的搜索 | 出现“先对后错”的搜索 | 总查询数非单调的搜索 |',
        '|---|---|---:|---:|---:|---:|']
    for s in summaries:
        lines.append(f"| {s['model']} | {s['partition']} | {s['calls']} | {s['calls_with_eligible_prefixes']} | {s['action_loss_nonmonotone_calls']} | {s['query_nonmonotone_calls']} |")
    lines += ['', '表中不是模型相互独立的样本数：RF140和RF142的反例来自相同搜索。枚举的是固定分数所决定的数学分段，不据此挑选新阈值。原始动作差异、联合不合格、查询变化的全部分段与反例均保存。', '',
        '## 这个现象不是本轮失败的主要原因', '',
        'RF144的校准记录没有出现先对后错，拟合记录有1次。三份模型在各自最终校准阈值以及更高阈值处，现有校准数据上都没有再出现动作或查询不合格。原程序每次抬阈值后会重新执行完整搜索，确实检查了新暴露的状态。因此，不能把损失不单调说成原程序漏审，也不能把换成LTT当成提升收益的证据。', '',
        '直接瓶颈仍是：少数继续样本拿到了极高停止分数，使统一阈值排除了大量原本有效的停止。GRID144改善了平均排序，却只留下1次开发验证提前停止。统计校准可以筛除不可靠配置，无法凭空把这些高分错误和正确停止分开。', '',
        '可以构造单调风险上界：对阈值t，检查所有更高阈值是否存在不合格搜索；也可以检查所有分数高于t的可停前缀是否存在继续标签。二者都上界实际误停，后者还可能计入永远不会先到达的前缀。已验证这些上界，不选择或采用新阈值。上界解决适用条件，可能减少停止机会，不应预先称它能改善效率。', '',
        '## 校准证据有多大', '',
        '现有校准记录有198个前缀、82次搜索、8个生成场景副本，但只有4个日历日期组，其中67次搜索有可提前停止的前缀。相同日期的副本、同次搜索的多个前缀不能重复计作独立校准样本；4个日期组本身也没有建立iid或可交换抽样资格，且这些日期已反复用于开发。', '',
        '仅为理解样本量，假设另有预先固定的策略、独立同分布日期，并把“一天内是否发生任何动作差异”记为二元损失。n个日期零失败时，单侧95%精确二项上界为1−0.05^(1/n)。n=4时约52.71%；若想在同一理想设定下零失败排除5%以上风险，至少需59个独立日期；排除1%以上风险需299个。', '',
        '这些不是当前模型的置信区间，也不是采集规模指令：当前策略用同一校准数据选择阈值，并已反复开发，还不满足该示例前提。日期内任一错误的概率也不同于平均每次搜索的错误率。CRC中单位上界B=1、n=4时修正项1/(n+1)=0.2，说明基本程序的有限样本代价，不表示真实错误率至少20%。', '',
        '原要求仍是保持原动作并减少查询，没有把允许误差改成5%或1%。有限零错误样本无法排除任意小的正失败概率；完全沿用原搜索具有结构性的动作一致性，无需靠样本证明。', '',
        '## 下一步如何推进', '',
        '不再自动换类别系数或校准公式，也暂不扩大数据。先把下一次训练要解决的问题定得更窄：在一个固定决策深度判断是否必须继续，保留现有动作保持标签和原补查规则，避免把所有深度混成同一种判断。固定256是已有强对照，但并不据开发验证的零差异宣布它普遍可靠。', '',
        '本轮已按原拟合/校准数据列出各深度的继续样本、日期支持。若拆分后负例太少，就不单独拟合一个模型；若有支持，下一方案应明确哪个深度、输入、损失以及如何用原校准完整搜索评价，训练前再通知。不根据本轮分段挑一个表现好的阈值。', '',
        '风险控制作为另一个层次保留：先冻结策略族、完整搜索或日期损失单位、抽样方案和多重检验方式，再设计新的校准证据。它负责限制错误，模型负责区分哪些状态值得继续，两者不能互相替代。', '',
        '## 审计范围', '',
        f"三份固定模型各584个拟合/校准前缀、243次完整搜索，解析分段在边界两侧共{review['interval_boundary_whole_search_checks']}次独立完整搜索重放核对；{curve_checks}个分组损失点和原9个校准历史阈值复核通过。未载入开发验证分数进行本轮重放，未调用fit/predict、无新物理仿真，最终8测试未生成。旧源文件、分数、权重、标签和阈值哈希全保持。", '',
        '审计纠正一项描述计数：构造字典时的空组插入使“有可停前缀的搜索数”误计为全部搜索数，现改为拟合155、校准67。原输出和冻结源码保存在pre_audit_snapshots/counter_01，收据counter_correction.json；分段、反例、动作及查询结果均未改变。', '',
        '这是一轮只读机制与可行性诊断；不构成新质量、实测时间或跨场景泛化结果。']
    (O / 'GRID145_report.md').write_text('\n'.join(lines)+'\n', encoding='utf8')
    G.save('complete.json', dict(passed=True, kind='read_only_risk_control_feasibility', new_fits=0, physical=0,
        validation_replay=0, adopted_policy=False, new_risk_tolerance=None, ready_to_scale=False))
    G.save('state.json', dict(stage='complete'))
    G.verify()
    G.archive('complete')
    c = R / '论文协作资料/FRAME_风险校准可行性_GRID145'
    c.mkdir(exist_ok=False)
    names = ['GRID145_report.md', '论文阅读与适用条件.md', 'design.json', 'family_summary.json',
             'all_nonmonotone_transitions.json', 'risk_support_by_depth.json', 'sample_size_illustration.json',
             'counter_correction.json', 'review.json', 'decision.json', 'complete.json']
    for n in names: shutil.copy2(O/n, c/n)
    entry = '''\n\n## 2026-10-24 — GRID145风险控制适用性诊断完成

读CALM NeurIPS2022、CRC ICLR2024、LTT原文；固定RF140/142/144各584拟合/校准前缀、243完整搜索，零拟合/predict/物理/开发验证重放。阈值解析分段4180边界核对通过；RF140/142校准同一调用出现128补查选210正确→192换站选162错误，风险/总查询不必单调。RF144校准无此反转，且三原最终阈值以上校准仍零不合格，不能把非单调当当前失败主因或原校准漏审。主瓶颈仍是高分端错误排序；4校准日期组不可冒充198独立样本，CRC/LTT不直接给现有模型未来保证。固定深度继续风险仅规划，按深度支持另存，不训练/换目标/新阈值/扩大/最终8。计数空组问题保留原快照，纠正eligible155/67；科学分段未变。报告research_archive/power-grid-2026/protocols/GRID145_risk_feasibility/complete/GRID145_report.md
'''
    for name in ['RESEARCH_STATE.md', 'DECISIONS.md', 'EXPERIMENT_LOG.md']:
        p = REPO/'docs'/name
        old = p.read_text(encoding='utf8')
        if name == 'RESEARCH_STATE.md':
            title, rest = old.split('\n', 1)
            p.write_text(title+'\n'+entry+'\n'+rest, encoding='utf8')
        else: p.write_text(old+entry, encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — GRID145后

1. 当前保留RF140与全部固定预算/失败模型；风险校准不能修复高分排序，未采用新策略或阈值。
2. 下一次训练先形成固定深度继续风险方案，依据risk_support_by_depth.json核对负例/日期支持；不预先选择好结果，不因样本少丢反例。训练前通知，原动作保持/查询目标和完整搜索评价不变。
3. 若研究统计风险保证，另定完整搜索或日期损失、独立抽样与阈值族/检验顺序；4旧校准日期和重复前缀不作独立证据。先解决模型区分，再决定是否值得采集，不自动扩大。
4. 无开发验证新阈值/模型选择，无新物理/RL/论文修改；最终8封存。旧分数/标签/权重保持，所有不利结果保留。
''', encoding='utf8')
    p = REPO/'research_archive/power-grid-2026/README.md'
    p.write_text('# 最新：GRID145只读风险控制可行性\n\n[报告](protocols/GRID145_risk_feasibility/complete/GRID145_report.md)；三篇原文适用性、阈值非单调反例、日期聚类支持。无训练或策略更换。\n\n'+p.read_text(encoding='utf8'), encoding='utf8')
    count = 0
    for phase in G.A.iterdir():
        if phase.is_dir():
            for n, h in G.read(phase/'archive_manifest.json').items():
                assert G.sha(phase/n) == h
                count += 1
    for p in c.iterdir(): assert G.sha(p) == G.sha(O/p.name)
    # Preserve nested audit snapshots in a manifest-bearing archive phase too.
    d = G.A/'counter_snapshot'
    shutil.copytree(O/'pre_audit_snapshots/counter_01', d)
    (d/'archive_manifest.json').write_text(json.dumps({p.name: G.sha(p) for p in d.iterdir()}, indent=2)+'\n', encoding='utf8')
    count += len(list(d.iterdir()))-1
    G.save('publication_verification.json', dict(passed=True, archive_files_checked=count,
        collaboration_files_checked=len(names), original_sources_verified=True, scientific_results_unchanged_by_counter_correction=True))
    d = G.A/'verification'
    d.mkdir()
    shutil.copy2(O/'publication_verification.json', d/'publication_verification.json')
    (d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':G.sha(d/'publication_verification.json')},indent=2)+'\n', encoding='utf8')
    print(json.dumps(dict(completed=True, corrections=fixes, depth_support=support, curve_checks=curve_checks), ensure_ascii=False))

if __name__ == '__main__': main()
