"""Independent label accounting; preserve and document an unused scope excess."""
from pathlib import Path
import collections,copy,json,pickle,shutil,subprocess,sys
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.verify();assert C.read(C.O/'complete.json')['passed'];assert not (C.O/'publication_audit.json').exists()
    rows=C.read(C.O/'fit_label_rows.json');held=C.read(C.O/'held_diagnostic_rows.json');pools=C.read(C.O/'pool_statistics.json');summary=C.read(C.O/'summary.json')
    cases={c['sample_id']:c for c in pickle.loads((C.R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'}
    task=C.read(C.R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];checked=0
    for r in [*rows,*held]:
        c=cases[r['sample_id']];j=r['position'];assert r['gain']==float(c['gains'][j]);assert r['queries']==int(c['queries'][j]);assert abs(r['NET']-(r['gain']-fee*r['queries'])/scale)<1e-12;checked+=1
    for pool in pools:
        f,s=pool['fold'],pool['seed'];rr=[r for r in rows if r['fold']==f and r['seed']==s and r['domain']=='STOP'];positive=[r for r in rr if r['gain']>0]
        vals=sorted({r['gain'] for r in positive});threshold=next(v for v in vals if sum(r['mass'] for r in positive if r['gain']<=v)>=.95*sum(r['mass'] for r in positive))
        assert threshold==pool['fit_only_threshold']
        dates=collections.defaultdict(float);calls=collections.defaultdict(float);starts=collections.defaultdict(float)
        for r in rr:dates[r['date']]+=r['mass'];calls[r['group']]+=r['mass'];starts[r['sample_id']]+=r['mass']
        assert all(abs(v-1/12)<1e-12 for v in dates.values())
        for d in dates:
            gs={r['group'] for r in rr if r['date']==d};assert all(abs(calls[g]-1/(12*len(gs)))<1e-12 for g in gs)
        hh=[r for r in held if r['fold']==f and r['seed']==s];assert all(r['fit_p95']==threshold and r['at_least_fit_p95']==(r['gain']>=threshold) for r in hh)
        assert not {r['date'] for r in rr}&{r['date'] for r in hh}
    # summarize() also emitted descriptive held-side quantiles although no held
    # quantile was allowed in the plan. They were never used as decision cutoffs
    # or in the report. Keep originals; canonical copies omit these extra fields.
    canonical_pools=copy.deepcopy(pools);drop=['positive_gain_median','positive_gain_p95','fit_p95_tail_positive_label_share','fit_p95_tail_reward_share']
    for p in canonical_pools:
        for key in drop:p['HELD_STOP'].pop(key)
    canonical_summary=copy.deepcopy(summary)
    for key in ['fit_p95_tail_positive_label_share','fit_p95_tail_reward_share']:canonical_summary['domains']['HELD_STOP'].pop(key)
    C.save('canonical_pool_statistics.json',canonical_pools);C.save('canonical_summary.json',canonical_summary)
    note='原run.py的通用summarize函数还计算并保存了留出侧的描述性分位数及其尾部占比，超出了方案“不计算留出分位数”的限定。这些额外字段没有参与任何高收益分组、门控动作、训练、报告表格或结论。原源代码、原输出与完整归档保留；canonical_pool_statistics.json与canonical_summary.json移除这几项字段，作为本阶段正式统计版本。所有报告中的留出高收益分组均逐项重验为仅由相应拟合STOP侧95分位阈值确定。本修正不增加训练或评价策略。\n'
    (C.O/'scope_addendum.md').write_text('# 统计范围补充说明\n\n'+note,encoding='utf8',newline='\n')
    C.save('publication_audit.json',dict(passed=True,protocol_scope_excess_documented_and_canonicalized=True,originals_preserved=True,scope_excess='Unused descriptive held quantiles emitted by generic summary function; no held-derived threshold used in policy or report',all_fit_only_thresholds_independently_reproduced=True,all_date_and_call_masses_independently_equal=True,raw_label_records_checked=checked,fit_records=len(rows),held_records=len(held),new_fits=0,new_NET_forwards=0,new_physics=0))
    dest=C.R/'论文协作资料/FRAME_门控标签幅度_GRID195'
    for name in ['canonical_pool_statistics.json','canonical_summary.json','scope_addendum.md','publication_audit.json']:shutil.copy2(C.O/name,dest/name)
    C.archive('audited')
    # Reference canonical fields explicitly without changing previous frozen text.
    p=C.REPO/'docs/EXPERIMENT_LOG.md';p.write_text(p.read_text(encoding='utf8')+'\nGRID195范围补充：原通用汇总附带未使用的留出描述分位数，正式canonical统计已移除并保留原件；所有实际分组阈值均来自拟合侧、独立复核通过。详见audited/scope_addendum.md。\n',encoding='utf8',newline='\n')
    print('GRID195_AUDITED',checked)
def delivery():
    C.verify();assert C.read(C.O/'publication_audit.json')['passed'];head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote
    n=0
    for mf in C.A.rglob('archive_manifest.json'):
        for name,h in C.read(mf).items():assert C.sha(mf.parent/name)==h;n+=1
    dest=C.R/'论文协作资料/FRAME_门控标签幅度_GRID195';names=[]
    for p in dest.iterdir():
        assert C.sha(p)==C.sha(C.O/p.name)==C.sha(C.A/'audited'/p.name)
        ap=C.A/'audited'/p.name;assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes();names.append(p.name)
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,archive_files_checked=n,collaboration_files_checked=len(names),committed_bytes_match=True,new_fits=0,new_physics=0)
    C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
