"""Check contribution conservation and deliver a table-escaped report copy."""
from pathlib import Path
import collections,json,shutil,subprocess,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
def git(*args):return subprocess.check_output(['git','-C',str(C.REPO),*args])
def audit():
    C.verify();assert C.read(C.O/'complete.json')['passed'];assert not (C.O/'publication_audit.json').exists();pairs=C.read(C.O/'all_pair_accounting.json');calls=C.read(C.O/'all_call_contributions.json');rep=C.read(C.O/'all_call_repetition.json');summary=C.read(C.O/'summary.json');assert len(pairs)==7134 and len(calls)==666 and len(rep)==222
    pred=C.read(C.NEW/'gate_predictions.json');pi={(r['seed'],r['sample_id']):r for r in pred};groups=collections.defaultdict(list)
    for p in pairs:groups[p['seed'],p['group']].append(p)
    for call in calls:
        rr=groups[call['seed'],call['group']]
        for key in ['actual_gain_delta','actual_query_delta','matched_gain_delta','matched_query_delta','matched_NET_delta']:assert abs(sum(r[key] for r in rr)-call[key])<1e-12
    for r in rep:
        rr=[p for p in calls if p['group']==r['group']];assert len(rr)==3;assert sum(p['actual_gain_delta']>0 for p in rr)==r['actual_positive_seeds'];assert sum(p['matched_NET_delta']>0 for p in rr)==r['matched_positive_NET_seeds']
    pm=C.read(C.NEW/'predictive_metrics.json')['all12_mean']['gain_readout'];assert abs(pm['normalized_MSE']-summary['mean_profiles']['held_STOP']['normalized_MSE'])<1e-12
    spent=0.;waste=0.;rawspent=0;rawwaste=0
    for r in pairs:
        if r['new_append']:
            p=pi[r['seed'],r['sample_id']];q=p['actual_next_queries'];spent+=r['macro_mass']*q;rawspent+=q
            if p['actual_next_gain']==0:waste+=r['macro_mass']*q;rawwaste+=q
    C.save('wasted_query_audit.json',dict(macro_added_query_total=spent,macro_zero_gain_query_total=waste,macro_zero_gain_query_fraction=waste/spent,raw_added_queries=rawspent,raw_zero_gain_queries=rawwaste,raw_zero_gain_query_fraction=rawwaste/rawspent,conditional_pool_profile_fraction=summary['mean_profiles']['held_STOP']['zero_gain_query_fraction'],different_weighting_explicit=True))
    p=C.O/'GRID197_report.md';text=p.read_text(encoding='utf8');fixed=text.replace('|call',r'\|call');assert fixed!=text
    fixed+='\n## 发布格式核对\n\n调用标识中的竖线已在本v2副本转义，避免Markdown表格错列；原报告、源代码及原归档保留，数值未变。\n\n按原全量日期宏平均口径，本轮追加候选查询中零奖励占比为'+f'{100*waste/spent:.2f}%；上文84.32%是先逐条件池计算比例再取均值，权重不同，两者不能混用。\n'
    (C.O/'GRID197_report_v2.md').write_text(fixed,encoding='utf8',newline='\n')
    C.save('publication_audit.json',dict(passed=True,all7134pairs_and_666seedcalls_rechecked=True,all222_repetition_spectra_rechecked=True,held_profile_MSE_matches_independent_GRID196_evaluation=True,zero_gain_queries_independently_accounted=True,report_format_only_revision=True,original_report_sha256=C.sha(p),corrected_report_sha256=C.sha(C.O/'GRID197_report_v2.md'),new_fits=0,new_NET_forwards=0,new_physics=0));C.archive('audited')
    dest=C.R/'论文协作资料/FRAME_收益改善与浪费_GRID197'
    for n in ['GRID197_report_v2.md','publication_audit.json','wasted_query_audit.json']:shutil.copy2(C.O/n,dest/n)
    p=C.REPO/'research_archive/power-grid-2026/README.md';text=p.read_text(encoding='utf8');text=text.replace('protocols/GRID197_gain_opportunity_audit/complete/GRID197_report.md','protocols/GRID197_gain_opportunity_audit/audited/GRID197_report_v2.md',1);p.write_text(text,encoding='utf8',newline='\n')
    p=C.REPO/'docs/EXPERIMENT_LOG.md';p.write_text(p.read_text(encoding='utf8')+'\nGRID197发布核对：v2仅转义调用标识竖线，原稿保留；全量日期宏平均零奖励追加查询比例'+f'{100*waste/spent:.2f}%，与条件池平均84.32%权重不同，已分别记录。\n',encoding='utf8',newline='\n')
    print('GRID197_AUDITED',json.dumps(C.read(C.O/'wasted_query_audit.json')))
def delivery():
    C.verify();assert C.read(C.O/'publication_audit.json')['passed'];count=0
    for mf in C.A.rglob('archive_manifest.json'):
        for name,h in C.read(mf).items():assert C.sha(mf.parent/name)==h;count+=1
    dest=C.R/'论文协作资料/FRAME_收益改善与浪费_GRID197';names=[]
    for p in dest.iterdir():
        ap=C.A/'audited'/p.name;assert C.sha(p)==C.sha(C.O/p.name)==C.sha(ap);assert git('show','HEAD:'+ap.relative_to(C.REPO).as_posix())==ap.read_bytes();names.append(p.name)
    head=git('rev-parse','HEAD').decode().strip();remote=git('ls-remote','lntano-src','refs/heads/main').decode().split()[0];assert head==remote
    rec=dict(passed=True,verified_publication_head=head,remote_main=head,archive_files_checked=count,collaboration_files_checked=len(names),committed_bytes_match=True,new_fits=0,new_physics=0)
    C.save('delivery_receipt.json',rec);d=C.A/'delivery';d.mkdir(exist_ok=False)
    for p in [Path(__file__),C.O/'delivery_receipt.json']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:C.sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n');print(json.dumps(rec))
if __name__=='__main__':globals()[sys.argv[1]]()
