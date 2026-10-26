"""Locate existing learned savings against FRAME; no new predictions or fits."""
from pathlib import Path
import collections,hashlib,json,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid184_search_adapter';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID184_search_adapter'
sys.path.insert(0,str(R/'work/grid184_search_adapter'));from run import read,sha
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
rs=read(O/'review.json')['runrows'];lookup={(r['repeat'],r['week'],r['method']):r for r in rs};cache={}
def calls(row):
    k=(row['repeat'],row['week'],row['method'])
    if k not in cache:
        folder=O/'runs'/f"r{k[0]}__{k[1]}__{k[2]}";out={}
        for p in folder.glob('call[0-9][0-9][0-9].json'):
            c=read(p)
            with np.load(p.with_name(p.stem+'_state.npz')) as z:h=hashlib.sha256(z['base_action'].tobytes()).hexdigest()
            out[c['step'],c['state_hash'],h]=c
        cache[k]=out
    return cache[k]
families={}
for mode in ['FACTOR','CONSENSUS']:
    counts=collections.Counter();details=[];pairs=[]
    for r in rs:
        if not r['method'].startswith(mode+'_seed'):continue
        b=lookup[r['repeat'],r['week'],'FRAME_GUARD'];pairs.append(dict(seed=r['method'].split('seed')[1],date=r['date'],repeat=r['repeat'],control_reduction=1-r['controller_s']/b['controller_s'],candidate_query_reduction=1-r['candidate_queries']/b['candidate_queries']))
        if r['repeat']!=0:continue
        a_call=calls(r);b_call=calls(b)
        for k in set(a_call)&set(b_call):
            a,c=a_call[k],b_call[k];delta=a['public_queries']-c['public_queries'];same=a['chosen_pool_id']==c['chosen_pool_id'];kind=('fewer' if delta<0 else 'more' if delta>0 else 'same')+('_same_chosen' if same else '_different_chosen');counts[kind]+=1
            details.append(dict(seed=r['method'].split('seed')[1],date=r['date'],step=k[0],class_name=kind,query_delta=delta,learner_chosen=a['chosen_pool_id'],frame_chosen=c['chosen_pool_id']))
    families[mode]=dict(counts=dict(counts),pairs=details,macro_control_reduction=float(np.mean([v['control_reduction'] for v in pairs])),macro_candidate_query_reduction=float(np.mean([v['candidate_query_reduction'] for v in pairs])))
save(O/'baseline_aligned_call_tradeoffs.json',dict(families=families,scope='Onlysameobservedstate+base,repeat0,actualboughtfeedback. Prefixes/seedsnotindependentscenarios;querysavingsnotwholeweekqualityproof.'))
f=families['FACTOR'];c=families['CONSENSUS'];report=(O/'GRID184_report.md').read_text(encoding='utf8');old='前者是可以继续研究的具体省查询机会，但这些调用不独立、不能替代整周结果；不因整体组合暂未胜出就删掉这些有利点。';assert old in report
report=report.replace(old,'本轮相对FACTOR未找到同状态、同动作且省查询的共识调用，不能凭空声称该组合增加了省查询机制。')
report+=f"\n\n## 原学习模块已有的收益仍值得保留\n\n进一步只用已有记录，将原FACTOR与未学习FRAME配对：相同真实观测和基础动作下有{f['counts'].get('fewer_same_chosen',0)}个查询更少且最终动作相同的调用。FACTOR相对FRAME在同等已运行时域的控制时间宏平均减少{100*f['macro_control_reduction']:.2f}%，候选查询减少{100*f['macro_candidate_query_reduction']:.2f}%。共识对FRAME也有{c['counts'].get('fewer_same_chosen',0)}个同动作省查询调用，但相对FACTOR没有新增此类调用。上述调用按全部三种子计数，可能共享日期和物理状态，不当独立样本。\n\n因此应保留原学习模块的有效省查询点，重点检查便宜共识为什么覆盖了原FACTOR的一些停止决定；当前证据不支持把组合直接替代FACTOR，也不需要立即否定学习路线或追加新实验。基准对照上的整周成本失败仍须分开解释。\n"
(O/'GRID184_report.md').write_text(report,encoding='utf8',newline='\n');dest=R/'论文协作资料/FRAME_小闭环结果_GRID184';shutil.copy2(O/'GRID184_report.md',dest/'GRID184_report.md');shutil.copy2(O/'baseline_aligned_call_tradeoffs.json',dest/'baseline_aligned_call_tradeoffs.json')
pre=read(O/'publication_preflight.json');pre.update(report_sha256=sha(O/'GRID184_report.md'),factor_vs_frame_same_action_savings_calls=f['counts'].get('fewer_same_chosen',0),consensus_vs_frame_same_action_savings_calls=c['counts'].get('fewer_same_chosen',0));save(O/'publication_preflight.json',pre)
for p in [O/'GRID184_report.md',O/'baseline_aligned_call_tradeoffs.json',O/'publication_preflight.json',Path(__file__)]:shutil.copy2(p,A/'final_delivery'/p.name)
save(A/'final_delivery/archive_manifest.json',{p.name:sha(p) for p in (A/'final_delivery').iterdir() if p.is_file() and p.name!='archive_manifest.json'})
raw=A/'raw_records';save(raw/'archive_manifest.json',{p.relative_to(raw).as_posix():sha(p) for p in raw.rglob('*') if p.is_file() and p.name!='archive_manifest.json'})
msg=f"已对已有记录补充定位：FACTOR对FRAME有{f['counts'].get('fewer_same_chosen',0)}个同状态同动作省查询调用，控制时间宏平均少{100*f['macro_control_reduction']:.2f}%/候选少{100*f['macro_candidate_query_reduction']:.2f}%。C对FRAME有{c['counts'].get('fewer_same_chosen',0)}个，但C对F为0，不给组合捏造增益；保留原学习模块正向，下一步只解释C覆盖F停止的少数分歧。无新预测/拟合/物理。"
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID184保留有效学习点\n\n'+msg+'\n'
    if n=='RESEARCH_STATE.md':a,b=t.split('\n',1);t=a+'\n'+entry+'\n'+b
    else:t+=entry
    p.write_text(t,encoding='utf8',newline='\n')
print('GRID184_EXISTING_SAVINGS',json.dumps({mode:{k:v for k,v in row.items() if k!='pairs'} for mode,row in families.items()}))
