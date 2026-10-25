"""Separate a better helper from how the fixed outer gate uses it; no new policy."""
from pathlib import Path
import collections,hashlib,importlib.util,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid175_helper_gate_audit';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID175_helper_gate_audit'
sp=importlib.util.spec_from_file_location('g175_qualified',R/'work/grid174_power_helper/run.py');D=importlib.util.module_from_spec(sp);sp.loader.exec_module(D);read=D.read;sha=D.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def main():
    assert not O.exists();O.mkdir(parents=True);D.verify_trained();assert read(D.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID175_HELPER_GATE_MECHANISM',authorization='User sustain modest learning improvements; inspect current fixed records,no newfit.',
        question='POWER8 helper improves alone whilefixedgateworsens. Quantify helperfirstdifferences, exactgatevisibility, actualcompletegatedifferences without proposing a new gate.',
        population='All7134OOFseed-startpairs,all16dates/4folds/3seeds and10746heldstate-seedrows.',
        identities='Gatego=F AND(R OR C);changingCcanchangegateonlywhenFCONT andRSTOP. Distinguish stateidentity from fulltrajectory effects. Helperalone contributions cannot be called literal transferable gate gains.',
        prohibitions=['newrules/thresholdscans','newmodels/forwards/fits','newdata/physics/final8','paper']))
    paths=[D.O/n for n in ['complete.json','review.json','design.json','state_scores.npz','summary.json','family_summary.json','oof_trajectories.json','all7134_paired_events.json','source_hashes.json','code_freeze.json','trained_freeze.json']]+[p/n for p in D.sources() for n in ['state_scores.npz','metadata.json','fold_arrays.npz']]+[R/'outputs/grid102_search_rl/cases.pkl',Path(__file__),O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    scores=np.load(D.O/'state_scores.npz');statecounts=collections.defaultdict(collections.Counter);state_records=[]
    for fold,path in enumerate(D.sources()):
        old=np.load(path/'state_scores.npz');meta=read(path/'metadata.json');hi=np.load(path/'fold_arrays.npz')['held_ids']
        for seed in [0,1,2]:
            F=old[f'FACTOR_seed{seed}'];RR=old[f'ROUTED_seed{seed}'];c0=scores[f'fold{fold}_ZERO8_seed{seed}'];c1=scores[f'fold{fold}_POWER8_seed{seed}']
            f=F[:,1]>F[:,0];r=RR[:,1]>RR[:,0];a=c1[:,1]>c1[:,0];b=c0[:,1]>c0[:,0]
            gate0=f&(r|b);gate1=f&(r|a);assert np.array_equal((gate0!=gate1),f&~r&(a!=b))
            for i in hi:
                if a[i]==b[i]:continue
                direction='NEW_CONTINUE' if a[i] else 'NEW_STOP';reason='FACTOR_STOP' if not f[i] else ('ROUTED_CONTINUE' if r[i] else 'ACTIVE_GATE_PIVOT')
                statecounts[direction][reason]+=1;state_records.append(dict(fold=fold,seed=seed,sample_id=meta[i]['sample_id'],position=meta[i]['position'],date=meta[i]['date'],physical_state_group=meta[i]['physical_state_group'],direction=direction,reason=reason))
    save('changed_state_gate_visibility.json',state_records);save('state_visibility_summary.json',{k:dict(v) for k,v in statecounts.items()})
    tr=read(D.O/'oof_trajectories.json');by={(x['method'],x['sample_id']):x for x in tr};cases={c['sample_id']:c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'};datecounts=collections.Counter(c['date'] for c in cases.values());lookups=[{(m['sample_id'],m['position']):i for i,m in enumerate(read(p/'metadata.json'))} for p in D.sources()];folds={date:f for f,p in enumerate(D.sources()) for date in read(p/'design.json')['fold']['held_dates']};helper=[];gate=[]
    for seed in [0,1,2]:
        for c in cases.values():
            sid=c['sample_id'];fold=folds[c['date']];weight=1/(3*16*datecounts[c['date']]);a=by[f'POWER8_HELPER_seed{seed}',sid];b=by[f'ZERO8_HELPER_seed{seed}',sid];F=by[f'FACTOR_seed{seed}',sid];ga=by[f'POWER8_GATE_seed{seed}',sid];gb=by[f'ZERO8_GATE_seed{seed}',sid];old=np.load(D.sources()[fold]/'state_scores.npz')
            def event(x,y):
                j=min(x['batches'],y['batches']);changed=x['batches']!=y['batches'];nd=x['net_utility']-y['net_utility']
                return dict(seed=seed,sample_id=sid,date=c['date'],physical_state_group=c['physical_state_group'],new_batches=x['batches'],old_batches=y['batches'],direction='UNCHANGED' if not changed else ('NEW_CONTINUE' if x['batches']>y['batches'] else 'NEW_STOP'),net_delta=nd,gain_delta=x['gain']-y['gain'],saved_queries=y['queries']-x['queries'],same_choice=x['chosen_pool_id']==y['chosen_pool_id'],weight=weight,first_difference=j if changed else None)
            h=event(a,b)
            if h['direction']!='UNCHANGED':
                j=h['first_difference'];i=lookups[fold][sid,j];r=old[f'ROUTED_seed{seed}'][i];route=bool(r[1]>r[0])
                h['gate_access']='FACTOR_ALREADY_STOPPED' if F['batches']<=j else ('EARLIER_GATE_STOP' if min(ga['batches'],gb['batches'])<j else ('ROUTED_CONTINUE' if route else 'REACHABLE_HELPER_PIVOT'))
                h['next_gain']=float(c['gains'][j]);h['next_queries']=int(c['queries'][j])
            helper.append(h);g=event(ga,gb)
            if g['direction']!='UNCHANGED':
                j=g['first_difference'];i=lookups[fold][sid,j];fq=old[f'FACTOR_seed{seed}'][i];rq=old[f'ROUTED_seed{seed}'][i];assert fq[1]>fq[0] and not rq[1]>rq[0]
                g['next_gain']=float(c['gains'][j]);g['next_queries']=int(c['queries'][j])
            gate.append(g)
    save('all7134_helper_pairs.json',helper);save('all7134_gate_pairs.json',gate)
    # Group complete-trajectory contributions; no single-state score is substituted for return.
    def decompose(rows,field):
        result=[]
        for k in sorted({r[field] for r in rows}):
            rr=[r for r in rows if r[field]==k];result.append(dict(kind=k,pairs=len(rr),physical_groups=len({r['physical_state_group'] for r in rr}),positive_pairs=sum(r['net_delta']>0 for r in rr),negative_pairs=sum(r['net_delta']<0 for r in rr),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr)))
        return result
    hd=decompose(helper,'direction');gd=decompose(gate,'direction');access=decompose([h for h in helper if h['direction']!='UNCHANGED'],'gate_access');save('helper_direction_decomposition.json',hd);save('gate_direction_decomposition.json',gd);save('helper_gate_access_decomposition.json',access)
    family=read(D.O/'family_summary.json');h0=family['ZERO8_HELPER'];h1=family['POWER8_HELPER'];g0=family['ZERO8_GATE'];g1=family['POWER8_GATE'];dh=h1['mean_net_utility']-h0['mean_net_utility'];dg=g1['mean_net_utility']-g0['mean_net_utility'];interaction=(g1['mean_net_utility']-h1['mean_net_utility'])-(g0['mean_net_utility']-h0['mean_net_utility']);assert abs(dg-dh-interaction)<1e-12
    assert abs(dh-sum(r['net_contribution'] for r in hd))<1e-12 and abs(dg-sum(r['net_contribution'] for r in gd))<1e-12
    s=read(D.O/'summary.json');seed_pairs=[]
    for seed in [0,1,2]:
        a=s[f'POWER8_HELPER_seed{seed}'];b=s[f'ZERO8_HELPER_seed{seed}'];seed_pairs.append(dict(seed=seed,delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],relative=a['macro']['mean_net_utility']/b['macro']['mean_net_utility']-1,per_date=[dict(date=x['date'],delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    dates=[dict(date=v['date'],delta=float(np.mean([next(z['delta'] for z in p['per_date'] if z['date']==v['date']) for p in seed_pairs]))) for v in seed_pairs[0]['per_date']]
    key=dict(helper_net_delta=dh,helper_relative_change=dh/h0['mean_net_utility'],helper_query_reduction=1-h1['mean_queries']/h0['mean_queries'],helper_gain_capture_percentage_points=100*(h1['gain_capture']-h0['gain_capture']),helper_positive_seeds=sum(p['delta']>0 for p in seed_pairs),helper_positive_dates=sum(p['delta']>0 for p in dates),helper_positive_seed_dates=sum(v['delta']>0 for p in seed_pairs for v in p['per_date']),gate_net_delta=dg,gate_relative_change=dg/g0['mean_net_utility'],outer_composition_difference=interaction,helper_relative_to_factor=h1['mean_net_utility']/family['FACTOR']['mean_net_utility']-1)
    save('key_changes.json',key);save('helper_paired_summary.json',dict(pairs=seed_pairs,per_date=dates))
    lines=['# GRID175：小模型变好了，组合为什么没有跟着变好','',
        f"同容量配对下，电气输入让小模型单独决策的局部净效用提高{key['helper_relative_change']:.2%}，三个种子都提高，10/16个日期平均提高；平均查询减少{key['helper_query_reduction']:.2%}，收益保留率提高{key['helper_gain_capture_percentage_points']:.2f}个百分点。这个正向结果应保留，不能因为组合平均下降就说输入没有用。",'',
        f"但原组合的效用变化为{key['gate_relative_change']:+.2%}。小模型与组合是两个不同层次：小模型变好，不代表固定外层规则能利用它的全部新判断。POWER8小模型效用仍比原FACTOR低{abs(key['helper_relative_to_factor']):.2%}，不能宣布已经整体取代原方法。",'',
        '## 先看小模型的实际变化','','| 首次变化方向 | 配对数 | 净正 / 净负 | 宏平均效用贡献 |','|---|---:|---:|---:|']
    for r in hd:lines.append(f"| {r['kind']} | {r['pairs']} | {r['positive_pairs']} / {r['negative_pairs']} | {r['net_contribution']:+.8f} |")
    lines+=['','## 固定外层规则能看到哪些改变','',
        '外层决定为F继续 AND（R继续 OR C继续）。因此只有F继续且R停止的位置，小模型C的改变才可能改变外层决定；F已经停止时，小模型无法重新打开搜索；R继续时，外层本来就会继续。逐状态的这个恒等式已经全量核对。','',
        '| 小模型变化方向 | F停止屏蔽 | R继续屏蔽 | 可影响外层的位置 |','|---|---:|---:|---:|']
    for direction,counts in statecounts.items():lines.append(f"| {direction} | {counts['FACTOR_STOP']} | {counts['ROUTED_CONTINUE']} | {counts['ACTIVE_GATE_PIVOT']} |")
    lines+=['','小模型完整搜索首次分歧所在的位置还受更早停止影响，分布如下。表中贡献是小模型独立运行的效用变化，不能把被屏蔽部分直接当成改组合后一定能获得的收益。','',
        '| 首次分歧位置的外层状态 | 配对数 | 净正 / 净负 | 小模型效用贡献 |','|---|---:|---:|---:|']
    for r in access:lines.append(f"| {r['kind']} | {r['pairs']} | {r['positive_pairs']} / {r['negative_pairs']} | {r['net_contribution']:+.8f} |")
    lines+=['','## 组合自身的变化仍须单独看','','| 首次变化方向 | 配对数 | 净正 / 净负 | 宏平均效用贡献 |','|---|---:|---:|---:|']
    for r in gd:lines.append(f"| {r['kind']} | {r['pairs']} | {r['positive_pairs']} / {r['negative_pairs']} | {r['net_contribution']:+.8f} |")
    lines+=['',f"数值分解严格守恒：小模型效用变化{dh:+.8f}，加上外层组合相对于小模型的变化{interaction:+.8f}，得到组合效用变化{dg:+.8f}。这是不同策略完整轨迹之间的差值恒等式，不是声称某条注意力或一个字段造成全部结果。",'',
        '## 下一步','',
        '保留电气输入这条正向线索，下一步应同时看有利机会识别和组合约束的取舍。先围绕固定原模型停止位置中，小模型新增继续的实际收益、代价与公开证据做只读核查，再决定一个具体组合或学习方案。不能直接删除外层限制、挑最好种子，或从这轮留出结果挑触发阈值。',
        '', '当前没有新训练、模型前向或新门控实验。全部24份模型、负向日期和完整搜索仍保留；结果限于旧同网局部补查任务，不宣称整周质量或完整控制提速。']
    (O/'GRID175_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    for p,h in read(O/'source_hashes.json').items():assert sha(R/p)==h,p
    D.verify_trained();save('review.json',dict(passed=True,all7134_helper_and_gate_pairs=True,state_gate_identity_rows=10746,weighted_fulltrajectory_decompositions=True,new_fits=0,new_forwards=0,new_physics=0,final8_absent=True))
    save('decision.json',dict(helper_improvement_retained=True,three_seed_positive=True,outer_gate_not_improved=True,next='Read-only newlyprofitablecontinue opportunities excluded by oldFSTOP;do not remove gate or tune threshold yet.',adopted=False,ready_to_scale=False));save('complete.json',dict(passed_audit=True,new_fits=0,new_forwards=0,new_physics=0))
    dest=A/'complete';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
    collab=R/'论文协作资料/FRAME_小模型改善与组合差异_GRID175';collab.mkdir(exist_ok=False)
    for n in ['GRID175_report.md','key_changes.json','helper_paired_summary.json','helper_direction_decomposition.json','gate_direction_decomposition.json','state_visibility_summary.json','review.json','decision.json','complete.json']:shutil.copy2(O/n,collab/n);assert sha(collab/n)==sha(O/n)
    msg=f"GRID175保留POWER8小模型正向：四折16日期效用+{key['helper_relative_change']:.2%}、查询少{key['helper_query_reduction']:.2%}、收益保留率+{key['helper_gain_capture_percentage_points']:.2f}pp，3/3种子、10/16日期正；组合-1.10不抹去小模型进步。全10746状态F AND(R OR C)可见性、7134成对完整轨迹分解守恒。下一步只读原FSTOP屏蔽的新有利继续机会，不直接删限制/调阈值。无新拟合/前向/物理/最终8。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID175小模型改善与组合\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 保留小模型改善，查清组合取舍\n\n1. POWER8小模型四折3种子均正、效用+4.36%、查询少10.47%、收益保留率+5.39pp，保留；组合-1.10与小模型进步分开，不宣布整体替代。\n2. 先只读追F原STOP屏蔽的小模型新增继续机会：完整收益/查询/物理组/公开信息；不直接删限制或选新阈值救结果。新组合/学习实验先给针对性设计和训练通知。\n3. 旧小改善和新反例全部保留；不新物理/数据/最终8，不改论文，不拿局部NET或旧计时称整周质量/控制提速。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID175小模型的正向改善\r\n\r\n[报告](protocols/GRID175_helper_gate_audit/complete/GRID175_report.md)；电气输入小模型效用+4.36%、3种子均正，区分组合限制。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID175_COMPLETE',json.dumps(key,ensure_ascii=False))
if __name__=='__main__':main()
