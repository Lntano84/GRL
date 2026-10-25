"""One fixed cheap-model consensus with FACTOR delegation; zero fitting/forwards."""
from pathlib import Path
import collections,hashlib,importlib.util,json,math,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid179_cheap_consensus';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID179_cheap_consensus'
sp=importlib.util.spec_from_file_location('g179_state',R/'work/grid178_power_block_ablation/run.py');S=importlib.util.module_from_spec(sp);sp.loader.exec_module(S);D=S.D;read=S.read;sha=S.sha;H=D.H
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    S.verify_trained()
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
def prepare():
    assert not O.exists();O.mkdir(parents=True);S.verify_trained();assert read(S.O/'complete.json')['passed_audit'];assert read(D.O/'complete.json')['passed_audit']
    save('design.json',dict(id='GRID179_CHEAP_CONSENSUS',authorization='User continue adopting useful small-model decisions; announced onefixed no-retraining composition.',
        rationale='STATE4 andPOWER8 improve separately butouterFstopblocksnewprofitablecontinues. Test one symmetric agreement/delegation rule using both already-trained physical views.',
        rule='IfSTATE4go==POWER8go,use thatgo;else useFACTORgo. Equivalent to majority ofthree strictq1>q0 decisions. No threshold/weight/seed/fold/modelsearch.',
        interpretation='Ordinary consensus/delegation,not a newNNarchitecture. Two physicalhelpers overlap and arenot independent confidence estimates. Previously exposed16dates andmodels;developmental comparison,not independent confirmation.',
        compute='Cheappairfirst;FACTOR neededonlyondisagreement. No ROUTED helper in thisnewrule. Actual feature/inference orcontroltiming not claimed by saved-score arithmetic.',
        query_bound='Unlike oldFAND(RORC),canreopenFSTOP and canincreasequeries. Only original full-local candidate budget is an upperbound. Never reuse old query-nonincreasing proof.',
        population='All4existingOOFfolds,16originaltrainingdates,2378starts x3seeds,10746heldstate-seedrows. Allseeds/dates/negativeoutcomes retained. Original4dev not newlyscored;final8sealed.',
        evaluation='Complete fixed-order localreplay with old best-observedaction. CompareFACTOR,STATE4/POWER8alone andtheir oldgates. Decompose firstdifferenceNEWCONT/NEWSTOP,fulltrajectory net,queries,gain,capture,physicalgroups.',
        prohibited=['extra composition/rule/thresholdscan','newforwards/fits','newdata/physical/final8','paper','automatic adoption/scale']))
    paths=[S.O/n for n in ['complete.json','review.json','trained_freeze.json','state_scores.npz','oof_trajectories.json','family_summary.json']]+[D.O/n for n in ['complete.json','review.json','state_scores.npz','family_summary.json']]+[p/n for p in D.sources() for n in ['metadata.json','fold_arrays.npz','state_scores.npz','design.json']]+[R/'outputs/grid102_search_rl/cases.pkl',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in W.glob('*.py')});verify();archive('frozen');print('GRID179_FROZEN')
def analyse():
    verify();assert not (O/'review.json').exists();q4=np.load(S.O/'state_scores.npz');q8=np.load(D.O/'state_scores.npz');cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];task=read(D.O/'design.json');fee=task['cost_per_query'];scale=task['gain_scale'];tr=[];decisions={};checks=collections.Counter();statecells=collections.Counter();blocked=[]
    for fold,path in enumerate(D.sources()):
        old=np.load(path/'state_scores.npz');meta=read(path/'metadata.json');lookup={(m['sample_id'],m['position']):i for i,m in enumerate(meta)};hi=np.load(path/'fold_arrays.npz')['held_ids'];dates=set(read(path/'design.json')['fold']['held_dates'])
        for seed in [0,1,2]:
            F=old[f'FACTOR_seed{seed}'];a=q4[f'fold{fold}_seed{seed}'];b=q8[f'fold{fold}_POWER8_seed{seed}'];f=F[:,1]>F[:,0];s=a[:,1]>a[:,0];p=b[:,1]>b[:,0];go=np.where(s==p,s,f);majority=(f.astype(int)+s.astype(int)+p.astype(int))>=2;assert np.array_equal(go,majority);decisions[f'fold{fold}_seed{seed}']=go;checks['held_logic_rows']+=len(hi)
            for i in hi:
                kind='DELEGATE_DISAGREEMENT' if s[i]!=p[i] else ('AGREE_CONTINUE' if s[i] else 'AGREE_STOP');statecells[kind]+=1
                if not f[i] and s[i] and p[i]:blocked.append(dict(fold=fold,seed=seed,sample_id=meta[i]['sample_id'],position=meta[i]['position'],date=meta[i]['date'],physical_state_group=meta[i]['physical_state_group']))
            for c in cases:
                if c['date'] not in dates:continue
                ids=[lookup[c['sample_id'],j] for j in range(len(c['states']))];r=H.trajectory(c,go[ids],f'CONSENSUS_seed{seed}',fee,scale)
                pos=next((j for j,i in enumerate(ids) if not go[i]),len(ids));assert pos==r['batches'];assert r['queries']==sum(int(v) for v in c['queries'][:pos]);assert abs(r['gain']-math.fsum(float(v) for v in c['gains'][:pos]))<1e-12;assert r['chosen_pool_id']==c['chosen'][pos];assert r['queries']<=r['full_queries'];checks['independent_saved_searches']+=1;tr.append(r)
    prior=read(S.O/'oof_trajectories.json');keep=['FACTOR','POWER8_HELPER','STATE4_HELPER','POWER8_GATE','STATE4_GATE'];tr +=[r for r in prior if any(r['method']==f'{a}_seed{s}' for a in keep for s in [0,1,2])]
    summary={name:H.summarize([r for r in tr if r['method']==name]) for name in {r['method'] for r in tr}};assert all(v['cases']==2378 and len(v['per_date'])==16 for v in summary.values());family={a:{k:float(np.mean([summary[f'{a}_seed{s}']['macro'][k] for s in [0,1,2]])) for k in ['mean_net_utility','mean_gain','mean_queries','query_fraction','gain_capture','physical_group_mean_utility']} for a in ['CONSENSUS',*keep]}
    by={(r['method'],r['sample_id']):r for r in tr};datecounts=collections.Counter(c['date'] for c in cases);events=[]
    for seed in [0,1,2]:
        for c in cases:
            a=by[f'CONSENSUS_seed{seed}',c['sample_id']];b=by[f'FACTOR_seed{seed}',c['sample_id']];j=min(a['batches'],b['batches']);direction='UNCHANGED' if a['batches']==b['batches'] else ('NEW_CONTINUE' if a['batches']>b['batches'] else 'NEW_STOP');dg=a['gain']-b['gain'];qs=b['queries']-a['queries'];nd=a['net_utility']-b['net_utility'];assert abs(nd-(dg+fee*qs)/scale)<1e-12
            cat='UNCHANGED' if direction=='UNCHANGED' else ('GAIN_INCREASE_NET_POSITIVE' if dg>0 and nd>0 else ('SAME_GAIN_ACTION_LESS_QUERY' if qs>0 and dg==0 and a['chosen_pool_id']==b['chosen_pool_id'] else ('OTHER_NET_POSITIVE' if nd>0 else 'NET_NEGATIVE')))
            row=dict(seed=seed,sample_id=c['sample_id'],date=c['date'],physical_state_group=c['physical_state_group'],direction=direction,category=cat,gain_delta=dg,saved_queries=qs,net_delta=nd,consensus_batches=a['batches'],factor_batches=b['batches'],same_choice=a['chosen_pool_id']==b['chosen_pool_id'],weight=1/(3*16*datecounts[c['date']]))
            if direction!='UNCHANGED':row.update(first_difference=j,next_gain=float(c['gains'][j]),next_queries=int(c['queries'][j]),next_net=float((c['gains'][j]-fee*c['queries'][j])/scale))
            events.append(row)
    def groups(field):
        out=[]
        for value in sorted({e[field] for e in events}):
            rr=[e for e in events if e[field]==value];out.append(dict(kind=value,pairs=len(rr),starts=len({r['sample_id'] for r in rr}),physical_groups=len({r['physical_state_group'] for r in rr}),positive_pairs=sum(r['net_delta']>0 for r in rr),negative_pairs=sum(r['net_delta']<0 for r in rr),net_contribution=math.fsum(r['weight']*r['net_delta'] for r in rr)))
        return out
    directions=groups('direction');categories=groups('category');pairs=[]
    for seed in [0,1,2]:
        a=summary[f'CONSENSUS_seed{seed}'];b=summary[f'FACTOR_seed{seed}'];pairs.append(dict(seed=seed,utility=a['macro']['mean_net_utility'],factor_utility=b['macro']['mean_net_utility'],delta=a['macro']['mean_net_utility']-b['macro']['mean_net_utility'],relative=a['macro']['mean_net_utility']/b['macro']['mean_net_utility']-1,per_date=[dict(date=x['date'],delta=x['mean_net_utility']-y['mean_net_utility']) for x,y in zip(a['per_date'],b['per_date'])]))
    dates=[dict(date=x['date'],delta=float(np.mean([next(v['delta'] for v in p['per_date'] if v['date']==x['date']) for p in pairs]))) for x in pairs[0]['per_date']];nd=family['CONSENSUS']['mean_net_utility']-family['FACTOR']['mean_net_utility'];assert abs(nd-sum(r['net_contribution'] for r in directions))<1e-12
    key=dict(relative_utility_vs_factor=nd/family['FACTOR']['mean_net_utility'],query_reduction_vs_factor=1-family['CONSENSUS']['mean_queries']/family['FACTOR']['mean_queries'],gain_capture_percentage_points_vs_factor=100*(family['CONSENSUS']['gain_capture']-family['FACTOR']['gain_capture']),positive_seeds=sum(p['delta']>0 for p in pairs),positive_dates=sum(v['delta']>0 for v in dates),positive_seed_dates=sum(v['delta']>0 for p in pairs for v in p['per_date']),disagreement_state_fraction=statecells['DELEGATE_DISAGREEMENT']/sum(statecells.values()),original_stop_reopened_rows=len(blocked),relative_utility_vs_state4=family['CONSENSUS']['mean_net_utility']/family['STATE4_HELPER']['mean_net_utility']-1)
    np.savez_compressed(O/'fixed_decisions.npz',**decisions);save('oof_trajectories.json',tr);save('summary.json',summary);save('family_summary.json',family);save('all7134_paired_events.json',events);save('direction_decomposition.json',directions);save('category_decomposition.json',categories);save('state_cells.json',dict(statecells));save('original_stop_reopened_states.json',blocked);save('paired_summary.json',dict(pairs=pairs,per_date=dates));save('key_changes.json',key);verify();save('review.json',dict(passed=True,checks=dict(checks),all16dates_all3seeds=True,new_fits=0,new_model_forwards=0,new_physics=0,query_increase_explicitly_allowed=True,full_local_budget_bound_only=True,original4dev_not_scored=True,final8_absent=True));print('GRID179_ANALYSED',json.dumps(key,ensure_ascii=False))
def finish():
    verify();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();k=read(O/'key_changes.json');f=read(O/'family_summary.json');p=read(O/'paired_summary.json');di=read(O/'direction_decomposition.json');ca=read(O/'category_decomposition.json')
    lines=['# GRID179：便宜模型共识、分歧委托原模型','',f"这一条固定组合相对原FACTOR的留出局部净效用变化{k['relative_utility_vs_factor']:+.2%}，平均查询变化{-k['query_reduction_vs_factor']:+.2%}，收益保留率变化{k['gain_capture_percentage_points_vs_factor']:+.2f}个百分点；{k['positive_seeds']}/3种子、{k['positive_dates']}/16日期正向。",'',
        '先看STATE4与POWER8的严格继续/停止决定：二者一致时采用共同决定，不一致时委托原FACTOR。这等价于三票多数，但运行时只需在两个便宜模型分歧时运行较贵FACTOR；没有新训练、模型前向、权重或阈值搜索。它是普通共识/委托，不能包装成新的神经网络结构。',
        '', '这条规则同时允许纠正原FACTOR的继续与停止，因此不再具有“查询不增加”的旧证明。多查无利候选、停早漏收益和当前奖励变化都必须保留；仍只评估原候选路径，不超过原完整同站候选预算。它没有改变合法反馈、排序或已观测最佳动作规则。','',
        '| 方案 | 局部净效用 | 平均查询 | 完整同站查询比例 | 收益保留率 | 调用均衡效用 |','|---|---:|---:|---:|---:|---:|']
    for a,v in f.items():lines.append(f"| {a} | {v['mean_net_utility']:.8f} | {v['mean_queries']:.3f} | {v['query_fraction']:.2%} | {v['gain_capture']:.2%} | {v['physical_group_mean_utility']:.8f} |")
    lines+=['','## 新继续与新停止分开看','','| 首次分歧 | 配对数 | 净正/净负 | 物理组 | 净效用贡献 |','|---|---:|---:|---:|---:|']
    for r in di:lines.append(f"| {r['kind']} | {r['pairs']} | {r['positive_pairs']}/{r['negative_pairs']} | {r['physical_groups']} | {r['net_contribution']:+.8f} |")
    lines+=['','| 变化类型 | 配对数 | 物理组 | 净效用贡献 |','|---|---:|---:|---:|']
    for r in ca:lines.append(f"| {r['kind']} | {r['pairs']} | {r['physical_groups']} | {r['net_contribution']:+.8f} |")
    lines+=['','| 种子 | 相对FACTOR净效用变化 |','|---|---:|']
    for r in p['pairs']:lines.append(f"| {r['seed']} | {r['relative']:+.2%} |")
    lines+=['','| 日期 | 三种子平均效用差 |','|---|---:|']
    for r in p['per_date']:lines.append(f"| {r['date']} | {r['delta']:+.8f} |")
    lines+=['',f"10746条留出状态×种子记录中，两个便宜模型分歧占{k['disagreement_state_fraction']:.2%}，{k['original_stop_reopened_rows']}条原STOP因二者都继续被重新打开。这是全部保存状态的比例，不是实际执行路径的运行时间节省；真实模型/特征计算开销未计时。",'',
        '两个小模型共享数据和部分特征，意见一致不等于统计独立或可靠置信度。规则从已暴露开发结果所揭示的问题出发，当前四折也均是历史开发日期，不是新的盲验证。所有3种子、16日期和不利条目均保留，不能只取正向日期解释机制。',
        '', '保留已经重复出现的正向判断，若有改善继续追收益来自哪些实际重新打开的机会；若结果混杂，也与小模型本身的进步分开。不从方向分解临时挑阈值、改票数或追加新组合救结果。无新物理、最终8或论文改动。']
    (O/'GRID179_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');save('decision.json',dict(positive_mean=k['relative_utility_vs_factor']>0,all_seed_win_not_required_for_research=True,adopted=False,ready_to_scale=False,next='Trace the fixedconsensus firstdifferences and compute deployment costs before anynewphysical proposal; no rule scan.'));save('complete.json',dict(passed_audit=True,new_fits=0,new_model_forwards=0,new_physics=0,final8_absent=True));archive('complete')
    collab=R/'论文协作资料/FRAME_便宜模型共识委托_GRID179';collab.mkdir(exist_ok=False)
    for n in ['GRID179_report.md','key_changes.json','family_summary.json','paired_summary.json','direction_decomposition.json','category_decomposition.json','review.json','decision.json','complete.json']:shutil.copy2(O/n,collab/n)
    msg=f"GRID179仅一条固定便宜STATE4/POWER8共识、分歧委托F（普通三票多数），零拟合/前向。对F局部效用{k['relative_utility_vs_factor']:+.2%}、查询变化{-k['query_reduction_vs_factor']:+.2%}、收益保留率{k['gain_capture_percentage_points_vs_factor']:+.2f}pp，{k['positive_seeds']}/3种子、{k['positive_dates']}/16日期正。明确不再承诺查询不增加，仅原完整local预算界；全部正反方向保留，不临时改组合。开发性非盲确认，无新物理/最终8/论文改动。"
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;old=p.read_text(encoding='utf8');entry='\n\n## 2026-10-25 — GRID179便宜共识委托\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a,b=old.split('\n',1);old=a+'\n'+entry+'\n'+b
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 固定共识的机会与开销\n\n1. 保留GRID179全部日期种子的正反方向，核对共同前缀重新打开与提前停止的实际收益和查询代价，不从结果改票数/阈值。\n2. 若继续工程核查，必须测当前两个便宜模型共识后委托F的实际调用成本，不能沿用旧171时间或旧查询不增加证明。\n3. 新物理/数据扩大先给冻结候选、对照与范围等待决定，最终8和论文不动。小模型输入与组合成绩分别报告。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes(f'# 最新：GRID179便宜共识委托\r\n\r\n[报告](protocols/GRID179_cheap_consensus/complete/GRID179_report.md)；固定规则对F效用{k["relative_utility_vs_factor"]:+.2%}，查询与收益分列。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID179_COMPLETE',msg)
if __name__=='__main__':globals()[sys.argv[1]]()
