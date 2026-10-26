"""Read-only attribution of existing gate differences; no fitting or policy search."""
from pathlib import Path
import collections,hashlib,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究'); W=Path(__file__).parent; O=R/'outputs/grid194_gate_case_attribution'; REPO=R/'work/GRL-results-integration'; A=REPO/'research_archive/power-grid-2026/protocols/GRID194_gate_case_attribution'
DIRS={190:R/'outputs/grid190_extension_gate',192:R/'outputs/grid192_gate_support',193:R/'outputs/grid193_gate_precondition'}
BLOCKS={'main_hidden':(0,64),'relation_hidden':(64,80),'NET_scores':(80,82),'query_charge':(82,83)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    dest=A/phase; dest.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
def prepare():
    assert not O.exists();O.mkdir(parents=True)
    save('design.json',dict(id='GRID194_READONLY_GATE_ATTRIBUTION',authorization='Continue existing learning mechanism analysis. No new fits, NET forwards or physics.',
        scope='All 4681 eligible held gate decisions across16 exposed historical training dates/3seeds. Compare 190/192/193 without filtering convergence, seeds, dates or outcomes.',
        categories='Bothappend/bothskip/new-only productive/new-only unchanged/dropped-old productive/dropped-old unchanged. Productive denotes changed local selected action, not positive NET or whole-week benefit.',
        attribution='Reconstruct old and new logits from identical cached83features. Decompose difference into fixed groups mainhidden64/relationhidden16/NETscores2/knowncharge1/intercept. Signed linear accounting, not causal or independent importance.',
        summaries='Raw seed-prefix pair counts plus unique starts, physical groups,dates; physical-group means over pairs and medians. State changes from original support experiment192 versus numerical control193 also counted. No thresholds learned or alternative strategy constructed.',
        label_boundary='Future next gain/NET only outcomes for descriptive strata; never model input or decision. No posthoc ensemble, fits, hyperparameter selection, new data, final8 or manuscript.'))
    files=[W/'run.py',O/'design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json']
    for d in DIRS.values():
        assert read(d/'complete.json')['passed']
        files += [d/'complete.json',d/'gate_predictions.json',d/'oof_trajectories.json',d/'family_summary.json',d/'fits.json']+list(d.glob('held_*.npz'))+list((d/'models').glob('*.npz'))
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files}); verify(); archive('planned'); print('GRID194_PLANNED')
def evaluate():
    verify();assert not (O/'complete.json').exists()
    predictions={k:read(d/'gate_predictions.json') for k,d in DIRS.items()}
    ix={k:{(v['seed'],v['sample_id']):v for v in rows} for k,rows in predictions.items()}
    assert all(set(m)==set(ix[190]) for m in ix.values()); assert len(ix[190])==4681
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];cm={c['sample_id']:c for c in cases}; nc=collections.Counter(c['date'] for c in cases)
    task=read(R/'outputs/grid102_search_rl/design.json'); fee,scale=task['cost_per_query'],task['gain_scale']
    rows=[]; maxerr=0.
    for f in range(4):
        for s in range(3):
            old=np.load(DIRS[190]/f'held_f{f}_s{s}.npz');new=np.load(DIRS[193]/f'held_f{f}_s{s}.npz'); a=np.load(DIRS[190]/'models'/f'f{f}_s{s}.npz');b=np.load(DIRS[193]/'models'/f'f{f}_s{s}.npz')
            assert np.array_equal(old['ids'],new['ids']) and np.array_equal(old['features'],new['features'])
            assert np.array_equal(a['mean'],b['mean']) and np.array_equal(a['scale'],b['scale'])
            ps=[p for p in predictions[190] if p['fold']==f and p['seed']==s]; assert len(ps)==len(old['ids'])
            z=(old['features']-a['mean'])/a['scale']; db=b['coef'][0]-a['coef'][0]; bias=float(b['intercept'][0]-a['intercept'][0])
            lo=z@a['coef'][0]+a['intercept'][0];ln=z@b['coef'][0]+b['intercept'][0]
            assert np.allclose(lo,old['logit'],atol=1e-8,rtol=1e-9) and np.allclose(ln,new['logit'],atol=1e-8,rtol=1e-9)
            for i,p in enumerate(ps):
                key=(s,p['sample_id']); q=ix[193][key]; m=ix[192][key]; c=cm[p['sample_id']]
                assert p['position']==q['position']==m['position']
                assert p['actual_next_gain']==q['actual_next_gain']==m['actual_next_gain']
                assert bool(lo[i]>0)==p['gate_continue'] and bool(ln[i]>0)==q['gate_continue']
                if p['gate_continue']==q['gate_continue']:cat='BOTH_APPEND' if q['gate_continue'] else 'BOTH_SKIP'
                elif q['gate_continue']:cat='NEW_ONLY_PRODUCTIVE' if p['actual_next_change'] else 'NEW_ONLY_NO_CHANGE'
                else:cat='DROP_OLD_PRODUCTIVE' if p['actual_next_change'] else 'DROP_OLD_NO_CHANGE'
                con={k:float(z[i,j:k0]@db[j:k0]) for k,(j,k0) in BLOCKS.items()};con['intercept']=bias;err=abs(sum(con.values())-(ln[i]-lo[i]));maxerr=max(maxerr,err);assert err<1e-7
                x=old['features'][i]; cost=float(x[82]); assert np.isclose(cost,fee*p['actual_next_queries']/scale,rtol=2e-6,atol=1e-8)
                rows.append(dict(fold=f,seed=s,sample_id=p['sample_id'],physical_state_group=p['physical_state_group'],date=p['date'],category=cat,old_logit=float(lo[i]),new_logit=float(ln[i]),logit_delta=con,net_continue_margin=float(x[81]-x[80]),known_query_charge=cost,queries=p['actual_next_queries'],next_gain=p['actual_next_gain'],next_NET=p['actual_next_NET'],positive_NET=p['actual_next_NET']>0,old_continue=p['gate_continue'],new_continue=q['gate_continue'],numerical_change_vs192=m['gate_continue']!=q['gate_continue'],macro_mass=1/(3*16*nc[c['date']])) )
    categories={}
    for cat in sorted({r['category'] for r in rows}):
        rr=[r for r in rows if r['category']==cat]; groups=collections.defaultdict(list)
        for r in rr:groups[r['physical_state_group']].append(r)
        fields=['net_continue_margin','known_query_charge','queries','next_gain','next_NET']
        summary={k:dict(pair_median=float(np.median([r[k] for r in rr])),physical_group_mean=float(np.mean([np.mean([r[k] for r in gg]) for gg in groups.values()]))) for k in fields}
        contrib={k:float(np.mean([np.mean([r['logit_delta'][k] for r in gg]) for gg in groups.values()])) for k in [*BLOCKS,'intercept']}
        categories[cat]=dict(pairs=len(rr),unique_starts=len({r['sample_id'] for r in rr}),physical_groups=len(groups),dates=len({r['date'] for r in rr}),positive_NET_pairs=sum(r['positive_NET'] for r in rr),summary=summary,group_mean_signed_logit_change=contrib,
            query_contribution=float(sum(r['macro_mass']*r['queries']*(int(r['new_continue'])-int(r['old_continue'])) for r in rr)),gain_contribution=float(sum(r['macro_mass']*r['next_gain']*(int(r['new_continue'])-int(r['old_continue'])) for r in rr)))
    novel=[r for r in rows if r['category']=='NEW_ONLY_PRODUCTIVE']; lost=[r for r in rows if r['category']=='DROP_OLD_PRODUCTIVE']; stable=[r for r in novel if not r['numerical_change_vs192']]
    switch=[r for r in rows if r['numerical_change_vs192']]
    result=dict(categories=categories,novel22_still_novel_without_numerical_change=len(stable),numerical_control_switches=len(switch),numerical_control_switches_by_pool=dict(collections.Counter(f"{r['fold']}/{r['seed']}" for r in switch)),max_logit_decomposition_error=maxerr)
    save('case_rows.json',rows);save('summary.json',result)
    q=categories['NEW_ONLY_PRODUCTIVE']; l=categories['DROP_OLD_PRODUCTIVE']
    loss_ratio=abs(l['gain_contribution'])/q['gain_contribution']
    lines=['# GRID194：门控互补案例和丢失收益的只读核查','',
        f"GRID193相对GRID190新找回{q['pairs']}个种子—起点配对、来自{q['physical_groups']}个物理调用组；同时放弃{l['pairs']}个原有效追加、来自{l['physical_groups']}个调用组。按相同日期宏平均权重，后者丢失的奖励约为前者找回奖励的{loss_ratio:.2f}倍。新增好例保留，但目前不能把它们直接叠加成净改善。",'',
        '## 先把样本口径说清楚','',
        '22不是22个独立场景；它是三个模型种子、多个前缀起点的配对数。同一物理调用可能有多个起点，不作为独立重复证据。所有4681个可追加停止点都纳入，未筛日期、种子或收敛情况。改选意味着局部动作变化；收益是否抵得上查询费用另列。','',
        '|类别|种子—起点配对|不同起点|物理调用组|日期数|下一批NET为正的配对|','|---|---:|---:|---:|---:|---:|']
    labels=dict(BOTH_APPEND='双方追加',BOTH_SKIP='双方跳过',NEW_ONLY_PRODUCTIVE='新找回有效追加',NEW_ONLY_NO_CHANGE='新增但无改选',DROP_OLD_PRODUCTIVE='丢失旧有效追加',DROP_OLD_NO_CHANGE='省去旧无效追加')
    for cat,v in categories.items():lines.append(f"|{labels[cat]}|{v['pairs']}|{v['unique_starts']}|{v['physical_groups']}|{v['dates']}|{v['positive_NET_pairs']}|")
    lines+=['','## 两类有效追加的区别','',
        '|类别|原NET继续评分差（中位数）|已知下一批代理费用（中位数）|下一批奖励增量（物理组均值）|下一批NET（物理组均值）|','|---|---:|---:|---:|---:|']
    for cat in ['NEW_ONLY_PRODUCTIVE','DROP_OLD_PRODUCTIVE']:
        v=categories[cat]['summary'];lines.append(f"|{labels[cat]}|{v['net_continue_margin']['pair_median']:.6f}|{v['known_query_charge']['pair_median']:.6f}|{v['next_gain']['physical_group_mean']:.9f}|{v['next_NET']['physical_group_mean']:.6f}|")
    lines+=['','原NET继续评分差和已知查询费用在决定前可见；未来奖励及NET只作核查标签。这里没有由分组结果拟定新阈值，也没有证明这些特征能在新日期预测收益。','',
        '## 决策为什么翻转','',
        '对每个停止点，精确重建两个门控的logit，再将差值分成预先规定的五部分。以下各列先对每个物理组内部取均值，再对组取均值；总和等于对应组均值logit变化。它是同一输入上的线性记账，不是因果归因：两组隐表示、NET评分及费用有相关性，不能据此删某个模块。','',
        '|类别|主干隐表示64维|关系隐表示16维|NET两项评分|已知费用|截距|','|---|---:|---:|---:|---:|---:|']
    for cat in ['NEW_ONLY_PRODUCTIVE','NEW_ONLY_NO_CHANGE','DROP_OLD_PRODUCTIVE','DROP_OLD_NO_CHANGE']:
        v=categories[cat]['group_mean_signed_logit_change'];lines.append('|'+labels[cat]+'|'+ '|'.join(f'{v[k]:+.6f}' for k in [*BLOCKS,'intercept'])+'|')
    lines+=['',f"12池原始83维输入、顺序和标准化完全相同；两门控的全部4681分数逐项重现，五部分分解最大误差{maxerr:.3g}。门控参数变了，原NET本身的评分与查询过程没有变。",'',
        '## 排除数值优化造成的假象','',
        f"GRID192与数值对照GRID193之间，共有{len(switch)}个停止点的门控动作翻转。上述{len(novel)}个新找回配对中，{len(stable)}个在数值对照前已经被GRID192追加。完整翻转明细与所属折/种子保存在case_rows.json和summary.json；未把那份仍达迭代上限的模型剔除。",'',
        '## 目前能支持什么','',
        '需要解决的不是单纯让模型多补查：混合覆盖既省掉旧无效追加，又错过旧有效追加，还增加新的无效追加。评价下一项学习改动时，要同时看找回奖励、丢失奖励与查询费用，不能只数改选次数。','',
        '已有标签把极小改选和大幅收益改选都标作1，当前两个门控都未充分体现机会的价值差异。下一步可先在拟合侧审查高收益样本的覆盖、集中程度，以及是否被同一日期/物理调用的多个前缀放大，再确定是否需要收益幅度监督。本轮不据此训练新模型，不自动组合两个门控。','',
        '本轮零拟合、零神经网络前向、零电网预测、零新数据。全部未来反馈仅用于事后解释；不是独立测试、闭环整周质量或实际时间结论。']
    (O/'GRID194_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    old=read(DIRS[190]/'family_summary.json')['GATE'];new=read(DIRS[193]/'family_summary.json')['GATE']
    assert abs(sum(v['gain_contribution'] for v in categories.values())-(new['mean_gain']-old['mean_gain']))<1e-12
    assert abs(sum(v['query_contribution'] for v in categories.values())-(new['mean_queries']-old['mean_queries']))<1e-10
    verify();save('review.json',dict(passed=True,all4681_scores_reconstructed=True,all12_cached_features_and_normalizers_identical=True,logit_group_sums_and_query_gain_accounting_pass=True,descriptive_not_causal=True,new_fits=0,new_NET_forwards=0,new_physics=0))
    save('complete.json',dict(passed=True,all_seeds_dates_and_counterexamples_retained=True,new_fits=0,new_NET_forwards=0,new_physics=0,no_new_decision_rule=True));archive('complete')
    dest=R/'论文协作资料/FRAME_门控案例归因_GRID194';dest.mkdir(exist_ok=False)
    for name in ['GRID194_report.md','summary.json','review.json','complete.json']:shutil.copy2(O/name,dest/name);assert sha(O/name)==sha(dest/name)
    msg=f'GRID194只读4681门控分解完成：新找回{len(novel)}配对/{q["physical_groups"]}组，但放弃{len(lost)}旧有效配对/{l["physical_groups"]}组，丢失宏平均奖励为找回{loss_ratio:.2f}倍。{len(stable)}/{len(novel)}新例在数值对照前已存在；192→193翻转{len(switch)}。83维输入同、五块logit守恒，非因果重要性；保留190与新例，不自动组合/加阈值/训练。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID194门控互补与损失核查\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':h,t0=t.split('\n',1);t=h+'\n'+entry+'\n'+t0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID194后\n\n1. 保留GRID190现有候选，GRID192/193互补小改善和全部反例保留，不按事后好例自动组合。\n2. 优先只读核查收益幅度监督所需拟合标签：高收益机会在日期/调用组/前缀上的集中和覆盖，区分标签幅度问题与跨日期迁移。\n3. 形成可检验的最小单变量训练协议后先通知；不扫描阈值、择种子、扩日期/物理/网络或最终8，不改论文。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID194门控互补与损失核查\n\n[报告](protocols/GRID194_gate_case_attribution/complete/GRID194_report.md)；只读重建全部门控差异与logit贡献，保留GRID190及互补好例，不自动训练或组合。\n\n'.encode('utf8')+p.read_bytes())
    print('GRID194_COMPLETE',json.dumps(result,ensure_ascii=False))
if __name__=='__main__':globals()[sys.argv[1]]()
