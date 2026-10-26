"""Read-only attribution of GRID196 gain improvements, wasted queries and transfer."""
from pathlib import Path
import collections,hashlib,json,pickle,shutil,sys
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid197_gain_opportunity_audit';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID197_gain_opportunity_audit';NEW=R/'outputs/grid196_gain_readout';OLD=R/'outputs/grid190_extension_gate';FIT=R/'outputs/grid192_gate_support'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8',newline='\n')
def verify():
    for p,h in read(O/'frozen_inputs.json').items():assert sha(R/p)==h,p
def prepare():
    assert not O.exists();O.mkdir(parents=True);assert read(NEW/'complete.json')['passed']
    save('design.json',dict(id='GRID197_READONLY_GAIN_OPPORTUNITIES',authorization='User继续吧; interpret existing gain head before any further fit.',
        scope='All7134seed-start trajectories/4681eligibleSTOPs across16existingdates. Compare196vs190;no subsetselection forheadline.',
        accounting='Bothappend/bothskip/new-onlychanged/new-onlyzero/droppedoldchanged/droppedoldzero, macrodate/seed/start weights. Allphysicalcall gain/query/net contributions saved,1/2/3seed repetition spectrum retained,not requireallwin.',
        matched='Use alreadyfrozen196expected-querycomparison probabilities. Attribute exactexpected reward andquery differences to everyphysicalcall;sum must reproduce all48date-seed and three seedmacros.',
        transfer='Read out fixed196 coefficients on existingfitSTOP,fitCONT andheldSTOP cached83features. Date/call/start balanced bias,normalizedMSE,appendrate,zero-gainquery fraction. No model updates/NNforwards. Fit83coordinate minmax envelope fixed perpool;outside indicator onlydescriptor,not OODproof or decisionrule.',
        labels='All actualgain/NET andfuturezero labels are only descriptive outcomes;no futureonlinefeatures. No new bins/thresholds,rule,ensemble,training,physics,data,final8 or paper.',
        limits='Sharedteacherpaths/seeds/prefixes not independent scenarios. Localgain andproxyfee differ fromfullweek quality andreal latency. Preserve allcounterexamples.'))
    files=[W/'run.py',O/'design.json',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid113_relational_learning/train_metadata.json',OLD/'gate_predictions.json',OLD/'oof_trajectories.json',OLD/'family_summary.json',NEW/'gate_predictions.json',NEW/'oof_trajectories.json',NEW/'family_summary.json',NEW/'expected_budget_comparison.json',NEW/'complete.json',NEW/'fits.json']+list((NEW/'models').glob('*.npz'))+list((NEW/'fit_labels').glob('*.npz'))+list((FIT/'fit_features').glob('*.npz'))+list(OLD.glob('held_f*.npz'))
    save('frozen_inputs.json',{p.relative_to(R).as_posix():sha(p) for p in files});verify();archive('planned');print('GRID197_PLANNED')
def hierarchy(rows):
    ds={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set);npref=collections.Counter(r['sample_id'] for r in rows)
    for r in rows:gd[r['date']].add(r['group']);gs[r['group']].add(r['sample_id'])
    w=np.array([1/(len(ds)*len(gd[r['date']])*len(gs[r['group']])*npref[r['sample_id']]) for r in rows]);assert abs(w.sum()-1)<1e-10;return w
def profile(value,actual,charge,queries,w):
    guess=np.maximum(value,0);selected=guess>charge;zero=actual==0;spent=float(w@(queries*selected))
    return dict(rows=len(value),mean_predicted_normalized_gain=float(w@guess),mean_actual_normalized_gain=float(w@actual),mean_prediction_bias=float(w@(guess-actual)),normalized_MSE=float(w@((guess-actual)**2)),append_mass=float(w@selected),query_mass=spent,zero_gain_query_fraction=float(w@(queries*selected*zero)/spent) if spent else 0.,selected_gain=float(w@(actual*selected)),available_gain=float(w@actual),gain_capture_at_gate=float(w@(actual*selected)/(w@actual)) if w@actual>0 else 0.)
def evaluate():
    verify();assert not (O/'complete.json').exists();meta=read(R/'outputs/grid113_relational_learning/train_metadata.json');cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='train'];cm={c['sample_id']:c for c in cases};nc=collections.Counter(c['date'] for c in cases);task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];new=read(NEW/'oof_trajectories.json');old=read(OLD/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in new};oi={(r['method'],r['sample_id']):r for r in old};pred=read(NEW/'gate_predictions.json');pi={(r['seed'],r['sample_id']):r for r in pred};match=read(NEW/'expected_budget_comparison.json')['GRID190'];mi={(r['seed'],r['date']):r for r in match['per_date_seed']};categories=collections.defaultdict(list);call=collections.defaultdict(lambda:dict(actual_gain_delta=0.,actual_query_delta=0.,matched_gain_delta=0.,matched_query_delta=0.,matched_NET_delta=0.,rows=0));pairs=[]
    for s in range(3):
        for c in cases:
            sid=c['sample_id'];a=oi[f'GATE_seed{s}',sid];b=ni[f'GATE_seed{s}',sid];base=ni[f'NET_seed{s}',sid];assert a['baseline_stop']==b['baseline_stop'];key=('BOTH_APPEND' if b['append'] else 'BOTH_SKIP') if a['append']==b['append'] else ('NEW_ONLY_CHANGED' if b['chosen_pool_id']!=base['chosen_pool_id'] else 'NEW_ONLY_ZERO') if b['append'] else ('DROP_OLD_CHANGED' if a['chosen_pool_id']!=base['chosen_pool_id'] else 'DROP_OLD_ZERO')
            if a['append']==b['append']:assert a['gain']==b['gain'] and a['queries']==b['queries']
            mass=1/(3*16*nc[c['date']]);mr=mi[s,c['date']];pa,pb=mr['OLD']['keep_probability'],mr['GAIN']['keep_probability'];ga=a['gain']-base['gain'];gb=b['gain']-base['gain'];qa=a['queries']-base['queries'];qb=b['queries']-base['queries'];mg=mass*(pb*gb-pa*ga);mq=mass*(pb*qb-pa*qa)
            r=dict(seed=s,sample_id=sid,group=c['physical_state_group'],date=c['date'],category=key,macro_mass=mass,actual_gain_delta=mass*(b['gain']-a['gain']),actual_query_delta=mass*(b['queries']-a['queries']),matched_gain_delta=mg,matched_query_delta=mq,matched_NET_delta=(mg-fee*mq)/scale,old_append=a['append'],new_append=b['append'])
            if b['eligible']:
                p=pi[s,sid];r.update(next_gain=p['actual_next_gain'],next_queries=p['actual_next_queries'],next_NET=p['actual_next_NET'],predicted_normalized_gain=p['predicted_normalized_gain'],predicted_NET=p['predicted_net'])
            pairs.append(r);categories[key].append(r);cc=call[s,c['physical_state_group']];cc['rows']+=1
            for k in ['actual_gain_delta','actual_query_delta','matched_gain_delta','matched_query_delta','matched_NET_delta']:cc[k]+=r[k]
    summarized={k:dict(pairs=len(rr),starts=len({r['sample_id'] for r in rr}),calls=len({r['group'] for r in rr}),dates=len({r['date'] for r in rr}),actual_gain_delta=float(sum(r['actual_gain_delta'] for r in rr)),actual_query_delta=float(sum(r['actual_query_delta'] for r in rr)),matched_gain_delta=float(sum(r['matched_gain_delta'] for r in rr)),matched_query_delta=float(sum(r['matched_query_delta'] for r in rr)),next_positive_NET_pairs=sum(r.get('next_NET',0)>0 for r in rr)) for k,rr in categories.items()}
    calls=[dict(seed=s,group=g,date=g.split('_')[0],**v) for (s,g),v in sorted(call.items())];groups=sorted({r['group'] for r in calls});repetition=[]
    for group in groups:
        rr=[r for r in calls if r['group']==group];assert len(rr)==3;repetition.append(dict(group=group,date=rr[0]['date'],actual_positive_seeds=sum(r['actual_gain_delta']>0 for r in rr),actual_negative_seeds=sum(r['actual_gain_delta']<0 for r in rr),matched_positive_NET_seeds=sum(r['matched_NET_delta']>0 for r in rr),matched_negative_NET_seeds=sum(r['matched_NET_delta']<0 for r in rr),total_actual_gain_delta=sum(r['actual_gain_delta'] for r in rr),total_matched_NET_delta=sum(r['matched_NET_delta'] for r in rr),seed_details=rr))
    profiles=[];envelopes=[]
    for f in range(4):
        for s in range(3):
            z=np.load(FIT/'fit_features'/f'f{f}_s{s}.npz');lab=np.load(NEW/'fit_labels'/f'f{f}_s{s}.npz');cp=np.load(NEW/'models'/f'f{f}_s{s}.npz');x=z['standard'];value=x@cp['coef']+cp['intercept'][0];n=int(z['original_STOP_count']);w=z['mass'];yy=lab['target'];charge=lab['known_charge'];queries=lab['queries'];stats={}
            for domain,idx in [('fit_STOP',np.arange(n)),('fit_CONT',np.arange(n,len(value)))]:ww=w[idx]/w[idx].sum();stats[domain]=profile(value[idx],yy[idx],charge[idx],queries[idx],ww)
            hz=np.load(OLD/f'held_f{f}_s{s}.npz');hx=(hz['features']-cp['mean'])/cp['scale'];hv=hx@cp['coef']+cp['intercept'][0];pp=[r for r in pred if r['fold']==f and r['seed']==s];hr=[dict(date=r['date'],group=r['physical_state_group'],sample_id=r['sample_id']) for r in pp];hw=hierarchy(hr);assert len(pp)==len(hv)
            actual=np.array([r['actual_next_gain']/scale for r in pp]);hcharge=np.array([r['known_charge'] for r in pp]);hq=np.array([r['actual_next_queries'] for r in pp]);assert np.allclose(np.maximum(hv,0),[r['predicted_normalized_gain'] for r in pp],atol=1e-10,rtol=1e-10);stats['held_STOP']=profile(hv,actual,hcharge,hq,hw);profiles.append(dict(fold=f,seed=s,domains=stats))
            outside=(hx<x.min(axis=0))|(hx>x.max(axis=0));anyout=outside.any(axis=1)
            for i,r in enumerate(pp):envelopes.append(dict(seed=s,sample_id=r['sample_id'],fold=f,group=r['physical_state_group'],date=r['date'],outside_any_fit_coordinate=bool(anyout[i]),outside_coordinate_count=int(outside[i].sum()),append=r['gate_continue'],gain=r['actual_next_gain'],NET=r['actual_next_NET']))
    oldf=read(OLD/'family_summary.json')['GATE'];newf=read(NEW/'family_summary.json')['GATE'];dg=newf['mean_gain']-oldf['mean_gain'];dq=newf['mean_queries']-oldf['mean_queries'];assert abs(sum(r['actual_gain_delta'] for r in pairs)-dg)<1e-12 and abs(sum(r['actual_query_delta'] for r in pairs)-dq)<1e-10
    for s in range(3):
        rr=[r for r in pairs if r['seed']==s];expected=next(r for r in match['per_seed'] if r['seed']==s);assert abs(3*sum(r['matched_NET_delta'] for r in rr)-expected['NET_delta'])<1e-10;assert abs(sum(r['matched_query_delta'] for r in rr))<1e-10
    meanprofile={domain:{key:float(np.mean([p['domains'][domain][key] for p in profiles])) for key in profiles[0]['domains'][domain] if key!='rows'} for domain in ['fit_STOP','fit_CONT','held_STOP']}
    spec={k:sum(r['actual_positive_seeds']==k for r in repetition) for k in range(4)};mspec={k:sum(r['matched_positive_NET_seeds']==k for r in repetition) for k in range(4)}
    summary=dict(category_accounting=summarized,actual_positive_seed_repetition_spectrum=spec,matched_positive_NET_seed_repetition_spectrum=mspec,physical_call_count=len(groups),mean_profiles=meanprofile,all_actual_reward_delta=dg,all_actual_query_delta=dq,all_matched_NET_delta=sum(r['matched_NET_delta'] for r in pairs))
    save('all_pair_accounting.json',pairs);save('all_call_contributions.json',calls);save('all_call_repetition.json',repetition);save('fixed_readout_transfer.json',profiles);save('fit_coordinate_envelopes.json',envelopes);save('summary.json',summary)
    report(summary,repetition,calls);verify();save('review.json',dict(passed=True,all7134pairs_and_222calls_perseed=True,actual_and_expected_contributions_conserve=True,all4681held_readouts_reproduced=True,fit_only_coordinate_bounds=True,repetition_is_shared_seed_evidence_not_independent_scenarios=True,new_fits=0,new_NET_forwards=0,new_physics=0));save('complete.json',dict(passed=True,new_fits=0,new_NET_forwards=0,new_physics=0,no_new_rule_or_threshold=True,all_dates_seeds_counterexamples_kept=True));archive('complete')
    dest=R/'论文协作资料/FRAME_收益改善与浪费_GRID197';dest.mkdir(exist_ok=False)
    for n in ['GRID197_report.md','summary.json','all_call_repetition.json','fixed_readout_transfer.json','review.json','complete.json']:shutil.copy2(O/n,dest/n);assert sha(O/n)==sha(dest/n)
    update_docs(summary);print('GRID197_COMPLETE',json.dumps(summary,ensure_ascii=False))
def report(summary,repetition,calls):
    cats=summary['category_accounting'];profile=summary['mean_profiles'];labels=dict(BOTH_APPEND='双方追加',BOTH_SKIP='双方跳过',NEW_ONLY_CHANGED='本轮新增且改选',NEW_ONLY_ZERO='本轮新增但零收益',DROP_OLD_CHANGED='丢失旧有效追加',DROP_OLD_ZERO='省去旧零收益追加')
    lines=['# GRID197：收益读出找回了什么，查询又花在哪里','',
        '本轮只重读GRID196与GRID190的全部保存记录，不训练、不调阈值。关注新收益能否在不同模型种子下重复出现，同时解释新增的零收益查询及原有效追加的丢失。','',
        '## 全量新旧差异','',
        '|类别|种子—起点配对|物理调用组|日期数|局部奖励变化贡献|候选查询变化贡献|下一批NET为正的配对|','|---|---:|---:|---:|---:|---:|---:|']
    for k,v in sorted(cats.items()):lines.append(f"|{labels[k]}|{v['pairs']}|{v['calls']}|{v['dates']}|{v['actual_gain_delta']:+.9f}|{v['actual_query_delta']:+.6f}|{v['next_positive_NET_pairs']}|")
    lines+=['','贡献按16日期、三个种子、每日期全部起点的原报告口径计算；表内求和逐项复现GRID196相对GRID190的奖励与查询变化。没有将未完成查询算成时间节省，也没有排除任何反向配对。','',
        '## 有多少改善能重复出现','',
        '|同一物理调用上出现正改善的种子数|实际奖励改善的调用数|同预期预算NET改善的调用数|','|---|---:|---:|']
    for k in range(4):lines.append(f"|{k}|{summary['actual_positive_seed_repetition_spectrum'][k]}|{summary['matched_positive_NET_seed_repetition_spectrum'][k]}|")
    lines+=['','共222个物理调用。三个种子共用相同教师数据，重复改善可作为机制线索，不等于三个独立场景验证；只在一个或两个种子中出现的改善也完整保留，不设置全种子获胜门槛。','',
        '同预期预算使用GRID196训练前固定的逐日期、种子稀释概率。每个调用的奖励和查询分配可能变化，但同一日期、种子的总预期查询相同；所有调用贡献之和精确复现原三个种子的NET差。','',
        '## 冻结读出在训练与留出停止点的表现','',
        '|数据域|平均预测奖励（归一化）|实际平均奖励（归一化）|预测偏差|追加比例|追加查询中零奖励占比|','|---|---:|---:|---:|---:|---:|']
    for k,label in [('fit_STOP','拟合侧原NET停止'),('fit_CONT','拟合侧原NET继续'),('held_STOP','留出侧原NET停止')]:
        p=profile[k];lines.append(f"|{label}|{p['mean_predicted_normalized_gain']:.6f}|{p['mean_actual_normalized_gain']:.6f}|{p['mean_prediction_bias']:+.6f}|{100*p['append_mass']:.2f}%|{100*p['zero_gain_query_fraction']:.2f}%|")
    lines+=['','这是12个池的描述均值：每个域内沿用日期/物理调用/起点/位置均衡，再对池取均值，非12个独立样本。拟合与留出人群不同；训练拟合好并不保证部署条件下的均值预测准确。没有为了消除偏差而重新校准预测或调整查询费用。','',
        '每个拟合池还冻结了83维输入的逐坐标最小/最大值，并保存全部留出停止点是否越界。这个描述不能证明不可学习或分布外因果关系，也没有用于过滤结果、设置停止规则或择取模型。','',
        '## 三个种子的正负来源','',
        '下表列出各种子同预期预算NET贡献绝对值最大的三个正向及三个反向调用，方便逐案例追踪；这是解释性展示，主结果仍包括所有调用，完整明细见all_call_contributions.json。','',
        '|种子|方向|物理调用|同预期预算NET贡献|实际奖励贡献|实际查询贡献|','|---|---|---|---:|---:|---:|']
    for s in range(3):
        rr=[r for r in calls if r['seed']==s]
        for sign in [1,-1]:
            selected=sorted([r for r in rr if sign*r['matched_NET_delta']>0],key=lambda r:sign*r['matched_NET_delta'],reverse=True)[:3]
            for r in selected:lines.append(f"|{s}|{'正' if sign>0 else '反'}|{r['group']}|{3*r['matched_NET_delta']:+.8f}|{3*r['actual_gain_delta']:+.9f}|{3*r['actual_query_delta']:+.6f}|")
    lines+=['','上表乘三后为相应种子的日期宏平均贡献；不把某个调用的事后收益当作下一轮可直接使用的触发标签。','',
        '## 下一步的依据','',
        '保留GRID196的新奖励机会，尤其是跨种子重复的调用；同时核查零收益追加的高估是否在拟合STOP域已经存在，还是主要出现在留出STOP域。两者决定应优先处理输出目标/稀疏收益建模，还是训练支持与跨日期迁移，不能只靠观察总体奖励上升作选择。','',
        '下一项学习若拆分“有无收益”和“有收益时多大”，必须分别学习对应条件量；不能将已有无条件奖励预测直接乘一个改选概率，造成重复计入事件概率。当前没有拟合这种模型，也不按事后好例合并旧门控。','',
        '本轮没有新增神经网络前向、电网预测、数据、最终测试、论文或图表。局部奖励和代理查询效用仍不能代替整周成本与实际时延。']
    (O/'GRID197_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
def update_docs(summary):
    sp=summary['actual_positive_seed_repetition_spectrum'];mp=summary['matched_positive_NET_seed_repetition_spectrum'];held=summary['mean_profiles']['held_STOP'];msg=f'GRID197只读7134配对/4681停止点完成，实际奖励正向调用按1/2/3种子重复为{sp[1]}/{sp[2]}/{sp[3]}，同预期NET为{mp[1]}/{mp[2]}/{mp[3]}，全222调用及反例保留。留出STOP预测/实际归一化均值{held["mean_predicted_normalized_gain"]:.5f}/{held["mean_actual_normalized_gain"]:.5f}，零收益查询占追加{100*held["zero_gain_query_fraction"]:.2f}%（12池描述均值）；贡献守恒、原系数回读复现。保留196奖励线索和190对照，不自动阈值、校准、组合、拟合或物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID197收益机会与零收益查询核查\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,r=t.split('\n',1);t=h+'\n'+entry+'\n'+r
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID197后\n\n1. 保留GRID196跨种子重复的新奖励机会与GRID190对照，不以总体同预算尚未占优而删除小改善。\n2. 根据fitSTOP/fitCONT/heldSTOP校准差，研究稀疏收益的事件与条件幅度能否分开识别；先资格化标签和数据域，不能把无条件收益再乘事件概率。\n3. 新训练前通知固定范围，不按事后调用拟规则、择种子或扫描阈值/费用；不自动扩物理、日期、最终8或论文。\n',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID197收益改善与无效查询核查\n\n[报告](protocols/GRID197_gain_opportunity_audit/complete/GRID197_report.md)；全量调用贡献和种子重复谱保留，固定回归输出校准核查，零拟合。\n\n'.encode('utf8')+p.read_bytes())
if __name__=='__main__':globals()[sys.argv[1]]()
