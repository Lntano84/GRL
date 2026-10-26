"""Audited internal learning comparisons; no new FRAME rollout."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def hierarchy(rows):
    ds={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set)
    for r in rows:gd[r['date']].add(r['physical_state_group']);gs[r['physical_state_group']].add(r['sample_id'])
    w=np.array([1/(len(ds)*len(gd[r['date']])*len(gs[r['physical_state_group']])) for r in rows]);assert abs(w.sum()-1)<1e-10;return w
def main():
    C.trained();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists();pred=read(O/'gate_predictions.json');oldp={(r['seed'],r['sample_id']):r for r in read(C.LINEAR/'gate_predictions.json')};task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];predictive=[];audits=[]
    for rec in read(O/'fits.json')['records']:
        f,s=rec['fold'],rec['seed'];name=f'f{f}_s{s}';cp=np.load(O/'models'/f'{name}.npz');source=np.load(C.EVENT/'models'/f'{name}.npz');assert np.array_equal(cp['event_coef'],source['coef'][0]) and np.array_equal(cp['event_intercept'],source['intercept']);lab=np.load(O/'fit_labels'/f'{name}.npz');z=np.load(C.FIT/'fit_features'/f'{name}.npz');x=z['standard'][lab['indices']];y,w=lab['target'],lab['sample_weight'];model=pickle.loads((O/'models'/f'{name}.pkl').read_bytes());assert model.get_params()['random_state']==s
        for k,v in C.PARAMS.items():assert model.get_params()[k]==v
        worst=0.;nodes=0;leaves=0
        for tree in model.estimators_:
            idx=tree.apply(x);nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1));assert tree.tree_.max_depth<=4
            for j in np.unique(idx):
                mask=idx==j;assert mask.sum()>=10;mean=float(w[mask]@y[mask]/w[mask].sum());err=abs(mean-float(tree.tree_.value[j,0,0]));worst=max(worst,err);assert err<1e-8
        assert nodes==rec['total_nodes'] and leaves==rec['total_leaves'];fitguess=model.predict(x);assert abs(float(w@((fitguess-y)**2)/w.sum())-rec['fit_positive_MSE'])<1e-10;audits.append(dict(fold=f,seed=s,nodes=nodes,leaves=leaves,leaf_mean_error=worst,event_frozen=True))
        pp=[r for r in pred if r['fold']==f and r['seed']==s];hw=hierarchy(pp);actual=np.array([r['actual_next_gain']/scale for r in pp]);yes=actual>0;pw=hw[yes]/hw[yes].sum();guess=np.array([r['predicted_normalized_gain'] for r in pp]);mu=np.array([r['conditional_normalized_gain'] for r in pp]);p=np.array([r['event_probability'] for r in pp]);assert np.allclose(p,[oldp[s,r['sample_id']]['event_probability'] for r in pp],atol=1e-12,rtol=1e-10);assert np.allclose(guess,p*mu,atol=1e-12,rtol=1e-12);assert mu.min()>0 and mu.min()>=y.min()-1e-8 and mu.max()<=y.max()+1e-8;conditionalmean=float(w@y/w.sum());predictive.append(dict(fold=f,seed=s,expected_gain_MSE=float(hw@((guess-actual)**2)),expected_gain_MAE=float(hw@abs(guess-actual)),zero_gain_MSE=float(hw@(actual**2)),positive_conditional_MSE=float(pw@((mu[yes]-actual[yes])**2)),positive_conditional_MAE=float(pw@abs(mu[yes]-actual[yes])),positive_constant_MSE=float(pw@((conditionalmean-actual[yes])**2)),mean_expected_gain=float(hw@guess),mean_actual_gain=float(hw@actual)))
    keys=[k for k in predictive[0] if k not in ['fold','seed']];pm={k:float(np.mean([r[k] for r in predictive])) for k in keys};C.save('forest_audit.json',dict(passed=True,all12_native_model_params_leafmeans_and_eventweights_verified=True,pools=audits));C.save('predictive_metrics.json',dict(weighting='date/call/eligible-startbalanced perpool,positivesrenormalizedonlyforconditionalmetrics',all12_mean=pm,pools=predictive,all12poolsnotindependent=True))
    _,_,cases=C.B.load();tr=read(O/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in tr};metrics=read(O/'search_metrics.json');family=read(O/'family_summary.json');comparisons={};actualpairs={}
    for name,folder in [('GRID198',C.LINEAR),('GRID199',C.CONSTANT)]:
        oi={(r['method'],r['sample_id']):r for r in read(folder/'oof_trajectories.json')};om=read(folder/'search_metrics.json');matched=[];actual=[]
        for s in range(3):
            for base in metrics[f'NET_seed{s}']['per_date']:
                d=base['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);arms={'OLD':[oi[f'GATE_seed{s}',sid] for sid in ids],'TREE':[ni[f'GATE_seed{s}',sid] for sid in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',sid] for sid in ids]};costs={k:sum(r['queries']-r['baseline_queries'] for r in rr) for k,rr in arms.items()};budget=min(costs['OLD'],costs['TREE']);row=dict(seed=s,date=d,matched_extra_query_total=budget,original_extra_query_totals=costs)
                for key,rr in arms.items():
                    prob=budget/costs[key] if costs[key] else 0.;gain=prob*sum(r['gain']-r['baseline_gain'] for r in rr);matches=prob*sum(r['action_match']-ni[f'NET_seed{s}',r['sample_id']]['action_match'] for r in rr);row[key]=dict(keep_probability=prob,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+gain/N,action_match_rate=base['action_match_rate']+matches/N,mean_net_utility=base['mean_net_utility']+(gain-fee*budget)/(scale*N))
                assert len({row[k]['mean_queries'] for k in arms})==1;matched.append(row);a=next(r for r in metrics[f'GATE_seed{s}']['per_date'] if r['date']==d);b=next(r for r in om[f'GATE_seed{s}']['per_date'] if r['date']==d);actual.append(dict(seed=s,date=d,NET_delta=a['mean_net_utility']-b['mean_net_utility'],gain_delta=a['mean_gain']-b['mean_gain'],query_delta=a['mean_queries']-b['mean_queries'],match_delta_pp=100*(a['action_match_rate']-b['action_match_rate'])))
        fields=['mean_queries','mean_gain','action_match_rate','mean_net_utility'];macro={k:{field:float(np.mean([r[k][field] for r in matched])) for field in fields} for k in arms};perseed=[]
        for s in range(3):
            rr=[r for r in matched if r['seed']==s];perseed.append(dict(seed=s,NET_delta=float(np.mean([r['TREE']['mean_net_utility']-r['OLD']['mean_net_utility'] for r in rr])),gain_delta=float(np.mean([r['TREE']['mean_gain']-r['OLD']['mean_gain'] for r in rr])),match_delta_pp=100*float(np.mean([r['TREE']['action_match_rate']-r['OLD']['action_match_rate'] for r in rr])),positive_NET_dates=sum(r['TREE']['mean_net_utility']>r['OLD']['mean_net_utility'] for r in rr)))
        comparisons[name]=dict(prespecified=True,analytic_expected_queries_not_hardbudget_or_actualtime=True,macro=macro,per_seed=perseed,per_date_seed=matched);actualpairs[name]=actual
    C.save('expected_budget_comparison.json',comparisons);C.save('all_actual_date_seed_differences.json',actualpairs);nc=collections.Counter(c['date'] for c in cases);spent=0.;waste=0.
    for p in pred:
        if p['gate_continue']:
            q=p['actual_next_queries'];mass=1/(3*16*nc[p['date']]);spent+=mass*q
            if p['actual_next_gain']==0:waste+=mass*q
    C.save('wasted_query_audit.json',dict(date_macro_extra_queries=spent,date_macro_zero_gain_queries=waste,date_macro_zero_gain_query_fraction=waste/spent))
    new=family['GATE'];linear=read(C.LINEAR/'family_summary.json')['GATE'];constant=read(C.CONSTANT/'family_summary.json')['GATE'];direct=read(C.DIRECT/'family_summary.json')['GATE'];net=family['NET'];deltas={}
    for name,comp in comparisons.items():
        ma=comp['macro'];deltas[name]=dict(tree_minus_old_NET=ma['TREE']['mean_net_utility']-ma['OLD']['mean_net_utility'],tree_minus_old_gain=ma['TREE']['mean_gain']-ma['OLD']['mean_gain'],tree_minus_old_match_pp=100*(ma['TREE']['action_match_rate']-ma['OLD']['action_match_rate']),tree_minus_random_NET=ma['TREE']['mean_net_utility']-ma['ADD1_RANDOM']['mean_net_utility'])
    baseseeds=[]
    for s in range(3):
        a,b=metrics[f'GATE_seed{s}']['macro'],metrics[f'NET_seed{s}']['macro'];baseseeds.append(dict(seed=s,gain_delta=a['mean_gain']-b['mean_gain'],query_delta=a['mean_queries']-b['mean_queries'],NET_delta=a['mean_net_utility']-b['mean_net_utility']))
    decision=dict(internal_learning_only=True,no_new_FRAME_closedloop_comparison=True,new_forest_heads=12,new_trees=768,all_fixed_seeds_dates_kept=True,tree_vs_base_learning_NET=new['mean_net_utility']-net['mean_net_utility'],tree_vs_base_learning_per_seed=baseseeds,matched=deltas,retain_low_query_tree_and_higher_reward_linear_candidates=True,no_automatic_replacement_ensemble_scaling_or_parametersearch=True);C.save('decision.json',decision)
    counts=read(O/'gate_counts.json');fits=read(O/'fits.json')['records'];lines=['# GRID201：浅随机树条件收益头的内部改进','',f"本轮固定12个树回归头完成，事件概率和原神经网络不变。相对线性条件头，查询{linear['mean_queries']:.3f}→{new['mean_queries']:.3f}，奖励增量保留率{100*linear['gain_capture']:.2f}%→{100*new['gain_capture']:.2f}%。它更谨慎，须结合同预算选择结果判断，不以查询少或奖励单项定胜负。",'',
        '## 固定改动与容量','',
        '条件头改为ExtraTrees回归，每头64棵树，深度最多4，叶子至少10行，所有特征可参与随机划分、不自助重采样、单线程；随机种子对应原NET种子0/1/2。仍只用GRID198相同的正例目标、条件权重、83维输入和标准化。没有调参、换数据、重新训练概率头或神经网络。','',
        '每个树叶输出其中训练正例的加权均值，森林输出这些均值的平均。逐叶重建权重均值，逐停止点核对原生预测，全部结果为正且位于拟合正例标签范围；没有额外部署裁剪。但这同时改变容量及归纳偏置，不能把效果只归因于消除负值或外推。','',
        f"12个森林实际节点数为{min(r['total_nodes'] for r in fits)}–{max(r['total_nodes'] for r in fits)}、叶子数为{min(r['total_leaves'] for r in fits)}–{max(r['total_leaves'] for r in fits)}；全部共768棵树。报告节点而不将其说成与84参数线性头同容量。线上额外树推理成本尚未进入真实控制计时。",'',
        '## 内部学习版本比较','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点候选查询|NET效用|','|---|---:|---:|---:|---:|']
    for label,z in [('基础学习搜索NET',net),('GRID196直接收益回归',direct),('GRID198线性条件收益',linear),('GRID199条件正例均值',constant),('GRID201浅树条件收益',new),('NET停止后固定追加一批',family['ADD1'])]:lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.6f}|")
    lines+=['',f"本轮255次追加中，{counts['append_changes']}次改选、{counts['append_no_change']}次零收益、{counts['append_positive_NET']}次奖励超过代理费用。日期宏平均追加查询中零奖励占{100*waste/spent:.2f}%。这些是共享教师路径的种子—前缀配对，非独立场景。",'',
        '## 同预期查询量比较','',
        '方案在训练前固定：逐日期、种子取双方较小追加预算，对各自已选择批次按成本决定的统一概率保留，解析计算相同预期查询的结果。固定追加一批同样稀释作随机参照。没有执行新随机策略，也不是硬预算或真实时延匹配。']
    for name,comp in comparisons.items():
        lines+=['',f'与{name}比较：','', '|方法|每起点预期查询|预期动作一致率|预期奖励增量|预期NET效用|','|---|---:|---:|---:|---:|']
        for k,label in [('OLD',name),('TREE','GRID201'),('ADD1_RANDOM','随机追加参照')]:
            z=comp['macro'][k];lines.append(f"|{label}|{z['mean_queries']:.6f}|{100*z['action_match_rate']:.6f}%|{z['mean_gain']:.9f}|{z['mean_net_utility']:.9f}|")
        lines+=['','|种子|树头NET差|奖励差|动作一致率差（百分点）|NET正向日期数|','|---|---:|---:|---:|---:|']
        for p in comp['per_seed']:lines.append(f"|{p['seed']}|{p['NET_delta']:+.8f}|{p['gain_delta']:+.9f}|{p['match_delta_pp']:+.4f}|{p['positive_NET_dates']}/16|")
    lines+=['','## 预测误差与正收益漏失','', '|诊断量|12池描述均值|','|---|---:|',f"|总体归一化收益MSE|{pm['expected_gain_MSE']:.6f}|",f"|固定零预测MSE|{pm['zero_gain_MSE']:.6f}|",f"|总体收益MAE|{pm['expected_gain_MAE']:.6f}|",f"|正例条件收益MSE|{pm['positive_conditional_MSE']:.6f}|",f"|正例固定均值MSE|{pm['positive_constant_MSE']:.6f}|",f"|预测/实际总体收益均值|{pm['mean_expected_gain']:.6f} / {pm['mean_actual_gain']:.6f}|",'',
        '误差按各池日期/调用/停止起点均衡；正例条件误差仅在实际正收益样本上归一。没有将正输出或低MSE当作查询选择有效的证明；少查也可能漏掉大收益。所有种子、反向日期及预测都保留。','',
        '## 当前定位','',
        '保留树头作为低查询内部候选，同时保留线性头和直接收益回归的较高奖励版本，不因一个平均值选择性删除前一阶段改善。用户当前要求先改进学习模块，因此本轮没有启动新的原FRAME闭环对比。','',
        '本轮仅16个已暴露历史训练日期上的折外教师路径研究，不是新的完整周质量、真实时延、跨电网或最终测试结论。全部12森林在留出评分前发布，原NET/ADD1轨迹、概率模型和叶子均值均已核对。']
    (O/'GRID201_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('publication_audit.json',dict(passed=True,all12forest_parameters_leafmeans_and_eventmodels_verified=True,all96expected_query_budgets_equal=True,all_actual_and_predictive_metrics_retained=True,no_new_FRAME_comparison=True,new_forest_heads=12,new_trees=768,new_NET_forwards=0,new_physics=0));C.trained();C.save('complete.json',dict(passed=True,all12heads_all16dates_all3seeds=True,new_forest_heads=12,new_trees=768,event_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));C.archive('complete');dest=R/'论文协作资料/FRAME_浅树条件收益_GRID201';dest.mkdir(exist_ok=False)
    for n in ['GRID201_report.md','decision.json','expected_budget_comparison.json','predictive_metrics.json','forest_audit.json','wasted_query_audit.json','review.json','publication_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f'GRID201固定12浅ExtraTrees条件头完成，共768树，原事件/NET不改。查询16.842→13.486、奖励保留69.60%→66.58%，相对基础学习NET代理效用差{new["mean_net_utility"]-net["mean_net_utility"]:+.8f}；同预期查询对198 NET差{deltas["GRID198"]["tree_minus_old_NET"]:+.8f}，对199{deltas["GRID199"]["tree_minus_old_NET"]:+.8f}。保留低查询/高奖励多版本、全部种子反例，不启动新原FRAME闭环。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID201浅树条件收益内部试验\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,t0=t.split('\n',1);t=h+'\n'+entry+'\n'+t0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID201后\n\n1. 按用户要求继续学习模块内部改进，暂不启动新的原FRAME闭环对比；保留低查询树头与较高奖励线性/直接收益版本及全部反例。\n2. 下一步先只读追踪树头相对198保留和漏掉的高收益状态，检查保守性来自概率还是条件幅度，不按单个总体均值盲目调阈值。\n3. 新训练先通知固定范围，不扫描参数、择种子、扩日期/物理/最终8或论文。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID201浅树条件收益内部试验\n\n[报告](protocols/GRID201_tree_amount/complete/GRID201_report.md)；固定12头/768树，保留低查询与高奖励多版本，不启动新FRAME闭环。\n\n'.encode('utf8')+p.read_bytes());print('GRID201_COMPLETE',json.dumps(dict(decision=decision,per_seed={k:v['per_seed'] for k,v in comparisons.items()},predictive=pm,wasted=waste/spent),ensure_ascii=False))
if __name__=='__main__':main()
