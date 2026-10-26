"""Constant-amount comparator audit, matched budgets and report."""
from pathlib import Path
import collections,json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def hierarchy(rows):
    ds={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set)
    for r in rows:gd[r['date']].add(r['physical_state_group']);gs[r['physical_state_group']].add(r['sample_id'])
    w=np.array([1/(len(ds)*len(gd[r['date']])*len(gs[r['physical_state_group']])) for r in rows]);assert abs(w.sum()-1)<1e-10;return w
def main():
    C.qualified();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists();pred=read(O/'gate_predictions.json');oldpred={(r['seed'],r['sample_id']):r for r in read(C.PARENT/'gate_predictions.json')};refstats={(r['fold'],r['seed']):r for r in read(C.PARENT/'predictive_metrics.json')['pools']};task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];predictive=[]
    for f in range(4):
        for s in range(3):
            cp=np.load(O/'models'/f'f{f}_s{s}.npz');event=np.load(C.EVENT/'models'/f'f{f}_s{s}.npz');lab=np.load(C.PARENT/'fit_labels'/f'f{f}_s{s}.npz');constant=float(lab['sample_weight']@lab['target']/lab['sample_weight'].sum());assert cp['conditional_constant'][0]==constant and np.array_equal(cp['event_coef'],event['coef'][0]);pp=[r for r in pred if r['fold']==f and r['seed']==s];hw=hierarchy(pp);actual=np.array([r['actual_next_gain']/scale for r in pp]);yes=actual>0;pw=hw[yes]/hw[yes].sum();guess=np.array([r['predicted_normalized_gain'] for r in pp]);prob=np.array([r['event_probability'] for r in pp]);assert np.allclose(prob,[oldpred[s,r['sample_id']]['event_probability'] for r in pp],atol=1e-12,rtol=1e-10);assert np.allclose(guess,prob*constant,atol=1e-12,rtol=1e-12);cmse=float(pw@((constant-actual[yes])**2));assert abs(cmse-refstats[f,s]['fit_positive_mean_MSE_on_held_positives'])<1e-10
            predictive.append(dict(fold=f,seed=s,fit_positive_constant=constant,expected_gain_MSE=float(hw@((guess-actual)**2)),expected_gain_MAE=float(hw@abs(guess-actual)),positive_conditional_MSE=cmse,mean_expected_gain=float(hw@guess),mean_actual_gain=float(hw@actual)))
    keys=[k for k in predictive[0] if k not in ['fold','seed']];pm={k:float(np.mean([r[k] for r in predictive])) for k in keys};C.save('predictive_metrics.json',dict(pools=predictive,all12_mean=pm,all12_conditional_MSE_exactly_matches_prior_GRID198_diagnostic=True))
    _,_,cases=C.B.load();tr=read(O/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in tr};metrics=read(O/'search_metrics.json');comparisons={};actualpairs={}
    for name,folder in [('GRID198',C.PARENT),('GRID190',C.OLD)]:
        old=read(folder/'oof_trajectories.json');oi={(r['method'],r['sample_id']):r for r in old};om=read(folder/'search_metrics.json');matched=[];actual=[]
        for s in range(3):
            for base in metrics[f'NET_seed{s}']['per_date']:
                d=base['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);arms={'OLD':[oi[f'GATE_seed{s}',sid] for sid in ids],'CONSTANT':[ni[f'GATE_seed{s}',sid] for sid in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',sid] for sid in ids]};budgets={k:sum(r['queries']-r['baseline_queries'] for r in rr) for k,rr in arms.items()};budget=min(budgets['OLD'],budgets['CONSTANT']);row=dict(seed=s,date=d,matched_extra_query_total=budget,original_extra_query_totals=budgets)
                for key,rr in arms.items():
                    p=budget/budgets[key] if budgets[key] else 0.;extra=p*sum(r['gain']-r['baseline_gain'] for r in rr);matches=p*sum(r['action_match']-ni[f'NET_seed{s}',r['sample_id']]['action_match'] for r in rr);row[key]=dict(keep_probability=p,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+extra/N,action_match_rate=base['action_match_rate']+matches/N,mean_net_utility=base['mean_net_utility']+(extra-fee*budget)/(scale*N))
                assert len({row[k]['mean_queries'] for k in arms})==1;matched.append(row);a=next(r for r in metrics[f'GATE_seed{s}']['per_date'] if r['date']==d);b=next(r for r in om[f'GATE_seed{s}']['per_date'] if r['date']==d);actual.append(dict(seed=s,date=d,NET_delta=a['mean_net_utility']-b['mean_net_utility'],gain_delta=a['mean_gain']-b['mean_gain'],query_delta=a['mean_queries']-b['mean_queries'],match_delta_pp=100*(a['action_match_rate']-b['action_match_rate'])))
        fields=['mean_queries','mean_gain','action_match_rate','mean_net_utility'];macro={k:{field:float(np.mean([r[k][field] for r in matched])) for field in fields} for k in arms};perseed=[]
        for s in range(3):
            rr=[r for r in matched if r['seed']==s];perseed.append(dict(seed=s,NET_delta=float(np.mean([r['CONSTANT']['mean_net_utility']-r['OLD']['mean_net_utility'] for r in rr])),gain_delta=float(np.mean([r['CONSTANT']['mean_gain']-r['OLD']['mean_gain'] for r in rr])),match_delta_pp=100*float(np.mean([r['CONSTANT']['action_match_rate']-r['OLD']['action_match_rate'] for r in rr])),positive_NET_dates=sum(r['CONSTANT']['mean_net_utility']>r['OLD']['mean_net_utility'] for r in rr)))
        comparisons[name]=dict(prespecified=True,analytic_expected_queries_not_hardbudget_or_walltime=True,macro=macro,per_seed=perseed,per_date_seed=matched);actualpairs[name]=actual
    C.save('expected_budget_comparison.json',comparisons);C.save('all_actual_date_seed_differences.json',actualpairs)
    nc=collections.Counter(c['date'] for c in cases);spent=0.;waste=0.;rawspent=0;rawwaste=0
    for p in pred:
        if p['gate_continue']:
            q=p['actual_next_queries'];mass=1/(3*16*nc[p['date']]);spent+=mass*q;rawspent+=q
            if p['actual_next_gain']==0:waste+=mass*q;rawwaste+=q
    C.save('wasted_query_audit.json',dict(date_macro_extra_queries=spent,date_macro_zero_gain_queries=waste,date_macro_zero_gain_query_fraction=waste/spent,raw_extra_queries=rawspent,raw_zero_gain_queries=rawwaste,raw_zero_gain_query_fraction=rawwaste/rawspent))
    family=read(O/'family_summary.json');new=family['GATE'];learned=read(C.PARENT/'family_summary.json')['GATE'];direct=read(C.DIRECT/'family_summary.json')['GATE'];old=read(C.OLD/'family_summary.json')['GATE'];deltas={}
    for name,comp in comparisons.items():
        ma=comp['macro'];deltas[name]=dict(constant_minus_old_NET=ma['CONSTANT']['mean_net_utility']-ma['OLD']['mean_net_utility'],constant_minus_old_gain=ma['CONSTANT']['mean_gain']-ma['OLD']['mean_gain'],constant_minus_old_match_pp=100*(ma['CONSTANT']['action_match_rate']-ma['OLD']['action_match_rate']),constant_minus_random_NET=ma['CONSTANT']['mean_net_utility']-ma['ADD1_RANDOM']['mean_net_utility'])
    decision=dict(new_optimization_fits=0,training_derived_means_recomputed=12,readout_parameters_constant85_vs_learned168=True,all_seeds_dates_and_counterexamples_kept=True,matched=deltas,do_not_claim_learned_amount_necessary_from_unmatched_quality=True,no_automatic_replacement_or_scale=True);C.save('decision.json',decision)
    lines=['# GRID199：条件收益回归是否超过一个训练正例均值','',
        '本轮给GRID198补充简单对照：冻结同一组事件概率模型，将学习的条件收益函数替换为对应拟合池的正收益加权均值。这个均值已在GRID198预测误差诊断中使用，本轮只重新计算并封存为实际门控；没有新的优化训练，但12个常数仍是由训练数据估计的统计量。','',
        '## 比较条件','',
        '两个方法共享相同的事件概率、83维可见输入、原标准化、半STOP/半实际CONT训练分布、正例条件权重和查询费用；动作都是“概率乘条件收益大于下一批已知费用时追加一批”。均值只来自拟合正例，不使用留出标签，全部12份常数模型发布后才评分。','',
        '常数方法含84个冻结事件参数加1个统计均值，共85个读出参数；GRID198为168个。它是检验条件收益头是否必要的简单基线，不是同容量消融。没有扫描其他均值、阈值、校准系数或数据域。','',
        '## 全量局部结果','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点候选查询|NET效用|','|---|---:|---:|---:|---:|']
    for label,z in [('原NET',family['NET']),('GRID190改选门控',old),('GRID196无条件收益回归',direct),('GRID198概率乘学习条件收益',learned),('GRID199概率乘拟合正例均值',new),('停止后固定追加一批',family['ADD1'])]:lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.6f}|")
    counts=read(O/'gate_counts.json');lines+=['',f"常数模型追加{counts['append']}次，其中{counts['append_changes']}次改选、{counts['append_no_change']}次未改选，{counts['append_positive_NET']}次实际局部奖励超过代理查询费用；零奖励候选占追加查询{100*waste/spent:.2f}%。其原始奖励增量更高时，也必须同时考虑额外查询。",'',
        '## 同预期候选查询比较','',
        '按预先冻结方案，逐日期、种子取双方较小追加预算，以成本确定的统一概率稀释各自已选批次。相同预期查询下比较奖励、动作一致与NET，并保留随机ADD1参照。这是解析期望，不是在线硬预算或实测时延。']
    for name,comp in comparisons.items():
        lines+=['',f'与{name}比较：','', '|方法|每起点预期查询|预期动作一致率|预期奖励增量|预期NET效用|','|---|---:|---:|---:|---:|']
        for key,label in [('OLD',name),('CONSTANT','GRID199常数收益'),('ADD1_RANDOM','随机追加参照')]:
            z=comp['macro'][key];lines.append(f"|{label}|{z['mean_queries']:.6f}|{100*z['action_match_rate']:.6f}%|{z['mean_gain']:.9f}|{z['mean_net_utility']:.9f}|")
        lines+=['','|种子|常数模型NET差|奖励差|动作一致率差（百分点）|NET为正日期数|','|---|---:|---:|---:|---:|']
        for p in comp['per_seed']:lines.append(f"|{p['seed']}|{p['NET_delta']:+.8f}|{p['gain_delta']:+.9f}|{p['match_delta_pp']:+.4f}|{p['positive_NET_dates']}/16|")
    lines+=['','全部三个种子、96个日期—种子比较及反向结果保留。不能从一个平均数或某个有利种子断言条件头必须保留，也不能将同预期比较写成真实控制提速。','',
        '## 预测误差核对','',
        '|量|12池描述均值|','|---|---:|',f"|总体归一化奖励MSE|{pm['expected_gain_MSE']:.6f}|",f"|总体归一化奖励MAE|{pm['expected_gain_MAE']:.6f}|",f"|正例上的条件收益MSE|{pm['positive_conditional_MSE']:.6f}|",f"|预测/实际总体奖励均值|{pm['mean_expected_gain']:.6f} / {pm['mean_actual_gain']:.6f}|",'',
        '误差按每池日期/物理调用/停止起点均衡；条件误差仅在实际正收益停止点重新归一。12池条件MSE逐项复现GRID198此前的拟合正例均值诊断，确认没有为本基线改均值。它与门控决策结果共同报告，不以误差小替代查询选择证据。','',
        '## 研究定位','',
        '这是在看到GRID198结果后提出、随后先冻结再评价的新基线，属于已暴露历史数据上的开发研究，不是新的独立验证。原198条件头、196无条件收益、190低查询门控和全部反例保留；是否继续条件幅度学习，结合本轮相同预算结果决定，而非因为多加一个模块就默认其有效。','',
        '未生成新数据、运行神经网络前向或电网仿真；最终8场景和论文未改。局部奖励、代理查询效用、整周质量与真实时延保持分开。']
    (O/'GRID199_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('publication_audit.json',dict(passed=True,all12_fit_only_means_and_frozen_event_models_verified=True,all12_positive_conditional_MSE_matches_GRID198=True,all96_expected_query_budgets_match=True,training_derived_constants=12,new_optimization_fits=0,new_NET_forwards=0,new_physics=0));C.qualified();C.save('complete.json',dict(passed=True,all12constantmodels_all16dates_all3seeds=True,training_derived_constants=12,new_optimization_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_条件收益均值对照_GRID199';dest.mkdir(exist_ok=False)
    for n in ['GRID199_report.md','decision.json','expected_budget_comparison.json','predictive_metrics.json','wasted_query_audit.json','review.json','publication_audit.json','complete.json']:shutil.copy2(O/n,dest/n);assert C.sha(O/n)==C.sha(dest/n)
    msg=f'GRID199同事件模型乘拟合正例均值完成，无新优化/12训练均值，85参数对198的168。常数奖励保留{100*new["gain_capture"]:.2f}%/查询{new["mean_queries"]:.3f}，198为69.60%/16.842；同预期查询常数减198 NET差{deltas["GRID198"]["constant_minus_old_NET"]:+.8f}，动作差{deltas["GRID198"]["constant_minus_old_match_pp"]:+.4f}pp。所有日期/种子/随机对照保留，不按原始质量高就声称条件头无用或必须。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID199条件收益均值基线完成\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,t0=t.split('\n',1);t=h+'\n'+entry+'\n'+t0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID199后\n\n1. 对照198学习条件幅度与199拟合常数的全部同预期预算结果，保留190/196权衡及所有反例，不凭平均预测误差替代决策证据。\n2. 后续先只读追踪两种幅度读出在同一事件概率下的分歧，分开收益过滤与正例幅度估计；不自动组合、扫描阈值或重训。\n3. 新训练先通知固定范围，不自动扩物理、日期、最终8或论文。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID199条件收益均值基线\n\n[报告](protocols/GRID199_constant_amount/complete/GRID199_report.md)；冻结同事件模型，正例均值对学习幅度，零新优化、12训练统计均值。\n\n'.encode('utf8')+p.read_bytes());print('GRID199_COMPLETE',json.dumps(dict(decision=decision,per_seed={k:v['per_seed'] for k,v in comparisons.items()},predictive=pm,wasted=waste/spent),ensure_ascii=False))
if __name__=='__main__':main()
