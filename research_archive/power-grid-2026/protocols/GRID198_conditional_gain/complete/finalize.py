"""Complete audits and prespecified matched-query comparisons for conditional heads."""
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
    C.trained();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];pred=read(O/'gate_predictions.json');eventold={(r['seed'],r['sample_id']):r for r in read(C.EVENT/'gate_predictions.json')};fits=read(O/'fits.json');audits=[];predictive=[]
    for rec in fits['records']:
        f,s=rec['fold'],rec['seed'];z=np.load(C.FIT/'fit_features'/f'f{f}_s{s}.npz');lab=np.load(O/'fit_labels'/f'f{f}_s{s}.npz');cp=np.load(O/'models'/f'f{f}_s{s}.npz');original=np.load(C.EVENT/'models'/f'f{f}_s{s}.npz');direct=np.load(C.DIRECT/'fit_labels'/f'f{f}_s{s}.npz');idx=lab['indices'];y=lab['target'];w=lab['sample_weight'];x=z['standard'][idx];assert np.array_equal(cp['event_coef'],original['coef'][0]) and np.array_equal(cp['event_intercept'],original['intercept']);assert np.array_equal(idx,np.flatnonzero(direct['target']>0));assert np.array_equal(y,direct['target'][idx]);expected=z['sample_weight'][idx]*z['sample_weight'].sum()/z['sample_weight'][idx].sum();assert np.allclose(w,expected,atol=1e-12,rtol=1e-12);assert (y>0).all();raw=x@cp['coef']+cp['intercept'][0];err=raw-y;value=float((w@(err*err)+cp['coef']@cp['coef'])/(2*w.sum()));assert abs(value-rec['objective'])<1e-10;g=np.r_[x.T@(w*err)+cp['coef'],w@err]/w.sum();denom=1+max(float(abs(x.T@(w*raw)).max()),float(abs(x.T@(w*y)).max()))/w.sum();relative=float(abs(g).max())/denom;assert relative<1e-8;audits.append(dict(fold=f,seed=s,rows=len(y),event_frozen=True,conditional_weights_correct=True,objective=value,relative_stationarity=relative))
        pp=[r for r in pred if r['fold']==f and r['seed']==s];hw=hierarchy(pp);actual=np.array([r['actual_next_gain']/scale for r in pp]);yy=actual>0;prob=np.array([r['event_probability'] for r in pp]);logit=np.array([r['event_logit'] for r in pp]);mu=np.array([r['conditional_normalized_gain'] for r in pp]);guess=prob*mu;oldprob=np.array([eventold[s,r['sample_id']]['probability'] for r in pp]);assert np.allclose(prob,oldprob,atol=1e-12,rtol=1e-10);assert np.allclose(guess,[r['predicted_normalized_gain'] for r in pp],atol=1e-12,rtol=1e-12)
        event_error=-(1-prob[yy])*actual[yy];amount_error=prob[yy]*(mu[yy]-actual[yy]);assert np.allclose(event_error+amount_error,guess[yy]-actual[yy],atol=1e-12,rtol=1e-12);pw=hw[yy]/hw[yy].sum();conditionalmean=float(w@y/w.sum());predictive.append(dict(fold=f,seed=s,event_Brier=float(hw@((prob-yy)**2)),event_BCE=float(hw@(np.logaddexp(0,logit)-yy*logit)),expected_gain_MSE=float(hw@((guess-actual)**2)),expected_gain_MAE=float(hw@abs(guess-actual)),zero_gain_MSE=float(hw@(actual**2)),conditional_positive_MSE=float(pw@((mu[yy]-actual[yy])**2)),conditional_positive_MAE=float(pw@abs(mu[yy]-actual[yy])),fit_positive_mean_MSE_on_held_positives=float(pw@((conditionalmean-actual[yy])**2)),mean_expected_gain=float(hw@guess),mean_actual_gain=float(hw@actual),positive_event_error_component=float(pw@event_error),positive_amount_error_component=float(pw@amount_error)))
    pkeys=[k for k in predictive[0] if k not in ['fold','seed']];pm={k:float(np.mean([r[k] for r in predictive])) for k in pkeys};C.save('endpoint_audit.json',dict(passed=True,all12_positiveonly_sampling_objectives_and_frozen_event_models_verified=True,pools=audits));C.save('predictive_metrics.json',dict(weighting='perpooldate/call/eligible-startbalanced,positivesrenormalizedonlyforconditionalmetrics,12overlappingpoolsdescriptive',normalized_gain_scale=scale,all12_mean=pm,pools=predictive,positive_error_decomposition_is_accounting_not_causal=True))
    tr=read(O/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in tr};metrics=read(O/'search_metrics.json');family=read(O/'family_summary.json');_,_,cases=C.B.load();comparisons={};actualpairs={}
    for name,folder in [('GRID190',C.OLD),('GRID196',C.DIRECT)]:
        old=read(folder/'oof_trajectories.json');oi={(r['method'],r['sample_id']):r for r in old};om=read(folder/'search_metrics.json');matched=[];actual=[]
        for s in range(3):
            for base in metrics[f'NET_seed{s}']['per_date']:
                d=base['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);arms={'OLD':[oi[f'GATE_seed{s}',sid] for sid in ids],'HURDLE':[ni[f'GATE_seed{s}',sid] for sid in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',sid] for sid in ids]};budgets={k:sum(r['queries']-r['baseline_queries'] for r in rr) for k,rr in arms.items()};budget=min(budgets['OLD'],budgets['HURDLE']);row=dict(seed=s,date=d,matched_extra_query_total=budget,original_extra_query_totals=budgets)
                for key,rr in arms.items():
                    prob=budget/budgets[key] if budgets[key] else 0.;extra=prob*sum(r['gain']-r['baseline_gain'] for r in rr);matches=prob*sum(r['action_match']-ni[f'NET_seed{s}',r['sample_id']]['action_match'] for r in rr);row[key]=dict(keep_probability=prob,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+extra/N,action_match_rate=base['action_match_rate']+matches/N,mean_net_utility=base['mean_net_utility']+(extra-fee*budget)/(scale*N))
                assert len({row[k]['mean_queries'] for k in arms})==1;matched.append(row);a=next(v for v in metrics[f'GATE_seed{s}']['per_date'] if v['date']==d);b=next(v for v in om[f'GATE_seed{s}']['per_date'] if v['date']==d);actual.append(dict(seed=s,date=d,NET_delta=a['mean_net_utility']-b['mean_net_utility'],gain_delta=a['mean_gain']-b['mean_gain'],match_delta_pp=100*(a['action_match_rate']-b['action_match_rate']),query_delta=a['mean_queries']-b['mean_queries']))
        fields=['mean_queries','mean_gain','action_match_rate','mean_net_utility'];macro={k:{field:float(np.mean([r[k][field] for r in matched])) for field in fields} for k in arms};perseed=[]
        for s in range(3):
            rr=[r for r in matched if r['seed']==s];perseed.append(dict(seed=s,NET_delta=float(np.mean([r['HURDLE']['mean_net_utility']-r['OLD']['mean_net_utility'] for r in rr])),gain_delta=float(np.mean([r['HURDLE']['mean_gain']-r['OLD']['mean_gain'] for r in rr])),match_delta_pp=100*float(np.mean([r['HURDLE']['action_match_rate']-r['OLD']['action_match_rate'] for r in rr])),positive_NET_dates=sum(r['HURDLE']['mean_net_utility']>r['OLD']['mean_net_utility'] for r in rr)))
        comparisons[name]=dict(prespecified=True,analytic_expectation_not_online_hardbudget_or_actualtime=True,macro=macro,per_seed=perseed,per_date_seed=matched);actualpairs[name]=actual
    C.save('expected_budget_comparison.json',comparisons);C.save('all_actual_date_seed_differences.json',actualpairs)
    spent=0.;waste=0.;rawspent=0;rawwaste=0;counts_bydate=collections.Counter(c['date'] for c in cases)
    for p in pred:
        if p['gate_continue']:
            q=p['actual_next_queries'];mass=1/(3*16*counts_bydate[p['date']]);spent+=mass*q;rawspent+=q
            if p['actual_next_gain']==0:waste+=mass*q;rawwaste+=q
    C.save('wasted_query_audit.json',dict(date_macro_extra_queries=spent,date_macro_zero_gain_queries=waste,date_macro_zero_gain_query_fraction=waste/spent,raw_extra_queries=rawspent,raw_zero_gain_queries=rawwaste,raw_zero_gain_query_fraction=rawwaste/rawspent))
    nc=read(O/'gate_counts.json');new=family['GATE'];old=read(C.OLD/'family_summary.json')['GATE'];direct=read(C.DIRECT/'family_summary.json')['GATE'];event=read(C.EVENT/'family_summary.json')['GATE'];net=family['NET'];deltas={}
    for name,comp in comparisons.items():
        ma=comp['macro'];deltas[name]=dict(matched_NET_delta=ma['HURDLE']['mean_net_utility']-ma['OLD']['mean_net_utility'],matched_gain_delta=ma['HURDLE']['mean_gain']-ma['OLD']['mean_gain'],matched_match_delta_pp=100*(ma['HURDLE']['action_match_rate']-ma['OLD']['action_match_rate']),matched_NET_delta_vs_random=ma['HURDLE']['mean_net_utility']-ma['ADD1_RANDOM']['mean_net_utility'])
    decision=dict(all12_newconditionalfits_complete=True,existing_event_and_NET_frozen=True,total_readout_parameters=168,all_seeds_dates_kept=True,local_gain_capture_delta_vs196=new['gain_capture']-direct['gain_capture'],mean_query_delta_vs196=new['mean_queries']-direct['mean_queries'],mean_NET_delta_vs196=new['mean_net_utility']-direct['mean_net_utility'],matched=deltas,retain_190_196_198_development_tradeoffs=True,no_automatic_replacement_ensemble_thresholdsearch_or_scale=True,no_same_capacity_latency_or_fullweek_quality_claim=True);C.save('decision.json',decision)
    lines=['# GRID198：将补查发生概率与条件收益幅度分开估计','',
        f"固定12份条件收益头完成。相对GRID196，候选查询{direct['mean_queries']:.3f}→{new['mean_queries']:.3f}，奖励增量保留率{100*direct['gain_capture']:.2f}%→{100*new['gain_capture']:.2f}%，动作一致率{100*direct['action_match_rate']:.2f}%→{100*new['action_match_rate']:.2f}%。查询减少，也放弃了一部分收益；本报告同时保留二者，不用单项上升定输赢。",'',
        '## 本轮方法','',
        '概率部分直接复用GRID193全部12份改选分类器，不更新任何参数。幅度部分只用同一混合训练人群中下一批奖励严格增加的样本，拟合“已发生收益时”的条件平均奖励。预测的补查收益等于两部分输出相乘，再扣除公开已知查询费用；大于零才追加下一批，最多64候选。','',
        '两个模型对应相同的半STOP/半实际CONT训练分布。条件头的权重为原采样权重限制到正例后重新归一，总权重保持原NSTOP；没有把零收益样本赋予虚构的条件幅度标签。原83维表示、标准化、神经网络、全局搜索与候选排序均不变。','',
        '每个新增幅度头为84参数岭回归，alpha=1固定，截距不惩罚，加权中心化SVD闭式求解。加上冻结的84参数事件头，共168个读出参数，多于GRID196的84个，因此不是同容量消融。条件分布改变，亦不声称相同统计正则效果。','',
        '加权经验均值恒等式E[g]=P(g>0)E[g|g>0]在拟合标签上逐池核对。恒等式不保证估计出的概率和条件均值准确，也不使混合分布概率自动成为部署STOP域的校准概率。没有将GRID196无条件收益预测再次乘概率。','',
        '原GRID193事件头中11份达到原坐标梯度标准、1份达到既定迭代上限；本轮全部冻结保留，没有重训或按收敛结果筛选。所有12份新配对权重推送后才读取留出模型结果。','',
        '## 全部日期与种子的局部结果','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点候选查询|NET效用|','|---|---:|---:|---:|---:|']
    for label,z in [('原NET',net),('GRID190停止点改选门控',old),('GRID193混合改选门控',event),('GRID196无条件收益回归',direct),('GRID198概率乘条件收益',new),('停止后固定追加一批',family['ADD1'])]:lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.6f}|")
    lines+=['',f"本轮892次追加中，{nc['append_changes']}次改选、{nc['append_no_change']}次未改选，{nc['append_positive_NET']}次实际下一批奖励超过固定查询代理费用。按全量日期宏平均，零奖励候选查询占追加查询{100*waste/spent:.2f}%。三个模型种子及同调用多个前缀不是独立场景。",'',
        '## 同预期候选查询比较','',
        '训练前已固定逐日期、种子取双方较小追加预算，并对各自选中的批次以统一概率保留，解析计算相同预期查询量下的收益。概率只由查询数决定；无条件ADD1以同样方式稀释作为随机追加参照。这里没有执行新随机策略，不是硬预算或实际时延匹配。']
    for name,comp in comparisons.items():
        lines+=['',f'与{name}比较：','', '|方法|每起点预期查询|预期动作一致率|预期奖励增量|预期NET效用|','|---|---:|---:|---:|---:|']
        for key,label in [('OLD',name),('HURDLE','GRID198'),('ADD1_RANDOM','随机追加参照')]:
            z=comp['macro'][key];lines.append(f"|{label}|{z['mean_queries']:.6f}|{100*z['action_match_rate']:.6f}%|{z['mean_gain']:.9f}|{z['mean_net_utility']:.9f}|")
        lines+=['','|种子|NET差|奖励差|动作一致率差（百分点）|NET差为正的日期数|','|---|---:|---:|---:|---:|']
        for p in comp['per_seed']:lines.append(f"|{p['seed']}|{p['NET_delta']:+.8f}|{p['gain_delta']:+.9f}|{p['match_delta_pp']:+.4f}|{p['positive_NET_dates']}/16|")
    lines+=['','全部96个配对预算、三个种子和全部反向日期保留。随机参照与模型相比也不能省略，不因其中某个结果有利而换比较口径。','',
        '## 两部分分别学得怎样','',
        '下列误差先在各池内按日期/调用/停止起点均衡，再报告12池描述均值；条件误差仅对正收益状态重新归一。标签使用既有奖励尺度归一化，未在留出数据上拟合阈值或校准器。','',
        '|诊断量|数值|','|---|---:|',f"|事件概率Brier误差|{pm['event_Brier']:.6f}|",f"|事件概率BCE|{pm['event_BCE']:.6f}|",f"|总体收益预测MSE|{pm['expected_gain_MSE']:.6f}|",f"|固定零预测MSE|{pm['zero_gain_MSE']:.6f}|",f"|总体收益预测MAE|{pm['expected_gain_MAE']:.6f}|",f"|正例条件收益MSE|{pm['conditional_positive_MSE']:.6f}|",f"|拟合正例均值在留出正例上的MSE|{pm['fit_positive_mean_MSE_on_held_positives']:.6f}|",f"|预测/实际总体奖励均值|{pm['mean_expected_gain']:.6f} / {pm['mean_actual_gain']:.6f}|",'',
        '所有留出事件概率逐项复现GRID193缓存；条件头只使用正收益拟合标签。对实际正例，预测误差还核对为“概率乘幅度误差”与“漏掉事件概率对应的收益”两项之和。这是误差记账，不能据此作某一头的因果归因，也没有用真实事件替代预测事件运行策略。','',
        '## 目前保留的结论','',
        '两部分估计提供了新的质量—查询权衡，应与GRID190和GRID196同时保留。是否更会选择补查，按前述同预期查询与随机追加结果判断，不能只看追加次数下降。少量、重复的改善仍有研究价值，不要求全部日期或种子获胜。','',
        '保护的是原NET已经评估的局部最佳动作，不保证逐状态优于其他学习门控，也不保证整周存活或成本。当前仍是16个已暴露历史训练日期上的折外教师路径实验；没有新的闭环、电网、实际时延或最终测试。','',
        '审核包括12个条件目标/权重/原目标函数及相对数值驻点误差、冻结事件模型、21402条轨迹、4681个概率乘幅度决定，以及96组预期预算匹配。原NET与ADD1全量重现。']
    (O/'GRID198_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('publication_audit.json',dict(passed=True,all12_positive_only_targets_objectives_and_event_models_checked=True,all4681_event_probabilities_match_GRID193=True,all96_expected_query_budgets_match=True,all168_readout_parameters_disclosed=True,new_conditional_fits=12,event_and_NET_fits=0,new_physics=0));C.trained();C.save('complete.json',dict(passed=True,all12conditionalfits_all16dates_all3seeds=True,new_conditional_fits=12,event_and_NET_fits=0,new_NET_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_概率与条件收益_GRID198';dest.mkdir(exist_ok=False)
    for n in ['GRID198_report.md','decision.json','expected_budget_comparison.json','predictive_metrics.json','wasted_query_audit.json','review.json','publication_audit.json','complete.json']:shutil.copy2(O/n,dest/n);assert C.sha(O/n)==C.sha(dest/n)
    msg=f'GRID198新增12个正例条件收益岭头，冻结193概率/原NET；84新增+84冻结读出参数，非同容量。查询196的19.857→16.842、奖励保留72.19%→69.60%、动作91.572%→91.185%，零收益追加候选{100*waste/spent:.2f}%。同预期查询对190 NET差{deltas["GRID190"]["matched_NET_delta"]:+.8f}，对196{deltas["GRID196"]["matched_NET_delta"]:+.8f}；全部种子/日期/随机对照/反例保留，不自动替换或扩物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID198概率与条件收益分解完成\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,t0=t.split('\n',1);t=h+'\n'+entry+'\n'+t0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID198后\n\n1. 保留190/196/198开发权衡和重复小改善，按全部种子、日期及同预期预算/随机追加比较解释，不自动替换或组合。\n2. 下一步先只读对照事件概率与条件幅度误差，追踪198保留/丢失的196高收益机会及零收益查询；不把两个误差项作因果归因。\n3. 新训练先通知固定范围，不扫描阈值/费用/正则、不自动扩日期/网络/物理/最终8或论文。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID198发生概率乘条件收益\n\n[报告](protocols/GRID198_conditional_gain/complete/GRID198_report.md)；12条件头与冻结事件头，查询与奖励共同下降，全部预期预算和随机对照保留。\n\n'.encode('utf8')+p.read_bytes());print('GRID198_COMPLETE',json.dumps(dict(decision=decision,per_seed={k:v['per_seed'] for k,v in comparisons.items()},predictive=pm,wasted=waste/spent),ensure_ascii=False))
if __name__=='__main__':main()
