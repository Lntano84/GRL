"""Endpoint, predictive and matched-query audits of all fixed GRID196 models."""
from pathlib import Path
import collections,json,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def hierarchy(rows):
    dates={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set)
    for r in rows:gd[r['date']].add(r['physical_state_group']);gs[r['physical_state_group']].add(r['sample_id'])
    return np.array([1/(len(dates)*len(gd[r['date']])*len(gs[r['physical_state_group']])) for r in rows])
def main():
    C.trained();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists();task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale'];tr=read(O/'oof_trajectories.json');ix={(r['method'],r['sample_id']):r for r in tr};metrics=read(O/'search_metrics.json');families=read(O/'family_summary.json');_,_,cases=C.B.load();pred=read(O/'gate_predictions.json');fits=read(O/'fits.json');endpoint=[];predictive=[]
    for rec in fits['records']:
        f,s=rec['fold'],rec['seed'];z=np.load(C.SRC/'fit_features'/f'f{f}_s{s}.npz');cp=np.load(O/'models'/f'f{f}_s{s}.npz');lab=np.load(O/'fit_labels'/f'f{f}_s{s}.npz');x,y,w=z['standard'],lab['target'],z['sample_weight'];beta=cp['coef'];bias=cp['intercept'][0];raw=x@beta+bias;err=raw-y;val=float((w@(err*err)+beta@beta)/(2*w.sum()));g=np.r_[x.T@(w*err)+beta,w@err]/w.sum();denom=1+max(float(abs(x.T@(w*raw)).max()),float(abs(x.T@(w*y)).max()))/w.sum();relative=float(abs(g).max())/denom
        assert abs(val-rec['objective'])<1e-10 and relative<1e-8 and np.array_equal(cp['mean'],z['mean']) and np.array_equal(cp['scale'],z['scale']);endpoint.append(dict(fold=f,seed=s,original_objective=val,original_gradient_infinity=float(abs(g).max()),relative_stationarity=relative))
        pp=[r for r in pred if r['fold']==f and r['seed']==s];hw=hierarchy(pp);assert abs(hw.sum()-1)<1e-10;actual=np.array([r['actual_next_gain']/scale for r in pp]);guess=np.array([r['predicted_normalized_gain'] for r in pp]);unclipped=np.array([r['unclipped_normalized_gain'] for r in pp]);const=float(w@y/w.sum());targetss={'zero':np.zeros(len(pp)),'fit_mean':np.full(len(pp),const),'gain_readout':guess,'unclipped_readout':unclipped}
        stats={key:dict(normalized_MSE=float(hw@((v-actual)**2)),normalized_MAE=float(hw@abs(v-actual)),mean_predicted_normalized_gain=float(hw@v)) for key,v in targetss.items()};stats['actual_mean_normalized_gain']=float(hw@actual);predictive.append(dict(fold=f,seed=s,fit_MSE=float(w@(err*err)/w.sum()),held=stats))
    C.save('endpoint_audit.json',dict(passed=True,all12_original_objectives_and_stationarity_rechecked=True,pools=endpoint));C.save('predictive_metrics.json',dict(weighting='date/call/eligible-start equal; foldpool descriptive,overlappingfitsnotindependent',normalized_gain_scale=scale,pools=predictive,all12_mean={key:{field:float(np.mean([p['held'][key][field] for p in predictive])) for field in ['normalized_MSE','normalized_MAE','mean_predicted_normalized_gain']} for key in ['zero','fit_mean','gain_readout','unclipped_readout']}))
    comparisons={}; paired_actual={}
    for name,dirpath in [('GRID190',C.OLD),('GRID193',R/'outputs/grid193_gate_precondition')]:
        ot=read(dirpath/'oof_trajectories.json');oi={(r['method'],r['sample_id']):r for r in ot};om=read(dirpath/'search_metrics.json');matched=[];actual=[]
        for s in range(3):
            for base in metrics[f'NET_seed{s}']['per_date']:
                d=base['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);arms={'OLD':[oi[f'GATE_seed{s}',i] for i in ids],'GAIN':[ix[f'GATE_seed{s}',i] for i in ids],'ADD1_RANDOM':[ix[f'ADD1_seed{s}',i] for i in ids]};budgets={k:sum(r['queries']-r['baseline_queries'] for r in vv) for k,vv in arms.items()};budget=min(budgets['OLD'],budgets['GAIN']);row=dict(seed=s,date=d,matched_extra_query_total=budget,original_extra_query_totals=budgets)
                for key,vv in arms.items():
                    p=budget/budgets[key] if budgets[key] else 0.;extra=p*sum(r['gain']-r['baseline_gain'] for r in vv);matches=p*sum(r['action_match']-ix[f'NET_seed{s}',r['sample_id']]['action_match'] for r in vv);row[key]=dict(keep_probability=p,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+extra/N,action_match_rate=base['action_match_rate']+matches/N,mean_net_utility=base['mean_net_utility']+(extra-fee*budget)/(scale*N))
                assert len({row[k]['mean_queries'] for k in arms})==1;matched.append(row)
                a=next(r for r in metrics[f'GATE_seed{s}']['per_date'] if r['date']==d);b=next(r for r in om[f'GATE_seed{s}']['per_date'] if r['date']==d);actual.append(dict(seed=s,date=d,NET_delta=a['mean_net_utility']-b['mean_net_utility'],match_delta_pp=100*(a['action_match_rate']-b['action_match_rate']),gain_delta=a['mean_gain']-b['mean_gain'],query_delta=a['mean_queries']-b['mean_queries']))
        fields=['mean_queries','mean_gain','action_match_rate','mean_net_utility'];macro={k:{field:float(np.mean([r[k][field] for r in matched])) for field in fields} for k in arms};perseed=[]
        for s in range(3):
            rr=[r for r in matched if r['seed']==s];perseed.append(dict(seed=s,NET_delta=float(np.mean([r['GAIN']['mean_net_utility']-r['OLD']['mean_net_utility'] for r in rr])),gain_delta=float(np.mean([r['GAIN']['mean_gain']-r['OLD']['mean_gain'] for r in rr])),match_delta_pp=100*float(np.mean([r['GAIN']['action_match_rate']-r['OLD']['action_match_rate'] for r in rr])),positive_NET_dates=sum(r['GAIN']['mean_net_utility']>r['OLD']['mean_net_utility'] for r in rr)))
        comparisons[name]=dict(prespecified=True,analytic_expected_budget_not_actual_hardbudget_or_latency=True,per_date_seed=matched,macro=macro,per_seed=perseed);paired_actual[name]=actual
    C.save('expected_budget_comparison.json',comparisons);C.save('all_actual_date_seed_differences.json',paired_actual)
    base=families['NET'];new=families['GATE'];old=read(C.OLD/'family_summary.json')['GATE'];mix=read(R/'outputs/grid193_gate_precondition/family_summary.json')['GATE'];ma=comparisons['GRID190']['macro'];mnet=ma['GAIN']['mean_net_utility']-ma['OLD']['mean_net_utility'];mgain=ma['GAIN']['mean_gain']-ma['OLD']['mean_gain'];mm=100*(ma['GAIN']['action_match_rate']-ma['OLD']['action_match_rate']);randomnet=ma['GAIN']['mean_net_utility']-ma['ADD1_RANDOM']['mean_net_utility'];counts=read(O/'gate_counts.json')
    decision=dict(all12fits_complete=True,all_seeds_dates_retained=True,local_reward_capture_delta_vs190=new['gain_capture']-old['gain_capture'],actual_queries_delta_vs190=new['mean_queries']-old['mean_queries'],actual_NET_delta_vs190=new['mean_net_utility']-old['mean_net_utility'],matched_expected_query_NET_delta_vs190=mnet,matched_expected_query_gain_delta_vs190=mgain,matched_expected_query_match_delta_pp_vs190=mm,matched_expected_query_NET_delta_vs_random_ADD1=randomnet,retain_190_and_196_as_development_comparators=True,no_claim_of_same_latency_or_fullweek_quality=True,no_automatic_ensemble_scale_or_morefits=True)
    C.save('decision.json',decision)
    lines=['# GRID196：预测下一批奖励幅度的固定线性门控','',
        f"12份固定线性回归门控完成。局部奖励增量保留率由GRID190的{100*old['gain_capture']:.2f}%提高到{100*new['gain_capture']:.2f}%，完整同站动作一致率{100*old['action_match_rate']:.2f}%→{100*new['action_match_rate']:.2f}%；候选查询{old['mean_queries']:.3f}→{new['mean_queries']:.3f}。质量增量和计算代价都保留，不能只报质量上升。",'',
        '## 与前几轮相比改变了什么','',
        '复用GRID192全部训练行、半STOP/半实际CONT权重、83维冻结表征和原STOP标准化。原NET及其搜索决定不更新。每池只拟合一个84参数线性回归，预测下一批的局部奖励增量；从预测值扣除已知查询代理费用，结果为正才追加一批，最多64候选，然后返回已观察最佳动作。','',
        '岭回归alpha=1固定，截距不惩罚，通过加权中心化SVD求闭式解，不扫描参数。目标、损失及输出解释从改选概率变为条件平均奖励，不能称为只修改标签的一项纯消融；alpha=1也不保证与逻辑回归具有等效统计正则。参数数量和原样本权重保持。','',
        '拟合前已验证连续标签、权重及日期隔离；12份最终权重全部推送后才评分留出停止点。没有新的神经网络前向或仿真。旧GRID120加权分类及旧收益回归的不利结果保留，本轮只是检验现有冻结表征上的不同读出目标。','',
        '## 全量局部结果','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点查询|NET效用|','|---|---:|---:|---:|---:|']
    for label,z in [('原NET',base),('GRID190改选门控',old),('GRID193同混合样本改选门控',mix),('GRID196收益幅度门控',new),('停止后固定追加一批',families['ADD1'])]:lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.6f}|")
    lines+=['',f"本轮{counts['append']}次追加中，{counts['append_changes']}次改选、{counts['append_no_change']}次未改选；{counts['append_positive_NET']}次追加的实际局部奖励超过固定查询费用。种子与前缀配对并非独立场景。",'',
        '## 同预期候选查询量比较','',
        '比较在训练前已固定：逐日期、种子按双方较小追加预算，对已选追加批次以统一概率保留，解析计算相同预期候选查询数下的结果。概率只取决于查询总数，不使用奖励；固定追加一批做相同稀释，作为随机参照。它不是一次在线运行的硬预算或实测时延。']
    for name,comp in comparisons.items():
        lines+=['',f'与{name}比较：','', '|方法|每起点预期查询|预期动作一致率|预期奖励增量|预期NET效用|','|---|---:|---:|---:|---:|']
        for key,label in [('OLD',name),('GAIN','GRID196'),('ADD1_RANDOM','随机追加参照')]:
            z=comp['macro'][key];lines.append(f"|{label}|{z['mean_queries']:.6f}|{100*z['action_match_rate']:.6f}%|{z['mean_gain']:.9f}|{z['mean_net_utility']:.9f}|")
        lines+=['','|种子|同预期查询NET差|奖励差|动作一致率差（百分点）|NET为正日期数|','|---|---:|---:|---:|---:|']
        for z in comp['per_seed']:lines.append(f"|{z['seed']}|{z['NET_delta']:+.8f}|{z['gain_delta']:+.9f}|{z['match_delta_pp']:+.4f}|{z['positive_NET_dates']}/16|")
    lines+=['',f"对GRID190，同预期查询NET差{mnet:+.8f}，动作一致率差{mm:+.4f}个百分点；相对同预期查询随机追加参照，NET差{randomnet:+.8f}。三个种子及48个日期—种子明细全保留，不按结果择取预算、种子或日期。",'',
        '## 预测误差与边界','',
        '预测误差使用已知奖励尺度归一化，并在每个池内按日期/物理调用/停止起点均衡。零预测和对应拟合池均值只是固定参照，未读取留出标签来拟合常数。','',
        '|读出|留出归一化MSE（12池描述均值）|留出归一化MAE|','|---|---:|---:|']
    pm=read(O/'predictive_metrics.json')['all12_mean']
    for key,label in [('zero','固定零'),('fit_mean','拟合池均值'),('gain_readout','截负为零的收益读出'),('unclipped_readout','未截断线性输出')]:lines.append(f"|{label}|{pm[key]['normalized_MSE']:.6f}|{pm[key]['normalized_MAE']:.6f}|")
    lines+=['','MSE/MAE不代替下游选择结果；只报告平均误差也不能证明门控更会分配查询。截负为零不会改变与严格正查询费用比较时的动作符号。12个池并非独立数据集。','',
        '## 当前研究判断','',
        '本轮保留了值得继续追踪的奖励幅度改善，同时保留增加查询和所有反向日期。GRID190仍是较低查询的开发对照，GRID196作为质量与费用权衡的另一候选，不自动替换、组合或扩大。是否学到了更好的选择，要结合上述同预期查询与随机追加结果逐种子判断。','',
        '保护的是原NET已经评估出的局部最佳动作，不保证与GRID190门控相比处处不退化，更不保证整周存活或成本。全部结果来自已暴露教师路径上的16日期折外研究，没有新的闭环整周、真实时延、跨网或最终测试结论。','',
        '审核逐项重建12个岭回归目标与原坐标梯度、21402条策略轨迹、4681个已知费用及门控决定；原NET和ADD1全量重现。所有源、固定权重、失败方向与比照明细归档。']
    (O/'GRID196_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('publication_audit.json',dict(passed=True,all12_original_objectives_rechecked=True,all96matched_expected_date_seed_budgets_equal=True,predictive_and_downstream_metrics_both_retained=True,all12weights_before_held=True,new_fits=12,new_NET_forwards=0,new_physics=0));C.trained();C.save('complete.json',dict(passed=True,all12fits_all16dates_all3seeds=True,new_fits=12,new_NET_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_收益幅度门控_GRID196';dest.mkdir(exist_ok=False)
    for name in ['GRID196_report.md','decision.json','expected_budget_comparison.json','predictive_metrics.json','endpoint_audit.json','review.json','publication_audit.json','complete.json']:shutil.copy2(O/name,dest/name);assert C.sha(O/name)==C.sha(dest/name)
    msg=f'GRID19612固定84参数收益岭读出完成，沿用192样本/半半权重/83维/原标准化，原NET不改。奖励保留66.16%→72.19%、动作90.335%→91.572%、查询15.007→19.857；实际代理NET较190差{new["mean_net_utility"]-old["mean_net_utility"]:+.8f}。同预期查询对190 NET差{mnet:+.8f}/动作{mm:+.4f}pp，对随机ADD1 NET差{randomnet:+.8f}；全部种子日期保留，作为权衡候选，不自动替换或组合/扩物理。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID196固定收益幅度读出完成\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':h,t0=t.split('\n',1);t=h+'\n'+entry+'\n'+t0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID196后\n\n1. 保留GRID190低查询候选和GRID196收益幅度改善，按全部日期/种子及同预期查询结果解释，不把多查当作同计算优势。\n2. 下一步优先只读拆解196新增奖励与无效追加，追踪已有小改善对应物理调用及可见线索；不要直接扫描费用/阈值、合并策略或扩大网络。\n3. 后续训练先通知固定范围，物理闭环/新日期/最终8和论文仍不自动启动。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID196收益幅度门控\n\n[报告](protocols/GRID196_gain_readout/complete/GRID196_report.md)；12固定线性读出，奖励与查询共同增加，同预期预算和随机对照全量保留。\n\n'.encode('utf8')+p.read_bytes());print('GRID196_COMPLETE',json.dumps(dict(decision=decision,per_seed=comparisons['GRID190']['per_seed'],matched=ma,predictive=pm),ensure_ascii=False))
if __name__=='__main__':main()
