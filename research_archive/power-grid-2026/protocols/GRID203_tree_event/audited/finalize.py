"""Prespecified held metrics and internal expected-query comparisons; zero fits."""
from pathlib import Path
import collections,json,pickle,shutil,sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent))
import run as C
R,O,read=C.R,C.O,C.read

def hierarchy(rows):
    ds={r['date'] for r in rows};gd=collections.defaultdict(set);gs=collections.defaultdict(set)
    for r in rows:gd[r['date']].add(r['physical_state_group']);gs[r['physical_state_group']].add(r['sample_id'])
    w=np.array([1/(len(ds)*len(gd[r['date']])*len(gs[r['physical_state_group']])) for r in rows]);assert abs(w.sum()-1)<1e-10
    return w

def profile(p,mu,y,w):
    yes=y>0;positive=w[yes]/w[yes].sum();pc=np.clip(p,1e-15,1-1e-15)
    return dict(event_AUC=float(roc_auc_score(yes,p,sample_weight=w)),event_Brier=float(w@((p-yes)**2)),event_BCE=float(-w@(yes*np.log(pc)+(~yes)*np.log1p(-pc))),actual_event_mean=float(w@yes),predicted_event_mean=float(w@p),expected_gain_MSE=float(w@((p*mu-y)**2)),expected_gain_MAE=float(w@abs(p*mu-y)),positive_conditional_MSE=float(positive@((mu[yes]-y[yes])**2)),mean_expected_gain=float(w@(p*mu)),mean_actual_gain=float(w@y),gain_weighted_event_probability=float(w@(y*p)/(w@y)))

def main():
    C.trained();assert read(O/'review.json')['passed'] and not (O/'complete.json').exists()
    task=read(R/'outputs/grid102_search_rl/design.json');fee,scale=task['cost_per_query'],task['gain_scale']
    pred=read(O/'gate_predictions.json');prev={(r['seed'],r['sample_id']):r for r in read(C.OLD/'gate_predictions.json')};pools=[];checks=[]
    for rec in read(O/'fits.json')['records']:
        f,s=rec['fold'],rec['seed'];name=f'f{f}_s{s}';model=pickle.loads((O/'models'/f'{name}_event.pkl').read_bytes());amount=pickle.loads((O/'models'/f'{name}_amount.pkl').read_bytes());z=np.load(C.FIT/'fit_features'/f'{name}.npz');lab=np.load(C.LAB/f'{name}.npz');x,y,w=z['standard'],z['y'],z['sample_weight']
        for k,v in C.PARAMS.items():assert model.get_params()[k]==v
        assert model.get_params()['random_state']==s and len(model.estimators_)==64 and np.array_equal(model.classes_,[0,1]);assert C.sha(O/'models'/f'{name}_amount.pkl')==C.sha(C.OLD/'models'/f'{name}.pkl')
        nodes=leaves=0;error=0.;manual=[]
        for tree in model.estimators_:
            leaf=tree.apply(x);v=tree.tree_.value[leaf,0,:];manual.append(v[:,1]/v.sum(axis=1));assert tree.tree_.max_depth<=4;nodes+=tree.tree_.node_count;leaves+=int(np.sum(tree.tree_.children_left==-1))
            for j in np.unique(leaf):
                mask=leaf==j;assert mask.sum()>=10;truth=float(w[mask]@y[mask]/w[mask].sum());val=tree.tree_.value[j,0,:];error=max(error,abs(truth-float(val[1]/val.sum())))
        assert error<1e-9 and nodes==rec['nodes'] and leaves==rec['leaves'];p=model.predict_proba(x)[:,1];assert np.allclose(p,np.mean(manual,axis=0),atol=1e-11,rtol=1e-10);assert abs(float(w@((p-y)**2)/w.sum())-rec['fit_Brier'])<1e-10
        n=int(z['original_STOP_count']);mu=amount.predict(x);domains={}
        for domain,idx in [('fit_STOP',np.arange(n)),('fit_CONT',np.arange(n,len(p)))]:
            ww=z['mass'][idx]/z['mass'][idx].sum();domains[domain]=profile(p[idx],mu[idx],lab['target'][idx],ww)
        pp=[r for r in pred if r['fold']==f and r['seed']==s];hw=hierarchy(pp);actual=np.array([r['actual_next_gain']/scale for r in pp]);hp=np.array([r['event_probability'] for r in pp]);hm=np.array([r['conditional_normalized_gain'] for r in pp]);assert np.allclose(hm,[prev[s,r['sample_id']]['conditional_normalized_gain'] for r in pp],atol=1e-10,rtol=1e-10);domains['held_STOP']=profile(hp,hm,actual,hw)
        pools.append(dict(fold=f,seed=s,domains=domains));checks.append(dict(fold=f,seed=s,nodes=nodes,leaves=leaves,max_leaf_error=error,amount_bytes_frozen=True))
    means={d:{k:float(np.mean([r['domains'][d][k] for r in pools])) for k in pools[0]['domains'][d]} for d in ['fit_STOP','fit_CONT','held_STOP']}
    oldpm=read(C.OLD/'predictive_metrics.json')['all12_mean'];assert abs(means['held_STOP']['positive_conditional_MSE']-oldpm['positive_conditional_MSE'])<1e-10
    C.save('predictive_metrics.json',dict(weighting='perpool date/call/start balanced;positive metrics renormalized on realized positives;12pool descriptive means not independent',BCE_clip='1e-15 metric only;native p used for decisions',mean_profiles=means,pools=pools));C.save('forest_audit.json',dict(passed=True,all12_params_leaf_probabilities_fit_Brier_and_frozen_amounts=True,pools=checks))
    _,_,cases=C.B.load();tr=read(O/'oof_trajectories.json');ni={(r['method'],r['sample_id']):r for r in tr};metrics=read(O/'search_metrics.json');family=read(O/'family_summary.json');comparisons={};actualpairs={}
    for name,folder in [('GRID201',C.OLD),('GRID198',C.LINEAR)]:
        oi={(r['method'],r['sample_id']):r for r in read(folder/'oof_trajectories.json')};om=read(folder/'search_metrics.json');matched=[];actual=[]
        for s in range(3):
            for base in metrics[f'NET_seed{s}']['per_date']:
                d=base['date'];ids=[c['sample_id'] for c in cases if c['date']==d];N=len(ids);arms={'OLD':[oi[f'GATE_seed{s}',sid] for sid in ids],'NEW':[ni[f'GATE_seed{s}',sid] for sid in ids],'ADD1_RANDOM':[ni[f'ADD1_seed{s}',sid] for sid in ids]};costs={k:sum(r['queries']-r['baseline_queries'] for r in rr) for k,rr in arms.items()};budget=min(costs['OLD'],costs['NEW']);row=dict(seed=s,date=d,matched_extra_query_total=budget,original_extra_query_totals=costs)
                for key,rr in arms.items():
                    prob=budget/costs[key] if costs[key] else 0.;assert 0<=prob<=1;gain=prob*sum(r['gain']-r['baseline_gain'] for r in rr);matches=prob*sum(r['action_match']-ni[f'NET_seed{s}',r['sample_id']]['action_match'] for r in rr);row[key]=dict(keep_probability=prob,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+gain/N,action_match_rate=base['action_match_rate']+matches/N,mean_net_utility=base['mean_net_utility']+(gain-fee*budget)/(scale*N))
                    # Direct per-start mixture independently checks macro formulas.
                    direct=[]
                    for r in rr:
                        b=ni[f'NET_seed{s}',r['sample_id']];direct.append([b['queries']+prob*(r['queries']-b['queries']),b['gain']+prob*(r['gain']-b['gain']),float(b['action_match'])+prob*(int(r['action_match'])-int(b['action_match']))])
                    val=np.mean(direct,axis=0);assert np.allclose(val,[row[key]['mean_queries'],row[key]['mean_gain'],row[key]['action_match_rate']],atol=1e-10,rtol=1e-10)
                assert len({row[k]['mean_queries'] for k in arms})==1;matched.append(row);a=next(r for r in metrics[f'GATE_seed{s}']['per_date'] if r['date']==d);b=next(r for r in om[f'GATE_seed{s}']['per_date'] if r['date']==d);actual.append(dict(seed=s,date=d,NET_delta=a['mean_net_utility']-b['mean_net_utility'],gain_delta=a['mean_gain']-b['mean_gain'],query_delta=a['mean_queries']-b['mean_queries'],match_delta_pp=100*(a['action_match_rate']-b['action_match_rate'])))
        fields=['mean_queries','mean_gain','action_match_rate','mean_net_utility'];macro={k:{field:float(np.mean([r[k][field] for r in matched])) for field in fields} for k in arms};perseed=[]
        for s in range(3):
            rr=[r for r in matched if r['seed']==s];perseed.append(dict(seed=s,NET_delta=float(np.mean([r['NEW']['mean_net_utility']-r['OLD']['mean_net_utility'] for r in rr])),gain_delta=float(np.mean([r['NEW']['mean_gain']-r['OLD']['mean_gain'] for r in rr])),match_delta_pp=100*float(np.mean([r['NEW']['action_match_rate']-r['OLD']['action_match_rate'] for r in rr])),positive_NET_dates=sum(r['NEW']['mean_net_utility']>r['OLD']['mean_net_utility'] for r in rr),negative_NET_dates=sum(r['NEW']['mean_net_utility']<r['OLD']['mean_net_utility'] for r in rr),random_NET_delta=float(np.mean([r['NEW']['mean_net_utility']-r['ADD1_RANDOM']['mean_net_utility'] for r in rr]))))
        comparisons[name]=dict(prespecified=True,analytic_expected_queries_not_hardbudget_or_time=True,macro=macro,per_seed=perseed,per_date_seed=matched);actualpairs[name]=actual
    C.save('expected_budget_comparison.json',comparisons);C.save('all_actual_date_seed_differences.json',actualpairs);nc=collections.Counter(c['date'] for c in cases);spent=waste=0.
    for p in pred:
        if p['gate_continue']:
            mass=1/(3*16*nc[p['date']]);spent+=mass*p['actual_next_queries']
            if p['actual_next_gain']==0:waste+=mass*p['actual_next_queries']
    assert abs(spent-family['GATE']['added_queries'])<1e-10
    C.save('wasted_query_audit.json',dict(date_macro_extra_queries=spent,date_macro_zero_gain_queries=waste,date_macro_zero_gain_query_fraction=waste/spent))
    base=[]
    for s in range(3):
        a,b=metrics[f'GATE_seed{s}']['macro'],metrics[f'NET_seed{s}']['macro'];base.append(dict(seed=s,NET_delta=a['mean_net_utility']-b['mean_net_utility'],query_delta=a['mean_queries']-b['mean_queries'],gain_delta=a['mean_gain']-b['mean_gain']))
    deltas={name:dict(NET_delta=v['macro']['NEW']['mean_net_utility']-v['macro']['OLD']['mean_net_utility'],match_delta_pp=100*(v['macro']['NEW']['action_match_rate']-v['macro']['OLD']['action_match_rate']),random_NET_delta=v['macro']['NEW']['mean_net_utility']-v['macro']['ADD1_RANDOM']['mean_net_utility']) for name,v in comparisons.items()}
    decision=dict(internal_learning_only=True,no_new_FRAME_comparison=True,new_event_forests=12,new_event_trees=768,amount_and_NET_fits=0,all_fixed_seeds_dates_kept=True,base_learning_per_seed=base,matched=deltas,retain_all_versions_and_counterexamples=True,no_automatic_scaling_or_parameter_search=True)
    C.save('decision.json',decision)
    new=family['GATE'];old=read(C.OLD/'family_summary.json')['GATE'];count=read(O/'gate_counts.json');pm=means['held_STOP'];lines=['# GRID203：只更换事件概率头的内部学习试验','',f"固定条件收益树和原神经网络，只将事件概率的线性读出换成浅树分类器。每起点查询{old['mean_queries']:.3f}→{new['mean_queries']:.3f}，局部奖励增量保留{100*old['gain_capture']:.2f}%→{100*new['gain_capture']:.2f}%，代理净效用{old['mean_net_utility']:.6f}→{new['mean_net_utility']:.6f}。以下给出所有种子与日期，按预定查询口径判断改善。",'', '## 方法与评价范围','', '12个固定ExtraTreesClassifier，每个64棵树、深度至多4、叶子至少10行，随机种子0/1/2对应原学习种子。类别权重为空，未按收益大小改训练权重。83维表示、半STOP半CONT样本与原分层权重不变；GRID201的12个收益森林字节一致，原神经网络未训练或扩展。新增768棵事件树，加冻结收益树共1536棵活跃树，不能声称同容量或纯校准对照。', '', '每个分类叶输出训练样本加权正例比例，64棵树平均为事件概率p。p乘冻结的正例条件平均收益μ，超过既定查询费用才追加一批至多64个候选。没有0.5事件阈值、后验调参或概率校准；BCE中1e-15仅用于计算对数，不参与决策。', '', '原16个已暴露历史训练日期做4折日期隔离，每折12日拟合、4日留出，保留三种子。4681个可追加停止点、每政策7134个种子—起点轨迹来自共享教师路径；它们不是独立场景。全部权重在留出评分前已发布。本轮无新物理仿真、神经网络前向、最终8测试、原FRAME闭环或论文改动。', '', '## 内部版本结果','', '|方法|同站完整搜索动作一致率|局部奖励增量保留率|每起点候选查询|代理净效用|','|---|---:|---:|---:|---:|']
    for label,z in [('基础学习NET',family['NET']),('GRID196直接收益',read(C.DIRECT/'family_summary.json')['GATE']),('GRID198线性概率×线性幅度',read(C.LINEAR/'family_summary.json')['GATE']),('GRID201线性概率×浅树幅度',old),('GRID203浅树概率×固定浅树幅度',new),('NET后固定一批',family['ADD1'])]:lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.9f}|")
    lines+=['',f"{count['append']}次追加中{count['append_changes']}次改选、{count['append_no_change']}次零收益、{count['append_positive_NET']}次局部奖励超过代理费用。按日期宏平均追加查询，零收益部分占{100*waste/spent:.2f}%；它与简单的零收益追加次数比例不同。",'', '## 相同预期查询量','', '逐日期和种子取双方较小追加查询数，再以仅由查询成本决定的统一概率保留各策略追加，解析计算同一预期查询量。固定一批也按相同预算稀释作随机参照。该分析已在训练前规定，是解析期望比较，不是真实硬预算执行或实际时延测试。']
    for name,v in comparisons.items():
        lines+=['',f'对{name}：','', '|策略|预期查询|预期动作一致率|预期局部奖励|预期代理净效用|','|---|---:|---:|---:|---:|']
        for k,label in [('OLD',name),('NEW','GRID203'),('ADD1_RANDOM','随机追加参照')]:
            z=v['macro'][k];lines.append(f"|{label}|{z['mean_queries']:.6f}|{100*z['action_match_rate']:.6f}%|{z['mean_gain']:.9f}|{z['mean_net_utility']:.9f}|")
        lines+=['','|种子|净效用差|奖励差|动作一致率差/百分点|净效用正/负日期|相对随机净效用差|','|---|---:|---:|---:|---:|---:|']
        for z in v['per_seed']:lines.append(f"|{z['seed']}|{z['NET_delta']:+.9f}|{z['gain_delta']:+.9f}|{z['match_delta_pp']:+.4f}|{z['positive_NET_dates']}/{z['negative_NET_dates']}|{z['random_NET_delta']:+.9f}|")
        negative=sorted((r for r in actualpairs[name] if r['NET_delta']<0),key=lambda r:r['NET_delta'])
        lines+=['',f'未经预算稀释的实际策略在48个种子—日期配对中有{len(negative)}个净效用下降；所有配对见all_actual_date_seed_differences.json。以下保留降幅最大的三项：']
        for z in negative[:3]:lines.append(f"- 种子{z['seed']}，{z['date']}：净效用{z['NET_delta']:+.8f}，查询{z['query_delta']:+.4f}，动作一致率{z['match_delta_pp']:+.4f}个百分点。")
    lines+=['','## 事件识别和幅度误差','', '|数据域|事件AUC|Brier|BCE|实际事件率|预测概率均值|总体收益MSE|正例条件收益MSE|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for d,z in means.items():lines.append(f"|{d}|{z['event_AUC']:.6f}|{z['event_Brier']:.6f}|{z['event_BCE']:.6f}|{z['actual_event_mean']:.6f}|{z['predicted_event_mean']:.6f}|{z['expected_gain_MSE']:.6f}|{z['positive_conditional_MSE']:.6f}|")
    oldevent=read(R/'outputs/grid202_factor_bottleneck/summary.json')['mean_profiles']['held_STOP'];lines+=['',f"留出事件AUC由{oldevent['event_AUC']:.6f}变为{pm['event_AUC']:.6f}，Brier由{oldevent['event_Brier']:.6f}变为{pm['event_Brier']:.6f}。收益幅度模型没有变化，正例条件MSE仍为{pm['positive_conditional_MSE']:.6f}；不将零收益样本填为条件幅度0。上述为各池分层均衡后的12池描述均值。",'', '## 与基础学习NET及后续定位','', '|种子|代理净效用差|局部奖励差|查询差|','|---|---:|---:|---:|']
    for z in base:lines.append(f"|{z['seed']}|{z['NET_delta']:+.9f}|{z['gain_delta']:+.9f}|{z['query_delta']:+.6f}|")
    lines+=['', '本轮检验的是已有表示上的概率读出方式。改进可支持继续研究这个局部学习模块，不能据此宣布完整周成本、存活、控制时延或跨电网泛化改善。原始策略和同预期查询比较分开报告；三个种子及所有反向日期完整保留，不按结果调费用、择种子、改阈值或追加训练。下一步先解释恢复和损失案例，不自动启动新的原FRAME比较。']
    (O/'GRID203_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.trained();C.save('publication_audit.json',dict(passed=True,all12_native_leaf_probabilities_and_params_rechecked=True,all96_expected_query_budgets_equal_and_direct_mixtures_reproduced=True,frozen_amount_predictions_and_conditional_MSE_reproduced=True,all_seeds_dates_counterexamples_retained=True,new_fits=0,new_NET_forwards=0,new_physics=0));C.save('complete.json',dict(passed=True,new_event_forests=12,new_event_trees=768,amount_and_NET_fits=0,all12pairs_all16dates_all3seeds=True,new_NET_forwards=0,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_浅树概率_GRID203';dest.mkdir(exist_ok=False)
    for n in ['GRID203_report.md','decision.json','expected_budget_comparison.json','predictive_metrics.json','forest_audit.json','wasted_query_audit.json','review.json','publication_audit.json','complete.json']:shutil.copy2(O/n,dest/n)
    msg=f"GRID203只换事件概率头，12固定浅ExtraTrees分类森林/768新树，GRID201幅度森林和原NET冻结。查询{old['mean_queries']:.3f}→{new['mean_queries']:.3f}、奖励保留{100*old['gain_capture']:.2f}%→{100*new['gain_capture']:.2f}%；同预期查询对201净效用{deltas['GRID201']['NET_delta']:+.8f}，对198{deltas['GRID198']['NET_delta']:+.8f}。保留全部种子反例、只作教师路径内部改进，不称完整控制/新FRAME优势。"
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-27 — GRID203浅树事件概率读出完成\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':h,rest=t.split('\n',1);t=h+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID203后\n\n1. 继续学习模块内部改进，先只读追踪新旧概率头的恢复、损失与查询开销；全部种子/日期保留。\n2. 区分概率排序与校准、当前局部奖励与长期质量；不凭总体改善改费用、标签或门槛，不自动训练、扩大或新FRAME闭环。\n3. 下一训练须先通知固定范围；最终8、物理环境与论文不动。\n',encoding='utf8',newline='\n');p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID203浅树事件概率内部试验\n\n[报告](protocols/GRID203_tree_event/complete/GRID203_report.md)；冻结条件幅度、保留全部反例，不新FRAME闭环。\n\n'.encode('utf8')+p.read_bytes())
    print('GRID203_COMPLETE',json.dumps(dict(decision=decision,matched_seed={k:v['per_seed'] for k,v in comparisons.items()},predictive=means,waste=waste/spent),ensure_ascii=False))

if __name__=='__main__':main()
