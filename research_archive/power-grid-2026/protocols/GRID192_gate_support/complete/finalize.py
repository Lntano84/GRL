"""Audit support intervention and prespecified expected-query comparison."""
from pathlib import Path
import collections,pickle,shutil,sys
import numpy as np
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;OLD=C.OLD;read=C.read
def main():
    C.bind();C.B.trained();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();fit=read(O/'fits.json');pools=read(O/'preflight.json')['pools'];oldtr=read(OLD/'oof_trajectories.json');newtr=read(O/'oof_trajectories.json');ob={(v['method'],v['sample_id']):v for v in oldtr};nb={(v['method'],v['sample_id']):v for v in newtr};oldm=read(OLD/'search_metrics.json');newm=read(O/'search_metrics.json');oldf=read(OLD/'family_summary.json');newf=read(O/'family_summary.json');task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale'];_,lookup,cases=C.B.load();nc=collections.Counter(c['date'] for c in cases);cm={c['sample_id']:c for c in cases};diagnostics=[];checks=0
    for pool in pools:
        f=pool['fold'];s=pool['seed'];z=np.load(O/'fit_features'/f'f{f}_s{s}.npz');oz=np.load(OLD/'fit_features'/f'f{f}_s{s}.npz');cp=np.load(O/'models'/f'f{f}_s{s}.npz');n=int(z['original_STOP_count']);assert n==len(oz['y'])
        assert np.array_equal(z['mean'],oz['mean']) and np.array_equal(z['scale'],oz['scale']);assert np.array_equal(z['x'][:n],oz['x']);assert np.array_equal(z['y'][:n],oz['y']);assert np.allclose(z['sample_weight'][:n],.5*oz['sample_weight'],atol=1e-12);assert abs(z['sample_weight'].sum()-n)<1e-8
        assert not set(z['ids'][:n].tolist())&set(z['ids'][n:].tolist());a=z['standard'];y=z['y'];w=z['sample_weight'];logit=a@cp['coef'][0]+cp['intercept'][0];p=np.exp(-np.logaddexp(0.,-logit));grad=a.T@(w*(p-y))/w.sum()+cp['coef'][0]/w.sum();gb=float(w@(p-y)/w.sum());rec=next(v for v in fit['records'] if v['fold']==f and v['seed']==s)
        stoplogit=oz['standard']@cp['coef'][0]+cp['intercept'][0];diagnostics.append(dict(fold=f,seed=s,converged=rec['converged'],n_iter=rec['n_iter'],objective_gradient_infinity=max(float(abs(grad).max()),abs(gb)),max_abs_STOP_standard=float(abs(a[:n]).max()),max_abs_CONT_standard=float(abs(a[n:]).max()),STOP_fit_AUC=float(roc_auc_score(oz['y'],stoplogit,sample_weight=oz['mass'])),normalizer_and_L2_total_weight_match=True));checks+=1
    # Both old/new deterministic extra-batch decisions are thinned independently by cost only.
    matched=[]
    for seed in range(3):
        for base in newm[f'NET_seed{seed}']['per_date']:
            d=base['date'];ids=[c['sample_id'] for c in cases if c['date']==d];old=[ob[f'GATE_seed{seed}',sid] for sid in ids];new=[nb[f'GATE_seed{seed}',sid] for sid in ids];add=[nb[f'ADD1_seed{seed}',sid] for sid in ids];N=len(ids);qs=[sum(v['queries']-v['baseline_queries'] for v in rr) for rr in [old,new]];budget=min(qs);row=dict(seed=seed,date=d,matched_extra_query_total=budget,old_extra_query_total=qs[0],new_extra_query_total=qs[1])
            for key,rr in [('OLD',old),('MIX',new),('ADD1_RANDOM',add)]:
                q=sum(v['queries']-v['baseline_queries'] for v in rr);retain=budget/q if q else 0.;gain=retain*sum(v['gain']-v['baseline_gain'] for v in rr);recover=retain*sum(v['action_match']-nb[f'NET_seed{seed}',v['sample_id']]['action_match'] for v in rr)
                row[key]=dict(keep_probability=retain,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+gain/N,action_match_rate=base['action_match_rate']+recover/N,mean_net_utility=base['mean_net_utility']+(gain-fee*budget)/(scale*N),added_gain=gain/N)
            assert row['OLD']['mean_queries']==row['MIX']['mean_queries']==row['ADD1_RANDOM']['mean_queries'];matched.append(row)
    macro={k:{field:float(np.mean([r[k][field] for r in matched])) for field in ['mean_queries','mean_gain','action_match_rate','mean_net_utility','added_gain']} for k in ['OLD','MIX','ADD1_RANDOM']};perseed=[]
    for s in range(3):
        rr=[v for v in matched if v['seed']==s];perseed.append(dict(seed=s,NET_delta=float(np.mean([v['MIX']['mean_net_utility']-v['OLD']['mean_net_utility'] for v in rr])),match_delta_pp=100*float(np.mean([v['MIX']['action_match_rate']-v['OLD']['action_match_rate'] for v in rr])),positive_NET_dates=sum(v['MIX']['mean_net_utility']>v['OLD']['mean_net_utility'] for v in rr)))
    C.save('expected_budget_comparison.json',dict(prespecified_in_design=True,method='Uniformindependentthinning atmin old/new candidatequery total perdate/seed;no reward in probability;analytic expectation,not actualhardbudget/walltime',per_date_seed=matched,macro=macro,per_seed=perseed))
    contribution=collections.defaultdict(lambda:dict(pairs=0,query_delta=0.,gain_delta=0.,groups=set()));novel=[]
    for s in range(3):
        for c in cases:
            sid=c['sample_id'];old=ob[f'GATE_seed{s}',sid];new=nb[f'GATE_seed{s}',sid];base=nb[f'NET_seed{s}',sid];assert old['baseline_stop']==new['baseline_stop'];assert base['queries']<=new['queries'] and base['gain']<=new['gain'];assert not base['action_match'] or new['action_match']
            if old['append']==new['append']:key='BOTH_APPEND' if new['append'] else 'BOTH_SKIP';assert old['gain']==new['gain'] and old['queries']==new['queries'] and old['chosen_pool_id']==new['chosen_pool_id']
            elif new['append']:key='NEW_ONLY_PRODUCTIVE' if new['chosen_pool_id']!=base['chosen_pool_id'] else 'NEW_ONLY_NO_CHANGE'
            else:key='DROP_OLD_PRODUCTIVE' if old['chosen_pool_id']!=base['chosen_pool_id'] else 'DROP_OLD_NO_CHANGE'
            w=1/(3*16*nc[c['date']]);v=contribution[key];v['pairs']+=1;v['query_delta']+=w*(new['queries']-old['queries']);v['gain_delta']+=w*(new['gain']-old['gain']);v['groups'].add(c['physical_state_group'])
            if key=='NEW_ONLY_PRODUCTIVE':novel.append(dict(seed=s,sample_id=sid,date=c['date'],group=c['physical_state_group'],added_gain=new['gain']-base['gain'],added_queries=new['queries']-base['queries']))
            # Independently rebuild unchanged fixed-candidate outcomes.
            end=new['batches'];assert new['chosen_pool_id']==c['chosen'][end] and new['queries']==int(c['queries'][:end].sum()) and new['gain']==float(c['gains'][:end].sum())
    for v in contribution.values():v['physical_groups']=len(v.pop('groups'))
    assert abs(sum(v['gain_delta'] for v in contribution.values())-(newf['GATE']['mean_gain']-oldf['GATE']['mean_gain']))<1e-12;assert sum(v['pairs'] for v in contribution.values())==7134
    C.save('paired_case_accounting.json',dict(categories=dict(contribution),all_new_only_productive=novel));C.save('optimization_and_scale.json',dict(pools=diagnostics,all12kept=True,no_retry_or_iteration_extension=True))
    old=oldf['GATE'];new=newf['GATE'];base=newf['NET'];converged=sum(v['converged'] for v in fit['records']);match_delta=100*(new['action_match_rate']-old['action_match_rate']);net_delta=macro['MIX']['mean_net_utility']-macro['OLD']['mean_net_utility'];match_matched=100*(macro['MIX']['action_match_rate']-macro['OLD']['action_match_rate'])
    decision=dict(completed_fixed12fits=True,all12included=True,converged=converged,not_converged=12-converged,matched_query_NET_delta_vs190=net_delta,matched_query_match_delta_pp_vs190=match_matched,retain_GRID190_as_current_candidate=True,support_hypothesis_not_refuted_in_general=True,no_extra_retraining=True,no_physics=True)
    C.save('decision.json',decision)
    lines=['# GRID192：补充原NET继续状态，能否改善单批门控','',
        f"固定的12个同容量门控训练与评价完成。混合覆盖模型在全部16日期/3种子中，完整同站动作一致率{100*new['action_match_rate']:.3f}%，平均查询{new['mean_queries']:.3f}；原GRID190门控为{100*old['action_match_rate']:.3f}%和{old['mean_queries']:.3f}。新模型查得更少、质量增量也更小，需结合下方预先定义的同预期查询比较。当前保留GRID190作为候选版本，不用本轮替换它。",'',
        f"{converged}/12份模型在原定2000次上限内收敛，另{12-converged}份达到上限；全部保留、未延长迭代、未补训或择种子。因此本轮不是完全收敛条件下对覆盖假设的最终否定。",'',
        '## 只改变了什么','',
        '门控仍是83维冻结NET表示和84个逻辑回归参数，C=1/L2/lbfgs/阈值0.5不变，原NET参数和执行路径不变。训练分布固定为一半原STOP域、一半原NET实际停止前的全部CONTINUE域；补充样本包含正、负两类，不按未来收益挑选，也没有加入停止后的教师路径。','',
        '原STOP侧标准化完全复用；训练总权重仍等于原STOP样本数，保持正则尺度。补充域内部按日期/物理组/起点/继续步均衡，不是新增正例类别权重。混合改变了输入分布及对应标签先验，门控读数不能直接当作STOP域的校准概率。没有增加电网数据或模型容量。','',
        f"拟合侧每组增加{min(v['CONT_rows'] for v in pools)}–{max(v['CONT_rows'] for v in pools)}个继续状态。12份权重全部上传后才评价门控；留出83维表示复用GRID190已保存且冻结的数组，没有新的神经网络前向或电网预测。",'',
        '## 全量局部结果','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点查询|原NET效用|','|---|---:|---:|---:|---:|']
    for name,z in [('原NET',base),('原STOP门控（GRID190）',old),('混合覆盖门控',new),('停止后无条件ADD1',newf['ADD1'])]:lines.append(f"|{name}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.6f}|")
    lines+=['','两个门控都保留了原NET局部最佳选择不下降的边界。这不保证整周成本或存活：只是在同一教师状态与固定候选上追加一次，不删除原查询。','',
        '## 预先定义的同预期追加查询量比较','',
        '每个日期、每个种子取新旧门控较小的追加候选查询量；分别对各方法已经选出的追加批次以统一概率独立保留，使追加候选查询量的数学期望一致。概率只由查询量确定，不由奖励或动作正确与否决定。无条件ADD1以同一方法稀释，作为随机追加参照。没有抽样种子或实际物理运行。','',
        '|策略|每起点候选查询（期望）|完整同站动作一致率（期望）|每起点奖励增量（期望）|NET效用（期望）|','|---|---:|---:|---:|---:|']
    for key,label in [('OLD','原STOP门控'),('MIX','混合覆盖门控'),('ADD1_RANDOM','随机追加参照')]:
        z=macro[key];lines.append(f"|{label}|{z['mean_queries']:.6f}|{100*z['action_match_rate']:.6f}%|{z['mean_gain']:.9f}|{z['mean_net_utility']:.9f}|")
    lines+=['',f"同预期候选查询下，混合覆盖相对原门控的动作一致率变化为{match_matched:+.4f}个百分点，NET效用变化{net_delta:+.8f}。这是预定义的离线数学期望比较，不是一次运行的硬预算、门控推理时间或完整控制时间匹配。",'',
        '|种子|同预期查询NET效用差|动作一致率差（百分点）|NET差为正的日期数|','|---|---:|---:|---:|']
    for v in perseed:lines.append(f"|{v['seed']}|{v['NET_delta']:+.8f}|{v['match_delta_pp']:+.4f}|{v['positive_NET_dates']}/16|")
    lines+=['','全部48个日期—种子记录保留；不按是否收敛、某个种子或某个日期重新筛选主结果。','',
        '## 哪些选择发生了变化','',
        '|新旧门控的差异|配对数|物理组数|查询变化贡献|奖励变化贡献|','|---|---:|---:|---:|---:|']
    label=dict(BOTH_APPEND='双方都追加',BOTH_SKIP='双方都跳过',NEW_ONLY_PRODUCTIVE='新模型独有且改选',NEW_ONLY_NO_CHANGE='新模型独有但未改选',DROP_OLD_PRODUCTIVE='放弃原模型有效追加',DROP_OLD_NO_CHANGE='省掉原模型无效追加')
    for k,v in sorted(contribution.items()):lines.append(f"|{label[k]}|{v['pairs']}|{v['physical_groups']}|{v['query_delta']:+.6f}|{v['gain_delta']:+.9f}|")
    lines+=['','新模型独有的改善也完整保留在paired_case_accounting.json中，不因整体不足而删除。它们目前是同一开发数据上的互补线索，不据此自动组合两个门控。','',
        '## 迭代上限与尺度','',
        '两份达到上限的模型属于原固定预算结果，不算软件崩溃，也不能宣称已经优化到收敛。以下同时保留各池继续状态相对原STOP标准化的最大绝对值与最终目标梯度，帮助区分数值优化限制和统计迁移问题；不能据此直接延长预算救结果。','',
        '|折/种子|迭代次数|收敛|STOP标准化最大绝对值|CONT标准化最大绝对值|目标梯度无穷范数|','|---|---:|---|---:|---:|---:|']
    for v in diagnostics:lines.append(f"|{v['fold']}/{v['seed']}|{v['n_iter']}|{v['converged']}|{v['max_abs_STOP_standard']:.3g}|{v['max_abs_CONT_standard']:.3g}|{v['objective_gradient_infinity']:.3g}|")
    lines+=['','## 当前判断','',
        'GRID191确认了条件停止样本与留出停止样本的差异，但更多拟合覆盖并不自动转化为更好的门控选择。本轮只测试这一份固定半半混合配置，同时存在两份未收敛终点；不能由此否定所有支持补充方法，也不继续扫描混合比例、改门槛或扩大网络。','',
        '原GRID190的小而跨种子重复的改善仍保留。下一步应先看新旧互补实例和标准化尺度是否给出可验证的解释，再定下一次最小干预；不能把更多追加带来的质量变化当成学习选择变好。','',
        '## 核验与范围','',
        '12份模型的原STOP样本、83维标准化、总目标权重均逐项匹配；同一案例上原NET、ADD1重现，原NET查询与局部奖励边界保持；所有新旧选择差异及查询/奖励贡献守恒。同预期预算比较的查询逐日期、种子相等。旧GRID190全部模型和报告未改。','',
        '当前是已暴露同电网的保存教师路径折外研究；没有新的物理闭环、实际时延/整周成本或最终测试结果。']
    (O/'GRID192_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n');C.save('publication_audit.json',dict(passed=True,all12_normalization_L2weight_and_originalSTOProws_match=True,all7134oldnew_pairs_reconstructed=True,matched_expected_queries_equal_all48=True,all_model_convergence_statuses_retained=True,new_fits=12,new_physics=0,new_held_NET_forwards=0));C.verify();C.save('complete.json',dict(passed=True,execution_and_record_audit_complete=True,new_gate_fits=12,converged=converged,iteration_limited=12-converged,all12before_held_scoring=True,all16dates_all3seeds=True,new_physics=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_门控覆盖对照_GRID192';dest.mkdir(exist_ok=False)
    for n in ['GRID192_report.md','decision.json','expected_budget_comparison.json','paired_case_accounting.json','optimization_and_scale.json','review.json','publication_audit.json','complete.json']:shutil.copy2(O/n,dest/n);assert C.sha(O/n)==C.sha(dest/n)
    msg=f'GRID192固定12个半STOP/半实际CONT域同84参数门控完成，10收敛/2达2000上限全保留，原NET/STOP标准化/总权重/C1/.5不改。新动作89.918%/查询14.402，原190为90.335%/15.007；同预期候选查询新旧NET差{net_delta:+.8f}、动作差{match_matched:+.4f}pp。新旧独有好例保留，不混成新策略；保留GRID190当前版本，不据单次配置否定覆盖假设，不补训/扫描参数/物理。'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/n;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID192同容量覆盖干预完成\n\n'+msg+'\n'
        if n=='RESEARCH_STATE.md':a0,b0=t.split('\n',1);t=a0+'\n'+entry+'\n'+b0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID192后\n\n1. 保留GRID190当前门控的小幅重复改善；GRID192覆盖干预全量记录和2个迭代受限终点保留，不择种子、不自动补训或扫描混合比例。\n2. 先只读解释新旧互补案例、CONT相对STOP归一尺度及数值优化影响；在同预期候选预算下比较，不把质量改变等同选择更优。\n3. 新拟合前通知固定范围；原NET不改，不自动新物理、数据、最终8或论文。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID192同容量覆盖干预\r\n\r\n[报告](protocols/GRID192_gate_support/complete/GRID192_report.md)；12固定门控完整保留，含2个迭代上限；当前继续保留GRID190，未新物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID192_COMPLETE',C.json.dumps(dict(decision=decision,matched=macro,novel_productive=len(novel),cases=dict(contribution))))
if __name__=='__main__':main()
