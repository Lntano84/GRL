"""Independent endpoint audit and complete, non-selective numerical-control report."""
from pathlib import Path
import collections, shutil, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import run as C
R=C.R; O=C.O; read=C.read
OLD=R/'outputs/grid190_extension_gate'
def main():
    C.bind(); C.verify(); C.B.trained()
    assert read(O/'review.json')['passed'] and not (O/'complete.json').exists()
    fits=read(O/'fits.json'); diagnostics=[]
    for rec in fits['records']:
        f,s=rec['fold'],rec['seed']; name=f'f{f}_s{s}'
        z=np.load(O/'fit_features'/f'{name}.npz'); prior=np.load(C.SRC/'fit_features'/f'{name}.npz')
        assert set(z.files)==set(prior.files)
        for key in z.files: assert np.array_equal(z[key],prior[key]),(name,key)
        cp=np.load(O/'models'/f'{name}.npz'); oldcp=np.load(C.SRC/'models'/f'{name}.npz')
        for key in ['mean','scale']: assert np.array_equal(cp[key],oldcp[key])
        x,y,w=z['standard'],z['y'],z['sample_weight']; mass=w.sum()
        def direct(c):
            beta=c['coef'][0]; b=c['intercept'][0]; logit=x@beta+b
            p=np.exp(-np.logaddexp(0.,-logit)); e=w*(p-y)/mass
            val=float(w@(np.logaddexp(0.,logit)-y*logit)/mass+beta@beta/(2*mass))
            grad=np.r_[x.T@e+beta/mass,e.sum()]
            return val,float(abs(grad).max())
        val,grad=direct(cp); prev,prevgrad=direct(oldcp)
        assert abs(val-rec['objective'])<1e-9
        assert np.isclose(grad,rec['original_gradient_infinity'],rtol=1e-6,atol=1e-8)
        assert (grad<=1e-4)==rec['converged']
        diagnostics.append(dict(fold=f,seed=s,n_iter=rec['n_iter'],certified=rec['converged'],objective=val,original_gradient_infinity=grad,GRID192_objective=prev,GRID192_gradient_infinity=prevgrad,objective_delta=val-prev))
    C.save('endpoint_audit.json',dict(passed=True,unchanged_data_labels_weights_normalizer_all12=True,direct_original_objective_and_gradient_match=True,all_pools=diagnostics))
    oldtr=read(OLD/'oof_trajectories.json'); newtr=read(O/'oof_trajectories.json')
    ob={(v['method'],v['sample_id']):v for v in oldtr}; nb={(v['method'],v['sample_id']):v for v in newtr}
    newm=read(O/'search_metrics.json'); _,_,cases=C.B.load(); nc=collections.Counter(c['date'] for c in cases)
    task=read(R/'outputs/grid102_search_rl/design.json'); fee,scale=task['cost_per_query'],task['gain_scale']
    matched=[]
    for seed in range(3):
        for base in newm[f'NET_seed{seed}']['per_date']:
            date=base['date']; ids=[c['sample_id'] for c in cases if c['date']==date]; N=len(ids)
            arms={'OLD190':[ob[f'GATE_seed{seed}',i] for i in ids], 'NEW193':[nb[f'GATE_seed{seed}',i] for i in ids], 'ADD1_RANDOM':[nb[f'ADD1_seed{seed}',i] for i in ids]}
            budgets={k:sum(v['queries']-v['baseline_queries'] for v in rr) for k,rr in arms.items()}
            budget=min(budgets['OLD190'],budgets['NEW193']); row=dict(seed=seed,date=date,matched_extra_query_total=budget,original_extra_totals=budgets)
            for key,rr in arms.items():
                prob=budget/budgets[key] if budgets[key] else 0.
                gain=prob*sum(v['gain']-v['baseline_gain'] for v in rr)
                recover=prob*sum(v['action_match']-nb[f'NET_seed{seed}',v['sample_id']]['action_match'] for v in rr)
                row[key]=dict(keep_probability=prob,mean_queries=base['mean_queries']+budget/N,mean_gain=base['mean_gain']+gain/N,action_match_rate=base['action_match_rate']+recover/N,mean_net_utility=base['mean_net_utility']+(gain-fee*budget)/(scale*N))
            assert len({row[k]['mean_queries'] for k in arms})==1
            matched.append(row)
    keys=['mean_queries','mean_gain','action_match_rate','mean_net_utility']
    macro={k:{v:float(np.mean([r[k][v] for r in matched])) for v in keys} for k in arms}
    perseed=[dict(seed=s,NET_delta=float(np.mean([r['NEW193']['mean_net_utility']-r['OLD190']['mean_net_utility'] for r in matched if r['seed']==s])),match_delta_pp=100*float(np.mean([r['NEW193']['action_match_rate']-r['OLD190']['action_match_rate'] for r in matched if r['seed']==s]))) for s in range(3)]
    C.save('expected_budget_comparison.json',dict(prespecified=True,method='Cost-only uniform independent thinning per date and seed to smaller extra-query budget; analytic expectation, not hard budget or wall time',per_date_seed=matched,macro=macro,per_seed=perseed))
    categories=collections.defaultdict(lambda:dict(pairs=0,query_delta=0.,gain_delta=0.,groups=set())); novel=[]
    for s in range(3):
        for c in cases:
            sid=c['sample_id']; old=ob[f'GATE_seed{s}',sid]; new=nb[f'GATE_seed{s}',sid]; base=nb[f'NET_seed{s}',sid]
            assert old['baseline_stop']==new['baseline_stop']
            assert base['queries']<=new['queries'] and base['gain']<=new['gain']
            assert not base['action_match'] or new['action_match']
            if old['append']==new['append']:
                key='BOTH_APPEND' if new['append'] else 'BOTH_SKIP'
                assert old['queries']==new['queries'] and old['gain']==new['gain'] and old['chosen_pool_id']==new['chosen_pool_id']
            elif new['append']: key='NEW_ONLY_PRODUCTIVE' if new['chosen_pool_id']!=base['chosen_pool_id'] else 'NEW_ONLY_NO_CHANGE'
            else:key='DROP_OLD_PRODUCTIVE' if old['chosen_pool_id']!=base['chosen_pool_id'] else 'DROP_OLD_NO_CHANGE'
            v=categories[key]; weight=1/(3*16*nc[c['date']]); v['pairs']+=1; v['query_delta']+=weight*(new['queries']-old['queries']); v['gain_delta']+=weight*(new['gain']-old['gain']); v['groups'].add(c['physical_state_group'])
            if key=='NEW_ONLY_PRODUCTIVE': novel.append(dict(seed=s,sample_id=sid,date=c['date'],group=c['physical_state_group'],added_gain=new['gain']-base['gain'],added_queries=new['queries']-base['queries']))
            end=new['batches']; assert new['chosen_pool_id']==c['chosen'][end] and new['queries']==int(c['queries'][:end].sum()) and new['gain']==float(c['gains'][:end].sum())
    for v in categories.values(): v['physical_groups']=len(v.pop('groups'))
    oldf=read(OLD/'family_summary.json'); newf=read(O/'family_summary.json'); mix=read(C.SRC/'family_summary.json')
    assert sum(v['pairs'] for v in categories.values())==7134
    assert abs(sum(v['gain_delta'] for v in categories.values())-newf['GATE']['mean_gain']+oldf['GATE']['mean_gain'])<1e-12
    bystart=collections.defaultdict(set)
    for v in novel: bystart[v['sample_id']].add(v['seed'])
    C.save('paired_case_accounting.json',dict(categories=dict(categories),all_new_only_productive=novel,all3seed_new_productive_starts=sum(len(s)==3 for s in bystart.values())))
    cert=sum(r['converged'] for r in fits['records']); delta=macro['NEW193']['mean_net_utility']-macro['OLD190']['mean_net_utility']; md=100*(macro['NEW193']['action_match_rate']-macro['OLD190']['action_match_rate'])
    decision=dict(retain_GRID190_as_current_candidate=True,completed_fixed12_fromzero_fits=True,certified=cert,iteration_limited=12-cert,matched_expected_query_NET_delta_vs190=delta,matched_expected_query_match_delta_pp_vs190=md,no_automatic_more_fits_or_combination=True,numerical_limit_not_sole_explanation_for_current_gap=True,not_general_refutation_of_support_augmentation=True)
    C.save('decision.json',decision)
    lines=['# GRID193：保持目标函数不变的数值优化对照','',
        '结论：本轮数值优化改善了收敛情况，但没有把混合覆盖门控提升到GRID190的水平。保留GRID190已观察到的小幅、跨种子重复改善；本轮全部模型、未收敛终点和独有改善案例保留。','',
        '## 改了什么','',
        'GRID192增加实际继续状态后，部分特征在原STOP标准化下达到约209万，另有两份模型在2000次上限内未收敛。本轮只检验数值优化是否解释其效果不足：相同样本、标签、权重、标准化、正则强度与0.5阈值，重新从零拟合12份同为84参数的逻辑回归。原NET不变。','',
        '使用拟合侧二阶尺度作变量变换，同时将L2惩罚按变量变换还原，因此优化的原目标函数不变。收敛由原坐标梯度无穷范数≤0.0001判定，不能用缩放后的小梯度冒充收敛。12组目标及梯度等价、96处有限差分均核验通过。','',
        f'固定2000次上限下，{cert}/12份满足原梯度标准；fold3/seed0达到上限，原梯度约0.000396，原样保留。GRID192先前受限的fold0/seed2与fold2/seed0本轮均达到标准。未续跑、重试或选种子。12份最终权重全部发布后才评价留出决策。','',
        '## 局部质量与查询','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点候选查询|NET效用|','|---|---:|---:|---:|---:|']
    for label,z in [('原NET',newf['NET']),('STOP门控 GRID190',oldf['GATE']),('混合覆盖 GRID192',mix['GATE']),('相同目标数值对照 GRID193',newf['GATE']),('停止后固定追加一批',newf['ADD1'])]: lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{z['mean_net_utility']:.6f}|")
    lines+=['','所有数值均覆盖16个历史训练日期的折外预测、三个种子。动作参照是完整同站搜索的选择，奖励增量保留率不是整周成本。NET效用包含固定查询代理费用，不是实际控制时间。','',
        '## 同预期候选查询比较','',
        '按预先方案，在每日期、种子内，将新旧门控的追加批次按统一概率保留，使预期追加候选查询量相同。保留概率仅由查询数确定；固定追加一批按相同方式稀释作随机参照。以下为解析期望，未运行新的随机策略，不代表硬预算或同实际时间。','',
        '|方法|每起点预期查询|预期动作一致率|预期奖励增量|预期NET效用|','|---|---:|---:|---:|---:|']
    for key,label in [('OLD190','GRID190'),('NEW193','GRID193'),('ADD1_RANDOM','随机追加参照')]:
        v=macro[key]; lines.append(f"|{label}|{v['mean_queries']:.6f}|{100*v['action_match_rate']:.6f}%|{v['mean_gain']:.9f}|{v['mean_net_utility']:.9f}|")
    lines += ['',f'与GRID190相比，同预期查询下动作一致率变化{md:+.4f}个百分点、NET效用变化{delta:+.8f}。因此不能把查询较少理解为选择更准。','','|种子|预期NET效用差|预期动作一致率差（百分点）|','|---|---:|---:|']
    for v in perseed:lines.append(f"|{v['seed']}|{v['NET_delta']:+.8f}|{v['match_delta_pp']:+.4f}|")
    lines+=['','## 新旧选择差异','',
        '|差异|种子—起点配对数|物理组数|查询变化贡献|奖励变化贡献|','|---|---:|---:|---:|---:|']
    labels=dict(BOTH_APPEND='双方追加',BOTH_SKIP='双方跳过',NEW_ONLY_PRODUCTIVE='本轮独有且改选',NEW_ONLY_NO_CHANGE='本轮独有但未改选',DROP_OLD_PRODUCTIVE='放弃原有效追加',DROP_OLD_NO_CHANGE='省去原无效追加')
    for k,v in sorted(categories.items()):lines.append(f"|{labels[k]}|{v['pairs']}|{v['physical_groups']}|{v['query_delta']:+.6f}|{v['gain_delta']:+.9f}|")
    lines+=['',f"本轮独有的有效追加共{len(novel)}个种子—起点配对；全部保存在paired_case_accounting.json中，三个种子均出现的独有改选起点为{sum(len(s)==3 for s in bystart.values())}个。保留这些线索，也保留放弃旧改善的反例；不按事后结果组合模型。",'',
        '## 原目标函数终点核验','',
        '|折/种子|本轮迭代|原梯度标准达标|GRID192目标值|GRID193目标值|本轮原梯度无穷范数|','|---|---:|---|---:|---:|---:|']
    for v in diagnostics:lines.append(f"|{v['fold']}/{v['seed']}|{v['n_iter']}|{v['certified']}|{v['GRID192_objective']:.9f}|{v['objective']:.9f}|{v['original_gradient_infinity']:.6g}|")
    lines+=['','## 对下一步的判断','',
        '数值优化受限不能单独解释目前的迁移落差：先前两份受限模型已达标准，而整体质量提升仍较小。剩余一份受限也不能隐去，因此这里只判断这一固定配置，不能宣称所有覆盖补充都无效。','',
        '当前优先保留GRID190，不继续扫描混合比例或增大网络。值得追踪的是：哪些原门控遗漏的高收益状态能被稳定识别，以及省掉无效追加时为何同时丢掉有效追加。先在已有记录中比较可观察特征与收益幅度，再确定下一项有明确假设的训练。','',
        '## 审核范围','',
        '逐项核验12组原样本/标签/权重/标准化、原目标值与梯度；7134个新旧配对及21402条策略轨迹；原NET重现、同站奖励和查询边界、48组预期查询匹配及差异贡献守恒。没有新的神经网络前向、电网预测、生成数据或最终8测试访问。','',
        '这是保存教师路径上的开发研究，不是新的闭环整周质量、跨电网泛化或实际提速结论。缓存包装仅避免反复解压相同拟合数组，未改数值目标；原脚本及包装哈希均已保存。']
    (O/'GRID193_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('publication_audit.json',dict(passed=True,original_coordinate_objective_independently_checked_all12=True,all7134_pairs_reconstructed=True,all48_expected_query_budgets_equal=True,all_statuses_retained=True,new_fits=12,new_physics=0,new_NET_forwards=0))
    C.verify(); C.save('complete.json',dict(passed=True,execution_and_record_audit_complete=True,new_gate_fits=12,certified=cert,iteration_limited=12-cert,all12_published_before_held=True,all16dates_all3seeds=True,new_physics=0,final8_absent=True)); C.archive('complete')
    dest=R/'论文协作资料/FRAME_门控数值对照_GRID193'; dest.mkdir(exist_ok=False)
    for name in ['GRID193_report.md','decision.json','expected_budget_comparison.json','paired_case_accounting.json','endpoint_audit.json','review.json','publication_audit.json','complete.json']:
        shutil.copy2(O/name,dest/name); assert C.sha(O/name)==C.sha(dest/name)
    msg=f'GRID193固定12次从零相同目标预条件拟合完成，11原坐标梯度达标/1达2000上限，全部保留。原192两份受限均达标，样本/正则/阈值/原NET不改；动作89.942%/查询14.462，190为90.335%/15.007。同预期查询NET差{delta:+.8f}、动作差{md:+.4f}pp，保留190当前候选及所有互补案例。数值受限非唯一解释；不自动续训、组合、扩物理/数据或访问最终8。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/name; t=p.read_text(encoding='utf8'); entry='\n\n## 2026-10-26 — GRID193相同目标数值对照完成\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':head,rest=t.split('\n',1); t=head+'\n'+entry+'\n'+rest
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID193后\n\n1. 保留GRID190当前候选及其小幅跨种子改善；GRID192/193全部终点和反例保留，不扫描混合比例、择种子或延长受限拟合。\n2. 下一步先只读比较遗漏高收益、无效追加和新旧互补状态的在线可观测差异；区分收益幅度识别与单纯改选，不直接合并门控。\n3. 有明确可检验机制后再通知固定训练范围；不自动扩物理、日期、网络、最终8或论文。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md'; p.write_bytes('# 最新：GRID193相同目标数值优化对照\n\n[报告](protocols/GRID193_gate_precondition/complete/GRID193_report.md)；12终点全保留，11达原梯度标准/1达上限，当前保留GRID190。\n\n'.encode('utf8')+p.read_bytes())
    print('GRID193_COMPLETE',C.json.dumps(dict(decision=decision,matched=macro,novel=len(novel)),ensure_ascii=False))
if __name__=='__main__':main()
