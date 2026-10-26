"""Independent saved-search audit and reporting for fixed trained gates."""
from pathlib import Path
import collections,pickle,shutil,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import run as C
R=C.R;O=C.O;read=C.read
def main():
    C.trained();assert read(O/'review.json')['passed'];assert not (O/'complete.json').exists();meta,lookup,case_list=C.load();cases={c['sample_id']:c for c in case_list};tr=read(O/'oof_trajectories.json');pred=read(O/'gate_predictions.json');metrics=read(O/'search_metrics.json');families=read(O/'family_summary.json');paired=read(O/'paired_summary.json');counts=read(O/'gate_counts.json');fit=read(O/'fits.json');task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale'];by={};checks=collections.Counter();gp={(p['seed'],p['sample_id']):p for p in pred}
    C.save('interpretation_plan.json',dict(posthoc=True,source_code_sha256=C.sha(Path(__file__)),additional_diagnostic='Compute uniform independent Bernoulli extension expectation at same addedcandidatequery budget in each date/seed,p=gate_addedQ/all_eligibleQ. No simulated draws/newseeds, no exactrunbudget/walltimematching. Report all results even adverse;not a prespecified trained baseline or new confirmation.',new_fits=0,new_model_forwards=0,new_physics=0))
    for v in tr:
        c=cases[v['sample_id']];name=v['method'];method=name.rsplit('_seed',1)[0];seed=int(name[-1]);p=v['baseline_stop'];can=p<len(c['states']);add=(can and (method=='ADD1' or method=='GATE' and gp[seed,c['sample_id']]['gate_continue']));end=p+int(add)
        assert v['append']==add and v['batches']==end and v['queries']==sum(int(x) for x in c['queries'][:end]);assert abs(v['gain']-sum(float(x) for x in c['gains'][:end]))<1e-12;assert v['chosen_pool_id']==c['chosen'][end];assert abs(v['net_utility']-(v['gain']-fee*v['queries'])/scale)<1e-12
        assert v['action_match']==(v['chosen_pool_id']==c['chosen'][-1]);by.setdefault(name,{})[v['sample_id']]=v;checks['independent_trajectories']+=1
    restored=[];waste=[]
    for seed in range(3):
        for sid,n in by[f'NET_seed{seed}'].items():
            g=by[f'GATE_seed{seed}'][sid];a=by[f'ADD1_seed{seed}'][sid];assert n['queries']<=g['queries']<=a['queries'] and n['gain']<=g['gain']<=a['gain'];assert not n['action_match'] or g['action_match']
            if g['append'] and g['chosen_pool_id']!=n['chosen_pool_id']:restored.append(dict(seed=seed,sample_id=sid,date=g['date'],group=g['physical_state_group'],added_queries=g['queries']-n['queries'],added_gain=g['gain']-n['gain'],full_match=g['action_match']))
            elif g['append']:waste.append(dict(seed=seed,sample_id=sid,date=g['date'],group=g['physical_state_group'],added_queries=g['queries']-n['queries']))
    # Weighted expected random query reference;no observed outcome controls its probability.
    random_rows=[];nc=collections.Counter(c['date'] for c in case_list);fullg=collections.defaultdict(float)
    for c in case_list:fullg[c['date']]+=float(c['gains'].sum())
    for seed in range(3):
        for b in metrics[f'NET_seed{seed}']['per_date']:
            date=b['date'];pp=[p for p in pred if p['seed']==seed and p['date']==date];totalq=sum(p['actual_next_queries'] for p in pp);paidq=sum(p['actual_next_queries'] for p in pp if p['gate_continue']);pr=paidq/totalq if totalq else 0.;gain=pr*sum(p['actual_next_gain'] for p in pp);n=nc[date];recoverable=sum(cases[p['sample_id']]['chosen'][p['position']]!=cases[p['sample_id']]['chosen'][-1] and cases[p['sample_id']]['chosen'][p['position']+1]==cases[p['sample_id']]['chosen'][-1] for p in pp)
            row=dict(seed=seed,date=date,probability=pr,mean_queries=b['mean_queries']+paidq/n,mean_gain=b['mean_gain']+gain/n,mean_net_utility=b['mean_net_utility']+(gain-fee*paidq)/(scale*n),action_match_rate=b['action_match_rate']+pr*recoverable/n,gain_capture=b['gain_capture']+gain/fullg[date] if fullg[date] else None)
            gate=next(x for x in metrics[f'GATE_seed{seed}']['per_date'] if x['date']==date);assert abs(gate['mean_queries']-row['mean_queries'])<1e-12;random_rows.append(row)
    rnd={k:float(np.mean([v[k] for v in random_rows if v[k] is not None])) for k in ['mean_queries','mean_gain','mean_net_utility','action_match_rate','gain_capture']};C.save('query_matched_random_expectation.json',dict(posthoc=True,exact_expected_candidate_budget_only=True,not_actualrunbudget_or_walltime_matched=True,per_date_seed=random_rows,macro=rnd))
    # Keep old auxiliary and simple references explicitly separate, not pooled weights.
    oldref=read(R/'outputs/grid187_action_invariance_target/search_metrics.json');oldaux=read(R/'outputs/grid123_event_auxiliary/decision.json')['family_utility']['oof']
    C.save('case_accounting.json',dict(changed_action_pairs=restored,no_change_append_pairs=waste,changed_unique_starts=len({x['sample_id'] for x in restored}),changed_physical_groups=len({x['group'] for x in restored}),changed_dates=sorted({x['date'] for x in restored}),old_aux_OOF_utility_separate=oldaux))
    n=families['NET'];g=families['GATE'];a=families['ADD1'];match_pp=100*(g['action_match_rate']-n['action_match_rate']);q_pc=100*(g['mean_queries']/n['mean_queries']-1);net_pc=100*(g['mean_net_utility']/n['mean_net_utility']-1);recall=counts['append_changes']/counts['all_positive_changes'];precision=counts['append_changes']/counts['append'];base_rate=counts['all_positive_changes']/counts['eligible']
    C.save('decision.json',dict(gain_and_match_increment_observed=True,match_delta_pp=match_pp,query_change_percent=q_pc,NET_change_percent=net_pc,all_originalNET_decisions_and_quality_on_saved_candidate_problem_preserved=True,all3seed_local_match_increased=all(p['match_delta_pp']>0 for p in paired),not_same_compute_quality_claim=True,not_real_closedloop_quality_claim=True,random_query_expectation_posthoc=rnd,no_automatic_expansion=True,scope='Exposed samegrid teacherpath OOF; no physical latency/weekcost claim'))
    lines=['# GRID190：冻结NET后的单批学习追加','',
        f"12个84参数逻辑回归门控已训练并通过保存数据核验。原NET冻结不变。16日期折外、3种子平均：动作一致率{100*n['action_match_rate']:.2f}%→{100*g['action_match_rate']:.2f}%（+{match_pp:.2f}个百分点），奖励增量保留率{100*n['gain_capture']:.2f}%→{100*g['gain_capture']:.2f}%；候选查询{n['mean_queries']:.2f}→{g['mean_queries']:.2f}（+{q_pc:.2f}%），原NET效用变化{net_pc:+.2f}%。",'',
        '结论是已有可重复的局部质量补强，同时付出额外查询，尚未形成同计算成本优势。不能因为效用略降就删除动作改善，也不能把模型保留原NET能力这一结构性质说成所有追加都值得。','',
        '## 固定训练与输入','',
        f"原4个日期划分各用12日期拟合、4日期留出，每个冻结NET种子对应一个门控，共12次固定拟合。83维输入为原NET主干64维、反馈分支16维、两个当前价值输出及已知下一批费用。原NET参数和停止决定逐项保持，未新增环境特征或未来反馈。",'',
        f"门控固定L2/C=1/lbfgs最多2000次/阈值0.5，无类别重加权或参数扫描；全部收敛于{min(v['n_iter'] for v in fit['records'])}–{max(v['n_iter'] for v in fit['records'])}次迭代。纯门控拟合阶段耗时{fit['total_wall_s']:.3f}秒，表征抽取与核验另计。这不是在线控制耗时。",'',
        '标准化与日期→物理调用组→起点权重仅由本折拟合侧数据计算；12份最终权重全部上传后，才抽取留出门控表征并评分。只在原NET第一次STOP且仍有候选时判断一次；追加后直接按原规则返回已观察最佳动作。原4开发日期未新增评分，最终8未生成。','',
        '## 主结果：学习追加与无条件追加','',
        '|方法|完整同站动作一致率|奖励增量保留率|每起点查询|完整查询占比|原NET效用|','|---|---:|---:|---:|---:|---:|']
    for name,z in [('原NET',n),('NET停止后学习追加一批',g),('NET停止后无条件追加一批',a)]:lines.append(f"|{name}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{100*z['query_fraction']:.3f}%|{z['mean_net_utility']:.6f}|")
    for name,label in [('BATCH1','原固定BATCH1（总共一批）'),('FULL_LOCAL','完整同站搜索')]:
        z=oldref[name]['macro'];lines.append(f"|{label}|{100*z['action_match_rate']:.3f}%|{100*z['gain_capture']:.3f}%|{z['mean_queries']:.3f}|{100*z['query_fraction']:.3f}%|{z['mean_net_utility']:.6f}|")
    lines+=['','这些是同一批保存起点上的局部搜索指标，不是完整周成本或物理控制质量。完整同站最终动作不是全局最优或安全标签。原BATCH1从搜索起点只买一批，与NET停止后再追加一批不同，不能混为同一基线。','',
        f"相对无条件追加，学习门控购买的额外查询比例为{100*g['added_queries']/a['added_queries']:.2f}%；对应的奖励增量提高也更小。它提供了一种中间取舍，没有同时达到无条件追加的全部质量改善。",'',
        '## 全部种子与日期','',
        '|NET种子|动作一致率增量（百分点）|查询增量|NET效用增量|动作一致提高的日期数|','|---|---:|---:|---:|---:|']
    for p in paired:lines.append(f"|{p['seed']}|{p['match_delta_pp']:+.3f}|{p['query_delta']:+.3f}|{p['NET_delta']:+.6f}|{sum(v['match_delta_pp']>0 for v in p['per_date'])}/16|")
    lines+=['','所有种子均提高动作一致率；未提高的日期也保留。局部动作一致不下降由只追加候选的结构决定，增量大小和花费仍由门控选择决定；不能把非退化边界当作学习已充分有效的证据。','',
        '## 门控实际选中了什么','',
        f"7134个种子—起点配对中，有{counts['eligible']}个原NET停止点可追加。门控追加{counts['append']}次，其中{counts['append_changes']}次改变所选动作、{counts['append_no_change']}次没有改选；{counts['append_positive_NET']}次下一批NET为正。动作改变对应{len({x['sample_id'] for x in restored})}个不同起点、{len({x['group'] for x in restored})}个物理调用组，覆盖{len({x['date'] for x in restored})}个日期。多个前缀/种子非独立场景。",'',
        f"已追加批次的改选比例为{100*precision:.2f}%，全部可追加机会的改选比例为{100*base_rate:.2f}%；因此存在选择性信号。现有单批改选机会只追回{100*recall:.2f}%，仍漏掉{counts['missed_changes']}个，覆盖尚小。这个比例比较没有匹配查询大小，不能据此单独声称等预算胜过随机。",'',
        '## 事后诊断：与相同预期候选查询的随机追加比较','',
        '这项在主结果之后登记，属于描述性解释，不是预注册的新训练基线。每日期、每种子在可追加停止点以统一概率独立追加；概率只由学习门控已付候选查询量/所有可追加查询量确定。因此各日期候选查询量的数学期望与门控一致，没有用奖励来选概率。没有随机抽样或新的物理运行。','',
        '|策略|每起点查询（期望）|动作一致率|奖励增量保留率|NET效用|','|---|---:|---:|---:|---:|',
        f"|学习追加|{g['mean_queries']:.3f}|{100*g['action_match_rate']:.3f}%|{100*g['gain_capture']:.3f}%|{g['mean_net_utility']:.6f}|",
        f"|同预期查询随机追加|{rnd['mean_queries']:.3f}|{100*rnd['action_match_rate']:.3f}%|{100*rnd['gain_capture']:.3f}%|{rnd['mean_net_utility']:.6f}|",'',
        '此参照只匹配候选查询的期望，不匹配一次实际运行的硬预算，也不匹配门控推理耗时。全部48个日期—种子结果均在query_matched_random_expectation.json，不能当作新的独立实验或删去不利对照。','',
        '## 下一步的判断依据','',
        '本轮保留原价值模型，避免了纯二元替换时丢掉其继续决定的问题，并找到了分布于多个日期/物理组的小幅动作改善。下一步应解释命中、无效追加与遗漏的机会，尤其检查收益幅度及查询大小；不立即扩大网络或按当前结果降低阈值。','',
        '若要宣称竞争力，需要后续在明确范围内测量门控/表征开销和完整物理闭环质量。当前不自动扩数据或开始新物理回放，也不因保留原NET路径就承诺整周成本不升。','',
        '## 核验与材料','',
        '16565条拟合侧83维表征和原NET分数/STOP决定保持一致，所有冻结权重哈希不变。12门控均收敛，sklearn输出与手工线性/概率计算一致；全部权重先发布。7134条原NET保存搜索逐项重现，21402条三臂搜索按原查询数组、奖励、动作与费用重算，查询上下界和局部最佳选择性质通过。','',
        'search_metrics.json、paired_summary.json与gate_predictions.json保留全部种子/日期，case_accounting.json保留所有改选和无效追加；models/保存12份权重与原生估计器。没有新的电网预测、独立AC求解或最终测试。']
    (O/'GRID190_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    C.save('publication_audit.json',dict(passed=True,checks=dict(checks),all_query_gain_and_match_bounds_checked=True,all7134baseline_searches_reproduced=True,all12gates_converged=fit['all_converged'],random_diagnostic_matches_expected_query_per_date_seed=True,all_seeds_dates_and_adverse_results=True,new_physics=0));C.trained();C.save('complete.json',dict(passed=True,new_gate_fits=12,NET_retraining=0,all12weights_before_held=True,all16dates_all3seeds=True,new_physics=0,old4dev_new_scores=0,final8_absent=True));C.archive('complete')
    dest=R/'论文协作资料/FRAME_NET停止后学习追加_GRID190';dest.mkdir(exist_ok=False)
    for name in ['GRID190_report.md','family_summary.json','paired_summary.json','decision.json','review.json','publication_audit.json','complete.json','query_matched_random_expectation.json']:shutil.copy2(O/name,dest/name);assert C.sha(O/name)==C.sha(dest/name)
    msg=f'GRID19012个84参数LR门控全部收敛，NET冻结/先发布后held。16日期OOF动作一致88.984%→90.335%(+1.350pp)，增量保留63.75%→66.16%，候选12.884→15.007(+16.47%)，NET效用−1.27%；三个种子一致增量+1.112/+1.455/+1.484pp。278追加84改选/61起点/29物理组，194无改选；小增量保留，不当同计算或整周优势。无条件ADD1动作97.209%/查询34.392，事后同预期查询随机参照全量保留，未自动扩物理/数据/阈值。'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=C.REPO/'docs'/name;t=p.read_text(encoding='utf8');entry='\n\n## 2026-10-26 — GRID190冻结NET单批门控训练完成\n\n'+msg+'\n'
        if name=='RESEARCH_STATE.md':a0,b0=t.split('\n',1);t=a0+'\n'+entry+'\n'+b0
        else:t+=entry
        p.write_text(t,encoding='utf8',newline='\n')
    (C.REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID190后\n\n1. 保留冻结NET与单批门控的重复小改善，先只读解释84次改选、194次无改选、528次遗漏机会及收益/查询大小；原16日期与3种子全部保留，不因效用略降就删改善，也不声称同预算全优。\n2. 事后相同预期候选查询随机参照与ADD1强简单对照一起看，不按结果改阈值、挑日期或自动再训。准备真实闭环前明确推理/表征开销、范围与质量参照；当前未新物理。\n3. 原4开发未新评分，最终8未生成，论文/第二网不动。\n',encoding='utf8',newline='\n')
    p=C.REPO/'research_archive/power-grid-2026/README.md';p.write_bytes('# 最新：GRID190冻结NET后的学习追加\r\n\r\n[报告](protocols/GRID190_extension_gate/complete/GRID190_report.md)；12门控完成，动作小幅改善与额外查询分列，尚未新物理。\r\n\r\n'.encode('utf8')+p.read_bytes());print('GRID190_COMPLETE',C.json.dumps(dict(family_GATE=g,random_expected=rnd,changed_groups=len({x['group'] for x in restored}))))
if __name__=='__main__':main()
