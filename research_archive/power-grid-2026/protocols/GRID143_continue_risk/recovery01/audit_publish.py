"""Audit ordinal metrics, date-subset calibration and saved future labels."""
from run import *
def manual_ap(y,score):
    # Continue score=1-stop score: ascending stop-score order,group ties.
    count=sum(y)
    if not count:return None
    order=sorted(range(len(y)),key=lambda i:score[i]);tp=0;seen=0;ap=0.;offset=0
    while offset<len(order):
        end=offset+1
        while end<len(order) and score[order[end]]==score[order[offset]]:end+=1
        add=sum(y[order[j]] for j in range(offset,end));tp+=add;seen+=end-offset;ap+=(tp/seen)*(add/count);offset=end
    return ap
def main():
    verify();assert read(O/'state.json')['stage']=='analysed_pending_audit';assert not (O/'review.json').exists();rows,parts,scores=payload();metrics=read(O/'ranking_metrics.json');dm=read(O/'ranking_per_date.json');checks=collections.Counter()
    for record in metrics+dm:
        ids=[i for i,p in enumerate(parts) if p==record['partition'] and ('date' not in record or rows[i]['meta']['date']==record['date'])];pp=scores[record['model']][ids];rr=[rows[i] for i in ids];continue_y=[not r['target']['stop_label'] for r in rr]
        ap=manual_ap(continue_y,pp);assert ap is None and record['continue_ap'] is None or ap is not None and abs(ap-record['continue_ap'])<1e-12
        pos=[p for r,p in zip(rr,pp) if r['target']['stop_label']];neg=[p for r,p in zip(rr,pp) if not r['target']['stop_label']]
        if pos and neg:
            auc=sum(1 if p>q else .5 if p==q else 0 for p in pos for q in neg)/(len(pos)*len(neg));assert abs(auc-record['auc'])<1e-12
            assert record['hindsight_stop_scores_above_all_continue']==sum(p>max(neg) for p in pos)
        checks['rank_metric_groups']+=1
    for r in read(O/'same_depth_branch_pairwise_ranking.json'):
        ii=[i for i,p in enumerate(parts) if p==r['partition'] and rows[i]['meta']['visible_count']==r['visible_count'] and rows[i]['features']['prefix_safe']==r['prefix_safe']];p=scores[r['model']];pos=[p[i] for i in ii if rows[i]['target']['stop_label']];neg=[p[i] for i in ii if not rows[i]['target']['stop_label']]
        assert r['pairs']==len(pos)*len(neg) and r['correct']==sum(a>b for a in pos for b in neg) and r['ties']==sum(a==b for a in pos for b in neg);checks['stratified_rank_groups']+=1
    folds=read(O/'all_calibration_date_folds.json');histories=read(O/'all_calibration_date_histories.json');allcalls=read(O/'all_omitted_calibration_call_replays.json');D=read(G.PREV/'design.json')
    for fold in folds:
        model=fold['model'];omit=fold['omitted_calibration_date'];donor=set(D['calibration_dates'])-{omit};pp=scores[model]
        donormap={(r['meta']['call_id'],r['meta']['visible_count']):float(pp[i]) for i,r in enumerate(rows) if parts[i]=='calibration' and r['meta']['date'] in donor}
        testmap={(r['meta']['call_id'],r['meta']['visible_count']):float(pp[i]) for i,r in enumerate(rows) if parts[i]=='calibration' and r['meta']['date']==omit};assert not set(donormap)&set(testmap)
        saved=next(h['history'] for h in histories if h['model']==model and h['omitted_calibration_date']==omit);tau=.5
        for j,h in enumerate(saved):
            assert h['iteration']==j and h['threshold']==tau;rr=[G.replay(r,'MODEL',tau,donormap) for r in G.refs('train',donor)];bad=[r for r in rr if r['bad_learned_stop']];assert G.summarize(rr)==h['summary']
            if j<len(saved)-1:assert bad;tau=max(r['stopped_score'] for r in bad)
            else:assert not bad
        assert tau==fold['diagnostic_threshold'];rr=[G.replay(r,'MODEL',tau,testmap) for r in G.refs('train',{omit})];assert G.summarize(rr)==fold['heldout_summary']
        prior={r['call_id']:r for r in allcalls if r['model']==model and r['omitted_calibration_date']==omit}
        for r in rr:assert all(r[k]==prior[r['call_id']][k] for k in r)
        checks['calibration_folds']+=1;checks['omitted_calibration_calls']+=len(rr)
    paths=read(O/'all_continue_future_paths.json');refsmap={r['call_id']:r for r in G.refs()};rowmap={r['meta']['sample_id']:r for r in rows}
    for path in paths:
        ref=refsmap[path['call_id']];r=rowmap[path['sample_id']];assert not r['target']['stop_label'];a=G.replay(ref,'CAP'+str(path['visible_count']));b=G.replay(ref,'CAP'+str(path['next_global_queries']))
        assert a['selected']==r['target']['after_stop_choice_label'] and a['queries']==r['target']['stop_total_queries_known_now'];assert b['selected']==path['next_then_completion_choice'] and b['queries']==path['next_then_completion_queries'];assert (b['selected']==ref['frame_choice'])==path['next_then_completion_restores_action']
        c=G.B.call_at(ref['source']);n=path['visible_count'];checkpoints=list(range(n+64,ref['global_n']+1,64))
        if ref['global_n'] not in checkpoints:checkpoints.append(ref['global_n'])
        first=next(q for q in checkpoints if G.replay(ref,'CAP'+str(q))['selected']==ref['frame_choice']);assert first==path['first_restoring_global_queries'];checks['negative_future_paths']+=1
    weights=np.asarray([r['weight'] for r in read(G.PREV/'fit_sample_weights.json')]);y=np.asarray([r['target']['stop_label'] for r in rows[:386]],bool);mass=read(O/'fit_class_mass_diagnosis.json');assert abs(mass['base_continue_weight']-weights[~y].sum())<1e-10
    assert abs(mass['planning_only_equal_mass_continue_multiplier']*weights[~y].sum()-mass['planning_only_equal_mass_stop_multiplier']*weights[y].sum())<1e-10
    verify();save('review.json',dict(passed=True,checks=dict(checks),models_scores_labels_cutoffs_unchanged=True,donor_and_omitted_scores_separated=True,no_validation_derived_thresholds_applied=True,new_fits=0,new_predict_calls=0,new_physical=0,final_tests_absent=True))
    foldsumm=read(O/'calibration_sensitivity_summary.json');pathsumm=read(O/'continue_path_summary.json');strata=read(O/'same_depth_branch_pairwise_ranking.json');aggregate=[]
    for model in scores:
        for part in ['fit','calibration','validation']:
            ss=[r for r in strata if r['model']==model and r['partition']==part and r['pairs']];n=sum(r['pairs'] for r in ss);aggregate.append(dict(model=model,partition=part,pairs=n,within_depth_branch_auc=sum(r['correct']+.5*r['ties'] for r in ss)/n if n else None))
    save('conditional_ranking_summary.json',aggregate)
    save('decision.json',dict(decision='DIAGNOSIS_ONLY_NO_MODEL_OR_THRESHOLD_ADOPTION',ranking=metrics,calibration_sensitivity=[dict(model=f['model'],folds_with_errors=f['folds_with_heldout_action_differences'],heldout_action_differences=f['all_omitted_calibration_calls']['action_differences']) for f in foldsumm],ready_to_scale=False,new_fits=0,new_physical=0,final_tests_absent=True))
    lines=['# GRID143：少数继续样本与校准日期稳定性','',
        '只分析RF140和RF142固定模型的保存分数、标签与原反馈。无新拟合或模型预测，没有给验证集重设阈值。原RF140保留为参照，RF142原失败结论不变。','',
        '## 1. 专门检查必须继续的少数样本','',
        '将继续作为正类，用1减停止分数排序，报告平均精确率AP及该类比例。AP的基准随继续样本比例变化，不能拿原来停止类的高AP来替代。原阈值下同时报告继续召回和有效停止召回，避免通过几乎全继续取得看似很好的风险召回。','',
        '| 模型 | 分区 | 继续数量 | 继续类AP | 继续比例 | 继续召回 | 有效停止召回 |','|---|---|---:|---:|---:|---:|---:|']
    for r in metrics:
        lines.append(f"| {r['model']} | {r['partition']} | {r['continue_prefixes']}/{r['prefixes']} | {r['continue_ap']:.4f} | {100*r['continue_prevalence']:.2f}% | {100*r['continue_recall']:.2f}% | {100*r['valid_stop_recall']:.2f}% |")
    lines += ['',
        '以上是原路径可学习前缀的逐行指标，既有策略提前停止后未必访问后面的前缀，不能当作完整搜索动作差异次数。所有日期分别保存；同调用前缀、同日两份场景不是独立样本。','',
        '| 模型 | 分区 | 同深度/初始分支配对AUC | 真停止分数高于全部继续分数 |','|---|---|---:|---:|']
    for r in aggregate:
        m=next(v for v in metrics if v['model']==r['model'] and v['partition']==r['partition']);lines.append(f"| {r['model']} | {r['partition']} | {r['within_depth_branch_auc']:.4f} | {m['hindsight_stop_scores_above_all_continue']}/{m['stop_prefixes']} |")
    lines += ['',
        '后一列使用了答案，只描述两类在高分端的重叠，不是可部署阈值。没有在这些位置重放验证策略、挑选阈值或声称新增模型成功。条件配对只控制原查询深度和初始安全分支，不是因果检验。','',
        '## 2. 四个校准日期是否稳定','',
        '对每个固定模型，每次留出原四个校准日期之一，只用另外三个日期按原整次搜索程序校准，再重放留出日期。共八次全部保留；只检验校准日期，没有把这些诊断阈值用于验证日期，更没有选择一个替换原阈值。','',
        '| 模型 | 留出校准日期 | 三日期诊断阈值 | 留出动作差异 | 留出提前停止 |','|---|---|---:|---:|---:|']
    for f in folds:lines.append(f"| {f['model']} | {f['omitted_calibration_date']} | {f['diagnostic_threshold']:.9f} | {f['heldout_summary']['action_differences']}/{f['heldout_summary']['calls']} | {f['heldout_summary']['model_stops']} |")
    for f in foldsumm:lines += ['',f"{f['model']}：{f['folds_with_heldout_action_differences']}/4个留出日期出现动作差异，合计{f['all_omitted_calibration_calls']['action_differences']}/{f['all_omitted_calibration_calls']['calls']}次留出搜索。"]
    lines += ['',
        '这是已暴露校准记录的敏感性诊断，不是新的独立交叉验证。特征方案此前还参考了开发记录，因此不能将该结果包装成无偏泛化估计。它直接检查“校准日期上没有错”对另一个日期是否可靠；不构成安全证明。','',
        '## 3. 继续标签对应什么后续路径','',
        '| 分区 | 继续前缀 | 涉及调用/日期 | 下一批改善全局最好 | 下一批加原补查恢复原动作 | 需要更久 |','|---|---:|---:|---:|---:|---:|']
    for r in pathsumm:lines.append(f"| {r['partition']} | {r['prefixes']} | {r['calls']}/{r['dates']} | {r['next_incumbent_improves']} | {r['one_batch_restores']} | {r['needs_more_than_one_batch']} |")
    lines += ['',
        '改善全局最好候选与恢复原FRAME最终动作不是同一个指标。后者包含原条件补查。全部122项均用已保存后续反馈复算，是标签解释，不是新在线信息或物理回放；未改变原动作保持标签，也不把这些结果换算成整周成本。','',
        '## 4. 下一次若研究训练权重，应固定什么','',
        f"原拟合继续类55/386，但有效权重只有{100*mass['base_continue_weight_fraction']:.2f}%。仅按拟合数据计算的等类别质量归一化，停止乘数为{mass['planning_only_equal_mass_stop_multiplier']:.6f}、继续乘数为{mass['planning_only_equal_mass_continue_multiplier']:.6f}，两类乘数比{mass['continue_to_stop_multiplier_ratio']:.6f}。这是训练损失归一化的数学规划值，不是已训练模型，也不是经济损失或仿真时间价格。",'',
        '如果后续只检验权重，应回到表现更好的原18项表示，保留原模型结构、种子、日期、标签和搜索/校准；仅做一套由拟合分区计算的类别质量平衡，不扫描系数。类内原相对权重可以保留，但全体日期/调用总权重会改变，不能声称只是在保持原日期等权下简单改名字。','',
        '本轮并未执行该训练。若继续，应先封存一次拟合方案并通知，保留原RF140、RF142与全部简单对照，仍按整次搜索动作保持和查询增量判断。无论是否改善，不能用抬验证阈值救结果，也不直接升级网络或RL。最终8测试场景未生成。']
    (O/'GRID143_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed=True,analysis_only=True,new_fits=0,new_predictions=0,new_physical=0,thresholds_not_adopted=True,ready_to_scale=False,final_tests_absent=True));save('state.json',dict(stage='complete'));verify();archive('complete')
    c=R/'论文协作资料/FRAME_继续风险与校准诊断_GRID143';c.mkdir(exist_ok=False);names=['GRID143_report.md','design.json','ranking_metrics.json','ranking_per_date.json','conditional_ranking_summary.json','calibration_sensitivity_summary.json','all_calibration_date_folds.json','continue_path_summary.json','fit_class_mass_diagnosis.json','decision.json','review.json','complete.json']
    for n in names:shutil.copy2(O/n,c/n)
    val140=next(r for r in metrics if r['model']=='RF140' and r['partition']=='validation');val142=next(r for r in metrics if r['model']=='RF142' and r['partition']=='validation')
    entry=f'''\n\n## 2026-10-24 — GRID143继续少数类与校准日期诊断完成\n\n只读两固定模型726分数/122继续前缀，零拟合/predict/物理。继续正类开发AP RF140={val140['continue_ap']:.4f}、RF142={val142['continue_ap']:.4f}，基准22/142；同时报告继续召回与有效停止召回，不能多数类准确率代替。逐日期/同查询深度与初始分支排序全部保留。8个三日期校准/留一校准日重放只作稳定性诊断，原阈值不变，不用于开发验证调阈值。RF140留出差异{foldsumm[0]['all_omitted_calibration_calls']['action_differences']}/82、RF142 {foldsumm[1]['all_omitted_calibration_calls']['action_differences']}/82，非新独立验证。122路径独立复算下一批改善/恢复原动作和更长依赖，未来仅作标签。有效继续权重{100*mass['base_continue_weight_fraction']:.2f}%，只计算拟合内等类别质量规划因子，无新训练/阈值采用。下一干预若继续，原18特征、只改变类别质量权重，需先封存/训练前通知；不扩网络/物理/最终8。报告research_archive/power-grid-2026/protocols/GRID143_continue_risk/complete/GRID143_report.md\n'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(old+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — 继续少数类与校准稳定性\n\n1. GRID143完成两固定模型的继续类AP、同深度/分支排序和8个校准日期留出敏感性重放，原阈值不变，不选择诊断阈值。详见报告全部日期与反例。\n2. 继续标签的下一批改善、加补查恢复原动作与更长依赖分别保存；未来只作标签，动作保持不等于经济收益，不能换标签救结果。\n3. 若下一步训练，只建议原18项表示的一次类别质量平衡干预：由原拟合权重与标签计算，类内相对权重保留，全体日期总权重会改变。模型/日期/标签/搜索/校准不变，无系数扫描。先封存方案并在拟合前通知。\n4. 原RF140与RF142及CAP参照全保留。重复开发评价不当独立验证；最终8封存，无新物理/网络/RL/论文修改，不自动扩大。\n',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+'\n- [GRID143继续少数类与校准日期诊断](protocols/GRID143_continue_risk/complete/GRID143_report.md)：两固定模型排序、8个校准日敏感性重放、122后续路径；零拟合，原阈值不变。\n',encoding='utf8')
    count=0
    for d in A.iterdir():
        if d.is_dir():
            for n,h in read(d/'archive_manifest.json').items():assert sha(d/n)==h;count+=1
    for n in names:assert sha(c/n)==sha(O/n)
    verify();save('final_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),original_scores_models_cutoffs_labels_unchanged=True,new_fits=0,new_physical=0,final_tests_absent=True))
    d=A/'verification';d.mkdir(exist_ok=False);shutil.copy2(O/'final_verification.json',d/'final_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID143_COMPLETE',json.dumps(dict(continue_ap_old=val140['continue_ap'],continue_ap_new=val142['continue_ap'],folds=foldsumm,paths=pathsumm),ensure_ascii=False))
if __name__=='__main__':main()
