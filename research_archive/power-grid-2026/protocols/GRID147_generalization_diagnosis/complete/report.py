"""Final independent consistency checks and publication of the read-only diagnosis."""
import json,shutil
from pathlib import Path
import run as D

def main():
    D.verify();O,R,REPO=D.O,D.R,D.REPO
    assert D.read(O/'state.json')['stage']=='analysed_pending_report'
    for n in ['feedback_context_source_hashes.json','saved_state_source_hashes.json']:
        for p,h in D.read(O/n).items():assert D.sha(R/p)==h,p
    summ=D.read(O/'summary.json');ranks=D.read(O/'ranking_comparison.json');cases=D.read(O/'case_comparison_summary.json')
    pairs=D.read(O/'all_crossdate_fit_neighbors.json');inventory=D.read(O/'saved_state_inventory.json');ctx=D.read(O/'feedback_context.json')
    fit=D.G.rowfile(D.G.O/'fit_rows.json.gz');cal=D.G.rowfile(D.G.O/'calibration_rows.json.gz');val=D.G.rowfile(D.G.O/'validation_rows.json.gz');allrows=fit+cal+val
    ids={r['meta']['sample_id']:r for r in allrows}
    assert len(pairs)==len(allrows)==275 and len(ids)==275
    # Verify deterministic matching with an independent scalar distance calculation.
    pre=D.read(D.G.O/'preprocessing.json');names=pre['features'];xx=D.B.transform(allrows,names,pre['medians'])
    geom=D.read(O/'matching_geometry.json');scales=D.np.asarray(geom['full_scales']);active=D.np.flatnonzero(D.np.ptp(xx[:153],axis=0)>0)
    matched=0
    for i,p in enumerate(pairs):
        r=allrows[i];assert p['sample_id']==r['meta']['sample_id']
        for label,name in [(True,'STOP'),(False,'CONTINUE')]:
            candidates=[j for j,s in enumerate(fit) if bool(s['target']['stop_label'])==label and s['meta']['date']!=r['meta']['date'] and s['features']['prefix_safe']==r['features']['prefix_safe']]
            q=p['neighbors'][name]
            if not candidates:assert q is None;continue
            def distance(j):return (sum(((xx[i,k]-xx[j,k])/scales[k])**2 for k in active)/len(active))**.5
            j=min(candidates,key=lambda j:(distance(j),fit[j]['meta']['sample_id']))
            assert q['sample_id']==fit[j]['meta']['sample_id'] and abs(q['distance']-distance(j))<1e-12
            matched+=1
    oob=D.read(O/'row_out_of_bag_scores_diagnostic.json')
    for model in ['RF140','RF146']:
        q=[x for x in oob if x['model']==model];assert len(q)==153
        for x in q:assert ids[x['sample_id']]['meta']['date']==x['date'] and x['whole_date_oob_trees']==0
        rr=[ids[x['sample_id']] for x in q]
        for reading,key in [('fitted_all_trees','fitted_score'),('row_oob_diagnostic','row_oob_score')]:
            m=next(x for x in ranks if x['model']==model and x['partition']=='fit128' and x['reading']==reading)
            pp=[x[key] for x in q]
            assert abs(m['auc']-D.manual_auc(rr,pp))<1e-12
            assert abs(m['continue_ap']-D.G.manual_ap(rr,pp))<1e-12
    assert len(inventory['records'])==305 and all(r['state_hash_matches_call_and_before_step'] for r in inventory['records'])
    inv={r['call_id']:r for r in inventory['records']}
    assert len(inv)==305 and all(inv[r['meta']['call_id']]['date']==r['meta']['date'] for r in allrows)
    ranges={r['sample_id']:r for r in D.read(O/'fit_range_checks.json')}
    error=next(c for c in cases if c['case']['call_id']=='2037-03-02_100|call009')
    assert error['case']['stop_label'] is False and not ranges[error['case']['sample_id']]['outside_fit_range']
    assert error['case']['leaf_inbag_dates']['min']==8
    records=D.read(O/'bootstrap_leaf_supports.json')
    for c in cases:
        assert c['case']['sample_id'] in D.read(O/'prespecified_case_ids.json')
        assert c['case']['public_features']==ids[c['case']['sample_id']]['features']
    branch=[]
    for part,rr in [('fit',fit),('calibration',cal),('validation',val)]:
        for safe in [False,True]:
            ss=[r for r in rr if r['features']['prefix_safe']==safe]
            branch.append(dict(partition=part,prefix_safe=safe,rows=len(ss),continue_count=sum(not r['target']['stop_label'] for r in ss)))
    D.save('branch_summary.json',branch)
    review=D.read(O/'review.json');review.update(independent_neighbor_checks=matched,independent_oob_ap_checks=4,
        existing_prestate_vectors_verified=305,existing_before_step_hashes_verified=305,
        original_context_sources_verified=True,unseen_date_oob_claimed=False)
    D.save('review.json',review)
    def m(model,part,reading):return next(r for r in ranks if r['model']==model and r['partition']==part and r['reading']==reading)
    nf=m('RF146','fit128','fitted_all_trees');no=m('RF146','fit128','row_oob_diagnostic');of=m('RF140','fit128','fitted_all_trees');oo=m('RF140','fit128','row_oob_diagnostic')
    supplement=next(x for x in ctx['declared_case_public_supplemental'] if x['case_call_id']==error['case']['call_id'])
    extras={x['sample_id']:x['public_features'] for x in supplement['public_supplemental_by_sample']}
    es=extras[error['case']['sample_id']];ns=extras[error['neighbors']['neighbors']['STOP']['sample_id']]
    fields=['incumbent_station_candidates','incumbent_station_remaining','incumbent_station_seen_fraction','seen_station_fraction','next_batch_new_station_candidate_fraction']
    assert all(es[k]==ns[k] for k in fields)
    D.save('decision.json',dict(new_fit=False,new_policy_adopted=False,ready_to_scale=False,
        conclusion='Fixed18-summary/forest/data combination has weak row-out-of-bag and cross-date discrimination. Leaf20 coarseness and minority pooling are real,but not sufficient causal explanations. No simple range/coverage veto resolves the specific error.',
        next='Keep original multi-stageFRAME and all controls. Decode existing pre-decision state schema and examine current congestion/action-location/feedback relations with online-available information before choosing further learned inputs. No direct4460-dimensional fit or new collection/NN by default.',
        no_threshold_or_label_change=True,final8_absent=True))
    lines=['# GRID147：高分误停与跨日期区分不足的原因核查','',
        '结论：单入口没有解决的核心问题，是现有摘要、森林和数据组合尚未形成稳定的可迁移区分能力。拟合排序明显偏乐观；错误状态可以落在训练数值范围内，并与很多正确停止状态共享树叶支持。不能只凭树较浅就认定扩大模型会改善结果。','',
        '本轮只读RF146和RF140的固定模型、bootstrap抽样及已有反馈。无新拟合、物理仿真、阈值或策略重放，没有更改原动作保持标签。','',
        '## 拟合表现能否离开直接参与拟合的树','',
        '| 固定森林，相同153个128入口 | 拟合继续AP | 拟合AUC | 行袋外继续AP | 行袋外AUC |','|---|---:|---:|---:|---:|']
    for name in ['RF140','RF146']:
        a=m(name,'fit128','fitted_all_trees');b=m(name,'fit128','row_oob_diagnostic')
        lines.append(f"| {name} | {a['continue_ap']:.4f} | {a['auc']:.4f} | {b['continue_ap']:.4f} | {b['auc']:.4f} |")
    lines += ['',
        f"RF146拟合AUC从{nf['auc']:.3f}降到行袋外{no['auc']:.3f}，继续类AP从{nf['continue_ap']:.3f}降到{no['continue_ap']:.3f}；35/153=22.88%是此分区的继续比例。拟合数据内部就缺乏稳定的袋外区分支持，不能把失败全部归于验证日期不同。这里没有显著性检验，也没有把0.49解释成显著差于随机。",'',
        '行袋外分数仅聚合没有在bootstrap中抽到该行的现有树，不重训。抽样索引和float32路由按[scikit-learn1.7.2官方说明](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.RandomForestClassifier.html)核对。新模型每行可用13–35棵树，中位23；全森林有64棵树，聚合数量不同，不能把差值全部解释成某一种过拟合机制。', '',
        '两份森林的153个入口都没有任何一棵树把整个对应日期完全排除。旧森林还可能抽到同次搜索的其它深度行，预处理和权重也使用全部拟合数据。因此这些分数仅是行级袋外诊断，不是日期交叉验证、独立测试或风险保证。两森林拟合样本量不同，也不能仅据袋外值判定哪种结构更有竞争力。','',
        '## 高分错误不是简单的单变量越界','',
        f"开发误停{error['case']['call_id']}，停止分数0.906026469，冻结阈值0.897774783。它的36个变换后通道均未超出拟合取值范围；落入树叶的bootstrap日期数最少8、中位10、最多12。64棵树中只有3棵落入纯停止叶，其余多数叶已经见过继续样本。",'',
        '所以这个错误并非所有树都“从未见过继续”，也不是只靠一个日期的孤立小叶子支撑。少量继续样本与较多停止样本被放在同一组，平均后仍给出很高停止分数。训练范围包含该点只表示没有单通道越界，不排除联合分布变化。','',
        '在275个原始输入中，没有完全相同特征却不同标签的组；不能据此断言任务本来不可学习。完整64树叶签名则有混合标签组：RF146为4组9行，RF140为3组7行，说明当前森林确实把部分不同输入压到相同表示。混合组并非全部错误的原因，新模型的主误停也没有被证明来自原始信息完全相同。','',
        '## 相似的停止和继续状态混在一起','',
        '| 分区 | 继续状态 | 两类均找到跨日期拟合近邻 | 距停止近邻更近 |','|---|---:|---:|---:|']
    for q in summ['neighbors']:lines.append(f"| {q['partition']} | {q['continue_rows']} | {q['matched_both_labels']} | {q['closer_to_fit_stop']} |")
    lines += ['',
        '匹配只使用拟合153入口的中位数、IQR/极差缩放，在相同初始分支内排除同日期，从全部拟合停止与继续行中各选最近一条。保留所有275行的配对，未按案例效果挑近邻。停止是多数类，特征之间也有相关和重复编码；近邻数受距离定义、类样本数影响，不是新分类器或信息不可分的证明。','',
        '主误停与其最近正确停止拟合状态在当前站候选数12、未查数11、已查比例1/12、全站覆盖比例以及下一批新站候选比例上完全相同。简单站点数量或“输入没见过就继续”规则，不能把这一对区分开。此前GRID142加6项批次统计也未通过主筛查，不能把“再加几个统计量”当成已经有证据的解决方案。','',
        '## 分批反馈具体提供了什么','',
        '误停案例在128处的最好候选为2，所属31号站，预测最大负载率0.988408；停止并补查该站后仍返回2。再看64个候选，最好候选变为141、所属16号站，负载率0.947823；原保存的192停止标签表明，此时按原规则补查能找到355，保持原FRAME动作，共211次查询，原FRAME总查询333。','',
        '这条证据表明，后续反馈改变了补查位置，而不只是提高一个分类分数。它是同一保存状态的原反馈与标签证据，不是额外物理实验，也不能用单例解释全部闭环成本。','',
        '| 分区 | 128处继续标签 | 多一批后当前最好动作改善 | 多一批加原补查恢复原动作 |','|---|---:|---:|---:|']
    for q in ctx['original_continue_future_labels']:lines.append(f"| {q['partition']} | {q['continue_at128']} | {q['next_batch_incumbent_improves']} | {q['one_more_batch_then_original_completion_restores_action']} |")
    lines += ['',
        '76个128继续状态中52个在多一批并执行原补查后恢复原动作；其它状态仍需要更长路径。改善当前最好动作与恢复原最终动作是不同指标，不改标签。另存的128→192共同可学习前缀表排除了已到原强制停止的节点，因此计数与这张未来路径标签表不完全相同。','',
        '原RF140在192开发前缀AUC约0.752，但同深度校准只有0.468；开发192继续样本仅6个。这既说明多一批可能提供有用信息，也提醒不能凭开发分区就宣布192学习可靠，更不能按结果改成固定192。所有深度与全部反向结果均保存。','',
        '## 更具体的当前电网状态已有保存','',
        '40个原训练/开发验证场景共305次搜索，均找到决策前状态文件，观测4460维、基础动作1605维。观测向量字节哈希同时匹配搜索记录state_hash及原物理控制步骤before_hash。没有初始化环境、发送预测或生成新样本。40场景含同日期副本，不是305个独立场景。','',
        '当前18项模型摘要没有显式描述每条拥塞线路或动作作用位置。已保存向量使下一步可以核查这些信息是否可用，无需先重采数据；字段布局尚未解码，不能猜索引，也不直接把4460维输入小样本模型。FULL教师保存状态仍不同于实际FRAME路径，后续若形成有用策略还需要覆盖核查。','',
        '## 下一步判断','',
        '保留原FRAME分批机制和强简单对照，不继续推广128单入口。先读取冻结环境的观测字段布局，明确决策前拥塞位置、候选动作作用位置和已揭示反馈的对应关系，再检查这些信息是否能区分相似摘要下的不同结果。若只有数量差异而无稳定跨日期关系，就不追加模型。','',
        '当前证据不支持只改最小叶或扩大网络即可解决问题。模型容量可以作为后续单因素问题，但优先级低于确认输入里是否有稳定、与后续搜索收益相关的信息。任何新输入训练都要另行固定方案并训练前通知；原动作/查询目标和最终8测试不动。','',
        '## 核验与边界','',
        f"核对{review['native_apply_route_checks']}个树路由、{review['leaf_weight_reconstructions']}个bootstrap加权叶、306个行袋外聚合以及8组AUC；550个跨日期近邻由独立标量距离复核，4组袋外/拟合AP独立核对。最大原分数重构误差{review['max_original_score_error']:.3g}，原模型、标签、权重、阈值和分数文件均未改变。305个原决策前状态核对通过，未声称独立AC重求解。",'',
        '本轮未采纳任何新分数/阈值，未运行新策略，不改变旧结果。统计分箱、叶签名和配对均为已暴露开发记录上的诊断，不能冒充独立测试。']
    (O/'GRID147_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8',newline='\n')
    D.save('complete.json',dict(passed=True,new_fits=0,new_physical=0,new_policy_evaluations=0,ready_to_scale=False,
        fixed_tree_diagnostic_scores_only=True,final8_absent=True,original_sources_unchanged=True))
    D.save('state.json',dict(stage='complete'));D.verify();D.archive('complete')
    c=R/'论文协作资料/FRAME_跨日期原因核查_GRID147';c.mkdir(exist_ok=False)
    names=['GRID147_report.md','design.json','summary.json','ranking_comparison.json','case_comparison_summary.json',
           'neighbor_summary.json','tree_split_usage.json','branch_summary.json','feedback_context.json','saved_state_inventory.json',
           'review.json','decision.json','complete.json']
    for n in names:shutil.copy2(O/n,c/n)
    msg='GRID147只读核查：固定RF146拟合AUC0.7686→行袋外0.4901/AP0.2466，RF140同入口0.7630→0.5404；全部153入口均无整日期袋外树，非留日期验证。主误停单变量无越界，叶支持8–12日期，3/64纯停止叶；混合叶/全部跨日期近邻/分箱保留，未证明任务不可学。128继续76条中52条下一批加原补查恢复原动作；主例站31→16，192保存标签恢复355。旧192校准AUC0.468的反例完整保留。305原决策前状态哈希与before-step一致，可先查物理状态布局，无需采集。零训练/新策略/物理/阈值修改，最终8封存。报告research_archive/power-grid-2026/protocols/GRID147_generalization_diagnosis/complete/GRID147_report.md'
    entry='\n\n## 2026-10-24 — GRID147跨日期区分原因核查完成\n\n'+msg+'\n'
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);old=title+'\n'+entry+'\n'+rest
        else:old+=entry
        p.write_text(old,encoding='utf8',newline='\n')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — GRID147后

1. 保留原FRAME多阶段机制、RF140和全部简单/失败对照，不采用单入口或袋外诊断分数，不自动调最小叶/网络/阈值。
2. 下一步先解码既有305个决策前状态的冻结字段布局，核查拥塞位置、候选动作作用位置和已揭示反馈的对应关系；不猜向量索引，不直接4460维拟合，不初始化求解/新采集。
3. 先验证新可知信息是否能解释相似摘要下不同结果，保留反例和各日期支持；已有站点数量相同反例、六批次统计失败结果均不可忽略。若无稳定关系，不追加模型。
4. 任何训练另行冻结、训练前通知；无新物理/最终8/论文/RL或扩大。行OOB不是日期留出证据，查询或动作保持不冒充整周质量与实测时间。
''',encoding='utf8',newline='\n')
    p=REPO/'research_archive/power-grid-2026/README.md';original=p.read_bytes();prefix='# 最新：GRID147跨日期原因核查\r\n\r\n[报告](protocols/GRID147_generalization_diagnosis/complete/GRID147_report.md)；固定树袋外、相似状态与已保存电网观测核查，无新训练。\r\n\r\n'.encode('utf8');p.write_bytes(prefix+original)
    count=0
    for phase in D.A.iterdir():
        for n,h in D.read(phase/'archive_manifest.json').items():assert D.sha(phase/n)==h;count+=1
    for p in c.iterdir():assert D.sha(p)==D.sha(O/p.name)
    D.save('publication_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),
        fixed_models_data_labels_cutoffs_unchanged=True,additional_fits=0,additional_physical_calls=0))
    d=D.A/'verification';d.mkdir();shutil.copy2(O/'publication_verification.json',d/'publication_verification.json')
    (d/'archive_manifest.json').write_text(json.dumps({'publication_verification.json':D.sha(d/'publication_verification.json')},indent=2)+'\n',encoding='utf8',newline='\n')
    print('GRID147_COMPLETE',count,'archive files',matched,'neighbors checked')

if __name__=='__main__':main()
