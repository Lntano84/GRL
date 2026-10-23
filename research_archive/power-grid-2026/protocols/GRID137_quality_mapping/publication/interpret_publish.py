"""Interpret inherited quality evidence without fitting or changing prior outcomes."""
from run import *

def main():
    verify();assert read(O/'review.json')['passed'];assert not (O/'publication_receipt.json').exists()
    rows=read(O/'quality_evidence_ledger.json');out=[]
    for old in rows:
        r=dict(old)
        # Preserve the original generated ledger; make the distinction explicit in v2.
        del r['nominal_cost_supervision_known']
        r['quality_outcome_known']=r['quality_evidence_label'] in ['PASS_COMPLETE','FAIL_COST','FAIL_SURVIVAL']
        r['nominal_cost_comparison_known']=r['wholeweek_cost_comparable']
        r['single_omission_quality_pass']=True if r['quality_evidence_label']=='PASS_COMPLETE' else False if r['quality_evidence_label'].startswith('FAIL_') else None
        out.append(r)
    save('quality_training_evidence_v2.json',out)
    schema=dict(passed=True,original_ledger_retained=sha(O/'quality_evidence_ledger.json'),
        explanation='Original nominal_cost_supervision_known actually denoted composite quality known, including survival failure without comparable complete cost. v2 separates quality_outcome_known and nominal_cost_comparison_known. No values, thresholds or categories changed.',
        quality_known=sum(r['quality_outcome_known'] for r in out),cost_comparison_known=sum(r['nominal_cost_comparison_known'] for r in out),
        old_labels_unchanged=True,source_hashes_verified=True)
    assert schema['quality_known']==47 and schema['cost_comparison_known']==46
    save('derived_schema_clarification.json',schema)
    passed=[r for r in out if r['single_omission_quality_pass'] is True]
    changed=[r for r in passed if not r['physical_path_identical']]
    failures=[r for r in out if r['single_omission_quality_pass'] is False]
    def query_counts(rr):return dict(collections.Counter('saving' if r['queries_saved']>0 else 'increase' if r['queries_saved']<0 else 'equal' for r in rr))
    groups=[]
    for cohort in sorted({r['cohort'] for r in out}):
        rr=[r for r in out if r['cohort']==cohort]
        groups.append(dict(cohort=cohort,cases=len(rr),statuses=dict(collections.Counter(r['quality_evidence_label'] for r in rr)),
            physical_path_identical=sum(r['physical_path_identical'] for r in rr)))
    known=[r for r in out if r['quality_outcome_known'] and not r['pilot_known']]
    faildates=sorted({r['calendar_date'] for r in known if r['single_omission_quality_pass'] is False})
    folds=[]
    for date in sorted({r['calendar_date'] for r in known}):
        tr=[r for r in known if r['calendar_date']!=date];te=[r for r in known if r['calendar_date']==date]
        folds.append(dict(held_out_date=date,train_pass=sum(r['single_omission_quality_pass'] is True for r in tr),
            train_fail=sum(r['single_omission_quality_pass'] is False for r in tr),test_pass=sum(r['single_omission_quality_pass'] is True for r in te),test_fail=sum(r['single_omission_quality_pass'] is False for r in te)))
    assert len(faildates)==1
    detail=dict(passed=True,all_cases=56,calendar_dates=len({r['calendar_date'] for r in out}),scenario_environment_groups=len({(r['cohort'],r['week']) for r in out}),
        qualified_cases=len(passed),qualified_identical=sum(r['physical_path_identical'] for r in passed),qualified_action_changed=len(changed),
        qualified_query_outcomes=query_counts(passed),qualified_changed_query_outcomes=query_counts(changed),
        qualified_changed_cost_outcomes=dict(collections.Counter('higher' if r['skip_cost_minus_completion']>0 else 'lower' if r['skip_cost_minus_completion']<0 else 'equal' for r in changed)),
        failures=[{k:r[k] for k in ['case_id','week','step','pilot_known','quality_evidence_label','skip_steps','completion_steps','skip_quality','skip_cost_minus_completion','queries_saved','matched_frame_skip_horizon']} for r in failures],
        by_cohort=groups,nonpilot_known=len(known),nonpilot_failure_dates=faildates,nonpilot_date_folds=folds,
        caution='Query differences are complete episode counts only for matched horizons; they are not actual timing. All events are single omissions followed by original FRAME, not a many-skip policy.',
        no_new_training_or_predictions=True)
    save('training_feasibility.json',detail)
    text='''# GRID137：当前标签能否支持下一轮学习

本轮核对了119份既有运行记录、236600个控制步，没有重新训练或回放。56个单次取消案例全部找到原先审核的FULL参照，阈值继续使用整周成本不超过FULL的100.5%，并要求完成情况不退化。

## 查清的结果

56例来自21个场景/环境组、17个日历日期，不是56个独立场景。45例取消后完成整周且通过成本标准，1例提前结束，1例成本超标；另9例因FULL本身未完成，只能比较相对存活，不能作为整周成本达标正例。

45个通过案例中，30个物理轨迹完全相同，15个改变了轨迹。后15个仍有8个取消后整周成本更高、7个更低；“通过质量标准”并不等于“成本更优”。45个通过案例中，整周查询减少41个、增加3个、不变1个。因此学习目标也不能只剩“质量达标”这一个标签，还应区分是否有计算收益；查询减少不等于已验证真实运行提速。

## 两个明确不能取消的案例

- S025：2035-07-09，第1901步。原FRAME完成2017步，取消后1906步终止；提前结束的累计成本不能当省成本。它是此前已知试点。
- S028：2035-12-10，第1170步。取消后仍完成，但整周成本相对FULL上升1.10844%，超过原0.5%门槛；相对FRAME增加26727.485，整周查询还增加359次。这是非试点记录。

原11个MIXED中，10个通过了论文的完成/成本筛查，只有S028未通过。MIXED描述的是多指标权衡，直接把它全当“应保留补查”会改变任务。9个参照未完成的案例也不能因为原标签是SKIP，就填成完整周正例。原标签与风险向量完整保留。

## 旧模型回查说明什么

按原日历留组、原0.5读数，旧逻辑回归与随机森林都会建议跳过S028；所以并非换一个评价口径就把过去模型变好了。

原0.95读数下，逻辑回归建议跳过13例，其中11例有整周通过证据、2例仅相对存活可比；随机森林仅跳过1例且通过。旧0.5和0.95的全部模型/常数参照均保留，未新选阈值。这些只是既有预测的事后核对，旧概率预测的是原Pareto类别，不是整周质量概率，不能据此宣布新模型成功或可靠部署。

## 当前真正的训练瓶颈

按整周业务目标，明确失败只有两个日期，其中一个还是已知试点。排除三个试点后，53例里只有1个失败、43个通过、9个参照不足。若按日期留组，把唯一失败日期留作评价，训练部分就没有任何质量失败例。这样的数据无法认真检验“能否识别不能取消的状态”。更多同一路径前缀不能补足这个缺口，随机拆行会掩盖它。

因此现在不应直接用这56例训练一个二分类模型，然后用高准确率声称学会了补查价值。这里说明的是现有业务标签的覆盖不足，不是证明公开状态不可学习，也不是认定神经网络或强化学习一定无效。

## 下一阶段的具体任务建议（尚未执行）

动作先固定为“当前这一次完整补查保留或取消”，随后均继续原FRAME。目标分开记录：是否损失完成质量、整周成本及查询变化。整周FULL参照与成本余量只可用于事后答案，不能作为运行时输入；先前即时收益记录仍作辅助证据，不能冒充反事实整周答案。

如继续采集，应在采集前固定独立日期分组与状态抽取规则，取得同一干预前缀下的保留/取消成对结果，完整保留无收益、失败及参照不足。需要覆盖多个失败日期，而不是反复采同一个已知失败状态、择失败结果或把一例复制给多批动作。若自然采集仍没有足够失败覆盖，应如实报告无法验证识别能力，不能事后选样救结果。

这一步涉及新的物理采集和规模预算，需要单独确定方案；本轮没有启动。最终8测试仍未生成，原模型、阈值、数据和论文不变。
'''
    (O/'阶段结论与训练建议.md').write_text(text,encoding='utf8')
    save('decision.json',dict(decision='QUALITY_MAPPING_COMPLETE_BUSINESS_FAILURE_COVERAGE_INSUFFICIENT',
        ready_to_train_from_these_labels_alone=False,ready_to_scale=False,adopted=False,
        quality_known=47,pass_complete=45,fail_survival=1,fail_cost=1,reference_limited=9,
        nonpilot_known=44,nonpilot_failures=1,nonpilot_failure_dates=1,
        next_step='Specify prospective grouped paired business-label collection before training;requires separate scale/budget decision.',
        prohibited_leakage='FULL wholeweek cost/headroom,branch outcome and future control are labels or audit-only,never online features.',
        no_new_fit=True,no_new_physical=True))
    verify()
    save('publication_receipt.json',dict(passed=True,review_sha256=sha(O/'review.json'),feasibility_sha256=sha(O/'training_feasibility.json'),
        decision_sha256=sha(O/'decision.json'),new_fits=0,new_model_forwards=0,new_physical_replays=0,
        original_labels_models_predictions_unchanged=True,derived_schema_clarification=sha(O/'derived_schema_clarification.json'),final_tests_absent=True))
    collab=R/'论文协作资料/FRAME_整周质量标签核查_GRID137';collab.mkdir(parents=True,exist_ok=False)
    for n in ['GRID137_report.md','阶段结论与训练建议.md','quality_training_evidence_v2.json','training_feasibility.json','old_new_label_crosswalk.json','saved_readout_crosscheck.json','review.json','decision.json','publication_receipt.json','derived_schema_clarification.json']:
        shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    save('collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
    dest=A/'publication';dest.mkdir(parents=True,exist_ok=False)
    for p in [Path(__file__),*O.iterdir()]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    # Preserve the replaced action list with the publication record.
    shutil.copy2(REPO/'docs/NEXT_STEPS.md',dest/'NEXT_STEPS_before_GRID137.md')
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
    entry='''

## 2026-10-23 — GRID137既有单次取消的整周质量映射完成

119运行/236600步仅保存数据核对，全部56例按GRID91既有cohort/week/environment对应FULL，原0.5%门槛未变。45完成成本通过、1存活失败(S025试点)、1成本失败(S028非试点,+1.10844% FULL)、9因FULL未完仅相对存活可比；原SKIP/MIXED/KEEP与风险向量不变。45通过中30物理等价/15改变轨迹，查询41减少/3增加/1不变；不能称全部有计算优势。原11MIXED为10通过/1成本失败。非试点53只有1失败日期、43通过、9未知，按日期留出唯一失败后训练无失败例，不支持直接据此验证业务质量分类器。旧LR/RF原0.5均误放S028；原0.95 LR13建议含11通过2参照不足、RF仅1通过，完整保留全部读数，不事后声称模型成功。v2派生字段明确47复合质量已知与46完整成本可比，原误命名字段与收据保留，无类别/数值改动。零拟合/前向/物理；最终8封存。后续需预先固定成对业务标签采集、按日期隔离、明确预算再训练，不自动扩大；FULL周成本余量不可作为在线输入。报告：research_archive/power-grid-2026/protocols/GRID137_quality_mapping/publication/阶段结论与训练建议.md
'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n
        original=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':
            title,rest=original.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(original+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text('''# Next steps — 整周业务标签与学习任务

1. GRID137已完成56单次取消按原FULL+0.5%质量标准映射：45通过/2失败/9参照不足；不是56个独立场景，不直接混入2378个即时教师前缀。
2. 当前业务失败只跨2个日期，非试点只1个。现有数据单独不足以检验按日期泛化识别失败；不直接重训二分类或扩大网络。原0.5读数LR/RF误放成本失败也保留，不能只把旧MIXED改口径就称学习成功。
3. 下一阶段若采集：固定单次完整补查保留/取消、随后继续FRAME，预先固定状态抽取与独立日期分组、对照与总预算；完整保存成对周质量及查询/风险。新物理采集前告知用户并确定规模；不因正负比例调样救结果。
4. 标签/输入隔离：FULL整周成本、质量剩余空间、分支结果不能进入在线特征。质量合格与计算获益分开；逐批停止、多次在线取消需要另外证据，单次收益不可相加。
5. 最终8测试未生成；本轮无拟合/前向/物理/论文修改。保留原135全部种子、旧代理目标失败与所有旧标签/校准/模型，不恢复旧自动化。
''',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md'
    p.write_text(p.read_text(encoding='utf8')+'\n- [GRID137整周质量标签映射](protocols/GRID137_quality_mapping/publication/阶段结论与训练建议.md)：56既有单次取消45通过/2失败/9参照不足；失败日期覆盖不足，分开质量与计算收益，零训练/回放。\n',encoding='utf8')
    for phase in ['frozen','complete','publication']:
        root=A/phase
        for n,h in read(root/'archive_manifest.json').items():assert sha(root/n)==h,(phase,n)
    verify();print(json.dumps(detail,ensure_ascii=False));print('GRID137_PUBLICATION_VERIFIED')

if __name__=='__main__':main()
