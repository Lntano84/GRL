"""Explain the locked class-mass experiment without adding a candidate policy."""
from support import *
def main():
    verify_trained();assert read(O/'review.json')['passed'];assert not (O/'post_diagnosis.json').exists();sums=read(O/'summary.json');s=lambda name:next(r for r in sums if r['evaluation_partition']=='validation' and r['method']==name)
    old=s('RF140_CALIBRATED');new=s('RF144_CALIBRATED');fixed=s('RF144_FIXED_0.5');oldfixed=s('RF140_FIXED_0.5');risk=next(r for r in read(O/'continue_risk_metrics.json') if r['partition']=='validation');oldrisk=next(r for r in read(DIAG/'ranking_metrics.json') if r['model']=='RF140' and r['partition']=='validation');cond=next(r for r in read(O/'conditional_ranking_summary.json') if r['partition']=='validation');oldcond=next(r for r in read(DIAG/'conditional_ranking_summary.json') if r['partition']=='validation' and r['model']=='RF140')
    history=read(O/'calibration_history.json');h=history[-2];critical=max(h['bad_stops'],key=lambda r:r['score']);assert critical['score']==read(O/'cutoff.json')['primary_threshold'];ref=next(r for r in refs() if r['call_id']==critical['call_id']);c=B.call_at(ref['source']);truth={v['pool_id']:v for v in c['outcomes']};n=critical['global_queries'];early=truth[critical['selected']];original=truth[critical['original']];gap=B.P.key(original)[0]-B.P.key(early)[0]
    model=read(O/'model_export.json');savedrows=eligible_data('train');r=next(r for r in savedrows if r['meta']['call_id']==critical['call_id'] and r['meta']['visible_count']==n);pre=read(O/'preprocessing.json');score=float(exported_scores(model,transform([r],pre['features'],pre['medians']))[0]);assert abs(score-critical['score'])<1e-10
    badfolds=[r for r in read(O/'all_calibration_date_folds.json') if r['heldout_summary']['action_differences']]
    result=dict(scope='Read-only interpretation of already locked outputs,not a new model or threshold.',
        continue_ap=dict(old=oldrisk['continue_ap'],new=risk['continue_ap']),conditional_auc=dict(old=oldcond['within_depth_branch_auc'],new=cond['auc']),
        default_action_differences=dict(old=oldfixed['action_differences'],new=fixed['action_differences']),default_query_macro=dict(old=oldfixed['date_macro_query_reduction'],new=fixed['date_macro_query_reduction']),
        primary_action_differences=new['action_differences'],primary_queries=new['queries'],primary_pooled_query_saving=1-new['queries']/new['original_queries'],primary_macro_query_saving=new['date_macro_query_reduction'],
        calibration_bottleneck=dict(call_id=critical['call_id'],global_queries=n,score=critical['score'],early_choice=critical['selected'],original_choice=critical['original'],early_rho=early['rho'],original_rho=original['rho'],signed_reward_gap=gap,
            previous_cutoff=h['threshold'],previous_stops=h['summary']['model_stops'],previous_bad_calls=len(h['bad_stops']),final_cutoff=history[-1]['threshold'],final_stops=history[-1]['summary']['model_stops'],final_bad_calls=len(history[-1]['bad_stops']),public_features=r['features'],future_values_are_labels_only=True),
        date_robustness_failures=[dict(omitted_date=r['omitted_date'],action_differences=r['heldout_summary']['action_differences'],calls=r['heldout_summary']['calls']) for r in badfolds],
        new_fits_total=1,additional_fits=0,no_cutoff_or_label_change=True,new_physical=0)
    save('post_diagnosis.json',result)
    lines=['# GRID144：权重调整改善了什么，为什么仍未通过','',
        '本轮只有一份新模型，系数由拟合数据唯一确定，没有验证后重调。保留原RF140，不采用新模型。','',
        '## 有改善，但不是最终目标的改善','',
        f"开发验证继续类AP从{oldrisk['continue_ap']:.4f}升到{risk['continue_ap']:.4f}，同查询深度/初始分支内AUC从{oldcond['within_depth_branch_auc']:.4f}升到{cond['auc']:.4f}。默认0.5不再对全部前缀提议停止；完整搜索动作差异由16/62降为12/62。",'',
        '因此，本次权重干预确实改变了学习器的行为，少数继续样本的排序有所改善，不能笼统说完全没有学到。但0.5下仍有12次动作差异，不能采用这个读数；它相对旧模型也多花了查询，没有建立同质量计算优势。','',
        '## 最难的高分错误仍在支配统一阈值','',
        f"四个校准日期在阈值{h['threshold']:.9f}时有{h['summary']['model_stops']}次提前停止，仍有{len(h['bad_stops'])}次不保持原动作。最高分错误来自{critical['call_id']}的{n}查询处，分数为{critical['score']:.9f}。此时返回候选{critical['selected']}、预测最大负载率{early['rho']:.6f}，原FRAME返回{critical['original']}，一步奖励差为{gap:.9f}。当前负载率较低，仍不能据此排除原搜索后来找到不同的更优候选。",'',
        f"按原校准规则把阈值提高至{history[-1]['threshold']:.9f}后，校准动作差异变为0，但提前停止从{h['summary']['model_stops']}次降为{history[-1]['summary']['model_stops']}次。不能为了保留这些停止机会而在验证后放宽原动作保持目标，也不能据一步奖励差断言整周成本。",'',
        '开发验证同样只剩1次提前停止，少查64个候选；总量减少约0.267%，日期宏平均减少0.155%。62次均保持原动作，但查询节省低于原模型4.14%，也低于固定256的8.08%。数字下降不是计算口径改变，所有口径均保留。','',
        '## 校准日期稳定性也没有解决','',
        '留出十月校准日期、只用其它三个日期确定诊断阈值时，十月13次搜索出现2次动作差异。其它三个留出日期没有动作差异。原RF140相同敏感性诊断在留出十月时有1次差异。不能凭有限日期的零错误校准把分数当作可靠风险保证，也没有选择一个诊断阈值替换主模型。','',
        '## 下一步判断','',
        '这轮排除了“只要把继续样本权重提高，就能得到足够查询收益”的简单解释。继续调整另一个系数、增加摘要或扩大同一森林，目前缺乏依据。下一步应把高分端错误的区分能力与停止风险控制机制分开研究；若需要改变校准方法或训练目标，必须作为新的明确研究方案，先说明依据和评价，再运行，原目标与失败结果不改。本轮不追加训练，也不进入实际FRAME状态回放或扩大规模。']
    (O/'权重干预结果解释.md').write_text('\n'.join(lines)+'\n',encoding='utf8');verify_trained();archive('diagnosis')
    collab=R/'论文协作资料/FRAME_类别损失训练_GRID144'
    for name in ['post_diagnosis.json','权重干预结果解释.md']:shutil.copy2(O/name,collab/name)
    entry=f'\n\nGRID144结果解释：继续AP {oldrisk["continue_ap"]:.4f}→{risk["continue_ap"]:.4f}、同深度/分支AUC {oldcond["within_depth_branch_auc"]:.4f}→{cond["auc"]:.4f}、固定0.5动作差异16→12，说明权重确实改变学习行为；但统一零错校准受{critical["call_id"]}在192处分数{critical["score"]:.9f}的高分错误支配，13次停止/2错收紧至1次/0错，开发仅少64查询、宏平均0.155%。十月校准留出2/13差异，稳定性未解决，不改验证阈值或目标。保留RF140；不再自动尝试另一个权重或特征，下一路线应先区分高分端错误识别与停止风险控制并形成明确依据，无新拟合/物理。\n'
    for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf8')+entry,encoding='utf8')
    p=REPO/'docs/NEXT_STEPS.md';p.write_text(p.read_text(encoding='utf8')+'\n5. 权重改善了中间排序和默认动作差异，但主校准仅1次提前停止。停止自动尝试其它系数/摘要；下一步先研究高分尾部错误区分与停止风险控制，明确新方案依据后再运行，不改目标或验证阈值救结果。\n',encoding='utf8')
    files=0
    for phase in A.iterdir():
        if phase.is_dir():
            for name,h in read(phase/'archive_manifest.json').items():assert sha(phase/name)==h;files+=1
    for p in collab.iterdir():assert p.is_file() and sha(p)==sha(O/p.name)
    verify_trained();save('post_diagnosis_verification.json',dict(passed=True,archive_files_verified=files,collaboration_files_verified=len(list(collab.iterdir())),one_new_fit=True,additional_fits=0,original_data_models_cutoffs_labels_unchanged=True,final_tests_absent=True))
    d=A/'post_verification';d.mkdir(exist_ok=False);shutil.copy2(O/'post_diagnosis_verification.json',d/'post_diagnosis_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'post_diagnosis_verification.json':sha(d/'post_diagnosis_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID144_DIAGNOSIS_COMPLETE',files)
if __name__=='__main__':main()
