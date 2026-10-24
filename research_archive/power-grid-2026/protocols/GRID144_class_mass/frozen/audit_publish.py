"""Independent weight-only audit and whole-search evaluation publication."""
from run import *
def manual_continue_ap(rows,pp):
    y=[not r['target']['stop_label'] for r in rows];n=sum(y)
    if not n:return None
    order=sorted(range(len(rows)),key=lambda i:pp[i]);seen=0;tp=0;value=0.;j=0
    while j<len(order):
        end=j+1
        while end<len(order) and pp[order[end]]==pp[order[j]]:end+=1
        add=sum(y[order[k]] for k in range(j,end));tp+=add;seen+=end-j;value+=(tp/seen)*(add/n);j=end
    return value
def main():
    verify_trained();assert read(O/'state.json')['stage']=='validated_pending_audit';assert not (O/'review.json').exists();D=read(O/'design.json');cut=read(O/'cutoff.json');pre=read(O/'preprocessing.json')
    fit=rowfile(O/'fit_rows.json.gz');cal=rowfile(O/'calibration_rows.json.gz');val=eligible_data('validation');rows=fit+cal+val;parts=['fit']*386+['calibration']*198+['validation']*142
    assert sha(O/'fit_rows.json.gz')==sha(PREV/'fit_rows.json.gz') and sha(O/'calibration_rows.json.gz')==sha(PREV/'calibration_rows.json.gz');assert public_checks(rows)==726
    assert D['features']==read(PREV/'design.json')['features'] and D['model_parameters']==read(PREV/'design.json')['model_parameters']
    assert pre==read(PREV/'preprocessing.json')
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in D['features']] for r in fit]);median=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(18)];assert median==pre['medians']
    w0=np.asarray([r['weight'] for r in read(PREV/'fit_sample_weights.json')]);y=np.asarray([r['target']['stop_label'] for r in fit],bool);w=np.asarray([r['weight'] for r in read(O/'fit_sample_weights.json')]);expected=np.asarray([w0[i]*w0.sum()/(2*w0[y==y[i]].sum()) for i in range(386)])
    assert np.max(abs(w-expected))<1e-12 and abs(w[y].sum()-193)<1e-10 and abs(w[~y].sum()-193)<1e-10
    assert read(O/'fit_base_sample_weights.json')==read(PREV/'fit_sample_weights.json');assert read(O/'fit_sample_weights.json')==read(O/'planned_new_fit_sample_weights.json')
    for label in [False,True]:assert np.max((w/w0)[y==label])-np.min((w/w0)[y==label])<1e-12
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    with (PREV/'model.pkl').open('rb') as f:old=pickle.load(f)
    assert est.get_params()==old.get_params() and est.n_features_in_==old.n_features_in_==36
    for a,b in zip(est.estimators_samples_,old.estimators_samples_):assert np.array_equal(a,b)
    z=np.load(O/'train_scores.npz');stored=np.concatenate([z['fit'],z['calibration'],np.load(O/'validation_scores.npz')['scores']]);manual=exported_scores(read(O/'model_export.json'),transform(rows,pre['features'],pre['medians']));assert max(abs(manual-stored))<1e-10
    allmap=score_map(rows,stored);history=read(O/'calibration_history.json');tau=.5
    for i,h in enumerate(history):
        assert h['iteration']==i and h['threshold']==tau;rr=[replay(r,'MODEL',tau,allmap) for r in refs('train',set(D['calibration_dates']))];bad=[r for r in rr if r['bad_learned_stop']]
        assert summarize(rr)==h['summary'] and {r['call_id'] for r in bad}=={r['call_id'] for r in h['bad_stops']}
        if i<len(history)-1:assert bad;tau=max(r['stopped_score'] for r in bad)
        else:assert not bad
    assert tau==cut['primary_threshold'];assert read(A/'trained/cutoff.json')==cut and sha(A/'trained/model.pkl')==sha(O/'model.pkl') and not (A/'trained/validation_started.json').exists()
    pub=read(O/'prevalidation_publication.json');assert pub['passed'] and pub['local_sha']==pub['remote_sha'] and pub['utc']<read(O/'validation_started.json')['utc']
    risk=read(O/'continue_risk_metrics.json');daterisk=read(O/'continue_risk_per_date.json')
    for r in risk+daterisk:
        ids=[i for i,p in enumerate(parts) if p==r['partition'] and ('date' not in r or rows[i]['meta']['date']==r['date'])];rr=[rows[i] for i in ids];ap=manual_continue_ap(rr,stored[ids]);assert ap is None and r['continue_ap'] is None or ap is not None and abs(ap-r['continue_ap'])<1e-12
    conditional=read(O/'same_depth_branch_ranking.json');cond_summary=[]
    for r in conditional:
        ids=[i for i,p in enumerate(parts) if p==r['partition'] and rows[i]['meta']['visible_count']==r['visible_count'] and rows[i]['features']['prefix_safe']==r['prefix_safe']];pos=[float(stored[i]) for i in ids if rows[i]['target']['stop_label']];neg=[float(stored[i]) for i in ids if not rows[i]['target']['stop_label']]
        assert r['pairs']==len(pos)*len(neg) and r['correct']==sum(a>b for a in pos for b in neg) and r['ties']==sum(a==b for a in pos for b in neg)
    for part in ['fit','calibration','validation']:
        ss=[r for r in conditional if r['partition']==part and r['pairs']];pairs=sum(r['pairs'] for r in ss);cond_summary.append(dict(partition=part,pairs=pairs,auc=sum(r['correct']+.5*r['ties'] for r in ss)/pairs))
    save('conditional_ranking_summary.json',cond_summary)
    folds=read(O/'all_calibration_date_folds.json');hist=read(O/'all_calibration_date_histories.json');fold_calls=read(O/'all_omitted_calibration_call_replays.json')
    for fold in folds:
        omit=fold['omitted_date'];dates=set(D['calibration_dates'])-{omit};donors={(r['meta']['call_id'],r['meta']['visible_count']):float(z['calibration'][i]) for i,r in enumerate(cal) if r['meta']['date'] in dates};tests={(r['meta']['call_id'],r['meta']['visible_count']):float(z['calibration'][i]) for i,r in enumerate(cal) if r['meta']['date']==omit};assert not set(donors)&set(tests)
        tau2=.5;hh=next(r['history'] for r in hist if r['omitted_date']==omit)
        for i,h in enumerate(hh):
            assert h['threshold']==tau2;rr=[replay(r,'MODEL',tau2,donors) for r in refs('train',dates)];bad=[r for r in rr if r['bad_learned_stop']];assert summarize(rr)==h['summary']
            if i<len(hh)-1:assert bad;tau2=max(r['stopped_score'] for r in bad)
            else:assert not bad
        assert tau2==fold['diagnostic_cutoff'];rr=[replay(r,'MODEL',tau2,tests) for r in refs('train',{omit})];assert summarize(rr)==fold['heldout_summary']
        saved={r['call_id']:r for r in fold_calls if r['omitted_date']==omit}
        for r in rr:assert all(r[k]==saved[r['call_id']][k] for k in r)
    records=read(O/'all_search_replays.json');assert len(records)==305*11;refsmap={r['call_id']:r for r in refs()};prior={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(OTHER/'all_search_replays.json')};checks=collections.Counter();errors=[]
    for r in records:
        c=B.call_at(refsmap[r['call_id']]['source']);n=r['global_queries'];before=B.P.best(c['outcomes'][:n]);initial=B.P.best(c['outcomes'][:128]);extra=[]
        if before is not None and n<len(c['order']) and (initial is None or initial['rho']>=.9 or B.P.bid(initial)!=B.P.bid(before)):
            station=B.subids()[before['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if B.subids()[int(alias)]==station};extra=[v for v in c['outcomes'][n:] if v['pool_id'] in ids]
        chosen=B.P.best(c['outcomes'][:n]+extra);assert B.P.bid(chosen)==r['selected'] and n+len(extra)==r['queries'] and len(extra)==r['local_queries'];assert n<=refsmap[r['call_id']]['global_n'];checks['complete_searches']+=1
        if r['method'].startswith('RF144'):
            threshold=.5 if r['method'].endswith('0.5') else tau
            for q in r['model_decisions']:assert q['score']==allmap[r['call_id'],q['visible_count']] and q['stop']==(q['score']>threshold);checks['new_model_decisions']+=1
            if not r['action_match'] or r['queries']>r['original_queries']:errors.append(dict(method=r['method'],partition=r['evaluation_partition'],call_id=r['call_id'],date=r['date'],queries=r['queries'],original_queries=r['original_queries'],global_queries=n,score=r['stopped_score'],threshold=threshold,selected=r['selected'],original_selected=r['original_selected'],public_features=B.P.input_at(c,n)['features']))
        else:
            previous=prior[r['evaluation_partition'],r['method'],r['call_id']];assert all(r[k]==previous[k] for k in ['queries','global_queries','local_queries','selected','model_decisions']);checks['old_reference_rows_unchanged']+=1
    save('all_new_model_action_or_query_failures.json',errors);verify_trained()
    save('review.json',dict(passed=True,new_fits=read(O/'fit_receipt.json')['fits'],old_fits=0,only_fit_loss_weights_change=True,raw_features_labels_rows_dates_preprocessing_parameters_same=True,bootstrap_samples_equal=True,
        class_mass_balanced=True,within_class_relative_weights_preserved=True,overall_date_call_weight_totals_change=True,fit_only_factors=True,public_suffix_checks=726,manual_score_checks=726,max_score_error=float(max(abs(manual-stored))),
        continue_ap_groups_checked=len(risk+daterisk),calibration_reexecuted=True,date_sensitivity_folds_checked=4,checks=dict(checks),frozen_model_cutoff_published_before_validation=True,no_new_physical=True,final_tests_absent=True))
    sums=read(O/'summary.json');s=lambda part,name:next(r for r in sums if r['evaluation_partition']==part and r['method']==name);a=s('validation','RF144_CALIBRATED');oldmain=s('validation','RF140_CALIBRATED');batch=s('validation','RF142_CALIBRATED');cap=s('validation','CAP256');decision=read(O/'decision.json');weighting=read(O/'weighting_receipt.json');valrisk=next(r for r in risk if r['partition']=='validation');oldrisk=next(r for r in read(DIAG/'ranking_metrics.json') if r['model']=='RF140' and r['partition']=='validation');condval=next(r for r in cond_summary if r['partition']=='validation')
    lines=['# GRID144：只改变类别损失的一次训练','',
        '本轮回到RF140原18项特征和18个缺失标记，只调整拟合损失中的类别质量。原始记录、标签、日期、缺失处理、64树/深度4/最小叶20/种子20261022、搜索执行和校准程序保持不变。旧模型直接复用，不重训，不使用RF142新增6项。','',
        f"拟合仍为12日期386前缀（331停、55继续），校准4日期198前缀。原分层权重下继续总质量占{100*weighting['base_continue_mass_fraction']:.2f}%；停止样本乘{weighting['stop_multiplier']:.12f}，继续乘{weighting['continue_multiplier']:.12f}，两类总质量各193。系数仅由拟合数据唯一确定，未扫描。",'',
        '这保留类内相对权重，但改变总体日期与调用质量，不能称总体仍日期等权。它是训练损失归一化，不是电网经济损失比。重加权后的树分数不当作真实事件概率。','',
        f"完成一次新拟合，耗时{read(O/'fit_receipt.json')['elapsed_s']:.4f}秒。原四日期按整次搜索校准{len(history)}轮，最后阈值{tau:.17g}；权重、预处理、模型和阈值封存推送后，才对开发验证142前缀/62搜索评分。没有第二次拟合或验证后调整阈值。",'',
        '## 开发验证完整搜索','',
        '| 方法 | 原动作差异 | 候选查询 | 查询节省日期宏平均 | 提前停止 |','|---|---:|---:|---:|---:|']
    for name in D['reference_methods']:
        r=s('validation',name);lines.append(f"| {name} | {r['action_differences']}/{r['calls']} | {r['queries']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
    lines += ['',
        '所有CAP参照均带相同原强制停止、无可接纳动作继续和条件补查。日期宏平均先算场景比例、再同日期两份场景平均、最后日期平均；与汇总查询数量的比例权重不同，两者均保留。主读数预定为RF144_CALIBRATED，固定0.5读数完整保留，不能按结果选择较好的读数。','',
        '| 日期 | 新模型动作差异 | 新模型查询 | RF140查询 | CAP256查询 |','|---|---:|---:|---:|---:|']
    for r in decision['per_date']:lines.append(f"| {r['date']} | {r['new_action_differences']} | {r['new_model_queries']} | {r['original_model_queries']} | {r['cap256_queries']} |")
    lines += ['',
        f"主模型动作差异{a['action_differences']}/62，查询节省宏平均{100*a['date_macro_query_reduction']:.3f}%；原RF140为{oldmain['action_differences']}/62、{100*oldmain['date_macro_query_reduction']:.3f}%，RF142为{batch['action_differences']}/62、{100*batch['date_macro_query_reduction']:.3f}%，CAP256为{cap['action_differences']}/62、{100*cap['date_macro_query_reduction']:.3f}%。固定开发筛查通过={decision['passes_development_screen']}。",'',
        '主门槛不变：全部62次保持原动作，查询节省宏平均严格优于CAP256，每日期查询不更多且至少一个日期更少。即使通过，也只具备提出实际FRAME状态覆盖核查的资格，不自动部署、扩大或物理回放；未通过则保留失败，不添加第二套类别系数。','',
        '## 少数继续风险与排序','',
        '| 分区 | 继续类AP | 继续比例 | 继续召回 | 有效停止召回 | 同深度/分支AUC |','|---|---:|---:|---:|---:|---:|']
    for r in risk:
        cond=next(v for v in cond_summary if v['partition']==r['partition']);lines.append(f"| {r['partition']} | {r['continue_ap']:.4f} | {100*r['continue_prevalence']:.2f}% | {100*r['continue_recall']:.2f}% | {100*r['valid_stop_recall']:.2f}% | {cond['auc']:.4f} |")
    lines += ['',
        f"开发验证继续类AP：RF140 {oldrisk['continue_ap']:.4f}，RF144 {valrisk['continue_ap']:.4f}。对应继续比例为22/142=15.49%。原RF140同深度/分支AUC为0.5719，新模型为{condval['auc']:.4f}。这些是原路径前缀辅助指标，不等于整次策略结果；同调用前缀和同日期重复不是独立场景，不能只凭继续召回较高称模型成功。",'',
        '## 校准日期敏感性：不采用诊断阈值','',
        '固定本轮模型，每次留出一个原校准日期，仅用其余三个日期按原程序确定诊断阈值，检查留出日期。全部四组保留，只用校准记录，未将其阈值用于开发验证或替换主阈值。','',
        '| 留出校准日期 | 三日期诊断阈值 | 留出动作差异 | 留出提前停止 |','|---|---:|---:|---:|']
    for r in folds:lines.append(f"| {r['omitted_date']} | {r['diagnostic_cutoff']:.9f} | {r['heldout_summary']['action_differences']}/{r['heldout_summary']['calls']} | {r['heldout_summary']['model_stops']} |")
    lines += ['',
        '这是已暴露记录的稳定性诊断，不是独立泛化保证，不能据此选择一个更有利的阈值。主模型仍使用事前封存的四日期阈值。','',
        '## 拟合与校准完整读数','',
        '| 分区 | 读数 | 原动作差异 | 查询节省宏平均 | 提前停止 |','|---|---|---:|---:|---:|']
    for part in ['fit','calibration']:
        for name in ['RF140_FIXED_0.5','RF140_CALIBRATED','RF144_FIXED_0.5','RF144_CALIBRATED']:
            r=s(part,name);lines.append(f"| {part} | {name} | {r['action_differences']}/{r['calls']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
    lines += ['',
        '## 审计与范围','',
        '386项新权重按拟合类别质量独立重算，类内比例和两类质量通过；旧原始特征、标签、记录字节、预处理、模型参数及bootstrap抽样均一致。726前缀隐藏后缀核对、独立分数遍历、四日期校准与四个日期敏感性重放通过；3355次完整搜索的查询/补查集合、最优已观察动作及新旧参照记录复核通过。全部动作或查询反例保存。','',
        '同一开发验证日期已反复暴露，本轮不是新的独立验证。固定状态来自FULL教师，实际FRAME路径不同；保持原动作不等同整周成本通过，查询数量不等同实测完整控制时间。最终8测试场景未生成，没有新物理仿真、网络/RL、论文改动或扩大规模。']
    (O/'GRID144_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed_audit=True,new_fits=1,old_fits=0,development_screen_passed=decision['passes_development_screen'],adopted=False,ready_to_scale=False,new_physical=0,final_tests_absent=True));save('state.json',dict(stage='complete'));verify_trained();archive('complete')
    c=R/'论文协作资料/FRAME_类别损失训练_GRID144';c.mkdir(exist_ok=False);names=['GRID144_report.md','design.json','planned_weighting.json','weighting_receipt.json','fit_receipt.json','cutoff.json','continue_risk_metrics.json','continue_risk_per_date.json','conditional_ranking_summary.json','all_calibration_date_folds.json','summary.json','decision.json','review.json','complete.json']
    for n in names:shutil.copy2(O/n,c/n)
    entry=f'''\n\n## 2026-10-24 — GRID144单次类别损失干预完成\n\n用户继续并训练前通知。回原18特征/36维，仅将拟合损失两类质量各193，停止乘0.546904435964/继续乘5.829986276655，类内相对权重保留、总体日期调用质量改变；系数只取原386拟合标签和基权重，无扫描。原64树/深4/叶20/种子、日期、标签、预处理、bootstrap、搜索/校准不变，旧模型不重训，一次新拟合。主开发验证差异{a['action_differences']}/62、查询节省宏平均{100*a['date_macro_query_reduction']:.3f}%，RF140 {100*oldmain['date_macro_query_reduction']:.3f}%、CAP256 {100*cap['date_macro_query_reduction']:.3f}%，原筛查={decision['passes_development_screen']}。继续AP {valrisk['continue_ap']:.4f}、同深度/分支AUC {condval['auc']:.4f}；全部4校准日期留出敏感性仅诊断不采用。权重/阈值于新验证评分前封存推送，独立权重/726分数与后缀/3355搜索审计通过，原0.5及全部反例保留。不因结果换系数/标签/阈值，最终8未生成，非独立验证/整周成本/实测时间，不自动扩大。报告research_archive/power-grid-2026/protocols/GRID144_class_mass/complete/GRID144_report.md\n'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;oldtext=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=oldtext.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(oldtext+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text(f'''# Next steps — 类别损失一次干预后\n\n1. GRID144已完成原18特征、单一拟合内类别质量平衡，一次新拟合；主动作差异{a['action_differences']}/62、少查{100*a['date_macro_query_reduction']:.3f}%，原筛查={decision['passes_development_screen']}。继续AP {valrisk['continue_ap']:.4f}、条件AUC {condval['auc']:.4f}完整保留。\n2. 若通过，仅提出实际FRAME保存状态覆盖核查；未通过则先解释现有错误与排序/校准代价，不加第二个系数、标签或验证阈值救结果，不自动新训练。\n3. RF140/RF142/CAP及两读数全部保留，旧模型未重训；类内权重相对不变但总体日期质量改变，不冒称日期全等权或物理损失比。\n4. 最终8封存，重复开发评价不当独立验证，查询不当实测提速、动作保持不当整周质量；无新物理/RL/论文/扩大。\n''',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+f'\n- [GRID144类别损失一次训练](protocols/GRID144_class_mass/complete/GRID144_report.md)：原18特征、两类质量平衡，开发动作差异{a["action_differences"]}/62、少查{100*a["date_macro_query_reduction"]:.3f}%，原筛查{decision["passes_development_screen"]}。\n',encoding='utf8')
    count=0
    for phase in A.iterdir():
        if phase.is_dir():
            for n,h in read(phase/'archive_manifest.json').items():assert sha(phase/n)==h;count+=1
    for n in names:assert sha(c/n)==sha(O/n)
    verify_trained();save('final_verification.json',dict(passed=True,archive_files_verified=count,collaboration_files_verified=len(names),one_new_fit=True,old_models_original_data_unchanged=True,weights_cutoff_unchanged=True,final_tests_absent=True))
    d=A/'verification';d.mkdir(exist_ok=False);shutil.copy2(O/'final_verification.json',d/'final_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID144_COMPLETE',json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':main()
