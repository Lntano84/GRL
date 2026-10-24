"""Audit fixed representation-only intervention and retain all readouts."""
from run import *
def independent_added(c,n):
    tail=c['outcomes'][n-64:n];prior=c['outcomes'][n-128:n-64]
    def best_reward(rr):
        values=[float(np.float32(t['reward'])) for t in rr if t['strict_admissible']]
        return max(values) if values else None
    def positive_rho(rr):return np.asarray([t['rho'] for t in rr if np.isfinite(t['rho']) and t['rho']>0],float)
    a=best_reward(prior);b=best_reward(tail);rho=positive_rho(tail);pr=positive_rho(prior)
    return dict(tail64_best_reward=b,tail64_admissible_fraction=len([t for t in tail if t['strict_admissible']])/64,
        tail64_rho_median=float(np.percentile(rho,50)) if len(rho) else None,
        tail64_rho_iqr=float(np.percentile(rho,75)-np.percentile(rho,25)) if len(rho) else None,
        tail64_vs_prior64_best_reward=b-a if a is not None and b is not None else None,
        tail64_vs_prior64_median_rho=float(np.percentile(rho,50)-np.percentile(pr,50)) if len(rho) and len(pr) else None)
def main():
    verify_trained();assert read(O/'state.json')['stage']=='validated_pending_audit';assert not (O/'review.json').exists();D=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json')
    fit=rowfile('fit_rows.json.gz');cal=rowfile('calibration_rows.json.gz');val=eligible_data('validation');rows=fit+cal+val;assert len(rows)==726;oldD=read(PREV/'design.json')
    assert D['model_parameters']==oldD['model_parameters'] and D['sample_weight']==oldD['sample_weight']
    assert D['fit_dates']==oldD['fit_dates'] and D['calibration_dates']==oldD['calibration_dates'] and D['validation_dates']==oldD['validation_dates']
    assert not set(D['fit_dates'])&set(D['calibration_dates']) and not (set(D['fit_dates'])|set(D['calibration_dates']))&set(D['validation_dates'])
    assert read(O/'fit_sample_weights.json')==read(PREV/'fit_sample_weights.json')
    with (O/'model.pkl').open('rb') as f:est=pickle.load(f)
    with (PREV/'model.pkl').open('rb') as f:oldest=pickle.load(f)
    assert est.get_params()==oldest.get_params();assert len(est.estimators_)==64
    for a,b in zip(est.estimators_samples_,oldest.estimators_samples_):assert np.array_equal(a,b)
    with gzip.open(PREV/'fit_rows.json.gz','rt',encoding='utf8') as f:oldfit=json.load(f)
    with gzip.open(PREV/'calibration_rows.json.gz','rt',encoding='utf8') as f:oldcal=json.load(f)
    oldrows=oldfit+oldcal+B.eligible_data('validation')
    for r,s in zip(rows,oldrows):assert r['meta']==s['meta'] and r['target']==s['target'] and {k:r['features'][k] for k in oldD['features']}==s['features']
    assert public_checks(rows)==726
    for r in rows:
        c=B.call_at(r['meta']['source']);f=independent_added(c,r['meta']['visible_count']);assert {k:r['features'][k] for k in ADDED}==f
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in D['features']] for r in fit],float)
    medians=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(24)];assert medians==pre['medians'];assert medians[:18]==read(PREV/'preprocessing.json')['medians']
    z=np.load(O/'train_scores.npz');stored=np.concatenate([z['fit'],z['calibration'],np.load(O/'validation_scores.npz')['scores']]);xx=transform(rows,pre['features'],pre['medians']);manual=exported_scores(read(O/'model_export.json'),xx);assert xx.shape==(726,48) and max(abs(manual-stored))<1e-10
    fitonly=transform(fit,pre['features'],pre['medians']);oldx=transform(oldfit,oldD['features'],read(PREV/'preprocessing.json')['medians']);assert np.array_equal(fitonly[:,:18],oldx[:,:18]) and np.array_equal(fitonly[:,24:42],oldx[:,18:36])
    allmap=score_map(rows,stored);history=read(O/'calibration_history.json');tau=.5
    for i,h in enumerate(history):
        assert h['iteration']==i and h['threshold']==tau;rr=[replay(r,'MODEL',tau,allmap) for r in refs('train',set(D['calibration_dates']))];bad=[r for r in rr if r['bad_learned_stop']]
        assert summarize(rr)==h['summary'] and {r['call_id'] for r in bad}=={r['call_id'] for r in h['bad_stops']}
        if i<len(history)-1:assert bad;tau=max(r['stopped_score'] for r in bad)
        else:assert not bad
    assert tau==cut['primary_threshold'] and read(A/'trained/cutoff.json')==cut and sha(A/'trained/model.pkl')==sha(O/'model.pkl')
    assert not (A/'trained/validation_started.json').exists();pub=read(O/'prevalidation_publication.json');assert pub['passed'] and pub['local_sha']==pub['remote_sha'] and pub['utc']<read(O/'validation_started.json')['utc']
    records=read(O/'all_search_replays.json');assert len(records)==305*9;refsbyid={r['call_id']:r for r in refs()};checks=collections.Counter();errors=[]
    old={(r['evaluation_partition'],r['method'],r['call_id']):r for r in read(PREV/'all_search_replays.json')}
    for r in records:
        c=B.call_at(refsbyid[r['call_id']]['source']);n=r['global_queries'];seen=c['outcomes'][:n];before=P.best(seen);initial=P.best(c['outcomes'][:128]);safe=initial is not None and initial['rho']<.9;extra=[]
        if before is not None and n<len(c['order']) and (not safe or P.bid(before)!=P.bid(initial)):
            station=B.subids()[before['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if B.subids()[int(alias)]==station};extra=[t for t in c['outcomes'][n:] if t['pool_id'] in ids]
        paid=seen+extra;chosen=P.best(paid);assert len(paid)==r['queries'] and len(extra)==r['local_queries'] and P.bid(chosen)==r['selected'];assert n<=refsbyid[r['call_id']]['global_n'];checks['complete_searches']+=1
        if r['method'].startswith('RF142'):
            threshold=.5 if r['method'].endswith('0.5') else tau
            for v in r['model_decisions']:assert v['score']==allmap[r['call_id'],v['visible_count']] and v['stop']==(v['score']>threshold);checks['new_model_decisions']+=1
            if not r['action_match'] or r['queries']>r['original_queries']:
                errors.append(dict(method=r['method'],partition=r['evaluation_partition'],call_id=r['call_id'],date=r['date'],global_queries=n,queries=r['queries'],original_queries=r['original_queries'],selected=r['selected'],original_selected=r['original_selected'],score=r['stopped_score'],threshold=threshold,public_features=features(c,n)))
        else:
            prev=old[r['evaluation_partition'],r['method'].replace('RF140_','RF_'),r['call_id']]
            assert all(r[k]==prev[k] for k in ['selected','queries','local_queries','global_queries','model_decisions']);checks['old_reference_rows_unchanged']+=1
    save('all_new_model_action_or_query_failures.json',errors)
    sums=read(O/'summary.json');s=lambda part,name:next(v for v in sums if v['evaluation_partition']==part and v['method']==name)
    paired=[]
    for part in ['fit','calibration','validation']:
        new={r['call_id']:r for r in records if r['evaluation_partition']==part and r['method']=='RF142_CALIBRATED'};prior={r['call_id']:r for r in records if r['evaluation_partition']==part and r['method']=='RF140_CALIBRATED'}
        for k,r in new.items():paired.append(dict(partition=part,call_id=k,date=r['date'],new_queries=r['queries'],old_queries=prior[k]['queries'],new_minus_old_queries=r['queries']-prior[k]['queries'],new_preserves_action=r['action_match'],old_preserves_action=prior[k]['action_match']))
    save('all_new_vs_old_calibrated_calls.json',paired)
    review=dict(passed=True,new_fits=read(O/'fit_receipt.json')['fits'],old_fits=0,model_parameters_equal=True,weights_labels_rows_dates_equal=True,bootstrap_samples_equal=True,only_input_representation_changes=True,
        new_raw_features=6,input_dimensions=48,public_suffix_checks=726,independent_added_feature_checks=726,manual_score_checks=726,max_score_error=float(max(abs(manual-stored))),calibration_reexecuted=True,model_cutoff_published_before_validation=True,
        checks=dict(checks),all_readouts_retained=True,no_new_physical=True,final_tests_absent=True)
    verify_trained();save('review.json',review);decision=read(O/'decision.json');a=s('validation','RF142_CALIBRATED');b=s('validation','RF140_CALIBRATED');cap=s('validation','CAP256');node=read(O/'validation_node_metrics.json');oldnode=read(PREV/'validation_node_metrics.json')
    lines=['# GRID142：六项批次信息的一次训练比较','',
        '本轮只检验输入表示：原18项摘要增加6项最近批次统计。森林结构、种子、训练/校准日期、全部标签、分层样本权重、搜索执行和校准程序与GRID140完全一致；旧模型直接复用，没有重训。','',
        '新增信息为最近64候选的最佳可接纳奖励、可接纳比例、正有限负载率中位数和四分位距，以及最近64与前64的最佳奖励差和负载率中位数差。共24项输入与24项缺失标记，缺失中位数只取12个拟合日期。128查询处两段均为已经看到的初始前缀，不额外查询。','',
        f"12个拟合日期386前缀、4个校准日期198前缀，完成一次拟合，耗时{read(O/'fit_receipt.json')['elapsed_s']:.4f}秒。森林为64树、深度4、最小叶节点20、种子20261022。逐调用校准共{len(history)}轮，阈值由0.5按错误停止上调至{tau:.17g}。发布权重与阈值后才评分142个开发验证前缀、重放62次完整搜索。",'',
        '## 开发验证结果','',
        '| 方法 | 原动作差异 | 候选查询 | 相对FRAME查询减少宏平均 | 提前停止次数 |','|---|---:|---:|---:|---:|']
    for name in D['reference_methods']:
        r=s('validation',name);lines.append(f"| {name} | {r['action_differences']}/{r['calls']} | {r['queries']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
    lines += ['',
        '所有固定预算参照均带相同的原强制停止、无可用动作继续和条件补查。查询比例先按场景、再同日期两份场景平均、最后日期宏平均；它与总查询减少比例不是同一权重。两模型的固定0.5读数均保留，主读数事前固定为新模型校准结果。','',
        '| 日期 | 新模型动作差异 | 新模型查询 | 旧模型查询 | CAP256查询 |','|---|---:|---:|---:|---:|']
    for r in decision['per_date']:lines.append(f"| {r['date']} | {r['new_action_differences']} | {r['new_model_queries']} | {r['old_model_queries']} | {r['cap256_queries']} |")
    lines += ['',
        f"新模型校准读数动作差异{a['action_differences']}/62，查询节省宏平均{100*a['date_macro_query_reduction']:.3f}%；旧模型{b['action_differences']}/62、{100*b['date_macro_query_reduction']:.3f}%；CAP256为{cap['action_differences']}/62、{100*cap['date_macro_query_reduction']:.3f}%。事前固定筛查通过={decision['passes_development_screen']}。",'',
        '通过要求为62次全部保持原动作，查询宏平均严格优于CAP256，每日期不更多且至少一日期更少。没有按结果改变特征、权重或阈值。若未通过，保留失败并停止这一轮训练；若通过，也仅允许提出实际FRAME状态覆盖核查，不自动扩大或物理回放。','',
        '## 输入增量与校准代价','',
        '| 分区 | 模型 | 动作差异 | 查询节省宏平均 | 提前停止 |','|---|---|---:|---:|---:|']
    for part in ['fit','calibration']:
        for name in ['RF140_FIXED_0.5','RF140_CALIBRATED','RF142_FIXED_0.5','RF142_CALIBRATED']:
            r=s(part,name);lines.append(f"| {part} | {name} | {r['action_differences']}/{r['calls']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
    tm=read(O/'training_node_metrics.json');oldtm=read(PREV/'training_node_metrics.json')
    lines += ['',
        f"节点AUC：原模型拟合/校准/开发验证为{oldtm['fit_fixed']['auc']:.4f}/{oldtm['calibration_fixed']['auc']:.4f}/{oldnode['fixed']['auc']:.4f}；新模型为{tm['fit_fixed']['auc']:.4f}/{tm['calibration_fixed']['auc']:.4f}/{node['fixed']['auc']:.4f}。这是辅助排序指标，不能替代完整搜索中的动作保持和查询增量。",'',
        f"新模型固定0.5在开发验证的停止提议为{node['fixed']['stop_proposals']}/142，逐行准确率{100*node['fixed']['accuracy']:.2f}%，多数类准确率{100*node['fixed']['majority_stop_accuracy']:.2f}%。校准后提议{node['calibrated']['stop_proposals']}/142。完整校准历史、全部逐调用新旧差异和两种读数的全部动作/查询反例均保存。",'',
        '## 核验和范围','',
        '核对模型参数、每行标签/分区/顺序、样本权重与64棵树bootstrap采样均和旧模型一致；原18项值与缺失标记未变。独立计算新增6项统计，726条公开前缀及隐藏后缀扰动通过；独立树遍历重算726项分数，重执行全部校准轮次，并复核2745次搜索的已付查询集合、补查集合和最优已观察选择。', '',
        '新增特征来自已暴露GRID141记录的分析，本轮重复使用原开发验证日期，因此不是新独立测试。固定状态来自FULL教师，实际FRAME路径不同；动作保持不等同整周经济成本通过，查询量不等同实际完整控制时间。最终8个测试场景未生成，无新物理仿真、论文修改或扩大规模。']
    (O/'GRID142_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('complete.json',dict(passed_audit=True,new_fits=1,old_fits=0,development_screen_passed=decision['passes_development_screen'],adopted=False,ready_to_scale=False,new_physical=0,final_tests_absent=True));save('state.json',dict(stage='complete'));verify_trained();archive('complete')
    collab=R/'论文协作资料/FRAME_批次信息训练_GRID142';collab.mkdir(exist_ok=False);names=['GRID142_report.md','design.json','preflight.json','fit_receipt.json','cutoff.json','training_node_metrics.json','validation_node_metrics.json','summary.json','decision.json','review.json','complete.json']
    for n in names:shutil.copy2(O/n,collab/n)
    entry=f'''\n\n## 2026-10-24 — GRID142六项批次信息一次训练完成\n\n用户继续并在训练前通知。原18摘要加6批次统计与24缺失标记，64树/深度4/叶20/种子不变，原12日期386拟合/4日期198校准，一次新拟合；旧模型不重训。标签、权重、行顺序、日期、bootstrap采样、原搜索与校准全部相同，独立6特征/726后缀及概率/2745完整搜索核验通过。主新校准模型验证动作差异{a['action_differences']}/62、查询节省宏平均{100*a['date_macro_query_reduction']:.3f}%，旧校准{100*b['date_macro_query_reduction']:.3f}%、CAP256 {100*cap['date_macro_query_reduction']:.3f}%；原固定筛查通过={decision['passes_development_screen']}。阈值{tau:.17g}于验证前发布；两种读数和全部反例保留。新增表示基于已暴露开发数据，不是新独立验证、整周质量或实测时间；最终8未生成，不自动追加模型/阈值/物理或扩大。报告research_archive/power-grid-2026/protocols/GRID142_batch_representation/complete/GRID142_report.md\n'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(old+entry,encoding='utf8')
    (REPO/'docs/NEXT_STEPS.md').write_text(f'''# Next steps — 批次表示的一次训练结果\n\n1. GRID142完成一次输入表示干预，主开发验证动作差异{a['action_differences']}/62、少查{100*a['date_macro_query_reduction']:.3f}%，旧模型{100*b['date_macro_query_reduction']:.3f}%，CAP256 {100*cap['date_macro_query_reduction']:.3f}%；固定筛查={decision['passes_development_screen']}。原两种读数和全部反例保留。\n2. 若通过，仅提出实际FRAME保存状态覆盖资格检查；若未通过，先解释现有错误/校准代价，不立即改变权重、加特征或扩大网络救结果。没有自动新训练/物理授权。\n3. 本轮仅原18项加6批次摘要；模型参数、日期、标签、分层权重、bootstrap和搜索/校准都相同，旧模型未重训。来源于开发数据的特征不能把重复开发评价说成独立测试。\n4. 最终8封存，原动作保持不等于整周质量，候选查询不等于实际时间；旧学习失败、未完周和参考边界不变，不改论文。\n''',encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+f'\n- [GRID142六项批次信息训练](protocols/GRID142_batch_representation/complete/GRID142_report.md)：一次表示干预，开发验证动作差异{a["action_differences"]}/62、少查{100*a["date_macro_query_reduction"]:.3f}%，原筛查{decision["passes_development_screen"]}。\n',encoding='utf8')
    files=0
    for phase in A.iterdir():
        if phase.is_dir():
            for n,h in read(phase/'archive_manifest.json').items():assert sha(phase/n)==h;files+=1
    for n in names:assert sha(collab/n)==sha(O/n)
    verify_trained();save('final_verification.json',dict(passed=True,archive_files_verified=files,collaboration_files_verified=len(names),one_new_fit=True,old_model_and_original_data_unchanged=True,weights_cutoff_unchanged=True,final_tests_absent=True))
    d=A/'verification';d.mkdir(exist_ok=False);shutil.copy2(O/'final_verification.json',d/'final_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID142_COMPLETE',json.dumps(decision,ensure_ascii=False))
if __name__=='__main__':main()
