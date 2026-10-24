"""Audit one frozen fit and whole searches,then report all fixed readouts."""
from common import *
from run import rowfile,score_map
def main():
    verify_trained();assert read(O/'state.json')['stage']=='validated_pending_audit';assert not (O/'review.json').exists();D=read(O/'design.json');pre=read(O/'preprocessing.json');cut=read(O/'cutoff.json')
    fit=rowfile('fit_rows.json.gz');cal=rowfile('calibration_rows.json.gz');val=eligible_data('validation');rows=fit+cal+val
    trainz=np.load(O/'train_scores.npz');valz=np.load(O/'validation_scores.npz');stored=np.concatenate([trainz['fit'],trainz['calibration'],valz['scores']]);assert len(rows)==len(stored)==726
    assert not set(D['fit_dates'])&set(D['calibration_dates']);assert not (set(D['fit_dates'])|set(D['calibration_dates']))&set(D['validation_dates'])
    raw=np.asarray([[np.nan if r['features'][k] is None else float(r['features'][k]) for k in D['features']] for r in fit],float)
    med=[float(np.median(raw[np.isfinite(raw[:,j]),j])) if np.isfinite(raw[:,j]).any() else 0. for j in range(len(D['features']))];assert med==pre['medians']
    model=read(O/'model_export.json');xx=transform(rows,pre['features'],pre['medians']);manual=exported_scores(model,xx);assert np.max(abs(manual-stored))<1e-10
    public_checks=0
    for r in rows:
        c=call_at(r['meta']['source']);x=P.input_at(c,r['meta']['visible_count']);assert x['features']==r['features'];public_checks+=1
    weights=read(O/'fit_sample_weights.json');assert len(weights)==386
    for date in D['fit_dates']:assert abs(sum(r['weight'] for r in weights if r['date']==date)-386/12)<1e-10
    for call in {r['call_id'] for r in weights}:
        ww=[r for r in weights if r['call_id']==call];assert max(r['weight'] for r in ww)-min(r['weight'] for r in ww)<1e-12
    allmap=score_map(rows,stored);calrefs=refs('train',set(D['calibration_dates']));history=read(O/'calibration_history.json');tau=.5
    for j,h in enumerate(history):
        assert h['iteration']==j and h['threshold']==tau;rr=[replay(r,'MODEL',tau,allmap) for r in calrefs];bad=[r for r in rr if r['bad_learned_stop']]
        assert summarize(rr)==h['summary'];assert {r['call_id'] for r in bad}=={r['call_id'] for r in h['bad_stops']}
        if j<len(history)-1:assert bad;tau=max(r['stopped_score'] for r in bad)
        else:assert not bad
    assert tau==cut['primary_threshold'];assert not (A/'trained/validation_started.json').exists()
    assert read(A/'trained/cutoff.json')==cut and read(A/'trained/model_export.json')==model
    refsbyid={r['call_id']:r for r in refs()};replays=read(O/'all_search_replays.json');assert len(replays)==305*7;errors=[];checks=collections.Counter()
    for r in replays:
        c=call_at(refsbyid[r['call_id']]['source']);n=r['global_queries'];seen=c['outcomes'][:n];before=max((t for t in seen if t['strict_admissible']),key=lambda t:(float(np.float32(t['reward'])),-t['pool_id']),default=None)
        extra=[];prefix=max((t for t in c['outcomes'][:min(128,len(c['order']))] if t['strict_admissible']),key=lambda t:(float(np.float32(t['reward'])),-t['pool_id']),default=None)
        safe=prefix is not None and prefix['rho']<.9
        if before is not None and n<len(c['order']) and (not safe or P.bid(before)!=P.bid(prefix)):
            station=subids()[before['pool_id']];ids={int(rep) for alias,rep in c['aliases'].items() if subids()[int(alias)]==station};extra=[t for t in c['outcomes'][n:] if t['pool_id'] in ids]
        paid=seen+extra;selected=max((t for t in paid if t['strict_admissible']),key=lambda t:(float(np.float32(t['reward'])),-t['pool_id']),default=None)
        assert len(paid)==r['queries'] and len(extra)==r['local_queries'] and P.bid(selected)==r['selected'];assert n<=refsbyid[r['call_id']]['global_n'];checks['complete_searches']+=1
        if r['method'].startswith('RF'):
            threshold=.5 if r['method']=='RF_FIXED_0.5' else tau
            for decision in r['model_decisions']:
                assert decision['score']==allmap[(r['call_id'],decision['visible_count'])] and decision['stop']==(decision['score']>threshold);checks['model_decisions']+=1
            if not r['action_match'] or r['queries']>r['original_queries']:
                stopn=r['global_queries'];features=P.input_at(c,stopn)['features']
                errors.append(dict(method=r['method'],evaluation_partition=r['evaluation_partition'],call_id=r['call_id'],date=r['date'],visible_count=stopn,threshold=threshold,score=r['stopped_score'],selected=r['selected'],original=r['original_selected'],queries=r['queries'],original_queries=r['original_queries'],public_features=features))
    save('all_model_action_or_query_failures.json',errors)
    verify_trained();save('review.json',dict(passed=True,fits=read(O/'fit_receipt.json')['fits'],trees=len(model['trees']),
        independent_probability_checks=726,max_probability_error=float(np.max(abs(manual-stored))),raw_public_feature_checks=public_checks,checks=dict(checks),
        fit_only_medians_weights=True,calendar_isolation=True,whole_call_calibration_reexecuted=True,trained_snapshot_before_validation=True,
        all_readouts_retained=True,no_new_physical=True,final_tests_absent=True))
    sums=read(O/'summary.json');decision=read(O/'decision.json');s=lambda part,name:next(r for r in sums if r['evaluation_partition']==part and r['method']==name)
    calibrated=s('validation','RF_CALIBRATED');fixed=s('validation','RF_FIXED_0.5');cap=s('validation','CAP256');fit_receipt=read(O/'fit_receipt.json');node=read(O/'validation_node_metrics.json');trainmetrics=read(O/'training_node_metrics.json')
    lines=['# GRID140：一次浅层森林的全局搜索压缩训练','',
        '任务固定为：在原FRAME的128/64全局检查点尝试提前停止，然后保留原条件同站补查。原强制停止点保留，没有可接纳动作时继续原搜索。目标是在这些固定状态上保持原FRAME动作并减少候选查询。','',
        '本轮使用已安装scikit-learn随机森林，完成一次拟合。64棵树、深度4、最小叶节点20、原种子20261022与其它森林参数逐项沿用GRID133，不做模型、种子、参数或特征筛选。18个原公开摘要加18个缺失标记，共36维；缺失中位数仅用拟合日期。','',
        '## 拟合、校准与验证顺序','',
        f"原12个拟合日期386前缀（331停/55继续），4个校准日期198前缀（153停/45继续）。每日期总权重相同，同日期内每调用相同，同调用内前缀相同。拟合耗时{fit_receipt['elapsed_s']:.4f}秒。验证为原4个已暴露日期、8场景、62次完整搜索，142个可学习前缀。",'',
        f"固定0.5读数保留作辅助。主读数从0.5开始，按校准完整搜索中的错误停止逐次提高阈值；严格大于才停止，等分继续。共{len(history)}轮，最终阈值{tau:.17g}。模型、预处理、阈值与校准收据在新验证评分前封存并推送。没有第二次拟合、回训、变更费用或验证调阈值。",'',
        '校准仅说明四个已知校准日期上重放没有错误停止，不是统计安全保证或业务安全证书。阈值收紧后重新执行整次搜索，后面原先未访问的决策会被重新检查；不是把逐行准确率当整条策略结果。','',
        '## 验证完整搜索结果','',
        '| 方法 | 与原FRAME动作不同 | 候选查询 | 相对原FRAME查询减少宏平均 | 模型提前停止次数 |',
        '|---|---:|---:|---:|---:|']
    for name in D['reference_methods']:
        r=s('validation',name);lines.append(f"| {name} | {r['action_differences']}/{r['calls']} | {r['queries']} | {100*r['date_macro_query_reduction']:.3f}% | {r['model_stops']} |")
    lines += ['',
        'CAP128/192/256/512均保留相同补查、没有可用动作继续和原强制停止；不是普通不带补查的固定前缀策略。全部参照与两种模型读数保留，主结论固定使用RF_CALIBRATED，不看验证后选择较好的读数。查询宏平均先按场景计算，再同日期两份场景平均，最后日期平均；查询总量与宏平均比例不是相同权重。','',
        '## 四个日期全部保留','',
        '| 日期 | 校准模型动作差异 | 校准模型查询 | CAP256查询 | 校准模型查询减少 |','|---|---:|---:|---:|---:|']
    capdates={r['date']:r for r in cap['by_date']}
    for r in calibrated['by_date']:lines.append(f"| {r['date']} | {r['action_differences']} | {r['queries']} | {capdates[r['date']]['queries']} | {100*r['query_reduction']:.3f}% |")
    lines += ['', '## 拟合与校准结果','', '| 分区 | 读数 | 动作差异 | 模型提前停止 | 查询减少宏平均 |','|---|---|---:|---:|---:|']
    for part in ['fit','calibration']:
        for name in ['RF_FIXED_0.5','RF_CALIBRATED']:
            r=s(part,name);lines.append(f"| {part} | {name} | {r['action_differences']}/{r['calls']} | {r['model_stops']} | {100*r['date_macro_query_reduction']:.3f}% |")
    lines += ['',
        f"拟合节点AUC {trainmetrics['fit_fixed']['auc']:.4f}，校准节点AUC {trainmetrics['calibration_fixed']['auc']:.4f}，验证节点AUC {node['fixed']['auc']:.4f}。节点多数类准确率{100*node['fixed']['majority_stop_accuracy']:.2f}%，所以不能以较高逐行准确率证明学会了停止。全部节点分类指标、置信分数和完整搜索路径分别保存。",'',
        '## 固定筛查判读','',
        f"主模型验证动作差异={calibrated['action_differences']}；查询宏平均减少={100*calibrated['date_macro_query_reduction']:.3f}%，CAP256={100*cap['date_macro_query_reduction']:.3f}%；固定离线增量筛查通过={decision['passes_offline_screen']}。",'',
        '事前要求：62次验证全部保持原动作，查询宏平均严格优于CAP256，且每个日期查询不更多、至少一个日期更少。无论结果如何，都不改变规则、模型或阈值救结果。即便通过，也只具备下一步实际FRAME状态覆盖核查资格，不自动采集、闭环回放、扩网络或进入RL。','',
        '所有模型动作或查询反例保存在all_model_action_or_query_failures.json，包括拟合、校准与验证两种读数；不挑选一个最好日期、读数或模型种子。','',
        '## 核验与局限','',
        '独立树节点遍历复算全部726项分数，与库实现一致；逐个原始公开前缀核对输入，整次搜索的原强制停止、补查查询集合、最优已观察动作与严格阈值完成核对。按调用重执行全部校准轮次。原冻结记录与所有数据哈希一致，最终8测试场景未生成。','',
        '这些是FULL教师路径上的固定状态，验证也已经用于开发；动作保持不等于学习模型已经改善整周经济成本。候选查询减少不等于完整控制时间按同比例降低。没有新物理仿真、场景或整周质量结果，论文未改。学习模块采用现成随机森林；本轮新增的是任务对应、数据处理、按调用校准与搜索执行逻辑。']
    (O/'GRID140_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    complete=dict(passed_audit=True,fit_count=1,offline_screen_passed=decision['passes_offline_screen'],adopted=False,ready_to_scale=False,new_physical=0,final_tests_absent=True)
    save('complete.json',complete);save('state.json',dict(stage='complete'));save('publication_receipt.json',dict(passed=True,review_sha256=sha(O/'review.json'),decision_sha256=sha(O/'decision.json'),fit_count=1,validation_weights_unchanged=True,new_physical=0,final_tests_absent=True))
    collab=R/'论文协作资料/FRAME_全局压缩一次训练_GRID140';collab.mkdir(parents=True,exist_ok=False)
    names=['GRID140_report.md','design.json','fit_receipt.json','cutoff.json','training_node_metrics.json','validation_node_metrics.json','summary.json','decision.json','review.json','complete.json','publication_receipt.json']
    for n in names:shutil.copy2(O/n,collab/n);assert sha(O/n)==sha(collab/n)
    save('collaboration_receipt.json',dict(directory=collab.relative_to(R).as_posix(),hashes={p.name:sha(p) for p in collab.iterdir()}))
    archive('complete');entry=f'''

## 2026-10-24 — GRID140一次浅层森林全局压缩训练完成

用户继续并已在训练前通知。继承64树/深度4/叶20/种子20261022配置，一次拟合，原12日期386拟合、4日期198校准，18公开摘要加18缺失标记；日期/调用/前缀分层等权。逐完整搜索从0.5按错误停止严格提高阈值，{len(history)}轮最终{tau:.17g}，权重/阈值封存推送后才新评分验证。原强制停止、无可用动作继续和条件补查保留，仅尝试提前停止；同模型0.5辅助读数和全部CAP128/192/256/512参照保留。主校准模型验证动作差异{calibrated['action_differences']}/62，候选查询减少宏平均{100*calibrated['date_macro_query_reduction']:.3f}%，CAP256 {100*cap['date_macro_query_reduction']:.3f}%，逐日期不劣且严格增量固定筛查={decision['passes_offline_screen']}；辅助0.5动作差异{fixed['action_differences']}/62、查询减少{100*fixed['date_macro_query_reduction']:.3f}%。节点AUC拟合/校准/验证{trainmetrics['fit_fixed']['auc']:.4f}/{trainmetrics['calibration_fixed']['auc']:.4f}/{node['fixed']['auc']:.4f}。726分数独立树遍历、公开输入与整次查询/补查集合审计通过；全部反例保存。不是整周质量、实际提速或独立测试，最终8未生成，无新物理/数据/NN/RL/论文变更，不再拟合救结果，不自动扩大。报告research_archive/power-grid-2026/protocols/GRID140_global_compression/complete/GRID140_report.md
'''
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;old=p.read_text(encoding='utf8')
        if n=='RESEARCH_STATE.md':title,rest=old.split('\n',1);p.write_text(title+'\n'+entry+'\n'+rest,encoding='utf8')
        else:p.write_text(old+entry,encoding='utf8')
    nextsteps=f'''# Next steps — 全局压缩训练后的证据

1. GRID140已完成固定浅层森林一次拟合与逐调用校准，主验证动作差异{calibrated['action_differences']}/62，候选查询宏平均减少{100*calibrated['date_macro_query_reduction']:.3f}%；固定离线增量筛查={decision['passes_offline_screen']}。原0.5读数、CAP128/192/256/512与全部反例保留，不重新拟合/改阈值救结果。
2. 若通过，仅做已保存实际FRAME状态的输入/反馈覆盖资格检查，先确定所需候选反馈是否齐备；不自动新物理或扩大。若未通过，先读取全部错误停止与推理分数，解释泛化/校准/简单对照差距，再决定路线，不自动追加模型。
3. 本任务是提前结束原全局搜索、保留原补查与强制停止，监督是原动作保持+查询差；不是整周成本标签，也不是原FULL最优保护或直接RL奖励。节点准确率不能替代整次搜索结果。
4. 当前为已暴露FULL教师固定状态；实际FRAME路径与闭环质量、真实控制时间尚未验证。最终8封存，旧模型/标签/费用/结果及论文不变；不恢复旧自动化。
'''
    (REPO/'docs/NEXT_STEPS.md').write_text(nextsteps,encoding='utf8')
    p=REPO/'research_archive/power-grid-2026/README.md';p.write_text(p.read_text(encoding='utf8')+f'\n- [GRID140全局压缩一次训练](protocols/GRID140_global_compression/complete/GRID140_report.md)：一次森林+逐调用校准，验证动作差异{calibrated["action_differences"]}/62、查询减少{100*calibrated["date_macro_query_reduction"]:.3f}%，离线增量筛查{decision["passes_offline_screen"]}；零新物理。\n',encoding='utf8')
    files=0
    for root in A.iterdir():
        if root.is_dir():
            for n,h in read(root/'archive_manifest.json').items():assert sha(root/n)==h;files+=1
    for n,h in read(O/'collaboration_receipt.json')['hashes'].items():assert sha(collab/n)==h
    verify_trained();save('final_verification.json',dict(passed=True,archive_files_verified=files,collaboration_files_verified=len(names),fit_count=1,original_data_unchanged=True,weights_cutoff_unchanged=True,final_tests_absent=True))
    d=A/'verification';d.mkdir(parents=True,exist_ok=False);shutil.copy2(O/'final_verification.json',d/'final_verification.json');(d/'archive_manifest.json').write_text(json.dumps({'final_verification.json':sha(d/'final_verification.json')},indent=2)+'\n',encoding='utf8')
    print('GRID140_COMPLETE',json.dumps(dict(decision=decision,calibrated=calibrated,fixed=fixed),ensure_ascii=False))
if __name__=='__main__':main()
