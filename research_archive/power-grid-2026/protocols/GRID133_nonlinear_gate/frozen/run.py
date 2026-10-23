"""One fixed nonlinear full-completion risk gate; matched expensive-branch control."""
from pathlib import Path
import collections, datetime, gzip, hashlib, json, math, os, pickle, shutil, sys, time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
import numpy as np
R=Path('D:/Codex/GRL研究'); O=R/'outputs/grid133_nonlinear_gate'; W=Path(__file__).parent
P=R/'outputs/grid129_baseline_risk'; B=R/'outputs/grid130_calibrated_gate'; C=R/'outputs/grid132_conditional_gate'
REPO=R/'work/GRL-results-integration'; A=REPO/'research_archive/power-grid-2026/protocols/GRID133_nonlinear_gate'
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x): (O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def archive(phase):
    d=A/phase; d.mkdir(parents=True,exist_ok=False)
    for p in [*O.iterdir(),Path(__file__)]:
        if p.is_file(): shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def verify():
    for n in ['source_hashes.json','code_freeze.json']:
        for p,h in read(O/n).items(): assert sha(R/p)==h,p
    sealed()
def data():
    with np.load(P/'train_public_inputs.npz') as z: X=z['X'][:,:37].astype(float); ids=z['sample_ids']; q=z['known_queries']
    with gzip.open(P/'train_label_ledger.jsonl.gz','rt',encoding='utf8') as f: rows=[json.loads(l) for l in f]
    with np.load(B/'train_calibration_scores.npz') as z: fit=z['fitmask']; cal=z['calmask']
    assert all(str(i)==r['sample_id'] for i,r in zip(ids,rows))
    return X,q,rows,fit,cal
def prepare():
    assert not O.exists(); O.mkdir(parents=True); sealed(); assert read(C/'review.json')['passed']
    forest=read(R/'outputs/grid101_gain_learning/design.json')['forest']
    D=dict(id='GRID133_NONLINEAR_GATE',status='frozen_before_one_fit',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User asks continue autonomous research; one fixed forest announced before training.',
        question='Can a shallow nonlinear classifier improve expensive full-completion skip decisions with exactly the same37 public inputs?',
        model='RandomForestClassifier, binary full completion gain>0; no feature selection or tuning.',
        forest_parameters=forest, classifier_defaults=dict(criterion='gini',bootstrap=True,class_weight=None),new_fits=1,
        provenance='Forest structural parameters from GRID101 full-gain regressor; classifier changes target/output and model family relative to LR. Not an untried algorithm, nor a parameter-count/compute-matched architecture ablation.',
        fit='Same445 Q>64 original12-date cases and weights as GRID132;193 original4-date calibration cases.',
        features='Exactly original first37 float32 summaries; no geometry addition, scaler not needed by tree thresholds.',
        cheap_branch='Original GRID130 scores and GRID132 cheap threshold unchanged.',
        calibration='Same max no-improvement probability over gain-positive or action-different calibration witnesses; strict greater SKIP, ties fallback, no witnesses fallback.',
        reference=['FULL_LOCAL','BATCH1','GLOBAL','GROUP','EXP_PUBLIC37','EXP_COUNT1'],
        screens=read(C/'design.json')['primary_screen'], incremental_screen=read(C/'design.json')['incremental_screen'],
        order='Only train/calibration inputs in preparation/fit; freeze weights and thresholds before new validation scoring.',
        public_information_audit='Count exact37-vector label collisions and next-batch coverage on train/cal only; no claim absence of collisions proves sufficiency, nor scope mismatch proves new representation will work.',
        budget='One64-tree fit; wall cap120s detected and preserved, no retries, seeds/splits/threshold scan or extra fits.',
        scope='All4validation dates exposed development, zero physical calls/new data/final8tests; local reward increments not whole-week economic cost or actual speed.',
        prohibited=['refit','parameter/seed search','new geometry model in this unit','new validation calibration','physical rollout','scaleup','paper/figures'])
    save('design.json',D)
    paths=[P/n for n in ['train_public_inputs.npz','train_label_ledger.jsonl.gz']]+[B/n for n in ['train_calibration_scores.npz','validation_scores.npz','model.joblib']]
    paths += [C/n for n in ['fits.json','validation_results.json','validation_trajectories.json','training_started.json']]
    paths += [R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid101_gain_learning/design.json',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'work/grid101_gain_learning/features.py',R/'work/grid112_relational_plan/schema.py',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths}); save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))})
    X,q,rows,fit,cal=data(); want={r['sample_id']:i for i,r in enumerate(rows)}; public={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            # Public input archive is partition-mixed; do not inspect validation payloads.
            r=json.loads(line)
            if r['sample_id'] not in want or r['position']!=0: continue
            c=r['public_input']; public[r['sample_id']]=len(c['next_ids'])
    assert len(public)==2378
    audit={}
    for name,mask in [('fit',fit),('calibration',cal)]:
        ii=np.flatnonzero(mask&(q>64)); groups=collections.defaultdict(list)
        for i in ii: groups[X[i].astype(np.float32).tobytes()].append(i)
        conflict=[g for g in groups.values() if len({rows[i]['full_baseline_gain']>0 for i in g})>1]
        nq=np.array([public[rows[i]['sample_id']] for i in ii]); assert (nq==64).all()
        audit[name]=dict(expensive_prefixes=len(ii),all_pending_queries=int(q[ii].sum()),old_next_batch_candidates=int(nq.sum()),
            beyond_old_next_batch=int((q[ii]-nq).sum()),exact_summary_groups=len(groups),conflicting_exact_summary_groups=len(conflict),
            conflicting_sample_ids=[[rows[i]['sample_id'] for i in g] for g in conflict])
    save('information_audit.json',dict(cohorts=audit,models_fitted=0,
        findings=['37 summaries retain reward/rho statistics and station coverage but no action partition or association of each paid feedback with a partition.',
            'GRID112/113/116/119 relational inputs include all paid local feedback but ONLY the next<=64 unseen plans; current gate is all remaining completion.',
            'Next-batch input is scope-limited for this new target, but omitted candidate information is public at decision time; no evidence yet that it predicts the full tail.',
            'Prior relational failures and weak pairing-permutation response remain counterevidence; do not claim relations have not been tried.',
            'No exact collisions does not establish predictive sufficiency; artificial collisions or absence of train overlap do not prove impossibility.']))
    verify(); save('state.json',dict(stage='frozen_before_one_fit',fits=0)); archive('frozen'); print('GRID133_FROZEN')
def fit():
    verify(); assert not (O/'training_started.json').exists()
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    X,q,rows,fitmask,calmask=data(); ii=np.flatnonzero(fitmask&(q>64)); cc=np.flatnonzero(calmask&(q>64))
    counts=collections.Counter(r['date'] for r,m in zip(rows,fitmask) if m); weights=np.array([1/counts[rows[i]['date']] for i in ii]); weights/=weights.mean()
    previous=read(C/'training_started.json'); assert previous['fit_ids']==[rows[i]['sample_id'] for i in ii] and previous['calibration_ids']==[rows[i]['sample_id'] for i in cc]
    assert np.array_equal(weights,previous['weights'])
    y=np.array([r['full_baseline_gain']>0 for r in rows]); bad=np.array([r['full_baseline_gain']>0 or r['action_changed_if_skip'] for r in rows])
    D=read(O/'design.json'); save('training_started.json',dict(fits_planned=1,fit_ids=previous['fit_ids'],calibration_ids=previous['calibration_ids'],weights=weights.tolist()))
    t=time.perf_counter(); model=RandomForestClassifier(**D['forest_parameters'],**D['classifier_defaults']).fit(X[ii],y[ii],sample_weight=weights); elapsed=time.perf_counter()-t
    joblib.dump(model,O/'EXP_RF37.joblib'); assert elapsed<=120,'Preserve completed fit; do not repeat on cap failure'
    score=1-model.predict_proba(X)[:,list(model.classes_).index(True)]; lam=float(score[cc][bad[cc]].max()) if bad[cc].any() else 1.
    assert not ((score[cc]>lam)&bad[cc]).any()
    save('fits.json',dict(fits=1,fit_seconds=elapsed,fit_cases=len(ii),calibration_cases=len(cc),parameters=model.get_params(),
        threshold=dict(cheap=read(C/'fits.json')['thresholds']['GROUP']['cheap'],expensive=lam),fit_ids_weights_match_grid132=True,
        trees=len(model.estimators_),total_nodes=sum(e.tree_.node_count for e in model.estimators_),model_sha256=sha(O/'EXP_RF37.joblib')))
    np.savez_compressed(O/'train_calibration_scores.npz',score=score)
    save('evaluation_freeze.json',{n:sha(O/n) for n in ['EXP_RF37.joblib','fits.json','train_calibration_scores.npz','design.json']})
    save('state.json',dict(stage='frozen_before_validation',fits=1)); print(json.dumps(read(O/'fits.json'),ensure_ascii=False))
def manual(model,X):
    # Independently traverse serialized tree arrays; do not call predict/apply/decision_path.
    prob=np.zeros(len(X)); X=X.astype(np.float32)
    for est in model.estimators_:
        t=est.tree_
        for i,x in enumerate(X):
            node=0
            while t.children_left[node]!=-1: node=t.children_left[node] if x[t.feature[node]]<=t.threshold[node] else t.children_right[node]
            value=t.value[node,0]; prob[i]+=value[1]/value.sum()
    return 1-prob/len(model.estimators_)
def evaluate():
    verify(); assert read(O/'state.json')['stage']=='frozen_before_validation'; assert not (O/'evaluation_started.json').exists()
    import joblib
    for n,h in read(O/'evaluation_freeze.json').items(): assert sha(O/n)==h
    save('evaluation_started.json',dict(weights_thresholds_frozen=True))
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='validation']; assert len(cases)==590
    X=np.stack([c['states'][0,:37] for c in cases]).astype(float); Q=np.array([int(c['queries'].sum()) for c in cases]); exp=Q>64
    model=joblib.load(O/'EXP_RF37.joblib'); s=1-model.predict_proba(X[exp])[:,1]; err=float(abs(s-manual(model,X[exp])).max()); assert err<1e-12
    with np.load(B/'validation_scores.npz') as z: scores=z['score'].copy(); ids=z['sample_ids']
    assert all(str(i)==c['sample_id'] for i,c in zip(ids,cases)); scores[exp]=s
    T=read(O/'fits.json')['threshold']; D=read(R/'outputs/grid102_search_rl/design.json'); results=[]
    for i,c in enumerate(cases):
        k=len(c['gains']); skip=bool(scores[i]>T['expensive' if exp[i] else 'cheap']); p=0 if skip else k; G=math.fsum(float(x) for x in c['gains']); gain=0. if skip else G; paid=0 if skip else int(Q[i])
        results.append(dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group'],method='EXP_RF37',expensive=bool(exp[i]),batches=p,
            gain=gain,gain_lost=G-gain,queries=paid,full_gain=G,full_queries=int(Q[i]),chosen_id=c['chosen'][p],baseline_id=c['chosen'][-1],
            action_disagreement=c['chosen'][p]!=c['chosen'][-1],net_utility=(gain-D['cost_per_query']*paid)/D['gain_scale']))
    old=read(C/'validation_results.json'); per={}
    for date in sorted({c['date'] for c in cases}):
        rr=[r for r in results if r['date']==date]; gain=math.fsum(r['full_gain'] for r in rr); qp=sum(r['queries'] for r in rr)/sum(r['full_queries'] for r in rr); loss=math.fsum(r['gain_lost'] for r in rr)/gain if gain else 0.
        per[date]=dict(prefixes=len(rr),gain_loss=loss,gain_capture=1-loss,query_fraction=qp,query_saving=1-qp,action_disagreements=sum(r['action_disagreement'] for r in rr),utility=float(np.mean([r['net_utility'] for r in rr])))
    v=dict(per_date=per,macro={k:float(np.mean([d[k] for d in per.values()])) for k in ['gain_loss','gain_capture','query_fraction','query_saving','utility']},
        total_action_disagreements=sum(r['action_disagreement'] for r in results),expensive_skipped=sum(r['expensive'] and not r['queries'] for r in results),expensive_disagreements=sum(r['expensive'] and r['action_disagreement'] for r in results))
    old['EXP_RF37']=v; primary=v['total_action_disagreements']==0 and v['macro']['query_saving']>old['GLOBAL']['macro']['query_saving']
    dominance=all(per[d]['gain_loss']<=old['GROUP']['per_date'][d]['gain_loss'] and per[d]['query_fraction']<=old['GROUP']['per_date'][d]['query_fraction'] for d in per) and any(per[d]['query_fraction']<old['GROUP']['per_date'][d]['query_fraction'] for d in per)
    # Recompute every new row from saved prefix choices; cheap branch must exactly replay.
    group={r['sample_id']:r for r in read(C/'validation_trajectories.json') if r['method']=='GROUP'}; checks=0
    for c,r in zip(cases,results):
        p=r['batches']; assert r['queries']==sum(int(x) for x in c['queries'][:p]); assert abs(r['gain']-sum(float(x) for x in c['gains'][:p]))<1e-12
        assert r['chosen_id']==c['chosen'][p]
        if not r['expensive']: assert all(r[k]==group[r['sample_id']][k] for k in ['queries','gain_lost','chosen_id'])
        checks+=1
    # Auditable train/calibration separation without selecting by validation.
    Xt,q,rows,fitmask,calmask=data()
    with np.load(O/'train_calibration_scores.npz') as z: ts=z['score']
    trainerr=float(abs(ts-manual(model,Xt)).max()); assert trainerr<1e-12
    diagnostics={}
    from sklearn.metrics import roc_auc_score
    with np.load(C/'train_calibration_scores.npz') as z: lr=z['EXP_PUBLIC37']
    for name,m in [('fit',fitmask),('calibration',calmask)]:
        m=m&(q>64); y=np.array([r['full_baseline_gain']>0 for r in rows]); bad=np.array([r['full_baseline_gain']>0 or r['action_changed_if_skip'] for r in rows]); skip=ts>T['expensive']
        diagnostics[name]=dict(prefixes=int(m.sum()),RF_auc=float(roc_auc_score(y[m],1-ts[m])),LR_auc=float(roc_auc_score(y[m],1-lr[m])),
            skipped=int((m&skip).sum()),skipped_positive=int((m&skip&bad).sum()),skipped_queries=int(q[m&skip].sum()),full_queries=int(q[m].sum()),
            max_positive_no_gain_score=float(ts[m&bad].max()),zero_gain_above_all_positives=int((m&~bad&(ts>ts[m&bad].max())).sum()))
    save('train_calibration_diagnostics.json',diagnostics); save('validation_results.json',old); save('validation_trajectories.json',results)
    np.savez_compressed(O/'validation_scores.npz',score=scores,sample_ids=ids)
    save('decision.json',dict(primary_screen=bool(primary),incremental_pareto_vs_group=bool(dominance),adopted=False,ready_to_scale=False,
        reason='Fixed development diagnostic; no automatic deployment even if a local screen passes.'))
    verify(); save('review.json',dict(passed=True,new_fits=1,case_arithmetic_checks=checks,independent_tree_traversal_validation_error=err,independent_tree_traversal_train_error=trainerr,
        fit_ids_weights_match_grid132=True,cheap_branch_exact=True,all_weights_thresholds_frozen_before_validation=True,all_prior_results_preserved=True,
        physical_prediction_calls=0,new_data=0,final_tests_absent=True,source_hashes_unchanged=True))
    lines=['# GRID133：同信息非线性门控与输入范围核查','',
        '本轮只训练一份固定随机森林分类器，检验37维摘要中的非线性关系能否改善大补查识别。445条拟合、193条校准前缀、原12/4日期、样本权重及校准规则与GRID132完全相同；小补查分支不变。',
        '森林结构沿用GRID101的64棵树、深度4、叶节点至少20条、全部特征、固定种子20261022；本轮改为完整补查是否有增量的分类任务，不是原回归模型复用。模型家族和参数量不同，不能把差异全部归因于非线性。','',
        '## 原四个开发验证日期','', '| 方法 | 补查增量保留 | 查询节省 | 动作差异/590 | 局部净效用 |','|---|---:|---:|---:|---:|']
    for arm in ['FULL_LOCAL','BATCH1','GLOBAL','GROUP','EXP_PUBLIC37','EXP_COUNT1','EXP_RF37']:
        a=old[arm]; m=a['macro']; lines.append(f"| {arm} | {100*m['gain_capture']:.5f}% | {100*m['query_saving']:.3f}% | {a['total_action_disagreements']} | {m['utility']:.8f} |")
    lines+=['','比例先按日期汇总再宏平均。查询节省只指同站补查；增量指预测奖励增量，净效用含历史代理查询费，均不是整周成本或实际控制提速。', '',
        f"固定大组校准门槛{T['expensive']:.12f}；大组跳过{v['expensive_skipped']}条，动作差异{v['expensive_disagreements']}条。主筛查通过={primary}；逐日期共同改善GROUP={dominance}。不采用、不扩大。",'', '## 拟合与校准诊断','']
    for name,a in diagnostics.items(): lines.append(f"- {name}：RF AUC {a['RF_auc']:.4f}，匹配LR {a['LR_auc']:.4f}；跳过{a['skipped']}条，其中正收益/动作变化{a['skipped_positive']}条，省查询{a['skipped_queries']}/{a['full_queries']}。")
    lines+=['','AUC只描述排序，不代表极端保护门槛下的计算覆盖；校准零损失是构造性质，无统计风险保证。','', '## 逐日期与不利结果','']
    for date,a in per.items(): lines.append(f"- {date}：增量保留{100*a['gain_capture']:.5f}%，查询省{100*a['query_saving']:.3f}%，动作差异{a['action_disagreements']}。")
    lines+=['','## 输入范围与下一步','',
        '现有37维摘要有奖励/负载率统计、搜索进展和站内覆盖，缺少候选接线结构及结构与逐项已购反馈的对应。旧关系模型保留了配对信息，但未评估候选结构只包括下一批最多64个，训练目标也是下一批净收益；本轮是整段补查的跳过决策。',
        '这说明两个研究问题不同，不说明旧关系网络只要换目标就能成功。旧JOINT/FACTOR、固定主干及配对扰动的失败必须保留；本轮没有新训练任何关系网络。',
        '下一步如研究结构信息，应先仅用公开接线与已购反馈构建完整剩余候选输入，检查首批以外的结构是否提供补充证据，再冻结同样目标和容量匹配的对照。不能直接重复旧模型或根据验证错例设计新特征。',
        '全部日期已多轮暴露，前缀在同一物理调用内相关。本轮没有新物理预测/回放、数据生成、最终测试、论文或图表。独立树遍历、保存动作/查询/增量、小分支重现及来源哈希核验通过。']
    (O/'GRID133_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('state.json',dict(stage='complete',fits=1)); save('complete.json',dict(passed=True,new_fits=1,adopted=False,ready_to_scale=False)); archive('complete')
    print(json.dumps(dict(RF=v,diagnostics=diagnostics,primary_screen=primary,pareto=dominance),ensure_ascii=False))
if __name__=='__main__': {'prepare':prepare,'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
