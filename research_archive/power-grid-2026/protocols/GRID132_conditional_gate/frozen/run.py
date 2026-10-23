"""Two fixed expensive-branch LR fits;group-calibration control;no physical calls."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,math,os,pickle,shutil,sys,time
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid132_conditional_gate';P=R/'outputs/grid129_baseline_risk';B=R/'outputs/grid130_calibrated_gate'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID132_conditional_gate'
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*O.iterdir(),Path(__file__)]:
        if p.is_file():shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def verify():
    for fn in ['source_hashes.json','code_freeze.json']:
        for n,h in read(O/fn).items():assert sha(R/n)==h,n
    sealed()
def data():
    with np.load(P/'train_public_inputs.npz') as z:X=z['X'][:,:37].astype(float);ids=z['sample_ids'];q=z['known_queries']
    with gzip.open(P/'train_label_ledger.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(l) for l in f]
    with np.load(B/'train_calibration_scores.npz') as z:s=z['score'];fit=z['fitmask'];cal=z['calmask']
    assert all(str(sid)==r['sample_id'] for sid,r in zip(ids,rows))
    return X,q,rows,s,fit,cal
def threshold(s,bad):return float(s[bad].max()) if bad.any() else 1.

def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(R/'outputs/grid131_expensive_coverage/review.json')['passed']
    D=read(B/'design.json');X,q,rows,s,fit,cal=data()
    assert (q>64)[fit].sum()==445 and (q>64)[cal].sum()==193
    plan=dict(id='GRID132_CONDITIONAL_GATE',status='frozen_before_two_fits',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User asks continue and previously authorizes autonomous ideas with updates. Announced exactly two small expensive-branch LR fits before fitting.',
        question='Separate global-cutoff coupling from expensive-task prediction,with count-only matched control.',
        baseline='Original FRAME full locked-station completion;one gate,otherwise unchanged original completion.',
        groups='Known remaining count Q<=64 versus Q>64,original batch size boundary,no scan.',
        arms=dict(GLOBAL='Reuse original37LR and original global threshold.',GROUP='Reuse identical original scores;calibrate each fixed group separately.',EXP_PUBLIC37='Cheap branch reuses original scores/cheap threshold;expensive new37LR/expensive threshold.',EXP_COUNT1='Same but expensive LR only takes log_completion_queries,feature3.'),
        new_fits=2,model_parameters=D['model_parameters'],fit_dates=D['fit_dates'],calibration_dates=D['calibration_dates'],
        fit_data='Same445 expensive cases from12fitdates for both new models,all193expensive cases from4calibration dates kept.',
        weights='Restrict original global-fit per-case date weights to expensive subset,normalize mean1;identical IDs/weights/scaler roles for both new arms. Do not introduce conditional date reweighting.',
        normalization='New scalers fit only on445expensive fitcases;count model uses same scalar column as public model.',
        targets='Full completion gain>0;empirical calibration protects all positive gains or action differences. No class/loss weight changes.',
        calibration='For each fixed branch and model,lambda=max no-improvement score over positive/different-action calibration witnesses;strict greater SKIP,ties fallback,no witnesses fallback. No validation-selected lambda.',
        order='Fit both final models and calibrate all group thresholds before opening saved original validation cases for new scoring. Freeze all model/threshold hashes first.',
        evaluation='All4known validation dates,allarms andFULL_LOCAL/BATCH1;original GLOBAL exact replay. No physical/new scenarios/final8test.',
        primary_screen='Zero disagreements versus full baseline in all590validation cases and date-macro query saving strictly greater than originalGLOBAL.',
        incremental_screen='For each new expensive arm vsGROUP:gain loss no greater on any4date AND query fraction no greater on any4date,with strictly smaller query on at least1date. Joint quality/computation only,not cherry-picked aggregate.',
        interpretation='Both experiments are exposed development diagnostics,not statistical guarantee,independent test or learning optimality. GROUP changes calibration degrees of freedom. Specialized model changesfitpopulation and compositecapacity,not isolated architecture cause.',
        prohibited=['morethan2fits','refit/extraepoch','groupboundary/feature/modelselection','riskalpha/thresholdscan','validationcalibration','newdata/physics/test','paper/figure','autoscales'])
    save('design.json',plan)
    paths=[P/n for n in ['train_public_inputs.npz','train_label_ledger.jsonl.gz','cross_version_review.json']]
    paths+=[B/n for n in ['model.joblib','train_calibration_scores.npz','validation_scores.npz','validation_results.json','validation_trajectories.json','fits.json','design.json']]
    paths += [R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid131_expensive_coverage/cohorts.json',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))})
    save('state.json',dict(stage='frozen_before_fitting',fits=0));archive('frozen');print('GRID132_FROZEN')

def fit():
    verify();assert not (O/'training_started.json').exists(),'Do not repeat started fits'
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib
    X,q,rows,s,fitmask,calmask=data();exp=q>64;ii=np.flatnonzero(fitmask&exp);cc=np.flatnonzero(calmask&exp)
    allcounts=collections.Counter(r['date'] for r,yes in zip(rows,fitmask) if yes)
    w=np.array([1/allcounts[rows[i]['date']] for i in ii]);w/=w.mean()
    positive=np.array([r['full_baseline_gain']>0 for r in rows]);bad=np.array([r['full_baseline_gain']>0 or r['action_changed_if_skip'] for r in rows])
    D=read(O/'design.json');thresholds=dict(GLOBAL=read(B/'fits.json')['calibrated_lambda'],GROUP=dict(cheap=threshold(s[calmask&~exp],bad[calmask&~exp]),expensive=threshold(s[calmask&exp],bad[calmask&exp])))
    save('training_started.json',dict(new_fits_planned=2,fit_ids=[rows[i]['sample_id'] for i in ii],calibration_ids=[rows[i]['sample_id'] for i in cc],weights=w.tolist()))
    fits=[];cache={}
    for arm,columns in [('EXP_PUBLIC37',list(range(37))),('EXP_COUNT1',[3])]:
        selected=X[:,columns];start=time.perf_counter();scaler=StandardScaler().fit(selected[ii],sample_weight=w)
        model=LogisticRegression(**D['model_parameters']).fit(scaler.transform(selected[ii]),positive[ii],sample_weight=w)
        duration=time.perf_counter()-start;path=O/f'{arm}.joblib';joblib.dump(dict(scaler=scaler,classifier=model,columns=columns),path)
        ss=1-model.predict_proba(scaler.transform(selected))[:,1]
        logits=((selected-scaler.mean_)/scaler.scale_)@model.coef_[0]+model.intercept_[0]
        err=float(abs(1/(1+np.exp(logits))-ss).max());assert err<1e-12
        lam=threshold(ss[cc],bad[cc]);thresholds[arm]=dict(cheap=thresholds['GROUP']['cheap'],expensive=lam)
        assert not ((ss[cc]>lam)&bad[cc]).any()
        fits.append(dict(arm=arm,fit_seconds=duration,iterations=int(model.n_iter_[0]),converged=bool(model.n_iter_[0]<D['model_parameters']['max_iter']),
            fit_cases=len(ii),calibration_cases=len(cc),features=columns,model_sha256=sha(path),numpy_max_error=err))
        cache[arm]=ss;save('fit_progress.json',dict(fits_completed=len(fits),fits=fits))
    save('fits.json',dict(fits=2,fit_ids=[rows[i]['sample_id'] for i in ii],calibration_ids=[rows[i]['sample_id'] for i in cc],models=fits,
        no_validation_used=True,identical_fit_weights=True,thresholds=thresholds))
    np.savez_compressed(O/'train_calibration_scores.npz',**cache)
    save('evaluation_freeze.json',{n:sha(O/n) for n in ['EXP_PUBLIC37.joblib','EXP_COUNT1.joblib','fits.json','train_calibration_scores.npz','design.json']})
    save('state.json',dict(stage='allmodels_thresholds_frozen_before_validation',fits=2));print(json.dumps(read(O/'fits.json'),ensure_ascii=False))

def evaluate():
    verify();assert read(O/'state.json')['stage']=='allmodels_thresholds_frozen_before_validation';assert not (O/'evaluation_started.json').exists()
    import joblib
    for n,h in read(O/'evaluation_freeze.json').items():assert sha(O/n)==h
    save('evaluation_started.json',dict(all_fits_and_thresholds_frozen=True))
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='validation'];assert len(cases)==590
    with np.load(B/'validation_scores.npz') as z:oldscore=z['score'];ids=z['sample_ids']
    assert all(str(sid)==c['sample_id'] for sid,c in zip(ids,cases))
    X=np.stack([c['states'][0,:37] for c in cases]).astype(float);q=np.array([int(c['queries'].sum()) for c in cases]);exp=q>64
    scores={'GLOBAL':oldscore,'GROUP':oldscore};errors={}
    for arm in ['EXP_PUBLIC37','EXP_COUNT1']:
        mm=joblib.load(O/f'{arm}.joblib');xx=X[exp][:,mm['columns']];sc=mm['scaler'];model=mm['classifier']
        ss=1-model.predict_proba(sc.transform(xx))[:,1];manual=1/(1+np.exp(((xx-sc.mean_)/sc.scale_)@model.coef_[0]+model.intercept_[0]))
        errors[arm]=float(abs(manual-ss).max());assert errors[arm]<1e-12
        s=oldscore.copy();s[exp]=ss;scores[arm]=s
    task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale'];thresholds=read(O/'fits.json')['thresholds'];rows=[]
    for i,c in enumerate(cases):
        k=len(c['gains']);G=float(c['gains'].sum());Q=int(q[i]);choices={'FULL_LOCAL':k,'BATCH1':1}
        for arm,s in scores.items():
            lam=thresholds[arm] if arm=='GLOBAL' else thresholds[arm]['expensive' if exp[i] else 'cheap']
            choices[arm]=0 if s[i]>lam else k
        for arm,p in choices.items():
            g=float(c['gains'][:p].sum());paid=int(c['queries'][:p].sum())
            rows.append(dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group'],method=arm,expensive=bool(exp[i]),batches=p,
                gain=g,gain_lost=G-g,queries=paid,full_gain=G,full_queries=Q,chosen_id=c['chosen'][p],baseline_id=c['chosen'][-1],
                action_disagreement=c['chosen'][p]!=c['chosen'][-1],net_utility=(g-fee*paid)/scale))
    summary={};checks=0
    for arm in choices:
        aa=[r for r in rows if r['method']==arm];dd={}
        for d in sorted({c['date'] for c in cases}):
            rr=[r for r in aa if r['date']==d];G=math.fsum(r['full_gain'] for r in rr);Q=sum(r['full_queries'] for r in rr);loss=math.fsum(r['gain_lost'] for r in rr)/G if G else 0.;qp=sum(r['queries'] for r in rr)/Q
            dd[d]=dict(prefixes=len(rr),gain_loss=loss,gain_capture=1-loss,query_fraction=qp,query_saving=1-qp,
                action_disagreements=sum(r['action_disagreement'] for r in rr),utility=math.fsum(r['net_utility'] for r in rr)/len(rr))
        summary[arm]=dict(per_date=dd,macro={key:float(np.mean([v[key] for v in dd.values()])) for key in ['gain_loss','gain_capture','query_fraction','query_saving','utility']},
            total_action_disagreements=sum(v['action_disagreements'] for v in dd.values()),expensive_skipped=sum(r['expensive'] and r['queries']==0 for r in aa),
            expensive_disagreements=sum(r['expensive'] and r['action_disagreement'] for r in aa))
    old=read(B/'validation_results.json')['LR_CALIBRATED'];globalrows=[r for r in rows if r['method']=='GLOBAL']
    oldrows={r['sample_id']:r for r in read(B/'validation_trajectories.json') if r['method']=='LR_CALIBRATED'}
    for r in globalrows:
        oldr=oldrows[r['sample_id']];assert r['queries']==oldr['queries'] and r['chosen_id']==oldr['chosen_id'] and r['gain_lost']==oldr['gain_lost'];checks+=1
    for key in summary['GLOBAL']['macro']:assert abs(summary['GLOBAL']['macro'][key]-old['macro'][key])<1e-12
    decision={}
    for arm in ['GROUP','EXP_PUBLIC37','EXP_COUNT1']:
        v=summary[arm];primary=v['total_action_disagreements']==0 and v['macro']['query_saving']>summary['GLOBAL']['macro']['query_saving']
        dominate=all(v['per_date'][d]['gain_loss']<=summary['GROUP']['per_date'][d]['gain_loss'] and v['per_date'][d]['query_fraction']<=summary['GROUP']['per_date'][d]['query_fraction'] for d in v['per_date'])
        strict=any(v['per_date'][d]['query_fraction']<summary['GROUP']['per_date'][d]['query_fraction'] for d in v['per_date'])
        decision[arm]=dict(primary_screen=bool(primary),incremental_pareto_vs_group=bool(dominate and strict),adopted=False)
    cm={c['sample_id']:c for c in cases}
    for r in rows:
        c=cm[r['sample_id']];p=r['batches'];assert r['chosen_id']==c['chosen'][p] and r['queries']==sum(int(x) for x in c['queries'][:p])
        assert abs(r['gain_lost']-math.fsum(float(x) for x in c['gains'][p:]))<1e-12;checks+=1
    save('validation_results.json',summary);save('validation_trajectories.json',rows);save('decision.json',dict(arms=decision,adopted=False,ready_to_scale=False))
    np.savez_compressed(O/'validation_scores.npz',**scores)
    verify();save('review.json',dict(passed=True,checks=checks,new_fits=2,identical_expensive_fit_ids_and_weights=True,all_models_thresholds_frozen_before_evaluation=True,
        original_global_trajectory_exactly_replayed=True,independent_numpy_forward_errors=errors,all_dates_arms_adverse_results_retained=True,source_unchanged=True,
        physical_or_prediction_calls=0,new_data=0,final_tests_absent=True))
    names={'FULL_LOCAL':'原FRAME完整补查','BATCH1':'固定补一批','GLOBAL':'原模型统一校准','GROUP':'原模型分组校准','EXP_PUBLIC37':'大补查专门37维LR','EXP_COUNT1':'大补查仅数量LR'}
    F=read(O/'fits.json');lines=['# GRID132：共享门槛与大补查专门学习的匹配对照','',
        '本轮仅新增两个逻辑回归，各使用相同445条大补查拟合前缀、193条校准前缀。日期仍为12拟合/4校准，小补查分支复用GRID130原模型。未增加数据、网络或物理回放。',
        '大补查固定为剩余候选数>64。两模型使用原全局拟合权重在大组上的限制并归一化，保持相同样本与权重；37维与仅log剩余数量的区别事先固定。所有新权重/门槛在验证评分前封存。','',
        '## 全部原四验证日期','', '| 方法 | 补查增量保留率 | 同站查询比例 | 动作不一致/590 | 净效用 |','|---|---:|---:|---:|---:|']
    for arm,n in names.items():
        v=summary[arm];m=v['macro'];lines.append(f"| {n} | {100*m['gain_capture']:.4f}% | {100*m['query_fraction']:.2f}% | {v['total_action_disagreements']} | {m['utility']:.8f} |")
    lines+=['','比例先按日期计算再宏平均。动作不一致相对完整同站补查，不能直接当整周失败。净效用沿用历史代理费，不是实际时间或经济成本。','','## 新模型和校准门槛','']
    for f in F['models']:lines.append(f"- {f['arm']}：{f['features']}，迭代{f['iterations']}次，训练{f['fit_seconds']:.4f}秒，收敛{f['converged']}；大组门槛{thresholds[f['arm']]['expensive']:.12f}。")
    lines+=[f"- 原模型分组门槛：小组{thresholds['GROUP']['cheap']:.12f}，大组{thresholds['GROUP']['expensive']:.12f}。",'', '## 所有开发筛查与逐日期结果','']
    for arm in ['GROUP','EXP_PUBLIC37','EXP_COUNT1']:
        lines.append(f"### {names[arm]}")
        lines.append(f"原定零动作差异且更省查询={decision[arm]['primary_screen']}；相对原模型分组校准的逐日期质量/查询共同改进={decision[arm]['incremental_pareto_vs_group']}。大组跳过{summary[arm]['expensive_skipped']}条，大组动作差异{summary[arm]['expensive_disagreements']}条。")
        for d,v in summary[arm]['per_date'].items():lines.append(f"- {d}：增量保留{100*v['gain_capture']:.4f}%，查询比例{100*v['query_fraction']:.2f}%，动作差异{v['action_disagreements']}。")
        lines.append('')
    lines+=['## 解释边界','',
        'GROUP对GLOBAL只改变预先固定分组的校准，原权重完全相同；这是门槛耦合的对照。专门模型对GROUP同时改变拟合人群和组合模型容量，不能声称只有某种架构因素起作用；两专门模型之间则固定了样本/权重/算法/校准，检验原37维在这个子问题上能否超过数量本身。',
        '全部4验证日期已多轮暴露，只是开发结果，无统计风险保证。校准组零损失是构造性质，不代表新日期零误停。不能选一个最好模型或门槛替代全部预定结果；不自动扩大、做物理闭环或读取最终8测试。']
    (O/'GRID132_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('state.json',dict(stage='complete',fits=2));save('complete.json',dict(passed=True,new_fits=2,adopted=False,ready_to_scale=False,report='GRID132_report.md'));archive('complete')
    print(json.dumps(dict(macros={a:dict(v['macro'],disagreements=v['total_action_disagreements'],expensive_skipped=v['expensive_skipped'],expensive_disagreements=v['expensive_disagreements']) for a,v in summary.items()},decision=decision),ensure_ascii=False))

if __name__=='__main__':{'prepare':prepare,'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
