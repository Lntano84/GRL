"""One frozen LR fit,one empirical calibration,one known-development evaluation."""
from pathlib import Path
import collections
import datetime
import gzip
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import time

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import numpy as np

R=Path('D:/Codex/GRL研究');O=R/'outputs/grid130_calibrated_gate';P=R/'outputs/grid129_baseline_risk'
REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID130_calibrated_gate'
sys.path.insert(0,str(R/'work/grid107_direct_value'))
from common107 import sealed

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):
    (O/name).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def archive(phase):
    dest=A/phase;dest.mkdir(parents=True,exist_ok=False)
    for p in [*O.iterdir(),Path(__file__)]:
        if p.is_file():shutil.copy2(p,dest/p.name)
    (dest/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in dest.iterdir() if p.is_file()},indent=2)+'\n',encoding='utf8')
def verify():
    for name in ['source_hashes.json','code_freeze.json']:
        for n,h in read(O/name).items():assert sha(R/n)==h,n
    sealed()

def prepare():
    assert not O.exists();O.mkdir(parents=True);sealed();assert read(P/'review.json')['passed'] and read(P/'cross_version_review.json')['passed']
    d=read(P/'proposed_training_plan.json')
    assert d['new_fits']==1 and d['status']=='proposed_not_started'
    d.update(id='GRID130_CALIBRATED_GATE',status='frozen_before_one_fit',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User authorizes baseline-risk research and previously permits autonomous ideas with immediate updates. One-fit plan qualified in GRID129 and announced before training.',
        reads='Train/calibration from GRID129 only;known4validation cases evaluated after frozen fitted model and calibrated lambda receipt. No final test.',
        objective='Check whether independent empirical-zero-loss calibration protects full-local baseline gains while saving queries,relative to fixed0.5 same-model readout.',
        input_precision='GRID102 saved float32 public summaries first37,qualified against GRID101 within2.31e-7;new gates use exactly identical inputs.',
        budget='Exactly one LR fit,max_iter2000 inherited,no refit or alternate split/threshold;all adverse results retained.',
        caveat='Empirical zero calibration loss not a statistical CRC certificate;4calibration dates,seasonal exchangeability unverified,all validation dates exposed. No model-selected price or alpha.')
    save('design.json',d)
    paths=[P/n for n in ['plan.json','review.json','cross_version_review.json','proposed_training_plan.json','train_public_inputs.npz','train_label_ledger.jsonl.gz']]
    paths += [R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',O/'design.json']
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths});save('code_freeze.json',{Path(__file__).relative_to(R).as_posix():sha(Path(__file__))})
    save('state.json',dict(stage='frozen_before_fit',fits=0));archive('frozen');print('GRID130_FROZEN')

def inputs():
    with np.load(P/'train_public_inputs.npz') as z:X=z['X'][:,:37].astype(float);ids=z['sample_ids']
    with gzip.open(P/'train_label_ledger.jsonl.gz','rt',encoding='utf8') as f:rows=[json.loads(line) for line in f]
    assert all(str(i)==r['sample_id'] for i,r in zip(ids,rows))
    return X,rows

def fit():
    verify();assert not (O/'training_started.json').exists(),'Do not repeat a started or completed fit'
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    import joblib
    D=read(O/'design.json');X,rows=inputs();fitdates=set(D['fit_dates']);caldates=set(D['calibration_dates'])
    fitmask=np.array([r['date'] in fitdates for r in rows]);calmask=np.array([r['date'] in caldates for r in rows])
    assert not (fitmask&calmask).any() and (fitmask|calmask).all()
    counts=collections.Counter(r['date'] for r,yes in zip(rows,fitmask) if yes)
    w=np.array([1/counts[r['date']] for r,yes in zip(rows,fitmask) if yes]);w/=w.mean()
    y=np.array([r['full_baseline_gain']>0 for r in rows]);bad=np.array([r['full_baseline_gain']>0 or r['action_changed_if_skip'] for r in rows])
    save('training_started.json',dict(fit_ids=[r['sample_id'] for r,yes in zip(rows,fitmask) if yes],calibration_ids=[r['sample_id'] for r,yes in zip(rows,calmask) if yes],fits_started=1))
    started=time.perf_counter();scaler=StandardScaler().fit(X[fitmask],sample_weight=w)
    model=LogisticRegression(**D['model_parameters']).fit(scaler.transform(X[fitmask]),y[fitmask],sample_weight=w)
    duration=time.perf_counter()-started
    joblib.dump(dict(scaler=scaler,classifier=model),O/'model.joblib')
    score=1-model.predict_proba(scaler.transform(X))[:,1]
    assert np.isfinite(score).all() and (score>=0).all() and (score<=1).all()
    lam=float(score[calmask&bad].max()) if (calmask&bad).any() else 1.
    skip=score>lam
    assert not (skip&calmask&bad).any()
    # Independent logistic equation and calibration maximum;no model selection.
    logit=((X-scaler.mean_)/scaler.scale_)@model.coef_[0]+model.intercept_[0]
    manual=1/(1+np.exp(logit));error=float(abs(manual-score).max());assert error<1e-12
    np.savez_compressed(O/'train_calibration_scores.npz',score=score,fitmask=fitmask,calmask=calmask)
    fitreceipt=dict(fits=1,fit_seconds=duration,fit_prefixes=int(fitmask.sum()),calibration_prefixes=int(calmask.sum()),
        fit_dates=sorted(fitdates),calibration_dates=sorted(caldates),calibrated_lambda=lam,calibration_gain_losses=0,calibration_action_mismatches=0,
        calibration_skipped=int((skip&calmask).sum()),fixed_half_calibration_skipped=int(((score>.5)&calmask).sum()),
        fixed_half_calibration_missed_positive=int(((score>.5)&calmask&bad).sum()),
        calibration_positive_witnesses=int((bad&calmask).sum()),iterations=int(model.n_iter_[0]),
        converged=bool(model.n_iter_[0]<D['model_parameters']['max_iter']),model_sha256=sha(O/'model.joblib'),
        independent_numpy_max_error=error,fit_date_weights={d:float(w[[r['date']==d for r,yes in zip(rows,fitmask) if yes]].sum()) for d in sorted(fitdates)},
        validation_used_for_fit_or_calibration=False,empirical_only_no_statistical_guarantee=True)
    save('fits.json',fitreceipt);save('state.json',dict(stage='model_and_lambda_frozen_before_validation',fits=1))
    save('evaluation_freeze.json',{n:sha(O/n) for n in ['model.joblib','fits.json','train_calibration_scores.npz','design.json']})
    print(json.dumps(fitreceipt,ensure_ascii=False))

def evaluate():
    verify();assert read(O/'state.json')['stage']=='model_and_lambda_frozen_before_validation'
    assert not (O/'evaluation_started.json').exists(),'Do not repeat evaluation'
    import joblib
    for n,h in read(O/'evaluation_freeze.json').items():assert sha(O/n)==h
    save('evaluation_started.json',dict(fits=1,lambda_fixed=True))
    model=joblib.load(O/'model.joblib');lam=read(O/'fits.json')['calibrated_lambda']
    cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='validation']
    assert len(cases)==590 and len({c['date'] for c in cases})==4
    X=np.stack([c['states'][0,:37] for c in cases]).astype(float);s=1-model['classifier'].predict_proba(model['scaler'].transform(X))[:,1]
    logit=((X-model['scaler'].mean_)/model['scaler'].scale_)@model['classifier'].coef_[0]+model['classifier'].intercept_[0]
    manual=1/(1+np.exp(logit));assert abs(manual-s).max()<1e-12
    task=read(R/'outputs/grid102_search_rl/design.json');fee=task['cost_per_query'];scale=task['gain_scale']
    rows=[]
    for c,score in zip(cases,s):
        k=len(c['gains']);G=float(c['gains'].sum());Q=int(c['queries'].sum())
        choices={'FULL_LOCAL':k,'BATCH1':1,'LR_FIXED':0 if score>.5 else k,'LR_CALIBRATED':0 if score>lam else k}
        for arm,p in choices.items():
            g=float(c['gains'][:p].sum());q=int(c['queries'][:p].sum())
            rows.append(dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group'],method=arm,
                batches=p,gain=g,queries=q,full_gain=G,full_queries=Q,chosen_id=c['chosen'][p],baseline_id=c['chosen'][-1],
                action_disagreement=c['chosen'][p]!=c['chosen'][-1],gain_lost=G-g,score=float(score),net_utility=(g-fee*q)/scale))
    summary={};checks=0
    for arm in choices:
        aa=[r for r in rows if r['method']==arm];dd={}
        for d in sorted({r['date'] for r in aa}):
            rr=[r for r in aa if r['date']==d];G=math.fsum(r['full_gain'] for r in rr);Q=sum(r['full_queries'] for r in rr)
            loss=math.fsum(r['gain_lost'] for r in rr)/G if G else 0.;query=sum(r['queries'] for r in rr)/Q
            dd[d]=dict(prefixes=len(rr),gain_loss=loss,gain_capture=1-loss,query_fraction=query,query_saving=1-query,
                action_disagreements=sum(r['action_disagreement'] for r in rr),action_disagreement_fraction=sum(r['action_disagreement'] for r in rr)/len(rr),
                utility=math.fsum(r['net_utility'] for r in rr)/len(rr))
            assert abs(dd[d]['gain_capture']-math.fsum(r['gain'] for r in rr)/G)<1e-12 if G else True
            checks+=1
        summary[arm]=dict(per_date=dd,macro={key:math.fsum(v[key] for v in dd.values())/len(dd) for key in ['gain_loss','gain_capture','query_fraction','query_saving','action_disagreement_fraction','utility']},
            total_action_disagreements=sum(v['action_disagreements'] for v in dd.values()))
    cal=summary['LR_CALIBRATED'];fixed=summary['LR_FIXED']
    screen=cal['total_action_disagreements']==0 and cal['macro']['query_saving']>0
    np.savez_compressed(O/'validation_scores.npz',score=s,sample_ids=np.array([c['sample_id'] for c in cases]))
    save('validation_trajectories.json',rows);save('validation_results.json',summary)
    save('decision.json',dict(local_development_screen=bool(screen),adopted=False,ready_to_scale=False,
        gain_loss_reduction_vs_fixed=fixed['macro']['gain_loss']-cal['macro']['gain_loss'],
        extra_query_fraction_vs_fixed=cal['macro']['query_fraction']-fixed['macro']['query_fraction'],
        quality_vs_full_baseline_zero_mismatch=cal['total_action_disagreements']==0,
        interpretation='Known4date development only. No choice of fit/split/lambda after validation;no physical/time/independent safety claim.'))
    # Independent row-by-row action,loss and query reconstruction from source cases.
    cm={c['sample_id']:c for c in cases}
    for rr in rows:
        c=cm[rr['sample_id']];p=rr['batches']
        assert rr['chosen_id']==c['chosen'][p]
        assert abs(rr['gain_lost']-math.fsum(float(x) for x in c['gains'][p:]))<1e-12
        assert rr['queries']==sum(int(x) for x in c['queries'][:p]);checks+=1
    assert sha(O/'model.joblib')==read(O/'fits.json')['model_sha256'];verify()
    save('review.json',dict(passed=True,checks=checks,fit_count=1,fit_and_calibration_date_groups_disjoint=True,
        empirical_calibration_zero_loss=True,independent_numpy_model_scores=True,maximum_validation_forward_error=float(abs(manual-s).max()),
        model_and_lambda_frozen_before_validation=True,all_4_dates_retained=True,final_tests_absent=True,source_unchanged=True,
        physical_or_prediction_calls=0,new_scenarios=0,statistics='No statistical risk guarantee or independent confirmation.'))
    names={'FULL_LOCAL':'原FRAME完整同站补查','BATCH1':'固定补一批','LR_FIXED':'同模型固定门槛','LR_CALIBRATED':'同模型校准门槛'}
    F=read(O/'fits.json');lines=['# GRID130：基础规则回退下的一次门控开发试验','',
        f"仅训练一个逻辑回归，12日期拟合{F['fit_prefixes']}条前缀、另4日期校准{F['calibration_prefixes']}条。训练{F['fit_seconds']:.3f}秒，迭代{F['iterations']}次。没有训练神经网络/强化学习、扩大数据或物理回放。",'',
        '使用原37维公开输入、同站补查前一次决策；选择跳过或执行原完整补查。模型预测完整补查有无改进，沿用GRID101配置。',
        f"固定门槛：s>0.5跳过；校准门槛：s>{lam:.12f}跳过，其中s=1−P(完整补查增量>0)。门槛由校准有利案例的最大s确定，并列保留。校准门槛使用额外4日期标签，不能说与固定门槛信息预算相同。",'',
        f"校准侧固定门槛跳过{F['fixed_half_calibration_skipped']}条，漏掉{F['fixed_half_calibration_missed_positive']}条有利补查；校准门槛跳过{F['calibration_skipped']}条，局部增量与动作损失均为0（构造性质，不是验证结论）。",'',
        '## 已暴露四验证日期结果','',
        '| 方法 | 补查增量保留率 | 同站查询比例 | 相对完整补查动作不一致 | 扣费净效用 |',
        '|---|---:|---:|---:|---:|']
    for arm,n in names.items():
        v=summary[arm];m=v['macro'];lines.append(f"| {n} | {100*m['gain_capture']:.2f}% | {100*m['query_fraction']:.2f}% | {v['total_action_disagreements']}/590 | {m['utility']:.8f} |")
    lines+=['','增量与查询先按日期计算再宏平均；局部增量不是整周经济成本，查询不是实际控制时间。动作不一致只是相对完整补查的差异，也不能直接解释为整周损害。','','## 校准门槛逐日期','']
    for d,v in cal['per_date'].items():lines.append(f"- {d}：增量保留{100*v['gain_capture']:.2f}%，查询比例{100*v['query_fraction']:.2f}%，动作不一致{v['action_disagreements']}/{v['prefixes']}，净效用{v['utility']:.8f}。")
    lines+=['','## 判读','',
        f"预定开发筛查：四日期全部动作一致且查询宏平均有节省={screen}。不采用、不自动扩大规模；本轮只用于评估该设计。", 
        f"相对同权重固定门槛，增量损失宏平均减少{100*(fixed['macro']['gain_loss']-cal['macro']['gain_loss']):.2f}个百分点，查询比例增加{100*(cal['macro']['query_fraction']-fixed['macro']['query_fraction']):.2f}个百分点。保护收益的代价与剩余误停均如实保留。",'',
        '这是经验校准门槛，不是CRC统计保证。4校准日期与季节分布不足以支持现成套用的严格保证，不能把相关前缀充数。校准组零损失也不保证留出日期零损失。',
        '第一版决策是整段委托而非原逐批学习，模型也改回简单LR；不能将它与旧KL之间的差别单独归因于校准。真正匹配的机制对照是本轮同一模型的固定与校准读出。',
        '原验证日期已多次用于开发，未扩最终测试或新场景。所有不利日期/动作保留，不选新的lambda、风险alpha、分组或模型救结果。']
    (O/'GRID130_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    save('state.json',dict(stage='completed_development_evaluation',fits=1));save('complete.json',dict(passed=True,fit_count=1,local_development_screen=bool(screen),ready_to_scale=False,report='GRID130_report.md'))
    archive('complete');print(json.dumps(dict(summary=summary,decision=read(O/'decision.json')),ensure_ascii=False))

if __name__=='__main__':
    {'prepare':prepare,'fit':fit,'evaluate':evaluate}[sys.argv[1]]()
