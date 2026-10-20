"""Freeze label-blind expansion, primary labels and two fixed simple model families."""
from pathlib import Path
from collections import defaultdict,Counter
import ast,datetime,hashlib,importlib.util,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
digest=lambda v:hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()


def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def main():
    assert not (OUT/'design.json').exists()
    refs={r['week']:r for r in read(ROOT/'outputs/grid91_mechanism/design.json')['references']}
    rows=read(ROOT/'outputs/grid94_predictability/dataset.json');by=defaultdict(list)
    for r in rows:by[r['week']].append(r)
    prior=read(ROOT/'outputs/grid95_labels/design.json')['cases']
    assert read(ROOT/'outputs/grid95_labels/review.json')['passed_saved_data']
    selected=[]
    for week,rr in by.items():
        rr.sort(key=lambda r:r['step']);indices=sorted(set([0,(len(rr)-1)//2,len(rr)-1]))
        take=[rr[i] for i in indices]
        take += [r for r in rr if r['incumbent_rho']>=1 and r not in take]
        selected+=sorted(take,key=lambda r:r['step'])
    assert len(selected)==56 and len(by)==21
    cases=[]
    for index,r in enumerate(selected):
        src=ROOT/r['source'];c=read(src)
        with np.load(src.with_name(src.stem+'_state.npz')) as z:
            assert digest(z['observation'])==c['state_hash'];basehash=digest(z['base_action'])
        pilot=next((p for p in prior if p['week']==r['week'] and p['step']==r['step']),None)
        cases.append(dict(id=f'S{index:03d}',week=r['week'],cohort=r['cohort'],calendar_date=r['calendar_date'],step=r['step'],
            environment=refs[r['week']]['environment'],reference=refs[r['week']]['frame'],source=r['source'],
            state_hash=c['state_hash'],base_action_hash=basehash,completion_budget=r['completion_queries'],
            before_choice=r['before_choice'],after_choice=r['after_choice'],features={k:r[k] for k in read(ROOT/'work/grid94_predictability/design.json')['features']},
            pilot_known=pilot is not None,reuse_case_id=pilot['id'] if pilot else None,
            reused_branch=f"outputs/grid95_labels/runs/r0__{r['week']}__SKIP_{pilot['id']}" if pilot else None))
    shared=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,policy='FRAME_GUARD',order_mode='STATIC_ID',repeat=0,stage_phase='all')
    items=[]
    for week in by:
        cc=[c for c in cases if c['week']==week]
        items.append(dict(shared,week=week,environment=cc[0]['environment'],method='FRAME_VALUE_QUAL',watch_cases=[c['id'] for c in cc],omit_case=None,purpose='factual reconstruction'))
    for c in cases:
        if not c['pilot_known']:items.append(dict(shared,week=c['week'],environment=c['environment'],method='SKIP_'+c['id'],watch_cases=[c['id']],omit_case=c['id'],purpose='single omission'))
    assert len(items)==74 and sum(c['pilot_known'] for c in cases)==3
    features=read(ROOT/'work/grid94_predictability/design.json')['features']
    d=dict(stage='GRID96_COUNTERFACTUAL_VALUE_LEARNING',frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='2026-10-20 用户：再用这个标签试学习，仍用简单模型，规模稍大一些',
        sample=dict(eligible_population=109,selected=56,scenario_entries=21,calendar_groups=17,
            rule='Per original scenario take chronological first, middle floor((n-1)/2), last eligible completion, plus all pre-completion incumbent_rho>=1 states. Uses no new value labels.',
            pilot_known=3,pilot_rule='Known selected GRID95 cases may enter training when their calendar date is in training; exclude from headline evaluation and list separately. No synthetic duplication.'),
        cases=cases,execution_order=items,new_replays=74,factual_replays=21,single_omission_replays=53,reused_qualified_branches=3,
        intervention='Original controller/model/order/128/64/2/0.9 unchanged; omit completion exactly once, then original FRAME continuation. Fresh full prefixes reconstruct memory. No observation-only reset.',
        nominal_episode_steps=2017,horizon=dict(primary='scheduled original episode end, not factual early-failure time',diagnostic_steps=[1,12,288]),
        label=dict(classes=['KEEP','SKIP','MIXED'],unresolved='UNRESOLVED',survival_priority=True,
            rule='External attack mismatch => unresolved. Identical entire physical path with no more total forecast queries => SKIP equivalent-path (even if both inherit the same factual early failure). Otherwise complete-vs-fail favors the completing branch. Both fail with different paths => unresolved. Both complete => Pareto comparison of operational cost, all forecast queries, overload steps, max rho and overload-excess sum. One branch no worse in every component and strictly better in at least one => its label; otherwise MIXED.',
            numeric_tolerances=dict(cost='max(0.01,32*float64_epsilon*(abs(C_complete)+abs(C_skip)))',max_rho=1e-6,overload_excess_sum=1e-6,queries=0,overload_steps=0),
            limitations='Numeric tolerances are inherited diagnostic reporting tolerances, not safety guarantees or fitted practical-equivalence margins. Label scope original continuation and fixed sampled future, not a learned gated closed loop.'),
        learning=dict(features=features,models=dict(LOGISTIC=dict(C=1.,solver='lbfgs',max_iter=2000),FOREST64=dict(n_estimators=64,max_depth=4,min_samples_leaf=3,max_features='sqrt',random_state=20261020,n_jobs=1)),
            weighting='Equal total weight per training calendar date, normalized to mean1; no class oversampling.',
            preprocessing='Logistic training-fold weighted mean/std only, constant columns scale1; forest uses raw inputs.',
            validation=['leave calendar date out; G1/G2 same dates withheld together','old13 train, G1+G2 evaluate'],
            unknown_labels='Retain and report but exclude from fitting/scoring; missing training classes use explicitly padded probabilities; constant classifier if one class only.',
            decision='Predict multiclass preferences; proposed omission only when P(SKIP)>=0.95 and SKIP is argmax. Also report fixed0.5, no threshold selection.',
            metrics=['confusion matrix','balanced accuracy','macro F1','SKIP ROC AUC/AP/Brier','skip precision/coverage','missed KEEP separately from disputed MIXED','per-state raw value for proposed skips','calendar/cohort counts; pilots separate'],
            no_deployment=True,no_hyperparameter_search=True),
        timing='No efficiency benchmarking. Label collection first; no training until every physical child exits and saved-data audit passes.',
        caps=dict(episode_s=1200,wall_s=10800,physical_steps=74*2017,public_forecasts=1500000),
        forbidden=['RL','model/controller/threshold changes','extra cases after outcomes','changing label rules after results','new sampled scenario/second grid','sum overlapping per-state gains into whole-policy savings','automatic learned-gate deployment'])
    save('design.json',d);save('manifest.json',dict(planned=len(items),summaries=[]))
    # Freeze the previously omitted optimizer dependency now, before these new replays.
    codes=read(ROOT/'outputs/grid95_labels/code_freeze.json');data=read(ROOT/'outputs/grid95_labels/data_freeze.json')
    for rel,h in codes.items():assert sha(ROOT/rel)==h,rel
    for rel,h in data.items():assert sha(ROOT/rel)==h,rel
    for directory in [W,ROOT/'work/grid95_labels']:
        for p in directory.glob('*.py'):codes[p.relative_to(ROOT).as_posix()]=sha(p);ast.parse(p.read_text(encoding='utf-8'))
    for rel in ['work/grid02/ljn_nn/modules/convex_optim.py','work/grid94_predictability/data.py','work/grid94_predictability/design.json']:
        codes[rel]=sha(ROOT/rel)
    for rel in ['outputs/grid94_predictability/dataset.json','outputs/grid95_labels/design.json','outputs/grid95_labels/review.json','outputs/grid95_labels/audit_memory_v2_receipt.json']:
        data[rel]=sha(ROOT/rel)
    for c in cases:
        for rel in [c['reference']]+([c['reused_branch']] if c['reused_branch'] else []):
            for p in (ROOT/rel).iterdir():
                if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
    save('preflight.json',dict(passed=True,selected_states=56,scenarios=21,calendar_dates=17,
        fresh_physical_replays=len(items),pilot_reuses=3,features=len(features),pre_replay_optimizer_source_sha256=sha(ROOT/'work/grid02/ljn_nn/modules/convex_optim.py'),
        model_fits_before_collection=0,no_new_labels_read_for_sampling=True))
    for n in ['design.json','preflight.json']:codes[(OUT/n).relative_to(ROOT).as_posix()]=sha(OUT/n)
    save('data_freeze.json',data);codes[(OUT/'data_freeze.json').relative_to(ROOT).as_posix()]=sha(OUT/'data_freeze.json');save('code_freeze.json',codes)
    dst=REPO/'research_archive/power-grid-2026/protocols/GRID96_value_learning/frozen';dst.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[OUT/n for n in ['design.json','preflight.json','code_freeze.json','data_freeze.json']]:shutil.copy2(p,dst/p.name)
    note='用户重新授权以可靠单次标签做稍大规模简单学习。固定56状态/21场景记录/17日期，按时间位置和补查前过载取样；21事实资格+53单次取消，另3已知试点复用并不计主要成绩。存活优先、成本/查询/实际过载Pareto三类标签，未定保留；原策略和参数不改。标签核验后逻辑回归与64棵深度4浅树森林固定比较，按日期隔离，不调参、不部署。'
    for n in ['docs/DECISIONS.md','docs/EXPERIMENT_LOG.md']:
        p=REPO/n;s=p.read_text(encoding='utf-8');p.write_text(s.rstrip()+'\n\n## 2026-10-20 — GRID96_VALUE_LEARNING_frozen\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps(dict(states=len(cases),new_replays=len(items),pilot_reused=3,code_items=len(codes),data_items=len(data)),ensure_ascii=False))


if __name__=='__main__':main()
