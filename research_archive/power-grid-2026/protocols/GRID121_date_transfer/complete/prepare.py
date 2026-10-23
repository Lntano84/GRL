"""Four fixed grouped folds. Never split prefixes/scenario replicas of a date."""
import datetime
from common121 import *

def main():
    assert not O.exists();O.mkdir(parents=True);configure();verify120();assert read(R/'outputs/grid120_decision_objective/review.json')['passed'];sealed()
    original=read(R/'outputs/grid120_decision_objective/design.json');allcases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cases=[c for c in allcases if c['partition']=='train']
    assert len(cases)==2378;meta=[];raw=[];y=[]
    for c in cases:
        for j,x in enumerate(c['states']):
            raw.append(x);y.append(np.float32((c['gains'][j]-original['cost_per_query']*c['queries'][j])/original['gain_scale']))
            meta.append(dict(sample_id=c['sample_id'],position=j,date=c['date'],physical_state_group=c['physical_state_group'],partition='train'))
    raw=np.stack(raw);y=np.asarray(y,'f');oldmeta=read(S/'train_metadata.json');assert [(m['sample_id'],m['position']) for m in meta]==[(m['sample_id'],m['position']) for m in oldmeta]
    p=np.load(S/'train_targets.npz')['sampling_probability'];assert np.array_equal(y,np.load(S/'train_targets.npz')['target'])
    dates=sorted({c['date'] for c in cases});assert len(dates)==16
    folds=[dict(fold=k,held_dates=dates[k::4],fit_dates=[d for d in dates if d not in dates[k::4]]) for k in range(4)]
    save(O/'metadata.json',meta);np.savez_compressed(O/'train_arrays.npz',raw=raw,target=y,original_probability=p)
    with (O/'train_cases.pkl').open('xb') as f:pickle.dump(cases,f,protocol=5)
    fullmean,fullscale=normalize_stats(raw,meta,np.ones(len(y),bool));oldnorm=np.load(R/'outputs/grid102_search_rl/normalization.npz')
    assert np.allclose(fullmean,oldnorm['mean'],rtol=1e-12,atol=1e-12) and np.allclose(fullscale,oldnorm['scale'],rtol=1e-12,atol=1e-12)
    foldchecks={};streams={}
    for f in folds:
        k=f['fold'];hold=np.array([m['date'] in f['held_dates'] for m in meta]);ids=np.flatnonzero(~hold);hi=np.flatnonzero(hold)
        mean,scale=normalize_stats(raw,meta,~hold);np.savez_compressed(O/f'fold{k}_normalization.npz',mean=mean,scale=scale)
        prob=p[ids]/p[ids].sum();Z=float(prob@abs(y[ids]).astype(float));assert Z>0
        assert not set(f['fit_dates'])&set(f['held_dates'])
        assert not {meta[i]['physical_state_group'] for i in ids}&{meta[i]['physical_state_group'] for i in hi}
        foldchecks[str(k)]=dict(fit_states=len(ids),held_states=len(hi),fit_dates=f['fit_dates'],held_dates=f['held_dates'],weight_normalizer=Z,original_task_price_frozen=True)
        for seed in range(3):
            rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();count=np.zeros(len(y),np.int64)
            for _ in range(7250):
                ix=ids[rng.choice(len(ids),32,replace=True,p=prob)];h.update(ix.astype('<i8').tobytes());np.add.at(count,ix,1)
            assert count[hi].sum()==0;np.savez_compressed(O/f'fold{k}_seed{seed}_expected_exposure.npz',state_draws=count)
            streams[f'fold{k}_seed{seed}']=h.hexdigest()
    d=dict(id='GRID121_DATE_TRANSFER',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User authorizes more learning experiments; announced grouped date-held-out diagnosis after GRID120 train/validation reversal.',
        purpose='Compare Huber versus utility-weighted classification on dates excluded from each corresponding fit; diagnose whether GRID120 reversal is specific to original validation dates.',
        folds=folds,fold_rule='Sort16 original training calendar dates; held fold k uses indices k,k+4,k+8,k+12. All prefixes/replica scenarios/physical calls of a date stay together.',
        arms=['HUBER','UTILITY_BCE'],seeds=[0,1,2],fit_order=[(k,s,a) for k in range(4) for s in range(3) for a in ['HUBER','UTILITY_BCE']],
        training=dict(new_fits=24,updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=10.,wall_cap_s=1800,endpoint='All24 final-only fits before held-date predictions; no checkpoint/seed selection.'),
        architecture='Exact original6978parameter41-64-64-2 MLP, original random tensors per seed reused acrossfolds; no crossfold warmstart.',
        inputs='Same41 raw public summaries; recompute mean/scale on12 fit dates only with original equal-date/state-weight normalization rule; apply identical clipped transform to both arms and held dates.',
        sampling='Original date/prefix/position weights restricted and renormalized to12fitdates; exact same global index stream per fold/seed/arm; no held labels or optimization.',
        objectives=dict(HUBER='Original mean SmoothL1(q,[0,y]),beta1.',UTILITY_BCE='Mean abs(y)/Z * BCEWithLogits(q1-q0,1[y>0]); Z uses12fitdates only.'),
        cost_per_query=original['cost_per_query'],gain_scale=original['gain_scale'],
        historical_task_dependency='Price and gain_scale were historically defined using all16 training dates. They remain fixed task constants for this diagnostic, NOT a fully nested no-leakage estimation of the original task-design process. No fold-specific price/scale tuning.',
        evaluation='Each original training date predicted only by its held-out fold, combine OOF trajectories for16-date macro/3seeds. Also per-fold/date and physical-call-balanced,train-side paired utility.',
        criterion='Descriptive support for objective transfer only if utility family mean>Huber,all3paired OOFseed differences>0,>=12/16dates positive. Not statistical significance or qualification to scale.',
        limits='All16dates previously exposed in development; this is an OOF diagnostic,not new independent test. Each fit has12 rather than16dates; compare arms withinfold,not absolute old validation performance. No original4valdate evaluation,physical,final8tests,network/fee/threshold search.')
    save(O/'design.json',d);save(O/'preflight.json',dict(passed=True,fold_checks=foldchecks,sampling_streams=streams,full_normalization_reproduced=True,all_groups_disjoint=True,total_source_states=len(y),held_union_states=sum(x['held_states'] for x in foldchecks.values()),original_initial_hashes=read(R/'outputs/grid107_direct_value/preflight.json')['initial_q_hashes'],optimizer_instances=0,new_fits=0,new_physical_calls=0,final_test_materialized=False))
    code=list(W.glob('*.py'))+[R/f'work/{n}' for n in ['grid120_decision_objective/common120.py','grid115_matched_module_plan/matched_model.py','grid113_relational_learning/model.py','grid113_relational_learning/common.py','grid107_direct_value/common107.py','grid112_relational_plan/schema.py']]
    save(O/'code_freeze.json',{x.relative_to(R).as_posix():sha(x) for x in code})
    files=[x for x in O.iterdir() if x.is_file() and x.name!='code_freeze.json']+[R/f'outputs/{n}' for n in ['grid120_decision_objective/decision.json','grid120_decision_objective/review.json','grid120_decision_objective/code_freeze.json','grid120_decision_objective/data_freeze.json','grid100_data_foundation/design.json']]
    save(O/'data_freeze.json',{x.relative_to(R).as_posix():sha(x) for x in files});save(O/'state.json',dict(stage='qualified_before_push',new_fits=0));print('GRID121_PREFLIGHT_PASSED',json.dumps(foldchecks),flush=True)
if __name__=='__main__':main()
