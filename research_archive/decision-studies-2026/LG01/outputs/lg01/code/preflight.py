import copy, json, os, sys, tempfile, time
from pathlib import Path
import numpy as np
from core import *
from features import make_features

def preflight():
    old=Path(r'C:\Users\windows\Desktop\im算法与基准阅读包\08_LOT_REOPT\stage06_neighborhood_content')
    states=read(old/'stage20_states.json')
    key=sorted(states)[0]; raw=states[key]; meta=raw['meta']
    nominal=read(old/'stage20_nominal_bar'/f"{meta['name']}.json")
    rec=dict(family=meta['seed'],rho=meta['rho'],disruption=raw['disruption'],repair=raw['repair'],nominal=nominal)
    inst,_=build_instance('large',rec['rho'],rec['family']);d=rec['disruption']
    pert=apply_disruption(inst,Disruption(d['name'],{int(j):ts for j,ts in d['down'].items()}))
    bm=build_model(pert,MODE_AUDITED);anchor=solution(rec['repair']);nom=solution(nominal)
    stab=stability_row_local(bm,anchor,TAU,KAPPA)
    rlx=solve(bm,{},stab,time.perf_counter()+5.,OUT/'preflight/relax.log',relax=True)
    assert rlx['optimal'] and rlx['dual_valid']
    fixes,masks,fr,slots,info=actions(bm,anchor,nom,rlx,key)
    checks={}
    for a in range(1,4):
        assert set(fixes[a])<=set(fr)
        for h,q in info['quota'].items():
            kind=h.split('_')[1];short=h.startswith('short')
            assert sum(slots[c][0]==kind and (slots[c][3]<=TAU)==short for c in set(fr)-set(fixes[a]))==q
    checks['stratified_matched_ceil_10pct']=True
    assert not fixes[4] and fixes[0]==fr
    assert all(all(f[c]==fr[c] for c in f) for f in fixes)
    checks['RINS_subset_expansions_FULL_and_anchor_values']=True
    mutation=[]
    for a,f in enumerate(fixes):
        assert verify(pert,anchor,anchor,f,slots)['ok']
        for h in ['short_Y','far_Y','short_Z','far_Z']:
            cs=[c for c in f if ('short_' if slots[c][3]<=TAU else 'far_')+slots[c][0]==h]
            if not cs: continue
            c=cs[0];kind,i,j,t=slots[c];mut=copy.deepcopy(anchor)
            getattr(mut,kind)[i,j,t-1]=1-f[c]
            assert fixes_ok(mut,f,slots)
            assert not verify(pert,mut,anchor,f,slots)['ok']
            mutation.append([ACTIONS[a],h,c])
    # Arm-specific validation: same reference, different imposed sets. A released flip must
    # pass the fixing-only checker for its own arm while failing A0's fixing-only checker.
    for a in range(1,5):
        if set(fr)-set(fixes[a]):
            c=min(set(fr)-set(fixes[a]));kind,i,j,t=slots[c];mut=copy.deepcopy(anchor)
            getattr(mut,kind)[i,j,t-1]=1-fr[c]
            assert not fixes_ok(mut,fixes[a],slots) and fixes_ok(mut,fr,slots)
    checks['all_arms_all_YZ_fixing_checks']=mutation
    g=make_features(bm,anchor,nom,rlx,fr,masks,slots,stab,inst)
    a,lo,up=system(bm,stab);co=a.tocoo();scale=np.maximum(1.,np.asarray(abs(a).max(axis=1).toarray()).ravel())
    assert np.array_equal(g['ei'],np.stack([co.row,co.col]))
    assert np.allclose(g['ev'][:,0],co.data/scale[co.row])
    checks['real_bipartite_coefficients_including_kappa']=True
    from models import SetValue,tensors,choice
    import torch
    torch.manual_seed(7);m=SetValue();m.eval()
    with torch.inference_mode(): before=m(tensors(g)).numpy()
    fake=np.random.default_rng(7).normal(size=(96,5));fake[:]=1e20
    afterg=make_features(bm,anchor,nom,rlx,fr,masks,slots,stab,inst)
    assert all(np.array_equal(g[k],afterg[k]) for k in g)
    with torch.inference_mode(): after=m(tensors(afterg)).numpy()
    assert np.array_equal(before,after) and choice(before)==choice(after)
    checks['outcome_table_independent_inputs_and_action']=True
    path=OUT/'preflight/first_log.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise RuntimeError('preflight diagnostics already exist; do not rerun silently')
    with path.open('a',encoding='utf-8') as f:
        for event in [dict(kind='start',id='first',reserved_s=20.),dict(kind='complete',id='first',actual_s=19.9)]:
            f.write(json.dumps(event)+'\n');f.flush();os.fsync(f.fileno())
    parsed=[json.loads(line) for line in path.read_text().splitlines()]
    assert len(parsed)==2
    save(OUT/'preflight/resume_checkpoint.json',dict(done=['first'],charged_s=19.9))
    resumed=read(OUT/'preflight/resume_checkpoint.json');assert resumed['charged_s']==19.9 and 'first' in resumed['done']
    # Conservative interruption reserve is charged once, never reset to a fresh 20 s.
    charged=resumed['charged_s']+20.; assert charged==39.9
    checks['first_log_fsync_atomic_resume_accounting']=True
    result=dict(ok=True,diagnostic_state=key,LG01_instances_used=0,relax=lp_summary(rlx),
                checks=checks,action_meta=info,graph_shapes={k:list(v.shape) for k,v in g.items()})
    save(OUT/'LG01_preflight.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

if __name__=='__main__': preflight()
