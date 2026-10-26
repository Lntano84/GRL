"""Fixed 50/50 STOP and pre-STOP CONTINUE support; frozen same84parameter gate."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import collections,contextlib,hashlib,importlib.util,io,json,pickle,shutil,sys,traceback
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid192_gate_support';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID192_gate_support';OLD=R/'outputs/grid190_extension_gate'
sp=importlib.util.spec_from_file_location('grid192_original_gate_protocol',R/'work/grid190_extension_gate/run.py');B=importlib.util.module_from_spec(sp);sp.loader.exec_module(B)
read=B.read;sha=B.sha
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda v:v.item())+'\n',encoding='utf8',newline='\n')
def bind():B.O=O;B.W=W;B.A=A
def verify():
    for n in ['source_freeze.json','code_freeze.json']:
        for p,h in read(O/n).items():assert sha(R/p)==h,p
    B.H.sealed()
def archive(phase):
    bind();B.archive(phase)
def prepare():
    assert not O.exists();O.mkdir(parents=True);B.H.sealed();assert read(R/'outputs/grid191_gate_diagnosis/complete.json')['passed']
    save('design.json',dict(id='GRID192_GATE_SUPPORT',authorization='User sustained:好样的继续;autonomouslearningexploration authorized;pretrainingnotice exactly12gates delivered. No physics expansion.',
        hypothesis='In-sample frozenNETSTOPselection removes most profitable fitpatterns;test exposure augmentation without changing capacity,valuebaseline or threshold.',
        change='Only gate fitting distribution:0.5 originalSTOPdomain mass +0.5 all actualNETCONTINUEstates before firstSTOP (including both labels,not selected by gain). No postSTOPteacherstates in added domain.',
        fixed='Original83features,84parameters,original STOPfit mean/scale unchanged;originalC1/L2/lbfgs2000/.5/no classweights. Total sampleweight equals originalNSTOP,not new rowcount,to hold effective L2 scale. FrozenNET untouched.',
        weights='STOPdomain original date/call/prefix weights;CONTdomain date/call/prefix then uniform actualpreSTOPpositions. Eachdomain normalized1;sumweights=NSTOP.',
        fits=12,wall_cap_s=600,all12published_before_held_gate_scoring=True,
        evaluation='All16oldtrain datesOOF/all3seeds. BASE_NET,oldSTOPgate,newMIXgate,unconditionalADD1. Prespecified cost-only uniform thinning of each arms selected extras to minimum candidatequery total per date/seed;analytic expectation not actual hardbudget/walltime. Also fixed old3seed4dev not newlyscored;final8sealed.',
        limitations='OriginalSTOPconditioning in-sample;CONTnotdeploymentgate domain. Mixtureoutputnot calibrated STOPconditionalprobability. More training rows/no new physicaldata;not same sampleexposure count,original normalization and totalobjectiveweight held. Pure support modification,not causal proof learnability.',
        prohibited=['extra fit/seed/mixture/threshold/label/feature search','NETtraining/newfeatures/physics/old4dev/final8','automaticextension/scale','paper']))
    files=[O/'design.json',R/'outputs/grid191_gate_diagnosis/summary.json',R/'outputs/grid191_gate_diagnosis/decision.json',R/'outputs/grid191_gate_diagnosis/complete.json',OLD/'oof_trajectories.json',OLD/'gate_predictions.json',OLD/'family_summary.json',OLD/'fits.json',OLD/'held_feature_checks.json',OLD/'complete.json']+list((OLD/'fit_features').iterdir())+list((OLD/'models').glob('*.npz'))+list(OLD.glob('held_f*.npz'))
    files += [R/p for p in read(OLD/'source_freeze.json')];save('source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});code=[W/'run.py',R/'work/grid190_extension_gate/run.py']+[R/p for p in read(OLD/'code_freeze.json')];save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code});verify();archive('planned');print('GRID192_PLANNED')
def cont_mass(rows):
    dates=sorted({v['date'] for v in rows});gd={d:{v['group'] for v in rows if v['date']==d} for d in dates};sc=collections.defaultdict(set);positions=collections.Counter(v['sample_id'] for v in rows)
    for v in rows:sc[v['group']].add(v['sample_id'])
    w=np.array([1/(len(dates)*len(gd[v['date']])*len(sc[v['group']])*positions[v['sample_id']]) for v in rows]);assert abs(w.sum()-1)<1e-10;return w
def cont_features(f,seed,ids):
    import torch
    path=B.sources()[f];z=np.load(path/'fold_arrays.npz');public=B.G.inputs('train');rows=[dict(public[i],global41=z['normalized'][i]) for i in ids];net=B.S.build(seed,'FACTOR');cp=torch.load(path/'models'/f'FACTOR_seed{seed}.pt',weights_only=False,map_location='cpu');net.load_state_dict(cp['state_dict']);net.eval();before=B.digest(net);got={};out=[];qs=[]
    hs=[net.base.q_net[3].register_forward_hook(lambda m,i,o:got.__setitem__('a',o.detach().numpy().copy())),net.delta.head[1].register_forward_hook(lambda m,i,o:got.__setitem__('b',o.detach().numpy().copy()))]
    try:
        with torch.inference_mode():
            for p in range(0,len(rows),32):
                rr=rows[p:p+32];q=net(**B.G.old_api().collate(rr)).numpy();out.append(np.concatenate([got['a'],got['b'],q,np.stack([v['known_query_cost'] for v in rr])],1));qs.append(q.copy())
    finally:
        for h in hs:h.remove()
    x=np.concatenate(out).astype(float);q=np.concatenate(qs);ref=np.load(path/'state_scores.npz')[f'FACTOR_seed{seed}'][ids];assert x.shape==(len(ids),83) and np.isfinite(x).all();assert np.allclose(q,ref,atol=3e-5,rtol=3e-5);assert np.array_equal(q[:,1]>q[:,0],ref[:,1]>ref[:,0]);assert (q[:,1]>q[:,0]).all();assert B.digest(net)==before
    return x,dict(rows=len(ids),max_score_error=float(abs(q-ref).max()),NET_weights_and_CONT_decisions_unchanged=True)
def qualify():
    verify();assert not (O/'preflight.json').exists();B.G.configure();(O/'fit_features').mkdir();meta,lookup,cases=B.load();pools=[]
    for f in range(4):
        source=B.sources()[f];fi=set(np.load(source/'fold_arrays.npz')['fit_ids'].tolist());fitdates={meta[i]['date'] for i in fi};cc=[c for c in cases if c['date'] in fitdates];qfile=np.load(source/'state_scores.npz')
        for seed in range(3):
            rr=[]
            for c in cc:
                ix=[lookup[c['sample_id'],j] for j in range(len(c['states']))];q=qfile[f'FACTOR_seed{seed}'][ix];stop=next((j for j,b in enumerate(q[:,1]>q[:,0]) if not b),len(ix))
                for j in range(stop):rr.append(dict(sample_id=c['sample_id'],date=c['date'],group=c['physical_state_group'],position=j,idx=ix[j],y=c['chosen'][j]!=c['chosen'][j+1]))
            ids=[v['idx'] for v in rr];assert set(ids)<=fi;cx,check=cont_features(f,seed,ids);cm=cont_mass(rr);old=np.load(OLD/'fit_features'/f'f{f}_s{seed}.npz');assert not set(ids)&set(old['ids'].tolist());n=len(old['y']);mass=np.r_[.5*old['mass'],.5*cm];sample_weight=n*mass;assert abs(sample_weight.sum()-n)<1e-8
            x=np.vstack([old['x'],cx]);y=np.r_[old['y'],np.array([v['y'] for v in rr],int)];standard=(x-old['mean'])/old['scale'];assert np.isfinite(standard).all();np.savez_compressed(O/'fit_features'/f'f{f}_s{seed}.npz',x=x,standard=standard,y=y,mass=mass,sample_weight=sample_weight,mean=old['mean'],scale=old['scale'],ids=np.r_[old['ids'],np.array(ids)],original_STOP_count=n,CONT_count=len(rr))
            pools.append(dict(fold=f,seed=seed,STOP_rows=n,CONT_rows=len(rr),CONT_positive=int(sum(v['y'] for v in rr)),CONT_groups=len({v['group'] for v in rr}),mixture_positive_mass=float(mass@y),original_positive_mass=float(old['mass']@old['y']),normalization_unchanged=True,total_weight_unchanged=True,fit_only=True,feature_check=check));save('qualification_progress.json',pools);print('QUALIFIED_SUPPORT',f,seed,n,len(rr),flush=True)
    save('preflight.json',dict(passed=True,pools=pools,held_new_scores=0,new_fits=0,NET_training=0,mixture_fixed_half=True,normalizer_and_total_weight_fixed=True));save('qualified_freeze.json',{p.relative_to(O).as_posix():sha(p) for p in [*list((O/'fit_features').iterdir()),O/'preflight.json']});verify();archive('qualified');print('GRID192_QUALIFIED')
def delegated(phase):
    bind();s=io.StringIO()
    try:
        with contextlib.redirect_stdout(s):getattr(B,phase)()
    except Exception:
        (O/f'{phase}_console.log').write_text(s.getvalue(),encoding='utf8');(O/f'{phase}_error.log').write_text(traceback.format_exc(),encoding='utf8');raise
    (O/f'{phase}_console.log').write_text(s.getvalue(),encoding='utf8');print(s.getvalue().replace('GRID190_','GRID192_'))
def train():delegated('train')
def publication():delegated('publication')
def cached_features(f,seed,ids,tag):
    assert tag=='held';z=np.load(OLD/f'held_f{f}_s{seed}.npz');assert np.array_equal(z['ids'],ids);rec=next(v for v in read(OLD/'held_feature_checks.json') if v['fold']==f and v['seed']==seed);return z['features'].copy(),dict(rec,reused_frozen_GRID190_features=True,new_NET_forwards=0)
def evaluate():
    B.features=cached_features;delegated('evaluate')
if __name__=='__main__':globals()[sys.argv[1]]()
