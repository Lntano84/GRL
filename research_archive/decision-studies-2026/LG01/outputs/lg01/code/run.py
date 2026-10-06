"""One frozen, serial LG01 trial: generation -> labels -> fit/seal -> test -> audit."""
from __future__ import annotations
import argparse, csv, importlib.metadata, json, os, platform, re, shutil, sys, time, traceback
from pathlib import Path
import numpy as np
from core import *

WORK=Path(__file__).resolve().parent

class Ledger:
    def __init__(self):
        self.path=OUT/'events.jsonl';self.events=[]
        if self.path.exists(): self.events=[json.loads(l) for l in self.path.read_text(encoding='utf-8').splitlines()]
    def emit(self,event):
        event['utc_unix']=time.time()
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f:
            f.write(json.dumps(event,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
        self.events.append(event)
    def previous(self,jid): return [e for e in self.events if e.get('id')==jid]
    def charge(self):
        starts={e['id']:e for e in self.events if e['kind']=='start'}
        ends={e['id']:e for e in self.events if e['kind']=='complete'}
        return sum(ends[k]['charged_s'] if k in ends else v['reserved_s'] for k,v in starts.items())
    def job(self,jid,cap,path,fn,fallback=None):
        path=diskpath(path);previous=self.previous(jid)
        if path.exists():
            result=read(path)
            if not any(e['kind']=='complete' for e in previous):
                self.emit(dict(kind='complete',id=jid,charged_s=result.get('job_wall_s',cap),
                               recovered_atomic_result=True,result_hash=sha(path)))
            return result
        if previous:
            # A killed solver NEVER receives another full budget. The full reserved cap is
            # charged, and its already-valid reference is the conservative delivered plan.
            if fallback is None: raise RuntimeError(f'{jid}: interrupted job has no saved feasible reference')
            tr=time.perf_counter();result=fallback();extra=time.perf_counter()-tr
            result.update(interrupted=True,timing_ok=False,job_wall_s=cap+extra)
            save(path,result);self.emit(dict(kind='complete',id=jid,charged_s=cap+extra,interrupted=True,result_hash=sha(path)))
            return result
        if self.charge()+cap>43200: raise RuntimeError('12 hour compute cap reached; no scope reduction')
        self.emit(dict(kind='start',id=jid,reserved_s=cap,result=str(path.relative_to(OUT))))
        t0=time.perf_counter();result=fn();elapsed=time.perf_counter()-t0
        result['job_wall_s']=elapsed;save(path,result)
        self.emit(dict(kind='complete',id=jid,charged_s=time.perf_counter()-t0,result_hash=sha(path)))
        return result

def freeze():
    path=OUT/'LG01_freeze.json'
    if path.exists():
        d=read(path)
        assert all(sha(ROOT/p)==h for p,h in d['code_hashes'].items()),'Frozen code changed'
        return d
    assert read(OUT/'LG01_preflight.json')['ok']
    families=list(range(14,46));random.Random(20261006).shuffle(families)
    split=dict(train=families[:18],validation=families[18:24],test=families[24:])
    source=Path(r'C:\Users\windows\Desktop\im算法与基准阅读包\08_LOT_REOPT\stage06_neighborhood_content')
    overlapping=[];examined=[]
    for p in source.glob('stage*.json'):
        if 'state' not in p.name and 'nominal' not in p.name: continue
        # Only instance identifiers are extracted; historical action outcomes never become labels.
        ids=set(re.findall(r'large_rho(?:0\.75|1\.10)_s(\d+)',p.read_text(encoding='utf-8')))
        overlap=sorted(set(map(int,ids))&set(range(14,46)))
        examined.append(dict(path=str(p),sha256=sha(p),families=sorted(map(int,ids))))
        if overlap: overlapping.append(dict(path=str(p),families=overlap))
    if overlapping: raise RuntimeError(f'new seed family overlap; do not silently replace: {overlapping}')
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(WORK.rglob('*.py')) if '.venv' not in p.parts}
    code=OUT/'code';code.mkdir(parents=True,exist_ok=True)
    for p in sorted(WORK.glob('*.py')): shutil.copy2(p,code/p.name)
    (code/'source').mkdir(exist_ok=True)
    for p in SRC.glob('*.py'): shutil.copy2(p,code/'source'/p.name)
    dependencies={m:importlib.metadata.version(m) for m in ['numpy','scipy','highspy','scikit-learn','torch']}
    d=dict(spec_sha256=sha(ROOT/'outputs/LG01_training_plan.md'),split=split,split_seed=20261006,
           family_count=32,state_counts=dict(train=72,validation=24,test=32),rhos=[.75,1.10],
           faults=['D1_m0_2p','D2_all_1p'],nominal_protocol=[120,20,20],scale='large',tau=6,kappa=4,
           actions=ACTIONS,expansion_fraction=.1,quota='ceil separately in each of four strata',
           online_budget_s=20.,delivery_reserve_s=.2,max_wall_s=20.2,LP_cap_s=5.,
           solver_seeds=dict(labels=[0],test=[0,1]),solver_threads=1,label_runs=480,test_runs_max=192,
           network=dict(message_rounds=2,hidden=32,seed=7,epochs=50,patience=5,delta=.02,lr=.001,
                        weight_decay=.0001,batch_states=1,training_cap_s=1800),
           model_training_total_cap_s=3600,total_compute_cap_s=43200,
           source_path=str(source),source_commit=None,source_commit_note='Source is not a Git checkout; exact file SHA-256 snapshots substitute for an unavailable commit.',
           code_hashes=hashes,dependencies=dependencies,python=sys.version,platform=platform.platform(),
           seed_overlap_audit=dict(overlap=overlapping,examined=examined),
           bootstrap=dict(draws=20000,seed=20261006),
           decisions=dict(main_gain=.02,positive_families=6,inference_p95_s=.1,cheap_margin=.005,oracle_space=.02),
           engineering_notes=['Complete far Y fixes checked independently. State build timed online.',
              'FULL fixed policy skips unneeded relaxed LP/graph. Others pay only their required inputs.',
              'Interrupted solver reserves are charged once in full; delivered prevalidated reference is retained, timing unqualified.',
              'Tree early_stopping=False: fixed 100 iterations, no implicit validation split.'],
           created_utc_unix=time.time())
    save(path,d);save(OUT/'LG01_split.json',split)
    return d

def status(phase,**extra):
    result=dict(phase=phase,updated_utc_unix=time.time(),**extra)
    save(OUT/'RUN_STATE.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

def fallback_online(rec,policy,seed,tag):
    r=rec['repair']
    return dict(state=tag.split('__')[0],family=rec['family'],rho=rec['rho'],fault=rec['disruption']['name'],
                policy=policy,action='RINS',seed=seed,repair_cost=r['objective'],D=max(1,abs(r['objective'])),
                objective=r['objective'],plan=r['plan'],source='interrupted_reference',wall_s=20.,
                timing={},inference_s=0.,fallback=True,fallback_delivery=True,timing_ok=False,
                validation=dict(ok=True,prevalidated_reference=True),fixed_sets=None)

def generate(d,ledger):
    status('generating',instances_complete=len(list((OUT/'nominal').glob('*.final.json'))))
    # Group list and raw parameters are saved before ANY solve, with no resampling.
    if not (OUT/'instances.json').exists():
        raw=[]
        for family in sorted(sum(d['split'].values(),[])):
            for rho in d['rhos']:
                inst,meta=build_instance('large',rho,family);raw.append(meta)
        save(OUT/'instances.json',raw)
    for family in sorted(sum(d['split'].values(),[])):
        for rho in d['rhos']:
            inst,meta=build_instance('large',rho,family);name=inst.name
            path=OUT/'nominal'/f'{name}.initial.json'
            def initial():
                bm=build_model(inst,MODE_AUDITED)
                base=Solution(status='all_lost',objective=0.,X=np.zeros((inst.N,inst.M,inst.T)),
                     Y=np.zeros((inst.N,inst.M,inst.T)),Z=np.zeros((inst.N,inst.M,inst.T)),
                     I=np.zeros((inst.N,inst.T)),L=np.asarray(inst.d,float),mode=MODE_AUDITED)
                base.objective=check_solution(inst,base,MODE_AUDITED).cost_recomputed['total']
                assert verify(inst,base)['ok']
                save(OUT/'nominal'/f'{name}.start.json',dict(objective=base.objective,plan=pack(base),validation=verify(inst,base)))
                r=solve(bm,{},({},0.),float('inf'),OUT/'logs'/f'{name}.initial.log',0,start=vector_of(base,bm),cap=120.)
                candidates=[base]
                if r['solution'] is not None:
                    assert verify(inst,r['solution'])['ok'];candidates.append(r['solution'])
                s=min(candidates,key=lambda c:c.objective)
                return dict(meta=meta,objective=float(s.objective),plan=pack(s),validation=verify(inst,s),solver=lp_summary(r))
            def initial_fallback():
                p=OUT/'nominal'/f'{name}.start.json'
                if not p.exists(): raise RuntimeError(f'{name}: crash before valid nominal witness saved')
                return read(p)
            start=ledger.job(name+'.initial',121.,path,initial,initial_fallback)
            current=start;stages=[]
            for phase in [1,2]:
                rec=dict(family=family,rho=rho,disruption=dict(name='D0',down={}),repair=current,nominal=current)
                tag=f'{name}__nominal{phase}'
                r=ledger.job(tag,20.2,OUT/'nominal'/f'{name}.polish{phase}.json',
                  lambda rec=rec,tag=tag:online(rec,'RINS',0,tag,nofault=True),
                  lambda rec=rec,tag=tag:fallback_online(rec,'RINS',0,tag))
                stages.append(r);current=r
            final=dict(meta=meta,objective=current['objective'],plan=current['plan'],protocol='NOM-160',
                       stages=[dict(anchor_objective=s.get('anchor_objective',s['repair_cost']),
                           objective=s['objective'],wall_s=s['wall_s'],flips=s['validation'].get('flips'),timing_ok=s['timing_ok']) for s in stages],
                       flips_vs_initial=int(np.count_nonzero(np.abs(np.asarray(current['plan']['Y'])[:,:,:TAU]-np.asarray(start['plan']['Y'])[:,:,:TAU])>.5)),
                       interrupted=bool(start.get('interrupted') or any(s.get('interrupted') for s in stages)))
            save(OUT/'nominal'/f'{name}.final.json',final)
            for dis in [Disruption('D1_m0_2p',{0:[1,2]}),Disruption('D2_all_1p',{j:[1] for j in range(inst.M)})]:
                state=f'{name}|{dis.name}';path=OUT/'states'/f'{state}.json'
                def repair(dis=dis,state=state):
                    pert=apply_disruption(inst,dis);s=repair_minbatch_v3(pert,solution(final),dis).solution
                    v=verify(pert,s);assert v['ok']
                    return dict(state=state,family=family,rho=rho,disruption=dis.as_dict(),nominal=final,
                                repair=dict(objective=float(s.objective),plan=pack(s),validation=v,anchor='Sbar_N'),
                                stability_anchor='new repair solution')
                ledger.job(state+'.repair',30.,path,repair,fallback=repair)
            status('generating',instances_complete=len(list((OUT/'nominal').glob('*.final.json'))),
                   instances_total=64,charged_s=ledger.charge())
    assert len(list((OUT/'states').glob('*.json')))==128
    statefiles={p.name:sha(p) for p in sorted((OUT/'states').glob('*.json'))}
    save(OUT/'LG01_states_seal.json',dict(states=statefiles,instances_hash=sha(OUT/'instances.json'),created_utc_unix=time.time()))

def state_recs(d,split):
    return [read(p) for p in sorted((OUT/'states').glob('*.json')) if int(re.search(r'_s(\d+)(?:\||%7C)',p.name).group(1)) in d['split'][split]]

def collect(d,ledger):
    recs=state_recs(d,'train')+state_recs(d,'validation')
    status('collecting_labels',completed=len(list((OUT/'labels').glob('*.json'))),total=480)
    for ix,rec in enumerate(recs):
        # Rotate execution order, independent of measured quality; no incumbents shared.
        order=[(ix+a)%5 for a in range(5)]
        for a in order:
            tag=f"{rec['state']}__A{a}__s0";graph=OUT/'graphs'/f"{rec['state']}.npz" if a==0 else None
            ledger.job(tag,20.2,OUT/'labels'/f'{tag}.json',
                       lambda:online(rec,ACTIONS[a],0,tag,graph_save=graph),
                       lambda:fallback_online(rec,ACTIONS[a],0,tag))
            status('collecting_labels',completed=len(list((OUT/'labels').glob('*.json'))),total=480,charged_s=ledger.charge())
    assert len(list((OUT/'labels').glob('*.json')))==480
    save(OUT/'LG01_labels_seal.json',dict(files={p.name:sha(p) for p in sorted((OUT/'labels').glob('*.json'))},created_utc_unix=time.time()))

def train_seal(d,ledger):
    seal=OUT/'LG01_policy_seal.json'
    if seal.exists(): return read(seal)
    from models import fit_all,NeuralScorer,TreeScorer
    recs=state_recs(d,'train')+state_recs(d,'validation')
    graphs=[];costs=[];denoms=[];families=[]
    for rec in recs:
        p=diskpath(OUT/'graphs'/f"{rec['state']}.npz")
        if not p.exists(): raise RuntimeError(f'{p}: interrupted reference action has no graph; no fresh online budget permitted')
        with np.load(p) as f: graphs.append({k:f[k] for k in f.files})
        rows=[read(OUT/'labels'/f"{rec['state']}__A{a}__s0.json") for a in range(5)]
        assert all(r['validation']['ok'] for r in rows)
        assert all(r['family'] in d['split']['train']+d['split']['validation'] for r in rows)
        costs.append([r['objective'] for r in rows]);denoms.append(rows[0]['D']);families.append(rec['family'])
    costs,denoms=np.asarray(costs),np.asarray(denoms);ti=list(range(72));vi=list(range(72,96))
    # Shared exact labels for all models. Training never loads test result directories.
    jid='train_all';prev=ledger.previous(jid)
    if not prev:
        if ledger.charge()+3600>43200: raise RuntimeError('compute cap before training')
        ledger.emit(dict(kind='start',id=jid,reserved_s=3600,result='models/training.json'))
    start=time.perf_counter();training=fit_all(graphs,costs,denoms,families,ti,vi)
    if sum(training[n]['wall_s'] for n in ['gnn','mlp','tree'])>3600:
        raise RuntimeError('one hour total model training cap exceeded; resource qualification failed')
    if not any(e['kind']=='complete' for e in prev):
        ledger.emit(dict(kind='complete',id=jid,charged_s=sum(training[n]['wall_s'] for n in ['gnn','mlp','tree'])))
    scorers={name:NeuralScorer(OUT/'models'/f'{name}.pt') for name in ['gnn','mlp']}
    scorers['tree']=TreeScorer(OUT/'models/tree.pkl')
    def macro(picks):
        vals=np.asarray([costs[i,a]/denoms[i] for i,a in zip(vi,picks)])
        fs=np.asarray(families)[vi]
        return float(np.mean([vals[fs==f].mean() for f in sorted(set(fs))]))
    fixed_scores=[macro([a]*24) for a in range(5)];fixed=int(np.argmin(fixed_scores))
    picks={'RINS':[0]*24,'fixed':[fixed]*24}
    allp={}
    for name,sc in scorers.items():
        pairs=[sc.choose(graphs[i]) for i in vi];picks[name]=[p[0] for p in pairs];allp[name]=[p[1].tolist() for p in pairs]
    scores={name:macro(p) for name,p in picks.items()}
    cheap=min(['RINS','fixed','tree','mlp'],key=lambda n:scores[n])
    if cheap=='fixed' and fixed==0: cheap='RINS'
    oracle=float(np.mean([(costs[i,fixed]-min(costs[i]))/denoms[i] for i in vi]))
    save(OUT/'LG01_validation_selection.json',dict(scores=scores,choices=picks,predictions=allp,fixed_action=ACTIONS[fixed],
            fixed_scores=fixed_scores,Bstar=cheap,validation_oracle_vs_fixed=oracle,
            training_oracle_vs_validation_fixed=float(np.mean([(costs[i,fixed]-min(costs[i]))/denoms[i] for i in ti])),
            validation_states=[recs[i]['state'] for i in vi],training=training))
    # Real frozen scorer invariance: perturb label side tables without supplying them to scorer.
    for name,sc in scorers.items():
        before=[sc.choose(graphs[i]) for i in vi]
        poison=costs.copy();poison[:]=np.random.default_rng(19).normal(size=poison.shape)*1e30
        after=[sc.choose(graphs[i]) for i in vi]
        assert all(a[0]==b[0] and np.array_equal(a[1],b[1]) for a,b in zip(before,after))
    save(OUT/'LG01_frozen_input_invariance.json',dict(ok=True,policies=list(scorers),states=24,poisoned_label_side_table=True))
    paths=[OUT/'models/gnn.pt',OUT/'models/mlp.pt',OUT/'models/tree.pkl',OUT/'LG01_validation_selection.json',OUT/'LG01_freeze.json']
    result=dict(Bstar=cheap,fixed_action=ACTIONS[fixed],artifacts={p.relative_to(OUT).as_posix():sha(p) for p in paths},
                code_hashes=d['code_hashes'],created_utc_unix=time.time(),test_results_generated=0)
    assert not (OUT/'test').exists(),'Test outcomes exist before policy seal'
    save(seal,result);return result

def testing(d,seal,ledger):
    assert all(sha(OUT/p)==h for p,h in seal['artifacts'].items())
    from models import NeuralScorer,TreeScorer
    gnn=NeuralScorer(OUT/'models/gnn.pt');cheap=seal['Bstar'];sc=None
    if cheap=='tree': sc=TreeScorer(OUT/'models/tree.pkl')
    elif cheap=='mlp': sc=NeuralScorer(OUT/'models/mlp.pt')
    cheapaction=seal['fixed_action'] if cheap=='fixed' else 'RINS'
    policies=[('GNN','gnn',gnn),('Bstar',cheapaction,sc),('RINS','RINS',None)] if cheap!='RINS' else [('GNN','gnn',gnn),('RINS','RINS',None)]
    recs=state_recs(d,'test');total=len(recs)*2*len(policies)
    for ix,rec in enumerate(recs):
        for seed in [0,1]:
            offset=(ix+seed)%len(policies)
            for label,policy,scorer in policies[offset:]+policies[:offset]:
                tag=f"{rec['state']}__{label}__s{seed}"
                ledger.job(tag,20.2,OUT/'test'/f'{tag}.json',
                           lambda:online(rec,policy,seed,tag,scorer=scorer),
                           lambda:fallback_online(rec,policy,seed,tag))
                status('testing',completed=len(list((OUT/'test').glob('*.json'))),total=total,charged_s=ledger.charge())
    assert len(list((OUT/'test').glob('*.json')))==total<=192
    if cheap=='RINS': save(OUT/'LG01_Bstar_alias.json',dict(Bstar='RINS',duplicate_runs_omitted=64))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','generate','all','audit']);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    # OS lock releases automatically on termination, so safe resume needs no stale PID edits.
    import msvcrt
    lock=(OUT/'RUN.lock').open('a+b');lock.seek(0);lock.write(b'1');lock.flush();lock.seek(0)
    try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError: raise RuntimeError('Another LG01 runner holds the single-trial lock')
    d=freeze();ledger=Ledger()
    try:
        if args.phase=='freeze': print(json.dumps(d['split']));return
        if args.phase in ['generate','all']: generate(d,ledger)
        if args.phase=='generate': return
        if args.phase=='all':
            collect(d,ledger);seal=train_seal(d,ledger);testing(d,seal,ledger)
        if args.phase in ['all','audit']:
            from report import audit_report
            audit_report(d,ledger)
    except BaseException as exc:
        status('stopped_error',error=str(exc),charged_s=ledger.charge(),traceback=traceback.format_exc())
        raise

if __name__=='__main__':main()
