"""Prespecify finite four-date, six-arm same-batch development comparison."""
import collections,copy,importlib.util,json,pickle,shutil,subprocess,sys
import numpy as np
from registry import ROOT,W,OUT,ARCHIVE,read,save,sha,test_sealed,utc

def main():
    recovering='--resume-prefit' in sys.argv
    if OUT.exists():
        assert recovering and not (OUT/'code_freeze.json').exists() and not (OUT/'runs').exists()
        assert read(OUT/'manifest.json')['summaries']==[]
        prior=read(OUT/'design.json')
    else:assert not recovering
    test_sealed();assert read(ROOT/'outputs/grid103_learning_mechanism/complete.json')['passed']
    for name in ['code_freeze.json','data_freeze.json']:
        for p,h in read(ROOT/'outputs/grid102_search_rl'/name).items():assert sha(ROOT/p)==h,p
    for p,h in read(ROOT/'outputs/grid103_learning_mechanism/freeze.json').items():assert sha(ROOT/p)==h,p
    OUT.mkdir(exist_ok=recovering);sys.path.insert(0,str(ROOT/'work/grid102_search_rl'))
    s=importlib.util.spec_from_file_location('grid104_preflight_policy',W/'policies.py');P=importlib.util.module_from_spec(s);s.loader.exec_module(P)
    from common import read as oldread
    trainplan=read(ROOT/'outputs/grid100_data_foundation/design.json')
    selected=[r for r in trainplan['scenario_plan'] if r['partition']=='validation' and r['replicate']==0]
    assert len(selected)==4 and all(r['scenario'].endswith('_100') for r in selected)
    metadata=read(ROOT/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']
    references={r['scenario']:str(__import__('pathlib').Path(next(m['source_call'] for m in metadata if m['scenario']==r['scenario'])).parent/'summary.json').replace('\\','/') for r in selected}
    names=['FRAME_GUARD','LOCAL_BATCH1','LOCAL_RF','LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2']
    methods=[dict(method=m,policy=m,rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,order_mode='STATIC_ID',
        model_seed=int(m[-1]) if m.startswith('LOCAL_DQN0_') else None) for m in names]
    order=[]
    for repeat in [0,1]:
        for row in selected if repeat==0 else selected[::-1]:
            i=selected.index(row);block=methods[i:]+methods[:i]
            if repeat:block=block[::-1]
            order.extend(dict(m,week=row['scenario'],date=row['date'],repeat=repeat,stage_phase='all',partition='validation_development') for m in block)
    D=dict(id='GRID104_LEARNED_LOCAL_DEVELOPMENT_CLOSED_LOOP',frozen_utc=utc(),authorization='User: autonomously advance existing idea, notify before scaling.',
        environment=trainplan['environment'],environment_seed=0,selected=selected,full_references=references,selection='All4 validation calendar dates, first pre-existing replicate100; no outcome-based date/replicate selection.',
        methods=methods,execution_order=order,repeats=2,parameters=dict(p=128,h=64,patience=2,safe=.9),
        learning='Only completion depth changes; same global FRAME, initial station, aliases/order, controller/reward/caches. Fixed models from GRID103, no fits.',
        timing='Same-process monotonic wall clock around complete ctl.propose, includes public context, visible-feature projection and single-state inference. Model loading and one dummy warm-up excluded as setup for all learned arms and retained in wall/setup records. Serialized vector/call audits outside control. No concurrent benchmark/training/heavy audit.',
        caps=dict(episode_s=1200,wall_s=14400,physical_steps=100000,public_forecasts=4000000),
        quality_threshold=.005,
        quality='Compare all4 scenarios with same-batch FRAME and BATCH1. If reference completes, require completion and whole-week cost<=100.5%; if reference fails early, require at least its steps and report no full-week cost comparison. Retain early failures. Old FULL only survival/whole-week-quality reference, never old timing.',
        scale_gate='Exploratory readiness only, not deployment: learner quality all4 against FRAME and BATCH1; identical horizon/completion to both for compute; macro actual control and matched-active query reduction>=5% versus both, >=3/4 faster dates, both timing-repeat macro time reductions positive. RF evaluated as one fixed candidate; DQN family needs all3 seeds quality and horizons, >=2/3 seeds compute against both plus family means. No best-seed selection.',
        limits='Four already exposed dates, same public grid; first-replicate development, not independent test or cross-grid generalization. Timing repeats/seeds are not independent scenarios. No training/new scenarios/threshold tuning/price sweep or final8 test access. Any scaling awaits user notification.')
    if recovering:assert {k:v for k,v in D.items() if k!='frozen_utc'}=={k:v for k,v in prior.items() if k!='frozen_utc'}
    save(OUT/'design.json',D);save(OUT/'manifest.json',dict(planned=48,summaries=[]))
    checks=collections.Counter()
    paths=sorted({r['source_call'] for r in metadata if r['scenario'] in {x['scenario'] for x in selected}})
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    policies={m:P.CompletionPolicy(m,int(m[-1]) if m.startswith('LOCAL_DQN0_') else None) for m in names}
    old=read(ROOT/'outputs/grid103_learning_mechanism/state_predictions.json');expected={(r['sample_id'],r['position']):r for r in old}
    with (ROOT/'outputs/grid102_search_rl/cases.pkl').open('rb') as f:cases=pickle.load(f)
    for c in cases:
        if c['partition']!='validation':continue
        for j,state in enumerate(c['states']):
            for m in ['LOCAL_RF','LOCAL_DQN0_0','LOCAL_DQN0_1','LOCAL_DQN0_2']:
                pp=policies[m];obs=np.clip((state-pp.mean)/pp.scale,-20,20).astype(np.float32)
                a,_=pp.predict(obs,int(c['queries'][j]));oldname='MYOPIC_RF64' if m=='LOCAL_RF' else m.replace('LOCAL_DQN0_','DQN_G0_')
                assert a==expected[c['sample_id'],j]['actions'][oldname];checks['old_single_state_predictions']+=1
    for p in paths:
        call=read(ROOT/p);truth={r['pool_id']:r for r in call['outcomes']};aliases={int(k):int(v) for k,v in call['aliases'].items()}
        for name,policy in policies.items():
            visited=[]
            def reveal(i,phase):
                assert i not in visited;visited.append(i);return dict(truth[i],phase=phase)
            result=policy.run(call['order'],reveal,subs,aliases,call['rho_limit'])
            global_visited=[]
            def global_reveal(i,phase):global_visited.append(i);return dict(truth[i],phase=phase)
            reference=P.G.run(call['order'],global_reveal,'FRAME_GUARD' if name=='FRAME_GUARD' else 'NO_LOCAL',subs,aliases,limit=call['rho_limit'])
            assert visited[:len(global_visited)]==global_visited
            if name=='FRAME_GUARD':assert result['chosen_pool_id']==reference['chosen_pool_id'] and visited==global_visited
            else:
                assert result['extra_selected_ids']==reference['completion_candidate_ids'][:result['local_queries']]
                assert result['local_queries']<=reference['completion_budget']
                if name=='LOCAL_BATCH1':assert result['local_queries']==min(64,reference['completion_budget'])
                valid=[r for r in result['outcomes'] if r['strict_admissible']]
                assert result['chosen_pool_id']==(max(valid,key=P.key)['pool_id'] if valid else None)
            assert len(visited)==len(set(visited))==result['public_queries'];checks['full_call_policy_replays']+=1
    for params in ['caps','execution_order','timing']:assert params in D
    assert len(order)==48
    with (OUT/'live_qualification.log').open('x',encoding='utf8') as log:
        subprocess.run([sys.executable,'-u',str(W/'qualify_live.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,timeout=120)
    assert read(OUT/'live_qualification.json')['passed']
    save(OUT/'preflight.json',dict(passed=True,checks=dict(checks),planned=48,new_model_fits=0,final_test_materialized=False,live_forecasts=1))
    inherited=read(ROOT/'outputs/grid100_data_foundation/bulk_code_freeze.json')
    for p,h in inherited.items():assert sha(ROOT/p)==h,p
    files=list(W.glob('*.py'))+[OUT/'design.json',OUT/'preflight.json',OUT/'live_qualification.json']
    files += [ROOT/'outputs/grid102_search_rl/normalization.npz',ROOT/'outputs/grid102_search_rl/design.json',ROOT/'work/grid102_search_rl/observations.py',ROOT/'work/grid101_gain_learning/features.py',ROOT/'work/grid91_mechanism/policies.py',ROOT/'outputs/grid103_learning_mechanism/models/next_batch_rf.joblib']
    files += [ROOT/f'outputs/grid103_learning_mechanism/models/dqn_gamma0_seed{s}_final.zip' for s in [0,1,2]]
    inherited.update({p.relative_to(ROOT).as_posix():sha(p) for p in files})
    save(OUT/'code_freeze.json',inherited)
    data={}
    for p in sorted((ROOT/D['environment']).iterdir()):
        if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
    for r in selected:
        receipt=read(ROOT/'outputs/grid100_data_foundation/generation_receipts'/f"{r['scenario']}.json")
        for p,h in receipt['files'].items():assert sha(ROOT/p)==h,p;data[p]=h
    for p in references.values():data[p]=sha(ROOT/p)
    save(OUT/'data_freeze.json',data)
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file()]:shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    (OUT/'RUN_STATE.md').write_text('FROZEN:48 serial existing-development rollouts; zero fits/new scenarios; final test sealed.\n',encoding='utf8')
    print(json.dumps(read(OUT/'preflight.json')),flush=True)
if __name__=='__main__':main()
