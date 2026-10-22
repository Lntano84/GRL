"""Freeze three serial March replays, two single interventions, before execution."""
import copy,re,shutil
import numpy as np
from common106 import ROOT,W,OUT,ARCHIVE,read,save,sha,digest,loadmod,sealed_test,utc
def main():
    assert not OUT.exists();sealed_test()
    old=ROOT/'outputs/grid104_learning_closedloop';assert read(old/'review.json')['passed_saved_data']
    inherited=read(old/'code_freeze.json');data=read(old/'data_freeze.json')
    for p,h in {**inherited,**data}.items():assert sha(ROOT/p)==h,p
    ref=old/'runs/r0__2037-03-02_100__FRAME_GUARD';calls=[(p,read(p)) for p in sorted(ref.glob('call[0-9][0-9][0-9].json'))]
    cases=[]
    for t in [108,552]:
        p,c=next((p,c) for p,c in calls if c['step']==t)
        with np.load(p.with_name(p.stem+'_state.npz')) as z:bh=digest(z['base_action'])
        cases.append(dict(id=f'MARCH_{t}',step=t,week='2037-03-02_100',source=p.relative_to(ROOT).as_posix(),
            state_hash=c['state_hash'],base_action_hash=bh,before_choice=c['pre_extra_choice'],after_choice=c['chosen_pool_id'],
            completion_budget=c['completion_budget']))
    assert [c['completion_budget'] for c in cases]==[198,13]
    base=dict(week='2037-03-02_100',date='2037-03-02',repeat=0,policy='FRAME_GUARD',rule='STATIC_ID',budget=128,
        checkpoint=None,cache_dc=True,cache_safe=True,order_mode='STATIC_ID',model_seed=None,stage_phase='all',partition='exposed_validation_diagnostic')
    order=[dict(base,method='FRAME_QUAL',omit_case=None,watch_cases=[c['id'] for c in cases])]
    order.extend(dict(base,method=f'SKIP_{c["step"]}',omit_case=c['id'],watch_cases=[c['id']]) for c in cases)
    D=dict(id='GRID106_TWO_SINGLE_DECISION_BRANCHES',frozen_utc=utc(),
        authorization='User: inspect these two decisions carefully; notify before any training. No training authorized this stage.',
        environment=read(old/'design.json')['environment'],reference=ref.relative_to(ROOT).as_posix(),cases=cases,execution_order=order,
        parameters=dict(p=128,h=64,patience=2,safe=.9),
        caps=dict(wall_s=1800,episode_s=500,physical_steps=6100,public_forecasts=100000),
        timing='Diagnostic wall clocks include memory witnesses; no speed claim or mixing historical control time.',
        primary='Single omission causal effect under otherwise original FRAME from the same full reconstructed prefix, until original week end.',
        horizons=dict(one_step=1,one_hour=12,one_day=288,remaining_week='Original2017-step endpoint; report all windows, do not select.'),
        label='First survival to endpoint, then raw operational cost and all/candidate queries separately. No fitted conversion weight. Opponent realization differences mark unresolved.',
        qualification='New full FRAME must reproduce all historical2017 physical steps and paid feedback before either intervention; branch prefix and numeric/used object RNG witnesses must match.',
        no_training=True,no_new_scenarios=True,final_test_untouched=True,
        limitations='Two intentionally selected failures on one exposed validation date. Not training data, independent test, scaling, or proof that first divergence explains all policy-week gaps.')
    OUT.mkdir();save(OUT/'design.json',D);save(OUT/'manifest.json',dict(planned=3,summaries=[]))
    G=loadmod('grid106_preflight_core','work/grid91_mechanism/policies.py');subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];checks=0
    for p,c in calls:
        truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):int(v) for k,v in c['aliases'].items()}
        for omit in [False,True] if c['step'] in [108,552] else [False]:
            seen=[]
            def reveal(i,phase):
                assert i not in seen;seen.append(i);return dict(truth[i],phase=phase)
            result=G.run(c['order'],reveal,'NO_LOCAL' if omit else 'FRAME_GUARD',subs,aliases,limit=c['rho_limit'])
            assert result['chosen_pool_id']==(c['pre_extra_choice'] if omit else c['chosen_pool_id'])
            assert result['public_queries']==c['public_queries']-(c['local_queries'] if omit else 0)
            checks+=1
    # Inherited RNG exclusions are qualified by unchanged source, not silently broadened.
    proof=['work/grid02/ljn_nn/modules/base_module.py','work/grid02/ljn_nn/modules/topology_heuristic.py','work/grid02/ljn_nn/modules/convex_optim.py']
    for p in proof:assert not re.search(r'space_prng|np\.random|random\.|\.sample\(|\.shuffle\(|\.permutation\(',(ROOT/p).read_text(encoding='utf8'))
    assert 'warm_start=False' in (ROOT/proof[-1]).read_text(encoding='utf8')
    for p in [W/'worker.py']:
        compile(p.read_text(encoding='utf8'),str(p),'exec')
    save(OUT/'preflight.json',dict(passed=True,saved_call_replays=checks,case_budgets=[198,13],original_source_hashes_checked=len(inherited)+len(data),
        no_replay_or_fits_yet=True,unused_rng_proof_sources=proof,expected_runs=3,mandatory_worker_keys=['caps','execution_order','timing']))
    paths=list(W.glob('*.py'))+[OUT/'design.json',OUT/'preflight.json']+[ROOT/p for p in proof]
    paths += [ROOT/'work/grid95_labels/label_hooks.py',ROOT/'work/grid95_labels/audit_memory_v2.py']
    inherited.update({p.relative_to(ROOT).as_posix():sha(p) for p in paths});save(OUT/'code_freeze.json',inherited)
    for p in ref.iterdir():
        if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'data_freeze.json',data)
    dest=ARCHIVE/'frozen';dest.mkdir(parents=True,exist_ok=False)
    for p in list(W.glob('*.py'))+list(OUT.iterdir()):
        if p.is_file():shutil.copy2(p,dest/p.name)
    save(dest/'archive_manifest.json',{p.name:sha(p) for p in dest.iterdir() if p.is_file()})
    save(OUT/'pipeline_state.json',dict(phase='frozen_not_started',planned=3,completed=0))
    print(read(OUT/'preflight.json'),flush=True)
if __name__=='__main__':main()
