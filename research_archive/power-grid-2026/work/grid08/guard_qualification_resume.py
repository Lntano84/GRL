"""Four training-week fixed-rule runs; no model selection or final-test use."""
import gzip,traceback
from two_step_guard import *
FOLDER=OUT/'guard_qualification';FOLDER.mkdir(exist_ok=True)
assert not (FOLDER/'finished.json').exists()
config={'weeks':['2035-10-29_15','2035-01-29_4'],
        'rules':['ALWAYS_ONE_STEP','ALWAYS_TWO_STEP'],
        'seed':0,'cadence':6,'max_physical_steps':8068,
        'scope':'Training-only engineering/failure qualification. V2 unchanged, no fitting. No claim of validation gain.'}
RESUME=json.loads((OUT/'resume_01.json').read_text())['plans']['guard']
records=json.loads((FOLDER/'manifest.json').read_text());physical=sum(r['steps'] for r in records)
assert len(records)==RESUME['completed_episodes']
done={(r['week'],r['rule']) for r in records}
start=time.perf_counter();env=make_env()
try:
    for week in config['weeks']:
        for rule in config['rules']:
            if (week,rule) in done:continue
            folder=FOLDER/(week+'__'+rule);folder.mkdir(exist_ok=False)
            env.seed(0);env.set_id(week);obs=env.reset();reward=0.;cost=0.;offered=chosen=veto=checked=0
            ctl=(ResidualControl if rule=='ALWAYS_ONE_STEP' else TwoStepResidualControl)(env,obs)
            horizon=env.max_episode_duration();rows=[];arrays={k:[] for k in ['action','rho','gen_p','load_p','actual_dispatch','curtailment_mw','storage_power']}
            t=time.perf_counter();forecast_steps=0;simulations=0
            write_json(folder/'metadata.json',{'scenario':week,'policy':rule,'horizon':horizon,'seed':0,
                'gen_cost_per_MW':env.gen_cost_per_MW.tolist(),'dt_hours':float(env.delta_time_seconds)/3600})
            with gzip.open(folder/'steps.jsonl.gz','wt',encoding='utf-8') as log:
                for step in range(1,horizon+1):
                    assert physical<config['max_physical_steps'] and time.perf_counter()-start<3600
                    oldcurt=float(obs.curtailment_mw.sum(dtype=np.float64));before=digest(obs.to_vect());ns=int(env.nb_highres_called)
                    base,rest,proposal=ctl.propose(obs,reward,6)
                    action=rest if rest is not None else base
                    offered+=proposal['offered'];chosen+=int(rest is not None)
                    guard=proposal.get('two_step_guard',{});checked+=guard.get('checked',False)
                    veto+=proposal['reason']=='two_step_restore_veto';forecast_steps+=guard.get('forecast_steps',0)
                    obs,reward,done,info=env.step(action);physical+=1
                    complete=bool(obs.current_step>=horizon or env.chronics_handler.done());fl=flags(info)
                    raw=float(info['rewards'][f'{PREFIX}_grid_operational_cost']);cost+=raw
                    ledger=public_cost_ledger(obs,env,oldcurt) if not fl['exceptions'] else None
                    tol=float(cost_rounding_tolerance(obs,env)) if ledger else None
                    if ledger:assert abs(ledger['recomputed_raw_cost']-raw)<=tol
                    sim=int(env.nb_highres_called)-ns;simulations+=sim
                    row={'step':step,'raw_cost':raw,'ledger':ledger,'ledger_tolerance':tol,
                         'before_hash':before,'after_hash':digest(obs.to_vect()),
                         'action_hash':digest(action_vector(action)),'done':bool(done),'complete':complete,
                         'proposal':proposal,**fl}
                    log.write(json.dumps(row,default=json_default,allow_nan=False)+'\n');log.flush()
                    for k in arrays:arrays[k].append(action_vector(action) if k=='action' else np.array(getattr(obs,k),copy=True))
                    if step%256==0 or done:print(json.dumps({'week':week,'rule':rule,'step':step,'veto':veto,'cost':cost}),flush=True)
                    if done:break
            np.savez_compressed(folder/'vectors.npz',**{k:np.stack(v) for k,v in arrays.items()})
            result={'week':week,'rule':rule,'steps':step,'complete':complete,'raw_cost':cost,
                    'offered':offered,'selected':chosen,'guard_checks':checked,'vetoes':veto,
                    'explicit_two_step_forecast_steps':forecast_steps,'native_highres_count':simulations,
                    'wall_s':time.perf_counter()-t,'path':str(folder.relative_to(ROOT))}
            write_json(folder/'summary.json',result);records.append(result);write_json(FOLDER/'manifest.json',records)
    write_json(FOLDER/'finished.json',{'physical_steps':physical,'records':records,'wall_s':time.perf_counter()-start,'wall_scope':'Resumed invocation only; interrupted attempt retained.'})
except Exception:
    write_json(FOLDER/'failure.json',{'physical_steps':physical,'traceback':traceback.format_exc()});raise
finally:env.close()
