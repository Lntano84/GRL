"""PARTIAL saved-data audit: nine completed runs, import timeout retained; no simulation or fitting."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid06'
OLD = ROOT / 'outputs/grid04'
started = time.perf_counter()

def load(p): return json.loads(p.read_text(encoding='utf-8'))
def lines(p): return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def save(name, value): (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
def stats(v):
    return dict(sum=float(sum(v)), p50=float(np.percentile(v,50)), p95=float(np.percentile(v,95)), max=float(max(v))) if v else None

design = load(OUT/'design_freeze.json')
assert sha(OUT/'design_freeze.json') == (OUT/'design_freeze.sha256').read_text().strip()
attempts=load(OUT/'run_manifest.json')
assert len(attempts)==10 and all(r['exit']==0 for r in attempts[:9]) and attempts[-1]['exit']!=0
assert [{k:r[k] for k in ('scenario','policy')} for r in attempts]==design['plan'][:10]
manifest=attempts[:9]
failed=attempts[-1]
failed_folder=OUT/'runs'/f"{failed['scenario']}__{failed['policy']}"
assert set(p.name for p in failed_folder.iterdir())=={'console.log','watchdog_timeout.json'}
watchdog=load(failed_folder/'watchdog_timeout.json')
assert not watchdog['import_seen'] and watchdog['wall_s']>design['caps']['import_s']
complete_weeks=design['selected_weeks'][:3]
for p,h in load(OUT/'execution_code_frozen.json').items(): assert sha(ROOT/p) == h
reused = load(OUT/'reused_assets_frozen.json')
for p,h in reused.items(): assert sha(ROOT/p) == h
pre = load(OUT/'qualification.json')
assert pre['passed'] and pre['physical_steps'] == pre['native_simulates'] == 0
for p,h in load(OUT/'data_files_frozen.json').items(): assert sha(ROOT/p) == h
pool = load(OLD/'pool_metadata.json')['action_hashes']
assert len(set(pool)) == len(pool) == 352

records, metas, summaries, expansion_details = {}, {}, [], []
vectors_checked = ledgers_checked = module_checks = shortlist_checks = 0
maximum_residual = 0.
for run in manifest:
    key = (run['scenario'],run['policy'])
    folder = OUT/'runs'/f"{key[0]}__{key[1]}"
    meta = load(folder/'metadata.json')
    end = load(folder/'completion.json')
    execution = load(folder/'execution_count.json')
    rows, events = lines(folder/'steps.jsonl'), lines(folder/'events.jsonl')
    assert len(rows) == end['steps'] == execution['physical_calls'] == execution['saved_steps']
    assert meta['scenario'] == key[0] and meta['policy'] == key[1]
    assert meta['seed'] == 0 and meta['max_steps'] == 2017 and len(rows) <= 2017
    assert meta['pool_size']==352 and meta['top_k']==(352 if key[1]=='NN352' else 20)
    assert meta['fixed_fallback'] == dict(enabled=key[1]=='FALLBACK90', threshold=.9, expanded_pool=352, forced_test=False)
    assert not any(e['kind'] == 'module_error' for e in events)
    by_step = defaultdict(list)
    for e in events: by_step[e['step']].append(e)
    assert set(by_step).issubset(set(range(1,len(rows)+1)))
    with np.load(folder/'initial.npz',allow_pickle=False) as z:
        previous = z['observation'].copy()
        previous_curtailed = float(z['curtailment_mw'].sum(dtype=np.float64))
    initial_hash = digest(previous)
    prices = np.asarray(meta['gen_cost_per_MW'],dtype=np.float64)
    dt = meta['delta_time_seconds']/3600
    assert dt == 1/12
    offsets = {}; cursor = 0
    for field in meta['action_vector_layout']:
        offsets[field['name']] = slice(cursor,cursor+field['size']); cursor += field['size']
    assert cursor == 1605
    for i,row in enumerate(rows,1):
        assert row['step'] == i and row['before_observation_hash'] == digest(previous)
        assert row['physical_observation_unchanged_during_act']
        file = folder/row['file']
        assert sha(file) == row['file_sha256']
        sim = [e for e in by_step[i] if e['kind'] == 'simulate']
        assert len(sim) == row['simulate_calls'] and all(e['horizon']==1 for e in sim)
        assert sum(e['wall_s'] for e in sim) <= row['act_wall_s'] + 1e-6
        with np.load(file,allow_pickle=False) as z:
            assert digest(z['observation']) == row['observation_hash']
            assert digest(z['action']) == row['action_hash']
            assert float(z['rho'].max()) == row['after_rho_max']
            assert np.isfinite(z['action']).all() and np.isfinite(z['rho']).all()
            assert int(np.count_nonzero(z['action'][offsets['_set_topo_vect']])) == row['changed_topology_entries']
            if row['ledger_eligible']:
                margin = float(max(prices[z['gen_p']>0]))
                loss = (z['gen_p'].astype(np.float64).sum()-z['load_p'].astype(np.float64).sum())*dt
                redisp = np.abs(z['actual_dispatch'].astype(np.float64)).sum()*dt
                current_curtailed = float(z['curtailment_mw'].astype(np.float64).sum())
                curt = current_curtailed*dt
                delta = (current_curtailed-previous_curtailed)*dt
                storage = np.abs(z['storage_power'].astype(np.float64)).sum()*dt
                cost = margin*(loss+redisp+delta+storage)
                for name,value in [('marginal_cost',margin),('losses_mwh',loss),('redispatch_mwh',redisp),
                    ('curtailed_mwh',curt),('curtailment_delta_mwh',delta),('storage_throughput_mwh',storage),('recomputed_raw_cost',cost)]:
                    assert abs(row['ledger'][name]-value) < 1e-8, (key,i,name)
                residual = abs(cost-row['raw_operational_cost'])
                assert residual <= row['cost_rounding_tolerance']
                maximum_residual = max(maximum_residual,residual); ledgers_checked += 1
            previous_curtailed = float(z['curtailment_mw'].astype(np.float64).sum())
            previous = z['observation'].copy(); vectors_checked += 1

        # Reconstruct the author's eligibility and float32 reward, without importing it.
        topo = [e for e in by_step[i] if e['kind']=='module' and e['module']=='topo_12_unsafe']
        short = [e for e in by_step[i] if e['kind']=='shortlist']
        fb = [e for e in by_step[i] if e['kind']=='fallback']
        topo_sim = [e for e in sim if e['module']=='topo_12_unsafe']
        if not topo:
            assert not short and not fb and not topo_sim
            continue
        assert len(topo)==1
        module = topo[0]
        initial_k=352 if key[1]=='NN352' else 20
        assert len(short)>=1 and short[0]['top_k']==initial_k
        initial_ids = short[0]['ids']
        assert len(set(initial_ids))==len(initial_ids)==initial_k and all(0<=j<352 for j in initial_ids)
        shortlist_checks += len(short)
        assert len(topo_sim)==module['simulate_calls']
        assert len({e['action_hash'] for e in topo_sim})==len(topo_sim), 'Cached candidates must not be simulated twice'
        def reward(e):
            eligible = e['rho_max'] is not None and 0<e['rho_max']<row['before_rho_max'] and not e['exceptions'] and not e['done']
            return np.float32(e['reward'] if eligible else -100.)
        initial_rewards = np.asarray([reward(e) for e in topo_sim[:initial_k]],dtype=np.float32)
        assert [e['action_hash'] for e in topo_sim[:initial_k]]==[pool[j] for j in initial_ids], 'Native base action differs: investigate before interpreting pool hashes'
        first_idx = int(np.argmax(initial_rewards)) if max(initial_rewards)>-100 else None
        first_hash = pool[initial_ids[first_idx]] if first_idx is not None else None
        first_rho = topo_sim[first_idx]['rho_max'] if first_idx is not None else None
        if key[1] in ('NN20','NN352'):
            assert not fb and len(short)==1 and len(topo_sim)==initial_k
            assert module['returned_action_hash']==first_hash and module['returned_none']==(first_hash is None)
        else:
            assert len(fb)==1
            event = fb[0]
            assert not event['forced_test'] and event['threshold']==.9
            assert event['first_ids']==initial_ids
            assert np.array_equal(initial_rewards,np.asarray(event['first_rewards'],dtype=np.float32))
            assert event['first_return_hash']==first_hash and event['first_selected_forecast_rho']==first_rho
            trigger = first_hash is None or first_rho>=.9
            assert event['trigger']==event['expanded']==trigger
            if trigger:
                assert len(short)==2 and short[1]['top_k']==352
                full = short[1]['ids']
                assert sorted(full)==list(range(352)) and full[:20]==initial_ids
                additional = [j for j in full if j not in set(initial_ids)]
                assert event['additional_ids']==additional and len(topo_sim)==352
                assert [e['action_hash'] for e in topo_sim[20:]]==[pool[j] for j in additional]
                rewards = dict(zip(initial_ids,initial_rewards))
                rewards.update(zip(additional,[reward(e) for e in topo_sim[20:]]))
                ordered = np.asarray([rewards[j] for j in full],dtype=np.float32)
                chosen = full[int(np.argmax(ordered))] if max(ordered)>-100 else None
            else:
                assert len(short)==1 and not event['additional_ids'] and len(topo_sim)==20
                chosen = initial_ids[first_idx]
            returned = pool[chosen] if chosen is not None else None
            assert event['returned_pool_row']==chosen and event['returned_action_hash']==returned
            assert module['returned_action_hash']==returned and module['returned_none']==(returned is None)
            expansion_details.append(dict(scenario=key[0], step=i, expanded=trigger,
                reason='no_candidate' if first_hash is None else ('rho_at_least_0.9' if trigger else 'rho_below_0.9'),
                first_selected_forecast_rho=first_rho, first_pool_row=initial_ids[first_idx] if first_idx is not None else None,
                returned_pool_row=chosen, changed_original_selection=returned!=first_hash,
                simulate_calls=len(topo_sim)))
        module_checks += 1

    final = rows[-1]
    complete = bool(final['current_step']>=meta['max_steps'] or final['chronics_done'])
    assert complete==end['reached_native_horizon']
    module_summary = defaultdict(lambda:dict(calls=0,simulate_calls=0,wall_s=0.))
    for e in events:
        if e['kind']=='module':
            m=module_summary[e['module']];m['calls']+=1;m['simulate_calls']+=e['simulate_calls'];m['wall_s']+=e['wall_s']
    valid = [r for r in rows if r['ledger_eligible']]
    run_fb = [e for e in expansion_details if e['scenario']==key[0]] if key[1]=='FALLBACK90' else []
    summaries.append(dict(**run, steps=len(rows), completed_native_week=complete, initial_observation_hash=initial_hash,
        native_raw_cost_sum=sum(r['raw_operational_cost'] for r in rows),
        verified_physical_cost_sum=sum(r['raw_operational_cost'] for r in valid),
        unverified_or_error_terminal_count=len(rows)-len(valid),
        unverified_or_error_terminal_cost_sum=sum(r['raw_operational_cost'] for r in rows if not r['ledger_eligible']),
        illegal_steps=sum(r['illegal'] for r in rows), ambiguous_steps=sum(r['ambiguous'] for r in rows),
        exception_steps=sum(bool(r['exceptions']) for r in rows), terminal_exception_types=final['exception_types'],
        simulate_calls=sum(r['simulate_calls'] for r in rows), act_time=stats([r['act_wall_s'] for r in rows]),
        modules=dict(module_summary), shortlist_calls=sum(e['kind']=='shortlist' for e in events),
        fallback_decisions=len(run_fb), expansions=sum(e['expanded'] for e in run_fb),
        expanded_changed_original_selection=sum(e['expanded'] and e['changed_original_selection'] for e in run_fb),
        trigger_reasons=dict(Counter(e['reason'] for e in run_fb)),
        solver_statuses=dict(Counter(e['problem_status'] for e in events if e['kind']=='solve' and not e['root_dispatch'])),
        losses_mwh=sum(r['ledger']['losses_mwh'] for r in valid), redispatch_mwh=sum(r['ledger']['redispatch_mwh'] for r in valid),
        physical_curtailed_mwh=sum(r['ledger']['curtailed_mwh'] for r in valid),
        storage_throughput_mwh=sum(r['ledger']['storage_throughput_mwh'] for r in valid),
        terminal_renewable_component=final['renewable_component'], terminal_assistant_component=final['assistant_component']))
    records[key]=rows; metas[key]=meta
    print('AUDITED',key,len(rows),flush=True)

paired=[]
common_fields = ['seed','parameters','rho_danger','rho_safe','optim_config','versions','action_vector_layout','gen_cost_per_MW','delta_time_seconds','gen_renewable']
original_meta=load(ROOT/'outputs/grid05/runs/2035-01-29_4__NN20/metadata.json')
for scene in complete_weeks:
    assert len({r['initial_observation_hash'] for r in summaries if r['scenario']==scene})==1
    for policy in design['policies']:
        assert all(metas[(scene,policy)][f]==original_meta[f] for f in common_fields),(scene,policy,'configuration mismatch')
    for reference,policy in [('NN20','FALLBACK90'),('NN20','NN352'),('FALLBACK90','NN352')]:
        ref,other=records[(scene,reference)],records[(scene,policy)]
        shared=[]
        for a,b in zip(ref,other):
            if a['before_observation_hash']!=b['before_observation_hash']:break
            shared.append((a,b))
        diffs=[i for i,(a,b) in enumerate(zip(ref,other),1) if a['action_hash']!=b['action_hash']]
        paired.append(dict(scenario=scene,reference=reference,other=policy,same_prior_prefix_steps=len(shared),first_action_difference=diffs[0] if diffs else None,
            identical_complete_actions=len(ref)==len(other) and not diffs,
            identical_complete_observations=len(ref)==len(other) and all(a['observation_hash']==b['observation_hash'] for a,b in zip(ref,other)),
            same_prior_reference_act_s=sum(a['act_wall_s'] for a,b in shared),same_prior_other_act_s=sum(b['act_wall_s'] for a,b in shared),
            same_prior_reference_simulates=sum(a['simulate_calls'] for a,b in shared),same_prior_other_simulates=sum(b['simulate_calls'] for a,b in shared)))

physical=sum(r['steps'] for r in summaries)
persisted=sum(p.stat().st_size for p in (OUT/'runs').rglob('*') if p.is_file())
assert physical<=design['caps']['physical_steps'] and persisted<=design['caps']['output_bytes']
save('per_run_partial.json',summaries);save('paired_trajectories_partial.json',paired);save('expansion_audit_partial.json',expansion_details)
columns=['scenario','policy','steps','completed_native_week','native_raw_cost_sum','verified_physical_cost_sum','simulate_calls','fallback_decisions','expansions','expanded_changed_original_selection','illegal_steps','ambiguous_steps','exception_steps','wall_s']
with (OUT/'GRID06_partial_per_run.csv').open('w',newline='',encoding='utf-8-sig') as f:
    writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
    for row in summaries:writer.writerow({c:row[c] for c in columns})
audit=dict(completed_data_audit_passed=True,matrix_complete=False,resource_acceptance_passed=False,
    planned_runs=12,completed_runs=9,failed_import_attempts=1,not_started_runs=2,
    completed_weeks=complete_weeks,physical_steps=physical,
    failed_attempt_physical_steps=0,failed_attempt_basis='No import_ready/initial/step files; fixed source constructs the environment and begins physical advancement only after writing import_ready. This is not a strategy result.',
    vectors_rehashed_and_rechecked=vectors_checked,public_ledgers_recomputed=ledgers_checked,
    max_cost_abs_residual=maximum_residual,topology_module_calls_reconstructed=module_checks,
    shortlists_checked=shortlist_checks,fallback_decisions_reconstructed=len(expansion_details),
    forced_production_expansions=0,cached_first20_not_resimulated=True,
    full_order_float32_reward_and_eligibility_reconstructed=True,
    all_completed_arms_native_configuration_matches_frozen_GRID05=True,
    reused_assets_unchanged=len(reused),execution_source_snapshot_unchanged=True,
    qualification_native_simulates=0,qualification_physical_steps=0,qualification_wall_s=pre['wall_s'],
    completed_process_wall_s=sum(r['wall_s'] for r in manifest),
    failed_process_wall_s=failed['wall_s'],recorded_process_wall_including_failure_s=sum(r['wall_s'] for r in attempts),
    import_watchdog=watchdog,import_cap_s=design['caps']['import_s'],
    import_limit_effectively_enforced=False,parent_total_wall_s=None,
    parent_total_wall_note='No normal MATRIX_COMPLETE clock record. Child process wall sum is not a substitute for parent orchestration elapsed time.',
    main_persisted_bytes=persisted,audit_wall_s=time.perf_counter()-started,
    normalized_competition_score_certified=False,
    scope='Audit of completed saved vectors/ledger/arithmetic/native feedback only. No full-matrix quality/resource acceptance, independent power flow, QP certification, neural rerun or training.')
save('GRID06_partial_audit.json',audit)
print(json.dumps(audit,indent=2),flush=True)
