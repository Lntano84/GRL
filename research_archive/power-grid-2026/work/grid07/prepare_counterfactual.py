"""Freeze one diagnostic intervention; do not rescue or rerun GRID06's matrix."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/grid07'
OUT.mkdir(parents=True, exist_ok=True)

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x): (OUT/name).write_text(json.dumps(x,indent=2),encoding='utf-8')

assert not (OUT/'counterfactual_design.json').exists(), 'Do not refreeze a launched design'
old = (ROOT/'work/grid06/run_one.py').read_text(encoding='utf-8')
code = old
def replace(a,b):
    global code
    assert code.count(a)==1,(a,code.count(a))
    code=code.replace(a,b)

replace('GRID06 worker, derived from frozen GRID05; sealed validation plus concurrent NN352.',
        'GRID07 single intervention: expand at archived first divergence only, NN20 otherwise.')
replace("choices=['NN20','FALLBACK90','NN352']", "choices=['SINGLE_EXPAND_221']")
replace("ROOT/'outputs/grid06/runs'", "ROOT/'outputs/grid07/runs'")
replace("ROOT/'outputs/grid06/design_freeze.json'", "ROOT/'outputs/grid07/counterfactual_design.json'")
replace("assert args.scenario in design['selected_weeks']", """assert args.scenario in design['selected_weeks']
reference_root=ROOT/'outputs/grid06/runs'
reference_rows={p:[json.loads(s) for s in (reference_root/f'{args.scenario}__{p}'/'steps.jsonl').read_text(encoding='utf-8').splitlines() if s] for p in ['NN20','FALLBACK90']}
prefix_checks=0
total_simulations=0
assert design['intervention_step']==221""")
replace("    if args.policy=='NN352':\n        agent.topo_12_unsafe.top_k=352\n", "")
replace("""    if args.policy=='FALLBACK90':
        install_rule(agent.topo_12_unsafe,agent.rho_safe,holder)
""", """    original_get_act=agent.topo_12_unsafe.get_act
    install_rule(agent.topo_12_unsafe,agent.rho_safe,holder)
    expanded_get_act=agent.topo_12_unsafe.get_act
    def only_first_divergence(*a,**kw):
        return (expanded_get_act if trace.step==design['intervention_step'] else original_get_act)(*a,**kw)
    agent.topo_12_unsafe.get_act=only_first_divergence
""")
replace("'top_k':352 if args.policy=='NN352' else 20", "'top_k':20")
replace("'enabled':args.policy=='FALLBACK90'", "'enabled':True,'only_at_step':221")
replace("""            assert native_calls==traced_calls,(native_calls,traced_calls)
""", """            assert native_calls==traced_calls,(native_calls,traced_calls)
            total_simulations+=native_calls
            assert total_simulations<=design['caps']['native_simulations']
            assert physical_calls<=design['caps']['physical_steps']
""")
replace("""            event_file.flush()
""", """            if step<=221:
                reference_policy='NN20' if step<221 else 'FALLBACK90'
                expected=reference_rows[reference_policy][step-1]
                assert record['before_observation_hash']==expected['before_observation_hash'],('engineering_before_state',step)
                assert record['action_hash']==expected['action_hash'],('engineering_action',step)
                assert record['observation_hash']==expected['observation_hash'],('engineering_after_state',step)
                assert abs(raw-expected['raw_operational_cost'])<=max(expected['cost_rounding_tolerance'] or 0,.002),('engineering_cost',step)
                prefix_checks+=1
                if step==221:
                    fb=[e for e in trace.events if e['kind']=='fallback']
                    assert len(fb)==1 and fb[0]['expanded'] and not fb[0]['forced_test']
                    assert fb[0]['returned_pool_row']==255
                    save('engineering_gate.json',{'passed':True,'prefix_steps_exactly_reproduced':220,
                        'intervention_action_and_after_observation_match_archived_fallback':True,
                        'gate_physical_calls':physical_calls,'gate_simulations':total_simulations,
                        'scope':'Exact saved hashes for NN20 steps 1..220 and archived FALLBACK90 step221; does not certify a counterfactual outcome.'})
            event_file.flush()
""")
replace("step%96==0", "step%48==0")
replace("'saved_steps':completed,'wall_after_import_s'", "'saved_steps':completed,'total_native_simulations':total_simulations,'prefix_checks':prefix_checks,'wall_after_import_s'")
worker = ROOT/'work/grid07/run_intervention.py'
worker.write_text(code,encoding='utf-8')
(OUT/'worker_derivation.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),code.splitlines(True),fromfile='frozen_GRID06_worker',tofile='GRID07_one_intervention')),encoding='utf-8')
design=dict(stage='GRID07_COUNTEREXAMPLE_DIAGNOSIS',environment_seed=0,
    selected_weeks=['2035-01-08_5'],data_root='work/grid03/formal_env_initialized',
    intervention_step=221,policy='SINGLE_EXPAND_221',
    target='Isolate the archived first topology expansion from later fallback decisions; continuation is unchanged NN20, including all original controller memory and continuous optimization.',
    engineering_gate='Before interpreting any outcome, steps1..220 must exactly match archived NN20 actions/states, and step221 must exactly match archived FALLBACK90 action/state. Native cost must match within the previously frozen rounding tolerance.',
    caps=dict(import_s=120,per_run_after_import_s=480,total_runner_s=650,
        physical_steps=2017,native_simulations=60000,output_bytes=268435456),
    interpretation='One deterministic intervention on an already exposed synthetic week. Post-hoc causal diagnosis of this controller/context, not a deployable rule, generalization test, optimality claim, or learning evidence.',
    restrictions=['One attempt, stop on failed gate/error/cap; no retries.',
        'No other scenarios, new seeds, models, thresholds, training, installs or changes to archived GRID06.',
        'Both branches complete: compare raw cost; otherwise completion/end first, no misleading cumulative-cost ranking.'])
save('counterfactual_design.json',design)
save('counterfactual_design.sha256',{'sha256':sha(OUT/'counterfactual_design.json')})
frozen={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'work/grid07/prepare_counterfactual.py',worker,ROOT/'work/grid07/run_counterfactual.py']}
save('counterfactual_execution_frozen.json',frozen)
save('counterfactual_reused_assets_frozen.json',json.loads((ROOT/'outputs/grid06/reused_assets_frozen.json').read_text(encoding='utf-8')))
print('ONE_INTERVENTION_FROZEN',flush=True)
