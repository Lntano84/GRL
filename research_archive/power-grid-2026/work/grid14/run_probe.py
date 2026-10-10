"""GRID14: one topology intervention, restored controller continuation, no fitting.

Candidates use public one-step observations only. Real continuations are diagnostic
outcomes and never available to the selection rules. Historical artifacts are read-only.
"""
import sys, argparse, gzip, os, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import (ROOT, np, json, hashlib, write_json, make_env, ResidualControl,
                    action_vector, digest, flags, public_cost_ledger,
                    cost_rounding_tolerance, PREFIX)
import cvxpy as cp

OUT = ROOT / 'outputs/grid14'
START = time.perf_counter()
COUNTS = {'physical_steps': 0, 'public_forecasts': 0, 'shadow_qp_calls': 0}
D = json.loads((OUT / 'design.json').read_text())

def memory(optim):
    values = {name: np.array(val.value, copy=True) for name, val in vars(optim).items()
              if isinstance(val, cp.Parameter) and val.value is not None}
    values['flow_computed'] = optim.flow_computed.copy()
    values['_storage_setpoint'] = optim._storage_setpoint.copy()
    return values

def memory_hash(values):
    # CVXPY validates an assigned scalar integer as float64 (15 -> 15.0).
    # Compare numeric state canonically; raw int/float bytes are not memory equivalence.
    return hashlib.sha256(b''.join(name.encode() + str(values[name].shape).encode() +
                                  np.ascontiguousarray(values[name], dtype=np.float64).tobytes()
                                  for name in sorted(values))).hexdigest()

def clone_optim(source, env):
    """New solver, exact public controller memory; never deep-copy the environment."""
    clone = type(source)(env, env.action_space, config=dict(source.config), verbose=False)
    saved = memory(source)
    for name, value in saved.items():
        target = getattr(clone, name)
        if isinstance(target, cp.Parameter):
            target.value = value.copy()
        else:
            setattr(clone, name, value.copy())
    assert memory_hash(memory(clone)) == memory_hash(saved)
    assert all(np.array_equal(memory(clone)[name], value, equal_nan=True)
               for name, value in saved.items())
    return clone

def valid_prediction(future, done, info):
    fl = flags(info)
    return bool(not done and not fl['illegal'] and not fl['ambiguous'] and
                not fl['exceptions'] and np.isfinite(future.rho).all() and
                0 < float(future.rho.max()) < D['final_rho_lt'])

def run_episode(week, rule, baseline=None):
    path = OUT / 'runs' / f'{week}__{rule}'
    path.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    env = make_env(); env.seed(D['environment_seed']); env.set_id(week); obs = env.reset()
    ctl = ResidualControl(env, obs)
    target = None if baseline is None else json.loads((baseline / 'probe.json').read_text())
    references = []
    if baseline is not None:
        with gzip.open(baseline / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f:
            references = [json.loads(line) for line in f]
    context = {'step': 0, 'probe': None, 'calls_after_start': 0, 'selected': None,
               'original_shadow': None, 'prefix_checks': 0, 'actual_gate': False}
    vectors = {}; rows = []; cost = 0.; offered = 0; selected_restore = 0
    delivered_gates = []
    arrays = {name: [] for name in ['action', 'observation', 'rho', 'gen_p', 'load_p',
                                   'actual_dispatch', 'curtailment_mw', 'storage_power']}
    reward = 0.; sim_total = 0; n_qp = 0
    horizon = int(env.max_episode_duration()); assert horizon <= 2017

    def install(module, module_name):
        original = module.get_act
        def get_act(observation, base_action, reward, done=False, **kwargs):
            nonlocal n_qp
            result = original(observation, base_action, reward, done=done, **kwargs)
            step = context['step']
            if target is not None:
                if step != target['action_step']:
                    return result
                assert module_name == target['module']
                assert digest(observation.to_vect()) == target['before_hash']
                assert memory_hash(memory(ctl.base.optim)) == target['optim_memory_before']
                chosen = target['choices'][rule]
                a = module.tested_action[chosen['source_index']].copy()
                assert digest(action_vector(a)) == chosen['topology_hash']
                context['selected'] = chosen
                return a
            if context['probe'] is not None or not (D['search_start_step'] <= step <= D['search_end_step']):
                return result
            context['calls_after_start'] += 1
            if context['calls_after_start'] > D['max_probes_per_week'] or result is None:
                return result
            scores = np.asarray(module.resulting_rewards)
            admissible = np.flatnonzero(scores > module.null_action_reward)
            if len(admissible) < 2:
                return result
            order = sorted(admissible.tolist(), key=lambda i: (-float(scores[i]), i))[:D['candidate_beam']]
            assert order[0] == int(np.argmax(scores))
            before_mem = memory_hash(memory(ctl.base.optim))
            before_obs = digest(observation.to_vect())
            candidates = []
            local_vectors = {'before_observation': observation.to_vect().copy(),
                             'incoming_base_action': action_vector(base_action)}
            for i in order:
                topology = module.tested_action[i].copy()
                combined = base_action.copy() + topology
                topo_future, _, topo_done, topo_info = observation.simulate(combined, time_step=1)
                delivered = combined.copy(); statuses = []
                needs_qp = bool(topo_future.rho.max() > ctl.base.rho_safe or topo_info.get('exception', []))
                if needs_qp:
                    shadow = clone_optim(ctl.base.optim, env)
                    solver = shadow._solve_problem
                    def solve(prob, solver_type=None):
                        answer = solver(prob, solver_type)
                        statuses.append(str(prob.status))
                        return answer
                    shadow._solve_problem = solve
                    delivered = shadow.get_act(observation, combined.copy(), reward)
                    n_qp += 1; COUNTS['shadow_qp_calls'] += 1
                future, _, future_done, info = observation.simulate(delivered, time_step=1)
                qualified = valid_prediction(future, future_done, info)
                ledger = public_cost_ledger(future, env, float(observation.curtailment_mw.sum(dtype=np.float64))) if qualified else None
                row = {'source_index': i, 'source_score': float(scores[i]),
                       'topology_hash': digest(action_vector(topology)),
                       'delivered_hash': digest(action_vector(delivered)),
                       'topology_forecast_rho': float(topo_future.rho.max()),
                       'final_forecast_rho': float(future.rho.max()),
                       'final_forecast_cost': ledger['recomputed_raw_cost'] if ledger else None,
                       'qualified': qualified, 'qp_used': needs_qp, 'solve_status': statuses,
                       'done': bool(future_done), **flags(info)}
                candidates.append(row)
                local_vectors[f'c{i}_topology'] = action_vector(topology)
                local_vectors[f'c{i}_delivered'] = action_vector(delivered)
                local_vectors[f'c{i}_forecast'] = future.to_vect().copy()
                assert memory_hash(memory(ctl.base.optim)) == before_mem
                assert digest(observation.to_vect()) == before_obs == digest(env.get_obs().to_vect())
            qualified = [c for c in candidates if c['qualified']]
            original_row = candidates[0]
            # First event with a qualified original and at least two different delivered alternatives.
            if not original_row['qualified'] or len({c['delivered_hash'] for c in qualified}) < 2:
                write_json(path / f'rejected_probe_{step}.json', {'action_step': step, 'module': module_name,
                           'reason': 'original_or_alternative_forecast_screen_failed', 'candidates': candidates})
                return result
            choices = {'AUTHOR': original_row,
                       'FINAL_COST': min(qualified, key=lambda c: (c['final_forecast_cost'], c['source_index'])),
                       'FINAL_RHO': min(qualified, key=lambda c: (c['final_forecast_rho'], c['source_index']))}
            probe = {'scenario': week, 'action_step': step, 'module': module_name,
                     'before_hash': before_obs, 'optim_memory_before': before_mem,
                     'source_admissible': len(admissible), 'source_tested': len(scores),
                     'candidate_beam': len(candidates), 'candidates': candidates, 'choices': choices,
                     'rule_inputs': 'Public observation, current controller memory, original source shortlist and one-step final-action forecast only.'}
            context['probe'] = probe; context['original_shadow'] = original_row
            vectors.update(local_vectors)
            write_json(path / 'probe.json', probe)
            np.savez_compressed(path / 'probe_vectors.npz', **vectors)
            print(json.dumps({'probe': week, 'step': step, 'module': module_name,
                              'candidates': len(candidates), 'unique_rule_actions': len({c['delivered_hash'] for c in choices.values()})}), flush=True)
            return result
        module.get_act = get_act
    install(ctl.base.topo_12_unsafe, 'topo_12_unsafe')
    install(ctl.base.topo_n1_unsafe, 'topo_n1_unsafe')
    write_json(path / 'metadata.json', {'scenario': week, 'rule': rule, 'horizon': horizon,
               'cost_per_MW': env.gen_cost_per_MW.tolist(), 'dt_hours': float(env.delta_time_seconds)/3600.,
               'continuation': 'Unchanged original NN20/N1/unsafe QP plus ALWAYS_RESTORE cadence6.',
               'target_action_step': target['action_step'] if target else None})
    def delivered_gate(act, reference, vector_path, step):
        with np.load(vector_path) as saved:
            expected = env.action_space(); expected.from_vect(saved[f"c{reference['source_index']}_delivered"])
        differences = {}
        for name in type(act).attr_list_vect:
            a = np.asarray(act._get_array_from_attr_name(name), dtype=float)
            b = np.asarray(expected._get_array_from_attr_name(name), dtype=float)
            if name in ('_redispatch', '_storage_power', '_curtail'):
                gap = float(np.nanmax(np.abs(a-b))) if a.size else 0.
                assert np.allclose(a, b, rtol=0., atol=D['continuous_gate_atol'], equal_nan=True), (name, gap)
                differences[name] = gap
            else:
                assert np.array_equal(a, b, equal_nan=True), ('Discrete action mismatch', name)
        future, _, done, info = obs.simulate(act, time_step=1)
        assert valid_prediction(future, done, info)
        actual_cost = public_cost_ledger(future, env, oldcurt)['recomputed_raw_cost']
        cost_tol = max(.002, float(cost_rounding_tolerance(future, env)))
        rho_gap = abs(float(future.rho.max()) - reference['final_forecast_rho'])
        cost_gap = abs(actual_cost - reference['final_forecast_cost'])
        assert rho_gap <= D['forecast_rho_gate_atol'] and cost_gap <= cost_tol, (rho_gap, cost_gap, cost_tol)
        gate = {'step': step, 'rule': rule, 'discrete_exact': True, 'continuous_differences': differences,
                'actual_public_rho': float(future.rho.max()), 'actual_public_cost': actual_cost,
                'rho_gap': rho_gap, 'cost_gap': cost_gap, 'cost_tolerance': cost_tol,
                'actual_action_hash': digest(action_vector(act)), 'shadow_action_hash': reference['delivered_hash']}
        delivered_gates.append(gate); write_json(path / 'delivered_gates.json', delivered_gates)
        np.savez_compressed(path / f'delivered_gate_{step}.npz', action=action_vector(act), forecast=future.to_vect())
    try:
        with gzip.open(path / 'steps.jsonl.gz', 'wt', encoding='utf-8') as log:
            for step in range(1, horizon + 1):
                assert time.perf_counter() - START < D['caps']['phase_wall_s']
                assert time.perf_counter() - started < D['caps']['per_episode_wall_s']
                context['step'] = step
                before = digest(obs.to_vect()); oldcurt = float(obs.curtailment_mw.sum(dtype=np.float64))
                counter = int(env.nb_highres_called)
                base, restore, proposal = ctl.propose(obs, reward, cadence=D['cadence'])
                act = restore if proposal['offered'] else base
                offered += int(proposal['offered']); selected_restore += int(proposal['offered'])
                ah = digest(action_vector(act))
                if context['probe'] is not None and step == context['probe']['action_step']:
                    delivered_gate(act, context['original_shadow'], path / 'probe_vectors.npz', step)
                    context['actual_gate'] = True
                if target is not None and step == target['action_step']:
                    assert context['selected'] is not None
                    delivered_gate(act, context['selected'], baseline / 'probe_vectors.npz', step)
                    context['actual_gate'] = True
                obs, reward, done, info = env.step(act)
                COUNTS['physical_steps'] += 1
                native = int(env.nb_highres_called) - counter
                COUNTS['public_forecasts'] += native; sim_total += native
                assert COUNTS['physical_steps'] <= D['caps']['physical_steps']
                assert COUNTS['public_forecasts'] <= D['caps']['public_forecasts']
                raw = float(info['rewards'][f'{PREFIX}_grid_operational_cost']); cost += raw
                fl = flags(info); eligible = bool((obs.gen_p > 0).any()) and not fl['exceptions']
                ledger = public_cost_ledger(obs, env, oldcurt) if eligible else None
                tolerance = float(cost_rounding_tolerance(obs, env)) if eligible else None
                if ledger: assert abs(ledger['recomputed_raw_cost'] - raw) <= tolerance
                complete = bool(obs.current_step >= horizon or env.chronics_handler.done())
                row = {'step': step, 'before_hash': before, 'action_hash': ah,
                       'after_hash': digest(obs.to_vect()), 'raw_cost': raw, 'ledger': ledger,
                       'ledger_tolerance': tolerance, 'restore': bool(proposal['offered']),
                       'current_step': int(obs.current_step), 'rho': float(obs.rho.max()),
                       'done': bool(done), 'complete': complete, 'simulations': native, **fl}
                if target is not None and step <= target['action_step']:
                    ref = references[step-1]; assert before == ref['before_hash']
                    if step < target['action_step']:
                        assert ah == ref['action_hash'] and row['after_hash'] == ref['after_hash']
                        assert raw == ref['raw_cost']; context['prefix_checks'] += 1
                rows.append(row); log.write(json.dumps(row, allow_nan=False) + '\n'); log.flush()
                for name in arrays:
                    value = action_vector(act) if name == 'action' else obs.to_vect() if name == 'observation' else getattr(obs, name)
                    arrays[name].append(np.array(value, copy=True))
                if step % 256 == 0 or done:
                    write_json(path / 'progress.json', {'step': step, 'cost': cost, 'counts': COUNTS})
                    print(json.dumps({'week': week, 'rule': rule, 'step': step, 'cost': cost, 'wall_s': time.perf_counter()-started}), flush=True)
                if done: break
        np.savez_compressed(path / 'vectors.npz', **{k: np.stack(v) for k, v in arrays.items()})
        summary = {'scenario': week, 'rule': rule, 'steps': step, 'complete': complete,
                   'cost': cost, 'offered': offered, 'selected_restore': selected_restore,
                   'simulations': sim_total, 'shadow_qp_calls': n_qp,
                   'probe_found': context['probe'] is not None if target is None else True,
                   'actual_gate': context['actual_gate'], 'prefix_exact_steps': context['prefix_checks'],
                   'wall_s': time.perf_counter() - started,
                   'illegal': sum(r['illegal'] for r in rows), 'ambiguous': sum(r['ambiguous'] for r in rows),
                   'exception_steps': sum(bool(r['exceptions']) for r in rows)}
        write_json(path / 'summary.json', summary)
        return summary
    except Exception:
        # Account forecasts from the failed, not-yet-committed decision too.
        missing_forecasts = int(env.nb_highres_called) - counter
        COUNTS['public_forecasts'] += missing_forecasts
        np.savez_compressed(path / 'partial_vectors.npz', **{k: np.stack(v) for k, v in arrays.items() if v})
        write_json(path / 'failure.json', {'traceback': traceback.format_exc(), 'counts': COUNTS,
                   'uncommitted_decision_forecasts': missing_forecasts,
                   'completed_steps': len(rows), 'wall_s': time.perf_counter() - started})
        raise
    finally:
        write_json(OUT / 'resources.json', {**COUNTS, 'wall_s': time.perf_counter() - START,
                   'training_runs': 0, 'phase': 'RUNNING'})
        env.close()

def main():
    summaries = []
    assert not (OUT / 'finished.json').exists()
    for week in D['training_weeks']:
        baseline = OUT / 'runs' / f'{week}__AUTHOR'
        summaries.append(run_episode(week, 'AUTHOR'))
        write_json(OUT / 'manifest.json', summaries)
        if not summaries[-1]['probe_found']:
            continue
        assert summaries[-1]['actual_gate']
        probe = json.loads((baseline / 'probe.json').read_text())
        seen = {probe['choices']['AUTHOR']['delivered_hash']}
        for rule in ['FINAL_COST', 'FINAL_RHO']:
            key = probe['choices'][rule]['delivered_hash']
            if key in seen:
                write_json(baseline / f'{rule}_identical.json', {'same_delivered_action_as_previous_rule': True,
                           'delivered_hash': key, 'reason': 'One-time intervention identical; no additional physical rollout needed.'})
                continue
            seen.add(key)
            summaries.append(run_episode(week, rule, baseline))
            write_json(OUT / 'manifest.json', summaries)
    total_bytes = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert total_bytes <= D['caps']['output_bytes']
    write_json(OUT / 'resources.json', {**COUNTS, 'wall_s': time.perf_counter() - START,
               'training_runs': 0, 'output_bytes': total_bytes, 'phase': 'COMPLETE'})
    write_json(OUT / 'finished.json', {'passed_engineering': True, 'summaries': summaries,
               'resources': COUNTS, 'wall_s': time.perf_counter() - START})

if __name__ == '__main__':
    main()
