"""Fixed, legal adaptive enumeration; reuse the author's original greedy evaluator."""
import time
import numpy as np
from trace_agent import action_vector, digest


def attach_rank_trace(module, trace, policy):
    original=module.get_top_k
    def ranked(gym_obs, top_k):
        started=time.perf_counter()
        ids=original(gym_obs,top_k)
        module.grid05_last_rank_ids=[int(i) for i in ids]
        trace.events.append({'kind':'shortlist','step':trace.step,'module':'topo_12_unsafe',
            'policy':policy,'ids':module.grid05_last_rank_ids.copy(),'top_k':int(top_k),
            'wall_s':time.perf_counter()-started})
        return ids
    module.get_top_k=ranked


def install_rule(module, threshold, holder, force_expand_for_test=False):
    """Install before Telemetry so both stages are one original module call.

    No thresholds or source greedy eligibility are changed. On expansion, cached
    float32 rewards are merged in the full neural order, preserving its tie rule.
    The force switch is only used by the explicitly labelled zero-step preflight.
    """
    original_get_act=module.get_act
    original_candidates=module._get_tested_action
    assert module.top_k==20

    def search(observation, base_action, reward, done=False, **kwargs):
        trace=holder['trace']
        first=original_get_act(observation,base_action,reward,done=done,**kwargs)
        first_ids=module.grid05_last_rank_ids.copy()
        first_actions=list(module.tested_action)
        first_rewards=module.resulting_rewards.copy()
        assert len(first_ids)==len(first_actions)==len(first_rewards)==20
        assert len(set(first_ids))==20 and np.isfinite(first_rewards).all()
        selected_rho=None
        if first is not None:
            key=(digest(action_vector(first+base_action)),1)
            assert key in trace.forecasts,'The original selected candidate has no recorded forecast'
            selected_rho=float(np.max(trace.forecasts[key][0]))
            assert np.isfinite(selected_rho)
        trigger=first is None or selected_rho>=threshold
        expand=bool(trigger or force_expand_for_test)
        event={'kind':'fallback','step':trace.step,'module':'topo_12_unsafe','threshold':float(threshold),
            'first_ids':first_ids,'first_rewards':first_rewards.tolist(),
            'first_return_hash':digest(action_vector(first)) if first is not None else None,
            'first_selected_forecast_rho':selected_rho,'trigger':bool(trigger),
            'forced_test':bool(force_expand_for_test),'expanded':expand,'additional_ids':[],
            'returned_pool_row':None,'returned_action_hash':None}
        if not expand:
            event['returned_pool_row']=first_ids[int(np.argmax(first_rewards))]
            event['returned_action_hash']=digest(action_vector(first))
            trace.events.append(event)
            return first

        gym_obs=module.gym_env.observation_space.to_gym(observation)
        full_ids=[int(i) for i in module.get_top_k(gym_obs,top_k=352)]
        assert sorted(full_ids)==list(range(352))
        seen=set(first_ids)
        additional=[i for i in full_ids if i not in seen]
        assert len(additional)==332
        additional_actions=[module.gym_env.action_space.from_gym(i) for i in additional]
        try:
            module._get_tested_action=lambda obs:additional_actions
            original_get_act(observation,base_action,reward,done=done,**kwargs)
            later_rewards=module.resulting_rewards.copy()
        finally:
            module._get_tested_action=original_candidates
        assert len(later_rewards)==332 and np.isfinite(later_rewards).all()
        scores=dict(zip(first_ids,first_rewards))
        scores.update(zip(additional,later_rewards))
        actions=dict(zip(first_ids,first_actions))
        actions.update(zip(additional,additional_actions))
        module.tested_action=[actions[i] for i in full_ids]
        module.resulting_rewards=np.asarray([scores[i] for i in full_ids],dtype=first_rewards.dtype)
        result=None
        if np.max(module.resulting_rewards)>module.null_action_reward:
            j=int(np.argmax(module.resulting_rewards))
            result=module.tested_action[j]
            event['returned_pool_row']=full_ids[j]
            event['returned_action_hash']=digest(action_vector(result))
        event['additional_ids']=additional
        trace.events.append(event)
        return result
    module.get_act=search

