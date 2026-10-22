"""Numeric prefix-only projection. No metadata, raw calendar vectors or targets."""
import math
import struct
import numpy as np

NAMES = [
 'log_pool_size','queried_fraction','log_visible_count','log_completion_queries',
 'completion_fraction_remaining','station_pool_fraction','station_seen_fraction',
 'station_admissible_fraction','station_observed_fraction','admissible_fraction',
 'terminal_fraction','illegal_fraction','ambiguous_fraction','exception_fraction',
 'incumbent_reward','incumbent_rho','reward_gap','missing_second_admissible',
 'admissible_reward_mean','admissible_reward_std','admissible_rho_mean',
 'admissible_rho_std','admissible_rho_min','rho_limit','admissible_reward_range',
 'initial_prefix_has_admissible','gain_since_initial_prefix','incumbent_changed',
 'initial_incumbent_rho','last_batch_has_admissible','last_batch_admissible_fraction',
 'last_batch_reward_gain','previous_prefix_has_admissible','improvement_fraction',
 'station_reward_std','station_rho_mean','incumbent_relative_position']


def reward(row):
    return struct.unpack('f', struct.pack('f', row['reward']))[0]


def best(rows):
    return max((r for r in rows if r['strict_admissible']),
               key=lambda r: (reward(r), -r['pool_id']), default=None)


def project(x):
    visible, order = x['visible'], x['public_order']
    n, total = len(visible), len(order)
    assert 0 < n < total and n == x['visible_count']
    assert [r['pool_id'] for r in visible] == order[:n]
    incumbent = best(visible); assert incumbent is not None
    assert incumbent['pool_id'] == x['incumbent_id']
    subs = x['public_subids']; station = subs[incumbent['pool_id']]
    targets = {int(rep) for original, rep in x['public_aliases'].items() if subs[int(original)] == station}
    pending = [j for j in order[n:] if j in targets]
    assert pending == x['remaining_same_station_ids'] and len(pending) == x['known_completion_queries']
    station_seen = [r for r in visible if r['pool_id'] in targets]
    station_valid = [r for r in station_seen if r['strict_admissible']]
    valid = [r for r in visible if r['strict_admissible']]
    rewards = np.array([reward(r) for r in valid]); rhos = np.array([r['rho'] for r in valid])
    ranked = sorted(valid, key=lambda r:(reward(r),-r['pool_id']), reverse=True)
    initial, previous = best(visible[:128]), best(visible[:-64])
    recent = visible[-64:]; recent_valid = [r for r in recent if r['strict_admissible']]
    running = None; improvements = 0
    for row in visible:
        if row['strict_admissible'] and (running is None or reward(row) > running):
            improvements += 1; running = reward(row)
    v = [
        math.log1p(total), n/total, math.log1p(n), math.log1p(len(pending)),
        len(pending)/(total-n), len(targets)/total, len(station_seen)/len(targets),
        len(station_valid)/len(station_seen), len(station_seen)/n, len(valid)/n,
        sum(bool(r['done']) for r in visible)/n, sum(bool(r['illegal']) for r in visible)/n,
        sum(bool(r['ambiguous']) for r in visible)/n, sum(bool(r['exceptions']) for r in visible)/n,
        reward(incumbent), incumbent['rho'],
        reward(ranked[0])-reward(ranked[1]) if len(ranked)>1 else 0., float(len(ranked)<2),
        rewards.mean(), rewards.std(), rhos.mean(), rhos.std(), rhos.min(), x['rho_limit'], np.ptp(rewards),
        float(initial is not None), reward(incumbent)-reward(initial) if initial else 0.,
        float(initial is not None and initial['pool_id']!=incumbent['pool_id']), initial['rho'] if initial else 0.,
        float(bool(recent_valid)), len(recent_valid)/len(recent),
        reward(incumbent)-reward(previous) if previous else 0., float(previous is not None), improvements/n,
        np.std([reward(r) for r in station_valid]), np.mean([r['rho'] for r in station_valid]),
        order[:n].index(incumbent['pool_id'])/n]
    assert len(v) == len(NAMES) and np.isfinite(v).all()
    return np.asarray(v, dtype=np.float64)
