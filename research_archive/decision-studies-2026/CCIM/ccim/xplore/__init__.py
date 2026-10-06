"""Exploratory influence maximization: discover a network under a survey budget, then seed it well.

The problem
-----------
Instead of "the network is known, choose the best seeds", the question is "the network is mostly unknown
and surveys cost money -- whom should we ask about, so that the seeds we finally choose still spread
well?".  A survey of ``u`` reveals the true neighbours of ``u``; the survey budget is small; the payoff is
measured on the *complete* network, which the policy never sees.

Modules
-------
``icm``   independent-cascade estimation and the greedy seed selector (faithful to the published code)
``env``   the discovery environment and the published exploration policies

Reference implementation: ``kage08/graph_sample_rl`` (official code for Kamarthi, Vijayan, Wilder,
Ravindran and Tambe, "Influence maximization in unknown social networks: Learning Policies for Effective
Graph Sampling", AAMAS 2020).  Defaults follow ``train.py``: 5 free seeds, 5 queries, IC ``p = 0.1``,
greedy seed budget 10, 100 live-edge samples.
"""

from .icm import INFL_BUDGET, PROP_PROBAB, SAMPLES, LiveEdgeObjective, influence_on
from .env import (EXTRA_SEEDS, QUERY_BUDGET, DiscoveryEnv, EpisodeResult, StepRecord,
                  change_surveys, policy_degree_max, policy_degree_min, policy_random,
                  run_change_episode, run_episode)

__all__ = [
    "INFL_BUDGET", "PROP_PROBAB", "SAMPLES", "LiveEdgeObjective", "influence_on",
    "EXTRA_SEEDS", "QUERY_BUDGET", "DiscoveryEnv", "EpisodeResult", "StepRecord",
    "change_surveys", "policy_degree_max", "policy_degree_min", "policy_random",
    "run_change_episode", "run_episode",
]
