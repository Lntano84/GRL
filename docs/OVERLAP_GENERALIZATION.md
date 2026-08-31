# Explicit Overlap Experiment

This experiment compares the existing `Direct-MLP` marginal-gain predictor with
`Direct-MLP+Overlap`. The second model is still a direct MLP; it is not a GNN.

## Features

For `(S, v)`, features are computed from the undirected topology projection:

- `nearest_seed_distance_norm`: shortest distance from `v` to the nearest seed,
  clipped and divided by `distance_cap`.
- `unreachable_flag`: 1 when no seed is reachable in the topology projection.
- `empty_seed_set_flag`: 1 only when `S` is empty.
- `neighborhood_overlap_1hop` and `neighborhood_overlap_2hop`: fraction of the
  candidate's closed k-hop neighborhood covered by the union of seed closed
  k-hop neighborhoods.
- `neighborhood_jaccard_2hop`: Jaccard overlap of the two closed 2-hop sets.
- `candidate_2hop_seed_fraction`: fraction of seeds within two hops of `v`.
- `seed_union_2hop_graph_coverage`: fraction of graph nodes in the union of
  seed 2-hop neighborhoods. This and neighborhood overlap are local coverage
  proxies, not community-detection output.
- `direct_seed_neighbor_fraction`: fraction of the candidate's closed 1-hop
  neighborhood that is itself a seed.

No Louvain or other community detector is used because the repository has no
validated community implementation for this protocol.

## Run

```bash
PYTHONPATH=src python scripts/run_overlap_experiment.py --config configs/smoke/overlap_nethept.yaml
PYTHONPATH=src python scripts/run_overlap_experiment.py --config configs/overlap_nethept.yaml
```

The result directory contains one subdirectory per seed, per-seed metrics and
checkpoints, `aggregate_metrics.csv`, and `results.json`. The existing
`marginal_dataset.json` is reused unchanged. Its SHA-256 and split counts are
recorded in `results.json`.

Sequential evaluation reranks a Degree Top-20 pool rebuilt at every step inside
the restricted top-200-by-degree universe. It reports pool RankingLoss, Top-1,
and final spread separately from the held-out dataset metrics.

## Cross-graph protocol

The current results are NetHEPT-only. A minimal cross-graph test should train
on NetHEPT and test on a teacher-specified second graph after regenerating its
embedding and marginal dataset with identical IC probability, feature
definitions, normalization, and a graph-local disjoint split. Test-graph
marginal labels must not enter training or tuning; test-graph test metrics must
not be used for model selection; node IDs and label-bearing caches must not be
reused across graphs. No cross-graph result is claimed until the graph file,
preprocessing parameters, embedding, and labels are available.
