# TAIM minimum probe: no evidence for an expensive-foresight gap here

This is a **small exploratory implementation check**, not a reproduction of
Tong et al.'s paper or a GRL result. The frozen design and caveats are in
`PLAN.md`; the complete per-state/trial output is in `result.json`.

## Scope and validity checks

Two real small undirected graphs were converted to independent directed IC
arcs: football (115 nodes, 613 undirected edges) and polbooks (105, 441).
For each graph, p=0.1 and p=0.2; one top-20-degree seed was invested before
the observed round. Each resulting state had two rounds left, analyzed with
one and two remaining seeds. Four development and eight held-out base states
were generated per graph/probability stratum; no state was dropped for being
uninteresting. The 32 held-out base states yield 64 budget-specific decisions.
Development and held-out states have zero full-state overlap; each stratum's
held-out eight states are unique. There were 512 independent paired
confirmation worlds per decision, 32,768 paired trial rows in total. Decision
streams and confirmation streams have different keyed seeds.

The implementation passed six tests, including an exact four-world regression
on the attachment's hand example: with one remaining seed the correct
choice is to wait (90 rather than 85 additional activated nodes); with two
it is to seed both centers now (120 rather than 110). These exclude the most
obvious error of making waiting impossible in the simulator.

## Held-out comparison

The development-selected cheap arm was **invest all remaining seeds now**.
FF at theta 0.2 and 0.4 tied it on development. Mean final activated nodes
across the held-out state/budget combinations:

| Decision policy | Mean | Mean decision time, including shared current greedy ordering |
|---|---:|---:|
| Best cheap: all now | 20.5638 | 12.12 ms |
| SOF, 8 simulations per allocation | 20.5342 | 13.46 ms |
| SOF, 64 simulations per allocation | 20.5638 | 21.87 ms |

High-SOF and cheap selected **exactly the same current seed count and nodes
in all 64/64 decisions**. The observed paired difference is therefore
identically zero in this finite sample, not a statistical equivalence claim
for a population of states. Low-SOF differed on only 2/64 decisions; its
mean difference from cheap is -0.0295, with a 95% state-cluster bootstrap
interval [-0.0730, 0.0000]. High-minus-low is +0.0295 [0.0000, 0.0730].
The high-minus-cheap stratum means are all exactly zero.

The natural states are quite early and small: median active set 3, median
frontier 2, maximum frontier 6, and 3/32 held-out base states have an empty
frontier. That may explain why the value of waiting does not overcome the
lost propagation round here, but the experiment does not establish this as
a causal explanation. It does demonstrate that the hand-example mechanism
does not automatically create a useful gap on these two real graphs.

## Decision and limitations

The proposed gate is **closed for this setting**. Spending 64 rather than 8
SOF samples, or using SOF rather than the cheap selected schedule, has not
bought practical quality here; 64-sample SOF is also under 22 ms per decision
on this implementation. Do not start GRL training on this result.

This does **not** refute TAIM or its use of learning. The probe used only
two small graphs, two fixed probabilities, one early-state generator,
budgets 1/2, and two remaining rounds. Its common node selector uses 32
Monte Carlo worlds; published methods may use stronger selectors or larger
horizons. FF and SOF follow the local equations/decision structure in
[Tong et al.](https://arxiv.org/pdf/2001.01742), but they have not been
checked against the authors' implementation. The hand example is a mechanism
regression, not a real-data positive result.

One audit qualification: the first full run exposed that the intended
budget-specific frontier thresholds had been encoded as one shared threshold.
After seeing that run, the code was corrected to the 25 threshold pairs
specified in `PLAN.md` and the same states were rerun. The selected cheap arm
and every held-out comparison stayed unchanged. Because the held-out states
had already been viewed, this is a **development probe**, not an untouched
confirmatory paper result. No subsequent training or post-hoc parameter
search was performed.
