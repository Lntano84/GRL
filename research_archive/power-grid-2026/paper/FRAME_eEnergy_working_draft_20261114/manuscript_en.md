# Allocating Simulation Queries in Online Power-Grid Topology Search

## Abstract

Online topology control uses predictive simulation to compare switching actions before execution. A global candidate order can place configurations of the same substation far apart, so a search can identify a promising substation and stop before checking its stronger configurations. FRAME combines feedback-driven global expansion with a completion pass at the substation of the current incumbent. We evaluate the allocation of simulator queries in closed-loop Grid2Op runs and compare a guarded, mixed-order configuration with the original heuristic, a matched learned-policy ablation, and cost-aware Bayesian stopping. On four 2039 development horizons, the guarded mixed-order configuration used 20.95% fewer candidate queries than the original heuristic. Operating cost decreased on two of the three jointly completed weeks and held steady on the third. A matched ablation produced the same trajectories and query counts with and without the residual correction. An independent cost-aware baseline saved a further 2.18% of candidate queries across its four matched horizons and incurred 0.0861% higher operating cost in one of the three completed weeks. Per-week controller timings quantify how search workload translates into wall-clock cost.

## 1 Introduction

Topology controllers change the busbar connections of grid components to manage line loading. A hybrid controller can combine a topology proposal with redispatch, storage, and curtailment, and evaluate the joint action in a simulator before dispatch. An online controller has to choose which candidate simulations merit computation at each decision. The action chosen at one step can change the states and search calls encountered later in the week. [1, 2]

A global candidate order helps the search find promising configurations early. The order interleaves actions at different substations, and several batches without improvement can leave effective configurations of the incumbent substation untested. Once a promising substation has been found, a controller can use the same structural information that defines the topology library to decide where to direct additional queries.

FRAME separates global search depth from the location of local verification. Its prefix search uses observed feedback to decide when to end global expansion. Its completion pass locks the incumbent's substation and evaluates eligible representatives from that group. Action canonicalization retains the aliases needed to recover group membership. We examine the local completion mechanism with a deletion ablation, then compare ordering and stopping variants on closed-loop trajectories. The evaluation reports candidate queries, total forecasts, controller time, completion, and operating cost as distinct outcomes.

## 2 Related work

Grid2Op provides a sequential simulation benchmark for network operations. The LJN controller combines continuous control with discrete topology proposals. Work on learned topology action rankings uses graph features and supervised labels to prioritize configurations for predictive validation. FRAME addresses the allocation of candidate evaluations after a priority order is available. [1–3]

Cost-aware Bayesian optimization treats evaluations as costly information and stops when further expected improvement cannot justify their cost. Our independent comparator adapts the LogEIPC rule to the same station-specific pool accessed by the local control policy. The comparison tests whether a method designed around immediate search improvement retains its savings after the decisions are applied in closed loop. [4]

## 3 Method

### Candidate evaluation

At control step \(t\), the simulator receives the current observation, a base action, and a candidate topology change. Available actions are filtered and combined with the base action. Equal joint action vectors share a representative, and the alias map retains their original substation memberships. A predictive call returns reward, maximum predicted loading \(\rho\), and simulator flags. A valid candidate satisfies the implementation's strict loading bound \(0<\rho<\min(\rho_t,\theta_t)\), where \(\theta_t\) is the threshold supplied by the controller. The flags must indicate a nonterminal, legal, unambiguous transition without exceptions. The selection key is the pair of single-precision reward and negative representative ID. This key specifies both improvement and tie-breaking.

### Global expansion and local completion

Global search starts with at most 128 candidates and proceeds in batches of at most 64. If the initial incumbent is below 0.9 loading, subsequent batches verify that incumbent until the search reaches a plateau or exhausts the pool. If the prefix fails that loading check, global expansion can end when a qualifying incumbent appears or when two successive batches leave an available incumbent unimproved. If no candidate is admissible, the search continues through the pool.

At global exit, FRAME locks the substation of the best admissible candidate when a local completion is warranted. The completion pool is the set of unqueried representatives with an original alias at that station. If the initially qualifying incumbent survives the verification phase, FRAME returns it without a completion pass. A completion pass keeps the original incumbent in consideration and adds at most one query per unqueried representative. The best observed selection key is therefore nondecreasing within that pass. The control loop evaluates operating cost and survival after the resulting actions are applied.

### Ordering and completion controls

The fixed mixed order keeps the first 64 representatives in the original order, then inserts up to 64 distinct candidates from a DC-based loading order. The remaining representatives keep their original relative order. Unsupported DC candidates preserve their original slots in the DC ranking, and calls outside the supported input conditions use the original order. The simulator still determines admissibility and final selection.

The learned completion controls share a frozen base network. It can request the next local batch. At the first native stop, the 208 parent gate compares predicted improvement probability times conditional positive improvement with a known query charge. It can authorize one further batch. The 261 variant adds a bounded-input linear correction to the normalized parent margin and can cancel a purchase the parent would otherwise make. The two guarded configurations also complete the remaining local pool when they would stop at this gate and the admissible incumbent has predicted loading above 0.9. The correction's inputs and gate are held fixed within the ablation.

The independent LogEIPC arm shares the mixed global order, global search, initial incumbent, and station lock. A fixed Matérn Gaussian process on public command vectors conditions on revealed feedback. The comparator queries the remaining candidate with the largest expected improvement per query cost and stops when that ratio is at most one. It uses the historical per-query proxy price and its own local stopping rule. Operating cost and control time are measured separately from this proxy objective.

## 4 Evaluation

### Design and metrics

The 2039 comparison uses four previously observed development weeks, with scenario 100 and environment seed 0. The learned arms use one frozen pool. GRID330 evaluates the original FRAME heuristic, original-order protected learned policies, and the corresponding mixed-order policies. GRID334 compares the mixed-order guarded 261 policy with independent LogEIPC in fresh paired runs. Each arm has one run per week within its own comparison. The same closed-loop controller handles actions and continuous operation. The earlier 2035 deletion ablation is reported separately.

A full week contains 2,017 simulator steps under the replay convention. Operating-cost comparisons require both arms to complete the week. Candidate queries count evaluations in topology search. Total forecasts include the rest of the instrumented controller. Controller time measures cumulative decision time, including search, scoring, and selection. Timing comparisons use matched horizons within a run batch. The historical March FRAME trajectory supplies counts and costs. Its timing is excluded from the GRID330 pair.

### Does local completion recover a missed configuration?

In the earlier fixed-order deletion ablation, an exhaustive reference evaluated all 909 candidates at step 1733 of the 2035 March development week. The global-only variant stopped at 256 queries and returned candidate 26. FRAME spent 198 further queries at the selected substation and recovered the exhaustive winner, candidate 309. All three variants completed that week. Operating cost was 4.810% above the exhaustive reference for the global-only policy and 0.372% above it with local completion. This case illustrates why an incumbent can be a useful guide for selecting a group even when the incumbent itself can still be improved.

### What do the current policy comparisons show?

Table 1 reports per-week differences. Candidate deltas count the first arm minus its paired reference. Cost and time deltas are relative to the reference.

| Week | Arm vs. reference | Steps per arm | Candidate delta | Cost delta | Controller time delta |
| --- | --- | ---: | ---: | ---: | ---: |
| 2039-03-14 | Guarded mixed 261 vs. Original FRAME | 2017 | -1221 | -0.94% | — |
| 2039-06-13 | Guarded mixed 261 vs. Original FRAME | 2017 | -215 | -0.54% | +3.06% |
| 2039-09-12 | Guarded mixed 261 vs. Original FRAME | 451 | 0 | — | +2.29% |
| 2039-12-12 | Guarded mixed 261 vs. Original FRAME | 2017 | -179 | 0.00% | +0.90% |
| 2039-03-14 | Guarded mixed 261 vs. Guarded mixed 208 | 2017 | 0 | 0.00% | +1.78% |
| 2039-06-13 | Guarded mixed 261 vs. Guarded mixed 208 | 2017 | 0 | 0.00% | +0.01% |
| 2039-09-12 | Guarded mixed 261 vs. Guarded mixed 208 | 451 | 0 | — | -0.29% |
| 2039-12-12 | Guarded mixed 261 vs. Guarded mixed 208 | 2017 | 0 | 0.00% | +0.28% |
| 2039-03-14 | LogEIPC vs. Guarded mixed 261 | 2017 | 151 | +0.09% | -2.99% |
| 2039-06-13 | LogEIPC vs. Guarded mixed 261 | 2017 | -146 | 0.00% | -0.32% |
| 2039-09-12 | LogEIPC vs. Guarded mixed 261 | 451 | 0 | — | +0.12% |
| 2039-12-12 | LogEIPC vs. Guarded mixed 261 | 2017 | -138 | 0.00% | +0.09% |

September ends for both arms at step 451 in each pair. All other displayed pairs complete the week. The September interval has one exception step per arm, and the audited runs have zero illegal or ambiguous steps.

Across the four GRID330 horizons, mixed-order protected 261 evaluates 1,615 fewer candidates than original FRAME, a 20.95% reduction. The March and June operating costs decrease and December is unchanged. The two same-day completed pairs with FRAME timings show controller time increases of 3.06% and 0.90%. Comparing the two protected 261 configurations within GRID330, changing only the order reduces queries by 1,528. Under the same mixed order and protection condition, the 208 and 261 runs have identical delivered trajectories, costs, candidate counts, and total forecasts. The correction cancels zero purchases.

In GRID334, LogEIPC uses 133 fewer candidate queries and 72 fewer total forecasts across the matched horizons. Cumulative controller time over the three commonly completed weeks is 227.16 seconds for LogEIPC and 229.82 seconds for guarded 261, a 1.16% difference. Individual complete weeks span `−2.99%` to `+0.09%`. June and December follow identical trajectories and yield equal costs despite fewer LogEIPC queries.

### Where did locally cheaper search become more expensive?

At the first March divergence, step 520, both policies have identical input state, 192 identical global evaluations, and the same current incumbent. LogEIPC stops. The base network in 261 requests the final 35 candidates before the residual and local protection are consulted. The added evaluation replaces the incumbent and lowers maximum predicted loading from 0.899714 to 0.896234. Over the two resulting March trajectories, LogEIPC uses 105 fewer local candidates and 256 more global candidates, for 151 additional candidate queries and 0.0861% higher operating cost. The query accounting follows the observed trajectories, which separate after step 520.

## 5 Discussion and conclusion

The experiments distinguish three mechanisms. Substation completion recovers an important configuration in the earlier fixed-order ablation. Mixed ordering reduces candidate search in the later guarded configurations. The matched learned-policy ablation measures the residual separately and observes identical outcomes on the four development horizons. The independent stopping comparison shows that immediate savings in a local search can accompany additional search later in closed loop.

The current results cover one simulator environment, previously used development dates, and one model pool for the recent comparisons. The elapsed-time measurements describe single pairs and vary by week. The evidence supports reporting topology evaluation effort and closed-loop operating outcomes together, with separate measurement of actual controller time.

## References

1. Marot et al. 2021. *Learning to run a Power Network Challenge: a Retrospective Analysis*. Proceedings of the NeurIPS 2020 Competition and Demonstration Track. [PMLR](https://proceedings.mlr.press/v133/marot21a.html)
2. La Javaness. *L2RPN 2023 LJN Agent*. [Source implementation](https://github.com/lajavaness/l2rpn-2023-ljn-agent)
3. Hassouna et al. *Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach*. ECML PKDD Applied Data Science track. [Publisher DOI](https://doi.org/10.1007/978-3-032-06129-4_8)
4. Xie et al. 2026. *Cost-aware Stopping for Bayesian Optimization*. International Conference on Machine Learning. [PMLR](https://proceedings.mlr.press/v306/xie26c.html)

