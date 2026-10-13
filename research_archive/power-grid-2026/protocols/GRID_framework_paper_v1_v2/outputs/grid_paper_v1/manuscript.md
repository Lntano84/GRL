# Progressive Topology Search with Substation Completion for Efficient Power Grid Control

**Working manuscript, 13 October 2026.** This draft presents the framework as the main subject. The primary fresh-date confirmation is frozen in GRID62 and is not yet available in this version. Development results, historical model comparisons, and pending confirmation are explicitly separated. Authors, target venue, formatting, and final archival references remain to be supplied. This is a full first draft, not a submission-ready manuscript.

## Abstract

Simulation-backed power grid controllers often evaluate many topology actions to obtain a feasible intervention. Ranking or early stopping can reduce this cost, but a favorable immediate forecast does not establish the quality of the subsequent control trajectory. We study a replaceable search layer for an existing hybrid continuous and topology controller. The layer expands a ranked action prefix using observed simulation feedback and, when its stopping rule fires during extended search, completes the unqueried candidates associated with the incumbent's substation. This completion preserves the best observed single-state selection key while limiting additional queries to a defined structural neighborhood. In a controlled experiment on four previously used date families, all methods share the action order, base controller, feasibility checks, and compilation caches. The complete framework reduces topology-search queries by 69.37% and actual controller time by 13.06% relative to exhaustive search while satisfying a predeclared operational-cost and survival screen. Removing completion produces a 4.81% worst complete-week cost increase, compared with 0.372% for the framework. The effect is concentrated in one development family rather than uniformly distributed. A final evaluation on all three remaining unaccessed date families is frozen separately. The method requires no newly trained graph model or reinforcement learning policy, and makes no global optimality, long-term safety, or hard real-time guarantee.

**Keywords:** power grid topology control; simulation-backed search; computational efficiency; structural completion; Grid2Op.

## 1. Introduction

Topology changes give grid operators an additional way to relieve congestion. A practical controller must nevertheless coordinate these discrete actions with continuous redispatch, curtailment, restoration, and operating restrictions. Existing hybrid agents provide a useful foundation: they offer a working control loop and let a search component validate candidate interventions before execution. The remaining question is how much candidate simulation is necessary at a particular decision.

Evaluating every eligible action provides a strong immediate reference, but its computation can be redundant. Conversely, selecting the first action whose forecast satisfies a line-loading threshold, or stopping after a fixed prefix, can skip another action that changes the subsequent trajectory substantially. The number of avoided simulations therefore cannot stand alone as the performance objective. Controller time, complete-episode operational cost, and survival must be assessed jointly.

This paper studies a specific compromise. We retain an existing controller and replace only its expensive topology-search hook. A progressive search first evaluates a ranked prefix, then buys additional batches when the received feedback does not support stopping. Before an extended-search stop, it completes the current incumbent's substation neighborhood. The completion addresses a concrete failure mode: a promising substation may have several action configurations spread across the global order, so plateau stopping may discover the substation but miss a materially better configuration within it.

The contribution is not a new graph neural network architecture or a claim that simulation verification is new. We make three more specific contributions:

1. A fully specified search layer that combines feedback-driven batch expansion with incumbent-substation completion, while keeping ranking replaceable.
2. A conditional single-state non-degradation property and an explicit additional-query count for completion, together with the limitations of both statements.
3. A controlled closed-loop study that isolates search from ranking and shared implementation caches, reporting actual controller time, queries, operational cost, survival, and adverse cases.

The controlled development study already supports a quality–computation tradeoff. Its small and previously exposed date set does not establish broad generalization. We therefore reserve the remaining three unaccessed date families for a final frozen confirmation and do not tune the method on those outcomes.

## 2. Related work and positioning

The public LJN agent provides modular continuous optimization and topology search, including learned action ranking. We use that implementation as the base system and retain its existing controller context [1]. Neither hybrid control nor a learned Top-K proposal is attributed to this paper.

SoftGNN learns action preferences from multiple useful interventions. Its agent validates ranked actions until the first forecast below a loading threshold is found; when no such action is available, the basic variant returns a do-nothing action. It also handles disconnected endpoints and offers a restricted contingency-based extension [2]. We adopt the first-verified-safe principle as a controlled search reference but strengthen its exhausted fallback to the best admissible queried candidate. Our experiment is an adaptation of this search policy, not a reproduction of the complete published agent.

Generalizable GNNs for topology control examine graph representations and learned action selection, including verification and hybrid search variants [3]. These precedents motivate separating the ranking model from the search policy. Our current evidence does not establish that graph ranking improves over the author's strong frequency order.

| Component | Existing precedent | Role in this paper |
|---|---|---|
| Continuous/discrete hybrid control | LJN agent [1] | Retained base system |
| Candidate ranking and simulation verification | [1–3] | Shared interface, replaceable ranking |
| First verified action below a loading threshold | SoftGNN [2] | Strengthened search-policy reference |
| Disconnected-endpoint handling | SoftGNN [2] | Shared legality preprocessing |
| Feedback-driven expansion plus incumbent-substation completion | Specific policy examined here | Candidate framework contribution, tested by controlled ablation |
| Compilation reuse | Conventional implementation optimization | Shared engineering support, not a unique learning contribution |

This table establishes the scope of the proposed contribution rather than an exhaustive priority claim. A final literature pass should check the combination of adaptive termination and substation-level completion before a submission claims novelty. Broad neighborhood search or progressive widening should not be renamed as a new general theory.

## 3. Problem and information model

At a topology-search request, let x be the current public observation and b the base action supplied by the hybrid controller. A fixed library contains topology candidates with original identifiers and substation metadata. The controller preprocesses the library using current legality restrictions and disconnected-line endpoints. Candidates producing an identical current physical action vector are represented by a canonical action; an alias map retains their original identities.

Let A(x,b) denote the resulting ordered representatives. A query simulates a candidate combined with b and returns its forecast reward r, maximum loading rho, termination flag, and legality/exception flags. The online search sees only actually dispatched query responses. Full-search traces from other arms are used in audits after execution and never supplied to the controller.

An action is admissible when its forecast has positive maximum loading strictly below the passed limit, does not terminate, and has no illegal, ambiguous, or exceptional feedback. The limit is the smaller of current maximum loading and the threshold passed by the unchanged base controller. This is a one-step admissibility test, not a guarantee of future safe operation.

Admissible actions are compared with the base implementation's key

`f(a) = (float32(r(a)), -original_id(a))`,

ordered lexicographically. The physical episode cost is separately computed from the operational-cost ledger. Maximizing the single-step selection key is not equivalent to minimizing full-week operational cost.

More precisely, the reported cost follows the installed Grid2Op `L2RPNSandBoxScore` convention: at each step, a marginal generation-cost coefficient multiplies the sum of estimated loss energy, absolute redispatch energy, **the change in curtailed energy**, and storage-throughput energy. The curtailment term is a change relative to the previous observation, not total curtailed energy. A separate public-state ledger reconstructs this reward with a floating-point tolerance. Thus our cost is a particular benchmark accounting quantity; a percentage improvement is not an independently established reduction in production electricity expenditure or total renewable curtailment.

The objective is a measured tradeoff: reduce the time spent in controller decisions without introducing earlier episode failure or an unacceptable complete-episode cost increase relative to exhaustive search. The 0.5% cost allowance used in the experiments is an investment screen fixed before evaluation. It is not a mathematical equivalence margin or a universal engineering safety standard.

## 4. Progressive search with substation completion

### 4.1 Progressive expansion

The search evaluates up to 128 initial representatives and maintains the best admissible incumbent. If the initial incumbent is below the inherited rho=0.9 threshold, the search returns it. If the prefix exhausts the pool, there is no unqueried action to inspect.

Otherwise the search proceeds in batches of 64. An improving batch resets the stale-batch counter; a batch without a better admissible key increments it. Extended search stops when the incumbent is below 0.9 or when at least 256 representatives have been queried and two consecutive batches have not improved the key. If no admissible incumbent exists, it continues until exhaustion. The schedule is fixed and is not an outcome-selected hyperparameter sweep.

### 4.2 Structural completion

When extended search would stop with an admissible incumbent and unqueried representatives remain, the framework identifies the incumbent's original substation. It queries all still-unqueried canonical representatives having an original alias assigned to that substation. Every new admissible response can replace the incumbent using the same key. Completion is one pass over this chosen neighborhood; it does not recursively expand the neighborhood of a replacement action.

The initial safe-prefix return does not trigger completion. It is therefore incorrect to claim that every returned action has a fully explored substation neighborhood. Completion does not examine all other substations and can still miss the full-search winner.

### 4.3 Algorithm

```text
Input: public state x, base action b, ordered canonical pool A,
       original-substation metadata and alias map
C <- query first min(128, |A|) representatives
a <- best admissible element of C, if any
if pool exhausted or a exists with rho(a) < 0.9:
    return a
stale <- 0
for each next batch of at most 64 representatives:
    old_key <- key(a), if a exists
    query batch and update a
    stale <- 0 if key improves else stale + 1
    if a exists and (rho(a) < 0.9 or (|C| >= 256 and stale >= 2)):
        break
if a exists and the extended search left unqueried representatives:
    s <- original substation of a
    L <- unqueried canonical representatives with an alias at s
    query L once and update a
return a, or no topology proposal if none is admissible
```

Returning no topology proposal delegates to the retained base controller. We do not replace this with an undocumented universal do-nothing policy.

### 4.4 Conditional property and cost

Let C be the queried candidate set before a triggered completion, and let a_C be its best admissible action. Let L be the unqueried representatives selected by the original-substation alias rule. Since a_C remains available after querying L, selecting the best admissible element of C union L yields

`f(a_after) >= f(a_C)`.

The proof is simply inclusion: an existing incumbent is not removed, and updates require a larger key. The additional query count is exactly the number of previously unqueried representatives in L. Canonicalization and the query registry prevent repeated purchases, so total queries cannot exceed |A|.

This elementary property clarifies what completion protects. It does not prove that the chosen reward is globally optimal, that the full-week cost cannot increase, or that all future contingencies remain feasible. In the worst case the framework can still query the complete pool. Its schedule is not a hard real-time deadline.

### 4.5 Implementation and ranking

The main framework experiment uses the author's existing frequency-descending action-library order, represented by increasing frozen library ID. Every arm uses the same order. The frequency prior is a strong external artifact, not an uninformative random ordering, and the author's original teacher-date exposure is unknown.

An existing GNN residual ranker remains an optional implementation. Historical experiments did not establish a stable additional benefit over the strong frequency ordering. Its training, teachers, graph construction, and inference costs must be disclosed if included; they are not needed for the main framework claim. This paper does not report newly trained RL.

DC and safe-QP compilation caches are shared by all applicable arms. First construction and relevant current-state updates are accounted for. Numerical and path-equivalence qualification of these caches was performed before the search comparison. Their benefit is not added arithmetically to a search speed percentage measured against a different reference.

## 5. Experimental protocol

### 5.1 Environment and base controller

Experiments run in the public `l2rpn_idf_2023` Grid2Op environment with LightSim2Grid and the retained LJN-derived hybrid controller. Environment rules, disturbances, forecasts, reconnect/restoration logic, and continuous control are shared across arms. The topology library contains 909 original candidates; the current canonical pool can be smaller after masking and equivalence handling. The exact software, compatibility changes, library hashes, and machine metadata are included in the reproducibility receipts.

We use a single CPU worker and bounded single-thread numerical libraries. A week is run until natural completion or environment termination. Earlier failures are retained and never treated as cheaper successful runs. The expensive topology hook is designated N-1 by the base implementation; counting its simulations does not certify exhaustive N-1 security analysis over every line.

### 5.2 Policies and fairness

| Policy | Difference from the shared system |
|---|---|
| FULL | Query all current canonical representatives and retain the best admissible key |
| FIRST_SAFE_BEST | Query in order until the first admissible rho<0.9 action; if none exists, keep the best admissible exhausted result |
| NO_LOCAL | Same progressive expansion and stopping as FRAME, with completion disabled |
| FRAME | Progressive expansion and incumbent-substation completion |

FULL and FIRST_SAFE_BEST are primary search references; NO_LOCAL is an internal mechanism ablation. No arm is weakened by withholding common legality preprocessing or caches. All methods have the same ranking. The comparison isolates search policy and does not independently compare neural architectures.

### 5.3 Data roles

GRID61 contains four previously exposed development dates with two balanced timing repeats per policy, totaling 32 trajectories. It is used to study mechanism and illustrate failure cases.

GRID62 contains all three date families still unaccessed in the local experiment history: 2035-04-09, 2035-05-14, and 2035-11-19. A primary episode variant is selected by a fixed hash before payload extraction. All variants of these dates are excluded from training and tuning. The order, policies, thresholds, and outcomes-independent selection are frozen first. This yields 24 trajectories with the same two-repeat design. The final pool is small and not seasonally balanced; it cannot support cross-grid or production generalization claims.

Earlier eight-date GNN/PPO/SoftGNN tables remain historical supplementary experiments. They are not merged with the new framework confirmation as though all observations had an identical protocol or independent status.

### 5.4 Metrics and measurement

Actual controller time is the wall-clock interval around each complete controller decision, including search preparation, ranking, simulation, and continuous-control work inside that decision. Initialization, physical `env.step`, audit, and artifact serialization are reported separately. Avoided queries are a secondary computational metric rather than a substitute for time.

For each date, timing repeats are averaged before computing its method/FULL ratio. Dates receive equal weight in the macro average. The pooled ratio is reported separately when useful. Cost changes are normalized by the corresponding same-batch FULL complete-week cost. Cost ratios are withheld on truncated failed episodes, and speed ratios require comparable horizons. Mean controller time does not imply a reduction in p95 decision latency.

The frozen fresh-date screen requires no earlier failure, all complete-week cost increases no greater than 0.5%, at least two complete comparable dates, a mean controller-time reduction of at least 5%, and a positive time direction on at least two dates. These are finite investment criteria, not statistical significance or equivalence tests.

### 5.5 Audit and reproducibility

Every trajectory saves public observations, delivered action vectors, simulation feedback, query phases, per-step cost ledgers, and timing components. An offline replay checks the stopping/completion decisions using only the dispatched responses. Query responses at matched public states are compared across arms. Repeated runs are checked for identical physical paths, and operating-system power events are inspected for timing contamination.

These checks audit saved vectors, feedback, and accounting. They do not independently rerun an AC power-flow solver. Hash receipts preserve original and failed records; the published lightweight archive identifies raw local payloads without silently replacing completed experiments.

## 6. Results

### 6.1 Controlled development comparison

| Policy | Quality screen, runs | Completed weeks, distinct dates | Controller-time reduction | Query reduction | Worst complete-week cost increase |
|---|---:|---:|---:|---:|---:|
| FULL | 8/8 | 3/4 | 0.00% | 0.00% | 0.00000% |
| FIRST_SAFE_BEST | 4/8 | 3/4 | 7.03% | 51.45% | 3.95164% |
| NO_LOCAL | 6/8 | 3/4 | 14.25% | 71.84% | 4.81017% |
| FRAME | 8/8 | 3/4 | 13.06% | 69.37% | 0.37200% |

The June trajectory fails at step 1901 for every method; both repeats preserve that result. A quality-screen pass is not an episode-completion count. The method does not eliminate this shared base-controller failure.

NO_LOCAL saves more computation but misses an important action in March. Completing the incumbent's substation reduces that date's complete-week cost increase from 4.81017% to 0.37200%, a difference of 4.43817 percentage points relative to FULL cost. The overall controller-time saving drops by about 1.19 percentage points when completion is included. This is the observed cost of the quality repair, rather than evidence that completion is free.

The controlled comparison also retains an adverse example. FIRST_SAFE_BEST has a lower September cost than FRAME by 0.729% of FULL cost. Its October cost increase of 0.51914% is only slightly beyond the fixed 0.5% screen. FRAME does not dominate every policy on every date or metric.

### 6.2 Single-state mechanism

At the documented March step 1733, NO_LOCAL stops after 256 queries and selects action 26. Completing the incumbent's substation adds 198 queries and finds action 309, also selected by FULL after 909 queries. This establishes a concrete skipped-local-configuration mechanism at one state. It does not identify every cause of a weekly cost difference or demonstrate global optimality.

The saved-state qualification reconstructs 90 existing FRAME calls exactly, and completion changes the choice in 12 of them. The live GRID61 audit verifies 63,616 physical rows, 70,358 matched feedback items, and 16/16 repeated physical paths. These checks support the implementation interpretation; they are not additional independent date samples.

### 6.3 Final frozen confirmation

**PENDING IN V1.** Insert GRID62's audited per-date cost, survival, controller time, query counts, and frozen verdict here. The table must retain all three selected dates, all four policies, both timing repeats, and failures. No winner-selecting revision of the method is permitted using this result. A negative or mixed result changes the strength of the conclusion rather than disappearing from the manuscript.

### 6.4 Historical ranking and system evidence

On an earlier eight-date matched engineering comparison, the quality-strengthened GNN-integrated controller reduced actual time relative to FULL by about 15.75% and queries by about 70%, with a worst complete-week cost increase of approximately 0.080%. That comparison was not uniformly faster than resource-matched PPO or SoftGNN adaptations, and its decision-latency tail was not improved.

The author's frequency order is a competitive ranker. Subsequent limited comparisons did not establish a stable GNN increment over it. We therefore use the frequency order in the main framework experiment and interpret learned ranking as an optional auxiliary module. Historical model results are not substituted for the new policy-only confirmation.

Compilation reuse also has a measurable but separate contribution: same-path qualification demonstrated reduced repeated DC/safe-QP construction overhead. Those optimizations are provided to the full-search baseline as well. Percentages from different date sets, controller versions, and denominators are not summed into a synthetic total improvement.

## 7. Discussion and limitations

The framework exploits a structural asymmetry between exploring another substation and finishing candidate configurations of a substation already supported by feedback. That interpretation is supported by the known single-state recovery and the controlled NO_LOCAL contrast. The main quality improvement in development is concentrated in one date. It is not evidence that every congestion episode needs completion.

The method has no hard query or wall-clock budget below exhaustive search. A large incumbent neighborhood may consume much of the remaining pool. The fixed patience schedule can miss a better distant action. Completion's monotone-key property is restricted to a current state and does not extend to operational cost or unknown future disturbances.

The evaluation covers one grid and a small date pool. Remaining fresh dates are not seasonally balanced. Author action-library exposure is unknown, even though our local training/tuning exclusions are explicit. The complete published SoftGNN or Generalizable-GNN agents are not reproduced in the policy-only matrix. Claims of broad state-of-the-art superiority, production readiness, or novel GNN/RL capability would exceed the evidence.

The accumulated experiment history is substantial, but negative model revisions and shared-controller failures are retained. Model tuning on the final confirmation would invalidate its intended role. A submission should make the data roles and adaptation scope visible without turning the main paper into an exhaustive experiment diary.

## 8. Conclusion

We presented a replaceable topology-search framework combining feedback-driven progressive expansion with incumbent-substation completion. Its single-state property describes what completion preserves and what it costs. A common-order closed-loop ablation demonstrates a measured quality–computation tradeoff and shows that simple aggressive stopping can save more queries while degrading complete-week operational cost. The evidence supports studying the framework independently of GNN architecture claims. The final conclusion and headline numbers will be revised only after the separately frozen three-date confirmation is audited; any failed or mixed outcome will remain part of the report.

## References

[1] Lajavaness. *L2RPN 2023 LJN Agent*. Official source implementation. https://github.com/lajavaness/l2rpn-2023-ljn-agent. Source version and compatibility changes are recorded in the experiment receipts.

[2] Mohamed Hassouna, Clara Holzhüter, Malte Lehna, Matthijs de Jong, Jan Viebahn, Bernhard Sick, and Christoph Scholz. *Learning Topology Actions for Power Grid Control: A Graph-Based Soft-Label Imitation Learning Approach*. arXiv:2503.15190v2, 2025. https://arxiv.org/abs/2503.15190.

[3] Matthijs de Jong, Jan Viebahn, and Yuliya Shapovalova. *Generalizable Graph Neural Networks for Robust Power Grid Topology Control*. arXiv:2501.07186v1, 2025. https://arxiv.org/abs/2501.07186.

## Appendix A. Evidence entry points

- Main mechanism data: `outputs/grid61/GRID61_report.md`, `review.json`, `per_run.csv`.
- Exact policies: `work/grid60/policies.py`, `work/grid27/policy.py`.
- Frozen confirmation: `outputs/grid62/design.json`, `reservation.json`, `code_freeze.json`.
- Common worker/timers: `work/grid55/run.py`, `worker.py`, with the policy-only adapter.
- Model and common-cache history: `outputs/grid55b/qualified_view/`, `outputs/grid57/`, `outputs/grid59/`, `outputs/grid56/`.
- Public small-artifact archive: https://github.com/Lntano84/GRL/tree/main/research_archive/power-grid-2026/protocols.

The repository's chronological state documents and source manifests resolve historical corrections. Raw observation/action vectors and checkpoints have local hash receipts and are not silently included as a lightweight source-only download.
