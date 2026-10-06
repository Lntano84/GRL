# Validity fixes: three bugs that retracted the Go/No-Go conclusions

> 2026-09-18. All three were found by the PI reading the runner, and all three were confirmed by
> direct measurement before anything was changed. The Go/No-Go report is retracted; this file
> records what was wrong, the evidence, and what now prevents a recurrence.

---

## Bug 1 — the objective function was never wired in

**What was wrong.** Every arm reported `run_overexposure(...).spread`, which counts positively
activated nodes **in the whole graph**. The source model is *targeted*: its objective is
``|positive_at_end ∩ D|`` for an explicit target set ``D``. The runner used ``D`` to restrict
candidate *eligibility* and never to restrict *scoring*.

**Evidence.** Congress-Twitter, `D` = top 20% by out-degree, 20 seeds:

```
|D|                      95
run.spread (whole graph) 145      <- what the runner reported
|positive ∩ D|            24      <- what the contract requires
|positive \ D|           121
```

So **84% of the reported quantity lay outside the target set**, and it was not a constant offset:
the untargeted count is dominated by nodes the objective does not count.

**A second, independent error on top of it.** The gate's absolute tolerance was computed in *target
nodes* (`|D| / 1000`) and compared against a difference in *whole-graph* spread. The units did not
match, so even the tolerance was meaningless.

**What it invalidated.** The Gate 1(b) verdict, the quality–cost curves, and every conclusion drawn
from them — that is, items 1, 2 and 3 of the Go/No-Go report.

**Fix.**
* `src/grl/oracle/targeted_mc.py` — `TargetedMonteCarloOracle`. Every number it produces is
  ``|positive_at_end ∩ D|``: `spread`, `score`, `score_with_uncertainty` and `state` all route
  through the contract. `is_exact = False`, so the reference is named after its Monte-Carlo budget
  rather than called an oracle.
* The runner's `paired_target_counts` scores every arm as ``len(run.positive & D)`` on shared
  windows, and `paired_spreads` (whole-graph) is deleted.

**The guard, and the bug the guard itself had.** The oracle asserts the range of what it returns.
The first version applied one range to everything and rejected
``paired marginal = -4.0 is outside [0, |D|]``, silently removing three arms from the run. A
*level* must lie in ``[0, |D|]``; a *marginal* is a difference of two levels and may be negative —
indeed a negative marginal is the phenomenon the paper is about, since adding a seed can push a
target past its upper threshold. There are now two checks, `_check_level` and `_check_marginal`,
and the distinctness is the point: forbidding negative marginals would have hidden the effect under
study. The observed marginals at email-Eu-core ``|S|/n = 0.20`` were **−1 to −9 target nodes**,
which is direct evidence that the contracted objective is non-monotone there.

---

## Bug 2 — patience was never enabled

**What was wrong.** `run_delta2_patience` forwarded the runner's global ``--stopping`` rule into
``should_stop``. The default is ``fill_budget``, whose ``should_stop`` always returns ``False``.

**Evidence.**

```
FILL_BUDGET.should_stop([], [100, 99, 99, 99, 99])  ->  False
PATIENCE_2.should_stop([],  [100, 99, 99, 99, 99])  ->  True
```

The arm was therefore a re-rank-only arm carrying one extra cascade per step.

**What it invalidated.** The claim that "the stopping lever never fires, so the mechanism the paper
proposes has not produced a gain anywhere it has been tried". That claim measured the absence of a
rule.

**Fix.** The arm owns its rule: ``rule = PATIENCE_2``, ignoring the global flag, and the rule
actually used is recorded per cell. The observed value the rule reads is the **contracted**
``|positive ∩ D|``, so the rule cannot be reading a different objective from the one it is judged by.

**What the arm does now that it is enabled.** At Congress-Twitter ``k = 48`` it **stops after 3
seeds** (``stopped_early = True``) and reaches 25.21 target nodes against 28.24 for filling the
budget — so the lever fires, and on that cell it fires too early and costs about 3 target nodes.
That is a real measurement, and it is the opposite of what the retracted run appeared to show.

---

## Bug 3 — the "independent" replicates were 99.5% the same

**What was wrong.** Trial seeds were ``base + offset`` for ``offset in range(trials)``. With
replicate bases 20260917 and 20260918 and 200 trials, the two streams overlap in every draw but one.

**Evidence.**

```
replicate A windows: 20260917 .. 20261116
replicate B windows: 20260918 .. 20261117
|A ∩ B| = 199 / 200   (99.5%)
```

Sharing windows *between arms* is what makes the comparison paired and is desirable. Sharing them
between *replicates* means the second replicate is not a replicate.

**Fix.** ``grl.oracle.trial_seeds(base, n, stride=1_000_003)`` gives disjoint streams, and the
streams are namespaced by purpose as well:

```
SELECT_NS =      0     ranking and candidate probes
EVAL_NS   = 500_000    the scored quantity
STATE_NS  = 900_000    exposure reads
```

so a method never selects on the windows it is scored on. Verified: ``|A ∩ B| = 0/200`` and the
evaluation stream does not intersect the selection stream.

---

## What the retraction does and does not say

**Does say.** The three experiments in the Go/No-Go report measured the wrong objective, with a
disabled mechanism and non-independent replicates. None of their conclusions may be used.

**Does not say.** That sequential decision-making is worthless, or that the paper must change
direction. The retracted runs cannot support that, and the corrected runs are not finished.

**Unaffected.** The raw-weight finding: 7 of 9 graphs exceed the model's ``Σω ≤ 1`` allowance in raw
form (ca-HepPh by 4.9×), and applying a normalisation strategy to an already-normalised graph made
``sum_to_one`` and ``clip_to_one`` byte-identical. That is about preprocessing, not about the
objective, and it stands.

---

## How to check this has not regressed

```bash
# the three bugs, as executable checks
python -m pytest tests/test_targeted_oracle.py tests/test_stream_independence.py -q

# every objective requirement against the artifacts
python scripts/audit/verify_go_no_go_artifacts.py
```

The artifacts now also carry ``TargetedObjectiveError`` guards on every level and marginal, so a
regression to whole-graph counting raises instead of producing a plausible table.
