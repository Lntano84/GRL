# Corrected validation: sequential vs static and screening, on the contracted objective

> 2026-09-18. Both stages **complete**. Supersedes the retracted numbers in
> `docs/GO_NO_GO_REPORT.md`, whose three validity bugs are documented in `docs/VALIDITY_FIXES.md`.
>
> Every figure below is ``|positive_at_end ∩ D|`` --- the contracted objective --- read from
> `docs/results/validation_small.json` and `validation_large.json`. Stage A is 8/8 cells
> (`complete: true`, reference requested and run); stage B is 8/8 (`complete: true`).

---

## Design

| | stage A | stage B |
| --- | --- | --- |
| fractions | `\|S\|/n = 0.01` | `\|S\|/n = 0.20` |
| realised `k` | 5 (Congress-Twitter), 10 (email-Eu-core) | 95, 201 |
| graphs | Congress-Twitter, email-Eu-core | same |
| pools × seeds | 2 × 2 = 4 cells per graph | 2 × 2 |
| full-pool MC reference | **yes** (`mc_greedy_mc50`) | no — `O(k·\|pool\|·MC)` would be ~2M cascades per cell |
| cells | 8 | 8 |
| wall time | 2,159 s | 9,140 s |

Both stages run `delta2_sequential`, `delta2_patience`, `patience_static` (the new control),
`degree_static`, `delta2_static`, `random_pruning`, `selective_analytic`, `adaptive_selective`.
Streams are namespaced (select 0 / eval 500k / stop 700k / state 900k) and the reference is named
after its MC budget because it is an estimator, not an optimum.

---

## Gate 1(b): sequential vs static — **DOES NOT PASS**, in either regime

| stage | cells | mean lift | median lift | positive | max lift |
| --- | --- | --- | --- | --- | --- |
| A (`\|S\|/n = 0.01`) | 8 | −11.59% | **−11.46%** | **0/8** | −0.95% |
| B (`\|S\|/n = 0.20`) | 8 | −3.40% | **−3.60%** | 4/8 | **+2.20%** |

The threshold is +5%. **The largest lift anywhere in 16 cells is +2.20%.**

Stage B splits cleanly by graph, and the split is explained by the scorer diagnosis:

| graph | `ρ_degree` | `ρ_δ₂` (targeted) | lift |
| --- | --- | --- | --- |
| Congress-Twitter | +0.352 | **−0.248** | **+0.28% … +2.20%** (4/4 positive) |
| email-Eu-core | −0.472 | **+0.590** | **−7.48% … −8.34%** (0/4) |

That is the opposite of what the score correlations would predict, and it is worth stating plainly:
the graph where the analytic score is *worse than degree* is the graph where sequential selection
gains, and the graph where the score is *much better than degree* is the graph where it loses. A
2% gain does not compensate for a 8% loss, and neither is near the threshold.

---

## The patience stop is a large, consistent loss — and the control says why

`delta2_patience` fires in **6 of 8** stage-A cells and **8 of 8** stage-B cells, and it is the worst
arm in the set:

| stage | median lift vs best static | positive |
| --- | --- | --- |
| A | **−15.84%** | 0/8 |
| B | **−15.36%** | 0/8 |

Worst cell: email-Eu-core `k = 10`, where it stops after **3 of 10** seeds and reaches 32.65 against
44.63 for filling the budget --- a 27% loss.

The new `patience_static` control shares the stopping rule, the observation cost and the state read,
and differs only in that the ranking is frozen after one read (`state_reads == 1`, asserted by test):

| stage | `patience_static` | `delta2_patience` |
| --- | --- | --- |
| A median target count | **40.05** | 30.67 |
| B median target count | **39.86** | 30.82 |
| A median cascades | 125 | 175 |
| B median cascades | 138 | 175 |

**The control beats the sequential arm at lower cost in both stages.** So the stopping rule's effect
is not attributable to re-ranking --- re-ranking makes it worse. And since both are far below the
arms that simply fill the budget (40.62 / 42.94), the stopping rule itself is the problem: with the
contracted objective the observed count keeps improving for longer than the rule is willing to wait.

---

## Quality–cost: the gate now discriminates, and it rejects the screening arms

Stage A carries the reference, so the gate is evaluable there.

| arm | median cascades | median target count | PASS | FAIL | UNDECIDED |
| --- | --- | --- | --- | --- | --- |
| `degree_static` | **0** | **40.62** | **3** | 0 | 5 |
| `mc_greedy_mc50` | 44,000 | 40.44 | — | — | — (reference) |
| `delta2_static` | 25 | 40.23 | 1 | 0 | 7 |
| `random_pruning` | 3,375 | 40.23 | 0 | 0 | 8 |
| `patience_static` | 125 | 40.05 | 0 | 0 | 8 |
| `delta2_sequential` | 188 | 34.72 | 0 | **6** | 2 |
| `selective_analytic` | 3,562 | 34.68 | 0 | **5** | 3 |
| `adaptive_selective` | 3,938 | 34.68 | 0 | **5** | 3 |
| `delta2_patience` | 175 | 30.67 | 0 | **6** | 2 |

**Every sequential and screening arm FAILS the quality gate**, while the zero-cost `degree_static`
passes three cells outright. The criterion's cost half (≥30% fewer cascades) is satisfied by
everything, so quality is the only binding constraint --- and the free arms win it.

Stage B has no reference, so its cost table stands alone:

| arm | median cascades | median target count |
| --- | --- | --- |
| `degree_static` | **0** | 42.94 |
| `random_pruning` | 66,600 | **42.97** |
| `delta2_static` | 25 | 42.38 |
| `delta2_sequential` | 3,700 | 41.16 |
| `selective_analytic` | 70,300 | 41.07 |
| `adaptive_selective` | 77,700 | 41.07 |
| `patience_static` | 138 | 39.86 |
| `delta2_patience` | 175 | 30.82 |

At `|S|/n = 0.20` the screening budget is so large that `random_pruning` costs 66,600 cascades to
match what `degree_static` gets for zero.

---

## What the answer to the PI's question is

**Why does analytic screening cost more than random pruning and do no better?** The scorer diagnosis
(`docs/results/scorer_diagnosis.json`) answers it, and the answer is graph-dependent:

* **Not** because the score ignored the target set. Targeting *helped* on average (+0.151), and the
  fix was right. It hurts only on Congress-Twitter, where the targeted score is **negative**
  (−0.136 to −0.248) while degree is **positive** (+0.080 to +0.352) --- on that graph the analytic
  score is worse than free degree.
* **Not** because it chases already-active targets: 0% of targets are already positive, and 0% of the
  score's top-10 point at one, in every cell.
* **Not** because two hops is the wrong depth: hops 2 and 3 give identical correlations.

So the score is simply not reliable across graphs, and where it fails it fails badly. That is the
thing to fix, and it is a property of the scoring function rather than of the pipeline.

A finding from the same run that corrects the paper: **68% of candidates have a negative marginal**
in the contracted objective at Congress-Twitter `|S|/n = 0.20`. The regime table measured 0.5% mean
and 4% worst, but over `run.spread`. **Non-monotonicity is far more prevalent in the targeted
objective than the whole-graph count ever showed.**

---

## What this does and does not support

**Supports.** The sample-efficiency framing is not supported: on the contracted objective, with a
real Monte-Carlo baseline, a zero-cost heuristic matches the reference, and every arm that spends
cascades either ties or loses. The sequential and screening arms **fail the quality gate**.

**Does not support.** That the paper must change direction. This is two graphs and two fractions. The
regime split in stage B is real and unexplained, and the diagnosis says the scoring function --- not
the sequential idea --- is the weak component. Fixing the score is the next experiment.

**Also does not support** any claim built on Congress-Twitter alone, which is the graph the earlier
work was built on and the graph where the analytic score is anti-correlated with the truth.
