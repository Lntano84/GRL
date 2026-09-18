# GRL Decision Log

## 2026-09-02 — Use marginal gain as a candidate signal

The intended learning target is conditional marginal gain `Delta(v|S)`, evaluated by within-state
ranking and downstream decision quality rather than aggregate regression error.

## 2026-09-05 — Abandon the static-IC learning-versus-RIS claim

The same-protocol OPIM-C comparison showed that the old static-IC route did not establish a quality
or runtime advantage over mature RIS. That claim is permanently abandoned. No pre-fix static-IC
number may be reused as evidence for the current paper.

## 2026-09-15 — Change the setting to threshold-dependent overexposure

Human-authorized route change: study the threshold-window overexposure setting and target DASFAA
2027. The motivation is that the standard non-negative seed-independent coverage representation
does not match the objective, so the method must be evaluated with an explicit state-tracking
oracle. RL and an additional predictor architecture remain out of scope until the core evidence
exists.

## 2026-09-18 — Current post-fix paper decision

The state-machine correction and MC=300 sweep changed the evidence boundary:

- keep the exact coverage counterexample;
- keep the corrected 8-graph regime table;
- withdraw the old negative-share, baseline, calibration, surrogate-gap, and learned-scorer numbers;
- do not claim a general sample-efficient algorithm until post-fix quality--cost experiments pass;
- treat the measurement protocol as a valid contribution, not as a substitute for missing algorithmic
  evidence.

The current DASFAA submission remains conditional. If corrected sequential experiments cannot show
near-oracle quality with a reproducible reduction in expensive oracle work, remove the
sample-efficiency framing and retarget the manuscript.

## 2026-09-18 — Reference audit decision

Only bibliographic records verified against publisher/official proceedings pages or DBLP may remain
in the submission. The audit corrected five materially wrong records, removed one unsupported
related-work claim, one duplicate entry and one uncited entry, and aligned the prose with the
published methods. Short venue names are retained to keep the fully verified bibliography inside
the 16-page LNCS limit; DOI, volume/issue, article number and page data remain in BibTeX.

## 2026-09-18 — Pre-registered Gate 1: (a) PASSES, (b) NOT YET MEASURED

The 2026-09-15 entry set a gate with a stated consequence: **if either part fails, DASFAA is dropped
and the target moves to CIKM 2027.** Status, recorded here so it cannot drift:

| part | criterion | status |
| --- | --- | --- |
| **(a)** | Spearman correlation between true marginal gain and degree under overexposure collapses (≈0 or negative) | **PASSES.** At `\|S\|/n ≥ 20%` over 16 saturated cells: mean `ρ_degree = −0.177`, negative in 12/16 cells and on 7 of 8 graphs; `ρ_δ₂ = +0.420`. Source: `docs/results/signflip_fixedmodel_mc300.json` |
| **(b)** | sequential decision beats the static optimum by a material margin (>5%) | **NOT MEASURED.** This is the `sequential vs static` re-run and it is the only experiment that gates the venue |

**Deadline for (b): 2026-10-13** — 25 days from today. The budget is thinner than "early October"
suggests, and (b) must be run before the supporting curves, not after.

Note that (a) passing is a weaker statement than it looks. It says degree is a poor ranker under
overexposure; it does not say a *sequential* policy beats a *static* one, which is what (b) asks and
what the paper's framing depends on. `docs/results/stage4b_confounds_removed.json` further shows
that the state dependence a learned predictor would need is absent on 5 of 6 graphs measured, so (b)
should be attempted with the training-free closed form as the primary candidate and a learned
component as the ablation, not the reverse.

**Gate 2 is also open** and its control does not exist yet: a quality–cost Pareto curve including a
**random-pruning** control, to show the learned/screening component rather than the parallelism is
responsible for any gain. `random-pruning` is not implemented in this repository.

## 2026-09-18 — The proposed Go/No-Go criterion is not evaluable as written

The criterion under discussion — "quality loss ≤ 1% **and** ≥ 30% fewer online cascades" — must not
be frozen in that form. It cannot be evaluated in the regime the paper is about.

At `|S|/n ≥ 20%` on Congress-Twitter the measured marginals are `+0.70` target nodes with a paired
standard error of `0.05`–`0.39` (`docs/results/signflip_fixedmodel_mc300.json`). One percent of that
marginal is `0.007` target nodes — one to two orders of magnitude **below** the error of measuring
it. A method could lose 1% of quality, pass the gate, and be indistinguishable from one that lost
20%; equally, a genuinely equivalent method could fail it on noise alone.

**Decision:** freeze the criterion as an **absolute tolerance in target-count units with a paired
confidence interval**, a stated failure fraction, wall-clock runtime, and a break-even `Q` for
`C_offline + Q·C_online`. The relative form may be *reported* alongside, but it must not be the
pass/fail rule. This is consistent with the 2026-09-15 entry's own "evidence requirements", which
already asked for an absolute tolerance near zero; the relative phrasing crept in afterwards.

Corollary for reporting: because the marginals are near zero in the regime of interest, every
quality claim there needs its paired standard error printed next to it. A bare mean is not
interpretable, and this project has already published one table of bare means that turned out to be
Monte-Carlo artefact (`docs/WITHDRAWN_RESULTS.md` W8).
