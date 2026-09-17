# DASFAA 2027 draft — status

> Updated 2026-09-17. Target: **submission 2026-11-25** (69 days from the date above).
> Working title: *When Reverse Reachability Has No Domain: Influence Maximization under
> Threshold-Dependent Overexposure*.

---

## What is written

| file | state | notes |
| --- | --- | --- |
| `sections/abstract.tex` | **complete draft** | all numbers from `notes/CLAIMS.md` |
| `sections/introduction.tex` | **complete draft** | includes the two mistakes we made and corrected |
| `sections/preliminaries.tex` | **complete draft** | model, two observations that drive everything |
| `sections/inapplicability.tex` | **complete draft** | Props. 1–2, Cor. 1, Table 1 — the core contribution |
| `sections/regime.tex` | **complete draft** | Table 2, the seed-fraction claim |
| `sections/analysis.tex` | **complete draft** | the closed form, Table 3 (calibration) |
| `sections/experiments.tex` | **complete draft** | Tables 4–7, incl. the SNR caveat |
| `sections/limitations.tex` | **complete draft** | 6 numbered threats, incl. the negative result and the surrogate gap (Table 8) |
| `sections/conclusion.tex` | **complete draft** | |
| `sections/related_work.tex` | **complete draft** | citations need verification (see below) |
| `references.bib` | **needs verification** | 13 entries; 8 carry `TODO(verify)` |
| `paper.tex` | complete | LNCS class; needs `llncs.cls` to build |

Rough length: the drafted sections are estimated at **11–13 LNCS pages** with the seven tables,
inside the 16-page budget. This is an estimate from word count, not a compiled measurement.

---

## What is NOT done

### Blocking

1. **The paper does not compile here.** No LaTeX toolchain on this machine and
   `paper/iclr2027/tools/tectonic` is absent from the working copy. `build.sh` documents the
   command; `llncs.cls` and `splncs04.bst` must be downloaded from Springer.
2. **Citations are unverified.** 8 of 13 entries carry `TODO(verify)`. Do not submit with a
   guessed venue or page range.
3. **DASFAA requirements are partially unverified.** See `notes/REQUIREMENTS.md` — the page
   limit, template version and submission portal all need confirming on the official site.

### Content gaps that a reviewer will notice

4. **Three of the four datasets of the source model are not covered.** We now include **Trust
   Bitcoin-Alpha** (parsed by `scripts/data/convert_bitcoin_alpha.py`; it reproduces the source's
   `|V| = 3783` exactly and behaves as the other graphs do --- `ρ_degree` flips from $+0.897$ at
   0\% to $-0.679$ at saturation). Still missing: **Occupywallstnyc** (not on SNAP; needs another
   source) and **Congress-Twitter / Wiki-Vote in the source's exact versions** --- we use the raw
   Wiki-Vote (7,115 nodes) rather than its 889-node version, and the source's Congress-Twitter
   `|V| = 333` versus our 475. A benchmark section that omits the source model's own datasets is
   a real weakness; it is now one dataset rather than three.
5. **No comparison against the source model's own algorithms** (incremental greedy, the
   upper-bound method) or against `Max_Degree` / `IMRank` / `PageRank` / `CELF` as it reports
   them. Our baselines are degree, exact MC-greedy and our own closed form.
6. ~~**The surrogate gap `λ(S) − σ(S)` is not measured**~~ — **done**, see Table 8 and
   `docs/SURROGATE_GAP_FINDING.md`. It is now a result plus an open question: the bound is tight
   at low coverage (9.8%) but inverts under our reading once saturated, and the per-realisation
   containment its proof invokes holds in 0/180 draws. **Before submitting, get the authors'
   reading of `σ^κ`/`σ^τ`** — the check is cheap and being wrong is expensive.
7. **Only one graph is used for the main comparison** (Congress-Twitter, Table 4). Table 2
   covers 8 graphs but only for rank correlation.
8. **No figures.** All results are tables. At least one figure showing the sign flip against
   `|S|/n` and one showing regret vs.\ fraction would help.

### Known soft spots in the argument

9. §4's claim that RR sets are empty is proven for the **non-seed source**. A reviewer may ask
    about seeds: since seeds are positive by definition they could serve as a base case. Our
    response is that the identity then reduces to `Pr[S ∩ R ≠ ∅]` counting only seeds, which
    carries no information about reachability — this is stated but deserves an explicit
    lemma/remark rather than a sentence.
10. `δ2` is a **heuristic** with no approximation guarantee. The paper's honest position is
    empirical. A reviewer may ask for a bound; we have none.

---

## Suggested order of work, 69 days

| window | task | why first |
| --- | --- | --- |
| days 1–3 | obtain `llncs.cls`, compile, verify requirements | everything downstream depends on knowing the real page budget |
| days 3–7 | verify all citations | cheap, and a wrong reference is a desk-reject risk |
| **days 3–10** | **resolve the `σ^κ`/`σ^τ` reading with the authors** | the surrogate result depends on it; ask before publishing it |
| days 7–20 | parse Trust Bitcoin-Alpha; find Occupywallstnyc; add both | closes the biggest credibility gap (#4) |
| days 14–28 | implement the source model's incremental greedy + upper bound as baselines | reviewers will ask why the source model is not compared |
| days 28–45 | figures; extend Table 4 to 3 graphs | presentation |
| days 45–60 | full rewrite pass for LNCS length; anonymity check | |
| days 60–69 | buffer | |

---

## Honesty constraints carried from the analysis

These are not style preferences; each is a place where being wrong is easy.

* **Report `|S|/n` next to every `|S|`.** Our own two major errors came from ignoring it.
* **Never report a rank correlation measured below `MC = 300`** in the negative-marginal regime.
* **Never claim density causes the sign flip.** 8 graphs, `ρ = −0.405`, one counterexample.
* **Never claim degree is universally harmful.** NetHEPT wins at `|S|/n = 20%`.
* **Never present the learned-scorer result as anything but a limitation at 480 states.**
