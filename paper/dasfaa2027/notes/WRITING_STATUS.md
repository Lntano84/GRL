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

### Resolved since the first draft

* ~~**The paper does not compile.**~~ **It compiles.** Toolchain installed on 2026-09-17:
  Tectonic 0.17.0 from `C:\Users\windows\tools\latex\tectonic\tectonic.exe`, plus `llncs.cls`
  and `splncs04.bst` copied from the CTAN bundle into `src/dasfaa2027/`. **Current build: 18
  pages, 0 undefined citations, 0 overfull hboxes.** `build.sh` documents the command.
* ~~**No baselines from the source model.**~~ **Done.** `evaluate_source_model_baselines.py`
  implements \textsc{IGA} (greedy on the objective) and two readings of the upper-bound method
  (\textsc{UB}-$\lambda$ on the surrogate, \textsc{UB}-$\kappa$ on $\sigma^\kappa$ alone). Result
  (Table 6): \textsc{IGA} spends **13,875 cascades per query** to reach a mean marginal of
  $+132.21$ against the closed form's $+133.20$ at zero cost, and is *negative* on
  Congress-Twitter at $|S|/n = 10\%$. The \textsc{UB} arms are the weakest policies tested
  because the surrogate saturates after one seed.
* ~~**The surrogate gap is unmeasured.**~~ **Done**, see Table 8 and
  `docs/SURROGATE_GAP_FINDING.md`. Now a result plus an open question.
* ~~**Trust Bitcoin-Alpha not parsed.**~~ **Done** — ninth graph, reproduces the source's
  `|V| = 3783` exactly.

* ~~**The datasets' other baselines are missing.**~~ **Done.** `src/grl/baselines/classic_im.py`
  implements \texttt{Random}, \texttt{Max\_Degree}, \textsc{IMRank}, \texttt{PageRank} and
  \textsc{CELF}, and all nine policies are compared in Table 6. Result: \textsc{IGA} **and** its
  exact variant \textsc{CELF} both spend $13{,}875$ cascades per query to land at or below the
  zero-cost closed form, and both go negative on Congress-Twitter at $|S|/n = 10\%$. In the
  saturated subset $\delta_2$ averages $+8.13$ against $+4.66$ and $+4.74$ for the two greedies.
  \textsc{CELF} is implemented as the **exact** greedy, not a lazy one, because the lazy
  skip-test needs monotone submodularity and this objective has neither; that methodological point
  is now made in the related work.

### Still blocking

1. **The paper is 19 pages against a 16-page limit.** Needs ~3 pages of trimming. Cheapest
   candidates, in order:
   (a) move the SNR caveat table (Table 3) to an appendix --- it is a methodological warning, not
   a result;
   (b) merge Table 3 into Table 4 --- both tell the "calibration fails, ranking survives" story;
   (c) tighten the related-work paragraphs, especially the CELF one;
   (d) drop Table 7 (the NetHEPT counterexample) and state it in a sentence.
   ⚠️ First confirm whether references count toward the limit — see `notes/REQUIREMENTS.md`.
2. **Citations are unverified.** 9 of 14 entries carry `TODO(verify)`. Do not submit with a
   guessed venue or page range.
3. **DASFAA requirements are partially unverified** — page limit, template version, portal.

### Content gaps that a reviewer will notice

4. **Two of the four datasets of the source model are still missing or approximate.** We include
   **Trust Bitcoin-Alpha** exactly (`|V| = 3783` matches). Still missing: **Occupywallstnyc** (not
   on SNAP; needs another source). Still approximate: the source uses Congress-Twitter with
   `|V| = 333` and Wiki-Vote with `|V| = 889`, whereas we use 475 and 7,115. Documenting the
   version difference is the minimum; matching them would be better.
5. **Only three graphs appear in the main comparison** (Table 6). Table 2 covers nine graphs but
   only for rank correlation.
6. **No figures.** All results are tables. One figure showing the sign flip against `|S|/n`, and
   one showing the marginal against fraction for the baseline table, would help a lot.

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
