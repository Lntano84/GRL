# GRL Decision Log

## 2026-09-02 — Use marginal gain as the primary learned target
**Decision:** Center the learning component on conditional marginal gain Δ(v|S), rather than direct final-set/solution prediction.

**Rationale:** Existing validation indicates marginal gain is more learnable and maps naturally to sequential influence-maximization decisions.

**Implication:** Future model evaluation must include conditional ranking quality and downstream decision quality.

## 2026-09-02 — Develop toward a learning-augmented certified oracle
**Decision:** Treat the learned predictor as part of an oracle/decision system with uncertainty, trust, certification, or fallback behavior rather than as a standalone regressor.

**Rationale:** This gives a clearer algorithmic contribution and connects predictive accuracy to reliable influence-maximization decisions and computational savings.

**Alternatives not used as the main route:**
- Direct solution/set prediction as the primary model target.
- Purely adding a generic RL module without first establishing reliable marginal-gain estimation and decision benefit.

## 2026-09-02 — Make project state independent of a single Codex session
**Decision:** Persist research state in `AGENTS.md` and `docs/{RESEARCH_STATE,DECISIONS,EXPERIMENT_LOG,NEXT_STEPS}.md`.

**Rationale:** Multiple computers and ChatGPT/Codex entry points should be able to resume from repository state without relying on local chat history.

---

## 2026-09-15 — HUMAN AUTHORIZATION: change setting from static IC to overexposure

**Decision (human-authorized):** Authorize the setting change required by the 2026-09-05
`P0 paper-positioning gate`. The target becomes **DASFAA 2027 (deadline 2026-11-25, LNCS 16 pages,
double-blind)**, and the evaluation setting changes from **vanilla static IC** to
**overexposure-aware diffusion (threshold-window model)**.

**Rationale:** The 2026-09-05 record listed three candidate reframings. The overexposure setting
satisfies **all three at once**:

1. *Expensive / non-standard black-box influence oracle* — RR/RIS is unusable under overexposure
   (per-node activation probability is non-monotone in the active-neighbour set, so the coverage
   identity `σ(S) = n·Pr[S∩R≠∅]` has no valid premise). The only ground-truth oracle is
   state-tracking Monte-Carlo, which is genuinely expensive.
2. *Repeated-query / changing-state setting where learned representations amortize* — overexposure
   makes exposure state explicit and time-dependent, so per-step conditioning is intrinsic rather
   than artificial.
3. *Safe fallible-advice methodology with a clearly motivated setting* — with **no** reliable
   classical fallback (greedy's guarantee degrades to `γ/k`; no learning-based method exists in
   this setting), audit + progressive verification + fallback stops being defensive decoration and
   becomes the algorithmic core.

**Implication — the following are now UNFROZEN** (the 2026-09-05 gate had frozen them):
- multi-graph evaluation (≥3 graphs, including 2–3 standard IM graphs);
- `k = {5, 10, 20}` budget sweeps;
- broad ablations (learned-only / audit-only / no-fallback / state-awareness on-off);
- a new IM setting (overexposure threshold-window model).

**Still frozen / still not authorized:** adding RL, adding a new predictor architecture, or
locally tuning runtime to hide a structural comparison.

**Deliberately abandoned claim:** "learned advice reduces runtime/quality vs mature RIS on static IC".
The 2026-09-05 OPIM-C comparison (quality 500.940 vs 502.116; ~24.984 s vs ~0.602 s) makes that
claim untenable and it must not be resurrected.

**New primary claim to be earned:** in a setting where no certified classical method exists,
learned marginal advice with **auditing and fallback** preserves near-oracle quality while reducing
expensive oracle computation, and degrades gracefully as advice quality collapses.

**Evidence requirements before the DASFAA submission:**
- Gate 1 (by 2026-10-13): (a) Spearman correlation between true marginal gain and degree under
  overexposure collapses (≈0 or negative); (b) sequential decision beats the static optimum by a
  material margin (>5%). **If either fails, DASFAA is dropped and the target moves to CIKM 2027.**
- Gate 2: quality–cost Pareto curve including a **random-pruning control**, to show the learned
  component rather than the parallelism/screening is responsible for any gain.

**Record of the merge:** this decision was taken after consolidating `GaoYucen/GRL` and
`Lntano84/GRL` into a single repository (see `MERGE_NOTES.md`).
