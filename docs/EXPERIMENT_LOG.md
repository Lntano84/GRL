# GRL Experiment Log

## 2026-10-13 — Framework mechanism manuscript unit

GRID61方法/机制稿与三面板科学图完成。宏平均FRAME控制减少13.06%、查询减少69.37%，对FIRST_SAFE_BEST控制少6.06%；去局部补全虽控制减少14.25%，最坏成本高FULL4.81%，FRAME0.372%。九月普通停止成本优于FRAME0.729% FULL、共同失败与额外补查开销保留。图形首次因bundled Python缺matplotlib中断，仅改用既有项目环境，已有数值报告hash不变；拥挤标签旧图与标签修订图分开归档。未新增回放/训练，原32条实际运行wall2975.99s、审计63616步/70358匹配反馈/16重复路径。

## 2026-10-13 — User-facing grid data review

应用户要求整理当前对比数据与解释，导出六页XLSX、GRID55B明确来源的128条逐次成绩、GRID59全部16条成绩、原始CSV/JSON及哈希数据包。来源摘要hash、逐次总账与全部汇总行核对通过，工作簿公式输入变更后重算并恢复，关键导出缓存值与来源一致；未声称原生Excel执行。说明包含约70%N1查询/15.75%宏平均控制/13.92%汇总控制、0.08045%成本尾部、72ms单步p95以及强频次基线增量不足。仅数据交付，无新回放、拟合或科学裁决变更。

## 2026-10-13 — GRID59 post-audit evidence update

只读复算16条GRID59回放的查询/控制比例与8条GNN—STATIC差值分解，无新采集或拟合。两种子平均查询少2.901%，控制平均多0.570%；六月共同1901步失败，完整周成本只比较三周。九月查询少但控制路径和成本不同，图/先验/前向约0.039秒不能单独解释搜索外耗时增加。贡献表、同批差异CSV、来源哈希及收尾记录归档GRID59_closeout；原历史证据不覆盖。

This file is a concise index of experiments worth remembering. Raw logs, checkpoints, copied worktrees, and large outputs remain in their project-local result directories.

Power-grid archive snapshots use `.gitattributes` with `-text` to preserve original bytes across checkouts. The original-file SHA-256 manifests therefore remain verifiable on other operating systems; Git's automatic line-ending normalization is disabled for this archive. This is an artifact transport fix, not an experiment rerun or result change.

## 2026-10-10 — GRID22/23 complete deployment comparisons

[Reports, source snapshots, selected tables and owned model weights](../research_archive/power-grid-2026/README.md) cover GRID00–23. The actual included files have an independent SHA-256 archive manifest; large traces/data and external PPO artifacts stay local.

GRID22: 28 runs, 53,828 physical steps, 139,770 public forecasts. K32 GNN seed0 completed 3/4 weeks; seed1 2/4, losing September at181 steps. GRID23: 24 runs, 47,712 steps, 134,946 forecasts; same weights, shared exact canonical tie handling and K128. Both GNN seeds now complete3/4, matching FULL survival on these weeks. Both fail the quality screen: March cost +8.073%; complete-week mean increases2.691%/3.070%. Public candidate queries decrease82.530%/81.937%. Graph cost gains against PPO/MLP/bias are only0.024%/0.031%/0.015% and fail the fixed increment threshold.

Both saved-data audits pass; a separate reconciliation checks1,217 sealed-file hashes and recomputes comparison arithmetic. No independent AC rerun or new fitting/RL. Earlier GRID22 interruption preserves14 completed runs and archives305 readable partial steps, at least4,253 forecasts and unknown unlogged work. Its costs are not zero and are separate from successful trajectory totals.

This is development evidence on four exposed weeks, not final confirmation. The model's training loss uses next-step risk; the actual controller additionally optimizes continuous actions and evaluates long-term cost. A tiny missed risk optimum can precede a much larger trajectory cost difference, but its causal mechanism remains unproved. Wall-time results have non-search variation and are not attributed wholesale to the GNN.

## 2026-09-18 — Current-state override after model correction

This entry is the current interpretation of all later evidence. Entries below that predate the
state-machine correction are historical and must not be quoted as current measurements.

**Correction.** The old implementation froze a positive node and rejected the literal boundary
`delta=tau=1`. The corrected model re-evaluates non-seed nodes and counts only final positives.

**Completed evidence.** `docs/results/signflip_fixedmodel_mc300.json` now contains 8 graphs and 24
unique graph/fraction cells at MC=300 with paired per-trial estimation. In saturated cells,
degree is negative on 7/8 graphs and `delta2` is the better ranker on 8/8. The old 8--35% negative
marginal-share claim is withdrawn; the corrected mean is 0.5% and the worst cell is 4.0%.

**Current limitation.** Corrected Stage 4b has only one usable state-dependence/headroom cell
(Congress-Twitter, k=1). No post-fix multi-graph sample-efficiency or quality--cost claim is yet
confirmed. The next experiment must re-run sequential selection, Full-MC, degree, `delta2`, and
random-pruning under the frozen contract.

**Artifact rule.** A result is current only when its JSON, generating script, model contract, MC
budget, random seeds, and status are all recorded. The current state summary is maintained in
`docs/RESEARCH_STATE.md`; this log retains historical provenance but does not override it.

## 2026-09-18 — DASFAA page-limit rebuild

The anonymous LNCS draft was rebuilt after removing the pre-fix tables. Tightening the conclusion,
removing unresolved bibliography TODO notes, and keeping the references complete reduced the PDF
from 17 to **16 pages**. This was an editorial/format gate, not a new algorithmic result; the
post-fix quality--cost evidence remains pending.

## 2026-09-01 to 2026-09-02 — Marginal-gain predictability validation
**Goal:** Determine whether the model genuinely learns conditional marginal gain Δ(v|S), especially candidate ranking under a fixed seed set/state.

**Current retained conclusion:** Marginal-gain prediction is sufficiently promising to remain the core learning target, but strict unseen-state validation and state-conditioning diagnostics remain important before treating the predictor as a reliable oracle.

**Related work:** NetHEPT marginal-gain predictability tests and strict state-conditioning checks documented in the GRL Notion workspace.

## 2026-09-02 — ICLR 2027 route pre-experiments
**Goal:** Test components around a learning-augmented certified marginal-gain oracle, including uncertainty/trust logic and ensemble-style pretests.

**Status:** Active / exploratory. Multiple project-local pre-experiment directories exist under `/workspace/GRL`. Exact metrics should be promoted into this log only after the corresponding result/config artifact has been checked.

## 2026-09-03 — Strict state-conditioning diagnostics
**Goal:** Determine whether high candidate-ranking accuracy actually reflects dependence on the current seed set S.

**Verified finding:** Strict unseen-state ranking is strong, but zero/shuffled seed-mask and controlled-overlap diagnostics show that the original predictor relies heavily on candidate-intrinsic strength. State-sensitive difficult-state supervision plus same-candidate delta loss restores strong cross-state sensitivity without destroying ordinary candidate ranking. A candidate-conditioned residual variant improves state-response amplitude further, but candidate-specific drop-magnitude calibration remains unresolved.

**Decision:** Do not spend the current phase on small predictor-metric tuning. Keep state-aware supervision as the default learning direction and move to end-to-end sequential IM evaluation.

## 2026-09-03 — First end-to-end learning-augmented IM prototype
**Protocol:** NetHEPT; shared fixed candidate pool of 128 nodes; seed budget 10; selection MC 40; final spread MC 1000. Full-MC, learned-only, and selective methods use the same candidate pool. Selective baseline scores all candidates with the learned state-aware predictor and exactly refines only the learned Top-M candidates at each step.

**Results:**
- Full-MC greedy: spread 444.911; 1,235 exact candidate evaluations; 49,400 MC candidate samples.
- Learned strict: spread 325.880 (73.25% of Full-MC); 0 exact candidate evaluations.
- Learned state-aware: spread 321.526 (72.27%); 0 exact candidate evaluations.
- Selective Top-8: spread 412.186 (92.64%); 80 exact candidate evaluations = 6.48% of Full-MC.
- Selective Top-16: spread 414.101 (93.08%); 160 exact candidate evaluations = 12.96% of Full-MC.
- Selective Top-32: spread 422.776 (95.02%); 320 exact candidate evaluations = 25.91% of Full-MC.
- Degree baseline: spread 299.100 (67.23%); Degree Discount: 302.340 (67.96%).

**Interpretation:** This is a usable framework/result anchor, not the final paper result. Selective correction materially closes the quality gap while using much fewer exact candidate evaluations. The next algorithmic priority is to improve learned shortlist recall and replace fixed Top-M with adaptive trust/certification; full-graph/RIS scaling comes after that.

**Compact artifact:** `docs/results/nethept_end_to_end_20260903.json`.


## 2026-09-03 — Sequential shortlist and adaptive certification
**Protocol:** NetHEPT, 128-candidate shared pool, budget 10, MC40 selection oracle, MC1000 final spread evaluation.

**Sequential shortlist diagnostic:** true Full-MC winner ranks under the state-aware predictor are `[1, 57, 87, 2, 2, 42, 1, 8, 1, 32]` (Top-32 recall 0.70; Top-64 recall 0.90). This explains the fixed Top-M quality ceiling.

**Adaptive result:** residual-envelope adaptive refinement with `beta=0.5` obtains spread **443.626** versus Full-MC **444.911** (ratio **0.9971**) using **512** exact candidate evaluations versus **1235** (fraction **0.4146**). With per-step MC-world reuse it uses 400 live-edge worlds and takes 32.4s in the current NetworkX prototype.

**Interpretation:** the main predictor metric for the next phase is sequential hard-state Top-K recall, and the main algorithmic direction is adaptive certification/fallback rather than a fixed shortlist. See `docs/results/nethept_adaptive_certification_20260903.json`.

## 2026-09-03 — Two-level adaptive certification with progressive MC

Starting from the validated adaptive shortlist rule, we separated two decisions: (1) how many candidates need exact verification, and (2) how many common-random-number MC worlds are needed inside the current shortlist. The final implementation uses progressive MC budgets 5→10→20→40 and caches candidate/world gains across rounds.

On NetHEPT (128-node candidate pool, budget=10, final spread MC=1000), the recommended configuration is residual_beta=0.5, confidence_z=0.5, bootstrap_mc=10. It selects the same seeds and obtains the same measured spread as the fixed adaptive beta=0.5 baseline: 443.626 (99.71% of Full-MC 444.911). Exact candidate evaluations are 504 versus 1235 for Full-MC; MC candidate-samples are 18,280 versus 49,400 for Full-MC and 20,480 for fixed adaptive. Selection time is 23.59s in this implementation. The per-step verified shortlist sizes are [16,16,16,72,64,112,96,80,16,16], with final MC budgets [40,10,10,20,40,20,5,40,20,40].

Bootstrap ablation 10/20/40 produced identical selected seeds and spread. Bootstrap=10 was best on compute: 18,280 samples and 320 generated live-edge worlds, versus 18,920/360 for bootstrap=20 and 18,920/400 for bootstrap=40.

## 2026-09-04 — Online trust gate repairs robustness failure

A controlled predictor-corruption stress test exposed a failure of the residual-envelope-only heuristic: when learned scores were fully randomized (clean-vs-corrupted Spearman approximately 0), the ungated method became falsely confident, used only 224 exact candidate evaluations / 5,560 MC candidate-samples, and final spread fell to 369.541 (83.06% of the Full-MC reference).

We therefore added an online trust audit. At every greedy step it evaluates the predicted Top-16 plus eight rank-spaced sentinel candidates using MC20. The predictor is trusted only if audit learned-vs-exact Spearman exceeds a threshold and no sentinel candidate unexpectedly beats the audited head. Untrusted steps ignore learned ranking and fall back to Full-MC40 over all available candidates.

On NetHEPT (128 candidates, budget=10), tau=0.3 gives the best first-pass consistency/robustness tradeoff. Clean predictor: spread 443.591 (99.70% of Full-MC), 802 exact candidate evaluations, 31,440 MC candidate-samples, 3/10 fallback steps. Moderate corruption alpha=0.5: spread 446.189 under the finite evaluation protocol, 769 exact evaluations, 30,060 samples, 5/10 fallbacks. Strong corruption alpha=0.75: spread 444.911, 1,044 exact evaluations, 41,480 samples, 8/10 fallbacks. Fully randomized predictor alpha=1.0: all 10 steps fall back, exactly recovering the Full-MC reference cost (1,235 candidates / 49,400 samples) and measured spread 444.911.

Values slightly above the Full-MC reference at intermediate corruption levels are finite-MC / trajectory effects and must not be interpreted as outperforming exact greedy. The important result is graceful fallback: as prediction trust collapses, the method approaches classical Full-MC behavior instead of becoming cheaper and catastrophically worse.

## 2026-09-04 — Multi-seed trust calibration: robustness trend survives independent randomness

**Protocol:** NetHEPT; shared 128-candidate pool; budget 10; five independent repeats. Predictor corruption, MC-oracle worlds, and final-spread evaluation use separate repeat-specific random seeds. Each repeat has its own same-seed Full-MC40 reference. The frozen trust-progressive configuration is `tau=0.3`, audit MC20, Top-16 audit head, four rank-spaced sentinels, and the validated progressive 5→10→20→40 fast path.

**Aggregate results:**
- Full-MC reference spread: **445.154 ± 3.154**, 49,400 MC candidate-samples.
- Clean predictor (`alpha=0`): mean quality ratio **1.0005 ± 0.0082**, mean **26,004 samples = 52.6%** of Full-MC, **4.4 ± 1.5 fallback steps**.
- Moderate corruption (`alpha=0.5`, predictor Spearman ≈0.624): quality ratio **1.0030 ± 0.0093**, **24,542 samples = 49.7%**, **4.0 ± 1.6 fallbacks**.
- Strong corruption (`alpha=0.75`, predictor Spearman ≈0.248): quality ratio **1.0019 ± 0.0045**, **41,171 samples = 83.3%**, **7.6 ± 0.5 fallbacks**.
- Randomized predictor (`alpha=1`, predictor Spearman ≈−0.040): quality ratio **0.9998 ± 0.0005**, **47,161 samples = 95.5%**, **9.4 ± 0.9 fallbacks**.

**Interpretation:** the desired learning-augmented behavior is now visible across independent randomness: solution quality remains essentially at the Full-MC reference while oracle effort rises sharply as predictor quality collapses. A randomized predictor no longer causes the earlier catastrophic 83.06% quality failure; the solver approaches classical Full-MC cost automatically. It is not necessary for every random-predictor step to trigger literal full fallback, because trusted progressive verification can occasionally certify a decision safely.

**Current bottleneck:** clean-case **efficiency variance / false distrust**, not robustness. Across the five clean repeats, fallback count ranges from 2/10 to 6/10 and sample fraction from 41.4% to 64.2%. Therefore the next step is statistical/held-out calibration of the audit trust score, not another hand-tuned threshold sweep.

**Caveat:** quality ratios slightly above 1 are finite-MC / trajectory effects, not evidence of outperforming Full-MC greedy. This remains a 128-candidate prototype; full-graph/RIS and multi-dataset evidence are still required.

**Compact artifact:** `docs/results/nethept_trust_calibration_multiseed_20260904.json`.

## 2026-09-04 — Audited residual gate replaces Spearman trust score

Held-out calibration showed that audit Spearman is not a reliable trust statistic: unsafe steps can have higher local learned-vs-exact Spearman than safe steps, and the conservative calibrated threshold drove clean-case fallback close to Full-MC cost. We therefore replaced the Spearman gate with an **audited residual gate**. The audit evaluates the learned head plus rank-spaced sentinel candidates, estimates the upper tail of `exact - learned` residuals, and trusts the learned ranking only when the audited best head candidate dominates the learned best outsider after adding this audited residual upper bound; sentinel surprise still forces distrust.

The frozen prototype configuration is `residual_q=1.0`, `residual_beta=0`, audit Top-16 + 8 sentinels, audit MC20, followed by the validated progressive 5→10→20→40 fast path. Extra residual safety margins (`beta=0.5/1.0`) were too conservative and rapidly approached Full-MC cost.

Five independent NetHEPT repeats (128-candidate shared pool, budget 10) give:
- Clean predictor: quality ratio **1.0070 ± 0.0070**, **20,715 ± 1,997 samples = 41.9% ± 4.0%** of Full-MC, **0.8 ± 0.45 fallbacks**.
- `alpha=0.5`: quality ratio **1.0079 ± 0.0063**, **18,944 ± 3,756 samples = 38.3% ± 7.6%**, **1.2 ± 1.3 fallbacks**.
- `alpha=0.75`: quality ratio **1.0006 ± 0.0145**, **28,621 ± 5,010 samples = 57.9% ± 10.1%**, **3.8 ± 1.3 fallbacks**.
- Random predictor `alpha=1`: quality ratio **1.0042 ± 0.0051**, **41,893 ± 3,821 samples = 84.8% ± 7.7%**, **7.8 ± 1.3 fallbacks**.

Ratios slightly above 1 are finite-MC/trajectory noise and are not interpreted as outperforming Full-MC greedy. The important result is the cost response: useful predictions retain a large oracle saving, while degraded/random predictions automatically consume much more oracle work and preserve solution quality. The earlier Spearman gate is now retained as a failed/diagnostic ablation rather than the main method.

**Compact artifact:** `docs/results/nethept_audited_residual_multiseed_20260904.json`.

## 2026-09-04 — Candidate-scale stress test exposes proposal + certificate scaling failures

The audited-residual gate was next tested beyond the original 128-candidate prototype. This test is intentionally diagnostic rather than a paper-ready scalability claim.

### Fixed audit rule does not scale
Using the frozen 128-pool rule (Top-16 + 8 rank-spaced sentinels, audit MC20, residual q=1, beta=0):
- Pool 256, two repeats: clean quality ratio **0.9895 ± 0.0047** at **36.6% ± 10.4%** of Full-MC samples; alpha=.75 remains ~1.0002 at 64.5% cost; random remains ~0.9971 at 82.6% cost.
- Pool 512, one repeat: clean **0.9173** quality at 35.0% cost; alpha=.75 **0.8857** at 20.4% cost; random **0.9591** at 64.5% cost.

Thus the 128-candidate success must not be extrapolated to larger candidate sets.

### Simply enlarging the audit head is not a fix
A separate scale-aware audit-budget pilot tested larger heads/sentinel counts. It failed structurally:
- pool256 Top32+8: clean ratio **0.9804**, random **0.9547**.
- pool512 Top32+16: clean **0.8916**, random **0.9803**.
- pool512 Top64+16: clean **0.9110**, random **0.9299**.

The current certificate compares the audited best against the first learned outsider after the head. Enlarging the head moves this outsider deeper to a lower learned score and can therefore make the certificate *easier* to satisfy. Audit size tuning alone is not a principled solution.

### Clean proposal ranking also degrades with candidate count
Along exact Full-MC trajectories, the learned rank of the true winner changes sharply with pool size:
- pool128: ranks `[1,57,87,2,2,42,1,8,1,32]`, mean **23.3**, max **87**, Top64 recall **0.9**.
- pool256: `[1,70,79,141,2,47,2,9,2,46]`, mean **39.9**, max **141**, Top64 recall **0.7**.
- pool512: `[1,144,81,173,2,48,13,160,11,83]`, mean **71.6**, max **173**, Top64 recall **0.5**.

Several large-pool hard states have learned-top1 regret above 40–50 marginal-influence units. Therefore the scaling failure is **two coupled problems**: (1) the learned proposal itself becomes weaker on large candidate sets, and (2) the current audited residual certificate lacks population-aware coverage of the unseen outsider set.

**Decision:** stop manual Top-K/sentinel tuning. The next P0 is large-candidate sequential hard-negative training for the proposal, followed separately by a population-aware statistical certificate whose confidence becomes stricter as the number of unseen outsiders grows.

## 2026-09-05 — Full-graph RR screening + 128 shortlist 闭环通过 {#FULLGRAPH_RR_SCREENING_20260905}

当前主线从“人为 Top-degree 128”升级为完整图输入：`NetHEPT 15,233 nodes -> independent RR screening -> Top-128 shortlist -> state-aware marginal proposal -> audited residual + progressive verification -> Full-MC fallback`。

3 个独立 repeat（screening RR 与 full-graph RR baseline 使用独立 50k RR samples）结果：
- independent RR-greedy seed recall in Top-128: **1.0000 ± 0.0000**；
- full-graph independent RR-sketch greedy spread: **506.7407 ± 3.5330**；
- screened Full-MC40 spread: **508.4677 ± 1.9645**；
- audited/progressive spread: **500.9403 ± 2.4410**；
- audited / screened Full-MC quality: **0.9852 ± 0.0035**；
- audited / full-graph RR-sketch quality: **0.9886 ± 0.0109**；
- expensive MC sample fraction: **0.5908 ± 0.0846**（约 40.9% reduction）；
- fallback: **4.0 ± 1.63 / 10 steps**；
- pairwise shortlist Jaccard: **0.6954 ± 0.0092**。

Interpretation: 128 不再是人为 toy pool，而是完整图上的 cheap screening budget。不同 RR 随机种子得到的 shortlist 并不完全相同，但独立 RR-greedy 的强 seed 在 3/3 repeats 中均被覆盖。当前可以认为 full-graph-input learning-augmented closure 已经跑通。

Caveat: 这里的 50k RR baseline 是 prototype RR-sketch greedy，不等同于正式 IMM/OPIM-C 等带理论采样量的 paper-grade RIS baseline；不能据此声称优于 RIS。下一步先验证 shortlist budget M={64,128,256}，再做 k={5,10,20} 和多图/正式 RIS baseline。


## 2026-09-05 P0 paper-value risk — official OPIM-C same-protocol comparison
Official OPIM-C (Tang et al., SIGMOD 2018; source `tangj90/OPIM`, commit `344cc3d5eaa13d8cdf9a9e75722e49d341981e8d`) is now running on the same NetHEPT IC data used by GRL. For `k=10`, `eps=0.1`, its returned original-node seed set is `[267, 6024, 47, 66, 1241, 3210, 37, 1434, 12464, 156]` (internal GRL indices `[1265, 8595, 308, 410, 3219, 5753, 228, 3519, 13507, 892]`).

Using exactly the three final-evaluation seeds from the existing full-graph closure (`1900401, 1901422, 1902443`) and `MC=1000`, OPIM-C obtains spread `504.017 / 501.763 / 500.568`, i.e. **502.116 ± 1.430**. The existing audited learning-augmented method is **500.940 ± 2.441**; full-graph independent 50k-RR greedy is **506.741 ± 3.533**; screened Full-MC40 is **508.468 ± 1.965**. Hence audited/OPIM-C = **0.99766**: quality is essentially the same at current Monte-Carlo resolution and must not be framed as an advantage over OPIM-C.

Runtime exposes a more serious paper-value issue. Across the three existing full-graph repeats, RR screening takes **0.482 ± 0.116 s** and audited sequential selection takes **24.502 ± 1.633 s**, for about **24.984 s** screen+selection. The official OPIM-C executable takes about **0.602 s** for optimization; its graph-format/reverse-graph step takes about **0.069 s** and successfully writes the reverse graph (the formatter returns code 1 despite the successful output). Thus the current prototype is roughly **37× slower** in optimization-path wall time while producing comparable spread.

**Interpretation:** the frozen method chain remains mechanically complete, but the paper-value closure is not. Before expanding to multi-graph, `k={5,10,20}`, or large ablations, we need a human decision on why learned advice is necessary relative to mature RIS. Plausible reframings are an expensive/nonstandard black-box influence oracle, a repeated-query/changing-state setting where learned representations can be amortized, or a safe fallible-advice methodology with a clearly motivated setting. Do not automatically add a new method module or locally tune runtime to hide this structural comparison.

Evidence is saved at `outputs/baselines/opim_c_nethept_k10_eps01/same_protocol_eval.json` on the 4090 workspace.

## 2026-09-05 P0-RESCUE — classical RR rescues runtime, learning value unresolved

A same-protocol rescue study replaced the old ~24.5 s sequential Monte-Carlo audit with independent RR verification inside the frozen full-graph-screened M=128 shortlist. Across three independent final-evaluation seeds, pure-RR verification gave: 1k RR = 335.657 ± 77.912 at 0.484 s; 2k = 461.309 ± 27.889 at 0.495 s; 5k = 453.223 ± 25.848 at 0.518 s; 10k = 486.321 ± 11.505 at 0.559 s; 20k = 502.446 ± 7.264 at 0.652 s; 50k = **506.588 ± 2.302 at 0.909 ± 0.037 s**.

The 50k verifier reaches **99.51% of official IMM spread** (IMM 509.081 ± 0.842 at ~0.287 s) and repairs the old GRL path (500.940 ± 2.441 at ~24.984 s), while 20k RR is close to the OPIM-C operating point (502.446 / 0.652 s versus 502.116 ± 1.430 / ~0.672 s). However, this rescue is entirely classical RR computation; 50k screening + 50k independent verification is not evidence of lower sampling cost than mature RIS.

The current tuned state-aware learned marginal predictor does not yet reduce the RR evidence needed for near-IMM quality. Learned-only remained poor (~324–333 spread in observed repeats), and hard learned-Top16 guidance was worse than pure RR at each completed low-budget pilot (1k: 378.812 vs 424.797; 2k: 384.294 vs 433.680; 5k: 374.293 vs 478.747; 10k: 402.223 vs 496.804).

**Paper-level judgment:** the engineering runtime bottleneck is largely solved, but the learning-value hypothesis in vanilla static IC is not. The current route cannot claim better quality, runtime, or demonstrated sample efficiency from learning relative to mature RIS. This is a **YELLOW / human paper-positioning review** boundary. Do not expand to multi-graph, k-sensitivity, broad ablations, a new predictor architecture, RL, or a new IM setting without human authorization.

## 2026-09-18 — Repository closure and reference audit

This was an editorial/artifact audit, not a new experiment. The corrected result JSON contains 24
unique graph/fraction cells; the source-rule documentation, state files, and claim ledger are
aligned; pre-fix withdrawn tables are absent from the submission PDF. The 12 references cited by
the manuscript were then checked against authoritative records. Five materially incorrect records
and their prose were corrected, an unsupported related-work claim was removed, and duplicate/unused
entries were deleted. The rebuilt anonymous PDF is 16 pages. Detailed provenance is in
`paper/dasfaa2027/notes/REFERENCE_AUDIT.md`.


## 2026-10-06 — Local result archive integration

将已定位本地实验报告、结果、方案、日志、代码快照与更正/作废记录整合为research_archive/decision-studies-2026。大文件使用同仓库Release及SHA-256对象清单；排除检查点、环境与第三方原始数据。没有运行新科研实验；这是归档核验，不是重做科学审计。

## 2026-10-10 — GRID24 quality-first safeguarded auxiliary search

[Report and archived sources](../research_archive/power-grid-2026/outputs/grid24/GRID24_report.md). Keep V1/PPO weights and K128, use author's existing rho_safe=0.9 to trigger cached full complement evaluation. 12 runs on four exposed development weeks: 23,856 physical steps, 89,064 public forecasts. All delivered action/observation/cost paths equal sealed GRID23 FULL; cost regret0 on joint-complete weeks, completion3/4 per method. Pooled search-query reduction34.0930%. All three rankers, including PPO, share the result. Preserved79 old-file hashes and625 historical archive files. Saved-data independent mask/alias/admission/choice/cost and model rank replay passed; no independent AC rerun. No new fits/RL/reserved-family evaluation. Two remaining workers ran concurrently; time is descriptive, not certified speed.

## 2026-10-10 — Controlled timing protocol and measurement repair

GRID25: four March pilot trajectories,8068 physical steps,48319 public forecasts, saved-data audit passed; original models and costs retained. Per-observation simulate restoration can create a bound-method reference cycle, so timings are diagnostic and remaining44 were not started. This does not establish the cause of runtime variation. GRID26 freezes48 serial runs (four development weeks/four methods/three balanced repeats) with a class-level timer.104 prior-state equivalence checks,67276 fake feedback dispatches and a direct lifecycle test pass; no new AC queries or fits in preflight. [Immutable protocol](../research_archive/power-grid-2026/protocols/GRID26_before_runs/README.md).

## 2026-10-10 — GRID26 actual serial control timing completed

48/48 runs,95424 physical steps,379368 public forecasts; saved-data audit and independent timing accounting pass. Same delivered action/observation/cost paths as sealed FULL; completion3/4 dates including same June failure. GNN0/GNN1 search savings19.55%/17.14%; total-control savings1.40%/0.88%; pooled query saving34.09%. FULL source fraction9.80%; repeat0 total slowdown. PPO search saving18.31%, total slowdown2.82% with large non-search variation; not a GNN-caused speed attribution.156 GNN/PPO call comparisons have equal choice and query count, but only96 identical queried sets and0 identical dispatch sequences. New fits/RL/reserved evaluations0. [Contribution report](../research_archive/power-grid-2026/outputs/grid26/GRID26_contribution_report.md).

## 2026-10-11 — GRID26 fallback review, no new simulation

Read104 saved call logs from repeat0/four exposed dates;26 distinct states and78 ranker/state rows.16/26 calls use full complement,accounting for91.41% of ranker queries;complement alone77.65%.All16 full pools lack admissiblerho<0.9 candidates.Thus fixedK128/guard cannot benefit in query count from a more accurate ranking on these states.14 full pools contain an admissible winner;shortlist coverageGNN0/GNN1/PPO=8/7/5,not an independent graph-increment confirmation.No new forecasts,rollouts,fits or hidden-family outcomes.Independent prefix-phase arithmetic agrees.[Review](../research_archive/power-grid-2026/outputs/grid26_review/FALLBACK_REVIEW.md).

## 2026-10-11 — GRID27 screen and GRID28 semantic completion qualification

156 legal feedback-replay rows per stage (26 exposed states x3 frozen rankers x2 fixed schedules).GRID27 plateau/fixed cutoff miss known March1733 action; no AC rollout.GRID28 adds only incumbent-substation completion,with aliases;GNN0/PPO PATIENCE64 now recover that action but still change2/26 decisions.78 real-runtime preflight checks and30074 fake feedback dispatches agree;no AC/refit in preflight.March protected/localGNN0/localPPO trial running under frozen design.One protectedFULL profile2017 physical steps exactly matches historical path;hierarchical optimizer solve share72.4%,N1search15.0% in that one instrumented March sample.Solver fine-statistics are partially captured,not zero for missing restoration calls.[Protocol snapshot](../research_archive/power-grid-2026/protocols/GRID28_before_results/README.md).

## 2026-10-11 — GRID28 March gate and GRID29 frozen qualification

GRID28 March3 runs and saved-data audit pass: all complete2017steps, localGNN0/localPPO cost+0.136116% versus protected, query saving54.15%/53.03%, single-repeat source saving46.22%/45.36%, controller saving4.87%/5.06%. No graph-specific speed attribution; remaining3 dates running. GRID29 generic DC-program cache design/code frozen before shadow/live outcomes, using only exposed states and no new weights. [GRID29 protocol](../research_archive/power-grid-2026/protocols/GRID29_before_results/README.md).

## 2026-10-11 — GRID28 frozen closed-loop matrix completed

GRID28完成12/12条开发回放，保存数据审计通过。GNN0查询节省50.90%、搜索时间减少43.18%、总控制减少8.83%；共同完成周成本平均增幅0.0454%、最大0.1361%。GNN0/PPO均完成3/4周，没有新增/提前失败；PPO同规则查询节省47.78%、搜索减少37.74%、控制减少3.45%。单次计时、开发数据及不同轨迹，不证明稳定端到端速度或GNN独有贡献；0.5%为投资筛查，不是等价。 Audited physical rows23856, source calls78, paid feedback29982; saved-data checks, no independent AC replay/refit. [Report](../research_archive/power-grid-2026/outputs/grid28/GRID28_accounting_report.md).

## 2026-10-11 — GRID29 V0 qualification blocked, V1 frozen

First real cached model rejected by is_dcp(dpp=True); no cached comparison/closed trajectory. Actual reactance is a Parameter. Preserve failed code/log and initial protocol; freeze V1 numeric reciprocal and numeric reactance-key invalidation plus real Parameter/triangle checks before rerun. [Revision protocol](../research_archive/power-grid-2026/protocols/GRID29_V1_before_results/README.md).

## 2026-10-11 — GRID29 V1 shadow qualification and live protocol

Six manual checks and130 stored exposed states pass, same convergence/finite flags and max flow difference0MW. Shadow time reduction39.43%, no new AC replay/refit; unified zero storage accumulator limits scope. Freeze4 original/cache protectedMarch live trajectories in balanced2-repeat order before runs. No whole-controller claim yet. [Snapshot](../research_archive/power-grid-2026/protocols/GRID29_PILOT_before_runs/README.md).

## 2026-10-11 — GRID29 cache pilot and GRID28 paired ranker review

GRID29工程数值资格130状态流量差0；4条三月闭环逐步动作/观测/成本相同，查询不变。两组控制时间减少10.93%/0.39%，汇总点估计6.19%，幅度不稳定，不能认定跨日期稳定整体提速或GNN贡献。V0非DPP/键缺陷已保留，V1修订与协议不覆盖历史。 Audit8068physicalrows/44sourcecalls/26236paidfeedback; no independentAC/refit. GRID28 matched26same-state/same-pool/same-choice calls, GNN query reduction5.98% vsPPO, directions7/4/15, common feedback6178/6178exact. Both branches4/4same delivered paths. GRID29B frozen before run for actual-memory isolated DC probes, not another total-time sweep.

## 2026-10-11 — GRID29B setup-name repair, no numerical outcome

profile.py shadowed stdlib profile during Torch/cProfile initialization and recursively entered the driver. Failure before control initialization, no delivered physical step/DC pair. Preserve original code/log/setup directory. Rename probe_dc.py only, refreeze before running; existing GRID29 result rows unchanged. Human directs continued diagnosis of10.93% versus0.39%, especially what worked in the10% pair.

## 2026-10-11 — GRID29B real-memory DC audit and GRID29C freeze

GRID29B真实控制记忆下153组DC配对：原34.611s、缓存13.268s，减少61.67%；154组保存三向量最大流量差0、2017步路径完全相同。实际原DC调用编译29.17s/后端0.070s，确认重复编译是这一DC组成的主要开销，不等于完整控制器根因。旧10.93%组恢复交付步骤省25.42s、其余多1.75s；旧0.39%组对应省10.32s、其余多9.62s。旧两组没有逐solver计时，不能事后识别各自DC收益。 Saved-data audit154DCtriples/2017physicalrows; no independent solve/new fit/reserved evaluation. GRID29C4balanced delivered-arm component traces frozen before results.

## 2026-10-11 — GRID29C delivered-arm component results

GRID29C完整组件配对4/4审计通过：两组完整控制提速6.45%/6.51%；DC省17.47/18.14s（9.61/9.94完整控制百分点），非DC部分分别多耗5.74/6.27s。680个配对solver调用的状态/迭代零改变，8068步动作/观测/成本/查询相同。重复编译节省机制成立，但旧10.93%/0.39%仅有逐步计时，具体历史波动成因无法完整追溯。 Sum-disjoint component savings reconstruct total; null stats retained. Unsafe author path still has13user_limit/3optimal_inaccurate/18optimal, restoration153optimal, both unchanged. GRID30 fixed qualification plan before first outcome.

## GRID29C narrative count correction

340+340=680paired explicit in-control solver calls; earlier682 narrative included initial DC incorrectly. Raw per-run statistics/review unchanged, no replay.

## 2026-10-11 — GRID30 replica-copy qualification repair

V0manual8pass but public-state335gen_per_bus includes2negative original author entries(min-0.0006708864). Setter replica refused them before original/shadow safe solve. Retain335step fragment and old code. V1copies arrays in-place exactly as original updater, assertion no clipping; new numeric/action gate still required. No deployment/new fit/hidden dates.

## 2026-10-11 — GRID30 qualified, GRID31 actual pilot frozen

GRID30安全QP共享编译复用资格通过：9项手工/复制检查、153次真实预求解参数/记忆三向量，最大差4.33e-10MW；306次动作重建相同，2017步实际路径/成本/查询仍原保护控制。隔离原QP42.45s、缓存15.24s，减少64.09%，不构成整套提速。V0原负值setter失败已保留，V1原位原样复制未裁剪。 Saved-data audit reconstructs306actions without new QP/AC solve.6actual delivered-arm profiles frozen before cached safe output; same original numeric/status/warm_start/cadence settings, no new fit/hidden dates.

## 2026-10-11 — GRID31 actual three-arm cache result

GRID31六条完整三月回放及审计通过：DC+安全QP共享复用相对原完整控制，两组减少23.49%/21.33%，汇总点估计22.42%；12102步动作/观测/成本/67950次预测相同，所有6项配对solver状态/迭代比较无差别。局部数值资格与完整时间分开，原10.93%/0.39%及6.45%/6.51%全部保留。仅三月2重复，不是GNN独有或封存泛化证据。 Saved-data check only, no independent new solve/refit. GRID32frozen before3exposed-week compatibility outcomes.

## 2026-10-11 — GRID32 other exposed-week compatibility

GRID32三组其他已暴露周配对通过：2035-06-25_8减少63.26%；2035-09-03_4减少61.89%；2035-10-01_5减少74.46%。11870步动作/观测/成本逐步相同，六月共同1901步失败保留，未新增/提前失败；每周仅一次配对，不作为独立封存泛化或GNN贡献。 No new fits/final dates. GRID33existing-ranker protocol frozen before output.

## 2026-10-11 — GRID33 existing-ranker matched screen

GRID33五个现成排序器130项同状态诊断及独立停止复算通过；查询7358/7530(GNN0/1)、7549/7826(MLP0/1)、7823(PPO)。全部找回1733关键动作；差距仅开发线索，不证明长期质量或图结构必要性。 No new fit/simulation/final date. GRID34common-engineering livecomparisonfrozen.

## 2026-10-11 — Saved-only component heterogeneity review

已完成用户要求的幅度诊断：三月可复用DC+安全QP占原控制39.7%、安全QP缓存命中77.8%、未复用unsafe占38.6%；其他三周可复用占73.4%–88.1%、命中90.0%–93.8%、unsafe占3.3%–8.2%，支持22%vs62%–74%的组件解释。旧10.93%/0.39%缺逐solver计时，精确波动原因仍不可恢复。GRID32继承的repeats=2/旧四条文案为元数据残留，实际冻结6条execution_order及审计一贯是3单配对；原JSON不改。 不把共享工程归于GNN，不推测温度/后台负载；GRID34所有强对照共用相同优化，正在公平闭环试点。

## 2026-10-11 — GRID34 fair cached March pilot gate

GRID34三月12条公平试点质量门槛通过：GNN1/MLP0/MLP1两次均逐步等同FULL；GNN0/PPO共同成本增加0.1361%，全部2017步完成。优化后FULL141–142秒，各排序器125–128秒；架构间仅小差距，MLP强对照保留，不能把总体节省全部归于GNN。 剩余36条按既定协议继续；相反6臂运行顺序全部保留，不能从一次较快时间识别GNN贡献。

## 2026-10-11 — GRID34 completed fair common-cache comparisons

GRID34全部48条公平闭环和保存数据审计完成；LOCAL_GNN0控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少11.34%/0.12%/1.36%/0.32%；LOCAL_GNN1控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少11.79%/0.62%/1.86%/0.82%。全部模型种子保留，六月共同1901步失败保留。两GNN均满足质量投资门槛，MLP0九月成本高0.60247%超过门槛但结果保留；共有编译复用不归于GNN，开发日期不作封存泛化。 48条重复内状态/反馈/完整向量/成本复核，未新增拟合或最终日期；相同路径子集和同输入查询比较单独记录。finished.wall_s仅最后剩余阶段，不误报为全部实验墙钟。

## 2026-10-11 — Public nonlearning order screens and quality pilot

GRID35/35B廉价顺序独立诊断通过：STATIC_ID/LOCAL_STRESS/原既有PHYSICS_TAIL查询7879/6965/6748，同FULL选择21/17/17（共26状态）；均找回1733已知关键动作，但最大rho差更大。便宜规则省查询不等于长期质量，GRID36三条三月闭环质量试点已冻结。 不从26个暴露状态认定可部署或图结构必需；静态原规则部件也给完整公平机会。

## 2026-10-11 — Cheap order quality and fair static comparator

GRID36三月廉价规则闭环核验完成：STATIC_ID成本+0.37200%通过0.5%筛查；PHYSICS_TAIL/LOCAL_STRESS成本+3.61164%/+4.01870%未通过，所有结果保留。静态顺序仅单次资格，不能据跨批时间宣告赢过GNN。GRID37冻结同一共有优化下FULL、两GNN、STATIC_ID四日期两平衡重复共32条，公平区分便宜停止与图模型增量。 新旧模型不覆盖，失败规则不原样扩展；先核对训练目标而非堆模块。

## 2026-10-11 — Focused target revision and limited baseline scope

按用户投稿目标收敛：主实验只保留2–3个最近且可用的成熟方法，MLP/静态顺序仅少量归因消融，不再扩规则组合。GRID37当前已启动批次跑完保留。训练集167状态/148有效目标峰值均与实际奖励最优对齐，没有代码错误；最优集合原目标均值权重24.44%。GRID38冻结仅改目标为一半原soft标签、一半奖励最优集合均匀标签，原初始化/数据顺序/网络/先验/训练轮数全部相同，两GNN两MLP公平处理；不覆盖原模型、不扫参数。 四次拟合尚未运行；与计时worker错开。

## 2026-10-11 — GRID37 completed

GRID37全部32条公平静态/GNN/完整搜索比较及保存数据核验完成；LOCAL_GNN0质量通过8/8、最大可比成本差0.136%；LOCAL_GNN1质量通过8/8、最大可比成本差0.001%；STATIC_ID质量通过8/8、最大可比成本差0.372%。两模型种子和两计时重复均保留，详见质量—时间主表。静态顺序属于必要消融，不扩展为方法清单。

## 2026-10-11 — GRID38 completed

GRID38一版目标修订的四次拟合与104项已有反馈排序核验完成；GNN0查询7358→7490、同FULL选择24→24/26、关键动作True；GNN1查询7530→7426、同FULL选择25→24/26、关键动作True；MLP0查询7549→7663、同FULL选择25→24/26、关键动作True；MLP1查询7826→7651、同FULL选择26→24/26、关键动作True。原/新模型、全部种子和训练记录保留，未新增教师仿真。离线查询与排序变化尚不是闭环质量或完整提速。

## 2026-10-11 — GRID39 before

GRID39已冻结12条有限闭环评价：一版目标修改的两个GNN种子与新运行FULL，4个已暴露开发周、单次计时、轮换顺序。控制/停止/守卫/缓存/物理求解不变；原模型和GRID37结果保留，新旧时间不作配对比较。不扩大规则或消融清单，最终8日期仍封存。

## 2026-10-11 — GRID39 complete

GRID39十二条一版目标修订闭环及保存数据核验完成；REVISED_GNN0质量通过4/4、最大可比成本差0.136%、控制时间减少11.03%；REVISED_GNN1质量通过4/4、最大可比成本差0.136%、控制时间减少11.29%。原版查询7312/7484，新版7380/7381；种子0成本不变、种子1四周平均成本增幅0.03403%（以FULL归一化）。没有稳定改进，暂不替换原配置；全模型保留。SoftGNN必要官方源码/论文配置已保存，尚未安装/拟合/回放。仅单次计时开发测量，不把跨批旧/新时间当配对模型提升。

## 2026-10-11 — GRID40 before

GRID40官方SoftGNN核心适配预检通过：167既有状态/148正标签、909动作、27维拓扑对象节点、3843309参数。原始GAT/输入转换源码未改；加入相同公开连续base信息、训练侧标准化、共同合法mask和搜索接口。两个种子固定64epochs/batch8尚未拟合；作者800max/256batch/分布式调参未复现，弱试点不否定公开方法。

## 2026-10-11 — GRID40 fit

GRID40两个原论文学习率的资源匹配拟合完成并核验目标/训练侧标准化；原模型保留。训练拟合存在平台，需要先做一版小批量优化适配（仅学习率降至0.001，其余初始权重/两种子/64轮/数据/结构不变），尚未读取SoftGNN开发预测，不拿欠拟合当作者方法失败。

## 2026-10-11 — GRID40B before

GRID40B仅降低学习率至0.001的两种子拟合已冻结；其余初始模型/数据/64轮/批量8/官方GAT/soft标签不变。用训练拟合平台而非评价质量提出这一版，原论文学习率结果全部保留。

## 2026-10-11 — GRID40 complete

GRID40两个固定SoftGNN核心适配拟合及26状态离线复核完成；SOFTGNN0查询7810、同FULL选择21/26、关键1733匹配True；SOFTGNN1查询7439、同FULL选择22/26、关键1733匹配True。这是公开方法资源匹配适配，不是作者已训练模型或忠实WCCI复现；闭环质量与实际速度尚未确定。

## 2026-10-11 — GRID40B complete

GRID40B两种子小批量学习率适配完成；seed0训练KL3.074→1.153；seed1训练KL2.992→0.679；统一配置资格True。选择仅用训练拟合，尚非闭环质量或近期公开方法性能结论。

## 2026-10-11 — GRID41 before

GRID41已冻结20条同条件闭环：原GNN两种子、近期SoftGNN适配两种子、成熟FULL。4暴露开发日期、单次轮换计时；共同控制/缓存/停止/保护不变，原版与学习率适配模型全部保留。SoftGNN统一配置由训练拟合确定，不按开发表现择种子。

## 2026-10-11 — GRID41 complete

GRID41二十条共同控制比较与保存数据核验完成；FULL质量通过4/4、查询22597、控制307.51秒；LOCAL_GNN0质量通过4/4、查询7312、控制272.25秒；LOCAL_GNN1质量通过4/4、查询7484、控制273.68秒；SOFTGNN0质量通过3/4、查询8054、控制267.58秒；SOFTGNN1质量通过3/4、查询8398、控制276.76秒。单次开发计时，近期SoftGNN是明确资源匹配适配；不可视为原论文完整复现或封存泛化。

## 2026-10-11 — GRID42/43 before

GRID41后续机制检查：SoftGNN三月578步的rho/FP32即时奖励与FULL完全并列但物理动作不同。GRID42导入冲突仅预检失败、保留；GRID42B独立复算130条缓存序列及隐藏反馈扰动通过。一条共同补查：完成已观测最佳奖励并列涉及的变电站，含安全前缀。原GNN0/1、SoftGNN0/1均找回578/1733，代价为查询增加；不是长期质量或图专属贡献。GRID43冻结6条三月试点，质量通过后18条其他开发日期；不重训、不打开最终8日期。

## 2026-10-11 — GRID42/43 pilot

GRID43共同并列补查三月六条试点完成，质量门槛未全部通过；FULL成本差0.000000、查询9578；LOCAL_GNN0成本差0.001361、查询3295；LOCAL_GNN1成本差0.000000、查询3613；SOFTGNN0成本差0.013741、查询3911；SOFTGNN1成本差0.013741、查询3818；LOCAL_PPO成本差0.001361、查询3378。结果保留，不选最佳种子。

## 2026-10-11 — GRID44 before

GRID43并列补查试点关闭：6条保留、18条未运行；不改善原模型成本且增加查询，当前仍用原GRID21权重和GRID28规则。GRID44首次8预留日期家族确认已冻结：96串行回放、两计时重复、原GNN/SoftGNN各两种子、作者PPO及FULL；预处理/缓存/控制/停止一致，先冻结后解压。不重新训练、不按确认结果调参；全部家族及变体持续排除训练。

## 2026-10-11 — GRID44 acquired

GRID44冻结8个首选序列已按官方索引安全解压并逐文件哈希，104文件、50541137字节；尚未运行控制器或读取评价成绩。96条冻结串行确认待运行，不重新训练。

## 2026-10-11 — GRID44 user shutdown

用户准备关机，GRID44在36/96条后保存停止，剩余60条。全部已完成轨迹、计量与哈希保留，调度器不再派发任务；未重训/调参、尚无整批确认裁决。下次按原 --phase all 跳过已完成身份续跑，不重新准备/解压。

## 2026-10-12 — GRID44 complete

GRID44八预留日期家族96条串行确认及保存数据核验完成；FULL完成16/16、无提前失败16/16、质量筛查16/16；LOCAL_GNN0完成16/16、无提前失败16/16、质量筛查14/16；LOCAL_GNN1完成16/16、无提前失败16/16、质量筛查14/16；SOFTGNN0完成16/16、无提前失败16/16、质量筛查14/16；SOFTGNN1完成16/16、无提前失败16/16、质量筛查12/16；LOCAL_PPO完成16/16、无提前失败16/16、质量筛查14/16。两原模型平均控制提速14.624%（家族bootstrap95%[10.650%,18.543%]），各有一个日期成本+0.88509%超过原0.5%筛查，联合筛查未通过；不是研究绝对失败。原GNN对PPO宏平均提速3.211%/2.481%，不普遍支配SoftGNN。48/48计时重复物理路径一致；两次计时/两种子不是独立日期，SoftGNN为资源匹配适配。全部日期使用后不再叫封存，禁止据其择模型赢家。

## 2026-10-12 — GRID45/47 before

GRID45独立复算96条确认并定位4月23日540步：两GNN与PPO均在rho0.92207平台停查、漏掉FULL844（真实奖励差0.00686），不能单动作因果归整周成本。GRID46只恢复作者原0.9缓冲要求，旧26状态×5排序器130项序列/未观测反馈核验通过，原GNN/PPO同FULL选择26/26；查询14893。GRID47冻结24条旧开发回放，先6三月，原两GNN质量及正提速通过才18其他旧日期；所有基线共用保护，不重新训练/复用GRID44确认当未见评价。回补为已有保护的质量—时间端点，不包装新安全理论。

## 2026-10-12 — GRID45/47 pilot

GRID47六条共同作者阈值回补试点完成；原两GNN门槛=True；FULL成本差0.000000、时间减少0.0000；LOCAL_GNN0成本差0.000000、时间减少0.0358；LOCAL_GNN1成本差0.000000、时间减少0.0375；SOFTGNN0成本差0.012900、时间减少0.0390；SOFTGNN1成本差0.061945、时间减少-0.0349；LOCAL_PPO成本差0.000000、时间减少0.0258。所有基线与种子保留，单次开发计时。

## 2026-10-12 — GRID45/47 complete

GRID47共同作者阈值回补24条旧开发比较与保存数据核验完成；FULL质量4/4、查询22597、控制321.67秒；LOCAL_GNN0质量4/4、查询14893、控制310.55秒；LOCAL_GNN1质量4/4、查询14893、控制310.32秒；SOFTGNN0质量3/4、查询15251、控制301.07秒；SOFTGNN1质量3/4、查询16057、控制313.99秒；LOCAL_PPO质量4/4、查询14893、控制306.97秒。保护是已有作者缓冲的回补，不是新安全理论或图专属；没有训练或GRID44重评价。

## 2026-10-12 — GRID48 before

按用户要求保留原快速结构。GRID48训练侧资格：±2残差导致72/148最佳目标无法排第一、70/148峰值概率质量超硬界；但没有任何目标数学上被排除128候选，不能把硬界当成闭环失败原因。仅移除末端tanh硬界，保留PPO先验KL0.1、原GNN结构、同初始权重/数据/目标/64轮/两种子；不采用严格GRID47保护，不训练或重评GRID44日期。序列化NumPy布尔缺陷原版保留，资格数值按相同规则修复。两次拟合尚未开始。

## 2026-10-12 — GRID48 complete

GRID48移除残差硬界的两次固定拟合及旧26状态复核完成；GNN0查询7358→7385、同FULL选择24→25/26、1733匹配True；GNN1查询7530→7279、同FULL选择25→24/26、1733匹配True。原模型全部保留，未训练/评价GRID44日期；仅缓存开发比较，实际质量和提速尚未证明。

## 2026-10-12 — GRID49 before

GRID49冻结原快速控制下的一次评分改动比较：原GNN两种子、新无硬界残差两种子、FULL；先5旧三月试点，两个新种子质量/存活合格才15其他旧开发回放。停止/保护/特征/缓存/控制均沿用原快版，不采用GRID47严格回补。没有GRID44训练/重评或新模型拟合；近期强基线原比较完整保留，不扩baseline菜单。

## 2026-10-12 — GRID49 pilot

GRID49原快控制五条评分改动试点完成；两新种子质量=False；FULL成本差0.000000、查询9578、控制141.16s；LOCAL_GNN0成本差0.001361、查询3007、控制126.22s；LOCAL_GNN1成本差0.000000、查询3122、控制125.95s；FREE_GNN0成本差0.000000、查询2967、控制126.07s；FREE_GNN1成本差0.013741、查询2882、控制121.77s。全部原/新权重、结果保留，不选种子。

## 2026-10-12 — GRID50 before

按用户保留原版优势：GRID49自由残差种子0成本零且约10.69%提速，种子1成本+1.374%故不替换原版、15条未跑。GRID50只做一版软压缩2*asinh(raw/2)，尺度仍原2、局部斜率1、先验KL0.1保留；同结构/初始化/数据/目标/64轮/两种子，原快停止不变。若仍混合，停止激活形式变更，转向训练覆盖；不在GRID44调参或择种子。

## 2026-10-12 — GRID50 complete

GRID50一版软压缩两次拟合及旧26状态复核完成；GNN0查询7358→7728、同FULL选择24→25/26、1733True；GNN1查询7530→7728、同FULL选择25→25/26、1733True。不挑种子，不重训/评价GRID44，原快模型仍默认；缓存表现不等于闭环质量/提速。

## 2026-10-12 — GRID51 before

GRID51冻结原快速控制下的一次评分改动比较：原GNN两种子、新软压缩残差两种子、FULL；先5旧三月试点，两个新种子质量/存活合格才15其他旧开发回放。停止/保护/特征/缓存/控制均沿用原快版，不采用GRID47严格回补。没有GRID44训练/重评或新模型拟合；近期强基线原比较完整保留，不扩baseline菜单。

## 2026-10-12 — GRID51 pilot

GRID51原快控制五条评分改动试点完成；两新种子质量=True；FULL成本差0.000000、查询9578、控制153.76s；LOCAL_GNN0成本差0.001361、查询3007、控制130.89s；LOCAL_GNN1成本差0.000000、查询3122、控制127.29s；COMPRESSED_GNN0成本差0.000000、查询3309、控制128.01s；COMPRESSED_GNN1成本差0.000000、查询3213、控制126.15s。全部原/新权重、结果保留，不选种子。

## 2026-10-12 — GRID51 complete

GRID51原快速控制20条评分改动比较与保存数据核验完成；FULL质量4/4、查询22597、控制328.62s；LOCAL_GNN0质量4/4、查询7312、控制295.48s；LOCAL_GNN1质量4/4、查询7484、控制279.75s；COMPRESSED_GNN0质量4/4、查询7682、控制278.91s；COMPRESSED_GNN1质量4/4、查询7682、控制281.65s。全部种子保留；旧开发非独立确认，实际改动不自动成为默认。

## 2026-10-12 — GRID52 before

GRID51软压缩候选保留：旧开发完成周成本与FULL相同、控制减少14.3%–15.1%，不声称已胜PPO/SoftGNN。GRID52新版新八日期确认已在解压前冻结：两模型种子、三类主对照、两次平衡计时共96条；全部先前暴露/训练/开发/已提取日期排除，按季度固定哈希选日期和变体。Q1/Q3剩余各两日期，选择受限如实披露。无新训练/教师，原GRID44结果及质量损失完整保留。

## 2026-10-12 — GRID52 acquired

GRID52八冻结首选序列解压核验完成：104文件、51506252字节。尚无控制器评价/新拟合，96条固定确认待运行。

## 2026-10-12 — GRID52 complete

GRID52新版八新日期96条确认及保存数据核验完成；FULL完成16/16、质量16/16、控制1330.81s；SOFTGNN0完成16/16、质量16/16、控制1117.88s；SOFTGNN1完成16/16、质量16/16、控制1102.17s；LOCAL_PPO完成16/16、质量12/16、控制1095.09s；COMPRESSED_GNN0完成16/16、质量16/16、控制1164.43s；COMPRESSED_GNN1完成16/16、质量16/16、控制1152.88s。新版两种子均通过原质量—时间筛查，完整周最大成本增幅均0.08045%，对FULL宏平均控制时间减少14.465%/14.748%、查询少70.00%/70.62%；平均两种子时间减少14.606%（八家族bootstrap95%[7.240%,22.175%]）。但控制时间比PPO约长3.24%–3.57%、比SoftGNN1约长4.34%–4.60%，不能主张全面领先；PPO最大成本增幅12.434%，SoftGNN两种子最大增幅0.177%/0.169%且均通过质量筛查。48/48计时重复路径一致。全部种子/失败保留；两次重复不是独立家族；SoftGNN为资源匹配适配；GNN独有归因未自动建立，不把旧GRID44失误视为已经修复。

## 2026-10-12 — GRID53 timing diagnosis

GRID53已有212份计时/1272项逐搜索账目核查完成：GRID44原版对PPO快2.48%–3.21%、对SoftGNN1快2.25%–2.93%，但慢于SoftGNN0；GRID52新版在不同日期上对这两项均略慢，不能据跨日期比较认定模型退步。GRID51同批新版seed0快5.606%、seed1慢0.678%，结果混合。新版前向仅约2.3ms/搜索、共同PPO先验约233ms；一份旧可见状态剖析确认909次as_dict重复静态拓扑展开占先验剖析约85%。相对SoftGNN额外先验及不同路径连续优化共同形成时间差；相对PPO主要差值在连续优化，不能只归于神经网络或用公共缓存冒充图贡献。未训练、未新候选simulate、未改模型或停止规则；原新全部结果保留。

## 2026-10-12 — GRID54 before

GRID54先验静态元数据缓存资格已冻结：只替换909候选拓扑字典与动作映射的重复展开；每次仍读取新鲜动作字段验证快照，替换/原地变化即重建，不用可能陈旧的to_vect。130个旧开发状态精确比较先验分数/排序/最终模型排名与特征哈希，原新版与PPO公平共享；冷启动计入先验时间，无训练/候选simulate/保护改动。

## 2026-10-12 — GRID54 complete

GRID54等价先验缓存资格通过：130状态分数/排序/最终排名/特征逐项一致，四项替换/原地修改/恢复重建通过；局部先验累计29.966→3.615s（减少87.94%），含首次冷构造与每次动作新鲜字段校验。无新训练/候选simulate；不是闭环速度或GNN独有贡献，公平给旧新PPO。

## 2026-10-12 — GRID55 before

GRID55一次有限同日期旧新版及既有主对照比较已冻结：128条串行回放、8个既有暴露日期、两个平衡计时；原GNN/软压缩GNN/SoftGNN各两种子、作者PPO、FULL。共有GRID54先验缓存含首次构造和逐次内容核验，原GRID28快控制/缓存/合法信息不变。无新训练/参数/日期替换；此为配对选型与工程归因，不重新宣称独立确认。

## 2026-10-12 — GRID55 complete

GRID55有限同日期旧新版及既有强对照比较与保存数据核验完成；FULL质量16/16、控制1313.24s；SOFTGNN0质量16/16、控制1101.60s；SOFTGNN1质量16/16、控制1103.91s；LOCAL_PPO质量12/16、控制1130.60s；COMPRESSED_GNN0质量16/16、控制1151.80s；COMPRESSED_GNN1质量16/16、控制2392.41s；LOCAL_GNN0质量14/16、控制1072.01s；LOCAL_GNN1质量14/16、控制1069.77s。各版本/全部种子/失败保留，实际质量—时间及配对差见报告；已暴露日期不当独立确认，共有缓存不作为图模型独有。 **计时资格未通过：** Windows现代待机事件与worker125（新版种子1）及127（PPO）重合；前者控制1354.35s/CPU96.94s，不能把巨幅负提速解释为模型退步。原128条、RAW统计与选型均未改；64/64重复路径一致，成本证据有效（原版最坏+1.740%/+1.704%，新版均+0.08045%）。最终速度与版本选型仍未解决。仅提出为该日期补一次完整八臂计时，不覆盖原记录、不变模型/规则；受本批不重复已完成项/不改预算约束，尚未执行，需用户决定。请先读GRID55_review_status.md，再读RAW报告。

## 2026-10-13 — GRID55B before

用户2026-10-13继续后，按关机前具体方案冻结GRID55B：仅待机受影响日期2035-11-12_9全部八方法补新重复2，2400秒墙钟上限；原128条不覆盖，不改模型/控制/阈值/先验缓存，不新增训练。运行期临时防空闲待机，不修改永久电源计划。结束后核对电源日志、原两次物理路径/成本/合法反馈/查询，对全部八方法一致构建原重复0/新重复2计时视图，沿用原对称选型。不是新独立确认。

## 2026-10-13 — GRID55B complete

GRID55B八条补测与保存数据核验通过：16/16与原两重复的路径、向量、成本、反馈和查询一致，运行期无电源待机事件。原RAW128保留，七日期原0/1加受影响日期原0/新2组成明确来源的128槽位计时视图。FULL质量16/16、控制1311.18s；SOFTGNN0质量16/16、控制1092.66s；SOFTGNN1质量16/16、控制1095.14s；LOCAL_PPO质量12/16、控制1064.52s；COMPRESSED_GNN0质量16/16、控制1133.24s；COMPRESSED_GNN1质量16/16、控制1124.13s；LOCAL_GNN0质量14/16、控制1072.05s；LOCAL_GNN1质量14/16、控制1071.05s。原对称选型=COMPRESSED_CANDIDATE_QUALITY_SCREEN；这是质量—时间投资筛查，不是等价或独立确认；公共缓存不归于GNN。

## 2026-10-13 — GRID56 evidence closure

GRID56最终审计收口通过：84项冻结哈希、9项RAW锚点、174项新补测文件收据、3842项历史归档条目、128槽位来源与21项统计/bootstrap/选型公式。无新仿真或训练，不宣称独立AC重求解。完成质量—速度主表、科学图、方法结果工作稿、贡献边界与复现入口；训练教师费用按20训练周分离为50.68分钟/146521个N1查询，最终两拟合调用89.99秒，已有PPO成本仍未知。现成STATIC_ID开发消融接近GNN，下一项仅补最终配置的廉价顺序消融，不扩主对照。

## 2026-10-13 — GRID57 before

GRID57仅冻结一条必要廉价顺序消融：既有STATIC_ID在最终八个已暴露日期各一次回放，原控制/停止/合法信息/DC与安全QP缓存不变，排序是冻结的代表ID升序。无新训练/参数/日期/基线家族；主比较完整周质量、存活和查询，历史FULL/GNN只用于离线参照；新批墙钟不与旧批构成公平提速主张。

## 2026-10-13 — GRID57 complete

GRID57八条既有STATIC_ID最终配置消融与保存数据审计完成：质量尺度通过8/8，GNN平均相对其查询减少1.897%，正向4/8；裁决MIXED。无新拟合；不作跨批速度或新独立确认。

## 2026-10-13 — GRID57 prior review

GRID57固定顺序来源已核实：909动作counts非增且全部八运行行hash等同作者资产顺序，等价历史教师频次先验。当前GNN查询宏平均仅少1.90%（7有调用日期），汇总反而多6.69%；成本1好/6差/1平、平均高0.509个FULL归一化百分点。现有图模块增量未成立，整体提速仍保留。只保留“让GNN校正较强频次先验”的单次改进假设，不继续基线扩散。

## 2026-10-13 — GRID58 before

GRID58冻结一次先验替换：保留作者教师频次STATIC_ID为零残差基准，原软压缩GNN/尺度2/初始化/167状态148目标/64轮/两种子/优化器/停止/守卫均不变。仅两个拟合与26旧开发缓存状态资格；不训练最终八日期，不调参或新增教师。两种子须找回既知1733才另设有限旧开发闭环。

## 2026-10-13 — GRID58 complete

GRID58两个固定频次先验GNN拟合及52项旧开发缓存核验完成；seed0查询7457、STATIC7879、同FULL22/26、1733=True；seed1查询7520、STATIC7879、同FULL22/26、1733=True。旧开发闭环资格=True，尚未运行闭环，无新教师；缓存结果不作质量/提速证明。

## 2026-10-13 — GRID59 before

GRID59冻结16条四旧开发周有限配对试点：频次先验GNN两种子、强STATIC_ID、FULL；原快控制/停止/合法信息/DC安全QP缓存不变，单次平衡串行计时。无新拟合或最终八日期评价；先验证两种子质量，再按冻结查询条件看额外学习信号，不新增先验/常数变体。

## 2026-10-13 — GRID59 complete

GRID59有限频次先验闭环及保存数据审计完成；STATIC_ID质量4/4、对STATIC查询少0.000%、对STATIC控制少0.000%；FREQUENCY_GNN0质量4/4、对STATIC查询少3.127%、对STATIC控制少-0.635%；FREQUENCY_GNN1质量4/4、对STATIC查询少2.675%、对STATIC控制少-0.506%；裁决WEAK_OR_MIXED_STOP_REVISION。单次旧开发配对，不是封存确认。原新先验/模型/失败全部保留。

## 2026-10-13 — GRID60 before

用户已选择以框架作用为论文主线、GNN可替换辅助。GRID60仅做机制/近邻方法资格核对：共同频次排序下FULL、首个安全候选加增强回退、无变电站补全、现有完整框架；不新增训练/参数。

## 2026-10-13 — GRID60 complete

用户已选择以框架作用为论文主线、GNN可替换辅助。GRID60仅做机制/近邻方法资格核对：共同频次排序下FULL、首个安全候选加增强回退、无变电站补全、现有完整框架；不新增训练/参数。104项合法反馈扰动与90项原FRAME复算通过；实际局部补查改变12次选择。缓存状态数量/选择尚不是完整周机制因果证据。

## 2026-10-13 — GRID61 before

GRID61同频次顺序的框架机制矩阵：FULL/加强首个安全候选/去局部补全/现有FRAME，四旧开发日期、两个平衡重复32条；仅改变停止/补全策略。原GRID28基础控制、合法反馈和共有DC/safeQP缓存不变；无GNN拟合、阈值调节或外部基线扩散。

## 2026-10-13 — GRID61 complete

GRID61同频次顺序的框架机制矩阵：FULL/加强首个安全候选/去局部补全/现有FRAME，四旧开发日期、两个平衡重复32条；仅改变停止/补全策略。原GRID28基础控制、合法反馈和共有DC/safeQP缓存不变；无GNN拟合、阈值调节或外部基线扩散。裁决LOCAL_COMPLETION_SUPPORTED_IN_MATRIX；FULL质量8/8、控制减少0.000%;FIRST_SAFE_BEST质量4/8、控制减少7.032%;NO_LOCAL质量6/8、控制减少14.251%;FRAME质量8/8、控制减少13.059%。保留全部共同失败、较慢和较差结果；同排序机制比较不是新独立确认。

## 2026-10-13 — GRID62 before

GRID62冻结全部剩余三个未用日期、四既有同频次策略、两平衡计时重复，共24条。主方法FRAME、对照FULL/加强首个安全候选，NO_LOCAL只作已有内部消融。原控制/缓存/参数均不变，无训练；三个日期不能包装成八日期或跨电网确认。

## 2026-10-13 — GRID62 complete

GRID62冻结全部剩余三个未用日期、四既有同频次策略、两平衡计时重复，共24条。主方法FRAME、对照FULL/加强首个安全候选，NO_LOCAL只作已有内部消融。原控制/缓存/参数均不变，无训练；三个日期不能包装成八日期或跨电网确认。保存数据审计通过，裁决FRAMEWORK_CONFIRMED_IN_LIMITED_FRESH_MATRIX。全部失败/较慢/较差结果保留；非独立AC重求解。

## 2026-10-13 — GRID62 timing and claims review

GRID62三剩余封存日期24条及保存数据审计完成：FRAME质量3/3、完成2/3（五月四策略共同747步失败）、原宏平均控制少6.086%、查询少83.543%，最坏完整周成本+0.013871%，通过有限冻结尺度。两计时重复路径一致12/12。补充计时分解说明五月无搜索，计时差不能归于未调用策略；两有搜索日期FRAME控制少8.250%、FIRST_SAFE_BEST少11.921%（事后描述，不替换主判据）。FIRST在新日期也质量3/3；对成熟停止策略优势及补全普遍必要性未建立。新四月补全成本改善仅0.007818%FULL，新十一月补全未触发；旧三月机制实例保留。重复CPU/编译/求解时间变异较大，不宣称稳定全面提速、不加时或扫参数。

## 2026-10-13 — framework manuscript V1/V2

完整框架英文论文V1/V2与来源表、伪代码、单步性质、三日期封存主表、计时反例、科学图、表格独立复算和轻量复现入口完成。V1保留。原冻结确认通过，但FIRST_SAFE_BEST新日期也质量3/3，FRAME对它稳定优势未建立；主张限定为相对FULL有限质量—计算改善与已有具体机制，不扩大为普遍补全必要或GNN创新。尚需作者/目标会议格式和最终文献定位；不是可直接投稿或一键完整仿真复现的认证。

## 2026-10-13 — submission positioning and author materials

完成一次正式投稿定位与作者重写素材包：ECAI框架/搜索主方向、ECML2027 ADS应用备选、ICAPS2027较早但规划范围有条件，PAIS待当届细则与单位标准。官方来源17条、8行统一主表/28日期方法行、14项主张证据、6条BibTeX、代码映射/算法/性质、SVG与科学图、标准与AI使用事实说明已准备。按先平均重复再算比例统一GRID61 FRAME为13.1121%，原13.0588%保留，不改科学裁决。独立便携表格复算通过；769MB原始记录另建本地raw包并CRC/哈希校验，未推入Git。轻量附件齐全但尚不是匿名投稿或完整独立仿真安装包。无新运行/训练；两个离线材料目录/换行错误留档，未损及历史结果。
