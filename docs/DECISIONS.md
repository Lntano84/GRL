# GRL Decision Log

## 2026-10-10 — Power-grid closed-loop evidence overrides ranking-only optimism

The user authorized autonomous Grid2Op/L2RPN iteration with every old model and modification preserved. GRID22/23 are complete: 52 developmental trajectories and saved-data audits. The current role is supervised candidate ranking above a frozen published PPO prior, not a newly trained RL policy.

K128 with a common exact-reward tie rule removed the extra lost September week, but March cost remains 8.073% above FULL. Both GNN seeds fail the frozen auxiliary quality gate; matched-budget PPO/MLP/bias controls remove any established practical graph increment. Do not claim that raising the candidate budget solved the model, that near-best one-step rho guarantees whole-week cost, or that the neural architecture combination is new.

Preserve both configurations, both GNN seeds and V0/V1 initial/final weights. Separate public simulation counts from single instrumented wall timing, and compare costs only on jointly completed weeks. The source's N1 label means disconnected-line conditions, not an exhaustive N-minus-one security assessment.

The original suffix-reserved test was partly used at GRID19; it is developmental now. Eight distinct date families remain reserved for the eventual evaluation, with all variants protected. The published PPO's training identity is unknown. No new RL or final evaluation was started by GRID22/23. The next justified model question concerns candidate/object information and alignment to the delivered control outcome, rather than another budget or depth sweep. Earlier IM/LOT/FA/LG decisions remain historical and are not automatically reopened.

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


## 2026-10-06 — Evidence archive and current stopping decisions

保留全部有效与否定结果，协议/数据资格失败单独标注。LG01接受固定LOSS-EXPAND并停止当前GNN配置，不进入RL；LOT固定AB不再作为核心贡献；FA低预算线索仍不等于新算法贡献。以最终更正报告为准，不覆盖删除历史证据。

## 2026-10-10 — Retain quality protection before speed claims

Human goal: speed with adequate control quality. GRID24 uses published verify/fallback logic and unchanged author threshold; it is not a new threshold sweep or new network. Quality is recovered on all four exposed weeks with fewer public searches. Keep it as the current protected operating point, preserve unguarded models/results as ablations. Do not combine unguarded82% query reduction with guarded0 cost regret. PPO achieves the same result, so graph-specific novelty remains unestablished. Next evidence priority is controlled total-control timing, then untouched date-family evaluation under a frozen protocol. No new training, large K sweep, or final holdout opened this stage.

## 2026-10-10 — Benchmark actual control cost without audit overhead

Human authorized continued speed validation. Preserve weights/rules and use balanced serial repetitions with candidate archival/verification outside act time. Remove per-instance method-binding measurement cycles before formal repetitions. Keep GRID25 diagnostic rather than selecting its favorable timings. Report source-search acceleration separately from total-control acceleration; do not attribute non-search variation to GNN. Reserved8 date families remain unopened until protocol is finalized.

## 2026-10-10 — Separate source acceleration from whole control and graph increment

Keep all48 GRID26 repetitions rather than the favorable old GRID25 pilot. Exact quality preservation and17%–20% source acceleration are developmental evidence;0.88%–1.40% whole-controller point estimates with a reversed repeat do not establish stable significant total acceleration. Non-search time decomposition is accounting, not identified causality. PPO shares the same query reductions; do not label its non-search variation a graph advantage. Next check static action metadata caching with exact rank equivalence and the same optimization for PPO. Preserve checkpoints, guard and hidden families; no new training is implied by engineering optimization.

## 2026-10-11 — Improve the costly fallback decision before network capacity

The objective is useful same-quality acceleration,not to force GNN attribution.Whole-controller gains remain small and repeated timing reverses;PPO shares present savings.Offline diagnosis identifies16 fallback states with no safe candidate even in the full pool;under unchangedK128/guard better ranking cannot reduce their queries.Prioritize a separately evaluated legal staged-search/stopping design rather than blind depth/GPU/RL changes.Preserve protected baseline and allweights;coverage of privilegedFULL winners is not an online certificate.No guard was removed or threshold changed in this review.

## 2026-10-11 — Check closed-loop quality before interpreting staged-search savings

User authorizes both whole-controller and expensive-search acceleration experiments.GRID27 rho-small omissions include a known cost-critical action,so do not deploy that plateau alone.GRID28 semantic substation completion is a shared cheap rule,not graph-specific novelty or stopping proof.Keep all schedules/checkpoints;use matchingPPO,unchanged admission/continuous safeguards andpre-specified pilotgate.Exact-protected profiling is diagnostic;generic optimizer acceleration must be applied to all baselines.No immediate RL or hidden-date evaluation.

## 2026-10-11 — Qualify shared DC compilation reuse independently

Use per-optimizer topology/reactance-keyed bounded CVXPY/DC-program reuse, retaining the exact mathematical problem and original solve wrapper/warm_start=False. Frozen stored-state qualification precedes a balanced protected-March pilot. No blind direct-solver substitution or changed tolerances. Apply any engineering improvement to strong baselines; do not count it as GNN novelty. GRID28 remaining runs still serial, no reserved-date opening.

## 2026-10-11 — Retain staged-search tradeoff without graph attribution

Keep measured quality cost and one-repeat timing visible. Substation completion addresses the known omitted action, while the original protected guard remains available. Same stopping logic for GNN and PPO is essential; shared improvement is not GNN benefit. Whole-controller engineering qualification proceeds separately and is granted to all baselines. Do not train RL or open final date families from these developmental results.

## 2026-10-11 — Reject the non-DPP cache before rollout

V0 was blocked by actual Parameter division, unlike its ndarray toy; its object-valued reactance cache key was also wrong. Retain V0 and use value-keyed numeric reciprocal following CVXPY DPP documentation. Added real Parameter mutation and triangle-flow tests. This changes symbolic representation only, not mathematical constraints or solver settings. No valid cache shadow result/live trajectory existed at revision time.

## 2026-10-11 — Diagnose component speed before more total-time repeats

Cache pilot preserved exact quality, but10.93% versus0.39% paired total gains do not identify stable whole-speed magnitude. Keep both. Next one instrumented actual controller plus isolated same-memory DC pairs; never deploy probe output or label instrumented total as acceleration. Include positional solver arguments and null statistics correctly. GNN/PPO matched-query developmental signal does not identify GNN architecture or independent transfer; common engineering changes go to both.

## 2026-10-11 — Explain the10% pair with measured components

Live-memory isolated DC matches actual flows exactly and identifies compilation savings, but old paired trajectories lack per-solver timing. Do not attribute old total variation to a particular optimizer from residual arithmetic. Freeze full-arm component measurement with identical instrumentation/paths and retained two balanced orders. No model or numeric control change. Quota interruptions are not user cancellation; resume current work without repeated confirmation.

## 2026-10-11 — Extend proven compilation reuse only after numerical qualification

Whole-arm DC gains are real and match unchanged output/numeric iteration paths, but total saving is partly offset elsewhere. Historical10% is plausible, not a chosen robust guarantee. Safe restoration shows analogous repeated compilation overhead; examine bounded topology/coefficient keyed exact parameterization, with same solver/status/warm_start/cadence. Manual tests plus actual-memory isolated action comparison precede deployment. Generic optimization is available to all learned/nonlearned comparators.

## 2026-10-11 — Shared safe-QP cache qualifies; measure rather than add percentages

Isolated safe-QP outputs and converted actions match original at live inputs. This permits a6run three-arm original/DC/DC+safe measured pilot, not an assertion of64%total control speed. Preserve negative parameter copier failure and all previous timings. No numerical clipping, solver loosened tolerance or changed optimization cadence. Treat shared compilation reuse as engineering provided to every strong baseline, not GNN innovation.

## 2026-10-11 — Retain measured22% common acceleration without graph attribution

Full delivered-arm improvement is observed with exact path/cost/native solver behavior. This supports practical shared engineering, not graph-specific novelty. Before fair learned search comparison, check numerical/path compatibility on other exposed dates, including the known June failure. Do not choose only the10%or23%repeat, add component percentages, train immediately or open final date families.

## 2026-10-11 — Give shared solver reuse to all graph controls

Cross-week exact paths support common engineering, with strong magnitude heterogeneity and a preserved June failure. Next test existing MLPs rather than train or conflate common reuse with GNN. Known critical-action offline miss blocks unchanged semantic-stop deployment; it does not reject all possible MLP policies.

## 2026-10-11 — Keep both MLP seeds and both GNN seeds in fair live comparison

All existing rankers pass the known critical-action screen; do not omit MLP or select a best checkpoint. Give exhaustiveFULL and everyranker common caches. Reversed6arm paired order per exposeddate, shared quality gate and all components. Pilot cost/survival must be measured before widening. Offline matched-query counts are not full-controller acceleration or an architecture-proof.

## 2026-10-11 — Distinguish measured cache dose from unrecoverable historical noise

已完成用户要求的幅度诊断：三月可复用DC+安全QP占原控制39.7%、安全QP缓存命中77.8%、未复用unsafe占38.6%；其他三周可复用占73.4%–88.1%、命中90.0%–93.8%、unsafe占3.3%–8.2%，支持22%vs62%–74%的组件解释。旧10.93%/0.39%缺逐solver计时，精确波动原因仍不可恢复。GRID32继承的repeats=2/旧四条文案为元数据残留，实际冻结6条execution_order及审计一贯是3单配对；原JSON不改。 不把共享工程归于GNN，不推测温度/后台负载；GRID34所有强对照共用相同优化，正在公平闭环试点。

## 2026-10-11 — GRID34 fair cached March pilot gate

GRID34三月12条公平试点质量门槛通过：GNN1/MLP0/MLP1两次均逐步等同FULL；GNN0/PPO共同成本增加0.1361%，全部2017步完成。优化后FULL141–142秒，各排序器125–128秒；架构间仅小差距，MLP强对照保留，不能把总体节省全部归于GNN。 剩余36条按既定协议继续；相反6臂运行顺序全部保留，不能从一次较快时间识别GNN贡献。

## 2026-10-11 — GRID34 completed fair common-cache comparisons

GRID34全部48条公平闭环和保存数据审计完成；LOCAL_GNN0控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少11.34%/0.12%/1.36%/0.32%；LOCAL_GNN1控制时间相对共有优化FULL/PPO/MLP0/MLP1分别减少11.79%/0.62%/1.86%/0.82%。全部模型种子保留，六月共同1901步失败保留。两GNN均满足质量投资门槛，MLP0九月成本高0.60247%超过门槛但结果保留；共有编译复用不归于GNN，开发日期不作封存泛化。 48条重复内状态/反馈/完整向量/成本复核，未新增拟合或最终日期；相同路径子集和同输入查询比较单独记录。finished.wall_s仅最后剩余阶段，不误报为全部实验墙钟。

## 2026-10-11 — Public nonlearning order screens and quality pilot

GRID35/35B廉价顺序独立诊断通过：STATIC_ID/LOCAL_STRESS/原既有PHYSICS_TAIL查询7879/6965/6748，同FULL选择21/17/17（共26状态）；均找回1733已知关键动作，但最大rho差更大。便宜规则省查询不等于长期质量，GRID36三条三月闭环质量试点已冻结。 不从26个暴露状态认定可部署或图结构必需；静态原规则部件也给完整公平机会。

## 2026-10-11 — Cheap order quality and fair static comparator

GRID36三月廉价规则闭环核验完成：STATIC_ID成本+0.37200%通过0.5%筛查；PHYSICS_TAIL/LOCAL_STRESS成本+3.61164%/+4.01870%未通过，所有结果保留。静态顺序仅单次资格，不能据跨批时间宣告赢过GNN。GRID37冻结同一共有优化下FULL、两GNN、STATIC_ID四日期两平衡重复共32条，公平区分便宜停止与图模型增量。 新旧模型不覆盖，失败规则不原样扩展；先核对训练目标而非堆模块。
