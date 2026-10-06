# TAIM no-training minimum probe (frozen before outcome inspection)

Question: On naturally generated intermediate IC states, does a higher-sampling
one-step foresight decision beat the strongest *development-selected* cheap
policy enough to justify its additional decision time?

This is a deliberately small probe, **not** a faithful reproduction of Tong
et al.'s full experimental protocol or evidence of GRL's usefulness.

* Networks: the existing football (115/613) and polbooks (105/441) GML files;
  undirected ties become two independent directed IC arcs. Two fixed IC
  probabilities, 0.1 and 0.2, not adjusted after results.
* Natural state generation: choose the first seed uniformly from the top 20
  out-degree nodes, run one actual IC round; freeze the resulting active set
  and current propagation frontier. Initial horizon is 3; analyze the
  remaining two rounds, with remaining seed budget b=1 and b=2 for each state.
  Four development and eight held-out base states per graph/probability stratum.
  State-generation streams are separate from all decision and evaluation
  streams. No filtering for interesting spreads or policy disagreement.
* All policies see the known graph/probability/current active/current frontier,
  but never future live-edge outcomes. They share a 32-world Monte Carlo greedy
  current-node ordering. At the final round they share an *exact expected
  one-round* greedy selector. This keeps the timing decision the only policy
  difference. These are approximate greedy choices, not a guarantee of global
  node-optimality.
* Cheap arms: always invest all now; always wait; balanced (one now for b=2);
  a separate b=1 and b=2 frontier-size threshold from {0, .02, .05, .1,
  .2} (25 predetermined pairs; denominator is current active count);
  FF's Eq. (8)-style overlap and delay score with 32 worlds, theta in
  {.2,.4,.6,.8}. The best cheap *configuration* is selected using development
  states and 128 independent evaluation worlds, then frozen for the held-out
  states. Do not select a different cheap arm per held-out state or budget.
* SOF: enumerate i=0..b current seeds, simulate one feedback round, greedily
  select the remaining b-i seeds for the final round, score its exact expected
  one-round outcome. Low L=8 and high L=64 share only the nested selection
  prefix; neither shares draws with the evaluation stream.
* Held-out confirmation: 512 paired independent IC worlds per state/budget.
  Same world is used across arms; after the observed next-round feedback each
  arm is allowed to choose its own final-round seeds. Report outcomes paired
  by base state, with b=1/2 handled as a cluster (not 2 independent states),
  95% cluster-bootstrap intervals and full per-stratum results. Measure
  decision time separately from confirmation time. Primary contrast is
  high-SOF minus frozen best cheap, averaged over four strata and two budgets.
  Low-SOF minus cheap and high-minus-low are secondary.
* Stop: if cheap/low SOF closes the gap, or high SOF gives no practical gain.
  Uncertain: wide interval, inconsistent strata, or differences only at this
  narrow parameter choice. Only a stable positive high-SOF gain *and*
  substantial decision cost would justify a later learning proxy probe.

Implementation limitations must be reported: tiny graphs, two probabilities,
two budgets, two rounds, finite Monte Carlo node selection; FF and SOF are
literal local approximations to the published policy forms, not verified
against author code. No training is authorized by this plan.
