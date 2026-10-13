# Contribution and method-source ledger

Checked against primary sources on 2026-10-13. This is a scoped comparison, not exhaustive priority certification.

| Claim | Existing source | Our use and remaining boundary |
|---|---|---|
| Hybrid continuous and topology control | LJN official agent | Retained; not novel |
| Learned action ranking / Top-K proposal | LJN; SoftGNN; Generalizable GNNs | Replaceable auxiliary; incremental GNN benefit over frequency prior not established |
| Simulating candidates before execution | All three | Shared legality/feedback interface; not novel |
| Stop at first forecast below 0.9 | SoftGNN §4.4 | Primary reference, with stronger exhausted best-admissible fallback; not full published reproduction |
| Mask disconnected topology endpoints | SoftGNN §4.4.2 | Shared preprocessing; not novel |
| Progressive feedback stopping | General adaptive search idea | Exact frozen schedule is an implementation choice, not a new general theory |
| Complete incumbent's original-substation aliases before an extended stop | Proposed framework rule | Concrete difference to test; conditional key protection and query scope; broader literature still needs final review |
| Reuse compiled DC/safeQP problems | Conventional optimization engineering | Shared by all methods; effect reported separately |

Sources:

1. https://github.com/lajavaness/l2rpn-2023-ljn-agent
2. https://arxiv.org/html/2503.15190v2#S4.SS4
3. https://arxiv.org/html/2501.07186v1#S4.SS9

Broad web searches for progressive power-grid topology search and substation stopping did not identify an exact counterpart in the inspected sources. That does not establish absence or justify a first-ever claim. Unrelated search hits are not treated as evidence or added as baselines.
