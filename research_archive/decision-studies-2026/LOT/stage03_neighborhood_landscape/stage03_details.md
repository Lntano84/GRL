# stage03_details.md

Local release-benefit landscape. Produced by `python run.py` in `stage03_neighborhood_landscape`.

Environment: python 3.13.14, numpy 2.5.2, scipy 1.17.1.

Frozen: `audited_v1`, original disruptions, original repaired reference S^r, tau=2, kappa=2 for every local solve, 30 s cap per subproblem.

`F(S) = J(empty) - J(S)`; the baseline is `J(empty)`, not the repair cost.


## 0. Main table

| 状态 | 空集合 | 最优单变量 | 最优双变量 | 单变量排序 | 两步贪心 | 缺货优先 | 缺货+容量释放 | 贪心遗憾 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0|D1_m0_1p | 28 | 10 | 10 | 10 | 10 | 10 | 10 | 0 |
| S0|D2_m1_2p | 105 | 47 | 38 | 38 | 38 | 38 | 48 | 0 |
| S0|D3_both_1p | 93 | 65 | 65 | 65 | 65 | 65 | 65 | 0 |
| S1|D1_m0_1p | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0 |
| S1|D2_m1_2p | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0 |
| S1|D3_both_1p | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0 |
| S2|D1_m0_1p | 8 | 8 | 8 | 8 | 8 | 8 | 8 | 0 |
| S2|D2_m1_2p | 66 | 8 | 8 | 8 | 8 | 8 | 8 | 0 |
| S2|D3_both_1p | 46 | 46 | 46 | 46 | 46 | 46 | 46 | 0 |
| S3|D1_m0_1p | 5 | 5 | 5 | 5 | 5 | 5 | 5 | 0 |
| S3|D2_m1_2p | 62 | 5 | 5 | 5 | 5 | 5 | 5 | 0 |
| S3|D3_both_1p | 62 | 62 | 62 | 62 | 62 | 62 | 62 | 0 |

## 1. Cross-checks against stage 02

| 状态 | stage02 kappa=1 | 本轮最优单变量 | 一致 | stage02 kappa=2 | 本轮最优双变量 | 一致 |
|---|---:|---:|---|---:|---:|---|
| S0|D1_m0_1p | 10 | 10 | ok | 10 | 10 | ok |
| S0|D2_m1_2p | 47 | 47 | ok | 38 | 38 | ok |
| S0|D3_both_1p | 65 | 65 | ok | 65 | 65 | ok |
| S1|D1_m0_1p | 5 | 5 | ok | 5 | 5 | ok |
| S1|D2_m1_2p | 5 | 5 | ok | 5 | 5 | ok |
| S1|D3_both_1p | 5 | 5 | ok | 5 | 5 | ok |
| S2|D1_m0_1p | 8 | 8 | ok | 8 | 8 | ok |
| S2|D2_m1_2p | 8 | 8 | ok | 8 | 8 | ok |
| S2|D3_both_1p | 46 | 46 | ok | 46 | 46 | ok |
| S3|D1_m0_1p | 5 | 5 | ok | 5 | 5 | ok |
| S3|D2_m1_2p | 5 | 5 | ok | 5 | 5 | ok |
| S3|D3_both_1p | 62 | 62 | ok | 62 | 62 | ok |

## 2. Selected indices and diagnostics

| 状态 | 最优单变量 | 最优双变量 | R1 | 贪心首选 | 首选并列数 | 并列最好/最差 | 互补对数 | max C |
|---|---|---|---:|---|---:|---|---:|---:|
| S0|D1_m0_1p | `[0, 1, 1]` | `[[0, 0, 1], [0, 1, 1]]` | 0 | `[0, 1, 1]` | 1 | 10 / 10 | 0 | 0 |
| S0|D2_m1_2p | `[1, 0, 2]` | `[[1, 0, 1], [1, 0, 2]]` | 9 | `[1, 0, 2]` | 1 | 38 / 38 | 0 | 8 |
| S0|D3_both_1p | `[0, 0, 2]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 2]` | 1 | 65 / 65 | 1 | 15 |
| S1|D1_m0_1p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 5 / 5 | 0 | 0 |
| S1|D2_m1_2p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 5 / 5 | 0 | 0 |
| S1|D3_both_1p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 5 / 5 | 0 | 0 |
| S2|D1_m0_1p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 8 / 8 | 0 | 0 |
| S2|D2_m1_2p | `[1, 0, 1]` | `[[0, 0, 1], [1, 0, 1]]` | 0 | `[1, 0, 1]` | 1 | 8 / 8 | 0 | 0 |
| S2|D3_both_1p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 46 / 46 | 0 | 0 |
| S3|D1_m0_1p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 5 / 5 | 0 | 0 |
| S3|D2_m1_2p | `[1, 0, 1]` | `[[0, 0, 1], [1, 0, 1]]` | 0 | `[1, 0, 1]` | 1 | 5 / 5 | 0 | 0 |
| S3|D3_both_1p | `[0, 0, 1]` | `[[0, 0, 1], [0, 0, 2]]` | 0 | `[0, 0, 1]` | 8 | 62 / 62 | 0 | 0 |

### Greedy tie exposure

Greedy commits to ONE first variable and only ever examines pairs containing it. If the optimal pair is not reachable from any tied first choice, a greedy miss would be a tie-breaking artefact, not evidence about joint selection.

| 状态 | 首选并列数 | 从并列首选可达的变量对 | 全部变量对 | 最优对可达 |
|---|---:|---:|---:|---|
| S0|D1_m0_1p | 1 | 7 | 28 | yes |
| S0|D2_m1_2p | 1 | 7 | 28 | yes |
| S0|D3_both_1p | 1 | 7 | 28 | yes |
| S1|D1_m0_1p | 8 | 28 | 28 | yes |
| S1|D2_m1_2p | 8 | 28 | 28 | yes |
| S1|D3_both_1p | 8 | 28 | 28 | yes |
| S2|D1_m0_1p | 8 | 28 | 28 | yes |
| S2|D2_m1_2p | 1 | 7 | 28 | yes |
| S2|D3_both_1p | 8 | 28 | 28 | yes |
| S3|D1_m0_1p | 8 | 28 | 28 | yes |
| S3|D2_m1_2p | 1 | 7 | 28 | yes |
| S3|D3_both_1p | 8 | 28 | 28 | yes |

The reachable count is only 7 of 28 pairs when there is a single tied first choice, so the exposure is real even though it did not bite on these states.


## 3. Query counts (deployment cost is NOT the measured runtime)

| 方法 | 规则 | 每状态查询数 |
|---|---|---:|
| 最优双变量组 | 穷举所有 |S|<=2 | 37 |
| 最优单变量 | 穷举所有单变量 | 8 |
| 单变量收益排序 | 单变量打分后取前二 | 9 |
| 两步贪心 | 先最佳单变量，再加最佳第二变量 | 16 |
| 缺货优先 | 可见分数取前二，0 次额外求解 | 1 |
| 缺货+容量释放 | 缺货优先 + 同机同期搭档，0 次额外求解 | 1 |

The last two consult no subproblem at all; their `1` is the single evaluation of the chosen set. Measured wall time in the JSON is the cost of BUILDING the table and must not be reported as the deployment cost of any method.


## 4. Superadditivity and complementary pairs

- **S0|D1_m0_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S0|D2_m1_2p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 8
    - `[0, 0, 2]`+`[1, 0, 1]`: F=65, F_u=0, F_v=57, C=8
- **S0|D3_both_1p**: complementary pairs (both singles zero, pair positive) = 1; largest C = 15
    - `[0, 1, 2]`+`[1, 0, 2]`: F=15, F_u=0, F_v=0, C=15  <-- complementary
- **S1|D1_m0_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S1|D2_m1_2p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S1|D3_both_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S2|D1_m0_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S2|D2_m1_2p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S2|D3_both_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S3|D1_m0_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S3|D2_m1_2p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0
- **S3|D3_both_1p**: complementary pairs (both singles zero, pair positive) = 0; largest C = 0

## 5. Joint-benefit cases: what actually changed

Cases where `R1 > 0` (a second variable helps). The comparison is against the best SINGLE release, so the columns show what the second variable buys.

| 状态 | 集合 | 成本 | F(S) | 实际翻转位置 | κ 内翻转数 |
|---|---|---:|---:|---|---:|
| S0|D2_m1_2p | empty `[]` | 105 | 0 | `[]` | 0 |
| S0|D2_m1_2p | best single `[[1, 0, 2]]` | 47 | 58 | `[[1, 0, 2]]` | 1 |
| S0|D2_m1_2p | best pair `[[1, 0, 1], [1, 0, 2]]` | 38 | 67 | `[[1, 0, 1], [1, 0, 2]]` | 2 |

## 6. Complementary pairs (both singles worthless, pair valuable)

- **S0|D3_both_1p**: 1 complementary pair(s)
    - `[0, 1, 2]` + `[1, 0, 2]`: F(single)=0, F(single)=0, F(pair)=15, C=15
      greedy regret on this state = 0; optimal pair reachable from a tied first choice = True
