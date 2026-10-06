# stage04_details.md

Machine-generated detail tables. Produced by `stage04_run.py`.

Environment: python 3.13.14, numpy 2.5.2, scipy 1.17.1, Windows-11-10.0.26200-SP0

Model `audited_v1`, kappa = 4, tau per scale {'small': 3, 'medium': 6, 'large': 6}. `scipy.optimize.milp` has NO MIP-start parameter, so no run is warm started.


## 0. Nominal solves

| instance | N | M | T | obj | gap | proven optimal | feasible |
|---|---:|---:|---:|---:|---:|---|---|
| large_rho0.75_s0 | 24 | 3 | 18 | 5865.12 | 0.843595 | False | True |
| large_rho0.75_s1 | 24 | 3 | 18 | 3962.88 | 0.747086 | False | True |
| large_rho1.10_s0 | 24 | 3 | 18 | 7671.97 | 0.582893 | False | True |
| large_rho1.10_s1 | 24 | 3 | 18 | 6892.87 | 0.477326 | False | True |
| medium_rho0.75_s0 | 12 | 3 | 12 | 298.772 | 0.377633 | False | True |
| medium_rho0.75_s1 | 12 | 3 | 12 | 333.476 | 0.32671 | False | True |
| medium_rho1.10_s0 | 12 | 3 | 12 | 2192.74 | 0.227529 | False | True |
| medium_rho1.10_s1 | 12 | 3 | 12 | 2166.31 | 0.210751 | False | True |
| small_rho0.75_s0 | 6 | 2 | 6 | 84.1725 | 0.142589 | False | True |
| small_rho0.75_s1 | 6 | 2 | 6 | 81.0674 | 0.18551 | False | True |
| small_rho1.10_s0 | 6 | 2 | 6 | 463.241 | 0 | True | True |
| small_rho1.10_s1 | 6 | 2 | 6 | 769.113 | 9.78661e-05 | True | True |

## 1. Per-state 20 s results

| state | scale | rho | dis | nom | repair | FULL | EMPTY | S-12 | S-24 | J_best | src | FULL bound |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| large_rho0.75_s0|D1_m0_2p | large | 0.75 | D1_m0_2p | 5865.12 | 6656.22 | 13458.5 | 12937.4 | 11077.4 | 11025.7 | 2574.43 | FULL@120s | 2087.93 |
| large_rho0.75_s0|D2_all_1p | large | 0.75 | D2_all_1p | 5865.12 | 7595.56 | 12650 | 14548.4 | 12652.1 | 17618.9 | 3627.24 | FULL@120s | 2833.35 |
| large_rho0.75_s1|D1_m0_2p | large | 0.75 | D1_m0_2p | 3962.88 | 4691.75 | 13063.3 | 13563.5 | 14248.3 | 14855.3 | 2492.94 | FULL@120s | 1949.72 |
| large_rho0.75_s1|D2_all_1p | large | 0.75 | D2_all_1p | 3962.88 | 6051.6 | 14673 | 14154 | 13970.6 | 14938.5 | 3783.81 | FULL@120s | 2909.61 |
| large_rho1.10_s0|D1_m0_2p | large | 1.10 | D1_m0_2p | 7671.97 | 9394.14 | 24492.9 | 7468.92 | 19642.5 | 19198.8 | 5640.01 | FULL@120s | 4579.1 |
| large_rho1.10_s0|D2_all_1p | large | 1.10 | D2_all_1p | 7671.97 | 11554.2 | 23940.3 | 13455.9 | 18390 | 13082.5 | 6558.75 | FULL@120s | 5602.72 |
| large_rho1.10_s1|D1_m0_2p | large | 1.10 | D1_m0_2p | 6892.87 | 8675.63 | 20672.3 | 7969.47 | 7431.94 | 7699.25 | 6246.36 | FULL@120s | 5041.65 |
| large_rho1.10_s1|D2_all_1p | large | 1.10 | D2_all_1p | 6892.87 | 10795.1 | 19608.1 | 7760.58 | 8430.73 | 9841.73 | 6735.11 | FULL@120s | 5905.51 |
| medium_rho0.75_s0|D1_m0_2p | medium | 0.75 | D1_m0_2p | 298.772 | 1687.28 | 927.111 | 1625.01 | 1360.09 | 1310.7 | 927.111 | FULL@20s | 907.931 |
| medium_rho0.75_s0|D2_all_1p | medium | 0.75 | D2_all_1p | 298.772 | 2321.32 | 1668.15 | 1827.6 | 1771.56 | 1755.9 | 1666.25 | FULL@120s | 1640.67 |
| medium_rho0.75_s1|D1_m0_2p | medium | 0.75 | D1_m0_2p | 333.476 | 2045.67 | 949.003 | 1741.85 | 1526.39 | 1587.81 | 931.338 | FULL@120s | 889.355 |
| medium_rho0.75_s1|D2_all_1p | medium | 0.75 | D2_all_1p | 333.476 | 2269.96 | 1879.89 | 2090.32 | 1826.85 | 1749.99 | 1733.57 | FULL@120s | 1658.24 |
| medium_rho1.10_s0|D1_m0_2p | medium | 1.10 | D1_m0_2p | 2192.74 | 3913.56 | 3072.25 | 3530.68 | 3528.98 | 3511.97 | 3024.49 | FULL@120s | 2865.87 |
| medium_rho1.10_s0|D2_all_1p | medium | 1.10 | D2_all_1p | 2192.74 | 5852.4 | 4053.38 | 4695.04 | 4311.35 | 4301.06 | 4031.65 | FULL@120s | 3809.83 |
| medium_rho1.10_s1|D1_m0_2p | medium | 1.10 | D1_m0_2p | 2166.31 | 3990.98 | 3114.31 | 3714.49 | 3166.96 | 3072.79 | 3021.79 | FULL@120s | 2812 |
| medium_rho1.10_s1|D2_all_1p | medium | 1.10 | D2_all_1p | 2166.31 | 5774.5 | 4110.14 | 4399.03 | 4091.09 | 4079.68 | 4053.91 | FULL@120s | 3847.69 |
| small_rho0.75_s0|D1_m0_2p | small | 0.75 | D1_m0_2p | 84.1725 | 1025.59 | 655.126 | 886.127 | 763.749 | 655.126 | 655.126 | FULL@20s | 655.083 |
| small_rho0.75_s0|D2_all_1p | small | 0.75 | D2_all_1p | 84.1725 | 1618.49 | 1110.14 | 1446.48 | 1184.95 | 1110.14 | 1110.14 | FULL@20s | 1110.04 |
| small_rho0.75_s1|D1_m0_2p | small | 0.75 | D1_m0_2p | 81.0674 | 1392.27 | 675.704 | 1222.52 | 822.826 | 675.704 | 675.704 | FULL@20s | 675.64 |
| small_rho0.75_s1|D2_all_1p | small | 0.75 | D2_all_1p | 81.0674 | 1263.45 | 976.954 | 1054.99 | 1034.92 | 976.954 | 976.954 | FULL@20s | 976.872 |
| small_rho1.10_s0|D1_m0_2p | small | 1.10 | D1_m0_2p | 463.241 | 2171.56 | 1346.32 | 2082.47 | 1698.6 | 1346.32 | 1346.32 | FULL@20s | 1346.32 |
| small_rho1.10_s0|D2_all_1p | small | 1.10 | D2_all_1p | 463.241 | 2000.17 | 1796.21 | 1878.41 | 1868.32 | 1796.21 | 1796.21 | FULL@20s | 1796.04 |
| small_rho1.10_s1|D1_m0_2p | small | 1.10 | D1_m0_2p | 769.113 | 2208.5 | 1780.48 | 2164.28 | 2143.27 | 1780.48 | 1780.48 | FULL@20s | 1780.48 |
| small_rho1.10_s1|D2_all_1p | small | 1.10 | D2_all_1p | 769.113 | 2897.62 | 2069.8 | 2584.06 | 2138.55 | 2069.8 | 2069.8 | FULL@20s | 2069.8 |

## 2. gap_best by budget, all methods


## 1b. Scale / method / budget summary

Mean and worst `gap_best` over the states of each scale. A state where the method fell back to the repaired plan is counted at its fallback gap.

| scale | method | budget | states | mean gap | worst gap | proven optimal | fallbacks |
|---|---|---:|---:|---:|---:|---:|---:|
| small | FULL | 1 | 8 | 0.0103 | 0.0433 | 0 | 0 |
| small | FULL | 5 | 8 | 0.0006 | 0.0029 | 5 | 0 |
| small | FULL | 20 | 8 | 0.0000 | 0.0000 | 8 | 0 |
| small | EMPTY | 1 | 8 | 0.3254 | 0.8093 | 1 | 0 |
| small | EMPTY | 5 | 8 | 0.3252 | 0.8093 | 8 | 0 |
| small | EMPTY | 20 | 8 | 0.3252 | 0.8093 | 8 | 0 |
| small | SHORTAGE-12 | 1 | 8 | 0.1344 | 0.2617 | 0 | 0 |
| small | SHORTAGE-12 | 5 | 8 | 0.1312 | 0.2617 | 6 | 0 |
| small | SHORTAGE-12 | 20 | 8 | 0.1311 | 0.2617 | 8 | 0 |
| small | SHORTAGE-24 | 1 | 8 | 0.0103 | 0.0433 | 0 | 0 |
| small | SHORTAGE-24 | 5 | 8 | 0.0006 | 0.0029 | 5 | 0 |
| small | SHORTAGE-24 | 20 | 8 | 0.0000 | 0.0000 | 8 | 0 |
| medium | FULL | 1 | 8 | 2.4935 | 4.9844 | 0 | 0 |
| medium | FULL | 5 | 8 | 0.0250 | 0.0849 | 0 | 0 |
| medium | FULL | 20 | 8 | 0.0213 | 0.0844 | 0 | 0 |
| medium | EMPTY | 1 | 8 | 0.8426 | 2.7012 | 0 | 0 |
| medium | EMPTY | 5 | 8 | 0.3324 | 0.8857 | 0 | 0 |
| medium | EMPTY | 20 | 8 | 0.3215 | 0.8703 | 0 | 0 |
| medium | SHORTAGE-12 | 1 | 8 | 2.2135 | 4.5594 | 0 | 0 |
| medium | SHORTAGE-12 | 5 | 8 | 0.1909 | 0.6389 | 0 | 0 |
| medium | SHORTAGE-12 | 20 | 8 | 0.1895 | 0.6389 | 0 | 0 |
| medium | SHORTAGE-24 | 1 | 8 | 1.9708 | 4.9921 | 0 | 0 |
| medium | SHORTAGE-24 | 5 | 8 | 0.1872 | 0.7049 | 0 | 0 |
| medium | SHORTAGE-24 | 20 | 8 | 0.1791 | 0.7049 | 0 | 0 |
| large | FULL | 1 | 8 | 2.6712 | 4.2401 | 0 | 1 |
| large | FULL | 5 | 8 | 3.0059 | 4.2401 | 0 | 0 |
| large | FULL | 20 | 8 | 3.0059 | 4.2401 | 0 | 0 |
| large | EMPTY | 1 | 8 | 2.9358 | 4.4407 | 0 | 0 |
| large | EMPTY | 5 | 8 | 2.9358 | 4.4407 | 0 | 0 |
| large | EMPTY | 20 | 8 | 2.0027 | 4.4407 | 0 | 0 |
| large | SHORTAGE-12 | 1 | 8 | 2.4136 | 4.7155 | 0 | 1 |
| large | SHORTAGE-12 | 5 | 8 | 2.6407 | 4.7155 | 0 | 0 |
| large | SHORTAGE-12 | 20 | 8 | 2.2408 | 4.7155 | 0 | 0 |
| large | SHORTAGE-24 | 1 | 8 | 3.0365 | 4.9590 | 0 | 0 |
| large | SHORTAGE-24 | 5 | 8 | 3.0365 | 4.9590 | 0 | 0 |
| large | SHORTAGE-24 | 20 | 8 | 2.3925 | 4.9590 | 0 | 0 |

### Per state

| state | method | 1s | 5s | 20s |
|---|---|---:|---:|---:|
| large_rho0.75_s0|D1_m0_2p | FULL | 4.2278 | 4.2278 | 4.2278 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 4.0253 | 4.0253 | 4.0253 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 3.3029 | 3.3029 | 3.3029 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 3.2828 | 3.2828 | 3.2828 |
| large_rho0.75_s0|D2_all_1p | FULL | 2.4875 | 2.4875 | 2.4875 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 3.0109 | 3.0109 | 3.0109 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 2.4881 | 2.4881 | 2.4881 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 3.8574 | 3.8574 | 3.8574 |
| large_rho0.75_s1|D1_m0_2p | FULL | 4.2401 | 4.2401 | 4.2401 |
| large_rho0.75_s1|D1_m0_2p | EMPTY | 4.4407 | 4.4407 | 4.4407 |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 4.7155 | 4.7155 | 4.7155 |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 4.9590 | 4.9590 | 4.9590 |
| large_rho0.75_s1|D2_all_1p | FULL | 2.8778 | 2.8778 | 2.8778 |
| large_rho0.75_s1|D2_all_1p | EMPTY | 2.7407 | 2.7407 | 2.7407 |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 2.6922 | 2.6922 | 2.6922 |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 2.9480 | 2.9480 | 2.9480 |
| large_rho1.10_s0|D1_m0_2p | FULL | 0.6656 | 3.3427 | 3.3427 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 2.9034 | 2.9034 | 0.3243 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0.6656 | 2.4827 | 2.4827 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 2.4040 | 2.4040 | 2.4040 |
| large_rho1.10_s0|D2_all_1p | FULL | 2.6501 | 2.6501 | 2.6501 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 2.2122 | 2.2122 | 1.0516 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1.8039 | 1.8039 | 1.8039 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 2.3134 | 2.3134 | 0.9947 |
| large_rho1.10_s1|D1_m0_2p | FULL | 2.3095 | 2.3095 | 2.3095 |
| large_rho1.10_s1|D1_m0_2p | EMPTY | 2.4436 | 2.4436 | 0.2759 |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 2.2198 | 2.2198 | 0.1898 |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 2.4917 | 2.4917 | 0.2326 |
| large_rho1.10_s1|D2_all_1p | FULL | 1.9113 | 1.9113 | 1.9113 |
| large_rho1.10_s1|D2_all_1p | EMPTY | 1.7097 | 1.7097 | 0.1523 |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 1.4206 | 1.4206 | 0.2518 |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 2.0353 | 2.0353 | 0.4613 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 4.3778 | 0.0017 | 0.0000 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 0.9192 | 0.7829 | 0.7528 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 2.9005 | 0.4670 | 0.4670 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 4.9921 | 0.4306 | 0.4137 |
| medium_rho0.75_s0|D2_all_1p | FULL | 2.9272 | 0.0059 | 0.0011 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 0.3187 | 0.1031 | 0.0968 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 4.5594 | 0.0632 | 0.0632 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 2.3269 | 0.0538 | 0.0538 |
| medium_rho0.75_s1|D1_m0_2p | FULL | 4.9844 | 0.0190 | 0.0190 |
| medium_rho0.75_s1|D1_m0_2p | EMPTY | 2.7012 | 0.8857 | 0.8703 |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 2.3172 | 0.6389 | 0.6389 |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 1.0445 | 0.7049 | 0.7049 |
| medium_rho0.75_s1|D2_all_1p | FULL | 2.4332 | 0.0849 | 0.0844 |
| medium_rho0.75_s1|D2_all_1p | EMPTY | 1.0376 | 0.2131 | 0.2058 |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 1.8038 | 0.0538 | 0.0538 |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 3.1280 | 0.0095 | 0.0095 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 1.4809 | 0.0332 | 0.0158 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 0.6728 | 0.1737 | 0.1674 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0.2456 | 0.1739 | 0.1668 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 2.1164 | 0.1949 | 0.1612 |
| medium_rho1.10_s0|D2_all_1p | FULL | 0.5215 | 0.0054 | 0.0054 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 0.2783 | 0.1714 | 0.1645 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1.3538 | 0.0694 | 0.0694 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 0.7113 | 0.0668 | 0.0668 |
| medium_rho1.10_s1|D1_m0_2p | FULL | 1.7097 | 0.0306 | 0.0306 |
| medium_rho1.10_s1|D1_m0_2p | EMPTY | 0.3442 | 0.2292 | 0.2292 |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 1.9593 | 0.0480 | 0.0480 |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 0.0895 | 0.0295 | 0.0169 |
| medium_rho1.10_s1|D2_all_1p | FULL | 1.5135 | 0.0197 | 0.0139 |
| medium_rho1.10_s1|D2_all_1p | EMPTY | 0.4692 | 0.0997 | 0.0851 |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 2.5683 | 0.0125 | 0.0092 |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 1.3576 | 0.0076 | 0.0064 |
| small_rho0.75_s0|D1_m0_2p | FULL | 0.0026 | 0.0000 | 0.0000 |
| small_rho0.75_s0|D1_m0_2p | EMPTY | 0.3539 | 0.3526 | 0.3526 |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 0.1687 | 0.1658 | 0.1658 |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 0.0026 | 0.0000 | 0.0000 |
| small_rho0.75_s0|D2_all_1p | FULL | 0.0029 | 0.0029 | 0.0000 |
| small_rho0.75_s0|D2_all_1p | EMPTY | 0.3030 | 0.3030 | 0.3030 |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 0.0684 | 0.0674 | 0.0674 |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 0.0029 | 0.0029 | 0.0000 |
| small_rho0.75_s1|D1_m0_2p | FULL | 0.0008 | 0.0000 | 0.0000 |
| small_rho0.75_s1|D1_m0_2p | EMPTY | 0.8093 | 0.8093 | 0.8093 |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 0.2316 | 0.2177 | 0.2177 |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 0.0008 | 0.0000 | 0.0000 |
| small_rho0.75_s1|D2_all_1p | FULL | 0.0031 | 0.0000 | 0.0000 |
| small_rho0.75_s1|D2_all_1p | EMPTY | 0.0803 | 0.0799 | 0.0799 |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 0.0637 | 0.0595 | 0.0593 |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 0.0031 | 0.0000 | 0.0000 |
| small_rho1.10_s0|D1_m0_2p | FULL | 0.0000 | 0.0000 | 0.0000 |
| small_rho1.10_s0|D1_m0_2p | EMPTY | 0.5468 | 0.5468 | 0.5468 |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0.2617 | 0.2617 | 0.2617 |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 0.0000 | 0.0000 | 0.0000 |
| small_rho1.10_s0|D2_all_1p | FULL | 0.0072 | 0.0020 | 0.0000 |
| small_rho1.10_s0|D2_all_1p | EMPTY | 0.0458 | 0.0458 | 0.0458 |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 0.0401 | 0.0401 | 0.0401 |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 0.0072 | 0.0020 | 0.0000 |
| small_rho1.10_s1|D1_m0_2p | FULL | 0.0433 | 0.0000 | 0.0000 |
| small_rho1.10_s1|D1_m0_2p | EMPTY | 0.2156 | 0.2156 | 0.2156 |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 0.2079 | 0.2038 | 0.2038 |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 0.0433 | 0.0000 | 0.0000 |
| small_rho1.10_s1|D2_all_1p | FULL | 0.0227 | 0.0000 | 0.0000 |
| small_rho1.10_s1|D2_all_1p | EMPTY | 0.2485 | 0.2485 | 0.2485 |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 0.0332 | 0.0332 | 0.0332 |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 0.0227 | 0.0000 | 0.0000 |

## 3. Wall clock versus requested budget

| state | method | budget | wall | solver wall | feature+rank | n fixed | free Y | free Z | fallback |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| large_rho0.75_s0|D1_m0_2p | FULL | 20 | 20.02 | 20.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | FULL | 5 | 5.05 | 5.05 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 20 | 20.09 | 20.09 | 0.002 | 408 | 888 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 5 | 5.02 | 5.02 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 5 | 5.02 | 5.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 20 | 20.02 | 20.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 5 | 5.02 | 5.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1.02 | 1.02 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 20 | 20.02 | 20.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | FULL | 1 | 1.02 | 1.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 1 | 1.02 | 1.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | FULL | 20 | 20.02 | 20.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | FULL | 5 | 5.02 | 5.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 20 | 20.02 | 20.02 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 5 | 5.07 | 5.07 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 5 | 5.04 | 5.03 | 0.001 | 420 | 876 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | EMPTY | 20 | 20.02 | 20.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | EMPTY | 5 | 5.07 | 5.07 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | 1.08 | 1.08 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 20 | 20.06 | 20.06 | 0.002 | 420 | 876 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | FULL | 1 | 1.02 | 1.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s0|D2_all_1p | EMPTY | 1 | 1.07 | 1.07 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | FULL | 20 | 20.02 | 20.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | FULL | 5 | 5.14 | 5.14 | 0.002 | 0 | 1296 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 20 | 20.06 | 20.06 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 5 | 5.02 | 5.02 | 0.002 | 408 | 888 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 5 | 5.02 | 5.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | EMPTY | 20 | 20.09 | 20.09 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | EMPTY | 5 | 5.10 | 5.10 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 1 | 1.04 | 1.04 | 0.002 | 420 | 876 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 1 | 1.07 | 1.07 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 20 | 20.02 | 20.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | FULL | 1 | 1.02 | 1.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s1|D1_m0_2p | EMPTY | 1 | 1.02 | 1.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | FULL | 20 | 20.02 | 20.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | FULL | 5 | 5.02 | 5.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 20 | 20.03 | 20.03 | 0.001 | 408 | 888 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 5 | 5.03 | 5.03 | 0.002 | 408 | 888 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 5 | 5.02 | 5.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | EMPTY | 20 | 20.02 | 20.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | EMPTY | 5 | 5.03 | 5.03 | 0.000 | 432 | 864 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 1 | 1.02 | 1.02 | 0.002 | 408 | 888 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 20 | 20.05 | 20.05 | 0.002 | 420 | 876 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | FULL | 1 | 1.02 | 1.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho0.75_s1|D2_all_1p | EMPTY | 1 | 1.02 | 1.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | FULL | 20 | 20.03 | 20.03 | 0.002 | 0 | 1296 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | FULL | 5 | 5.03 | 5.02 | 0.002 | 0 | 1296 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 20 | 20.09 | 20.09 | 0.001 | 408 | 888 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 5 | 5.04 | 5.04 | 0.002 | 408 | 888 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 5 | 5.03 | 5.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 20 | 20.02 | 20.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 5 | 5.05 | 5.05 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1.03 | 1.03 | 0.002 | 420 | 876 | 1296 | YES |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1.02 | 1.02 | 0.002 | 408 | 888 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 20 | 20.02 | 20.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho1.10_s0|D1_m0_2p | FULL | 1 | 1.04 | 1.04 | 0.001 | 0 | 1296 | 1296 | YES |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 1 | 1.05 | 1.05 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | FULL | 20 | 20.03 | 20.03 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | FULL | 5 | 5.06 | 5.06 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 20 | 20.04 | 20.04 | 0.001 | 408 | 888 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 5 | 5.03 | 5.02 | 0.001 | 408 | 888 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 5 | 5.02 | 5.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | EMPTY | 20 | 20.03 | 20.03 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | EMPTY | 5 | 5.02 | 5.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.002 | 420 | 876 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | 1.06 | 1.06 | 0.002 | 408 | 888 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 20 | 20.06 | 20.06 | 0.002 | 420 | 876 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | FULL | 1 | 1.08 | 1.08 | 0.002 | 0 | 1296 | 1296 | no |
| large_rho1.10_s0|D2_all_1p | EMPTY | 1 | 1.02 | 1.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | FULL | 20 | 20.11 | 20.11 | 0.002 | 0 | 1296 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | FULL | 5 | 5.07 | 5.07 | 0.002 | 0 | 1296 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 20 | 20.02 | 20.02 | 0.002 | 408 | 888 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 5 | 5.06 | 5.06 | 0.001 | 408 | 888 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 5 | 5.02 | 5.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | EMPTY | 20 | 20.03 | 20.03 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | EMPTY | 5 | 5.02 | 5.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 1 | 1.04 | 1.04 | 0.001 | 408 | 888 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 20 | 20.02 | 20.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | FULL | 1 | 1.10 | 1.10 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho1.10_s1|D1_m0_2p | EMPTY | 1 | 1.05 | 1.05 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | FULL | 20 | 20.06 | 20.06 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | FULL | 5 | 5.05 | 5.05 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 20 | 20.03 | 20.03 | 0.002 | 408 | 888 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 5 | 5.02 | 5.02 | 0.002 | 408 | 888 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 5 | 5.02 | 5.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | EMPTY | 20 | 20.02 | 20.02 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | EMPTY | 5 | 5.06 | 5.06 | 0.000 | 432 | 864 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 1 | 1.04 | 1.04 | 0.001 | 420 | 876 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 1 | 1.06 | 1.06 | 0.001 | 408 | 888 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 20 | 20.02 | 20.02 | 0.001 | 420 | 876 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | FULL | 1 | 1.02 | 1.02 | 0.001 | 0 | 1296 | 1296 | no |
| large_rho1.10_s1|D2_all_1p | EMPTY | 1 | 1.07 | 1.07 | 0.000 | 432 | 864 | 1296 | no |
| medium_rho0.75_s0|D1_m0_2p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | FULL | 5 | 5.02 | 5.02 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 20 | 20.02 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1.01 | 1.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 1 | 1.01 | 1.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | FULL | 5 | 5.03 | 5.03 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 20 | 20.01 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 5 | 5.03 | 5.03 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | FULL | 1 | 1.02 | 1.02 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 1 | 1.02 | 1.02 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | EMPTY | 20 | 20.02 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s1|D1_m0_2p | EMPTY | 1 | 1.02 | 1.02 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 20 | 20.02 | 20.02 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | EMPTY | 20 | 20.01 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 1 | 1.02 | 1.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 1 | 1.02 | 1.02 | 0.001 | 192 | 240 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | FULL | 1 | 1.03 | 1.03 | 0.000 | 0 | 432 | 432 | no |
| medium_rho0.75_s1|D2_all_1p | EMPTY | 1 | 1.01 | 1.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 20 | 20.01 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1.01 | 1.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 20 | 20.02 | 20.02 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 1 | 1.01 | 1.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.000 | 204 | 228 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 20 | 20.01 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | 1.02 | 1.02 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 1 | 1.01 | 1.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.001 | 192 | 240 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | EMPTY | 20 | 20.01 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 1 | 1.01 | 1.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s1|D1_m0_2p | EMPTY | 1 | 1.01 | 1.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | FULL | 20 | 20.01 | 20.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 20 | 20.01 | 20.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 5 | 5.01 | 5.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 5 | 5.01 | 5.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | EMPTY | 20 | 20.01 | 20.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | EMPTY | 5 | 5.01 | 5.01 | 0.000 | 216 | 216 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 1 | 1.01 | 1.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.000 | 192 | 240 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 20 | 20.01 | 20.01 | 0.001 | 204 | 228 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 432 | 432 | no |
| medium_rho1.10_s1|D2_all_1p | EMPTY | 1 | 1.01 | 1.01 | 0.000 | 216 | 216 | 432 | no |
| small_rho0.75_s0|D1_m0_2p | FULL | 20 | 2.53 | 2.53 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | FULL | 5 | 2.72 | 2.72 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 20 | 2.73 | 2.73 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 5 | 2.72 | 2.72 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 5 | 2.92 | 2.92 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | EMPTY | 20 | 1.93 | 1.93 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | EMPTY | 5 | 1.93 | 1.92 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1.00 | 1.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1.00 | 1.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 20 | 2.88 | 2.88 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | FULL | 1 | 1.00 | 1.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s0|D1_m0_2p | EMPTY | 1 | 1.00 | 1.00 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s0|D2_all_1p | FULL | 20 | 7.16 | 7.16 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s0|D2_all_1p | FULL | 5 | 5.00 | 5.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 20 | 7.18 | 7.18 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 5 | 5.00 | 5.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 5 | 3.95 | 3.95 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s0|D2_all_1p | EMPTY | 20 | 2.25 | 2.25 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s0|D2_all_1p | EMPTY | 5 | 2.25 | 2.25 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | 1.01 | 1.01 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | 1.00 | 1.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 20 | 3.95 | 3.95 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s0|D2_all_1p | FULL | 1 | 1.00 | 1.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s0|D2_all_1p | EMPTY | 1 | 1.00 | 1.00 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | FULL | 20 | 2.80 | 2.80 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | FULL | 5 | 2.81 | 2.81 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 20 | 2.82 | 2.82 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 5 | 2.80 | 2.80 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 5 | 5.00 | 5.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | EMPTY | 20 | 4.94 | 4.94 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | EMPTY | 5 | 4.85 | 4.85 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 1 | 1.00 | 1.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | SHORTAGE-12 | 20 | 5.38 | 5.38 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s1|D1_m0_2p | EMPTY | 1 | 1.00 | 1.00 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s1|D2_all_1p | FULL | 20 | 8.82 | 8.82 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s1|D2_all_1p | FULL | 5 | 5.01 | 5.01 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 20 | 8.77 | 8.77 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 5 | 5.00 | 5.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 5 | 5.00 | 5.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s1|D2_all_1p | EMPTY | 20 | 4.55 | 4.55 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s1|D2_all_1p | EMPTY | 5 | 4.51 | 4.51 | 0.000 | 36 | 36 | 72 | no |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 1 | 1.00 | 1.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-24 | 1 | 1.00 | 1.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho0.75_s1|D2_all_1p | SHORTAGE-12 | 20 | 6.74 | 6.74 | 0.000 | 24 | 48 | 72 | no |
| small_rho0.75_s1|D2_all_1p | FULL | 1 | 1.00 | 1.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho0.75_s1|D2_all_1p | EMPTY | 1 | 1.00 | 1.00 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | FULL | 20 | 2.37 | 2.37 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | FULL | 5 | 2.39 | 2.39 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 20 | 2.41 | 2.41 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 5 | 2.40 | 2.40 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 5 | 1.06 | 1.06 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | EMPTY | 20 | 1.76 | 1.76 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | EMPTY | 5 | 1.76 | 1.76 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1.01 | 1.01 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1.00 | 1.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 20 | 1.06 | 1.06 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | FULL | 1 | 1.00 | 1.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s0|D1_m0_2p | EMPTY | 1 | 1.00 | 1.00 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s0|D2_all_1p | FULL | 20 | 15.56 | 15.56 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s0|D2_all_1p | FULL | 5 | 5.00 | 5.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 20 | 15.57 | 15.57 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 5 | 5.00 | 5.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 5 | 2.18 | 2.18 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s0|D2_all_1p | EMPTY | 20 | 1.65 | 1.65 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s0|D2_all_1p | EMPTY | 5 | 1.65 | 1.65 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | 1.00 | 1.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | 1.01 | 1.01 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 20 | 2.17 | 2.17 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s0|D2_all_1p | FULL | 1 | 1.01 | 1.01 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s0|D2_all_1p | EMPTY | 1 | 1.00 | 1.00 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | FULL | 20 | 2.43 | 2.43 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | FULL | 5 | 2.45 | 2.45 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 20 | 2.60 | 2.59 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 5 | 2.61 | 2.61 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 5 | 3.24 | 3.24 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | EMPTY | 20 | 0.64 | 0.64 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | EMPTY | 5 | 0.71 | 0.71 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 1 | 1.00 | 1.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-24 | 1 | 1.00 | 1.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | SHORTAGE-12 | 20 | 3.18 | 3.18 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | FULL | 1 | 1.00 | 1.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s1|D1_m0_2p | EMPTY | 1 | 0.64 | 0.64 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s1|D2_all_1p | FULL | 20 | 2.59 | 2.58 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s1|D2_all_1p | FULL | 5 | 2.59 | 2.59 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 20 | 2.58 | 2.58 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 5 | 2.58 | 2.58 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 5 | 1.92 | 1.92 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s1|D2_all_1p | EMPTY | 20 | 1.54 | 1.54 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s1|D2_all_1p | EMPTY | 5 | 1.55 | 1.55 | 0.000 | 36 | 36 | 72 | no |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 1 | 1.00 | 1.00 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-24 | 1 | 1.00 | 1.00 | 0.000 | 12 | 60 | 72 | no |
| small_rho1.10_s1|D2_all_1p | SHORTAGE-12 | 20 | 1.92 | 1.92 | 0.000 | 24 | 48 | 72 | no |
| small_rho1.10_s1|D2_all_1p | FULL | 1 | 1.00 | 1.00 | 0.000 | 0 | 72 | 72 | no |
| small_rho1.10_s1|D2_all_1p | EMPTY | 1 | 1.01 | 1.00 | 0.000 | 36 | 36 | 72 | no |

## 4. Repair logs

- **large_rho0.75_s0|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=33.7516 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=12.6505 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=6.97435 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=0 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=86.8356 Y=0 Z=0 | p1 note m0 t3: recovery period (delta=2) | p1 note m0 t3: still activated after the cancellation: lef
- **large_rho0.75_s0|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=33.7516 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=12.6505 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=6.97435 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=11.1153 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=20.1795 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=1.03272 Y=1 Z=1 | p1 cancelled m1 t1: down period: r
- **large_rho0.75_s1|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=17.4601 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=22.9153 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=23.2068 Y=1 Z=0 | p1 cancelled m0 t2: down period: removed X=17.8055 Y=1 Z=0 | p1 cancelled m0 t2: down period: removed X=22.4533 Y=1 Z=0 | p1 cancelled m0 t2: down period: removed X=18.9368 Y=1 Z=1 | p1 note m0 t3: recovery period (de
- **large_rho0.75_s1|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=17.4601 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=22.9153 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=23.2068 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=32.4601 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=32.2323 Y=1 Z=1 | p1 cancelled m2 t1: down period: removed X=21.1789 Y=1 Z=1 | p1 cancelled m2 t1: down period: r
- **large_rho1.10_s0|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=80.1589 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=1.77636e-15 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=100 Y=0 Z=0 | p1 note m0 t3: recovery period (delta=2) | p1 note m0 t3: still activated after the cancellation: left untouched | p1 note m0 t3: still activated after the cancellation: left untouched | p1 cancelled m0 t17: cascade: 
- **large_rho1.10_s0|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=80.1589 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=1.77636e-15 Y=0 Z=0 | p1 cancelled m1 t1: down period: removed X=-1.77636e-14 Y=0 Z=0 | p1 cancelled m1 t1: down period: removed X=48.0834 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=27.0293 Y=1 Z=1 | p1 cancelled m2 t1: down period: removed X=84.7443 Y=1 Z=1 | p1 note m0 t2: recovery p
- **large_rho1.10_s1|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=81.0397 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=3.55271e-15 Y=0 Z=0 | p1 cancelled m0 t1: down period: removed X=3.55271e-15 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=100 Y=0 Z=0 | p1 note m0 t3: recovery period (delta=2) | p1 note m0 t3: still activated after the cancellation: left untouched | p1 cancelled m0 t18: cascade: product
- **large_rho1.10_s1|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=81.0397 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=3.55271e-15 Y=0 Z=0 | p1 cancelled m0 t1: down period: removed X=3.55271e-15 Y=0 Z=0 | p1 cancelled m1 t1: down period: removed X=82.9209 Y=1 Z=1 | p1 cancelled m2 t1: down period: removed X=63.2413 Y=1 Z=1 | p1 note m0 t2: recovery period (delta=1) | p1 cancelled m0 t2: no activation remains (h
- **medium_rho0.75_s0|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=40.8977 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=31.5985 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=63.7936 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=18.3907 Y=1 Z=1 | p1 note m0 t3: recovery period (delta=2) | p1 note m0 t3: still activated after the cancellation: left untouched | p1 note m0 t3: still activated after the 
- **medium_rho0.75_s0|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=40.8977 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=31.5985 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=28.7244 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=10.5046 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=21.2003 Y=1 Z=0 | p1 cancelled m2 t1: down period: removed X=52.2753 Y=1 Z=0 | p1 cancelled m2 t1: down period: r
- **medium_rho0.75_s1|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=33.2272 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=-3.55271e-15 Y=0 Z=0 | p1 cancelled m0 t1: down period: removed X=31.6152 Y=1 Z=0 | p1 cancelled m0 t2: down period: removed X=39.8109 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=44.9969 Y=1 Z=1 | p1 note m0 t3: recovery period (delta=2) | p1 note m0 t3: still activated after the cancel
- **medium_rho0.75_s1|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=33.2272 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=-3.55271e-15 Y=0 Z=0 | p1 cancelled m0 t1: down period: removed X=31.6152 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=37.139 Y=1 Z=0 | p1 cancelled m1 t1: down period: removed X=30.6693 Y=1 Z=1 | p1 cancelled m2 t1: down period: removed X=21.24 Y=1 Z=0 | p1 cancelled m2 t1: down period:
- **medium_rho1.10_s0|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=84.5131 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=50.9907 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=30.0247 Y=1 Z=1 | p1 note m0 t3: recovery period (delta=2) | p1 cancelled m0 t3: no activation remains (had_prod=True carried=True): removed X=100 Y=0 | p1 cancelled m0 t8: cascade: production lost its activation, removed X=7.10543e-15
- **medium_rho1.10_s0|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=84.5131 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=82.9831 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=1.06581e-14 Y=-0 Z=0 | p1 cancelled m2 t1: down period: removed X=80.1473 Y=1 Z=1 | p1 note m0 t2: recovery period (delta=1) | p1 cancelled m0 t2: no activation remains (had_prod=True carried=True): removed X=50.9907 Y=0 | p1 note m0 t
- **medium_rho1.10_s1|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=86.8824 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=100 Y=0 Z=0 | p1 note m0 t3: recovery period (delta=2) | p1 note m0 t3: still activated after the cancellation: left untouched | p1 cancelled m0 t8: cascade: production lost its activation, removed X=7.10543e-15 | p1 cancelled m0 t10: cascade: production lost its activation, removed X=3.55271e-1
- **medium_rho1.10_s1|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=86.8824 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=85.0098 Y=1 Z=1 | p1 cancelled m2 t1: down period: removed X=86.3023 Y=1 Z=1 | p1 note m0 t2: recovery period (delta=1) | p1 cancelled m0 t2: no activation remains (had_prod=True carried=True): removed X=100 Y=0 | p1 note m1 t2: recovery period (delta=1) | p1 cancelled m1 t2: no activation remai
- **small_rho0.75_s0|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=49.1265 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=24.4242 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=27.1607 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=58.338 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=1.91278e-15 Y=0 Z=0 | p1 note m0 t3: recovery period (delta=2) | p1 cancelled m0 t3: no activation remains (had_pr
- **small_rho0.75_s0|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=49.1265 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=24.4242 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=37.8264 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=29.1654 Y=1 Z=0 | p1 note m0 t2: recovery period (delta=1) | p1 cancelled m0 t2: no activation remains (had_prod=True carried=True): removed X=27.1607 Y=0 | p1 note m0 t2: st
- **small_rho0.75_s1|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=32.0379 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=32.301 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=8.08925e-13 Y=2.38581e-14 Z=-0 | p1 cancelled m0 t1: down period: removed X=4.40536e-12 Y=1.23346e-13 Z=0 | p1 cancelled m0 t2: down period: removed X=48.3125 Y=-0 Z=0 | p1 cancelled m0 t2: down period: removed X=-0 Y=-5.13013e-16 Z=0 |
- **small_rho0.75_s1|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=32.0379 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=32.301 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=8.08925e-13 Y=2.38581e-14 Z=-0 | p1 cancelled m0 t1: down period: removed X=4.40536e-12 Y=1.23346e-13 Z=0 | p1 cancelled m1 t1: down period: removed X=4.15668e-12 Y=1.50041e-13 Z=0 | p1 cancelled m1 t1: down period: removed X=32.6349 Y=
- **small_rho1.10_s0|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=50.4735 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=26.561 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=37.9718 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=-7.40439e-14 Y=0 Z=0 | p1 cancelled m0 t2: down period: removed X=50.5659 Y=0 Z=-0 | p1 note m0 t3: recovery period (delta=2) | p1 cancelled m0 t3: no activation remains (had_
- **small_rho1.10_s0|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=50.4735 Y=1 Z=0 | p1 cancelled m0 t1: down period: removed X=26.561 Y=1 Z=1 | p1 cancelled m1 t1: down period: removed X=50.5583 Y=1 Z=-0 | p1 cancelled m1 t1: down period: removed X=20.7756 Y=1 Z=1 | p1 note m0 t2: recovery period (delta=1) | p1 note m0 t2: still activated after the cancellation: left untouched | p1 cancelled m0 t2: no activation remains
- **small_rho1.10_s1|D1_m0_2p**: p1 cancelled m0 t1: down period: removed X=85.5675 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=-0 Y=-2.77556e-15 Z=-2.77556e-15 | p1 cancelled m0 t2: down period: removed X=21.7939 Y=-0 Z=0 | p1 cancelled m0 t2: down period: removed X=-2.77556e-13 Y=0 Z=-0 | p1 cancelled m0 t2: down period: removed X=60.5692 Y=1 Z=1 | p1 cancelled m0 t2: down period: removed X=-3.48166e-13 Y=0 Z=0 | p1 no
- **small_rho1.10_s1|D2_all_1p**: p1 cancelled m0 t1: down period: removed X=85.5675 Y=1 Z=1 | p1 cancelled m0 t1: down period: removed X=-0 Y=-2.77556e-15 Z=-2.77556e-15 | p1 cancelled m1 t1: down period: removed X=86.6528 Y=1 Z=1 | p1 note m0 t2: recovery period (delta=1) | p1 cancelled m0 t2: no activation remains (had_prod=True carried=True): removed X=21.7939 Y=-0 | p1 cancelled m0 t2: no activation remains (had_prod=False ca
