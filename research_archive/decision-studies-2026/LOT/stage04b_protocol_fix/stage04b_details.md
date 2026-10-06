# stage04b_details.md

Protocol-fix retest. Budget 20 s per method, kappa=4, method order shuffled with seed 20260926.

Repaired reference plans are the FROZEN stage-04 ones, so the comparison baseline is unchanged.


## Per run

| state | method | repair | raw incumbent | raw valid | selected | source | reason | total s | overrun s |
|---|---|---:|---:|---|---:|---|---|---:|---:|
| large_rho0.75_s0|D1_m0_2p | EMPTY | 6656.22 | 12937.4 | True | 6656.22 | repair | incumbent_worse_or_tied | 20.03 | 0.03 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 6656.22 | 11077.4 | True | 6656.22 | repair | incumbent_worse_or_tied | 20.04 | 0.04 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 6656.22 | 11025.7 | True | 6656.22 | repair | incumbent_worse_or_tied | 20.03 | 0.03 |
| large_rho0.75_s0|D1_m0_2p | FULL | 6656.22 | 13458.5 | True | 6656.22 | repair | incumbent_worse_or_tied | 20.14 | 0.14 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 7595.56 | 14548.4 | True | 7595.56 | repair | incumbent_worse_or_tied | 20.11 | 0.11 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 7595.56 | 12652.1 | True | 7595.56 | repair | incumbent_worse_or_tied | 20.03 | 0.03 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 7595.56 | 17618.9 | True | 7595.56 | repair | incumbent_worse_or_tied | 20.09 | 0.09 |
| large_rho0.75_s0|D2_all_1p | FULL | 7595.56 | 12650 | True | 7595.56 | repair | incumbent_worse_or_tied | 20.03 | 0.03 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 9394.14 | 7468.92 | True | 7468.92 | solver | incumbent_better | 20.08 | 0.08 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 9394.14 | 7790.45 | True | 7790.45 | solver | incumbent_better | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 9394.14 | 7101.3 | True | 7101.3 | solver | incumbent_better | 20.03 | 0.03 |
| large_rho1.10_s0|D1_m0_2p | FULL | 9394.14 | 24492.9 | True | 9394.14 | repair | incumbent_worse_or_tied | 20.03 | 0.03 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 11554.2 | 7900.1 | True | 7900.1 | solver | incumbent_better | 20.09 | 0.09 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 11554.2 | 8410.61 | True | 8410.61 | solver | incumbent_better | 20.04 | 0.04 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 11554.2 | 9420.99 | True | 9420.99 | solver | incumbent_better | 20.03 | 0.03 |
| large_rho1.10_s0|D2_all_1p | FULL | 11554.2 | 23940.3 | True | 11554.2 | repair | incumbent_worse_or_tied | 20.04 | 0.04 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 1687.28 | 1625.01 | True | 1625.01 | solver | incumbent_better | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1687.28 | 1360.09 | True | 1360.09 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1687.28 | 1310.7 | True | 1310.7 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 1687.28 | 927.111 | True | 927.111 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 2321.32 | 1827.6 | True | 1827.6 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 2321.32 | 1771.56 | True | 1771.56 | solver | incumbent_better | 20.02 | 0.02 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 2321.32 | 1755.9 | True | 1755.9 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho0.75_s0|D2_all_1p | FULL | 2321.32 | 1668.15 | True | 1668.15 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 3913.56 | 3530.68 | True | 3530.68 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 3913.56 | 3528.98 | True | 3528.98 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 3913.56 | 3511.97 | True | 3511.97 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 3913.56 | 3072.25 | True | 3072.25 | solver | incumbent_better | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 5852.4 | 4695.04 | True | 4695.04 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 5852.4 | 4311.35 | True | 4311.35 | solver | incumbent_better | 20.01 | 0.01 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 5852.4 | 4301.06 | True | 4301.06 | solver | incumbent_better | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | FULL | 5852.4 | 4053.38 | True | 4053.38 | solver | incumbent_better | 20.01 | 0.01 |

## Free-variable accounting (three separate numbers)

| state | method | Y slots | Z slots | Z struct-pinned | free Y | free Z |
|---|---|---:|---:|---:|---:|---:|
| large_rho0.75_s0|D1_m0_2p | EMPTY | 1296 | 1296 | 72 | 864 | 1224 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1296 | 1296 | 72 | 876 | 1224 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1296 | 1296 | 72 | 888 | 1224 |
| large_rho0.75_s0|D1_m0_2p | FULL | 1296 | 1296 | 72 | 1296 | 1224 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 1296 | 1296 | 72 | 864 | 1224 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1296 | 1296 | 72 | 876 | 1224 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1296 | 1296 | 72 | 888 | 1224 |
| large_rho0.75_s0|D2_all_1p | FULL | 1296 | 1296 | 72 | 1296 | 1224 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 1296 | 1296 | 72 | 864 | 1224 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1296 | 1296 | 72 | 876 | 1224 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1296 | 1296 | 72 | 888 | 1224 |
| large_rho1.10_s0|D1_m0_2p | FULL | 1296 | 1296 | 72 | 1296 | 1224 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 1296 | 1296 | 72 | 864 | 1224 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1296 | 1296 | 72 | 876 | 1224 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1296 | 1296 | 72 | 888 | 1224 |
| large_rho1.10_s0|D2_all_1p | FULL | 1296 | 1296 | 72 | 1296 | 1224 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 432 | 432 | 36 | 216 | 396 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 432 | 432 | 36 | 228 | 396 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 432 | 432 | 36 | 240 | 396 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 432 | 432 | 36 | 432 | 396 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 432 | 432 | 36 | 216 | 396 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 432 | 432 | 36 | 228 | 396 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 432 | 432 | 36 | 240 | 396 |
| medium_rho0.75_s0|D2_all_1p | FULL | 432 | 432 | 36 | 432 | 396 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 432 | 432 | 36 | 216 | 396 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 432 | 432 | 36 | 228 | 396 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 432 | 432 | 36 | 240 | 396 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 432 | 432 | 36 | 432 | 396 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 432 | 432 | 36 | 216 | 396 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 432 | 432 | 36 | 228 | 396 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 432 | 432 | 36 | 240 | 396 |
| medium_rho1.10_s0|D2_all_1p | FULL | 432 | 432 | 36 | 432 | 396 |

`Z struct-pinned` counts Z_ij0 from (1.13) and Z_ijT under D2. Stage 04 reported `Z slots` as if all of them were free.


## Selection errors

None.


## Raw-incumbent problems (invalid incumbents are errors, not timeouts)

None.

