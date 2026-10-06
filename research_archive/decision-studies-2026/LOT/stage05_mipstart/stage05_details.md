# stage05_details.md

MIP-start comparison. highspy 1.15.1, budget 20 s, kappa=4, threads=1, run order shuffled with seed 20260926.


## Absolute costs, both seeds

| state | method | seed | J_repair | J_cold | J_warm | raw cold | raw warm | source cold | source warm | delta_start | delta_restrict |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|
| large_rho0.75_s0|D1_m0_2p | FULL | 0 | 6656.22 | 6656.22 | 6656.22 | 13458.5 | 6656.22 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D1_m0_2p | FULL | 1 | 6656.22 | 6656.22 | 6656.22 | 8768.52 | 6656.22 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 0 | 6656.22 | 6656.22 | 6656.22 | 12937.4 | 6656.22 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 1 | 6656.22 | 6656.22 | 6656.22 | 14715 | 6656.22 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 0 | 6656.22 | 6656.22 | 6656.22 | 11077.4 | 6656.22 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | 6656.22 | 6001.12 | 6656.22 | 6001.12 | 6656.22 | solver | solver | -0.0984198 | 0 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 0 | 6656.22 | 6656.22 | 6656.22 | 11025.7 | 6656.22 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | 6656.22 | 5546.33 | 6656.22 | 5546.33 | 6656.22 | solver | solver | -0.166745 | 0 |
| large_rho0.75_s0|D2_all_1p | FULL | 0 | 7595.56 | 7595.56 | 7595.56 | 12650 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | FULL | 1 | 7595.56 | 7595.56 | 7595.56 | 15178.4 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 0 | 7595.56 | 7595.56 | 7595.56 | 14548.4 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 1 | 7595.56 | 7595.56 | 7595.56 | 15699.8 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 0 | 7595.56 | 7595.56 | 7595.56 | 12652.1 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | 7595.56 | 7595.56 | 7595.56 | 14067.3 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 0 | 7595.56 | 7595.56 | 7595.56 | 17618.9 | 7595.56 | repair | solver | 0 | 0 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | 7595.56 | 7595.56 | 7595.56 | 14784.8 | 7595.56 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D1_m0_2p | FULL | 0 | 9394.14 | 9394.14 | 9394.14 | 23667.9 | 9394.14 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D1_m0_2p | FULL | 1 | 9394.14 | 9394.14 | 9394.14 | 19297.6 | 9394.14 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 0 | 9394.14 | 8747.95 | 7417.78 | 8747.95 | 7417.78 | solver | solver | 0.141595 | 0.210382 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 1 | 9394.14 | 7511.43 | 8084.04 | 7511.43 | 8084.04 | solver | solver | -0.0609534 | 0.13946 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0 | 9394.14 | 9394.14 | 9394.14 | 25766.6 | 9394.14 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | 9394.14 | 9394.14 | 9394.14 | 14936 | 9394.14 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 0 | 9394.14 | 9394.14 | 9394.14 | 13222.1 | 9394.14 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | 9394.14 | 9394.14 | 9394.14 | 16642.8 | 9394.14 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D2_all_1p | FULL | 0 | 11554.2 | 11554.2 | 11554.2 | 23940.3 | 11554.2 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D2_all_1p | FULL | 1 | 11554.2 | 11554.2 | 11554.2 | 21456.1 | 11554.2 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 0 | 11554.2 | 9080.05 | 7916.75 | 9080.05 | 7916.75 | solver | solver | 0.100682 | 0.314817 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 1 | 11554.2 | 9380.89 | 7840.14 | 9380.89 | 7840.14 | solver | solver | 0.133349 | 0.321448 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 0 | 11554.2 | 9516.68 | 11554.2 | 9516.68 | 11554.2 | solver | solver | -0.176346 | 0 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | 11554.2 | 8684.5 | 11554.2 | 8684.5 | 11554.2 | solver | solver | -0.24837 | 0 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 0 | 11554.2 | 11554.2 | 11554.2 | 15067.8 | 11554.2 | repair | solver | 0 | 0 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | 11554.2 | 11554.2 | 7930.82 | 17923 | 7930.82 | repair | solver | 0.3136 | 0.3136 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 0 | 1687.28 | 931.579 | 936.411 | 931.579 | 936.411 | solver | solver | -0.00286427 | 0 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 1 | 1687.28 | 946.015 | 938.614 | 946.015 | 938.614 | solver | solver | 0.0043867 | 0 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 0 | 1687.28 | 1622.91 | 1620.64 | 1622.91 | 1620.64 | solver | solver | 0.00134389 | -0.405524 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 1 | 1687.28 | 1622.92 | 1628.04 | 1622.92 | 1628.04 | solver | solver | -0.0030342 | -0.408604 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 0 | 1687.28 | 1306.98 | 1306.98 | 1306.98 | 1306.98 | solver | solver | 0 | -0.219625 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | 1687.28 | 1306.23 | 1310.23 | 1306.23 | 1310.23 | solver | solver | -0.00237074 | -0.220245 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 0 | 1687.28 | 1298.59 | 1298.59 | 1298.59 | 1298.59 | solver | solver | 0 | -0.214656 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | 1687.28 | 1301.5 | 1296.95 | 1301.5 | 1296.95 | solver | solver | 0.00270023 | -0.212375 |
| medium_rho0.75_s0|D2_all_1p | FULL | 0 | 2321.32 | 1675.03 | 1673.67 | 1675.03 | 1673.67 | solver | solver | 0.000585439 | 0 |
| medium_rho0.75_s0|D2_all_1p | FULL | 1 | 2321.32 | 1660.85 | 1673.67 | 1660.85 | 1673.67 | solver | solver | -0.00552303 | 0 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 0 | 2321.32 | 1825.78 | 1825.78 | 1825.78 | 1825.78 | solver | solver | 0 | -0.0655268 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 1 | 2321.32 | 1829.5 | 1829.5 | 1829.5 | 1829.5 | solver | solver | 0 | -0.0671281 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 0 | 2321.32 | 1772.53 | 1773.08 | 1772.53 | 1773.08 | solver | solver | -0.00023849 | -0.042826 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | 2321.32 | 1776.75 | 1769.09 | 1776.75 | 1769.09 | solver | solver | 0.00330097 | -0.0411044 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 0 | 2321.32 | 1750.69 | 1822.14 | 1750.69 | 1822.14 | solver | solver | -0.0307803 | -0.0639568 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | 2321.32 | 1773.93 | 1747.44 | 1773.93 | 1747.44 | solver | solver | 0.0114139 | -0.0317777 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 0 | 3913.56 | 3028.4 | 3028.4 | 3028.4 | 3028.4 | solver | solver | 0 | 0 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 1 | 3913.56 | 3141.1 | 3047.03 | 3141.1 | 3047.03 | solver | solver | 0.024037 | 0 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 0 | 3913.56 | 3500.15 | 3488.73 | 3500.15 | 3488.73 | solver | solver | 0.00291741 | -0.117625 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 1 | 3913.56 | 3511.6 | 3511.6 | 3511.6 | 3511.6 | solver | solver | 0 | -0.118709 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0 | 3913.56 | 3552.96 | 3552.96 | 3552.96 | 3552.96 | solver | solver | 0 | -0.134038 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | 3913.56 | 3482.54 | 3482.54 | 3482.54 | 3482.54 | solver | solver | 0 | -0.111284 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 0 | 3913.56 | 3494.58 | 3494.58 | 3494.58 | 3494.58 | solver | solver | 0 | -0.119121 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | 3913.56 | 3475.73 | 3502.73 | 3475.73 | 3502.73 | solver | solver | -0.00689785 | -0.116441 |
| medium_rho1.10_s0|D2_all_1p | FULL | 0 | 5852.4 | 4128.62 | 4128.62 | 4128.62 | 4128.62 | solver | solver | 0 | 0 |
| medium_rho1.10_s0|D2_all_1p | FULL | 1 | 5852.4 | 4067.62 | 4010.42 | 4067.62 | 4010.42 | solver | solver | 0.00977358 | 0 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 0 | 5852.4 | 4684.42 | 4684.42 | 4684.42 | 4684.42 | solver | solver | 0 | -0.0949694 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 1 | 5852.4 | 4721.18 | 4721.18 | 4721.18 | 4721.18 | solver | solver | 0 | -0.121447 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 0 | 5852.4 | 4348.12 | 4348.12 | 4348.12 | 4348.12 | solver | solver | 0 | -0.0375065 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | 5852.4 | 4346.46 | 4260 | 4346.46 | 4260 | solver | solver | 0.0147736 | -0.0426462 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 0 | 5852.4 | 4233.71 | 4347.22 | 4233.71 | 4347.22 | solver | solver | -0.0193952 | -0.0373527 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | 5852.4 | 4284.56 | 4309.46 | 4284.56 | 4309.46 | solver | solver | -0.00425583 | -0.0510979 |

## MIP start adoption

| state | method | seed | setSolution | adopted | reported start cost | log message |
|---|---|---:|---|---:|---:|---|
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 0 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| medium_rho1.10_s0|D2_all_1p | FULL | 0 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 0 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| medium_rho1.10_s0|D2_all_1p | FULL | 1 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| large_rho1.10_s0|D1_m0_2p | FULL | 0 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| medium_rho0.75_s0|D2_all_1p | FULL | 1 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 0 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 0 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 0 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 1 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| large_rho1.10_s0|D2_all_1p | FULL | 0 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| medium_rho0.75_s0|D2_all_1p | FULL | 0 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 1 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 1 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| large_rho1.10_s0|D1_m0_2p | FULL | 1 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | 1 | HighsStatus.kOk | True | 5852.4 | MIP start solution is feasible, objective value is 5852.39700024 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | 1 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| large_rho1.10_s0|D2_all_1p | FULL | 1 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho0.75_s0|D2_all_1p | FULL | 1 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 1 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 0 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| large_rho0.75_s0|D1_m0_2p | FULL | 1 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | 1 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | 0 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| large_rho1.10_s0|D2_all_1p | EMPTY | 0 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| medium_rho0.75_s0|D1_m0_2p | FULL | 1 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 0 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | 1 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | 0 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 2321.32 | MIP start solution is feasible, objective value is 2321.31696796 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| medium_rho1.10_s0|D1_m0_2p | FULL | 0 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 11554.2 | MIP start solution is feasible, objective value is 11554.2212859 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | 1 | HighsStatus.kOk | True | 9394.14 | MIP start solution is feasible, objective value is 9394.14345575 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | 0 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| large_rho0.75_s0|D2_all_1p | EMPTY | 1 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 3913.56 | MIP start solution is feasible, objective value is 3913.55503683 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | 1 | HighsStatus.kOk | True | 1687.28 | MIP start solution is feasible, objective value is 1687.27576204 |
| large_rho0.75_s0|D1_m0_2p | FULL | 0 | HighsStatus.kOk | True | 6656.22 | MIP start solution is feasible, objective value is 6656.22202755 |
| large_rho0.75_s0|D2_all_1p | FULL | 0 | HighsStatus.kOk | True | 7595.56 | MIP start solution is feasible, objective value is 7595.55603892 |

## Raw incumbent validity and timing

| state | method | start | seed | raw valid | flips | mip_gap | dual bound | total s | overrun s |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| medium_rho1.10_s0|D1_m0_2p | EMPTY | warm | 0 | True | 0 | 0.0254178 | 3400.06 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | cold | 1 | True | 1 | 0.0562186 | 4102.11 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | warm | 0 | True | 4 | 0.0786152 | 4005.47 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | FULL | warm | 0 | True | 3 | 0.0978574 | 3724.6 | 20.03 | 0.03 |
| large_rho1.10_s0|D2_all_1p | FULL | cold | 1 | True | 1 | 0.739943 | 5579.81 | 20.04 | 0.04 |
| medium_rho1.10_s0|D1_m0_2p | FULL | cold | 1 | True | 4 | 0.114964 | 2779.99 | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | cold | 0 | True | 0 | 0.0114565 | 1604.32 | 20.02 | 0.02 |
| large_rho0.75_s0|D2_all_1p | EMPTY | cold | 1 | True | 0 | 0.755267 | 3842.24 | 20.09 | 0.09 |
| large_rho0.75_s0|D1_m0_2p | FULL | cold | 0 | True | 2 | 0.862919 | 1844.91 | 20.20 | 0.20 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | warm | 0 | True | 4 | 0.0577843 | 4096.87 | 20.02 | 0.02 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | warm | 0 | True | 0 | 0.0530946 | 3309.04 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | cold | 0 | True | 2 | 0.573296 | 6429.51 | 20.05 | 0.05 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | warm | 0 | True | 1 | 0.0542989 | 1723.2 | 20.02 | 0.02 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | warm | 0 | True | 0 | 0.00950944 | 1808.42 | 20.03 | 0.03 |
| medium_rho1.10_s0|D2_all_1p | FULL | warm | 1 | True | 2 | 0.0690186 | 3733.63 | 20.03 | 0.03 |
| large_rho1.10_s0|D1_m0_2p | FULL | warm | 0 | True | 0 | 0.521407 | 4495.98 | 20.09 | 0.09 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | warm | 1 | True | 1 | 0.0548567 | 3310.58 | 20.02 | 0.02 |
| medium_rho0.75_s0|D2_all_1p | FULL | warm | 1 | True | 4 | 0.0363682 | 1612.8 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | warm | 1 | True | 1 | 0.0795449 | 3966.67 | 20.02 | 0.02 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | cold | 1 | True | 2 | 0.515442 | 2907.89 | 20.04 | 0.04 |
| medium_rho1.10_s0|D2_all_1p | FULL | cold | 0 | True | 3 | 0.0978574 | 3724.6 | 20.03 | 0.03 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | cold | 0 | True | 1 | 0.748376 | 2774.33 | 20.10 | 0.10 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | warm | 0 | True | 0 | 0.138212 | 6392.56 | 20.06 | 0.06 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | warm | 0 | True | 0 | 0.509558 | 3725.18 | 20.05 | 0.05 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | cold | 1 | True | 0 | 0.0317406 | 3400.14 | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | FULL | warm | 0 | True | 4 | 0.070848 | 870.069 | 20.02 | 0.02 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | cold | 0 | True | 0 | 0.269011 | 6394.65 | 20.05 | 0.05 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | warm | 0 | True | 0 | 0.0112543 | 4631.7 | 20.02 | 0.02 |
| medium_rho1.10_s0|D1_m0_2p | FULL | warm | 1 | True | 4 | 0.0919 | 2767.01 | 20.02 | 0.02 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | cold | 0 | True | 0 | 0.00950944 | 1808.42 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | FULL | warm | 0 | True | 0 | 0.516143 | 5590.59 | 20.13 | 0.13 |
| medium_rho0.75_s0|D2_all_1p | FULL | warm | 0 | True | 4 | 0.0365298 | 1612.53 | 20.03 | 0.03 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | cold | 1 | True | 0 | 0.0188891 | 4632 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | FULL | cold | 1 | True | 2 | 0.0821101 | 3733.63 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | EMPTY | warm | 1 | True | 0 | 0.119579 | 6902.63 | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | FULL | cold | 1 | True | 1 | 0.765394 | 4527.34 | 20.04 | 0.04 |
| medium_rho1.10_s0|D1_m0_2p | FULL | cold | 0 | True | 4 | 0.0883699 | 2760.78 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | EMPTY | cold | 1 | True | 0 | 0.264182 | 6902.63 | 20.05 | 0.05 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | cold | 0 | True | 1 | 0.791345 | 5376.35 | 20.04 | 0.04 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | warm | 0 | True | 0 | 0.443541 | 6429.45 | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | cold | 1 | True | 0 | 0.149821 | 6386.06 | 20.05 | 0.05 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | warm | 0 | True | 0 | 0.428827 | 6599.45 | 20.12 | 0.12 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | warm | 1 | True | 0 | 0.542991 | 3041.95 | 20.08 | 0.08 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | warm | 0 | True | 0 | 0.440681 | 5254.32 | 20.10 | 0.10 |
| medium_rho0.75_s0|D1_m0_2p | FULL | cold | 1 | True | 4 | 0.149516 | 804.571 | 20.02 | 0.02 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | cold | 0 | True | 1 | 0.738604 | 2895.59 | 20.04 | 0.04 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | cold | 0 | True | 0 | 0.0478848 | 3382.83 | 20.02 | 0.02 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | cold | 1 | True | 0 | 0.0116287 | 1808.22 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | warm | 1 | True | 0 | 0.428399 | 6604.4 | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | warm | 0 | True | 0 | 0.426258 | 5389.81 | 20.05 | 0.05 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | warm | 1 | True | 0 | 0.509015 | 3729.3 | 20.08 | 0.08 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | warm | 0 | True | 3 | 0.0162693 | 1744.24 | 20.04 | 0.04 |
| large_rho0.75_s0|D2_all_1p | FULL | cold | 1 | True | 1 | 0.821766 | 2705.31 | 20.04 | 0.04 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | cold | 0 | True | 0 | 0.0112543 | 4631.7 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | cold | 0 | True | 4 | 0.0577843 | 4096.87 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-12 | warm | 1 | True | 4 | 0.0389779 | 4093.96 | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | warm | 1 | True | 2 | 0.0143502 | 1278.34 | 20.06 | 0.06 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | cold | 0 | True | 2 | 0.0217043 | 1278.61 | 20.02 | 0.02 |
| large_rho1.10_s0|D1_m0_2p | FULL | warm | 1 | True | 0 | 0.517267 | 4534.86 | 20.05 | 0.05 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | cold | 1 | True | 1 | 0.0781348 | 3949.78 | 20.02 | 0.02 |
| medium_rho1.10_s0|D2_all_1p | SHORTAGE-24 | cold | 0 | True | 3 | 0.0586049 | 3985.6 | 20.03 | 0.03 |
| medium_rho1.10_s0|D2_all_1p | EMPTY | warm | 1 | True | 0 | 0.0188891 | 4632 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | cold | 0 | True | 3 | 0.306246 | 6602.24 | 20.06 | 0.06 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | warm | 0 | True | 0 | 0.563124 | 2907.94 | 20.07 | 0.07 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | warm | 1 | True | 0 | 0.0317406 | 3400.14 | 20.03 | 0.03 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | cold | 1 | True | 2 | 0.0214371 | 1278.23 | 20.03 | 0.03 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | cold | 1 | True | 2 | 0.027867 | 1724.5 | 20.04 | 0.04 |
| medium_rho1.10_s0|D1_m0_2p | EMPTY | cold | 0 | True | 0 | 0.0285969 | 3400.06 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | FULL | warm | 1 | True | 0 | 0.516844 | 5582.49 | 20.04 | 0.04 |
| large_rho0.75_s0|D2_all_1p | FULL | warm | 1 | True | 0 | 0.64383 | 2705.31 | 20.04 | 0.04 |
| medium_rho0.75_s0|D2_all_1p | FULL | cold | 1 | True | 4 | 0.0232916 | 1622.16 | 20.02 | 0.02 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | cold | 1 | True | 4 | 0.500103 | 2772.59 | 20.12 | 0.12 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | warm | 0 | True | 0 | 0.0456791 | 3390.67 | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | cold | 1 | True | 2 | 0.0179737 | 1278.11 | 20.04 | 0.04 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | cold | 1 | True | 1 | 0.641125 | 6432.11 | 20.04 | 0.04 |
| large_rho1.10_s0|D2_all_1p | FULL | cold | 0 | True | 2 | 0.766625 | 5587.05 | 20.04 | 0.04 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | warm | 1 | True | 0 | 0.0144942 | 1604.45 | 20.02 | 0.02 |
| large_rho0.75_s0|D2_all_1p | EMPTY | warm | 0 | True | 0 | 0.493786 | 3844.98 | 20.07 | 0.07 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | warm | 1 | True | 0 | 0.563258 | 2907.05 | 20.05 | 0.05 |
| large_rho0.75_s0|D1_m0_2p | FULL | warm | 1 | True | 0 | 0.720581 | 1859.87 | 20.04 | 0.04 |
| medium_rho0.75_s0|D2_all_1p | EMPTY | warm | 1 | True | 0 | 0.0116287 | 1808.22 | 20.03 | 0.03 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | warm | 0 | True | 0 | 0.0101106 | 1604.26 | 20.02 | 0.02 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | cold | 1 | True | 2 | 0.683694 | 5264.23 | 20.04 | 0.04 |
| large_rho1.10_s0|D2_all_1p | EMPTY | warm | 0 | True | 0 | 0.127495 | 6907.41 | 20.07 | 0.07 |
| large_rho0.75_s0|D1_m0_2p | FULL | cold | 1 | True | 3 | 0.786064 | 1875.9 | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | warm | 1 | True | 0 | 0.426998 | 5382.87 | 20.05 | 0.05 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | warm | 0 | True | 0 | 0.51142 | 3711.03 | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | FULL | cold | 0 | True | 0 | 0.809121 | 4517.71 | 20.04 | 0.04 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | cold | 0 | True | 4 | 0.601725 | 5266.03 | 20.05 | 0.05 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | cold | 1 | True | 3 | 0.639604 | 5382.87 | 20.05 | 0.05 |
| medium_rho0.75_s0|D1_m0_2p | FULL | warm | 1 | True | 4 | 0.0432861 | 897.985 | 20.02 | 0.02 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | warm | 1 | True | 0 | 0.584488 | 2765.74 | 20.04 | 0.04 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | cold | 0 | True | 2 | 0.705386 | 3727.48 | 20.15 | 0.15 |
| large_rho0.75_s0|D2_all_1p | EMPTY | cold | 0 | True | 0 | 0.735711 | 3844.98 | 20.10 | 0.10 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | cold | 1 | True | 0 | 0.0288961 | 3381.91 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-12 | cold | 1 | True | 2 | 0.239387 | 6605.54 | 20.04 | 0.04 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | cold | 0 | True | 2 | 0.0153844 | 1278.62 | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | warm | 0 | True | 2 | 0.0217043 | 1278.61 | 20.02 | 0.02 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-12 | cold | 1 | True | 0 | 0.734896 | 3729.3 | 20.11 | 0.11 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | warm | 0 | True | 2 | 0.0153844 | 1278.62 | 20.02 | 0.02 |
| medium_rho0.75_s0|D1_m0_2p | EMPTY | cold | 1 | True | 0 | 0.0116139 | 1604.07 | 20.03 | 0.03 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | warm | 1 | True | 3 | 0.0137537 | 1744.75 | 20.03 | 0.03 |
| large_rho1.10_s0|D1_m0_2p | EMPTY | warm | 1 | True | 0 | 0.211221 | 6376.52 | 20.09 | 0.09 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | cold | 0 | True | 4 | 0.0124614 | 1728.87 | 20.02 | 0.02 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | warm | 0 | True | 0 | 0.540084 | 3061.3 | 20.04 | 0.04 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | cold | 1 | True | 3 | 0.0181739 | 1744.46 | 20.02 | 0.02 |
| large_rho1.10_s0|D2_all_1p | EMPTY | cold | 0 | True | 0 | 0.239277 | 6907.41 | 20.04 | 0.04 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-24 | warm | 1 | True | 4 | 0.0109062 | 1728.38 | 20.03 | 0.03 |
| medium_rho0.75_s0|D2_all_1p | SHORTAGE-12 | cold | 0 | True | 3 | 0.0158874 | 1744.37 | 20.02 | 0.02 |
| large_rho0.75_s0|D2_all_1p | FULL | cold | 0 | True | 1 | 0.786295 | 2703.38 | 20.09 | 0.09 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | cold | 0 | True | 3 | 0.78944 | 3709.85 | 20.10 | 0.10 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | cold | 1 | True | 1 | 0.046074 | 3315.59 | 20.02 | 0.02 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | warm | 1 | True | 0 | 0.511589 | 3709.75 | 20.04 | 0.04 |
| medium_rho1.10_s0|D1_m0_2p | FULL | warm | 0 | True | 4 | 0.0883699 | 2760.78 | 20.03 | 0.03 |
| large_rho1.10_s0|D2_all_1p | SHORTAGE-24 | warm | 1 | True | 2 | 0.189484 | 6428.06 | 20.05 | 0.05 |
| large_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | warm | 1 | True | 0 | 0.438566 | 5274.19 | 20.04 | 0.04 |
| large_rho0.75_s0|D1_m0_2p | SHORTAGE-24 | warm | 0 | True | 0 | 0.582767 | 2777.19 | 20.05 | 0.05 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | cold | 0 | True | 0 | 0.763375 | 3061.3 | 20.04 | 0.04 |
| medium_rho0.75_s0|D2_all_1p | FULL | cold | 0 | True | 4 | 0.0378386 | 1611.65 | 20.02 | 0.02 |
| large_rho0.75_s0|D2_all_1p | EMPTY | warm | 1 | True | 0 | 0.494146 | 3842.24 | 20.05 | 0.05 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-12 | warm | 1 | True | 0 | 0.0288961 | 3381.91 | 20.03 | 0.03 |
| medium_rho0.75_s0|D1_m0_2p | FULL | cold | 0 | True | 4 | 0.0682896 | 867.961 | 20.05 | 0.05 |
| medium_rho1.10_s0|D1_m0_2p | SHORTAGE-24 | cold | 0 | True | 0 | 0.0530946 | 3309.04 | 20.03 | 0.03 |
| large_rho0.75_s0|D2_all_1p | SHORTAGE-24 | cold | 1 | True | 2 | 0.749084 | 3709.75 | 20.14 | 0.14 |
| medium_rho0.75_s0|D1_m0_2p | SHORTAGE-12 | warm | 1 | True | 2 | 0.0242761 | 1278.42 | 20.04 | 0.04 |
| large_rho0.75_s0|D1_m0_2p | FULL | warm | 0 | True | 0 | 0.72013 | 1862.88 | 20.13 | 0.13 |
| large_rho0.75_s0|D1_m0_2p | EMPTY | cold | 1 | True | 0 | 0.79183 | 3063.22 | 20.13 | 0.13 |
| large_rho0.75_s0|D2_all_1p | FULL | warm | 0 | True | 0 | 0.644084 | 2703.38 | 20.07 | 0.07 |
