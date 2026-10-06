# stage02_details.md

Machine-generated detail tables. Produced by `python run.py` in `stage02_repair_reopt`.

Environment: python 3.13.14, numpy 2.5.2, scipy 1.17.1, Windows-11-10.0.26200-SP0

Model: `audited_v1`. tau = 2, kappa in [0, 1, 2], per-solve time cap 30s.


## 0. Summary

| 检查 | 验收要求 | 结果 |
|---|---|---|
| 名义解、修复解、重优化解 | 全部通过独立验解 | PASS |
| 稳定性 | 短期翻转数不超过 kappa | PASS |
| 固定条件 | 集合外短期 Y 不变 | PASS |
| 参考点 | 相对 S^r 计数，另存相对 S^o | PASS |
| 嵌套可行域 | 扩大释放集合成本不升 | PASS |
| 基线关系 | J_kappa <= J_0 <= J_LP <= J_r | PASS |
| 机制测试 | 手算预期命中 | PASS |
| 数据保存 | 写盘后重读一致 | 见 `stage02_savecheck.py` |

Hard failures in this run: **0**


## 1. Mechanism tests (hand-written nominal plans)

| 实例 | 故障 | 名义成本 | 修复成本 | 预期 | J_LP | 预期 | 无故障不变 | 修复可行 | J_LP 可行 |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| M0_nodis | none | 5 | 5 | 5 OK | 5 | None OK | True | True | True |
| M1_shutdown | m0_1p | 5 | 30 | 30 OK | 30 | None OK | False | True | True |
| M2_repair_fits | m0_1p | 1 | 21 | 21 OK | 21 | None OK | False | True | True |
| M3_repair_fails | m0_1p | 1 | 20 | 20 OK | 20 | None OK | False | True | True |
| M4_repair_headroom | m0_1p | 1 | 21 | 21 OK | 11 | 11 OK | False | True | True |
| M5_capacity_occupied | m0_1p | 3 | 21 | None OK | 21 | None OK | False | True | True |
| M6_initial_stock | all_T | 4 | 11 | None OK | 11 | None OK | False | True | True |

### Repair logs

- **M0_nodis** (no disruption)
    - `note`: no disruption: nominal plan returned as is
- **M1_shutdown** (machine 0 down during [1])
    - `cancelled` m0 t1: removed X=3 Y=1 Z=0 from the nominal plan
    - `note` m0: disruption covers the whole horizon (delta=1=T): recovery step skipped
- **M2_repair_fits** (machine 0 down during [1])
    - `cancelled` m0 t1: removed X=1 Y=1 Z=1 from the nominal plan
    - `note` m0 t2: recovery period for machine 0 (delta=1)
    - `note` m0 t2: carry-over from t=1 was broken; ledger: need s+b*m=2, other items use 0, room=2
    - `note` m0 t2: trial restore=(feasible=True, cost=21) vs cancel=(feasible=True, cost=30)
    - `restored` m0 t2: set Y=1, X=m_i=1
- **M3_repair_fails** (machine 0 down during [1])
    - `cancelled` m0 t1: removed X=1 Y=1 Z=1 from the nominal plan
    - `note` m0 t2: recovery period for machine 0 (delta=1)
    - `note` m0 t2: carry-over from t=1 was broken; ledger: need s+b*m=2, other items use 0, room=1
    - `note` m0 t2: trial restore=(feasible=False capacity j0,t2: 2 > 1, cost=inf) vs cancel=(feasible=True, cost=20)
    - `refused` m0 t2: cancelled the recovery-period production: restoring would not fit
- **M4_repair_headroom** (machine 0 down during [1])
    - `cancelled` m0 t1: removed X=1 Y=1 Z=1 from the nominal plan
    - `note` m0 t2: recovery period for machine 0 (delta=1)
    - `note` m0 t2: carry-over from t=1 was broken; ledger: need s+b*m=2, other items use 0, room=4
    - `note` m0 t2: trial restore=(feasible=True, cost=21) vs cancel=(feasible=True, cost=30)
    - `restored` m0 t2: set Y=1, X=m_i=1
- **M5_capacity_occupied** (machine 0 down during [1])
    - `cancelled` m0 t1: removed X=1 Y=1 Z=1 from the nominal plan
    - `note` m0 t2: recovery period for machine 0 (delta=1)
    - `note` m0 t2: carry-over from t=1 was broken; ledger: need s+b*m=2, other items use 3, room=1
    - `note` m0 t2: trial restore=(feasible=False capacity j0,t2: 5 > 4, cost=inf) vs cancel=(feasible=True, cost=21)
    - `refused` m0 t2: cancelled the recovery-period production: restoring would not fit
- **M6_initial_stock** (machine 0 down during [1, 2])
    - `cancelled` m0 t1: removed X=0 Y=1 Z=1 from the nominal plan
    - `cancelled` m0 t2: removed X=1 Y=-0 Z=0 from the nominal plan
    - `note` m0: disruption covers the whole horizon (delta=2=T): recovery step skipped

### Plans (repaired)

- **M0_nodis**: `(i0,j0) X=[3] Y=[1] Z=[0] i0 I=[0] L=[0]`
- **M1_shutdown**: `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **M2_repair_fits**: `(i0,j0) X=[0,1] Y=[0,1] Z=[0,0] i0 I=[0,0] L=[1,1]`
- **M3_repair_fails**: `(i0,j0) X=[0,0] Y=[0,0] Z=[0,0] i0 I=[0,0] L=[1,1]`
- **M4_repair_headroom**: `(i0,j0) X=[0,1] Y=[0,1] Z=[0,0] i0 I=[0,0] L=[1,1]`
- **M5_capacity_occupied**: `(i0,j0) X=[0,2] Y=[0,1] Z=[0,0] i0 I=[0,0] L=[0,0] (i1,j0) X=[0,0] Y=[0,0] Z=[0,0] i1 I=[0,0] L=[0,2]`
- **M6_initial_stock**: `(i0,j0) X=[0,0] Y=[0,0] Z=[0,0] i0 I=[1,0] L=[0,1]`

## 2. Nominal solutions (solved, not constructed)

| 实例 | 成本 | 状态 | gap | 耗时(s) | 可行 | 最大违反 | 独立路径 |
|---|---:|---|---:|---:|---|---:|---|
| S0 | 8 | optimal | 0 | 0.019 | True | 0.0e+00 | optimal (8) |
| S1 | 5 | optimal | 0 | 0.020 | True | 0.0e+00 | binary_infeasible (-) |
| S2 | 8 | optimal | 0 | 0.009 | True | 0.0e+00 | optimal (8) |
| S3 | 5 | optimal | 0 | 0.015 | True | 0.0e+00 | binary_infeasible (-) |

### Nominal plans

- **S0** (cost 8): `(i0,j0) X=[2,3,0] Y=[1,0,0] Z=[1,0,0] (i0,j1) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] i0 I=[0,0,0] L=[0,0,0] (i1,j0) X=[0,0,3] Y=[0,0,1] Z=[0,0,0] (i1,j1) X=[2,3,0] Y=[1,0,0] Z=[1,0,0] i1 I=[0,0,0] L=[0,0,0]`
- **S1** (cost 5): `(i0,j0) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] (i0,j1) X=[0,0,2] Y=[0,0,1] Z=[0,0,0] i0 I=[0,0,0] L=[0,0,0] (i1,j0) X=[0,3,3] Y=[0,1,0] Z=[0,1,0] (i1,j1) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] i1 I=[0,0,0] L=[0,0,0]`
- **S2** (cost 8): `(i0,j0) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] (i0,j1) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] i0 I=[1,1,1] L=[0,0,0] (i1,j0) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] (i1,j1) X=[2,3,0] Y=[1,0,0] Z=[1,0,0] i1 I=[0,2,0] L=[0,0,0]`
- **S3** (cost 5): `(i0,j0) X=[0,1,2] Y=[0,1,0] Z=[0,1,0] (i0,j1) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] i0 I=[0,0,0] L=[0,0,0] (i1,j0) X=[0,0,0] Y=[0,0,0] Z=[0,0,0] (i1,j1) X=[3,0,0] Y=[1,0,0] Z=[0,0,0] i1 I=[0,0,0] L=[0,0,0]`

## 3. The twelve disruption states

| 状态 | 故障 | J_nom | J_r | J_LP | J_0 | J_1 | J_2 | J_r-J_LP | 最大违反 | 通过 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| S0|D1_m0_1p | D1_m0_1p | 8 | 48 | 28 | 28 | 10 | 10 | 20 | 0.0e+00 | PASS |
| S0|D2_m1_2p | D2_m1_2p | 8 | 105 | 105 | 105 | 47 | 38 | 0 | 0.0e+00 | PASS |
| S0|D3_both_1p | D3_both_1p | 8 | 116 | 96 | 93 | 65 | 65 | 20 | 0.0e+00 | PASS |
| S1|D1_m0_1p | D1_m0_1p | 5 | 5 | 5 | 5 | 5 | 5 | 0 | 8.9e-16 | PASS |
| S1|D2_m1_2p | D2_m1_2p | 5 | 5 | 5 | 5 | 5 | 5 | 0 | 0.0e+00 | PASS |
| S1|D3_both_1p | D3_both_1p | 5 | 5 | 5 | 5 | 5 | 5 | 0 | 0.0e+00 | PASS |
| S2|D1_m0_1p | D1_m0_1p | 8 | 8 | 8 | 8 | 8 | 8 | 0 | 0.0e+00 | PASS |
| S2|D2_m1_2p | D2_m1_2p | 8 | 103 | 103 | 66 | 8 | 8 | 0 | 0.0e+00 | PASS |
| S2|D3_both_1p | D3_both_1p | 8 | 67 | 48 | 46 | 46 | 46 | 19 | 0.0e+00 | PASS |
| S3|D1_m0_1p | D1_m0_1p | 5 | 5 | 5 | 5 | 5 | 5 | 0 | 0.0e+00 | PASS |
| S3|D2_m1_2p | D2_m1_2p | 5 | 62 | 62 | 62 | 5 | 5 | 0 | 0.0e+00 | PASS |
| S3|D3_both_1p | D3_both_1p | 5 | 62 | 62 | 62 | 62 | 62 | 0 | 0.0e+00 | PASS |

`J_r - J_LP` is the improvement obtainable WITHOUT changing any binary variable; it must not be attributed to variable-group selection later.


## 4. Release-set and kappa sweep

| 状态 | kappa | release | 成本 | 相对 S^r 翻转 | 相对 S^o 翻转 | 全期翻转(S^r) | 稳定 | 固定条件 |
|---|---:|---|---:|---:|---:|---:|---|---|
| S0|D1_m0_1p | 0 | R0_empty | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 0 | R1_one | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 0 | Rall | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 1 | R0_empty | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 1 | R1_one | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 1 | Rall | 10 | 1 | 3 | 3 | True | True |
| S0|D1_m0_1p | 2 | R0_empty | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 2 | R1_one | 28 | 0 | 2 | 2 | True | True |
| S0|D1_m0_1p | 2 | Rall | 10 | 1 | 3 | 1 | True | True |
| S0|D1_m0_1p | None | Rall | 8 | 3 | 3 | 4 | True | True |
| S0|D2_m1_2p | 0 | R0_empty | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 0 | R1_one | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 0 | Rall | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 1 | R0_empty | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 1 | R1_one | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 1 | Rall | 47 | 1 | 2 | 2 | True | True |
| S0|D2_m1_2p | 2 | R0_empty | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 2 | R1_one | 105 | 0 | 1 | 2 | True | True |
| S0|D2_m1_2p | 2 | Rall | 38 | 2 | 3 | 3 | True | True |
| S0|D2_m1_2p | None | Rall | 30 | 3 | 4 | 3 | True | True |
| S0|D3_both_1p | 0 | R0_empty | 93 | 0 | 3 | 1 | True | True |
| S0|D3_both_1p | 0 | R1_one | 93 | 0 | 3 | 1 | True | True |
| S0|D3_both_1p | 0 | Rall | 93 | 0 | 3 | 1 | True | True |
| S0|D3_both_1p | 1 | R0_empty | 93 | 0 | 3 | 1 | True | True |
| S0|D3_both_1p | 1 | R1_one | 65 | 1 | 4 | 2 | True | True |
| S0|D3_both_1p | 1 | Rall | 65 | 1 | 4 | 2 | True | True |
| S0|D3_both_1p | 2 | R0_empty | 93 | 0 | 3 | 1 | True | True |
| S0|D3_both_1p | 2 | R1_one | 65 | 1 | 4 | 2 | True | True |
| S0|D3_both_1p | 2 | Rall | 65 | 1 | 4 | 2 | True | True |
| S0|D3_both_1p | None | Rall | 65 | 3 | 4 | 4 | True | True |
| S1|D1_m0_1p | 0 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 0 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 0 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 1 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 1 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 1 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 2 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 2 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D1_m0_1p | 2 | Rall | 5 | 1 | 1 | 2 | True | True |
| S1|D1_m0_1p | None | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 0 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 0 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 0 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 1 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 1 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 1 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 2 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 2 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | 2 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D2_m1_2p | None | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 0 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 0 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 0 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 1 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 1 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 1 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 2 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 2 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | 2 | Rall | 5 | 0 | 0 | 0 | True | True |
| S1|D3_both_1p | None | Rall | 5 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 0 | R0_empty | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 0 | R1_one | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 0 | Rall | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 1 | R0_empty | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 1 | R1_one | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 1 | Rall | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 2 | R0_empty | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 2 | R1_one | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | 2 | Rall | 8 | 0 | 0 | 0 | True | True |
| S2|D1_m0_1p | None | Rall | 8 | 0 | 0 | 0 | True | True |
| S2|D2_m1_2p | 0 | R0_empty | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 0 | R1_one | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 0 | Rall | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 1 | R0_empty | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 1 | R1_one | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 1 | Rall | 8 | 1 | 2 | 1 | True | True |
| S2|D2_m1_2p | 2 | R0_empty | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 2 | R1_one | 66 | 0 | 1 | 1 | True | True |
| S2|D2_m1_2p | 2 | Rall | 8 | 1 | 2 | 1 | True | True |
| S2|D2_m1_2p | None | Rall | 8 | 1 | 2 | 1 | True | True |
| S2|D3_both_1p | 0 | R0_empty | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 0 | R1_one | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 0 | Rall | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 1 | R0_empty | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 1 | R1_one | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 1 | Rall | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 2 | R0_empty | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 2 | R1_one | 46 | 0 | 2 | 0 | True | True |
| S2|D3_both_1p | 2 | Rall | 46 | 2 | 2 | 2 | True | True |
| S2|D3_both_1p | None | Rall | 46 | 2 | 2 | 2 | True | True |
| S3|D1_m0_1p | 0 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 0 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 0 | Rall | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 1 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 1 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 1 | Rall | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 2 | R0_empty | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 2 | R1_one | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | 2 | Rall | 5 | 0 | 0 | 0 | True | True |
| S3|D1_m0_1p | None | Rall | 5 | 0 | 0 | 0 | True | True |
| S3|D2_m1_2p | 0 | R0_empty | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 0 | R1_one | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 0 | Rall | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 1 | R0_empty | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 1 | R1_one | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 1 | Rall | 5 | 1 | 2 | 1 | True | True |
| S3|D2_m1_2p | 2 | R0_empty | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 2 | R1_one | 62 | 0 | 1 | 0 | True | True |
| S3|D2_m1_2p | 2 | Rall | 5 | 1 | 2 | 1 | True | True |
| S3|D2_m1_2p | None | Rall | 5 | 1 | 2 | 1 | True | True |
| S3|D3_both_1p | 0 | R0_empty | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 0 | R1_one | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 0 | Rall | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 1 | R0_empty | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 1 | R1_one | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 1 | Rall | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 2 | R0_empty | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 2 | R1_one | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | 2 | Rall | 62 | 0 | 1 | 0 | True | True |
| S3|D3_both_1p | None | Rall | 62 | 0 | 1 | 0 | True | True |

## 5. Release index rule and release-set comparison

R1 is the lexicographically first short-term variable on a machine that is not down in that period. Stored per state so the choice cannot depend on candidate quality:

| 状态 | R1 |
|---|---|
| S0|D1_m0_1p | `[[0, 0, 2]]` |
| S0|D2_m1_2p | `[[0, 0, 1]]` |
| S0|D3_both_1p | `[[0, 0, 2]]` |
| S1|D1_m0_1p | `[[0, 0, 2]]` |
| S1|D2_m1_2p | `[[0, 0, 1]]` |
| S1|D3_both_1p | `[[0, 0, 2]]` |
| S2|D1_m0_1p | `[[0, 0, 2]]` |
| S2|D2_m1_2p | `[[0, 0, 1]]` |
| S2|D3_both_1p | `[[0, 0, 2]]` |
| S3|D1_m0_1p | `[[0, 0, 2]]` |
| S3|D2_m1_2p | `[[0, 0, 1]]` |
| S3|D3_both_1p | `[[0, 0, 2]]` |

### Same kappa, three release sets (R0 empty / R1 one / Rall)

| 状态 | kappa | R0 | R1 | Rall | R1 beats R0? | Rall beats R0? |
|---|---:|---:|---:|---:|---|---|
| S0|D1_m0_1p | 0 | 28 | 28 | 28 | no | no |
| S0|D1_m0_1p | 1 | 28 | 28 | 10 | no | yes |
| S0|D1_m0_1p | 2 | 28 | 28 | 10 | no | yes |
| S0|D2_m1_2p | 0 | 105 | 105 | 105 | no | no |
| S0|D2_m1_2p | 1 | 105 | 105 | 47 | no | yes |
| S0|D2_m1_2p | 2 | 105 | 105 | 38 | no | yes |
| S0|D3_both_1p | 0 | 93 | 93 | 93 | no | no |
| S0|D3_both_1p | 1 | 93 | 65 | 65 | yes | yes |
| S0|D3_both_1p | 2 | 93 | 65 | 65 | yes | yes |
| S1|D1_m0_1p | 0 | 5 | 5 | 5 | no | no |
| S1|D1_m0_1p | 1 | 5 | 5 | 5 | no | no |
| S1|D1_m0_1p | 2 | 5 | 5 | 5 | no | no |
| S1|D2_m1_2p | 0 | 5 | 5 | 5 | no | no |
| S1|D2_m1_2p | 1 | 5 | 5 | 5 | no | no |
| S1|D2_m1_2p | 2 | 5 | 5 | 5 | no | no |
| S1|D3_both_1p | 0 | 5 | 5 | 5 | no | no |
| S1|D3_both_1p | 1 | 5 | 5 | 5 | no | no |
| S1|D3_both_1p | 2 | 5 | 5 | 5 | no | no |
| S2|D1_m0_1p | 0 | 8 | 8 | 8 | no | no |
| S2|D1_m0_1p | 1 | 8 | 8 | 8 | no | no |
| S2|D1_m0_1p | 2 | 8 | 8 | 8 | no | no |
| S2|D2_m1_2p | 0 | 66 | 66 | 66 | no | no |
| S2|D2_m1_2p | 1 | 66 | 66 | 8 | no | yes |
| S2|D2_m1_2p | 2 | 66 | 66 | 8 | no | yes |
| S2|D3_both_1p | 0 | 46 | 46 | 46 | no | no |
| S2|D3_both_1p | 1 | 46 | 46 | 46 | no | no |
| S2|D3_both_1p | 2 | 46 | 46 | 46 | no | no |
| S3|D1_m0_1p | 0 | 5 | 5 | 5 | no | no |
| S3|D1_m0_1p | 1 | 5 | 5 | 5 | no | no |
| S3|D1_m0_1p | 2 | 5 | 5 | 5 | no | no |
| S3|D2_m1_2p | 0 | 62 | 62 | 62 | no | no |
| S3|D2_m1_2p | 1 | 62 | 62 | 5 | no | yes |
| S3|D2_m1_2p | 2 | 62 | 62 | 5 | no | yes |
| S3|D3_both_1p | 0 | 62 | 62 | 62 | no | no |
| S3|D3_both_1p | 1 | 62 | 62 | 62 | no | no |
| S3|D3_both_1p | 2 | 62 | 62 | 62 | no | no |

**Where releasing exactly ONE variable helps:** S0|D3_both_1p/k1, S0|D3_both_1p/k2

**Where one variable is useless but releasing the whole short-term set helps:** S0|D1_m0_1p/k1, S0|D1_m0_1p/k2, S0|D2_m1_2p/k1, S0|D2_m1_2p/k2, S2|D2_m1_2p/k1, S2|D2_m1_2p/k2, S3|D2_m1_2p/k1, S3|D2_m1_2p/k2


The second list is the region where a COMBINATION of setup variables must change, not a single one. It is a design input for the next round, not evidence that learned group selection is needed: R1 is a fixed index rule, not a merit-based choice, and no non-learning strong control has been run yet.


## 6. Repair logs for the twelve states

- **S0|D1_m0_1p** (machine 0 down during [1]): cancelled m0 t1: removed X=2 Y=1 Z=1 from the nominal plan | note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: carry-over from t=1 was broken; ledger: need s+b*m=2, other items use 0, room=5 | note m0 t2: trial restore=(feasible=True, cost=48) vs cancel=(feasible=True, cost=56) | restored m0 t2: set Y=1, X=m_i=1
- **S0|D2_m1_2p** (machine 1 down during [1, 2]): cancelled m1 t1: removed X=2 Y=1 Z=1 from the nominal plan | cancelled m1 t2: removed X=3 Y=0 Z=0 from the nominal plan | note m1 t3: recovery period for machine 1 (delta=2) | note m1 t3: no broken carry-over and no unactivated production: nothing to restore
- **S0|D3_both_1p** (machine 0 down during [1]; machine 1 down during [1]): cancelled m0 t1: removed X=2 Y=1 Z=1 from the nominal plan | cancelled m1 t1: removed X=2 Y=1 Z=1 from the nominal plan | note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: carry-over from t=1 was broken; ledger: need s+b*m=2, other items use 0, room=5 | note m0 t2: trial restore=(feasible=False activation i1,j1,t2, cost=inf) vs cancel=(feasible=False activation i1,j1,t2, cost=inf) | refused m0 t2: cancelled the recovery-period production: restoring would not fit | note m1 t2: recovery period for machine 1 (delta=1) | note m1 t2: carry-over from t=1 was broken; ledger: need s+b*m=3, other items use 0, room=4 | note m1 t2: trial restore=(feasible=True, cost=116) vs cancel=(feasible=True, cost=153) | restored m1 t2: set Y=1, X=m_i=2
- **S1|D1_m0_1p** (machine 0 down during [1]): note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: no broken carry-over and no unactivated production: nothing to restore
- **S1|D2_m1_2p** (machine 1 down during [1, 2]): note m1 t3: recovery period for machine 1 (delta=2) | note m1 t3: no broken carry-over and no unactivated production: nothing to restore
- **S1|D3_both_1p** (machine 0 down during [1]; machine 1 down during [1]): note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: no broken carry-over and no unactivated production: nothing to restore | note m1 t2: recovery period for machine 1 (delta=1) | note m1 t2: no broken carry-over and no unactivated production: nothing to restore
- **S2|D1_m0_1p** (machine 0 down during [1]): note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: no broken carry-over and no unactivated production: nothing to restore
- **S2|D2_m1_2p** (machine 1 down during [1, 2]): cancelled m1 t1: removed X=2 Y=1 Z=1 from the nominal plan | cancelled m1 t2: removed X=3 Y=0 Z=-0 from the nominal plan | note m1 t3: recovery period for machine 1 (delta=2) | note m1 t3: no broken carry-over and no unactivated production: nothing to restore
- **S2|D3_both_1p** (machine 0 down during [1]; machine 1 down during [1]): cancelled m1 t1: removed X=2 Y=1 Z=1 from the nominal plan | note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: no broken carry-over and no unactivated production: nothing to restore | note m1 t2: recovery period for machine 1 (delta=1) | note m1 t2: carry-over from t=1 was broken; ledger: need s+b*m=3, other items use 0, room=4 | note m1 t2: trial restore=(feasible=True, cost=67) vs cancel=(feasible=True, cost=103) | restored m1 t2: set Y=1, X=m_i=2
- **S3|D1_m0_1p** (machine 0 down during [1]): note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: no broken carry-over and no unactivated production: nothing to restore
- **S3|D2_m1_2p** (machine 1 down during [1, 2]): cancelled m1 t1: removed X=3 Y=1 Z=0 from the nominal plan | note m1 t3: recovery period for machine 1 (delta=2) | note m1 t3: no broken carry-over and no unactivated production: nothing to restore
- **S3|D3_both_1p** (machine 0 down during [1]; machine 1 down during [1]): cancelled m1 t1: removed X=3 Y=1 Z=0 from the nominal plan | note m0 t2: recovery period for machine 0 (delta=1) | note m0 t2: no broken carry-over and no unactivated production: nothing to restore | note m1 t2: recovery period for machine 1 (delta=1) | note m1 t2: no broken carry-over and no unactivated production: nothing to restore
