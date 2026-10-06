# stage01_details.md

Machine-generated detail tables. Produced by `python stage01_run.py`.

Environment: python 3.13.14, numpy 2.5.2, scipy 1.17.1, Windows-11-10.0.26200-SP0


## 0. Summary

| 检查 | 是否通过 | 最大误差/违反量 | 未解决问题 |
|---|---|---:|---|
| 手工实例 (A-G, 两种模式, 14 次求解) | PASS | 0.0e+00 | 无 |
| 独立枚举 (8 实例, MILP vs 全二元枚举+LP) | PASS | 目标 0.0e+00 / 约束 0.0e+00 | 无 |
| 固定与释放接口 (8 实例) | PASS | 0.0e+00 | 无。全部 Y 变量的单调性只对最优解成立，未来限时求解结果不保证单调 |

## 1. Hand-checkable instances A-G

| instance | mode | objective | hand expectation | agreement | checker | max violation | status |
|---|---|---:|---:|---|---|---:|---|
| A_ample | paper_literal | 5 | - | not-specified | PASS | 0.0e+00 | optimal |
| A_ample | audited_v1 | 5 | 5 | match | PASS | 0.0e+00 | optimal |
| B_short | paper_literal | 14 | - | not-specified | PASS | 0.0e+00 | optimal |
| B_short | audited_v1 | 14 | 14 | match | PASS | 0.0e+00 | optimal |
| C_incompat | paper_literal | 30 | - | not-specified | PASS | 0.0e+00 | optimal |
| C_incompat | audited_v1 | 30 | 30 | match | PASS | 0.0e+00 | optimal |
| D_down | paper_literal | 30 | - | not-specified | PASS | 0.0e+00 | optimal |
| D_down | audited_v1 | 30 | 30 | match | PASS | 0.0e+00 | optimal |
| E_early | paper_literal | 8 | - | not-specified | PASS | 0.0e+00 | optimal |
| E_early | audited_v1 | 8 | 8 | match | PASS | 0.0e+00 | optimal |
| F_carryover | paper_literal | 11 | 11 | match | PASS | 0.0e+00 | optimal |
| F_carryover | audited_v1 | 1 | 1 | match | PASS | 0.0e+00 | optimal |
| G_terminal_Z | paper_literal | 1 | 1 | match | PASS | 0.0e+00 | optimal |
| G_terminal_Z | audited_v1 | 10 | 10 | match | PASS | 0.0e+00 | optimal |

### Optimal plans

- **A_ample / paper_literal** (obj 5): `(i0,j0) X=[3] Y=[1] Z=[0] i0 I=[0] L=[0]`
- **A_ample / audited_v1** (obj 5): `(i0,j0) X=[3] Y=[1] Z=[0] i0 I=[0] L=[0]`
- A_ample: same objective in both modes
- **B_short / paper_literal** (obj 14): `(i0,j0) X=[2] Y=[1] Z=[0] i0 I=[0] L=[1]`
- **B_short / audited_v1** (obj 14): `(i0,j0) X=[2] Y=[1] Z=[0] i0 I=[0] L=[1]`
- B_short: same objective in both modes
- **C_incompat / paper_literal** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **C_incompat / audited_v1** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- C_incompat: same objective in both modes
- **D_down / paper_literal** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **D_down / audited_v1** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- D_down: same objective in both modes
- **E_early / paper_literal** (obj 8): `(i0,j0) X=[3,0] Y=[1,0] Z=[0,0] i0 I=[3,0] L=[0,0]`
- **E_early / audited_v1** (obj 8): `(i0,j0) X=[3,0] Y=[1,0] Z=[0,0] i0 I=[3,0] L=[0,0]`
- E_early: same objective in both modes
- **F_carryover / paper_literal** (obj 11): `(i0,j0) X=[1,1] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,1]`
- **F_carryover / audited_v1** (obj 1): `(i0,j0) X=[1,2] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,0]`
- F_carryover: **the two modes differ** by -10
- **G_terminal_Z / paper_literal** (obj 1): `(i0,j0) X=[1] Y=[1] Z=[1] i0 I=[0] L=[0]`
- **G_terminal_Z / audited_v1** (obj 10): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[1]`
- G_terminal_Z: **the two modes differ** by 9

## 2. Independent enumeration (audited_v1)

| instance | slots | configs | feasible | MILP obj | enum obj | delta | agree | expected | expectation | checker | max viol | enum time (s) |
|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|---:|---:|
| MI0_basic | 6 | 64 | 32 | 21 | 21 | 0.00e+00 | yes | 21 | match | PASS | 0.0e+00 | 0.114 |
| MI1_incompat | 6 | 64 | 6 | 39 | 39 | 0.00e+00 | yes | 39 | match | PASS | 0.0e+00 | 0.106 |
| MI2_zero_demand | 6 | 64 | 1 | 0 | 0 | 0.00e+00 | yes | 0 | match | PASS | 0.0e+00 | 0.099 |
| MI3_initial_inv | 6 | 64 | 32 | 3 | 3 | 0.00e+00 | yes | 3 | match | PASS | 0.0e+00 | 0.107 |
| MI4_tight | 6 | 64 | 12 | 57 | 57 | 0.00e+00 | yes | 57 | match | PASS | 0.0e+00 | 0.102 |
| MI5_zero_cap | 6 | 64 | 1 | 42 | 42 | 0.00e+00 | yes | 42 | match | PASS | 0.0e+00 | 0.099 |
| MI6_cheap_lost | 6 | 64 | 32 | 12 | 12 | 0.00e+00 | yes | 12 | match | PASS | 0.0e+00 | 0.108 |
| MI7_setup_time | 6 | 64 | 20 | 112 | 112 | 0.00e+00 | yes | 112 | match | PASS | 0.0e+00 | 0.106 |

### Optimal plans

- **MI0_basic** (obj 21): `(i0,j0) X=[2,2] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,0] (i1,j0) X=[2,2] Y=[1,1] Z=[0,0] i1 I=[0,0] L=[0,0]`
- **MI1_incompat** (obj 39): `(i0,j0) X=[0,0] Y=[0,0] Z=[0,0] i0 I=[0,0] L=[2,1] (i1,j0) X=[2,2] Y=[1,0] Z=[1,0] i1 I=[0,0] L=[0,0]`
- **MI2_zero_demand** (obj 0): `(i0,j0) X=[0,0] Y=[0,0] Z=[0,0] i0 I=[0,0] L=[0,0] (i1,j0) X=[0,0] Y=[0,0] Z=[0,0] i1 I=[0,0] L=[0,0]`
- **MI3_initial_inv** (obj 3): `(i0,j0) X=[0,0] Y=[0,0] Z=[0,0] i0 I=[2,0] L=[0,0] (i1,j0) X=[0,0] Y=[0,0] Z=[0,0] i1 I=[1,0] L=[0,0]`
- **MI4_tight** (obj 57): `(i0,j0) X=[1,3] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[2,0] (i1,j0) X=[0,2] Y=[0,1] Z=[0,0] i1 I=[0,0] L=[1,0]`
- **MI5_zero_cap** (obj 42): `(i0,j0) X=[0,0] Y=[0,0] Z=[0,0] i0 I=[0,0] L=[2,2] (i1,j0) X=[0,0] Y=[0,0] Z=[0,0] i1 I=[0,0] L=[1,1]`
- **MI6_cheap_lost** (obj 12): `(i0,j0) X=[2,2] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,0] (i1,j0) X=[0,1] Y=[0,1] Z=[0,0] i1 I=[0,0] L=[1,0]`
- **MI7_setup_time** (obj 112): `(i0,j0) X=[1,2] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[1,0] (i1,j0) X=[0,1] Y=[0,1] Z=[0,0] i1 I=[0,0] L=[3,0]`

## 3. Fix-and-release interface

| instance | free Y | ref obj | fix-all delta | fix-all ok | fix Y only obj | ok | ref feasible | monotone | sweep |
|---|---:|---:|---:|---|---:|---|---|---|---|
| MI0_basic | 2 | 21 | 0.00e+00 | PASS | 21 | PASS | True | yes | k0=21; k1=21; k2=21 |
| MI1_incompat | 2 | 39 | 0.00e+00 | PASS | 39 | PASS | True | yes | k0=39; k1=39; k2=39 |
| MI2_zero_demand | 2 | 0 | 0.00e+00 | PASS | 0 | PASS | True | yes | k0=0; k1=0; k2=0 |
| MI3_initial_inv | 2 | 3 | 0.00e+00 | PASS | 3 | PASS | True | yes | k0=3; k1=3; k2=3 |
| MI4_tight | 2 | 57 | 0.00e+00 | PASS | 57 | PASS | True | yes | k0=57; k1=57; k2=57 |
| MI5_zero_cap | 2 | 42 | 0.00e+00 | PASS | 42 | PASS | True | yes | k0=42; k1=42; k2=42 |
| MI6_cheap_lost | 2 | 12 | 0.00e+00 | PASS | 12 | PASS | True | yes | k0=12; k1=12; k2=12 |
| MI7_setup_time | 2 | 112 | 0.00e+00 | PASS | 112 | PASS | True | yes | k0=112; k1=112; k2=112 |

## 4. Where the two modes diverge


### F_carryover

- **paper_literal** obj 11: `(i0,j0) X=[1,1] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,1]`
  - activation upper bounds U_ijt: `{'i0,j0,t1': 1.0, 'i0,j0,t2': 1.0}`
  - Z_ijT pinned to zero: False
  - binding upper-bound rows: `['inv_balance[i0,t1]', 'inv_balance[i0,t2]', 'capacity[j0,t1]', 'carry_implies_setup[i0,j0,t1]', 'carry_implies_setup[i0,j0,t2]', 'no_two_carry[i0,j0,t1]', 'no_two_carry[i0,j0,t2]', 'prod_activation[i0,j0,t1]', 'prod_activation[i0,j0,t2]', 'one_carry[j0,t1]']`
- **audited_v1** obj 1: `(i0,j0) X=[1,2] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,0]`
  - activation upper bounds U_ijt: `{'i0,j0,t1': 2.0, 'i0,j0,t2': 2.0}`
  - Z_ijT pinned to zero: True
  - binding upper-bound rows: `['inv_balance[i0,t1]', 'inv_balance[i0,t2]', 'capacity[j0,t1]', 'capacity[j0,t2]', 'carry_implies_setup[i0,j0,t1]', 'carry_implies_setup[i0,j0,t2]', 'no_two_carry[i0,j0,t1]', 'no_two_carry[i0,j0,t2]', 'prod_activation[i0,j0,t2]', 'one_carry[j0,t1]']`

#### F_carryover / paper_literal : all 16 binary configurations, best first

| rank | assignment | optimal objective | status |
|---:|---|---:|---|
| 1 | `Y001=1, Y002=0, Z001=1, Z002=0` | 11 | optimal |
| 2 | `Y001=1, Y002=1, Z001=0, Z002=0` | 12 | optimal |
| 3 | `Y001=1, Y002=1, Z001=0, Z002=1` | 12 | optimal |
| 4 | `Y001=1, Y002=1, Z001=1, Z002=0` | 12 | optimal |
| 5 | `Y001=0, Y002=1, Z001=0, Z002=0` | 21 | optimal |
| 6 | `Y001=0, Y002=1, Z001=0, Z002=1` | 21 | optimal |
| 7 | `Y001=1, Y002=0, Z001=0, Z002=0` | 21 | optimal |
| 8 | `Y001=0, Y002=0, Z001=0, Z002=0` | 30 | optimal |
| 9 | `Y001=0, Y002=0, Z001=0, Z002=1` | - | infeasible |
| 10 | `Y001=0, Y002=0, Z001=1, Z002=0` | - | infeasible |
| 11 | `Y001=0, Y002=0, Z001=1, Z002=1` | - | infeasible |
| 12 | `Y001=0, Y002=1, Z001=1, Z002=0` | - | infeasible |
| 13 | `Y001=0, Y002=1, Z001=1, Z002=1` | - | infeasible |
| 14 | `Y001=1, Y002=0, Z001=0, Z002=1` | - | infeasible |
| 15 | `Y001=1, Y002=0, Z001=1, Z002=1` | - | infeasible |
| 16 | `Y001=1, Y002=1, Z001=1, Z002=1` | - | infeasible |

#### F_carryover / audited_v1 : all 8 binary configurations, best first

| rank | assignment | optimal objective | status |
|---:|---|---:|---|
| 1 | `Y001=1, Y002=0, Z001=1` | 1 | optimal |
| 2 | `Y001=1, Y002=1, Z001=0` | 12 | optimal |
| 3 | `Y001=1, Y002=1, Z001=1` | 12 | optimal |
| 4 | `Y001=0, Y002=1, Z001=0` | 21 | optimal |
| 5 | `Y001=1, Y002=0, Z001=0` | 21 | optimal |
| 6 | `Y001=0, Y002=0, Z001=0` | 30 | optimal |
| 7 | `Y001=0, Y002=0, Z001=1` | - | infeasible |
| 8 | `Y001=0, Y002=1, Z001=1` | - | infeasible |

### G_terminal_Z

- **paper_literal** obj 1: `(i0,j0) X=[1] Y=[1] Z=[1] i0 I=[0] L=[0]`
  - activation upper bounds U_ijt: `{'i0,j0,t1': 1.0}`
  - Z_ijT pinned to zero: False
  - binding upper-bound rows: `['inv_balance[i0,t1]', 'capacity[j0,t1]', 'carry_implies_setup[i0,j0,t1]', 'no_two_carry[i0,j0,t1]', 'prod_activation[i0,j0,t1]', 'one_carry[j0,t1]']`
- **audited_v1** obj 10: `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[1]`
  - activation upper bounds U_ijt: `{'i0,j0,t1': 1.0}`
  - Z_ijT pinned to zero: True
  - binding upper-bound rows: `['inv_balance[i0,t1]', 'lost_ub[i0,t1]', 'carry_implies_setup[i0,j0,t1]', 'prod_activation[i0,j0,t1]']`

#### G_terminal_Z / paper_literal : all 4 binary configurations, best first

| rank | assignment | optimal objective | status |
|---:|---|---:|---|
| 1 | `Y001=1, Z001=1` | 1 | optimal |
| 2 | `Y001=0, Z001=0` | 10 | optimal |
| 3 | `Y001=0, Z001=1` | - | infeasible |
| 4 | `Y001=1, Z001=0` | - | infeasible |

#### G_terminal_Z / audited_v1 : all 2 binary configurations, best first

| rank | assignment | optimal objective | status |
|---:|---|---:|---|
| 1 | `Y001=0` | 10 | optimal |
| 2 | `Y001=1` | - | infeasible |

### Mode-specific notes

- A_ample / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
- B_short / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
- C_incompat / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
- D_down / paper_literal: activation upper bound was negative for 1 (i,j,t) combination(s) because c_jt < s_i; clipped to 0 since the right-hand side of (1.8) cannot be negative. Affected (i,j,t,raw value): [(0, 0, 1, -1.0)]
- D_down / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
- E_early / paper_literal: activation upper bound was negative for 1 (i,j,t) combination(s) because c_jt < s_i; clipped to 0 since the right-hand side of (1.8) cannot be negative. Affected (i,j,t,raw value): [(0, 0, 2, -1.0)]
- E_early / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
- F_carryover / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
- G_terminal_Z / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j). Because (1.9) is kept exactly as published, X_ijT >= m_i (Y_ijT - Z_ijT) then degenerates to X_ijT >= m_i Y_ijT, so the final setup can no longer be used to exit below the minimum quantity.
