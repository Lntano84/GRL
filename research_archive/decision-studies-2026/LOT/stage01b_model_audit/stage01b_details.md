# stage01b_details.md

Machine-generated detail tables. Produced by `python run.py` in `stage01b_model_audit`.

Environment: python 3.13.14, numpy 2.5.2, scipy 1.17.1, Windows-11-10.0.26200-SP0


## 0. Summary

| 检查 | 结果 | 最大误差或违反量 |
|---|---|---:|
| 模型定义与实现一致 | PASS | U 见第 5 节 |
| 原 A–G 回归 | PASS | 0.0e+00 |
| 八个实例：独立构造 LP 与 MILP 一致 | PASS | 0.0e+00 |
| H：成本 2 | PASS（实测 2） | 0.0e+00 |
| I：成本 1 | PASS（实测 1） | 0.0e+00 |
| J：30 → 5 | PASS（30 → 5） | 0.0e+00 |
| 不相容固定条件被判不可行 | PASS | - |

## 1. Hand instances A-G (regression, both modes)

| instance | mode | objective | hand expectation | agreement | checker | max violation |
|---|---|---:|---:|---|---|---:|
| A_ample | paper_literal | 5 | None | not-specified | PASS | 0.0e+00 |
| A_ample | audited_v1 | 5 | 5 | match | PASS | 0.0e+00 |
| B_short | paper_literal | 14 | None | not-specified | PASS | 0.0e+00 |
| B_short | audited_v1 | 14 | 14 | match | PASS | 0.0e+00 |
| C_incompat | paper_literal | 30 | None | not-specified | PASS | 0.0e+00 |
| C_incompat | audited_v1 | 30 | 30 | match | PASS | 0.0e+00 |
| D_down | paper_literal | 30 | None | not-specified | PASS | 0.0e+00 |
| D_down | audited_v1 | 30 | 30 | match | PASS | 0.0e+00 |
| E_early | paper_literal | 8 | None | not-specified | PASS | 0.0e+00 |
| E_early | audited_v1 | 8 | 8 | match | PASS | 0.0e+00 |
| F_carryover | paper_literal | 11 | 11 | match | PASS | 0.0e+00 |
| F_carryover | audited_v1 | 1 | 1 | match | PASS | 0.0e+00 |
| G_terminal_Z | paper_literal | 1 | 1 | match | PASS | 0.0e+00 |
| G_terminal_Z | audited_v1 | 10 | 10 | match | PASS | 0.0e+00 |

### Optimal plans

- **A_ample / paper_literal** (obj 5): `(i0,j0) X=[3] Y=[1] Z=[0] i0 I=[0] L=[0]`
- **A_ample / audited_v1** (obj 5): `(i0,j0) X=[3] Y=[1] Z=[0] i0 I=[0] L=[0]`
- **B_short / paper_literal** (obj 14): `(i0,j0) X=[2] Y=[1] Z=[0] i0 I=[0] L=[1]`
- **B_short / audited_v1** (obj 14): `(i0,j0) X=[2] Y=[1] Z=[0] i0 I=[0] L=[1]`
- **C_incompat / paper_literal** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **C_incompat / audited_v1** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **D_down / paper_literal** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **D_down / audited_v1** (obj 30): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **E_early / paper_literal** (obj 8): `(i0,j0) X=[3,0] Y=[1,0] Z=[0,0] i0 I=[3,0] L=[0,0]`
- **E_early / audited_v1** (obj 8): `(i0,j0) X=[3,0] Y=[1,0] Z=[0,0] i0 I=[3,0] L=[0,0]`
- **F_carryover / paper_literal** (obj 11): `(i0,j0) X=[1,1] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,1]`
- **F_carryover / audited_v1** (obj 1): `(i0,j0) X=[1,2] Y=[1,0] Z=[1,0] i0 I=[0,0] L=[0,0]`
- **G_terminal_Z / paper_literal** (obj 1): `(i0,j0) X=[1] Y=[1] Z=[1] i0 I=[0] L=[0]`
- **G_terminal_Z / audited_v1** (obj 10): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[1]`

## 2. Independent LP path vs MILP (audited_v1)

| instance | configs | optimal | LP-infeas | binary-infeas | MILP | indep(MILP binaries) | delta | indep(full enum) | delta | baseline | agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| MI0_basic | 64 | 32 | 0 | 32 | 21 | 21 | 0.0e+00 | 21 | 0.0e+00 | 21 | match |
| MI1_incompat | 64 | 6 | 0 | 58 | 39 | 39 | 0.0e+00 | 39 | 0.0e+00 | 39 | match |
| MI2_zero_demand | 64 | 32 | 0 | 32 | 0 | 0 | 0.0e+00 | 0 | 0.0e+00 | 0 | match |
| MI3_initial_inv | 64 | 32 | 0 | 32 | 3 | 3 | 0.0e+00 | 3 | 0.0e+00 | 3 | match |
| MI4_tight | 64 | 12 | 20 | 32 | 57 | 57 | 0.0e+00 | 57 | 0.0e+00 | 57 | match |
| MI5_zero_cap | 64 | 1 | 31 | 32 | 42 | 42 | 0.0e+00 | 42 | 0.0e+00 | 42 | match |
| MI6_cheap_lost | 64 | 32 | 0 | 32 | 12 | 12 | 0.0e+00 | 12 | 0.0e+00 | 12 | match |
| MI7_setup_time | 64 | 20 | 12 | 32 | 112 | 112 | 0.0e+00 | 112 | 0.0e+00 | 112 | match |

## 3. tau-aware fix-and-release

Reference plan for every instance below produces nothing (all demand lost), so improvement is possible; the earlier round used the unrestricted optimum as the reference, which could never be improved.

| instance | T | tau | reference obj | frozen obj | released obj | expectation | ok |
|---|---:|---:|---:|---:|---:|---:|---|
| H_minlot_stock | 1 | 1 | 10 | 10 | 2 | 2 | PASS |
| I_tau_freedom | 2 | 1 | 10 | 1 | 1 | 1 | PASS |
| J_release_matters | 2 | 1 | 30 | 30 | 5 | 5 | PASS |

### Plans

- **H_minlot_stock** frozen (tau=1): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[1]`
- **H_minlot_stock** released: `(i0,j0) X=[2] Y=[1] Z=[0] i0 I=[1] L=[0]`
- **I_tau_freedom** frozen (tau=1): `(i0,j0) X=[0,1] Y=[0,1] Z=[0,0] i0 I=[0,0] L=[0,0]`
- **I_tau_freedom** released: `(i0,j0) X=[0,1] Y=[0,1] Z=[0,0] i0 I=[0,0] L=[0,0]`
- **J_release_matters** frozen (tau=1): `(i0,j0) X=[0] Y=[0] Z=[0] i0 I=[0] L=[3]`
- **J_release_matters** released: `(i0,j0) X=[3] Y=[1] Z=[0] i0 I=[0] L=[0]`

## 4. Negative control

- instance `D_fix_X`: zero capacity, so no production is possible

- fixing actually applied: `X[0,0,1] = 1.0` (flat variable index 0)

- with the fixing: **FAILED(status=2)**

- without the fixing: **optimal**, objective 30


## 5. paper_literal: negative U_ijt is kept and matters

| instance | negative U_ijt | configs | optimal | LP-infeasible | binary-rejected | MILP | independent | delta |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| D_down | `{'(0, 0, 1)': -1.0}` | 4 | 1 | 2 | 1 | 30 | 30 | 0.0e+00 |
| E_early | `{'(0, 0, 2)': -1.0}` | 16 | 2 | 6 | 8 | 8 | 8 | 0.0e+00 |
| G_terminal_Z | `{}` | 4 | 2 | 1 | 1 | 1 | 1 | 0.0e+00 |
| C_incompat | `{}` | 4 | 1 | 0 | 3 | 30 | 30 | 0.0e+00 |
| A_ample | `{}` | 4 | 3 | 0 | 1 | 5 | 5 | 0.0e+00 |
| B_short | `{}` | 4 | 3 | 0 | 1 | 14 | 14 | 0.0e+00 |

A negative U makes (1.8) read `X <= U*(Y+Z)` with U < 0, which forces X = 0 for every value of the binaries. The value is kept exactly as published.


## 6. U_ijt actually used by the code

| instance | mode | U_ijt |
|---|---|---|
| A_ample | paper_literal | `{'i0,j0,t1': 3.0}` |
| A_ample | audited_v1 | `{'i0,j0,t1': 5.0}` |
| B_short | paper_literal | `{'i0,j0,t1': 2.0}` |
| B_short | audited_v1 | `{'i0,j0,t1': 3.0}` |
| C_incompat | paper_literal | `{'i0,j0,t1': 3.0}` |
| C_incompat | audited_v1 | `{'i0,j0,t1': 5.0}` |
| D_down | paper_literal | `{'i0,j0,t1': -1.0}` |
| D_down | audited_v1 | `{'i0,j0,t1': 0.0}` |
| E_early | paper_literal | `{'i0,j0,t1': 3.0, 'i0,j0,t2': -1.0}` |
| E_early | audited_v1 | `{'i0,j0,t1': 5.0, 'i0,j0,t2': 0.0}` |
| F_carryover | paper_literal | `{'i0,j0,t1': 1.0, 'i0,j0,t2': 1.0}` |
| F_carryover | audited_v1 | `{'i0,j0,t1': 2.0, 'i0,j0,t2': 2.0}` |
| G_terminal_Z | paper_literal | `{'i0,j0,t1': 1.0}` |
| G_terminal_Z | audited_v1 | `{'i0,j0,t1': 1.0}` |
| H_minlot_stock | paper_literal | `{'i0,j0,t1': 1.0}` |
| H_minlot_stock | audited_v1 | `{'i0,j0,t1': 3.0}` |
| I_tau_freedom | paper_literal | `{'i0,j0,t1': -1.0, 'i0,j0,t2': 1.0}` |
| I_tau_freedom | audited_v1 | `{'i0,j0,t1': 0.0, 'i0,j0,t2': 2.0}` |
| J_release_matters | paper_literal | `{'i0,j0,t1': 3.0}` |
| J_release_matters | audited_v1 | `{'i0,j0,t1': 5.0}` |
| D_fix_X | paper_literal | `{'i0,j0,t1': -1.0}` |
| D_fix_X | audited_v1 | `{'i0,j0,t1': 0.0}` |

### Notes emitted by the model

- A_ample / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- B_short / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- C_incompat / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- D_down / paper_literal: paper_literal: U_ijt is negative for 1 (i,j,t) combination(s) because c_jt < s_i. The negative value is KEPT, not clipped: (1.8) then forces X = 0 and may make some binary configurations infeasible, which is the published formulation's behaviour. Affected (i,j,t,value): [(0, 0, 1, -1.0)]
- D_down / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- E_early / paper_literal: paper_literal: U_ijt is negative for 1 (i,j,t) combination(s) because c_jt < s_i. The negative value is KEPT, not clipped: (1.8) then forces X = 0 and may make some binary configurations infeasible, which is the published formulation's behaviour. Affected (i,j,t,value): [(0, 0, 2, -1.0)]
- E_early / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- F_carryover / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- G_terminal_Z / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- H_minlot_stock / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- I_tau_freedom / paper_literal: paper_literal: U_ijt is negative for 1 (i,j,t) combination(s) because c_jt < s_i. The negative value is KEPT, not clipped: (1.8) then forces X = 0 and may make some binary configurations infeasible, which is the published formulation's behaviour. Affected (i,j,t,value): [(0, 0, 1, -1.0)]
- I_tau_freedom / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- J_release_matters / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
- D_fix_X / paper_literal: paper_literal: U_ijt is negative for 1 (i,j,t) combination(s) because c_jt < s_i. The negative value is KEPT, not clipped: (1.8) then forces X = 0 and may make some binary configurations infeasible, which is the published formulation's behaviour. Affected (i,j,t,value): [(0, 0, 1, -1.0)]
- D_fix_X / audited_v1: D2 applied: Z_ijT fixed to 0 for all (i,j), because Z_ijT would describe a carry-over out of the last period, where no successor period exists. Effect on constraint (1.9), which is otherwise kept exactly as published: X_ijT >= m_i (Y_ijT - Z_ijT) degenerates to X_ijT >= m_i Y_ijT, so a setup in the final period must produce at least the minimum lot. A setup that produces more than the remaining demand is LEGAL and leaves end-of-horizon inventory, which is charged the holding cost h_i through the objective (1.1); this is an accepted feature of the main model, not a defect. It is exactly the mechanism instance H exercises.
