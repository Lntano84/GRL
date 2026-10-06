"""
HiGHS transfer layer for stage 05.

The model is NOT re-translated. The already-audited vectors from `lsp_model.build_model`
(objective, constraint matrix, row bounds, variable bounds, integrality) are handed to HiGHS
as-is, and the method's fixing rows and the stability row are appended exactly as
`stage04b_run.System` appends them. That keeps the MILP identical to the one used in
stage 04/04b and isolates the only change under study: whether the solver receives the
repaired plan as a MIP start.

Verified interface facts (highspy 1.15.1)
----------------------------------------
* `Highs.setSolution(num_col, indices, values)` returns `HighsStatus.kOk`; the values are
  taken as a candidate MIP start.
* Adoption is confirmed from the HiGHS log line
      `MIP start solution is feasible, objective value is <v>`
  A `kOk` from `setSolution` alone is NOT evidence of adoption, so the log is parsed and the
  parsed line is stored with every warm run.
* `Highs.getInfo()` exposes `objective_function_value`, `mip_dual_bound`, `mip_gap`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lsp_model import MODE_AUDITED, Solution, unpack

try:
    import highspy
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "highspy is required for stage 05. It is installed in the isolated environment:\n"
        "    stage05_mipstart/.venv-hs/Scripts/python.exe -m pip install "
        "--index-url https://pypi.org/simple highspy\n"
        "Run the stage-05 scripts with that interpreter.\n"
        f"(import error: {exc})"
    ) from exc

ADOPTION_RE = re.compile(
    r"MIP start solution is feasible,\s*objective value is\s*([-+0-9.eE]+)")
INFEASIBLE_START_RE = re.compile(r"MIP start solution is not feasible", re.I)


@dataclass
class HighsRun:
    status: str
    objective: Optional[float]
    mip_gap: Optional[float]
    dual_bound: Optional[float]
    solution: Optional[Solution]
    set_solution_status: Optional[str]
    start_adopted: bool
    start_reported_objective: Optional[float]
    start_message: Optional[str]
    log: str
    solved_wall_s: float
    transfer_wall_s: float
    n_cols: int
    n_rows: int


def _rowwise(matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SciPy CSR -> HiGHS rowwise arrays (start, index, value)."""
    csr = matrix.tocsr()
    return (csr.indptr.astype(np.int32), csr.indices.astype(np.int32),
            csr.data.astype(np.float64))


def build_highs(bm, fixed: Dict[int, float], stab_row: Tuple[Dict[int, float], float]):
    """Assemble the HiGHS model: base system + fixing rows + the stability row."""
    import highspy
    from scipy.sparse import coo_matrix, vstack as sp_vstack

    n = bm.idx.n
    A = bm.A.tocsr()
    row_lb = bm.row_lb.copy()
    row_ub = bm.row_ub.copy()
    lb = bm.var_lb.copy()
    ub = bm.var_ub.copy()

    coef, rhs = stab_row
    cols = list(coef.keys())
    A_stab = coo_matrix(([coef[c] for c in cols], ([0] * len(cols), cols)),
                        shape=(1, n)).tocsr()
    A = sp_vstack([A, A_stab], format="csr")
    row_lb = np.concatenate([row_lb, [-np.inf]])
    row_ub = np.concatenate([row_ub, [rhs]])

    if fixed:
        k = list(fixed.keys())
        A_fix = coo_matrix((np.ones(len(k)), (np.arange(len(k)), np.array(k))),
                           shape=(len(k), n)).tocsr()
        A = sp_vstack([A, A_fix], format="csr")
        v = np.array([fixed[c] for c in k], dtype=float)
        row_lb = np.concatenate([row_lb, v])
        row_ub = np.concatenate([row_ub, v])
        for c in k:
            lb[c] = fixed[c]
            ub[c] = fixed[c]

    start, index, value = _rowwise(A)
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("log_to_console", False)

    lp = highspy.HighsLp()
    lp.num_col_ = n
    lp.num_row_ = A.shape[0]
    lp.col_cost_ = np.asarray(bm.c, dtype=np.float64)
    lp.col_lower_ = lb.astype(np.float64)
    lp.col_upper_ = ub.astype(np.float64)
    lp.row_lower_ = row_lb.astype(np.float64)
    lp.row_upper_ = row_ub.astype(np.float64)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = start
    lp.a_matrix_.index_ = index
    lp.a_matrix_.value_ = value
    lp.integrality_ = [highspy.HighsVarType.kInteger if v == 1
                       else highspy.HighsVarType.kContinuous
                       for v in bm.integrality]
    return h, lp, int(n), int(A.shape[0])


def solve_highs(bm, fixed: Dict[int, float], stab_row, budget: float, seed: int,
                start_values: Optional[np.ndarray] = None,
                threads: int = 1) -> HighsRun:
    """Solve with HiGHS. When `start_values` is given it is submitted via setSolution.

    `setSolution` is called AFTER the full model (including the stability row and the fixing
    rows) is loaded, and the same `start_values` is submitted for every method, so all methods
    receive the same initial plan.
    """
    import time
    import os
    import tempfile
    import uuid
    t_tr0 = time.perf_counter()
    h, lp, n_cols, n_rows = build_highs(bm, fixed, stab_row)

    # Adoption evidence comes from the HiGHS log. `Highs` has no getLog() (that was a
    # callback in an earlier probe), so the documented `log_file` option is used and the file
    # is read back after the solve. The name MUST be unique per call: `id(h)` is NOT safe
    # because CPython reuses addresses after garbage collection, which made successive runs
    # collide on one path and a cold run read the previous warm run's log.
    log_path = os.path.join(tempfile.gettempdir(),
                            "hs_stage05_%d_%s.log" % (os.getpid(), uuid.uuid4().hex))
    if os.path.exists(log_path):
        os.remove(log_path)
    h.setOptionValue("output_flag", True)
    h.setOptionValue("log_to_console", False)
    h.setOptionValue("log_file", log_path)

    h.passModel(lp)
    h.setOptionValue("time_limit", float(budget))
    h.setOptionValue("random_seed", int(seed))
    h.setOptionValue("threads", int(threads))
    set_status = None
    if start_values is not None:
        set_status = str(h.setSolution(n_cols, np.arange(n_cols, dtype=np.int32),
                                       np.asarray(start_values, dtype=np.float64)))
    t_transfer = time.perf_counter() - t_tr0

    t0 = time.perf_counter()
    h.run()
    solved = time.perf_counter() - t0

    status = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    sol = h.getSolution()
    log_txt = ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            log_txt = fh.read()
    except OSError:
        log_txt = ""
    try:
        os.remove(log_path)          # Windows may still hold the handle; not an error
    except OSError:
        pass

    adopted = False
    reported_obj = None
    message = None
    m = ADOPTION_RE.search(log_txt)
    if m:
        adopted = True
        try:
            reported_obj = float(m.group(1))
        except ValueError:
            reported_obj = None
        message = m.group(0)
    elif INFEASIBLE_START_RE.search(log_txt):
        message = "MIP start solution is not feasible"

    out_sol = None
    obj = None
    if sol.col_value is not None and len(sol.col_value) == n_cols:
        x = np.asarray(sol.col_value, dtype=float)
        if np.all(np.isfinite(x)):
            X, Y, Z, I, L = unpack(x, bm.idx)
            obj = None
            try:
                obj = float(info.objective_function_value)
            except Exception:
                obj = None
            out_sol = Solution(status=status, objective=obj if obj is not None else float("nan"),
                               X=X, Y=Y, Z=Z, I=I, L=L,
                               mip_gap=(float(info.mip_gap)
                                        if np.isfinite(info.mip_gap) else None),
                               solve_time_s=solved, mode=MODE_AUDITED)

    return HighsRun(status=status, objective=obj,
                    mip_gap=(float(info.mip_gap) if np.isfinite(info.mip_gap) else None),
                    dual_bound=(float(info.mip_dual_bound)
                                if np.isfinite(info.mip_dual_bound) else None),
                    solution=out_sol, set_solution_status=set_status,
                    start_adopted=adopted, start_reported_objective=reported_obj,
                    start_message=message, log=log_txt, solved_wall_s=solved,
                    transfer_wall_s=t_transfer, n_cols=n_cols, n_rows=n_rows)


def repair_vector(bm, repair: Solution) -> np.ndarray:
    """The FULL variable vector (X, Y, Z incl. Z_ij0, I, L) of the repaired plan.

    Same initial plan is submitted for every method and both start conditions, as required.
    """
    n = bm.idx.n
    x = np.zeros(n)
    for i in range(bm.inst.N):
        for j in range(bm.inst.M):
            for t in range(1, bm.inst.T + 1):
                x[bm.idx.X(i, j, t)] = float(repair.X[i, j, t - 1])
                x[bm.idx.Y(i, j, t)] = float(repair.Y[i, j, t - 1])
                x[bm.idx.Z(i, j, t)] = float(repair.Z[i, j, t - 1])
            x[bm.idx.Z(i, j, 0)] = 0.0
        for t in range(1, bm.inst.T + 1):
            x[bm.idx.I(i, t)] = float(repair.I[i, t - 1])
            x[bm.idx.L(i, t)] = float(repair.L[i, t - 1])
    return x
