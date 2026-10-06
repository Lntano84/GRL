"""Q01-R steps R2/R3: the implicit re-execution path, and a correct classification of the reads.

Two corrections to the previous round, both driven by the audit:

R2. "no retry arm" was WRONG.  When a run is cancelled and the hidden runtime is below the current
    best, the official branch leaves ``mask = 0`` and ``explored_m = 0`` for the cell, so the cell is
    still eligible at the candidate filter on the next iteration and can be executed again, at a
    longer cap.  There is no separately NAMED retry action, but there is a path that re-executes a
    timed-out candidate, and that path's eligibility depends on the hidden runtime.

R3. "nine violations" was WRONG.  A replay environment must read the hidden runtime in order to
    produce the feedback for the run it just performed; that read is not automatically a violation.
    The test is whether the information read EXCEEDS what that execution could have revealed.
    This script classifies each read site on that basis and isolates the one that is clearly
    privileged regardless: the stopping rule, which reads the per-row minimum of the FULL hidden
    matrix (``opt_time``).
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIRROR = HERE / "limeqo_mirror" / "src"
SRC = MIRROR / "strategies"


def official_branch(hidden_x, best, cap):
    """The official state update for one candidate, with the deadline set to ``cap``.

    This mirrors limeqo.py:79-99 exactly, including the fact that the effective deadline is
    ``timeout_tolerance`` and NOT some separate cap:

        timeout_tolerance = min(alpha * min_observed[select], beta * pred_m[select, hint])
        ...
        if hidden_x >= min_observed:      explored_m[...] = 1
        if hidden_x >= timeout_tolerance: timeout_m[...] = timeout_tolerance ; continue
        mask[...] = 1 ; explored_m[...] = 1 ; cnt += 1
    """
    tol = cap                      # the deadline the run was actually given
    st = {"mask": 0, "explored_m": 0, "timeout_m": None, "counted": False}
    if hidden_x >= best:
        st["explored_m"] = 1
    if hidden_x >= tol:
        st["timeout_m"] = tol
        return st
    st["mask"] = 1
    st["explored_m"] = 1
    st["counted"] = True
    return st


def check_official():
    print("=" * 100)
    print("  R2 -- WHAT THE OFFICIAL BRANCH DOES, WITH THE DEADLINE SET CORRECTLY")
    print("=" * 100)
    print("  best b = 10; the deadline IS the tolerance, so cap = tolerance in every row.\n")
    print(f"  {'cap':>4} {'hidden x':>9} {'x>=b':>6} {'x>=cap':>7} {'mask':>5} {'explored_m':>11} "
          f"{'timeout_m':>10} {'counted':>8}  {'eligible again?':>16}")
    rows = []
    for cap in (2, 10):
        for x in (5, 20):
            st = official_branch(x, 10, cap)
            elig = (st["mask"] == 0 and st["explored_m"] == 0)
            rows.append({"cap": cap, "x": x, **st, "eligible_again": elig})
            print(f"  {cap:>4} {x:>9} {str(x >= 10):>6} {str(x >= cap):>7} {st['mask']:>5} "
                  f"{st['explored_m']:>11} {str(st['timeout_m']):>10} {str(st['counted']):>8}  "
                  f"{str(elig):>16}")
    print()
    print("  READ THIS TABLE")
    print("  * cap = 2: BOTH hidden values time out (correct, both charged 2 seconds).  They differ")
    print("    only in ``explored_m``: x=5 stays eligible for a later run, x=20 is retired.  That")
    print("    difference is decided by the hidden runtime, and 'runtime > 2' does not reveal it.")
    print("  * cap = 10: x=5 completes and x=20 times out, so the two states are ALLOWED to differ.")
    print("    My previous round wrongly treated the cap=10 row as if it had been cancelled at 2.")
    return rows


def trace_reexecution():
    print("\n" + "=" * 100)
    print("  R2b -- THE RE-EXECUTION PATH (retry), TRACED BY HAND ON THE BRANCH ABOVE")
    print("=" * 100)
    b, x = 10.0, 5.0
    print(f"  target cell: true runtime x = {x}, current best b = {b}")
    print()
    print(f"  iteration 1, cap = 2:")
    s1 = official_branch(x, b, 2)
    print(f"    branch result: mask={s1['mask']} explored_m={s1['explored_m']} "
          f"timeout_m={s1['timeout_m']} counted={s1['counted']}")
    print(f"    cost charged: 2 seconds (the deadline it ran to)")
    print(f"    eligible for a later run? {s1['mask'] == 0 and s1['explored_m'] == 0}"
          f"   <- the earlier claim that there is no retry path is false")
    print()
    print(f"  iteration 2, cap = 10 (the candidate filter checks explored_m, which is 0):")
    s2 = official_branch(x, b, 10)
    print(f"    branch result: mask={s2['mask']} explored_m={s2['explored_m']} "
          f"timeout_m={s2['timeout_m']} counted={s2['counted']}")
    print(f"    cost charged: {x} seconds (the run is redone FROM SCRATCH)")
    print()
    print(f"  total cost of the cell = 2 + {x} = {2 + x} seconds, i.e. {2 + x:.0f}")
    print("  This is the implicit retry the audit identified.")
    print()
    print("  CORRECT STATEMENT: there is no separately NAMED retry action, but there IS a path that")
    print("  re-executes a timed-out candidate, and whether a cell stays on that path is decided by")
    print("  the hidden runtime via explored_m.")
    print()
    print("  Consequence for the research question: 'introducing retry' cannot be sold as a new")
    print("  capability relative to this implementation.  What remains open is WHEN a re-execution is")
    print("  worth its cost, which this implementation does not decide explicitly.")
    return {"check1": {"cap": 2, "x": x, **s1}, "check3": {"cap": 10, "x": x, **s2},
            "total_cost": 2 + x}


def classify_reads():
    print("\n" + "=" * 100)
    print("  R3 -- READ SITES CLASSIFIED BY 'DOES THE READ EXCEED WHAT THE RUN REVEALED?'")
    print("=" * 100)
    sites = [
        ("strategies/limeqo.py", 89,
         "if dataset.matrix[select, hint] >= min_observed[select]:",
         "DECIDES explored_m for the candidate just executed",
         "VIOLATION: eligibility decided by a comparison (x >= b) that the run did not reveal"),
        ("strategies/limeqo.py", 92,
         "if dataset.matrix[select, hint] >= timeout_tolerance:",
         "decides measurement vs censored bound for the candidate just executed",
         "OK: this is the completion test for the run just performed, at the cap it was given; "
         "an environment must know it to report which kind of feedback occurred"),
        ("strategies/limeqo.py", 114,
         "if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:",
         "random fallback: eligibility of a randomly drawn cell",
         "VIOLATION, same shape as line 89"),
        ("strategies/limeqo.py", 35,
         "while min_observed.sum() > dataset.opt_time + 20:",
         "STOPPING RULE",
         "VIOLATION (privileged and global): opt_time is the per-row minimum of the FULL hidden "
         "matrix, i.e. the oracle optimum.  It may be used for post-hoc evaluation but not as a "
         "deployable stop condition"),
        ("strategies/limeqo.py", 105,
         "if min_observed.sum() <= dataset.opt_time + 50:",
         "STOPPING RULE inside the random fallback",
         "VIOLATION, same as line 35"),
        ("strategies/limeqo_plus.py", 114, "if dataset.matrix[select, hint] >= min_observed[select]:",
         "eligibility", "VIOLATION (same shape as limeqo.py:89)"),
        ("strategies/limeqo_plus.py", 117,
         "if dataset.matrix[select, hint] >= timeout_tolerance:",
         "completion test", "OK (same reasoning as limeqo.py:92)"),
        ("strategies/limeqo_plus.py", 139,
         "if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:",
         "random fallback eligibility", "VIOLATION (same shape as limeqo.py:114)"),
        ("strategies/greedy.py", 74, "if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:",
         "eligibility of a randomly drawn cell", "VIOLATION (same shape)"),
        ("strategies/random.py", 60, "if dataset.matrix[file_i, hint_i] >= min_observed[file_i]:",
         "eligibility of a randomly drawn cell", "VIOLATION (same shape)"),
        ("strategies/qo_advisor.py", 68,
         "if dataset.matrix[select, hint] >= min_observed[select]:",
         "eligibility", "VIOLATION (same shape)"),
    ]
    print(f"  {'file':<28} {'line':>4}  classification")
    for f, ln, code, role, verdict in sites:
        tag = verdict.split(":")[0]
        print(f"  {f:<28} {ln:>4}  {tag:<10} {role}")
    print()
    print("  VIOLATIONS (eligibility decided by an unrevealed comparison):")
    for f, ln, code, role, verdict in sites:
        if verdict.startswith("VIOLATION") and "same" not in verdict and "privileged" not in verdict:
            print(f"    {f}:{ln}")
    print("  VIOLATIONS (same shape, other files):")
    for f, ln, code, role, verdict in sites:
        if verdict.startswith("VIOLATION") and "same" in verdict:
            print(f"    {f}:{ln}")
    print("  PRIVILEGED STOPPING RULES:")
    for f, ln, code, role, verdict in sites:
        if "privileged" in verdict:
            print(f"    {f}:{ln}   {code}")
    print("  LEGITIMATE ENVIRONMENT READS (completion test at the cap actually used):")
    for f, ln, code, role, verdict in sites:
        if verdict.startswith("OK"):
            print(f"    {f}:{ln}")
    print()
    print("  The previous round reported '9 unmasked reads' without this distinction.  That count")
    print("  is withdrawn; the split above is what matters.")
    return [{"file": f, "line": ln, "code": code, "role": role, "verdict": verdict}
            for f, ln, code, role, verdict in sites]


def main() -> int:
    rows = check_official()
    retry = trace_reexecution()
    sites = classify_reads()
    (HERE / "q01r_mechanism.json").write_text(json.dumps(
        {"official_branch_table": rows, "retry_trace": retry, "read_sites": sites}, indent=2),
        encoding="utf-8")
    print(f"\n  wrote {HERE / 'q01r_mechanism.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
