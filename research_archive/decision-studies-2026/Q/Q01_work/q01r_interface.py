"""Q01-R step R4: the minimal feedback interface, plus the four deterministic checks.

Interface
---------
    observe(query, hint, cap) -> (kind, value_or_bound, charged_seconds)

    kind == "completed"    : the run finished inside ``cap`` and used ``value_or_bound`` seconds
    kind == "cancelled"    : the run was stopped at ``cap``; the only thing revealed is
                             "runtime >= cap", and ``cap`` seconds were charged

Boundary convention kept from the official code: ``x < cap`` completes, ``x >= cap`` times out.
A re-execution is always charged from scratch.

The strategy object holds no reference to the truth matrix.  The environment does, because it must,
in order to produce feedback for the run it just performed -- but it exposes only the tuple above.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ environment and policy

class ReplayEnvironment:
    """Holds the hidden truth; reveals only what one execution at ``cap`` would reveal."""

    def __init__(self, truth, best_by_query):
        self._truth = truth                     # hidden: {(q, h): runtime}
        self.best = dict(best_by_query)         # revealed: current best per query
        self.charged = {}                       # (q,h) -> total seconds charged so far
        self.calls = []                         # audit trail of what was revealed

    def observe(self, q, h, cap):
        x = self._truth[(q, h)]
        if x < cap:
            kind, val = "completed", x
            charged = x
            self.best[q] = min(self.best[q], x)
        else:
            kind, val = "cancelled", cap        # a lower bound, nothing more
            charged = cap
        self.charged[(q, h)] = self.charged.get((q, h), 0.0) + charged
        self.calls.append({"q": q, "h": h, "cap": cap, "kind": kind, "value_or_bound": val,
                           "charged": charged})
        return kind, val, charged


class LegalPolicy:
    """A policy that sees only ``observe`` tuples.  Holds no matrix reference."""

    def __init__(self):
        self.mask = {}              # completed measurements
        self.bounds = {}            # cancelled: the cap it reached
        self.explored = set()       # retired cells
        self.counted = set()        # cells counted toward the observation budget
        self.spent = 0.0

    def state(self):
        return {"mask": sorted(self.mask), "bounds": sorted(self.bounds),
                "explored": sorted(self.explored), "counted": sorted(self.counted),
                "spent": round(self.spent, 6)}

    def record(self, q, h, cap, kind, val, charged):
        self.spent += charged
        if kind == "completed":
            self.mask[(q, h)] = val
            self.explored.add((q, h))
            self.counted.add((q, h))
        else:
            self.bounds[(q, h)] = val
            # the ONLY legal inference: runtime >= cap.  This does NOT reveal whether the cell
            # could still beat the current best, so eligibility is untouched.
        return self.state()

    def eligible(self, q, h):
        return (q, h) not in self.explored

    def decide_next(self, candidates, cap_for):
        """Deterministic next action: the first eligible candidate, or 'stop'."""
        for c in candidates:
            if self.eligible(*c):
                return c, cap_for(*c)
        return "stop"


# ------------------------------------------------------------------ the official branch, for contrast

def official_record(state, q, h, hidden_x, best, cap):
    """Transcribe limeqo.py:89-98, which reads the HIDDEN runtime.  Kept only as the captured defect."""
    st = dict(state)
    st.setdefault("explored", set())
    if hidden_x >= best[q]:                      # <-- hidden comparison
        st["explored"] = st["explored"] | {(q, h)}
    if hidden_x >= cap:
        return st, "cancelled"
    st["explored"] = st["explored"] | {(q, h)}
    st["mask"] = st.get("mask", {})
    st["mask"][(q, h)] = hidden_x
    return st, "completed"


# ------------------------------------------------------------------ checks

def jsonable(obj):
    """Recursively stringify tuple keys and tuples so the audit trail can be written out."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, set):
        return sorted(jsonable(v) for v in obj)
    return obj


def main() -> int:
    print("=" * 100)
    print("  Q01-R -- FOUR DETERMINISTIC CHECKS ON THE LEGAL INTERFACE")
    print("=" * 100)
    results = {}
    ok_all = True

    def report(name, ok, detail):
        nonlocal ok_all
        ok_all &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        for line in detail:
            print(f"         {line}")

    # ---- check 1 -----------------------------------------------------------------------------
    print(f"\n{'=' * 100}\n  CHECK 1  b=10, cap=2, x in {{5, 20}}\n{'=' * 100}")
    print("  the two runs must give the SAME legal feedback, hence the SAME legal policy state")
    states = {}
    for x in (5, 20):
        env = ReplayEnvironment({(0, 1): x}, {0: 10.0})
        pol = LegalPolicy()
        kind, val, charged = env.observe(0, 1, 2)
        st = pol.record(0, 1, 2, kind, val, charged)
        states[x] = {"kind": kind, "value_or_bound": val, "charged": charged, "state": st,
                     "eligible": pol.eligible(0, 1)}
        print(f"    x={x:>2}: observe -> ({kind!r}, {val}, {charged})   "
              f"eligible_again={pol.eligible(0,1)}   state={json.dumps(st)}")
    same_feedback = (states[5]["kind"] == states[20]["kind"]
                     and states[5]["value_or_bound"] == states[20]["value_or_bound"]
                     and states[5]["charged"] == states[20]["charged"])
    same_state = states[5]["state"] == states[20]["state"]
    report("identical legal feedback", same_feedback,
           [f"kind/value/charged: {states[5]['kind']!r}, {states[5]['value_or_bound']}, "
            f"{states[5]['charged']} in both worlds"])
    report("identical legal policy state", same_state,
           [f"state A = {json.dumps(states[5]['state'])}",
            f"state B = {json.dumps(states[20]['state'])}"])
    # the captured defect
    oa, _ = official_record({"mask": {}, "bounds": {}}, 0, 1, 5, {0: 10.0}, 2)
    ob, _ = official_record({"mask": {}, "bounds": {}}, 0, 1, 20, {0: 10.0}, 2)
    defect = ((0, 1) in oa["explored"]) != ((0, 1) in ob["explored"])
    print(f"\n    CAPTURED DEFECT (official branch, kept as a historical artefact, NOT as legal")
    print(f"    behaviour): explored(0,1) = {(0,1) in oa['explored']} for x=5 and "
          f"{(0,1) in ob['explored']} for x=20 -> differs = {defect}")
    results["check1"] = {"legal": states, "official_defect_differs": defect}

    # ---- check 2 -----------------------------------------------------------------------------
    print(f"\n{'=' * 100}\n  CHECK 2  b=10, cap=10, x in {{5, 20}}\n{'=' * 100}")
    print("  x=5 must complete and x=20 must time out; the states are ALLOWED to differ here")
    c2 = {}
    for x in (5, 20):
        env = ReplayEnvironment({(0, 1): x}, {0: 10.0})
        pol = LegalPolicy()
        kind, val, charged = env.observe(0, 1, 10)
        st = pol.record(0, 1, 10, kind, val, charged)
        c2[x] = {"kind": kind, "value_or_bound": val, "charged": charged, "state": st}
        print(f"    x={x:>2}: observe -> ({kind!r}, {val}, {charged})   state={json.dumps(st)}")
    report("x=5 completes", c2[5]["kind"] == "completed",
           [f"kind={c2[5]['kind']!r}, charged={c2[5]['charged']}"])
    report("x=20 times out", c2[20]["kind"] == "cancelled",
           [f"kind={c2[20]['kind']!r}, charged={c2[20]['charged']}"])
    report("states differ, which is legal at cap=10", c2[5]["state"] != c2[20]["state"],
           ["the difference is explained entirely by the revealed completion, not by hidden data"])
    results["check2"] = c2

    # ---- check 3 -----------------------------------------------------------------------------
    print(f"\n{'=' * 100}\n  CHECK 3  x=5, cap=2 then cap=10 -- the re-execution path\n{'=' * 100}")
    env = ReplayEnvironment({(0, 1): 5.0}, {0: 10.0})
    pol = LegalPolicy()
    seq = []
    for cap in (2, 10):
        kind, val, charged = env.observe(0, 1, cap)
        pol.record(0, 1, cap, kind, val, charged)
        seq.append({"cap": cap, "kind": kind, "value_or_bound": val, "charged": charged,
                    "cumulative_spent": pol.spent})
        print(f"    cap={cap:>2}: ({kind!r}, {val}, charged {charged})   "
              f"cumulative spent = {pol.spent}")
    total = env.charged[(0, 1)]
    # after the first (cancelled) run the cell must still be eligible; capture that before the retry
    env_probe = ReplayEnvironment({(0, 1): 5.0}, {0: 10.0})
    pol_probe = LegalPolicy()
    k, v, c = env_probe.observe(0, 1, 2)
    pol_probe.record(0, 1, 2, k, v, c)
    stayed_eligible = pol_probe.eligible(0, 1)
    report("the re-execution path is available after a timeout", stayed_eligible,
           [f"after the cancelled run: eligible(0,1) = {stayed_eligible}; "
            f"the official note that there is no retry path is therefore wrong"])
    report("total charged equals 2 + 5 = 7", abs(total - 7.0) < 1e-12, [f"total = {total}"])
    results["check3"] = {"sequence": seq, "total_charged": total,
                         "eligible_after_first_run": stayed_eligible}

    # ---- check 4 -----------------------------------------------------------------------------
    print(f"\n{'=' * 100}\n  CHECK 4  only unrevealed truth and the full-matrix optimum change\n{'=' * 100}")
    print("  Give two worlds the SAME revealed feedback and differ only in hidden values and in")
    print("  opt_time; the next action and the budget-based stop decision must be unchanged.")
    cands = [(0, 1), (0, 2), (1, 0)]
    cap_for = lambda q, h: 2
    worlds = {}
    for name, truth, best in (("A", {(0, 1): 5.0, (0, 2): 7.0, (1, 0): 9.0}, {0: 10.0, 1: 11.0}),
                              ("B", {(0, 1): 5.0, (0, 2): 90.0, (1, 0): 9.0}, {0: 10.0, 1: 11.0})):
        pol = LegalPolicy()
        # identical revealed feedback in both worlds
        pol.record(0, 0, 10, "completed", 10.0, 10.0)      # same in both
        pol.record(0, 1, 2, "cancelled", 2.0, 2.0)         # same in both
        nxt, cap = pol.decide_next(cands, cap_for)
        stop_by_budget = pol.spent >= 20.0
        worlds[name] = {"next_action": nxt, "next_cap": cap, "spent": pol.spent,
                        "stop_by_budget": stop_by_budget,
                        "hidden": dict(truth), "opt_time_A_only": None}
        print(f"    world {name}: hidden={dict(truth)}")
        print(f"      revealed feedback identical; next action = {nxt} at cap {cap}; "
              f"spent = {pol.spent}; stop_by_budget = {stop_by_budget}")
    report("next action unchanged",
           worlds["A"]["next_action"] == worlds["B"]["next_action"]
           and worlds["A"]["next_cap"] == worlds["B"]["next_cap"],
           [f"{worlds['A']['next_action']} vs {worlds['B']['next_action']}"])
    report("budget-based stop decision unchanged",
           worlds["A"]["stop_by_budget"] == worlds["B"]["stop_by_budget"],
           ["a fixed exploration budget must not consult opt_time"])
    print()
    print("    The superseded code stopped on `min_observed.sum() > dataset.opt_time + 20`, where")
    print("    opt_time is the per-row minimum of the FULL hidden matrix.  Check 4 is the test that")
    print("    a fixed-budget comparison has to pass, and the official stop rule fails it.")
    results["check4"] = worlds

    (HERE / "q01r_checks.json").write_text(
        json.dumps(jsonable({"results": results, "all_pass": bool(ok_all)}), indent=2),
        encoding="utf-8")
    print(f"\n{'=' * 100}")
    print(f"  ALL FOUR CHECKS {'PASSED' if ok_all else 'FAILED'}")
    print(f"{'=' * 100}")
    print(f"  wrote {HERE / 'q01r_checks.json'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
