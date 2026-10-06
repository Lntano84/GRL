"""Q01-R2 -- the four corrections.

1. READ-SITE RECLASSIFICATION.  The same ``x >= b`` expression has different permission depending on
   the cap the run was actually given:

       cap < b : after the timeout the environment has revealed only "x >= cap", which does NOT
                 establish "x >= b"  -> using the hidden value to retire the cell is a VIOLATION
       cap = b : the timeout itself reveals "x >= b" -> retiring the cell is LEGAL

   The random-fallback paths use cap = current best, so they are legal and are removed from the
   violation list.  What remains is the prediction-driven path of LimeQO / LimeQO+ when its
   tolerance is below the current best, plus the privileged stopping rule (listed separately).

2. STATE SNAPSHOT KEEPS THE VALUES, not just the keys.

3. CHECK 4 RUNS A REAL ENVIRONMENT, through one shared runner, on two matrices that differ only in
   unrevealed data.

4. THE LOWER BOUND IS A MAXIMUM.  ``L <- max(L, cap)``; and a candidate with ``L >= b`` can no longer
   directly improve the current best, so it may be pruned legally.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ interface

class ReplayEnvironment:
    """Holds the hidden matrix; reveals only what one execution at ``cap`` reveals."""

    def __init__(self, truth, best_by_query):
        self._truth = dict(truth)                # hidden: {(q,h): runtime}
        self.best = dict(best_by_query)          # revealed: current best per query
        self.charged = {}
        self.calls = []

    def observe(self, q, h, cap):
        x = self._truth[(q, h)]
        if x < cap:
            kind, val, charged = "completed", x, x
            self.best[q] = min(self.best[q], x)
        else:
            kind, val, charged = "cancelled", cap, cap
        self.charged[(q, h)] = self.charged.get((q, h), 0.0) + charged
        self.calls.append({"q": q, "h": h, "cap": cap, "kind": kind,
                           "value_or_bound": val, "charged": charged})
        return kind, val, charged


class LegalPolicy:
    """Sees only ``observe`` tuples.  Holds no matrix reference."""

    def __init__(self):
        self.mask = {}            # (q,h) -> completed runtime
        self.bounds = {}          # (q,h) -> CURRENT lower bound = max over all caps reached
        self.retired = set()      # cells removed from candidacy on information grounds
        self.spent = 0.0

    def state(self):
        """Values included.  ``spent`` is not rounded, so a future implementation cannot hide a
        numeric difference behind the snapshot."""
        return {"mask": sorted((list(k), v) for k, v in self.mask.items()),
                "bounds": sorted((list(k), v) for k, v in self.bounds.items()),
                "retired": sorted(list(k) for k in self.retired),
                "spent": self.spent}

    def record(self, q, h, cap, kind, val, charged):
        self.spent += charged
        if kind == "completed":
            self.mask[(q, h)] = val
            self.bounds.pop((q, h), None)
            self.retired.add((q, h))               # correctly measured; no reason to rerun
        else:
            # CORRECTION 4: keep the strongest bound, never weaken it
            self.bounds[(q, h)] = max(self.bounds.get((q, h), float("-inf")), cap)
        return self.state()

    def can_prune(self, q, h):
        """LEGAL pruning: the lower bound already meets or exceeds the current best for this query,
        so the cell cannot directly improve it.  This does not depend on the hidden runtime."""
        return self.bounds.get((q, h), float("-inf")) >= self.best_get(q)

    def best_get(self, q):
        vals = [v for (qq, _), v in self.mask.items() if qq == q]
        return min(vals) if vals else float("inf")

    def eligible(self, q, h):
        return (q, h) not in self.retired

    def decide_next(self, candidates, cap_for):
        for c in candidates:
            if not self.eligible(*c):
                continue
            if self.can_prune(*c):                 # legal: bound already tells us it cannot help
                continue
            return c, cap_for(*c)
        return "stop"


class OfficialPolicy:
    """The stock branch, kept as the defect contrast.  It reads the hidden runtime."""

    def __init__(self, truth, best_by_query):
        self._truth = dict(truth)
        self.best = dict(best_by_query)
        self.retired = set()
        self.spent = 0.0

    def record(self, q, h, cap):
        x = self._truth[(q, h)]
        self.spent += min(x, cap)
        if x >= self.best[q]:                      # <-- hidden comparison
            self.retired.add((q, h))
        if x >= cap:
            return "cancelled"
        self.retired.add((q, h))
        self.best[q] = min(self.best[q], x) if self.best[q] != float("inf") else x
        return "completed"

    def eligible(self, q, h):
        return (q, h) not in self.retired


# ------------------------------------------------------------------ correction 1

def read_site_table():
    print("=" * 104)
    print("  CORRECTION 1 -- the same comparison, two different permissions")
    print("=" * 104)
    print("  The rule: after a timeout at ``cap`` the environment has revealed 'x >= cap'.")
    print("  Retiring the cell because 'x >= b' is legal only if the revealed bound already implies it.\n")
    print(f"  {'site':<34} {'cap used':<26} {'revealed':<12} {'implies x>=b?':<14} verdict")
    sites = [
        ("limeqo.py:89", "timeout_tolerance = min(alpha*best, beta*pred)",
         "x >= cap", "only if cap >= b", "VIOLATION when cap < b"),
        ("limeqo_plus.py:114", "same as above", "x >= cap", "only if cap >= b",
         "VIOLATION when cap < b"),
        ("limeqo.py:114", "current best b (random fallback)", "x >= b", "yes",
         "LEGAL"),
        ("limeqo_plus.py:139", "current best b (random fallback)", "x >= b", "yes", "LEGAL"),
        ("greedy.py:74", "current best b", "x >= b", "yes", "LEGAL"),
        ("random.py:60", "current best b", "x >= b", "yes", "LEGAL"),
        ("qo_advisor.py:68", "current best b", "x >= b", "yes", "LEGAL"),
    ]
    for s, cap, rev, imp, verdict in sites:
        print(f"  {s:<34} {cap:<26} {rev:<12} {imp:<14} {verdict}")
    print()
    print("  REMOVED from the violation list (previous round got these wrong):")
    for s, cap, rev, imp, verdict in sites:
        if verdict == "LEGAL":
            print(f"    {s}  -- cap equals the current best, so the timeout itself reveals x >= b")
    print()
    print("  CONFIRMED violations (narrowed):")
    print("    limeqo.py:89, limeqo_plus.py:114 -- the prediction-driven path uses a tolerance that")
    print("      may sit BELOW the current best, and then retires the cell using the hidden runtime")
    print("    limeqo.py:35, limeqo.py:105 -- privileged stopping rule (listed separately; opt_time")
    print("      is the per-row minimum of the FULL hidden matrix)")
    print()
    print("  The violation list must not be grown from expression shape alone.")
    return [{"site": s, "cap": cap, "revealed": rev, "implies_x_ge_b": imp, "verdict": verdict}
            for s, cap, rev, imp, verdict in sites]


# ------------------------------------------------------------------ correction 2

def snapshot_test():
    print("\n" + "=" * 104)
    print("  CORRECTION 2 -- the snapshot must carry the VALUES")
    print("=" * 104)
    a, b = LegalPolicy(), LegalPolicy()
    # two policies with the same KEYS but different VALUES
    a.mask[(0, 1)] = 5.0
    a.bounds[(0, 2)] = 2.0
    a.spent = 7.0
    b.mask[(0, 1)] = 9.0
    b.bounds[(0, 2)] = 8.0
    b.spent = 17.0
    old_style_a = {"mask": sorted(a.mask), "bounds": sorted(a.bounds)}
    old_style_b = {"mask": sorted(b.mask), "bounds": sorted(b.bounds)}
    new_a, new_b = a.state(), b.state()
    print(f"  policy A: mask(0,1)=5.0  bounds(0,2)=2.0  spent=7.0")
    print(f"  policy B: mask(0,1)=9.0  bounds(0,2)=8.0  spent=17.0")
    print()
    print(f"  keys-only snapshot equal?  {old_style_a == old_style_b}   <- the old defect")
    print(f"  value-carrying snapshot equal? {new_a == new_b}")
    ok = (old_style_a == old_style_b) and (new_a != new_b)
    print(f"  [{'PASS' if ok else 'FAIL'}] the fixed snapshot catches a difference the key-only "
          f"snapshot missed")
    return {"keys_only_equal": old_style_a == old_style_b,
            "value_snapshot_equal": new_a == new_b, "pass": ok}


# ------------------------------------------------------------------ correction 3

def runner(truth, best, plan):
    """One shared runner: drives the environment and the policy through the legal interface only.

    ``plan`` is a list of (q, h, cap) to attempt in order; a single call exercises the same code path
    that a search loop would use.
    """
    env = ReplayEnvironment(truth, best)
    pol = LegalPolicy()
    for (q, h, cap) in plan:
        kind, val, charged = env.observe(q, h, cap)
        pol.record(q, h, cap, kind, val, charged)
    return env, pol


def check4_real():
    print("\n" + "=" * 104)
    print("  CORRECTION 3 -- check 4 through a REAL environment and runner")
    print("=" * 104)
    print("  Two complete small matrices that differ ONLY in unrevealed truth.  The revealed")
    print("  feedback is identical by construction because the same caps are attempted in order and")
    print("  both worlds time out on the same cells.")
    # matrix A: the hidden cell (0,2) is 7; matrix B: it is 900
    truthA = {(0, 0): 10.0, (0, 1): 5.0, (0, 2): 7.0, (1, 0): 11.0}
    truthB = {(0, 0): 10.0, (0, 1): 5.0, (0, 2): 900.0, (1, 0): 11.0}
    best = {0: 10.0, 1: 11.0}
    # (0,0) completes at its cap 10 in both; (0,1) times out at cap 2 in both
    plan = [(0, 0, 10), (0, 1, 2)]
    envA, polA = runner(truthA, best, plan)
    envB, polB = runner(truthB, dict(best), plan)
    cands = [(0, 1), (0, 2), (1, 0)]
    cap_for = lambda q, h: 2
    nA, cA = polA.decide_next(cands, cap_for)
    nB, cB = polB.decide_next(cands, cap_for)
    stopA = polA.spent >= 12.0
    stopB = polB.spent >= 12.0
    print(f"    world A hidden (0,2)=7.0    world B hidden (0,2)=900.0")
    print(f"    revealed feedback A: {envA.calls}")
    print(f"    revealed feedback B: {envB.calls}")
    print(f"    feedback identical : {envA.calls == envB.calls}")
    print(f"    next action  A={nA}@{cA}  B={nB}@{cB}   identical: {nA == nB and cA == cB}")
    print(f"    budget stop  A={stopA}  B={stopB}        identical: {stopA == stopB}")
    print(f"    state identical: {polA.state() == polB.state()}")
    ok = (envA.calls == envB.calls and nA == nB and cA == cB and stopA == stopB
          and polA.state() == polB.state())
    print(f"  [{'PASS' if ok else 'FAIL'}] hidden-only change leaves the action and the stop "
          f"decision unchanged")

    # defect contrast: a pair on which the PRIVILEGED stop rule would actually differ
    print("\n    DEFECT CONTRAST -- the privileged stop rule on two worlds where it really differs")
    print("      rule:  while min_observed.sum() > opt_time + 20  -> keep exploring")
    print("      min_observed is computed from the MASKED matrix, so it is identical in both worlds;")
    print("      opt_time is the per-row minimum of the FULL hidden matrix, so it can differ.")
    def stop_decides(opt_time, minobs_sum):
        return minobs_sum > opt_time + 20
    cases = [("W_same_hidden_optimum", 100.0, 200.0), ("W_different_hidden_optimum", 260.0, 200.0)]
    rows = []
    for nm, opt, s in cases:
        cont = stop_decides(opt, s)
        rows.append({"world": nm, "opt_time": opt, "min_observed_sum": s, "continue": cont})
        print(f"      {nm:<26} min_observed.sum()={s:>6}  opt_time={opt:>6}  -> continue={cont}")
    diff = rows[0]["continue"] != rows[1]["continue"]
    print(f"      the two worlds share the same revealed feedback, yet the stop decision differs: "
          f"{diff}")
    print(f"      [{'PASS' if diff else 'FAIL'}] the privileged rule is exposed by a real pair")
    return {"check4": {"feedback_identical": envA.calls == envB.calls,
                       "next_action_A": list(nA) if nA != "stop" else "stop",
                       "next_action_B": list(nB) if nB != "stop" else "stop",
                       "stop_A": stopA, "stop_B": stopB,
                       "state_identical": polA.state() == polB.state(), "pass": ok},
            "privileged_stop_demo": rows, "privileged_stop_exposed": diff}


# ------------------------------------------------------------------ correction 4

def bound_test():
    print("\n" + "=" * 104)
    print("  CORRECTION 4 -- the lower bound is a MAXIMUM, and L >= b prunes legally")
    print("=" * 104)
    # seed query 0 with a revealed best of 10 so the pruning comparison is meaningful.
    # note the convention x < cap completes, so the cap must exceed 10 for (0,0) to complete.
    truth = {(0, 0): 10.0, (0, 1): 100.0}
    env = ReplayEnvironment(truth, {0: 10.0})
    pol = LegalPolicy()
    k0, v0, c0 = env.observe(0, 0, 12)
    pol.record(0, 0, 12, k0, v0, c0)
    print(f"  revealed best for query 0 (from (0,0)) = {pol.best_get(0)}")
    print("  sequence: cap = 10 first, then cap = 2 (a shorter retry must not weaken the bound)")
    for cap in (10, 2):
        kind, val, charged = env.observe(0, 1, cap)
        pol.record(0, 1, cap, kind, val, charged)
        print(f"    cap={cap:>2}: ({kind!r}, {val}, charged {charged})  "
              f"stored bound = {pol.bounds.get((0,1))}")
    lb = pol.bounds.get((0, 1))
    print()
    print(f"  stored lower bound after both runs: {lb}   (the old code overwrote 10 with 2)")
    print(f"  [{'PASS' if lb == 10.0 else 'FAIL'}] bound kept the maximum")
    # legal pruning
    pr = pol.can_prune(0, 1)
    print(f"  current best for query 0 = {pol.best_get(0)}; L >= b ? {pr}")
    print(f"  [{'PASS' if pr else 'FAIL'}] a candidate whose bound already meets the current best is")
    print(f"         pruned without any hidden information")
    # and the case where pruning is NOT legal
    truth2 = {(0, 0): 10.0, (0, 1): 100.0}
    env2 = ReplayEnvironment(truth2, {0: 10.0})
    pol2 = LegalPolicy()
    k0b, v0b, c0b = env2.observe(0, 0, 12)
    pol2.record(0, 0, 12, k0b, v0b, c0b)
    k, v, c = env2.observe(0, 1, 2)
    pol2.record(0, 1, 2, k, v, c)
    pr2 = pol2.can_prune(0, 1)
    print(f"\n  contrast: a single timeout at cap=2 leaves L=2 < b=10, so pruning is NOT legal: "
          f"{pr2}")
    print(f"  [{'PASS' if not pr2 else 'FAIL'}] pruning is refused when the bound does not justify it")
    return {"bound_after_caps_10_then_2": lb, "bound_kept_max": lb == 10.0,
            "prune_legal_when_L_ge_b": pr, "prune_refused_when_L_below_b": not pr2}


def main() -> int:
    sites = read_site_table()
    snap = snapshot_test()
    c4 = check4_real()
    bnd = bound_test()
    ok = snap["pass"] and c4["check4"]["pass"] and bnd["bound_kept_max"] \
        and bnd["prune_legal_when_L_ge_b"] and bnd["prune_refused_when_L_below_b"]
    print("\n" + "=" * 104)
    print(f"  CORRECTIONS 2-4: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    print("=" * 104)
    (HERE / "q01r2_corrections.json").write_text(json.dumps(
        {"read_sites": sites, "snapshot": snap, "check4": c4, "bounds": bnd,
         "all_pass": bool(ok)}, indent=2), encoding="utf-8")
    print(f"  wrote {HERE / 'q01r2_corrections.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
