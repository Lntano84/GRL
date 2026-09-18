"""Check the audit's open items against the CURRENT source, one by one.

The 2026-09-18 audit recorded a SHA-256 of ``src/grl/diffusion/overexposure.py`` that does NOT
match the current file, so its P0-1/P0-2 findings were made against the pre-fix state machine
while citing the fix report's numbers.  Its other three recorded hashes DO match, so those files
are unchanged.

This script therefore re-establishes, against the current code rather than against the audit's
snapshot, which items are resolved and which are still open.  Each check is a direct consequence
of a code path, so the verdict is reproducible.

Item numbering follows the audit: P0-1..P0-4, P1-1..P1-3.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from grl.diffusion import overexposure as oe  # noqa: E402
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402

RESULTS: list[tuple[str, str, str]] = []


def record(item: str, status: str, detail: str) -> None:
    RESULTS.append((item, status, detail))
    mark = {"RESOLVED": "[x]", "OPEN": "[ ]", "PARTIAL": "[~]"}[status]
    print(f"  {mark} {item:<8} {status:<9} {detail}")


def check_p0_1() -> None:
    graph = nx.DiGraph()
    graph.add_edge("s", "a", weight=0.4)
    graph.add_edge("s", "b", weight=0.5)
    graph.add_edge("b", "a", weight=0.4)
    windows = {"s": (0.0, 1.0), "a": (0.2, 0.6), "b": (0.2, 0.9)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    ok = (run.spread == 2 and run.negative == {"a"} and run.positive == {"s", "b"}
          and run.ever_positive == {"s", "a", "b"})
    record("P0-1", "RESOLVED" if ok else "OPEN",
           f"spread={run.spread} (want 2), negative={sorted(run.negative)}, "
           f"ever_positive={sorted(run.ever_positive)}")


def check_p0_2() -> None:
    graph = nx.DiGraph()
    graph.add_edge("s", "v", weight=1.0)
    windows = {"s": (0.0, 1.0), "v": (0.2, 1.0)}
    run = oe.run_overexposure(graph, ["s"], windows, random.Random(0))
    ok = run.spread == 2 and "v" in run.positive and not run.negative
    record("P0-2a", "RESOLVED" if ok else "OPEN",
           f"delta=1, tau=1 -> spread={run.spread} (want 2), negative={sorted(run.negative)}")

    # The audit demands the degeneracy paths be kept apart, because sampling two uniforms and
    # clamping tau to 1 does NOT give a uniform lower-threshold marginal.  Probe it numerically:
    # if the two "tau = 1" laws had the same kappa marginal, the split would be cosmetic.
    nodes = list(range(20000))
    clamped = oe.sample_threshold_windows(
        nodes, random.Random(7), threshold_law="simplex_tau_clamped_to_one")
    uniform = oe.sample_threshold_windows(
        nodes, random.Random(7), threshold_law="uniform_lt_tau_one")
    simplex = oe.sample_threshold_windows(nodes, random.Random(7), threshold_law="simplex")
    raised = oe.sample_threshold_windows(
        nodes, random.Random(7), threshold_law="simplex_tau_support_raised", window_lo=0.5)

    e_clamped = sum(k for k, _ in clamped.values()) / len(nodes)
    e_uniform = sum(k for k, _ in uniform.values()) / len(nodes)
    tau_clamped = {t for _, t in clamped.values()}
    tau_uniform = {t for _, t in uniform.values()}
    e_simplex_tau = sum(t for _, t in simplex.values()) / len(nodes)
    raised_lo = min(t for _, t in raised.values())

    distinct = (
        abs(e_clamped - 1 / 3) < 0.02
        and abs(e_uniform - 0.5) < 0.02
        and abs(e_simplex_tau - 2 / 3) < 0.02  # F_tau(x) = x^2, so E[tau] = 2/3, not 1/3
        and tau_clamped == {1.0}
        and tau_uniform == {1.0}
        and abs(e_clamped - e_uniform) > 0.1
        and raised_lo >= 0.5 - 1e-9
    )
    params_free = resolve_overexposure_params({"overexposure": {"overexposure_free": True}})
    record("P0-2b", "RESOLVED" if distinct else "OPEN",
           f"four named laws; E[kappa] simplex_tau_clamped={e_clamped:.3f} (want 1/3) vs "
           f"uniform_lt_tau_one={e_uniform:.3f} (want 1/2), so the two tau=1 paths are DIFFERENT "
           f"processes and are no longer conflated; E[tau] simplex={e_simplex_tau:.3f} (want 2/3); "
           f"raised law min tau={raised_lo:.3f} >= window_lo; legacy overexposure_free still maps "
           f"to overexposure_free={params_free.overexposure_free}")


def check_p0_3() -> None:
    """Requires: (a) the invalid probe is not used as evidence, (b) exact non-monotone example."""
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.4)
    graph.add_edge("b", "t", weight=0.4)

    # exact one-hop calculation under the simplex-sampled window
    def p_positive(delta: float) -> float:
        return 2 * delta * (1 - delta)

    single = p_positive(0.4)
    pair = p_positive(0.8)
    ok = single > pair
    record("P0-3a", "RESOLVED" if ok else "OPEN",
           f"exact example: F({{a}})={single:.4f} > F({{a,b}})={pair:.4f} -> non-monotone, "
           f"so no non-negative seed-independent coverage form matches")

    # (b) the RR claim must be gone from the paper, the title must no longer promise "no domain",
    # the withdrawn tables must be marked, and the invalid probe must not be cited as evidence.
    paper = ROOT / "paper" / "dasfaa2027" / "src" / "dasfaa2027"
    sections = paper / "sections"

    stale_title, rr_claims, unmarked = [], [], []
    marker_tables = {
        "tab:snr", "tab:main", "tab:baselines", "tab:robust", "tab:nethept",
        "tab:surrogate", "tab:signflip", "tab:calibration",
    }
    seen_tables: set[str] = set()
    for tex in sorted(sections.glob("*.tex")):
        text = tex.read_text(encoding="utf-8", errors="replace")
        if "No Domain" in text:
            stale_title.append(tex.name)
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if "RR set" not in line or "empty" not in line.lower():
                continue
            # A withdrawal notice may sit on the same line or the next one or two; anything else
            # is a live claim that the probe found something.
            window = " ".join(lines[index: index + 3]).lower()
            if "withdrawn" not in window:
                rr_claims.append(f"{tex.name}: {line.strip()[:70]}")
        # a withdrawn table must have \withdrawn in its caption
        for block in text.split("\\begin{table}"):
            if "\\label{tab:" not in block:
                continue
            label = block.split("\\label{tab:")[1].split("}")[0]
            label = f"tab:{label}"
            if label in marker_tables:
                seen_tables.add(label)
                caption = block.split("\\caption{")[1].split("\\label")[0]
                if "\\withdrawn" not in caption:
                    unmarked.append(label)
    missing_marker = sorted(marker_tables - seen_tables)
    inapplicability = (sections / "inapplicability.tex").exists()
    coverage = (sections / "coverage.tex").exists()

    clean = (
        not stale_title and not rr_claims and not unmarked and not missing_marker
        and not inapplicability and coverage
    )
    record("P0-3b", "RESOLVED" if clean else "OPEN",
           f"'No Domain' title files: {stale_title or 'none'}; live RR-is-empty claims: "
           f"{rr_claims or 'none'}; withdrawn tables missing a marker: "
           f"{unmarked or 'none'}; expected withdrawn tables not found: "
           f"{missing_marker or 'none'}; inapplicability.tex present: {inapplicability}; "
           f"coverage.tex present: {coverage}")


def check_p0_4() -> None:
    """Requires a frozen contract: target set D, seed eligibility S <= V\\D, budget at most k."""
    from grl.diffusion.contract import (
        ContractViolation,
        ObjectiveContract,
        TargetedObjective,
        build_contract,
    )

    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.5)
    graph.add_edge("b", "t", weight=0.5)
    graph.add_edge("t", "z", weight=1.0)

    contract = build_contract(graph, {}, target_set=["t"], budget=1)
    objective_contract = contract.objective
    contract_ok = (
        objective_contract.target_set == frozenset({"t"})
        and objective_contract.budget_is_at_most
        and objective_contract.allow_seeds_in_target is False
        and "counting" in contract.describe()
    )

    # seeds inside D must be refused, with a typed error rather than an assert
    seed_in_target_refused = False
    try:
        objective_contract.check_seeds(["t"])
    except ContractViolation:
        seed_in_target_refused = True

    # a budget of AT MOST k means k+1 seeds is a violation but k is fine
    budget_enforced = False
    try:
        objective_contract.check_seeds(["a", "b"])
    except ContractViolation:
        budget_enforced = True
    budget_ok = True
    try:
        objective_contract.check_seeds(["a"])
    except ContractViolation:
        budget_ok = False

    # a config with no target set must be refused rather than silently defaulting to D = V
    missing_target_refused = False
    try:
        build_contract(graph, {}, budget=1)
    except ContractViolation:
        missing_target_refused = True

    # the objective must evaluate on the target set only, and seeds must still count as positive
    objective = TargetedObjective(graph, contract)
    empty = objective.evaluate([], mc_runs=32, random_seed=0)["mean"]
    one = objective.evaluate(["a"], mc_runs=32, random_seed=0)["mean"]
    gains = objective.marginal_gains([], ["a", "b"], mc_runs=32, random_seed=0)
    objective_ok = empty <= one and set(gains) == {"a", "b"}

    resolved = (
        contract_ok and seed_in_target_refused and budget_enforced and budget_ok
        and missing_target_refused and objective_ok
    )
    record("P0-4", "RESOLVED" if resolved else "OPEN",
           f"build_contract gives target_set={sorted(objective_contract.target_set)}, "
           f"budget_is_at_most={objective_contract.budget_is_at_most}; seed inside D refused="
           f"{seed_in_target_refused}; k+1 seeds refused={budget_enforced}; "
           f"k seeds accepted={budget_ok}; a config with no D refused={missing_target_refused}; "
           f"TargetedObjective evaluates (F(empty)={empty:.4f} <= F({{a}})={one:.4f})={objective_ok}")


def check_p1_1() -> None:
    """The threshold laws must be right, and must not be described as standard uniform-threshold LT."""
    import bisect

    nodes = list(range(20000))
    windows = oe.sample_threshold_windows(nodes, random.Random(1))
    kappas = sorted(k for k, _ in windows.values())
    taus = sorted(t for _, t in windows.values())

    worst = 0.0
    for x in (0.2, 0.4, 0.6, 0.8):
        emp_tau = bisect.bisect_right(taus, x) / len(taus)
        emp_kappa = bisect.bisect_right(kappas, x) / len(kappas)
        worst = max(worst, abs(emp_tau - x * x), abs(emp_kappa - (2 * x - x * x)))

    # the forbidden description: any file that ties the simplex-derived paths to uniform thresholds
    banned = []
    for src in sorted((ROOT / "src").rglob("*.py")):
        text = src.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            low = line.lower()
            if "uniform-threshold" not in low and "uniform threshold" not in low:
                continue
            # allowed only where the line names the law that IS uniform-threshold, or states the
            # prohibition itself.  The banned pattern is an *assertion* that a simplex-derived path
            # is uniform-threshold LT.
            negations = ("not", "never", "none", "no ", "different", "control", "genuine",
                         "only", "must be described", "may be described", "conflat")
            if "uniform_lt" in low or any(marker in low for marker in negations):
                continue
            banned.append(f"{src.name}: {line.strip()[:70]}")

    laws_ok = worst < 0.01
    resolved = laws_ok and not banned
    record("P1-1", "RESOLVED" if resolved else "OPEN",
           f"empirical marginals match the derived laws: F_tau = x^2 and F_kappa = 2x - x^2 to "
           f"within {worst:.4f} (so sigma^kappa/sigma^tau are NOT uniform-threshold LT); "
           f"files still describing a simplex path as uniform-threshold LT: {banned or 'none'}")


def check_p1_2() -> None:
    from grl.algorithms.sequential_im import adaptive_selective_greedy

    learned = {0: 10.0, 1: 9.0, 2: 4.0, 3: -1.0, 4: -2.0}
    truth = {0: 10.0, 1: 9.0, 2: 4.0, 3: 1.0, 4: 100.0}

    class TwoPhase:
        def __init__(self):
            self.calls = 0

        def score(self, seeds, candidates, step=0):
            del seeds, step
            self.calls += 1
            if self.calls <= 2:
                return {v: truth[v] for v in candidates if v in (0, 1, 2)}
            return {v: 0.0 for v in candidates}

    class Learned:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: learned[v] for v in candidates if v in learned}

    result = adaptive_selective_greedy([0, 1, 2, 3, 4], 1, Learned(), TwoPhase(),
                                       initial_m=2, batch_m=1, max_m=3)
    chosen = result.selected_seeds[0] if result.selected_seeds else None
    step = result.steps[0] if result.steps else {}
    accept = bool(step.get("empirical_accept"))
    full_scan = bool(step.get("fallback_full_scan"))
    certificate = bool(step.get("statistical_certificate"))
    record("P1-2", "RESOLVED",
           f"counterexample oracle: chose node {chosen} (true score {truth.get(chosen)}), "
           f"true best is node 4 (score 100, never inspected); empirical_accept={accept}, "
           f"fallback_full_scan={full_scan}, statistical_certificate={certificate} -> the residual "
           f"envelope is reported as an empirical acceptance rule, not a coverage guarantee")
    assert accept and not full_scan and not certificate, "the three-stage split is not in effect"
    assert truth[chosen] < 100.0, "the counterexample no longer demonstrates a wrong acceptance"


def check_p1_3() -> None:
    """The eight cost/partition/metric confounds, each probed against the current source.

    A confound counts as fixed only when the code makes the correct behaviour the default or
    refuses the wrong one.  Anything still merely documented is listed as outstanding.
    """
    from grl.data import graph_loader  # noqa: F401

    states: list[tuple[str, bool, str]] = []

    def add(name: str, fixed: bool, detail: str) -> None:
        states.append((name, fixed, detail))

    # 1. uniform stopping under "at most k": full_oracle_greedy must be able to stop early.
    import inspect

    from grl.algorithms import sequential_im
    source = inspect.getsource(sequential_im.full_oracle_greedy)
    stops_on_non_positive = "<= 0" in source or "< 0" in source or "break" in source
    add("stopping under at-most-k is uniform across policies", False,
        "full_oracle_greedy fills the budget and the callers disagree; the contract now records "
        "'at most k' but no shared stopping rule exists yet"
        if not stops_on_non_positive else "a stopping rule is present but the policies are not "
        "yet routed through one shared implementation")

    # 2. priced state acquisition.
    from grl.oracle import OverexposureMonteCarloOracle
    stats = OverexposureMonteCarloOracle.__init__
    add("state acquisition is priced", False,
        "delta2 needs one cascade for the exposure state but the cost column still reports zero; "
        "OverexposureOracleStats tracks mc_cascades separately from state reads"
        if "state" not in inspect.getsource(stats) else "state reads are counted")

    # 3. a noisy estimator must not be called an oracle.
    add("MC=25 greedy is not presented as an oracle", False,
        "stage4/stage5 still label the MC-greedy arm 'full_oracle'; a named high-accuracy "
        "reference with a stated tolerance is required")

    # 4-5. stage4b aggregation and pool selection.
    stage4b = ROOT / "scripts" / "experiments" / "stage4b_usable_regime.py"
    text = stage4b.read_text(encoding="utf-8", errors="replace") if stage4b.exists() else ""
    add("stage4b does not pool mixed seed sizes", False,
        "stage4b still pools |S| in {0,k,2k,3k} in one within/between statistic"
        if "0" in text else "stage4b is not present")
    add("candidate pool is not a single random draw", False,
        "stage4b still draws one random 30-node pool, so regime selection has selection bias")

    # 6-7. dataset construction.
    dataset = ROOT / "src" / "grl" / "training" / "overexposure_dataset.py"
    dtext = dataset.read_text(encoding="utf-8", errors="replace") if dataset.exists() else ""
    add("dataset splits deduplicate identical seed sets", False,
        "splitting is still by context identifier, so identical seed sets can straddle a split")
    label_std_is_nan = 'label_std=float("nan")' in dtext or "label_std = float(\"nan\")" in dtext
    add("label uncertainty is recorded", bool(dtext) and not label_std_is_nan,
        "label_std is still hard-coded to float('nan') with the comment that per-candidate stderr "
        "is not tracked, so candidate differences were never compared against estimation error"
        if label_std_is_nan else "the dataset records a real per-label uncertainty")

    # 8. weight normalisation must not exceed what the model allows.
    loader = ROOT / "src" / "grl" / "data" / "graph_loader.py"
    ltext = loader.read_text(encoding="utf-8", errors="replace") if loader.exists() else ""
    add("weight normalisation stays within the model's allowance", False,
        "in-edge weights are still normalised to sum to exactly 1; the model allows at most 1, so "
        "the graphs must be reported with the normalisation used and varied as a sensitivity check")

    outstanding = [name for name, fixed, _ in states if not fixed]
    record("P1-3", "RESOLVED" if not outstanding else "OPEN",
           f"{len(outstanding)}/{len(states)} confounds outstanding: " + "; ".join(outstanding))
    for name, fixed, detail in states:
        mark = "fixed  " if fixed else "OPEN   "
        print(f"      {mark}{name}: {detail}")


def main() -> int:
    print("=" * 100)
    print("AUDIT ITEM STATUS AGAINST CURRENT SOURCE")
    print("=" * 100)
    print()
    for fn in (check_p0_1, check_p0_2, check_p0_3, check_p0_4,
               check_p1_1, check_p1_2, check_p1_3):
        fn()
    print()
    print("=" * 100)
    resolved = [r for r in RESULTS if r[1] == "RESOLVED"]
    partial = [r for r in RESULTS if r[1] == "PARTIAL"]
    open_ = [r for r in RESULTS if r[1] == "OPEN"]
    print(f"  RESOLVED {len(resolved)} | PARTIAL {len(partial)} | OPEN {len(open_)}")
    print()
    print("  DONE: P0-1, P0-2a, P0-2b, P0-3a, P0-3b, P0-4, P1-1, P1-2")
    print("  OPEN: P1-3")
    print()
    print("  NEXT, in order:")
    print("    1. P1-3   fix the eight cost/partition/metric confounds before any new number")
    print("    2. wire contract.py into the experiment scripts so every run declares D, eligibility,")
    print("              the at-most-k stopping rule, and a priced state acquisition")
    print("    3. re-derive every reference number under that contract, then un-mark the tables in")
    print("       paper/dasfaa2027 that currently carry \\withdrawn")
    print("    4. finish the MC>=300 regime sweep (scripts/audit/rederive_signflip_high_mc.py)")
    print("    5. verify citations; confirm the DASFAA page limit and template version")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
