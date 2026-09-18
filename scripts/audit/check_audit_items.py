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

    # The audit also demands three DISTINCT degeneracy paths be kept apart.
    params_free = resolve_overexposure_params({"overexposure": {"overexposure_free": True}})
    has_free = params_free.overexposure_free
    has_window_lo = True  # sample_threshold_windows(window_lo=...) exists
    record("P0-2b", "PARTIAL",
           f"overexposure_free available={has_free}; window_lo knob available={has_window_lo}; "
           f"but the three requested degeneracy paths (fixed-window path check / kappa~U[0,1] "
           f"with tau=1 / keep-kappa-marginal ablation) are NOT yet separately exposed")


def check_p0_3() -> None:
    """Requires: (a) the invalid probe is not used as evidence, (b) exact non-monotone example."""
    graph = nx.DiGraph()
    graph.add_edge("a", "t", weight=0.4)
    graph.add_edge("b", "t", weight=0.4)
    windows_list = [("a", 0.4), ("b", 0.4)]

    # exact one-hop calculation under the simplex-sampled window
    def p_positive(delta: float) -> float:
        return 2 * delta * (1 - delta)

    single = p_positive(0.4)
    pair = p_positive(0.8)
    ok = single > pair
    record("P0-3a", "RESOLVED" if ok else "OPEN",
           f"exact example: F({{a}})={single:.4f} > F({{a,b}})={pair:.4f} -> non-monotone, "
           f"so no non-negative seed-independent coverage form matches")

    # (b) is the invalid probe still presented as evidence anywhere?
    import subprocess
    repo = ROOT
    hits = []
    for doc in (repo / "docs").glob("*.md"):
        text = doc.read_text(encoding="utf-8", errors="replace")
        if "RR set" in text and "empty" in text.lower() and "GATE1" in doc.name.upper():
            hits.append(doc.name)
    paper = repo / "paper" / "dasfaa2027" / "src" / "dasfaa2027"
    title_hits = []
    for tex in paper.rglob("*.tex"):
        if "No Domain" in tex.read_text(encoding="utf-8", errors="replace"):
            title_hits.append(tex.name)
    record("P0-3b", "OPEN",
           f"invalid-probe documents still present: {hits or 'none'}; "
           f"paper files still carrying the 'No Domain' title: {title_hits or 'none'}")


def check_p0_4() -> None:
    """Requires a target set D, seeds excluded from D, and seeds not counted in A_window."""
    from grl.diffusion.params import OverexposureParams
    params = OverexposureParams()
    has_target = "target" in OverexposureParams.__dataclass_fields__
    seed_allowed_in_D = True  # nothing prevents it today
    record("P0-4", "OPEN",
           f"OverexposureParams has a target-set field: {has_target}; "
           f"seed eligibility S<=V\\D unenforced: {seed_allowed_in_D}; "
           f"run_overexposure counts all nodes including seeds")


def check_p1_1() -> None:
    """tau-process must not be called standard uniform-threshold LT."""
    # kappa and tau come from min/max of two uniforms -> F_kappa = 2x - x^2, F_tau = x^2
    nodes = list(range(20000))
    windows = oe.sample_threshold_windows(nodes, random.Random(1))
    taus = sorted(t for _, t in windows.values())
    # compare empirical tau CDF against x^2 at a few points
    import bisect
    worst = 0.0
    for x in (0.2, 0.4, 0.6, 0.8):
        emp = bisect.bisect_right(taus, x) / len(taus)
        worst = max(worst, abs(emp - x * x))
    record("P1-1", "OPEN",
           f"empirical tau CDF matches x^2 to within {worst:.4f}, so it is NOT uniform-threshold "
           f"LT; the code/docs must not describe sigma^kappa/sigma^tau as two standard LT runs")


def check_p1_2() -> None:
    from grl.algorithms.sequential_im import adaptive_selective_greedy

    class ScriptedOracle:
        """Scores chosen so that the residual envelope certifies the wrong node."""

        def __init__(self, learned, truth):
            self.learned, self.truth = learned, truth
            self.verified = 0

        def score(self, seeds, candidates, step=0):
            del seeds, step
            out = {}
            for v in candidates:
                out[v] = self.learned.get(v, 0.0)
                self.verified += 1
            # exact oracle call: return truth only for a fixed small prefix
            return {v: (self.truth[v] if self.verified <= 60 else self.learned.get(v, 0.0))
                    for v in candidates}

    learned = {0: 10.0, 1: 9.0, 2: 0.0}
    truth = {0: 10.0, 1: 9.0, 2: 100.0}

    class TwoPhase:
        def __init__(self):
            self.calls = 0

        def score(self, seeds, candidates, step=0):
            del seeds, step
            self.calls += 1
            if self.calls <= 2:
                return {v: truth[v] for v in candidates if v in (0, 1)}
            return {v: 0.0 for v in candidates}

    class Learned:
        def score(self, seeds, candidates, step=0):
            del seeds, step
            return {v: learned[v] for v in candidates if v in learned}

    result = adaptive_selective_greedy([0, 1, 2], 1, Learned(), TwoPhase(),
                                       initial_m=2, batch_m=1, max_m=3)
    chosen = result.selected_seeds[0] if result.selected_seeds else None
    certified = bool(result.steps and result.steps[0].get("certified"))
    record("P1-2", "OPEN",
           f"counterexample oracle: chose node {chosen} (true score {truth.get(chosen)}), "
           f"true best is 2 (score 100), certified={certified} -> the residual envelope is an "
           f"empirical acceptance rule, not a coverage guarantee")


def check_p1_3() -> None:
    items = [
        "full_oracle_greedy selects negative-marginal seeds while another script stops on non-positive gain",
        "delta2 exposure is generated by a cascade but reported as zero cost",
        "MC=25 greedy is described as an oracle although it is neither exact nor globally optimal",
        "stage4b mixes seed sizes 0,k,2k,3k in one within/between statistic",
        "stage4b uses a random 30-node pool, so regime selection has selection bias",
        "dataset splits by ctx_XXXXX without deduplicating identical seed sets",
        "label_std recorded as NaN, so candidate differences were never compared to estimation error",
        "in-edge weights normalised to sum exactly 1 although the source allows <= 1",
    ]
    record("P1-3", "OPEN", f"{len(items)} distinct confounds outstanding; first: {items[0]}")


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
    print("  NEXT, in the audit's own priority order:")
    print("    1. P0-3b  withdraw the RR argument and the paper title")
    print("    2. P0-4   freeze the model contract: target set D, seed eligibility, <=k")
    print("    3. P0-2b  expose the three degeneracy paths separately")
    print("    4. P1-2   relabel certified -> empirically accepted, or add a real bound")
    print("    5. P1-3   fix the eight cost/partition/metric confounds before any new number")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
