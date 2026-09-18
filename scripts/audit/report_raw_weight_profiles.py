"""Report the raw in-weight profile of each configured graph, before any normalisation.

Why this exists
---------------
The Go/No-Go runner's first version called ``normalise_in_weights`` on the output of
``evaluate_density_degree_signflip.load``, which normalises internally.  Every strategy therefore
became a no-op and ``sum_to_one`` and ``clip_to_one`` produced byte-identical columns.  The
identical columns were the tell.

This prints what the files actually carry, so that a degenerate strategy comparison is visible
*before* a sweep is run on it rather than inferred from suspicious agreement afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from evaluate_density_degree_signflip import GRAPHS  # noqa: E402
from grl.data import graph_loader  # noqa: E402
from grl.data.weights import in_weight_totals  # noqa: E402


def raw_profile(graph) -> dict:
    totals = sorted(in_weight_totals(graph).values())
    nonzero = [t for t in totals if t > 0.0]
    return {
        "n": len(totals),
        "n_with_in_edges": len(nonzero),
        "max_in_total": totals[-1] if totals else 0.0,
        "min_nonzero_in_total": nonzero[0] if nonzero else 0.0,
        "nodes_over_one": sum(1 for t in totals if t > 1.0 + 1e-9),
        "nodes_at_one": sum(1 for t in totals if abs(t - 1.0) <= 1e-9),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", nargs="+", default=list(GRAPHS), choices=list(GRAPHS))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "docs" / "results" / "raw_weight_profiles.json")
    args = parser.parse_args()

    rows = {}
    print(f"  {'graph':<20}{'n':>8}{'with in':>9}{'max total':>11}{'min nonzero':>13}"
          f"{'over 1':>8}{'at 1':>8}  strategies differ?")
    for name in args.graphs:
        path, directed = GRAPHS[name]
        graph, _ = graph_loader._parse_graph_file(path, directed, 0.01)
        profile = raw_profile(graph)
        rows[name] = profile
        differ = (profile["nodes_over_one"] > 0
                  or profile["nodes_at_one"] < profile["n_with_in_edges"])
        print(f"  {name:<20}{profile['n']:>8}{profile['n_with_in_edges']:>9}"
              f"{profile['max_in_total']:>11.4f}{profile['min_nonzero_in_total']:>13.4f}"
              f"{profile['nodes_over_one']:>8}{profile['nodes_at_one']:>8}  "
              f"{'yes' if differ else 'NO -- degenerate'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print()
    degenerate = [n for n, p in rows.items()
                  if p["nodes_over_one"] == 0 and p["nodes_at_one"] == p["n_with_in_edges"]]
    if degenerate:
        print(f"  sum_to_one and clip_to_one are IDENTICAL on: {', '.join(degenerate)}")
        print("  Those graphs already have every in-weight total at exactly 1, so the strategy")
        print("  comparison is degenerate there and must not be reported as a finding.")
    print(f"\n  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
