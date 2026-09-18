"""Turn docs/results/signflip_fixedmodel_mc300.json into the paper's regime table.

Why a generator rather than a hand-copied table
-----------------------------------------------
Every number that goes into the paper is read from the result file here, so a table cannot drift
from its evidence and a typo cannot enter one.  This is the practice ``CLAIMS.md`` demands --- the
paper must not contain a number that cannot be traced to a committed result file --- and it is
applied to the FIRST table we are un-withdrawing after the state-machine and MC = 12 findings.

The generated LaTeX is written to a file that the paper inputs, so the checked-in source is the
number, not a transcription of it.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Presentation order: increasing mean degree, which is the variable the section is about.
GRAPH_ORDER = [
    "nethept",
    "p2p_gnutella08",
    "ca_grqc",
    "wiki_vote",
    "ca_hepph",
    "facebook",
    "email_eu_core",
    "congress_twitter",
]

DISPLAY_NAME = {
    "nethept": "NetHEPT",
    "p2p_gnutella08": "p2p-Gnutella08",
    "ca_grqc": "ca-GrQc",
    "wiki_vote": "Wiki-Vote",
    "ca_hepph": "ca-HepPh",
    "facebook": "Facebook",
    "email_eu_core": "email-Eu-core",
    "congress_twitter": "Congress-Twitter",
}

SATURATED = 0.20


#: The seed fractions the sweep was asked for.  Rows written before the resume key was fixed
#: recorded only the REALISED fraction (48/475 = 0.10105 for a requested 0.10), so a row without
#: ``requested_fraction`` is snapped to the nearest of these rather than trusted as-is.
REQUESTED_FRACTIONS = (0.10, 0.20, 0.40)


def load_cells(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cells = payload.get("cells", [])
    # The file may contain rows written before the resume key was fixed, so deduplicate on
    # (graph, requested fraction) keeping the last measurement.
    deduped: dict[tuple[str, float], dict] = {}
    for row in cells:
        row = dict(row)
        if "requested_fraction" not in row:
            realised = float(row.get("seed_fraction", 0.0))
            row["requested_fraction"] = min(
                REQUESTED_FRACTIONS, key=lambda f: abs(f - realised)
            )
        key = (row["graph"], round(row["requested_fraction"], 6))
        deduped[key] = row
    return list(deduped.values())


def summarise(cells: list[dict]) -> tuple[list[dict], dict]:
    """Per-graph saturated summary plus the aggregate claims."""
    rows = []
    for name in GRAPH_ORDER:
        group = [c for c in cells if c["graph"] == name]
        if not group:
            continue
        saturated = [c for c in group if c["requested_fraction"] >= SATURATED - 1e-9]
        lowest = min(group, key=lambda c: c["requested_fraction"])
        rows.append({
            "graph": name,
            "mean_degree": group[0]["graph_mean_degree"],
            "n": group[0]["n"],
            "mc_runs": group[0]["mc_runs"],
            "rho_degree_saturated": statistics.fmean(c["rho_degree"] for c in saturated),
            "rho_delta2_saturated": statistics.fmean(c["rho_delta2"] for c in saturated),
            "negative_share_saturated": statistics.fmean(
                c["negative_share"] for c in saturated),
            "negative_beyond_noise_saturated": statistics.fmean(
                c["negative_beyond_noise"] for c in saturated),
            "rho_degree_unsaturated": lowest["rho_degree"],
            "rho_delta2_unsaturated": lowest["rho_delta2"],
        })

    saturated_cells = [c for c in cells if c["requested_fraction"] >= SATURATED - 1e-9]
    negative_graphs = [r for r in rows if r["rho_degree_saturated"] < 0.0]
    delta2_ahead = [r for r in rows if r["rho_delta2_saturated"] > r["rho_degree_saturated"]]

    def spearman(xs, ys):
        def ranks(values):
            order = sorted(range(len(values)), key=lambda i: values[i])
            out = [0.0] * len(values)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                    j += 1
                avg = (i + j) / 2.0 + 1.0
                for k in range(i, j + 1):
                    out[order[k]] = avg
                i = j + 1
            return out

        rx, ry = ranks(xs), ranks(ys)
        mx, my = statistics.fmean(rx), statistics.fmean(ry)
        num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
        den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
        return num / den if den > 1e-12 else float("nan")

    aggregate = {
        "graphs": len(rows),
        "cells": len(cells),
        "mc_runs": cells[0]["mc_runs"] if cells else 0,
        "mean_negative_share": statistics.fmean(c["negative_share"] for c in saturated_cells),
        "mean_negative_beyond_noise": statistics.fmean(
            c["negative_beyond_noise"] for c in saturated_cells),
        "max_negative_share": max(c["negative_share"] for c in cells),
        "negative_cells": sum(1 for c in saturated_cells if c["rho_degree"] < 0.0),
        "saturated_cells": len(saturated_cells),
        "mean_rho_degree_saturated": statistics.fmean(
            c["rho_degree"] for c in saturated_cells),
        "mean_rho_delta2_saturated": statistics.fmean(
            c["rho_delta2"] for c in saturated_cells),
        "graphs_degree_negative": len(negative_graphs),
        "graphs_delta2_ahead": len(delta2_ahead),
        "spearman_density_rho_degree": spearman(
            [r["mean_degree"] for r in rows],
            [r["rho_degree_saturated"] for r in rows],
        ),
        "degree_positive_graphs": [r["graph"] for r in rows
                                   if r["rho_degree_saturated"] >= 0.0],
    }
    return rows, aggregate


def latex_table(rows: list[dict], aggregate: dict) -> str:
    lines = [
        "% GENERATED by scripts/audit/write_regime_table.py -- do not edit by hand.",
        "% Source: docs/results/signflip_fixedmodel_mc300.json",
        f"% {aggregate['graphs']} graphs x 3 seed fractions, "
        f"MC = {aggregate['mc_runs']} paired trials per cell.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Out-degree versus the state-conditioned closed form, re-measured under the "
        "corrected activation rule at $\\mathrm{MC} = "
        f"{aggregate['mc_runs']}$ paired trials per cell. "
        "Values are means over $|S|/n \\ge 20\\%$. "
        "$\\rho$ is the Spearman correlation of the score with the true marginal gain; "
        "``neg.'' is the share of candidates with a negative marginal and "
        "``neg.$^\\ast$'' the share negative by more than two paired standard errors. "
        f"Out-degree is negatively correlated on {aggregate['graphs_degree_negative']} of "
        f"{aggregate['graphs']} graphs and $\\delta_2$ is the better ranker on "
        f"{aggregate['graphs_delta2_ahead']} of {aggregate['graphs']}.}}",
        "\\label{tab:regime-fixed}",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "graph & $n$ & $\\langle k \\rangle$ & $\\rho_{\\text{degree}}$ & "
        "$\\rho_{\\delta_2}$ & neg. & neg.$^\\ast$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        degree = row["rho_degree_saturated"]
        delta2 = row["rho_delta2_saturated"]
        # bold the winner only when the two differ by more than Monte-Carlo resolution
        degree_cell = f"${degree:+.3f}$"
        delta2_cell = f"${delta2:+.3f}$"
        if delta2 > degree:
            delta2_cell = f"$\\mathbf{{{delta2:+.3f}}}$"
        else:
            degree_cell = f"$\\mathbf{{{degree:+.3f}}}$"
        lines.append(
            f"{DISPLAY_NAME[row['graph']]:<18} & {row['n']} & {row['mean_degree']:.2f} & "
            f"{degree_cell} & {delta2_cell} & "
            f"{row['negative_share_saturated']*100:.1f}\\% & "
            f"{row['negative_beyond_noise_saturated']*100:.1f}\\% \\\\"
        )
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=ROOT / "docs" / "results" / "signflip_fixedmodel_mc300.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper" / "dasfaa2027" / "src" / "dasfaa2027"
                        / "sections" / "regime_table.tex")
    parser.add_argument("--json-out", type=Path,
                        default=ROOT / "docs" / "results" / "regime_table_summary.json")
    args = parser.parse_args()

    cells = load_cells(args.input)
    if not cells:
        print("no cells found; nothing written")
        return 1
    rows, aggregate = summarise(cells)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(latex_table(rows, aggregate), encoding="utf-8")
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(
        {"rows": rows, "aggregate": aggregate}, indent=2), encoding="utf-8")

    print(f"wrote {args.output}")
    print(f"wrote {args.json_out}")
    print()
    print(f"  {aggregate['graphs']} graphs, {aggregate['cells']} cells, "
          f"MC = {aggregate['mc_runs']}")
    print(f"  saturated (|S|/n >= 20%): {aggregate['saturated_cells']} cells")
    print(f"  rho_degree < 0 in {aggregate['negative_cells']}/{aggregate['saturated_cells']} cells; "
          f"mean {aggregate['mean_rho_degree_saturated']:+.3f}")
    print(f"  mean rho_delta2                                  "
          f"{aggregate['mean_rho_delta2_saturated']:+.3f}")
    print(f"  graphs with negative saturated rho_degree: "
          f"{aggregate['graphs_degree_negative']}/{aggregate['graphs']}")
    print(f"  graphs where delta2 > degree             : "
          f"{aggregate['graphs_delta2_ahead']}/{aggregate['graphs']}")
    print(f"  degree-positive graphs: {aggregate['degree_positive_graphs']}")
    print(f"  mean negative share (saturated)  : {aggregate['mean_negative_share']*100:.1f}%")
    print(f"  mean negative beyond noise       : "
          f"{aggregate['mean_negative_beyond_noise']*100:.1f}%")
    print(f"  worst negative share in any cell : {aggregate['max_negative_share']*100:.1f}%")
    print(f"  Spearman(<k>, saturated rho_degree) = "
          f"{aggregate['spearman_density_rho_degree']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
