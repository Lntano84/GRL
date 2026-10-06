"""Build the candidate-instance list for a possible larger-scale cost probe.

This script runs **no diffusion and no search**.  It only reads graphs that are already in the repo and
reports, for each candidate:

* size and directedness, as stored and after symmetrisation (the CCIM neighbourhood ``N(x)`` is
  undirected in the source paper, so the symmetrised reading is the one a CCIM run would use);
* the degree profile and the ``K``-core size for ``K = 2..6``.

Why the K-core matters and why ``K=4, T=8`` must not simply be carried over: a node can only be
activated by its neighbours if it has at least ``K`` of them, so non-seed activations live inside the
``K``-core.  On a graph whose ``K``-core is tiny, a threshold-``K`` cascade cannot spread at all; on a
graph where almost every node is in a dense ``K``-core with high degree, it may saturate immediately.
Either way the small-graph parameters stop meaning what they meant on Football and Polbooks.

Existing timing records are collected alongside, because the decision rule is: if the search is
already fast at the larger size, the probe is not worth starting.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

GRL = Path(r"C:\Users\windows\Desktop\_grl_merge\merged")
CCIM = Path(__file__).resolve().parents[1]
for extra in (GRL / "src", GRL / "scripts" / "experiments", CCIM):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import networkx as nx  # noqa: E402


def graph_manifest(path: Path) -> dict:
    """File-level facts about a graph, so a node/edge count can never be ambiguous again.

    Records the hash, how many lines were skipped as comments, how many self-loop and duplicate rows
    there are, whether the file stores each undirected pair once or twice, and what the simple
    undirected reading would be.  Dropping self-loops before versus after building the graph changes
    the node count by any node that appears ONLY as a self-loop, so that node is named explicitly.
    """
    import hashlib

    raw = path.read_bytes()
    data_lines: list[str] = []
    comment_lines = 0
    for line in raw.decode("utf-8", errors="replace").splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("%") or s.startswith("//"):
            comment_lines += 1
            continue
        data_lines.append(s)

    # A leading ``n m`` header is ambiguous with a real edge.  The loader in the GRL repo resolves it
    # with three conditions; the same rule is applied here so the two readings can be compared.  It
    # matters: without it the header is counted as an edge and its two numbers as nodes.
    header_skipped = None
    if data_lines:
        first = data_lines[0].split()
        if len(first) == 2 and all(p.isdigit() for p in first):
            n_hdr, m_hdr = int(first[0]), int(first[1])
            rest_ids = [int(p) for line in data_lines[1:] for p in line.split()[:2]
                        if p.lstrip("-").isdigit()]
            if m_hdr >= n_hdr and rest_ids and n_hdr >= max(rest_ids) / 10:
                header_skipped = [n_hdr, m_hdr]
                data_lines = data_lines[1:]

    rows: list[tuple[int, int]] = []
    non_numeric = 0
    for s in data_lines:
        parts = s.split()
        if len(parts) < 2:
            non_numeric += 1
            continue
        try:
            rows.append((int(parts[0]), int(parts[1])))
        except ValueError:
            non_numeric += 1
    self_loops = [r for r in rows if r[0] == r[1]]
    non_loop = [r for r in rows if r[0] != r[1]]
    as_directed = set(rows)
    unordered = {tuple(sorted(r)) for r in non_loop}
    both = sum(1 for (u, v) in unordered if (u, v) in as_directed and (v, u) in as_directed)
    all_ids = {x for r in rows for x in r}
    non_loop_ids = {x for r in non_loop for x in r}
    return {
        "file": path.name, "sha256": hashlib.sha256(raw).hexdigest().upper(),
        "comment_or_header_lines": comment_lines, "unparseable_lines": non_numeric,
        "leading_n_m_header_skipped": header_skipped,
        "data_rows": len(rows), "self_loop_rows": len(self_loops),
        "duplicate_rows_same_direction": len(rows) - len(as_directed),
        "as_directed_arcs": len(as_directed), "unordered_pairs": len(unordered),
        "pairs_with_both_directions_present": both,
        "pairs_with_only_one_direction_present": len(unordered) - both,
        "distinct_node_ids": len(all_ids),
        "nodes_only_present_as_self_loops": sorted(all_ids - non_loop_ids),
        "nodes_in_simple_undirected_graph": len(non_loop_ids),
        "symmetrised_edge_ratio": (len(unordered) / len(as_directed)) if as_directed else None,
        "rules": {"self_loops": "dropped", "parallel_edges": "collapsed",
                  "undirected_reading": "used (CCIM's N(x) is undirected)",
                  "node_ids": "kept as stored, no relabelling"},
    }


def k_core_size(graph: nx.Graph, k: int) -> int:
    """Size of the maximal subgraph with minimum degree >= k (0 if there is none)."""
    if k <= 0:
        return graph.number_of_nodes()
    try:
        return len(nx.k_core(graph, k=k))
    except nx.NetworkXError:
        return 0


def k_core_largest_component(graph: nx.Graph, k: int) -> int:
    """Largest connected component *inside* the k-core.

    A threshold-k cascade can only move between nodes that both sit in the k-core, so a k-core that is
    large but shredded into small pieces cannot carry a wide cascade.  This is the cheapest structural
    warning that "K is meaningful here" is false in either direction.
    """
    if k <= 0:
        return graph.number_of_nodes()
    try:
        core = nx.k_core(graph, k=k)
    except nx.NetworkXError:
        return 0
    if core.number_of_nodes() == 0:
        return 0
    return max((len(c) for c in nx.connected_components(core)), default=0)


def profile(graph: nx.Graph, label: str) -> dict:
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degs = [d for _, d in graph.degree()]
    row = {"label": label, "n": n, "m": m,
           "avg_degree": statistics.fmean(degs) if degs else 0.0,
           "min_degree": min(degs) if degs else 0,
           "median_degree": statistics.median(degs) if degs else 0,
           "max_degree": max(degs) if degs else 0}
    for k in (2, 3, 4, 5, 6):
        ge = sum(1 for d in degs if d >= k)
        row[f"deg_ge_{k}"] = ge
        row[f"kc_{k}"] = k_core_size(graph, k)
        row[f"kc_lcc_{k}"] = k_core_largest_component(graph, k)
    return row


def main() -> int:
    from evaluate_density_degree_signflip import GRAPHS, load  # noqa: E402

    rows = []
    print("loading graphs (parsing only, no diffusion)...", flush=True)
    for name, (path, directed) in GRAPHS.items():
        if not path.exists():
            print(f"  {name}: MISSING {path.name}", flush=True)
            continue
        try:
            g = load(name)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"  {name}: FAILED to load ({exc})", flush=True)
            continue
        n_as_loaded = g.number_of_nodes()
        m_as_loaded = g.number_of_edges()
        und = nx.Graph(g)                     # symmetrise: the CCIM neighbourhood is undirected
        und.remove_edges_from(nx.selfloop_edges(und))
        row = profile(und, name)
        row.update({"directed_as_stored": bool(directed or g.is_directed()),
                    "n_as_stored": n_as_loaded, "m_as_stored": m_as_loaded,
                    "manifest": graph_manifest(path)})
        # how many nodes vanish from the "can be activated by a neighbour" set at each K
        row["connected"] = nx.is_connected(und) if und.number_of_nodes() else False
        rows.append(row)
        print(f"  {name}: n={row['n']} m={row['m']} deg {row['min_degree']}-{row['max_degree']} "
              f"K4-core={row['kc_4']} K6-core={row['kc_6']}", flush=True)

    # ---- existing timing records ------------------------------------------------
    timings = []
    head = json.loads((GRL / "docs" / "results" / "shortlist_headroom.json").read_text(encoding="utf-8"))
    for block in head.get("blocks", []):
        if block.get("skipped"):
            continue
        runs = block.get("mc_per_batch", 0) * 2 * (1 + block.get("candidate_count", 0))
        secs = block.get("batch_seconds")
        if runs and secs:
            timings.append({"source": "GRL shortlist_headroom.json", "graph": block["graph"],
                            "diffuser": "overexposure (threshold-window, stochastic)",
                            "cascade_runs": runs, "seconds": secs,
                            "ms_per_cascade": secs / runs * 1000,
                            "note": "different diffusion model from CCIM"})
    gate2 = json.loads((CCIM / "results" / "scan_order_pilot.json").read_text(encoding="utf-8"))
    for gname, entry in gate2["graphs"].items():
        for method, arm in entry["arms"].items():
            ws = [r["wall_seconds"] for r in arm["runs"]]
            q = [r["budget_used"] for r in arm["runs"]]
            timings.append({"source": "CCIM scan_order_pilot.json", "graph": gname,
                            "diffuser": "CCIM deterministic threshold (this project)",
                            "cascade_runs": statistics.fmean(q), "seconds": statistics.fmean(ws),
                            "ms_per_cascade": statistics.fmean(ws) / statistics.fmean(q) * 1000,
                            "note": f"search arm = {method}, 5000-query budget"})
    gate1 = json.loads((CCIM / "results" / "gate1_multistep_value.json").read_text(encoding="utf-8"))
    for gname, entry in gate1["graphs"].items():
        for r in entry.get("dqn_runs", []):
            timings.append({"source": "CCIM gate1_multistep_value.json", "graph": gname,
                            "diffuser": "CCIM deterministic threshold (this project)",
                            "cascade_runs": r["cost"]["decision_evaluations"],
                            "seconds": r["seconds"],
                            "ms_per_cascade": r["seconds"] / max(1, r["cost"]["decision_evaluations"]) * 1000,
                            "note": f"DQN training, seed {r['train_seed']}"})

    out = {"script": Path(__file__).name,
           "note": "candidate list only; no diffusion and no search was run to build this",
           "candidates": rows, "timing_records": timings,
           "ccim_validated_graphs": ["football", "polbooks"],
           "validation_meaning": "cascade brute-force checked and Greedy reproduced Table 1 exactly; "
                                 "no other graph has been through the CCIM diffuser"}
    (CCIM / "results" / "candidate_instances.json").write_text(json.dumps(out, indent=2),
                                                              encoding="utf-8")

    print("\n" + "=" * 118)
    print("  CANDIDATE INSTANCES (symmetrised, since CCIM's N(x) is undirected)")
    print("=" * 118)
    hdr = (f"  {'graph':<17}{'n':>8}{'m':>9}{'dir':>5}{'deg min/med/max':>17}"
           f"{'>=4':>8}{'>=6':>8}{'K4core':>8}{'K6core':>8}{'conn':>6}")
    print(hdr)
    for r in sorted(rows, key=lambda x: x["n"]):
        deg_str = f"{r['min_degree']}/{r['median_degree']:.0f}/{r['max_degree']}"
        print(f"  {r['label']:<17}{r['n']:>8}{r['m']:>9}"
              f"{str(r['directed_as_stored']):>5}{deg_str:>17}"
              f"{r['deg_ge_4']:>8}{r['deg_ge_6']:>8}{r['kc_4']:>8}{r['kc_6']:>8}"
              f"{str(r['connected']):>6}")

    print("\n" + "=" * 118)
    print("  IS A THRESHOLD-K CASCADE STRUCTURALLY POSSIBLE?  (|k-core| and its largest component)")
    print("=" * 118)
    print(f"  {'graph':<18}{'n':>8}" + "".join(f"{'K=' + str(k):>19}" for k in (2, 3, 4, 5, 6)))
    for r in sorted(rows, key=lambda x: x["n"]):
        cells = []
        for k in (2, 3, 4, 5, 6):
            frac = r[f"kc_lcc_{k}"] / r["n"] * 100
            cells.append(f"{r[f'kc_{k}']:>7}({frac:>4.0f}%)".rjust(19))
        print(f"  {r['label']:<18}{r['n']:>8}" + "".join(cells))
    print("  (shown as |k-core| with the largest component of the k-core as a share of n;")
    print("   a large k-core shredded into small pieces still cannot carry a wide cascade)")

    print("\n" + "=" * 118)
    print("  EXISTING TIMING RECORDS")
    print("=" * 118)
    print(f"  {'source':<34}{'graph':<17}{'runs':>10}{'seconds':>10}{'ms/cascade':>12}   diffuser")
    for t in sorted(timings, key=lambda x: x["ms_per_cascade"]):
        print(f"  {t['source']:<34}{t['graph']:<17}{t['cascade_runs']:>10.0f}"
              f"{t['seconds']:>10.1f}{t['ms_per_cascade']:>12.3f}   {t['diffuser']}")
    print(f"\n  wrote {CCIM / 'results' / 'candidate_instances.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
