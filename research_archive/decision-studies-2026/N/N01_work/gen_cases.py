"""N01 case generation: 40 frozen four-set comparisons on ca-GrQc.

Generation rules, frozen before any diffusion result is observed
--------------------------------------------------------------
* candidates ``a,b,c,d`` are drawn uniformly from the **largest connected component** (they must be
  distinct) and must satisfy the structural condition
  ``(R2(a) u R2(b)) n (R2(c) u R2(d)) = empty`` where ``R2`` is forward reachability within two hops;
* 20 comparisons use a background ``U`` of eight nodes drawn uniformly from the nodes that are not
  candidates;
* 20 further comparisons use a background ``U`` of the eight highest-degree nodes that are not
  candidates (ties by ascending contiguous node index);
* quadruples never repeat, across both background types;
* at most 100,000 quadruple attempts in total.  If fewer than 40 valid comparisons are found the real
  number is reported and the condition is NOT relaxed;
* the sampling seed is fixed and written to disk.

Candidate selection reads graph structure only.  No diffusion value is consulted at any point here.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from n01_common import (  # noqa: E402
    K_SEEDS, Graph, check_2hop_disjoint, parse_ca_grqc, within_hops,
)

SAMPLING_SEED = 20261012
MAX_ATTEMPTS = 100_000
N_PER_GROUP = 20


def generate(graph: Graph, lcc: list[int], r2: dict[int, set[int]]):
    rng = random.Random(SAMPLING_SEED)
    degree_ranked = [v for v in graph.degree_ranking()]
    rank_of = {v: i for i, v in enumerate(degree_ranked)}

    attempts = 0
    rejected_struct = 0
    seen_quadruples: set[tuple[int, int, int, int]] = set()
    groups: dict[str, list[dict]] = {"uniform_U": [], "degree_U": []}

    # Attempts are shared across the two background types so the 100k cap is a single global budget.
    order = ["uniform_U"] * N_PER_GROUP + ["degree_U"] * N_PER_GROUP
    targets = {"uniform_U": N_PER_GROUP, "degree_U": N_PER_GROUP}

    while attempts < MAX_ATTEMPTS and any(len(groups[g]) < targets[g] for g in targets):
        attempts += 1
        quad = tuple(rng.sample(lcc, 4))
        if quad in seen_quadruples:
            continue
        a, b, c, d = quad
        struct = check_2hop_disjoint(r2, a, b, c, d)
        if not struct["holds"]:
            rejected_struct += 1
            continue
        seen_quadruples.add(quad)

        for group in order:
            if len(groups[group]) >= targets[group]:
                continue
            if group == "uniform_U":
                pool = [v for v in lcc if v not in quad]
                u = sorted(rng.sample(pool, K_SEEDS - 2))
            else:
                u = sorted([v for v in degree_ranked if v not in quad][:K_SEEDS - 2])
            if set(u) & set(quad):
                continue
            groups[group].append({"quadruple": list(quad), "U": u,
                                  "struct": struct, "candidates_in_lcc": True})
            break
        # a single accepted quadruple fills one pending slot; keep drawing for the other group

    cases = []
    for group, rows in groups.items():
        for row in rows:
            a, b, c, d = row["quadruple"]
            u = row["U"]
            cases.append({
                "group": group,
                "a": a, "b": b, "c": c, "d": d,
                "U": u,
                "S_ac": sorted([*u, a, c]),
                "S_bc": sorted([*u, b, c]),
                "S_ad": sorted([*u, a, d]),
                "S_bd": sorted([*u, b, d]),
                "struct": row["struct"],
                "degrees": {"a": graph.degree[a], "b": graph.degree[b],
                            "c": graph.degree[c], "d": graph.degree[d],
                            "U": [graph.degree[v] for v in u]},
                "hashes": None,
            })
    meta = {
        "sampling_seed": SAMPLING_SEED,
        "max_attempts": MAX_ATTEMPTS,
        "attempts_used": attempts,
        "accepted_quadruples": len(seen_quadruples),
        "rejected_on_structure": rejected_struct,
        "legal_rate": (len(seen_quadruples) / attempts) if attempts else 0.0,
        "target_per_group": N_PER_GROUP,
        "produced": {g: len(v) for g, v in groups.items()},
        "hit_attempt_cap": attempts >= MAX_ATTEMPTS,
        "condition_relaxed": False,
    }
    return cases, meta


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    comps = graph.components()
    lcc = comps[0]

    print("=" * 96)
    print("  N01 CASE GENERATION")
    print("=" * 96)
    print(f"  graph: n={graph.n} m={graph.m} arcs={graph.n_arcs} lcc={len(lcc)}")

    # Precompute the two-hop regions once -- they are a pure function of the frozen graph.
    r2 = {v: within_hops(graph, v, 2) for v in range(graph.n)}
    print(f"  two-hop regions precomputed for all {graph.n} nodes")

    cases, meta = generate(graph, lcc, r2)
    print(f"  attempts used {meta['attempts_used']}, accepted {meta['accepted_quadruples']}, "
          f"legal rate {meta['legal_rate']:.4%}, hit cap {meta['hit_attempt_cap']}")
    print(f"  produced {meta['produced']}")

    # ------- per-case verification (deliverable acceptance checks) -------
    problems = []
    for i, cs in enumerate(cases):
        cs["index"] = i
        for key in ("S_ac", "S_bc", "S_ad", "S_bd"):
            if len(cs[key]) != K_SEEDS:
                problems.append(f"case {i}: {key} has {len(cs[key])} seeds, expected {K_SEEDS}")
            if len(set(cs[key])) != K_SEEDS:
                problems.append(f"case {i}: {key} has duplicate seeds")
        a, b, c, d = cs["a"], cs["b"], cs["c"], cs["d"]
        st = check_2hop_disjoint(r2, a, b, c, d)
        if not st["holds"]:
            problems.append(f"case {i}: two-hop condition fails on recheck")
        cs["struct"] = st
        cad = {a, b, c, d}
        if len(cad) != 4:
            problems.append(f"case {i}: candidates not distinct")
        if set(cs["U"]) & cad:
            problems.append(f"case {i}: a candidate is inside U")
        # the differences that define the comparison
        assert set(cs["S_ac"]) ^ set(cs["S_bc"]) == {a, b}
        assert set(cs["S_ad"]) ^ set(cs["S_bd"]) == {a, b}
        assert set(cs["S_ac"]) ^ set(cs["S_ad"]) == {c, d}
        assert set(cs["S_bc"]) ^ set(cs["S_bd"]) == {c, d}

    # quadruples unique across the whole set
    quads = [tuple(sorted([cs["a"], cs["b"], cs["c"], cs["d"]])) for cs in cases]
    if len(set(quads)) != len(quads):
        problems.append("duplicate quadruples present")

    print(f"\n  per-case verification problems: {len(problems)}")
    for p in problems[:20]:
        print(f"    - {p}")

    degrees = [(cs["degrees"]["a"], cs["degrees"]["b"], cs["degrees"]["c"], cs["degrees"]["d"])
               for cs in cases]
    flat = [x for row in degrees for x in row]
    print(f"\n  candidate degrees across {len(cases)} comparisons: min {min(flat)}, "
          f"median {sorted(flat)[len(flat)//2]}, max {max(flat)}")
    for g in ("uniform_U", "degree_U"):
        sub = [cs for cs in cases if cs["group"] == g]
        if sub:
            dg = [x for cs in sub for x in
                  (cs["degrees"]["a"], cs["degrees"]["b"], cs["degrees"]["c"], cs["degrees"]["d"])]
            ug = [x for cs in sub for x in cs["degrees"]["U"]]
            print(f"    {g:<11} n={len(sub):<3} candidate deg mean {sum(dg)/len(dg):8.2f} | "
                  f"U deg mean {sum(ug)/len(ug):8.2f} "
                  f"(U in LCC: {sum(1 for cs in sub for v in cs['U'] if v in set(lcc))}"
                  f"/{sum(len(cs['U']) for cs in sub)})")

    artifact = {
        "task": "N01 -- long-range conditional ranking diagnostic under a fixed seed budget",
        "graph": {
            "manifest": manifest,
            "n_lcc": len(lcc),
            "n_components": len(comps),
            "node_mapping": "ascending original ca-GrQc node id -> contiguous index 0..n-1",
            "node_index_to_original_id": {str(i): graph.original_ids[i] for i in range(graph.n)},
        },
        "protocol": {
            "propagation_model": "independent cascade (live-edge view)",
            "graph_reading": "undirected simple graph; each edge becomes two propagation directions",
            "arc_probability": "p_{u,v} = 1 / deg(v), deg = undirected degree",
            "bidirectional_randomness": "the two directions are sampled independently",
            "rounds_cap": 100,
            "seed_budget_k": K_SEEDS,
            "activation_count_includes_seeds": True,
            "common_random_numbers": "one live/dead state per directed arc per world, shared by all "
                                     "four seed sets of a comparison; NOT per-set re-seeding",
        },
        "generation": meta,
        "verification": {"problems": problems, "n_problems": len(problems)},
        "cases": cases,
    }
    out = HERE / "N01_cases.json"
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
