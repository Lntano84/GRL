"""N02 step 2 -- structure-only selection of quadruples with a three-hop-interaction screen.

Frozen before any diffusion is seen:
  * sampling seed 2026092802, at most 100,000 attempts, candidates drawn uniformly and distinctly
    from the largest connected component;
  * the ORIGINAL condition must hold: (R_2(a) u R_2(b)) n (R_2(c) u R_2(d)) = empty, with R_h
    INCLUDING the source node and following the actual message-passing direction;
  * Q_3 = | |R_3(a)nR_3(c)| + |R_3(b)nR_3(d)| - |R_3(a)nR_3(d)| - |R_3(b)nR_3(c)| |,
    keep Q_3 > 0, take the top 64 by Q_3 descending, ties by the node-index tuple;
  * unordered four-node sets are unique, first occurrence wins;
  * unordered four-node sets already used by N01 or N01-S are excluded;
  * if fewer than 64 qualify, stop and report the shortfall -- the condition is NOT relaxed.

Q_3 is a STRUCTURAL screen only.  It is not a spread estimate and not evidence about learning.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "N01_work"))
sys.path.insert(0, str(HERE))

from n01_common import K_SEEDS, Graph, parse_ca_grqc  # noqa: E402
from n02_common import reach_within  # noqa: E402

SAMPLING_SEED = 2026092802
MAX_ATTEMPTS = 100_000
TARGET_QUADRUPLES = 64
N_UNIFORM_U = 128


def prior_quadruples() -> set[tuple[int, int, int, int]]:
    """Unordered four-node sets already used by N01 and N01-S."""
    used = set()
    for name in ("N01_cases.json", "N01S_cases.json"):
        path = ROOT / "N01_work" / name
        if not path.exists():
            continue
        art = json.loads(path.read_text(encoding="utf-8"))
        for cs in art["cases"]:
            used.add(tuple(sorted((cs["a"], cs["b"], cs["c"], cs["d"]))))
    return used


def main() -> int:
    arcs, undirected, node_ids, manifest = parse_ca_grqc()
    graph = Graph(arcs, undirected, node_ids)
    lcc = graph.components()[0]
    n = graph.n
    print("=" * 100)
    print("  N02 STEP 2 -- structure-only quadruple selection")
    print("=" * 100)
    print(f"  graph n={n} m={graph.m} lcc={len(lcc)} hash={manifest['sha256'][:16]}...")
    print(f"  sampling seed {SAMPLING_SEED}, max attempts {MAX_ATTEMPTS}")

    used_before = prior_quadruples()
    print(f"  unordered four-node sets already used by N01/N01-S: {len(used_before)}")

    # R_2 (inclusive) and R_3 (inclusive) for every node: pure structure, computed once.
    r2 = {v: reach_within(graph, v, 2, include_source=True) for v in range(n)}
    r3 = {v: reach_within(graph, v, 3, include_source=True) for v in range(n)}
    print("  R_2 and R_3 (source included) computed for all nodes")

    rng = random.Random(SAMPLING_SEED)
    seen: set[tuple[int, int, int, int]] = set()
    attempts = 0
    rejected_condition = 0
    rejected_prior = 0
    rejected_q3_zero = 0
    accepted: list[tuple[int, int, int, int, int]] = []   # (Q3, a, b, c, d)

    while attempts < MAX_ATTEMPTS:
        attempts += 1
        quad = tuple(rng.sample(lcc, 4))
        key = tuple(sorted(quad))
        if key in seen:
            continue
        seen.add(key)
        if key in used_before:
            rejected_prior += 1
            continue
        a, b, c, d = quad
        if (r2[a] | r2[b]) & (r2[c] | r2[d]):
            rejected_condition += 1
            continue
        q3 = abs(len(r3[a] & r3[c]) + len(r3[b] & r3[d])
                 - len(r3[a] & r3[d]) - len(r3[b] & r3[c]))
        if q3 <= 0:
            rejected_q3_zero += 1
            continue
        accepted.append((q3, a, b, c, d))

    print(f"\n  attempts used            : {attempts}")
    print(f"  distinct unordered sets  : {len(seen)}")
    print(f"  rejected on prior use    : {rejected_prior}")
    print(f"  rejected on the ORIGINAL two-hop condition : {rejected_condition}")
    print(f"  passed condition but Q_3 == 0              : {rejected_q3_zero}")
    print(f"  passed condition with Q_3 > 0              : {len(accepted)}")

    # top 64 by Q_3 desc, ties by node-index tuple
    accepted.sort(key=lambda t: (-t[0], (t[1], t[2], t[3], t[4])))
    selected = accepted[:TARGET_QUADRUPLES]
    pool_shortfall = len(accepted) < TARGET_QUADRUPLES
    print(f"  selected (top by Q_3)    : {len(selected)}"
          + ("   *** POOL SHORTFALL -- condition NOT relaxed ***" if pool_shortfall else ""))

    q3_vals = [t[0] for t in selected]
    if q3_vals:
        print(f"  Q_3 of selected: min {min(q3_vals)}, median "
              f"{sorted(q3_vals)[len(q3_vals)//2]}, max {max(q3_vals)}")
    cond_of_pool = len(accepted)
    print(f"  condition acceptance rate among distinct sets: "
          f"{cond_of_pool / max(1, len(seen) - rejected_prior):.4%}")

    # ---------------------------------------------------------------- backgrounds
    degree_ranked = graph.degree_ranking()
    top8_global = degree_ranked[:K_SEEDS - 2]
    print(f"\n  degree background (global top 8, ties by ascending index): {top8_global}")

    bg_rng = random.Random(SAMPLING_SEED + 1)      # separate stream, frozen
    comparisons = []
    for q3, a, b, c, d in selected:
        quad = (a, b, c, d)
        uniform_pool = [v for v in lcc if v not in quad]
        U_uniform = sorted(bg_rng.sample(uniform_pool, K_SEEDS - 2))
        U_degree = sorted(v for v in top8_global if v not in quad)
        for kind, U in (("uniform_U", U_uniform), ("degree_U", U_degree)):
            comparisons.append({
                "quadruple": list(quad), "Q3": q3, "background_kind": kind, "U": U,
                "S_ac": sorted([*U, a, c]), "S_bc": sorted([*U, b, c]),
                "S_ad": sorted([*U, a, d]), "S_bd": sorted([*U, b, d]),
            })

    # ---------------------------------------------------------------- verification
    problems = []
    for i, cs in enumerate(comparisons):
        a, b, c, d = cs["quadruple"]
        if len({a, b, c, d}) != 4:
            problems.append(f"cmp {i}: candidates not distinct")
        if set(cs["U"]) & {a, b, c, d}:
            problems.append(f"cmp {i}: candidate inside U")
        if len(cs["U"]) != K_SEEDS - 2:
            problems.append(f"cmp {i}: |U| = {len(cs['U'])}, expected {K_SEEDS - 2}")
        for key in ("S_ac", "S_bc", "S_ad", "S_bd"):
            if len(cs[key]) != K_SEEDS or len(set(cs[key])) != K_SEEDS:
                problems.append(f"cmp {i}: {key} wrong size/uniqueness")
        if (r2[a] | r2[b]) & (r2[c] | r2[d]):
            problems.append(f"cmp {i}: original two-hop condition violated")
    keys = [tuple(sorted(cs["quadruple"])) for cs in comparisons]
    if len(set(keys)) != len(selected):
        problems.append("quadruple repeated across backgrounds (should appear exactly twice)")
    if len(comparisons) != 2 * len(selected):
        problems.append(f"expected {2*len(selected)} comparisons, got {len(comparisons)}")

    print(f"\n  comparisons built: {len(comparisons)} "
          f"({len(selected)} quadruples x 2 backgrounds)")
    print(f"  verification problems: {len(problems)}")
    for p in problems[:10]:
        print(f"    - {p}")

    art = {
        "task": "N02 -- targeted reversal search retaining the local-indistinguishability condition",
        "step": "2: structure-only quadruple selection, frozen before any diffusion",
        "graph": {"manifest": manifest, "n_lcc": len(lcc),
                  "node_mapping": "ascending original id -> contiguous index 0..n-1"},
        "selection": {
            "sampling_seed": SAMPLING_SEED,
            "max_attempts": MAX_ATTEMPTS,
            "attempts_used": attempts,
            "target_quadruples": TARGET_QUADRUPLES,
            "selected_quadruples": len(selected),
            "pool_shortfall": pool_shortfall,
            "condition_relaxed": False,
            "rejected_prior_use": rejected_prior,
            "rejected_on_original_condition": rejected_condition,
            "passed_condition_q3_zero": rejected_q3_zero,
            "passed_condition_q3_positive": len(accepted),
            "r2_convention": "includes the source node; forward message-passing direction",
            "q3_formula": "abs(|R3(a)nR3(c)| + |R3(b)nR3(d)| - |R3(a)nR3(d)| - |R3(b)nR3(c)|)",
            "q3_note": "structural screen only; not a spread estimate and not evidence about learning",
            "tie_break": "Q_3 descending, then the node-index tuple",
        },
        "backgrounds": {
            "uniform_U": "8 nodes drawn uniformly from the LCC excluding the candidates",
            "degree_U": "the 8 highest-degree nodes excluding candidates, ties by ascending index",
            "global_degree_top8": top8_global,
            "background_rng_seed": SAMPLING_SEED + 1,
        },
        "verification": {"problems": problems, "n_problems": len(problems)},
        "comparisons": comparisons,
    }
    out = HERE / "N02_stage2_selection.json"
    out.write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"\n  wrote {out}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
