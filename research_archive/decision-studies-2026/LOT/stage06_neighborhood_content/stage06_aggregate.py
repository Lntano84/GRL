#!/usr/bin/env python
"""
Stage 06 aggregation, corrected.

The first version of the analysis aggregated like this:

    for state, by_cfg in acc.items():          # state = "instance|disruption"
        name = state.split("|")[0]
        for cfg, vals in by_cfg.items():
            inst[name][cfg] = sum(vals) / len(vals)   # <-- OVERWRITES per disruption
        inst_jr[name] = jr[state]                     # <-- OVERWRITES per disruption

For an instance with two disruptions the second disruption silently replaced the first, so the
"instance mean" never averaged the two disruptions, and the normaliser `J_r` was whichever
disruption happened to be visited last -- even though `J_r` differs between the two disruptions
in 16/16 instances (by 18%-132%).

The correct instance statistic pairs first and aggregates second:

    G_n(a,b) = (1/4) * sum_{d in {D1,D2}} sum_{s in {0,1}}
                   (J_{n,d,s,b} - J_{n,d,s,a}) / max(1, |J_r(n,d)|)

This module is the single source of truth for that definition; `stage06_analyse.py`,
`stage06_verdict.py` and `stage06_tables.py` all import it.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

CONFIGS = ("FULL", "EMPTY", "SHORTAGE-24", "WINDOW-24", "DEPENDENCY-24",
           "MATCHED-RANDOM-1", "MATCHED-RANDOM-2", "MATCHED-RANDOM-3")
RANDOMS = ("MATCHED-RANDOM-1", "MATCHED-RANDOM-2", "MATCHED-RANDOM-3")
THRESHOLD = 0.02          # engineering screen for this round, NOT a literature standard
SIMPLE_PICK = {"medium": "FULL", "large": "EMPTY"}


def load_runs(path: str = "stage06_runs.json") -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["runs"]


def pair_cells(runs) -> Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]]:
    """(instance, disruption, solver seed) -> config -> {cost, repair, source, ...}."""
    out: Dict[Tuple[str, str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for r in runs:
        inst, dis = r["state"].split("|")
        out[(inst, dis, int(r["seed"]))][r["config"]] = {
            "cost": float(r["selected_cost"]),
            "repair": float(r["repair_cost"]),
            "raw": (None if r.get("raw_objective") is None else float(r["raw_objective"])),
            "source": r["selected_source"],
            "improved": bool(r["improved_over_repair"]),
        }
    return out


def scale_of(instance: str) -> str:
    return instance.split("_")[0]


def scale_rho(instance: str) -> Tuple[str, str]:
    """`large_rho0.75_s2` -> ('large', '0.75').

    The rho is at split index 1; index 2 is the INSTANCE SEED. Grouping on the seed by mistake
    silently averages the group over rho (this was a real bug in the first analysis pass).
    """
    p = instance.split("_")
    return (p[0], p[1].replace("rho", ""))


def instance_g(cells, a: str, b: str) -> Dict[str, float]:
    """G_n(a,b): positive => a is cheaper than b.  Pair first, aggregate second."""
    acc: Dict[str, List[float]] = defaultdict(list)
    for (inst, dis, seed), d in cells.items():
        if a not in d or b not in d:
            continue
        ja, jb = d[a]["cost"], d[b]["cost"]
        jr = d[a]["repair"]                       # J_r depends on the disruption
        acc[inst].append((jb - ja) / max(1.0, abs(jr)))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def instance_g_random_mean(cells, a: str, randoms=RANDOMS) -> Dict[str, float]:
    """G_n(a, mean of the matched-random configs).

    The three random configs are averaged INSIDE each (disruption, seed) pair, so the comparison
    is paired on disruption, solver seed and repair normaliser.
    """
    acc: Dict[str, List[float]] = defaultdict(list)
    for (inst, dis, seed), d in cells.items():
        if a not in d:
            continue
        vals = [d[c]["cost"] for c in randoms if c in d]
        if not vals:
            continue
        rnd = sum(vals) / len(vals)
        acc[inst].append((rnd - d[a]["cost"]) / max(1.0, abs(d[a]["repair"])))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def instance_g_simple(cells, a: str, pick: Dict[str, str] = None) -> Dict[str, float]:
    """G_n(a, frozen simple strategy): medium -> FULL, large -> EMPTY."""
    pick = pick or SIMPLE_PICK
    inst_b = {inst: pick[scale_of(inst)] for (inst, _d, _s) in cells}
    acc: Dict[str, List[float]] = defaultdict(list)
    for (inst, dis, seed), d in cells.items():
        b = inst_b[inst]
        if a not in d or b not in d:
            continue
        acc[inst].append((d[b]["cost"] - d[a]["cost"]) / max(1.0, abs(d[a]["repair"])))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def instance_g_by_seed(cells, a: str, b_provider, seed: int) -> Dict[str, float]:
    """Same as above but restricted to ONE solver seed, so seed stability can be tested."""
    sub = {k: v for k, v in cells.items() if k[2] == seed}
    return b_provider(sub, a) if b_provider is instance_g_random_mean else b_provider(sub, a)


def describe(vals: List[float]) -> Dict[str, Any]:
    sv = sorted(vals)
    n = len(sv)
    med = sv[n // 2] if n % 2 else (sv[n // 2 - 1] + sv[n // 2]) / 2
    return {"n": n, "mean": sum(sv) / n, "median": med, "min": sv[0], "max": sv[-1],
            "pos": sum(1 for v in sv if v > 1e-9),
            "neg": sum(1 for v in sv if v < -1e-9),
            "zero": sum(1 for v in sv if abs(v) <= 1e-9),
            "ge2pct": sum(1 for v in sv if v >= THRESHOLD),
            "le_neg2pct": sum(1 for v in sv if v <= -THRESHOLD)}


def onemetric(cells, a: str, b: str) -> Dict[str, Any]:
    """describe() plus the scale / scale-rho breakdown, for a two-config comparison."""
    v = instance_g(cells, a, b)
    out = describe(list(v.values()))
    out["per_instance"] = sorted(v.items())
    for key, sel in (("by_scale", lambda n: scale_of(n)),
                     ("by_scale_rho", lambda n: "%s|%s" % scale_rho(n))):
        g: Dict[str, List[float]] = defaultdict(list)
        for n, x in v.items():
            g[sel(n)].append(x)
        out[key] = {k: {"n": len(x), "mean": sum(x) / len(x)} for k, x in sorted(g.items())}
    return out
