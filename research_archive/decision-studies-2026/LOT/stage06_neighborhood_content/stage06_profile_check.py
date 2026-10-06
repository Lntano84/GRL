#!/usr/bin/env python
"""
Stage 06 matched-random profile check, corrected.

    .venv-hs/Scripts/python.exe stage06_profile_check.py

`lsp_neighborhood._profile` keyed the (machine, period, repair-value) profile on the RAW float
value of the repaired Y.  The stored repaired plans are binary in exact arithmetic but not in
floating point: a single state yields six distinct values where two classes exist, e.g.

    -3.7263617956233694e-16, 0.0, 0.9999999999999998, 0.9999999999999999, 1.0, 1.0000000000000002

so a cell whose repaired Y is -3.7e-16 lands in a DIFFERENT stratum from an otherwise identical
cell whose repaired Y is exactly 0.0.  That fragments the strata, which makes `match_random`
see an undersized pool and fall back to "keep whatever is available", and it also makes the
`profile_preserved` bit flip to False for reasons that have nothing to do with the release set.

CLASSIFYING on the 0/1 class (threshold 0.5, the same rule the fixing conditions already use)
removes the artefact.  This script re-checks the profile of the ALREADY-RUN release sets; it
never regenerates them, so no historical result is overwritten.

The overlap fraction is recomputed from the stored sets as
    |R and D| / |D|          (D = DEPENDENCY-24's set, |D| = 24)
which is the intersection over the DEPENDENCY set, not over R.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_model import MODE_AUDITED, Solution
from stage06_aggregate import RANDOMS
from stage06_run import solution_from_record

YIndex = Tuple[int, int, int]


def profile_exact(s: Sequence[YIndex], repair: Solution) -> Dict[Any, int]:
    """The ORIGINAL keying: raw float repair value."""
    return dict(Counter((j, t, float(repair.Y[i, j, t - 1])) for (i, j, t) in s))


def profile_class(s: Sequence[YIndex], repair: Solution) -> Dict[Any, int]:
    """Corrected keying: the 0/1 class of the repaired Y (threshold 0.5)."""
    return dict(Counter((j, t, 1 if float(repair.Y[i, j, t - 1]) > 0.5 else 0)
                        for (i, j, t) in s))


def main() -> int:
    with open("stage06_runs.json", encoding="utf-8") as fh:
        runs = json.load(fh)["runs"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)

    # ---- per-state context, built once
    ctx: Dict[str, Any] = {}
    rels: Dict[Tuple[str, str], List[YIndex]] = {}
    for r in runs:
        rels[(r["state"], r["config"])] = [tuple(u) for u in r["release"]]

    print("=" * 96)
    print("=== matched-random profile preservation: exact float keying vs 0/1 class keying ===")
    print("=" * 96)
    tally = defaultdict(lambda: {"exact": 0, "class": 0, "n": 0})
    detail_bad = []
    for r in runs:
        cfg = r["config"]
        if cfg not in RANDOMS:
            continue
        state = r["state"]
        if state not in ctx:
            rec = ST[state]
            inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"],
                                     rec["meta"]["seed"])
            dis = next(d for d in disruptions_for(inst)
                       if d.name == rec["disruption"]["name"])
            pert = apply_disruption(inst, dis)
            ctx[state] = {"pert": pert, "repair": solution_from_record(pert, rec["repair"])}
        C = ctx[state]
        dep = rels[(state, "DEPENDENCY-24")]
        cur = rels[(state, cfg)]
        pe = profile_exact(cur, C["repair"]) == profile_exact(dep, C["repair"])
        pc = profile_class(cur, C["repair"]) == profile_class(dep, C["repair"])
        tally[cfg]["n"] += 1
        tally[cfg]["exact"] += int(pe)
        tally[cfg]["class"] += int(pc)
        if not pc:
            detail_bad.append((state, cfg, r["seed"]))

    print("  %-18s %6s %14s %14s" % ("config", "n", "exact-key ok", "0/1-class ok"))
    for cfg in RANDOMS:
        t = tally[cfg]
        print("  %-18s %6d %14d %14d" % (cfg, t["n"], t["exact"], t["class"]))
    te = sum(t["exact"] for t in tally.values())
    tc = sum(t["class"] for t in tally.values())
    tn = sum(t["n"] for t in tally.values())
    print("  %-18s %6d %14d %14d" % ("TOTAL", tn, te, tc))
    print()
    if detail_bad:
        print("  runs still failing under 0/1-class keying: %d -> %s" % (len(detail_bad),
                                                                         detail_bad[:6]))
    else:
        print("  with 0/1-class keying, ALL %d matched-random runs preserve the DEPENDENCY-24" % tn)
        print("  (machine, period, repair-Y-class) profile.  The earlier 190/192 was a")
        print("  floating-point classification artefact, not an undersized stratum.")
    print()

    # ---- overlap recomputed from the stored sets, both keyings
    print("=" * 96)
    print("=== overlap |R and D| / |D| recomputed from the STORED release sets ===")
    print("  (no release set is regenerated; the historical sets are used as they were run)")
    print("=" * 96)
    print("  %-18s %6s %10s %10s %10s %10s %10s"
          % ("config", "n", "mean", "median", "min", "max", "|R|"))
    for cfg in RANDOMS:
        vals, sizes = [], []
        for r in runs:
            if r["config"] != cfg:
                continue
            D = set(rels[(r["state"], "DEPENDENCY-24")])
            R = set(rels[(r["state"], cfg)])
            vals.append(len(R & D) / len(D))
            sizes.append(len(R))
        sv = sorted(vals)
        med = sv[len(sv) // 2] if len(sv) % 2 else (sv[len(sv) // 2 - 1] + sv[len(sv) // 2]) / 2
        print("  %-18s %6d %10.4f %10.4f %10.4f %10.4f %10s"
              % (cfg, len(vals), sum(vals) / len(vals), med, sv[0], sv[-1],
                 sorted(set(sizes))))
    print()

    # ---- how many distinct profile slots does a DEPENDENCY-24 set actually occupy?
    print("=== why the exact keying fragmented: distinct values in the stored repaired Y ===")
    worst = None
    for state, C in sorted(ctx.items()):
        un = np.unique(np.asarray(C["repair"].Y, float))
        if worst is None or len(un) > worst[1]:
            worst = (state, len(un), un)
    print("  worst state: %s -> %d distinct raw float values" % (worst[0], worst[1]))
    print("  values: %s" % ["%.17g" % v for v in worst[2]])
    print("  classes after thresholding at 0.5: %d"
          % len({1 if float(v) > 0.5 else 0 for v in worst[2]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
