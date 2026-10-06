#!/usr/bin/env python
"""
Stage 06 diagnostics for the report.

    .venv-hs/Scripts/python.exe stage06_diag.py

Prints:
  * matched-random overlap distribution (per config, per state), plus the
    fraction of DEPENDENCY-24's own members that a matched-random set reproduces;
  * release-set content: how many released cells are in the recovery window and how
    many carry a positive shortage score, by scale;
  * where each config declined to move at all (selected == repair) and where it moved;
  * per-run flat list written to stage06_per_run.csv.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict

from lsp_gen import apply_disruption, build_instance, disruptions_for
from lsp_neighborhood import (build_dependency_24, legal_candidates, rank_shortage,
                              rank_window, shortage_scores)
from stage06_run import solution_from_record

CONFIGS = ("FULL", "EMPTY", "SHORTAGE-24", "WINDOW-24", "DEPENDENCY-24",
           "MATCHED-RANDOM-1", "MATCHED-RANDOM-2", "MATCHED-RANDOM-3")


def main() -> int:
    with open("stage06_runs.json", encoding="utf-8") as fh:
        D = json.load(fh)
    runs = D["runs"]
    with open("stage06_states.json", encoding="utf-8") as fh:
        ST = json.load(fh)

    # ---------------------------------------------------------------- overlap
    print("=== matched-random overlap with DEPENDENCY-24 ===")
    print("  overlap_fraction is |R and D| / 24")
    by_cfg = defaultdict(list)
    for r in runs:
        mi = r.get("match_info") or {}
        if mi.get("overlap_fraction") is not None:
            by_cfg[r["config"]].append((r["state"], float(mi["overlap_fraction"]),
                                       mi.get("profile_preserved")))
    for c in CONFIGS:
        if c not in by_cfg:
            continue
        vals = [v for _, v, _ in by_cfg[c]]
        prof = sum(1 for _, _, p in by_cfg[c] if p)
        print("  %-16s n=%-3d mean=%.3f min=%.3f max=%.3f  distinct_set_sizes=%s  profile_ok=%d/%d"
              % (c, len(vals), sum(vals) / len(vals), min(vals), max(vals),
                 sorted({round(v * 24) for v in vals}), prof, len(by_cfg[c])))
    print()
    print("  per-state overlap (MATCHED-RANDOM-1):")
    for state, ov, _ in sorted(by_cfg.get("MATCHED-RANDOM-1", []))[::4]:
        print("    %-30s %.3f" % (state, ov))
    print()

    # ---------------------------------------------------------------- release content
    print("=== release-set composition by scale ===")
    print("  top24 = share drawn from the global top-24 shortage ranks")
    print("  win   = share inside the recovery window;  zero = share with shortage score 0")
    print("  %-16s %-7s %6s %7s %7s %7s %9s"
          % ("config", "scale", "n", "top24", "win", "zero", "mean_rank"))
    comp = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])
    dep_log_acc = defaultdict(list)
    for state, rec in sorted(ST.items()):
        inst, _ = build_instance(rec["meta"]["scale"], rec["meta"]["rho"], rec["meta"]["seed"])
        dis = next(d for d in disruptions_for(inst) if d.name == rec["disruption"]["name"])
        pert = apply_disruption(inst, dis)
        repair = solution_from_record(pert, rec["repair"])
        tau = rec["meta"]["tau"]
        legal = legal_candidates(pert, tau)
        sc = shortage_scores(pert, repair, tau)
        s_order = rank_shortage(pert, repair, tau, legal)
        top24 = set(s_order[:24])
        rank_of = {u: k for k, u in enumerate(s_order)}
        win = set(rank_window(pert, repair, tau, legal, dis)[:24])
        dep, dep_log = build_dependency_24(pert, repair, tau, legal, dis)
        dep_log_acc[rec["meta"]["scale"]].append(
            (dep_log["n_dependency_added"], dep_log["n_shortage_filled"]))
        sets = {"SHORTAGE-24": set(s_order[:24]), "WINDOW-24": win, "DEPENDENCY-24": set(dep)}
        for cfg, rel in sets.items():
            n = len(rel)
            key = (cfg, rec["meta"]["scale"])
            comp[key][0] += 1
            comp[key][1] += len(rel & top24) / n
            comp[key][2] += len(rel & win) / n
            comp[key][3] += sum(1 for u in rel if sc[u] <= 1e-9) / n
            comp[key][4] += sum(rank_of[u] for u in rel) / n
    for key in sorted(comp):
        n, t24, w, z, rk = comp[key]
        print("  %-16s %-7s %6d %7.3f %7.3f %7.3f %9.2f"
              % (key[0], key[1], n, t24 / n, w / n, z / n, rk / n))
    print()
    print("  DEPENDENCY-24 construction (mean over states):")
    for scale, vals in sorted(dep_log_acc.items()):
        print("    %-7s dependency-added=%.2f  shortage-filled=%.2f  (of 24, 8 are seeds)"
              % (scale, sum(v[0] for v in vals) / len(vals),
                 sum(v[1] for v in vals) / len(vals)))
    print()

    # ---------------------------------------------------------------- flat vs moved
    print("=== runs where the released neighbourhood produced NO improvement over the repair ===")
    print("  (selected_cost == repair_cost: the solver either did not move or only moved "
          "at equal cost)")
    print("  %-16s %8s %8s   %s" % ("config", "flat", "of", "flat states lump by scale"))
    for c in CONFIGS:
        sub = [r for r in runs if r["config"] == c]
        flat = [r for r in sub if abs(float(r["selected_cost"]) - float(r["repair_cost"])) < 1e-9]
        sc = defaultdict(int)
        for r in flat:
            sc[r["state"].split("_")[0]] += 1
        print("  %-16s %8d %8d   medium=%d large=%d"
              % (c, len(flat), len(sub), sc.get("medium", 0), sc.get("large", 0)))
    print()

    print("=== selected_source distribution ===")
    for c in CONFIGS:
        sub = [r for r in runs if r["config"] == c]
        src = defaultdict(int)
        for r in sub:
            src[r["selected_source"]] += 1
        print("  %-16s %s" % (c, dict(src)))
    print()

    print("=== timing (budget %.1f s) ===" % D["meta"]["budget_s"])
    tot = [r["timing"]["total_s"] for r in runs]
    ovr = [r["timing"]["overrun_s"] for r in runs]
    print("  total_s: mean=%.3f max=%.3f | overrun: mean=%.4f max=%.3f | overruns>0.1s: %d/%d"
          % (sum(tot) / len(tot), max(tot), sum(ovr) / len(ovr), max(ovr),
             sum(1 for v in ovr if v > 0.1), len(ovr)))
    print()

    # ---------------------------------------------------------------- per-run csv
    cols = ["state", "instance", "scale", "rho", "inst_seed", "disruption", "config", "seed",
            "repair_cost", "selected_cost", "selected_source", "raw_objective", "raw_valid",
            "n_released", "start_adopted", "start_reported_objective", "improved_over_repair",
            "gain_vs_repair_frac", "raw_flips_vs_repair", "selected_flips", "mip_gap",
            "highs_status", "total_s", "overrun_s", "overlap_fraction"]
    rows = []
    for r in sorted(runs, key=lambda z: (z["state"], z["config"], z["seed"])):
        inst = r["state"].split("|")[0]
        mi = r.get("match_info") or {}
        jr = float(r["repair_cost"])
        sc = float(r["selected_cost"])
        rows.append({
            "state": r["state"], "instance": inst, "scale": r["scale"], "rho": r["rho"],
            "inst_seed": r["inst_seed"], "disruption": r["disruption"],
            "config": r["config"], "seed": r["seed"],
            "repair_cost": "%.10g" % jr, "selected_cost": "%.10g" % sc,
            "selected_source": r["selected_source"],
            "raw_objective": "%.10g" % r["raw_objective"] if r["raw_objective"] is not None else "",
            "raw_valid": r["raw_valid"], "n_released": r["n_released"],
            "start_adopted": r["start_adopted"],
            "start_reported_objective": ("%.10g" % r["start_reported_objective"]
                                         if r["start_reported_objective"] is not None else ""),
            "improved_over_repair": r["improved_over_repair"],
            "gain_vs_repair_frac": "%.6f" % ((jr - sc) / max(1.0, abs(jr))),
            "raw_flips_vs_repair": r["raw_flips_vs_repair"],
            "selected_flips": r["selected_stability_flips"],
            "mip_gap": ("%.6f" % r["mip_gap"]) if r.get("mip_gap") is not None else "",
            "highs_status": r.get("highs_status", ""),
            "total_s": "%.4f" % r["timing"]["total_s"],
            "overrun_s": "%.4f" % r["timing"]["overrun_s"],
            "overlap_fraction": ("%.4f" % mi["overlap_fraction"]
                                 if mi.get("overlap_fraction") is not None else ""),
        })
    with open("stage06_per_run.csv", "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("wrote stage06_per_run.csv (%d rows)" % len(rows))
    return 0


_WIN_CACHE = {}


def _win(runs, state):
    if state not in _WIN_CACHE:
        _WIN_CACHE[state] = next(r["release"] for r in runs
                                 if r["state"] == state and r["config"] == "WINDOW-24")
    return _WIN_CACHE[state]


if __name__ == "__main__":
    raise SystemExit(main())
