"""Frozen validation on 100 new states: does the trained rule beat degree_max and random?

Nothing is retrained and nothing is added
-----------------------------------------
The ranker is the same deterministic fit on the same 60 states (20 development + 40 training) with the same
single configuration.  It is written to a file with a hash so "frozen" is checkable rather than asserted.
No privileged arm this round: two state batches already support the existence of headroom, and what is
missing is evidence that the *learned* rule beats a strong baseline.

Arm definitions, carried over unchanged
---------------------------------------
======================  ==========================================================================
``learned``             argmax of the frozen ranker over the candidate pool
``degree_max``          the max-visible-degree candidate in the pool
``degree_min``          the min-visible-degree candidate in the pool
``random``              uniform over the **candidate pool** (<= 8), *not* over the whole frontier
======================  ==========================================================================

The random arm sampled the candidate pool in every previous round; this is stated here explicitly so it
cannot drift silently.

Comparisons, fixed before running
---------------------------------
Two **primary**: ``learned - degree_max`` and ``learned - random``, Holm-corrected together.
One **secondary reconfirmation**: ``learned - degree_min``.  Reported 95% intervals are **unadjusted**.

100 states is a cost-control choice, not a power calculation.  No significance peeking, no gain-based
state selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import legal_ranker as LR  # noqa: E402
import xplore_learn as SIMP  # noqa: E402

OUTPUT = ROOT / "results" / "frozen_validation.json"
MODEL = ROOT / "results" / "frozen_ranker.json"
N_STATES = 100
NEW_NS = "vstates"          # new namespace; seeds 3000..3099, touched for the first time here
NEW_FROM = 3000
PRIMARY = ["learned - degree_max", "learned - random"]
SECONDARY = ["learned - degree_min"]
ARMS = ["learned", "degree_max", "degree_min", "random"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", type=int, default=N_STATES)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--model", type=Path, default=MODEL)
    args = ap.parse_args()
    g = pickle.loads((SIMP.DATA / f"{LR.GRAPH}.pkl").read_bytes())

    print("=" * 104)
    print("  FROZEN VALIDATION -- 100 new states, four arms, no retraining")
    print("=" * 104)

    # ---------------------------------------------------------------- freeze
    t0 = time.perf_counter()
    train_states = LR.collect(g, LR.DEV + LR.TRAIN, "simtrain")
    w, b, mean, std = LR.ranker_train(train_states)
    args.model.write_text(json.dumps({
        "features": LR.FEATURES, "weights": [float(x) for x in w], "bias": float(b),
        "standardiser_mean": [float(x) for x in mean], "standardiser_std": [float(x) for x in std],
        "config": {"L2": LR.L2, "lr": LR.LR, "epochs": LR.EPOCHS},
        "trained_on": {"dev": 20, "train": 40, "namespaces": ["valprobe", "rstates"]},
        "candidate_pool_max": LR.N_CAND, "candidate_rule": "degree_min + degree_max + fixed-seed fill",
        "tie_break": "sorted(frontier) then fixed RNG within a degree tie",
        "random_arm_scope": "candidate pool (<=8), NOT the full frontier",
        "state_rule": "5 fixed free seeds + 4 uniform-random-frontier surveys, fixed RNG streams",
        "budgets": {"sim_samples": LR.SIM_SAMPLES, "conf_samples": LR.CONF_SAMPLES}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    frozen model: {args.model.name}  sha256 {sha(args.model)[:16]}...")
    print(f"    frozen standardiser mean[0..2]={np.round(mean[:3], 4)}  "
          f"std[0..2]={np.round(std[:3], 4)}  (fit on the 60 training states, reused unchanged)")
    print(f"    frozen feature set: {LR.FEATURES}")

    # ---------------------------------------------------------------- new states
    print(f"\n    building {args.states} new states under namespace {NEW_NS}/{NEW_FROM}.."
          f"{NEW_FROM + args.states - 1}")
    known = {s["key"] for s in train_states}
    t_build = time.perf_counter()
    states, dup_new, dup_known = [], 0, 0
    seen = set()
    for k in range(args.states):
        i = NEW_FROM + k
        st = LR.build_state(g, NEW_NS, i)
        st["ns"], st["i"] = NEW_NS, i          # build_state does not carry these; collect() adds them
        st["key"] = LR.visible_key(st["env"])
        if st["key"] in known:
            dup_known += 1
        if st["key"] in seen:
            dup_new += 1
        seen.add(st["key"])
        cands = LR.candidates(st["env"], NEW_NS, i)
        if not cands:
            continue
        rows = []
        for c in cands:
            ch = LR.rollout(st["env"], c)
            rows.append({"cand": int(c), "feat": LR.features(st["env"], st, c), "chosen": ch})
        st["rows"] = rows
        states.append(st)
    print(f"    built {len(states)} usable states in {time.perf_counter() - t_build:.0f} s")
    print(f"    [{'PASS' if dup_known == 0 else 'FAIL'}] no new state repeats a training/dev state "
          f"({dup_known} repeats)")
    print(f"    [{'PASS' if dup_new == 0 else 'FAIL'}] no two new states are the same visible state "
          f"({dup_new} repeats)")
    print(f"    one last-step state per independent initial observation (by construction)")

    # ---------------------------------------------------------------- arms pick, then confirm
    t_decide = time.perf_counter()
    picked, cand_ids = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for st in states:
        rows = st["rows"]
        cands = [r["cand"] for r in rows]
        f = [r["feat"] for r in rows]
        s_sc = LR.ranker_scores(w, b, mean, std, f)          # feature construction + scoring
        rng = np.random.default_rng(LR.sseed("frozen", "randpick", st["ns"], st["i"]))
        pick = {"learned": int(cands[int(np.argmax(s_sc))]),
                "degree_max": int(cands[1]), "degree_min": int(cands[0]),
                "random": int(cands[int(rng.integers(len(cands)))])}
        for a, v in pick.items():
            picked[a].append(v)
            cand_ids[a].append(v)
    decide_seconds = time.perf_counter() - t_decide

    t_conf = time.perf_counter()
    conf = {a: [] for a in ARMS}
    for si, st in enumerate(states):
        # one confirmation value per candidate, then every arm reads it -- same action, same value
        cmap = {r["cand"]: LR.score(g, r["chosen"], f"frozen:{st['ns']}{st['i']}:{r['cand']}",
                                    LR.CONF_SAMPLES) for r in st["rows"]}
        for a in ARMS:
            conf[a].append(cmap[cand_ids[a][si]])
    conf_seconds = time.perf_counter() - t_conf
    n = len(states)

    # ---------------------------------------------------------------- comparisons
    vals = {a: np.array(conf[a]) for a in ARMS}
    means = {a: float(vals[a].mean()) for a in ARMS}

    def compare(x, y):
        d = vals[x] - vals[y]
        m = float(d.mean())
        h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
        return {"mean": m, "lo_unadjusted": m - h, "hi_unadjusted": m + h,
                "p_raw": float(stats.ttest_1samp(d, 0).pvalue),
                "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum()),
                "per_state": [float(x) for x in d]}

    res = {}
    for name in PRIMARY + SECONDARY:
        x, y = name.split(" - ")
        res[name] = compare(x, y)
    # Holm over the two primaries
    ordered = sorted(PRIMARY, key=lambda k: res[k]["p_raw"])
    prev = 0.0
    for j, k in enumerate(ordered):
        adj = min(1.0, max(prev, (len(ordered) - j) * res[k]["p_raw"]))
        res[k]["p_holm"] = adj
        prev = adj
    for k in SECONDARY:
        res[k]["p_holm"] = None

    print("\n" + "=" * 104)
    print(f"  RESULTS -- {n} new states, independent confirmation batch ({LR.CONF_SAMPLES} samples)")
    print("=" * 104)
    print(f"    arm means: " + "  ".join(f"{a}={means[a]:.2f}" for a in ARMS))
    print(f"\n    {'comparison':<26}{'mean':>8}{'95% CI (unadjusted)':>24}{'p_raw':>9}"
          f"{'p_holm':>9}{'W/L/T':>10}")
    for name in PRIMARY + SECONDARY:
        r = res[name]
        tag = " [PRIMARY]" if name in PRIMARY else " [secondary]"
        ph = f"{r['p_holm']:.4f}" if r["p_holm"] is not None else "-"
        print(f"    {name + tag:<26}{r['mean']:>+8.2f}"
              f"{f'[{r["lo_unadjusted"]:+.2f}, {r["hi_unadjusted"]:+.2f}]':>24}"
              f"{r['p_raw']:>9.4f}{ph:>9}{f'{r["wins"]}/{r["losses"]}/{r["ties"]}':>10}")

    dec_ok = all(res[k]["mean"] > 0 and res[k]["p_holm"] is not None and res[k]["p_holm"] < 0.05
                 for k in PRIMARY)
    one_ok = any(res[k]["mean"] > 0 and res[k]["p_holm"] is not None and res[k]["p_holm"] < 0.05
                 for k in PRIMARY)
    branch = ("both primaries positive after correction -> worth cross-graph validation; still no DQN"
              if dec_ok else
              "only one primary positive -> mixed result, no overall advantage, no automatic training"
              if one_ok else
              "neither primary established -> stop extending this ranker; keep the degree_min result but "
              "do not treat it as an established contribution, and do not read it as 'legal information "
              "is useless'")
    print(f"\n    BRANCH: {branch}")

    print("\n  DEPLOYMENT COST (separate from confirmation)")
    print(f"    decision time incl. feature construction + scoring + selection: "
          f"{decide_seconds:.2f} s total, {1000 * decide_seconds / n:.2f} ms per state")
    print(f"    final influence confirmation: {conf_seconds:.1f} s total "
          f"({conf_seconds / n:.2f} s per state) -- reported separately, never part of decision cost")

    args.output.write_text(json.dumps({
        "frozen_model_sha256": sha(args.model), "n_states": n, "arm_means": means,
        "comparisons": res, "primary": PRIMARY, "secondary": SECONDARY,
        "random_arm_scope": "candidate pool (<=8), NOT the full frontier",
        "intervals_note": "95% intervals are UNADJUSTED for multiple comparisons",
        "state_namespace": f"{NEW_NS}/{NEW_FROM}..{NEW_FROM + args.states - 1}",
        "duplicates": {"vs_training": dup_known, "within_new": dup_new},
        "cost": {"decision_seconds_total": decide_seconds,
                 "decision_ms_per_state": 1000 * decide_seconds / n,
                 "confirmation_seconds_total": conf_seconds,
                 "confirmation_s_per_state": conf_seconds / n},
        "branch": branch}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n    artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
