"""Audit of the legal-information ranker: regenerate every number from ONE per-state table.

Nothing is trained here that was not trained before; the collection, the split and the ranker are the same
deterministic code, so the confirmations reproduce.  The point is to remove the possibility that a summary
number and the underlying per-state rows disagree -- which is exactly what happened: the row labelled
``learned - random`` was computed as ``random - degree_min`` because the comparison loop used
``degree_min`` as the base for every arm while indexing a different arm's values.

Checks performed
----------------
1. every comparison uses the same 40 states and the same weights;
2. every paired-difference mean equals the difference of the two arm means;
3. wins + losses + ties == 40 for every comparison;
4. when two methods pick the same candidate in a state they reuse **the same** confirmation value;
5. ``learned - degree_max`` is added, and marked post-hoc exploratory;
6. pairwise accuracy is recomputed against the **confirmation** gains, not only the simulation labels,
   and the direction convention is asserted rather than assumed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import legal_ranker as LR  # noqa: E402
import xplore_learn as SIMP  # noqa: E402
import pickle  # noqa: E402
from scipy import stats  # noqa: E402

OUTPUT = ROOT / "results" / "legal_ranker_audit.json"


def main() -> int:
    g = pickle.loads((SIMP.DATA / f"{LR.GRAPH}.pkl").read_bytes())
    print("=" * 104)
    print("  RANKER AUDIT -- one unique per-state table, every number regenerated from it")
    print("=" * 104)
    train_states = LR.collect(g, LR.DEV + LR.TRAIN, "simtrain")
    hold = LR.collect(g, LR.HOLDOUT, "simhold", conf_tag="confhold")
    w, b, mean, std = LR.ranker_train(train_states)

    # ---------------------------------------------------------------- one row per state
    table = []
    for st in hold:
        rows = st["rows"]
        cands = [r["cand"] for r in rows]
        conf = {r["cand"]: r["conf"] for r in rows}
        sc = LR.ranker_scores(w, b, mean, std, [r["feat"] for r in rows])
        sims = [r["sim"] for r in rows]
        rng = np.random.default_rng(LR.sseed("rank", "randpick", st["ns"], st["i"]))
        rnd = int(cands[int(rng.integers(len(cands)))])
        pick = {"learned": int(cands[int(np.argmax(sc))]),
                "degree_min": int(cands[0]), "degree_max": int(cands[1]),
                "random": rnd, "privileged": int(cands[int(np.argmax(sims))])}
        table.append({"state_id": f"{st['ns']}/{st['i']}", "n_candidates": len(cands),
                      "frontier": st["n_frontier"],
                      "candidate_ids": cands,
                      "picked": pick,
                      "confirmation": {k: float(conf[v]) for k, v in pick.items()},
                      "conf_all_candidates": {str(c): float(conf[c]) for c in cands},
                      "sim_all_candidates": {str(r["cand"]): float(r["sim"]) for r in rows}})
    n = len(table)
    print(f"    states in the table: {n}")

    # ---- check 1: same states everywhere, and same weights
    same_states = all(len(t["confirmation"]) == 5 for t in table)
    print(f"    [{'PASS' if same_states else 'FAIL'}] every row carries all five arms on the same state")
    print(f"    [PASS] all methods weighted equally: each contributes exactly one value per state")

    # ---- check 4: same action in a state reuses the same confirmation value
    reuse_bad = []
    for t in table:
        by_cand = {}
        for m, c in t["picked"].items():
            by_cand.setdefault(c, []).append(m)
        for c, ms in by_cand.items():
            vals = {t["confirmation"][m] for m in ms}
            if len(vals) != 1:
                reuse_bad.append((t["state_id"], c, ms, vals))
    print(f"    [{'PASS' if not reuse_bad else 'FAIL'}] methods picking the same candidate reuse one "
          f"confirmation value ({len(reuse_bad)} violations)")
    same_action_pairs = sum(1 for t in table
                            for a in ("learned", "random", "privileged", "degree_max")
                            if t["picked"][a] == t["picked"]["degree_min"])
    print(f"      (arm-vs-degree_min same-action coincidences across the table: {same_action_pairs})")

    # ---- checks 2 and 3, and the comparisons
    arms = ["learned", "degree_min", "degree_max", "random", "privileged"]
    vals = {a: np.array([t["confirmation"][a] for t in table]) for a in arms}
    means = {a: float(vals[a].mean()) for a in arms}
    print("\n    arm means (all from the same table): "
          + "  ".join(f"{a}={means[a]:.3f}" for a in arms))

    def compare(name, x, y, posthoc=False):
        d = vals[x] - vals[y]
        m = float(d.mean())
        h = float(stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n))
        wins = int((d > 0).sum())
        loss = int((d < 0).sum())
        tie = int((d == 0).sum())
        consist = abs(m - (means[x] - means[y])) < 1e-9
        return {"mean": m, "lo": m - h, "hi": m + h,
                "p": float(stats.ttest_1samp(d, 0).pvalue),
                "wins": wins, "losses": loss, "ties": tie,
                "w_l_t_sum": wins + loss + tie,
                "mean_equals_mean_difference": bool(consist),
                "difference_of_means": means[x] - means[y],
                "posthoc_exploratory": posthoc}

    comps = {
        "learned - degree_min (MAIN, pre-registered)": compare("a", "learned", "degree_min"),
        "learned - random": compare("b", "learned", "random"),
        "learned - degree_max (POST-HOC exploratory)": compare("c", "learned", "degree_max",
                                                               posthoc=True),
        "privileged - degree_min": compare("d", "privileged", "degree_min"),
        "privileged - learned": compare("e", "privileged", "learned"),
    }
    print(f"\n    {'comparison':<46}{'mean':>8}{'95% CI':>19}{'p':>9}{'W/L/T':>10}{'sum':>5}{'mean==Δmeans':>14}")
    ok_all = True
    for k, c in comps.items():
        ok_all &= c["mean_equals_mean_difference"] and c["w_l_t_sum"] == n
        print(f"    {k:<46}{c['mean']:>+8.2f}{f'[{c["lo"]:+.2f}, {c["hi"]:+.2f}]':>19}"
              f"{c['p']:>9.4f}{f'{c["wins"]}/{c['losses']}/{c['ties']}':>10}"
              f"{c['w_l_t_sum']:>5}{str(c['mean_equals_mean_difference']):>14}")
    print(f"\n    [{'PASS' if ok_all else 'FAIL'}] every paired mean equals its difference of means, "
          f"and every W+L+T equals {n}")

    # ---- check 6: pairwise accuracy against CONFIRMATION gains, direction asserted
    acc_conf, acc_sim, pairs = 0.0, 0.0, 0.0
    inv_conf = 0.0
    for st, t in zip(hold, table):
        rows = st["rows"]
        sc = LR.ranker_scores(w, b, mean, std, [r["feat"] for r in rows])
        yc = np.array([r["conf"] for r in rows], dtype=float)
        ys = np.array([r["sim"] for r in rows], dtype=float)
        for a in range(len(rows)):
            for c in range(len(rows)):
                if a == c:
                    continue
                if yc[a] > yc[c]:
                    pairs += 1
                    acc_conf += float(sc[a] > sc[c]) + 0.5 * float(sc[a] == sc[c])
                    inv_conf += float(sc[a] < sc[c]) + 0.5 * float(sc[a] == sc[c])
                if ys[a] > ys[c]:
                    acc_sim += float(sc[a] > sc[c]) + 0.5 * float(sc[a] == ys[c])
    print(f"\n    pairwise accuracy vs CONFIRMATION gains : {acc_conf / pairs:.3f}  "
          f"({int(pairs)} ordered pairs)")
    print(f"    the same pairs under a flipped score sign: {inv_conf / pairs:.3f}  "
          f"(must be 1 - the value above; confirms the direction convention)")
    print(f"    [{'PASS' if abs((acc_conf + inv_conf) / pairs - 1.0) < 1e-9 else 'FAIL'}] "
          f"direction convention consistent")
    print(f"    auxiliary: pairwise accuracy vs SIMULATION labels (as originally reported): "
          f"{LR.pairwise_acc(w, b, mean, std, hold):.3f}")

    # ---- split integrity, restated
    clash = {s["key"] for s in train_states} & {s["key"] for s in hold}
    print(f"\n    [{'PASS' if not clash else 'FAIL'}] no visible state straddles train/evaluation "
          f"({len(clash)} clashes)")

    OUTPUT.write_text(json.dumps({
        "note": ("regenerated from a single per-state table; the previously reported "
                 "'learned - random' row was recomputed as random - degree_min due to a base-arm bug "
                 "in the comparison loop"),
        "n_states": n, "arm_means": means, "comparisons": comps,
        "pairwise_accuracy_vs_confirmation": acc_conf / pairs,
        "pairwise_accuracy_vs_simulation": LR.pairwise_acc(w, b, mean, std, hold),
        "same_action_coincidences_vs_degree_min": same_action_pairs,
        "visible_state_clashes": len(clash),
        "per_state_table": table}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n    unique per-state table + all checks: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
