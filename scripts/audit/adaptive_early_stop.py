"""Adaptive early stopping on paired gaps: does a cheap decision rule beat a fixed budget?

The idea under test
-------------------
Candidates that are easy to tell apart should not need the same number of simulations as candidates
that are hard to tell apart.  This script tests one specific, pre-fixed rule:

* candidates: the four configurations' **existing** ``degree`` top-8 -- the screening method is not
  changed, only how long it is allowed to run;
* checkpoints at **300** and **1000** trials, hard stop at **3000**;
* at a checkpoint, let ``b`` be the candidate with the highest mean marginal so far; for every other
  candidate ``v`` form the paired per-trial difference ``d_t = Delta_t(b) - Delta_t(v)`` and compute
  ``mean(d)`` and ``SE(d)``.  Stop and choose ``b`` only if **every** competitor satisfies
  ``mean(d) > 3 * SE(d)``;
* if any ``SE(d)`` is exactly zero, do **not** stop -- a finite sample with no observed variation is
  not evidence of certainty;
* if no checkpoint triggers, run to 3000 and take the highest mean, ties by ascending node id.

The ``3 * SE`` factor is a **pre-fixed heuristic, not a proven confidence bound**.  It carries no
95%-correct-selection guarantee, no other multiple is tried, and no threshold is searched for on
these four configurations.

Zero new simulation
-------------------
Everything is replayed from the 10000 per-trial selection marginals already on disk:

* trials ``[0, 3000)`` are the decision prefix -- a checkpoint may only read the part of it that has
  been revealed;
* trials ``[4000, 10000)`` are the evaluation segment;
* trials ``[3000, 4000)`` are deliberately unused.

Because the whole pool was simulated for all 10000 trials, an early stop that picks a different
candidate needs no re-run: its evaluation data already exists.

**This is an exploratory offline replay on historical data**, not an independent confirmation
experiment.  These trials have already been used for research analysis, and the evaluation segment is
inside the same stream as the decision prefix.

Two constraints are checked rather than argued
----------------------------------------------
1. **The decision must not read the evaluation segment.**  The whole replay is re-run with the
   evaluation segment replaced by garbage, and the choices must be identical.
2. **A non-triggering run must equal fixed-3000 exactly**, since that is what the rule falls back to.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for extra in (ROOT / "src", ROOT / "scripts" / "experiments", ROOT / "scripts" / "audit"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import selection_budget_curve as sbc  # noqa: E402

CHECKPOINTS = (300, 1000)
HARD_STOP = 3000
MULTIPLIER = 3.0
DECISION_TRIALS = 3000
EVAL_START = 4000
EVAL_END = 10000
CANDIDATES_SCREENED = 8

OUTPUT = ROOT / "docs" / "results" / "adaptive_early_stop.json"


def decision_cost(trials: int) -> int:
    """Cascades for one decision: one base run per trial plus one run per candidate screened."""
    return trials * (1 + CANDIDATES_SCREENED)


def load_series(index: int) -> dict[int, list[float]]:
    """Per-trial selection marginals for the 50 candidates, 10000 trials, from the frozen chunks."""
    stored = json.loads(sbc.SOURCE_CONFIG.with_name(sbc.SOURCE_CONFIG.name.format(index=index))
                        .read_text(encoding="utf-8"))
    candidates = [int(c) for c in stored["configuration"]["candidates"]]
    per: dict[int, list[float]] = {c: [] for c in candidates}
    for start, end in sbc.CHUNKS:
        path = sbc.SEL_CHUNK.with_name(sbc.SEL_CHUNK.name.format(index=index, start=start, end=end))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("payload_sha256") != sbc.payload_digest(payload):
            raise SystemExit(f"{path.name} changed after being written")
        if not payload.get("complete"):
            raise SystemExit(f"{path.name} is incomplete")
        for c in candidates:
            per[c].extend(float(v) for v in payload["per_trial_marginals"][str(c)])
    if any(len(per[c]) != EVAL_END for c in candidates):
        raise SystemExit(f"cfg{index}: expected {EVAL_END} trials per candidate")
    return per


def degree_top8(index: int) -> list[int]:
    frozen = json.loads(sbc.SOURCE_CHOICES.with_name(sbc.SOURCE_CHOICES.name.format(index=index))
                        .read_text(encoding="utf-8"))
    return [int(c) for c in frozen["choices"]["degree"]["top8"]]


def checkpoint_report(series: dict[int, list[float]], pool: list[int], trials: int,
                      multiplier: float = MULTIPLIER) -> dict:
    """The leader at ``trials`` and the paired gap to each of its competitors."""
    means = {c: statistics.fmean(series[c][:trials]) for c in pool}
    leader = max(pool, key=lambda c: (means[c], -c))
    competitors = []
    for v in sorted(pool):
        if v == leader:
            continue
        d = [series[leader][t] - series[v][t] for t in range(trials)]
        md = statistics.fmean(d)
        sd = statistics.pstdev(d) if trials > 1 else 0.0
        se = sd / math.sqrt(trials) if trials > 1 else float("inf")
        ratio = md / se if se > 0 and math.isfinite(se) else None
        competitors.append({
            "competitor": v, "mean_difference": md, "se": se, "sd": sd,
            "mean_over_se": ratio,
            "passes": bool(se > 0 and math.isfinite(se) and md > multiplier * se),
        })
    zero_se = [c["competitor"] for c in competitors if not (c["se"] > 0)]
    triggered = bool(competitors) and all(c["passes"] for c in competitors)
    return {
        "trials": trials, "leader": leader, "leader_mean": means[leader],
        "competitors": competitors,
        "competitors_with_zero_se": zero_se,
        "triggered": triggered,
        "blocked_by_zero_se": bool(zero_se) and all(
            c["passes"] for c in competitors if c["se"] > 0),
        "pool_means": {str(c): means[c] for c in sorted(pool)},
    }


def decide(series: dict[int, list[float]], pool: list[int],
           multiplier: float = MULTIPLIER) -> dict:
    """The pre-fixed rule.  Reads only trials ``[0, trials)`` at each checkpoint, never beyond."""
    trail = []
    for cp in CHECKPOINTS:
        rep = checkpoint_report(series, pool, cp, multiplier)
        trail.append(rep)
        if rep["triggered"]:
            return {"early_stopped": True, "stopped_at": cp, "chosen": rep["leader"],
                    "checkpoints": trail, "reason": "every competitor passed mean(d) > 3 SE(d)"}
        if rep["blocked_by_zero_se"]:
            trail[-1]["blocked_note"] = ("would have passed, but at least one competitor had "
                                         "SE(d) = 0, and the rule does not stop on that")
    rep = checkpoint_report(series, pool, HARD_STOP, multiplier)
    trail.append(rep)
    return {"early_stopped": False, "stopped_at": HARD_STOP, "chosen": rep["leader"],
            "checkpoints": trail,
            "reason": "no checkpoint triggered; fell back to the highest mean at 3000"}


def fixed_choice(series: dict[int, list[float]], pool: list[int], trials: int) -> int:
    means = {c: statistics.fmean(series[c][:trials]) for c in pool}
    return max(pool, key=lambda c: (means[c], -c))


def describe(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values)
    var = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n) if n > 1 else float("inf")
    return {"mean": mean, "sd": math.sqrt(var), "se": se, "n": n,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se}


def paired(a: list[float], b: list[float]) -> dict:
    out = describe([x - y for x, y in zip(a, b)])
    se = out["se"]
    out["p_two_sided"] = (2.0 * (1.0 - 0.5 * math.erfc(-abs(out["mean"]) / se / math.sqrt(2.0)))
                          if math.isfinite(se) and se > 0 else 1.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    eval_trials = EVAL_END - EVAL_START
    blocks = []
    for index in range(len(sbc.NEW_SEEDS)):
        series = load_series(index)
        pool = degree_top8(index)

        # ---- decisions first, then evaluation; the decision cannot see the evaluation segment ----
        adaptive = decide(series, pool)
        fixed1000 = fixed_choice(series, pool, 1000)
        fixed3000 = fixed_choice(series, pool, HARD_STOP)

        constraint_fallback = (adaptive["early_stopped"]
                               or adaptive["chosen"] == fixed3000)

        # constraint 1: the decision must be unchanged when the evaluation segment is destroyed
        garbled = {c: list(v) for c, v in series.items()}
        rng = random.Random(20260917 + index)
        for c in pool:
            for t in range(EVAL_START, EVAL_END):
                garbled[c][t] = rng.uniform(-5000.0, 5000.0)
        adaptive_garbled = decide(garbled, pool)
        constraint_blind = (adaptive_garbled["chosen"] == adaptive["chosen"]
                            and adaptive_garbled["stopped_at"] == adaptive["stopped_at"]
                            and [r["leader"] for r in adaptive_garbled["checkpoints"]]
                            == [r["leader"] for r in adaptive["checkpoints"]])

        def evaluate(candidate: int) -> dict:
            return describe(series[candidate][EVAL_START:EVAL_END])

        chosen = {"adaptive": adaptive["chosen"], "fixed1000": fixed1000, "fixed3000": fixed3000}
        gains = {k: evaluate(v) for k, v in chosen.items()}

        def contrast(a: str, b: str) -> dict:
            if chosen[a] == chosen[b]:
                return {"same_choice": True, "chosen": chosen[a],
                        "note": "both strategies selected the SAME candidate; the contrast is exactly "
                                "zero by construction and is not independent evidence"}
            return dict(paired(series[chosen[a]][EVAL_START:EVAL_END],
                               series[chosen[b]][EVAL_START:EVAL_END]),
                        same_choice=False, left=a, right=b,
                        definition=f"{a} minus {b}")

        blocks.append({
            "configuration": index, "random_seed": sbc.NEW_SEEDS[index],
            "degree_top8": pool,
            "adaptive": adaptive,
            "fixed1000": {"chosen": fixed1000, "trials": 1000,
                          "decision_cascades": decision_cost(1000)},
            "fixed3000": {"chosen": fixed3000, "trials": HARD_STOP,
                          "decision_cascades": decision_cost(HARD_STOP)},
            "adaptive_decision_cascades": decision_cost(adaptive["stopped_at"]),
            "same_choice_as_fixed3000": adaptive["chosen"] == fixed3000,
            "evaluation": {"segment": [EVAL_START, EVAL_END], "trials": eval_trials,
                           "gains": gains,
                           "adaptive_minus_fixed3000": contrast("adaptive", "fixed3000"),
                           "adaptive_minus_fixed1000": contrast("adaptive", "fixed1000"),
                           "fixed1000_minus_fixed3000": contrast("fixed1000", "fixed3000")},
            "constraints": {
                "decision_ignores_evaluation_segment": constraint_blind,
                "non_trigger_equals_fixed3000": constraint_fallback,
            },
        })

    n_early = sum(1 for b in blocks if b["adaptive"]["early_stopped"])
    artifact = {
        "script": Path(__file__).name, "code_version": sbc.code_version(),
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "status": "EXPLORATORY OFFLINE REPLAY ON HISTORICAL DATA -- not an independent confirmation "
                  "experiment; the evaluation segment lies inside the same stream as the decision "
                  "prefix and these trials have already been used for research analysis",
        "question": "can a pre-fixed paired-gap early-stop rule decide more cheaply than a fixed 3000 "
                    "trials without losing gain?",
        "rule": {
            "checkpoints": list(CHECKPOINTS), "hard_stop": HARD_STOP, "multiplier": MULTIPLIER,
            "stop_condition": "for EVERY competitor v: mean(d) > multiplier * SE(d), with "
                              "d_t = Delta_t(leader) - Delta_t(v) on the same trial",
            "zero_se_policy": "if any SE(d) is exactly zero, do not stop this round",
            "fallback": "run to 3000, take the highest mean, ties by ascending node id",
            "caveat": "3 SE is a PRE-FIXED HEURISTIC, not a proven confidence bound and not a "
                      "95%-correct-selection guarantee.  No other multiple was tried and no "
                      "threshold was searched on these configurations.",
        },
        "replay": {"decision_prefix": [0, DECISION_TRIALS],
                   "evaluation_segment": [EVAL_START, EVAL_END],
                   "unused": [DECISION_TRIALS, EVAL_START],
                   "new_simulation": "none"},
        "candidate_pool": "the four configurations' existing degree top-8; the screening method is "
                          "unchanged, only the budget is",
        "cost_model": "one base run per trial plus one run per candidate screened; "
                      "9 x stopping trials -- a replay-derived decision cost, NOT a measured speed-up",
        "summary": {"configurations": len(blocks), "early_stopped": n_early,
                    "ran_to_hard_stop": len(blocks) - n_early},
        "configurations": blocks,
        "complete": True,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ four-row table
    print("=" * 120)
    print("  ADAPTIVE EARLY STOP ON PAIRED GAPS -- exploratory offline replay, zero new simulation")
    print("=" * 120)
    print(f"  rule: checkpoints {list(CHECKPOINTS)}, hard stop {HARD_STOP}, "
          f"stop iff every competitor has mean(d) > {MULTIPLIER} * SE(d)")
    print(f"  decision prefix [0,{DECISION_TRIALS})   evaluation segment "
          f"[{EVAL_START},{EVAL_END})   unused [{DECISION_TRIALS},{EVAL_START})")
    print()
    print(f"  {'cfg':>3}{'stop':>7}{'candidate':>11}{'same as 3000':>14}{'decision casc':>15}"
          f"{'gain':>9}{'vs fixed3000':>24}")
    for b in blocks:
        a = b["adaptive"]
        g = b["evaluation"]["gains"]["adaptive"]
        c = b["evaluation"]["adaptive_minus_fixed3000"]
        if c["same_choice"]:
            diff = "same choice"
        else:
            diff = f"{c['mean']:+.3f} [{c['ci95_low']:+.3f},{c['ci95_high']:+.3f}]"
        print(f"  {b['configuration']:>3}{a['stopped_at']:>7}{a['chosen']:>11}"
              f"{str(b['same_choice_as_fixed3000']):>14}{b['adaptive_decision_cascades']:>15,}"
              f"{g['mean']:>9.3f}{diff:>24}")
    print()
    print(f"  {'cfg':>3}  {'fixed1000 cand':>15}{'casc':>8}{'gain':>9}   "
          f"{'fixed3000 cand':>15}{'casc':>8}{'gain':>9}")
    for b in blocks:
        print(f"  {b['configuration']:>3}  {b['fixed1000']['chosen']:>15}"
              f"{b['fixed1000']['decision_cascades']:>8,}"
              f"{b['evaluation']['gains']['fixed1000']['mean']:>9.3f}   "
              f"{b['fixed3000']['chosen']:>15}{b['fixed3000']['decision_cascades']:>8,}"
              f"{b['evaluation']['gains']['fixed3000']['mean']:>9.3f}")
    print()
    print("  ADAPTIVE minus FIXED-1000 (did the adaptive rule do anything right?)")
    for b in blocks:
        c = b["evaluation"]["adaptive_minus_fixed1000"]
        saved = b["fixed1000"]["decision_cascades"] - b["adaptive_decision_cascades"]
        if c["same_choice"]:
            print(f"    cfg{b['configuration']}  same choice ({c['chosen']});  "
                  f"cascades {'saved' if saved >= 0 else 'cost'} {abs(saved):,}")
        else:
            print(f"    cfg{b['configuration']}  {c['mean']:+.3f} "
                  f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  "
                  f"cascades {'saved' if saved >= 0 else 'cost'} {abs(saved):,}")
    print()
    print("  CHECKPOINT DIAGNOSTICS (leader and the seven paired gaps)")
    for b in blocks:
        for rep in b["adaptive"]["checkpoints"]:
            print(f"    cfg{b['configuration']} @{rep['trials']:<5} leader {rep['leader']:<7} "
                  f"triggered={str(rep['triggered']):<5} zero-SE={rep['competitors_with_zero_se']}")
            for c in rep["competitors"]:
                ratio = "n/a" if c["mean_over_se"] is None else f"{c['mean_over_se']:+.2f}"
                print(f"        vs {c['competitor']:<7} mean {c['mean_difference']:+.4f} "
                      f"SE {c['se']:.4f}  mean/SE {ratio:>7}  passes={c['passes']}")
    print()
    print("  CONSTRAINTS")
    for b in blocks:
        c = b["constraints"]
        print(f"    cfg{b['configuration']}  decision ignores evaluation segment: "
              f"{c['decision_ignores_evaluation_segment']}   non-trigger equals fixed3000: "
              f"{c['non_trigger_equals_fixed3000']}")
    print()
    print(f"  early stopped in {n_early}/{len(blocks)} configurations")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
