"""Successive-halving candidate elimination: 8 -> 4 -> 2 -> 1 at a fixed cost of 17000 cascades.

This is NOT a novel algorithm
----------------------------
Progressively eliminating trailing candidates is the standard successive-halving / racing idea, and
budget-allocation methods of this kind are well established in the literature.  Nothing here is
presented as new; this script asks only whether an off-the-shelf halving schedule beats **uniform
sampling at the same cost** on four configurations.

The two questions are different
-------------------------------
The previous prototype asked whether the *whole decision* could stop early; it could not, and that
line is archived.  This one asks whether simulation can be **withdrawn from trailing candidates**.
The saving is guaranteed by the schedule, so there is nothing to test about cost --- the only open
question is **quality**: does eliminating candidates early throw away the one that would have won?

The schedule, fixed in advance, no stage length or ratio adjusted
----------------------------------------------------------------
=========  =========================  ============================  ==============================
stage      candidates computed        decision data                 stage end
=========  =========================  ============================  ==============================
1          all 8                      trials ``[0, 1000)``           keep top 4 by mean
2          the surviving 4            cumulative ``[0, 2000)``       keep top 2 by mean
3          the surviving 2            cumulative ``[0, 3000)``       pick 1 by mean
=========  =========================  ============================  ==============================

Ties break by ascending node id.  An eliminated candidate never returns.

Cost
----
``8*1000 + 4*1000 + 2*1000 = 14000`` candidate runs plus ``3000`` shared base runs = **17000**
cascades, against fixed-3000's ``3000 * 9 = 27000``: **37.0% fewer**.

A near-equal-cost control is mandatory
--------------------------------------
``fixed1888`` costs ``1888 * 9 = 16992`` cascades --- 8 cascades less than the halving schedule, a
rounding difference.  If it performs about as well, then any gain **cannot be attributed to
allocating the budget cleverly**, because uniform sampling at the same cost achieves it.

Data split, unchanged
---------------------
Decisions read only trials ``[0, 3000)``; evaluation uses ``[4000, 10000)``; no new diffusion
simulation.  A candidate eliminated at stage ``s`` must never have its later trials read, and that is
**tested by garbling them** rather than assumed.

These four configurations are now **development data**.  This is a development diagnostic: even a
good result here is not an independently validated rule.
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

CANDIDATES_SCREENED = 8
#: (cumulative decision trials, how many candidates survive) -- fixed, never tuned.
STAGES = ((1000, 4), (2000, 2), (3000, 1))
DECISION_TRIALS = 3000
EVAL_START = 4000
EVAL_END = 10000
NEAR_COST_CONTROL = 1888

OUTPUT = ROOT / "docs" / "results" / "successive_halving.json"


# ------------------------------------------------------------------------------------ cost model
def uniform_cost(trials: int) -> int:
    """One base run per trial plus one run per candidate screened, for every trial."""
    return trials * (1 + CANDIDATES_SCREENED)


def halving_cost() -> dict:
    """Candidate runs per stage, plus the shared base runs, counted trial by trial."""
    previous, breakdown, total = 0, [], 0
    for trials, keep in STAGES:
        newly = trials - previous
        alive = CANDIDATES_SCREENED // (2 ** len(breakdown))
        candidate_runs = alive * newly
        base_runs = newly
        total += candidate_runs + base_runs
        breakdown.append({"stage": len(breakdown) + 1, "trials_read": [previous, trials],
                          "new_trials": newly, "candidates_alive": alive,
                          "candidate_runs": candidate_runs, "base_runs": base_runs,
                          "subtotal": candidate_runs + base_runs})
        previous = trials
    return {"per_stage": breakdown, "total": total,
            "candidate_runs": sum(s["candidate_runs"] for s in breakdown),
            "base_runs": sum(s["base_runs"] for s in breakdown)}


# ------------------------------------------------------------------------------------ the rule
def successive_halving(series: dict[int, list[float]], pool: list[int]) -> dict:
    """8 -> 4 -> 2 -> 1.  Reads only the trials the current stage is allowed to see."""
    alive = sorted(pool)
    trail, previous = [], 0
    for stage, (trials, keep) in enumerate(STAGES, start=1):
        means = {c: statistics.fmean(series[c][:trials]) for c in alive}
        order = sorted(alive, key=lambda c: (-means[c], c))
        survivors, eliminated = order[:keep], order[keep:]
        trail.append({
            "stage": stage, "new_trials_read": [previous, trials],
            "candidates_computed": list(alive),
            "cumulative_mean": {str(c): means[c] for c in alive},
            "ranking": order, "survivors": survivors, "eliminated": eliminated,
            "eliminated_at_trial": trials if eliminated else None,
        })
        alive = survivors
        previous = trials
    return {"chosen": alive[0], "trail": trail,
            "elimination_trial": {str(c): s["eliminated_at_trial"]
                                  for s in trail for c in s["eliminated"]}}


def uniform_choice(series: dict[int, list[float]], pool: list[int], trials: int) -> int:
    means = {c: statistics.fmean(series[c][:trials]) for c in pool}
    return max(pool, key=lambda c: (means[c], -c))


# ------------------------------------------------------------------------------------ statistics
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


def load_series(index: int) -> dict[int, list[float]]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    hcost = halving_cost()
    blocks = []
    for index in range(len(sbc.NEW_SEEDS)):
        series = load_series(index)
        pool = degree_top8(index)

        halving = successive_halving(series, pool)
        chosen = {
            "halving": halving["chosen"],
            "fixed1000": uniform_choice(series, pool, 1000),
            f"fixed{NEAR_COST_CONTROL}": uniform_choice(series, pool, NEAR_COST_CONTROL),
            "fixed3000": uniform_choice(series, pool, DECISION_TRIALS),
        }

        # ---- the eliminated candidates' later trials must never be read ----
        garbled = {c: list(v) for c, v in series.items()}
        rng = random.Random(20260917 + index)
        for c, at in halving["elimination_trial"].items():
            for t in range(int(at), DECISION_TRIALS):
                garbled[int(c)][t] = rng.uniform(-5000.0, 5000.0)
        garbled_halving = successive_halving(garbled, pool)
        constraint_post_elimination = (
            garbled_halving["chosen"] == halving["chosen"]
            and [s["ranking"] for s in garbled_halving["trail"]] == [s["ranking"] for s in halving["trail"]]
            and [s["survivors"] for s in garbled_halving["trail"]] == [s["survivors"] for s in halving["trail"]])

        def evaluate(candidate: int) -> dict:
            return describe(series[candidate][EVAL_START:EVAL_END])

        gains = {k: evaluate(v) for k, v in chosen.items()}

        def contrast(a: str, b: str) -> dict:
            if chosen[a] == chosen[b]:
                return {"same_choice": True, "chosen": chosen[a],
                        "note": "both methods selected the SAME candidate; the contrast is exactly "
                                "zero by construction and is not independent evidence"}
            return dict(paired(series[chosen[a]][EVAL_START:EVAL_END],
                               series[chosen[b]][EVAL_START:EVAL_END]),
                        same_choice=False, definition=f"{a} minus {b}")

        # was the candidate fixed-3000 would have picked eliminated before the end?
        f3000 = chosen["fixed3000"]
        f3000_eliminated_at = halving["elimination_trial"].get(str(f3000))
        blocks.append({
            "configuration": index, "random_seed": sbc.NEW_SEEDS[index], "degree_top8": pool,
            "halving": halving,
            "chosen": {k: int(v) for k, v in chosen.items()},
            "fixed3000_candidate_eliminated_early": f3000_eliminated_at is not None,
            "fixed3000_candidate_eliminated_at_trial": f3000_eliminated_at,
            "evaluation": {"segment": [EVAL_START, EVAL_END],
                           "trials": EVAL_END - EVAL_START,
                           "gains": gains,
                           "halving_minus_fixed1888": contrast("halving", f"fixed{NEAR_COST_CONTROL}"),
                           "halving_minus_fixed3000": contrast("halving", "fixed3000"),
                           "halving_minus_fixed1000": contrast("halving", "fixed1000"),
                           "fixed1888_minus_fixed3000": contrast(f"fixed{NEAR_COST_CONTROL}",
                                                                 "fixed3000")},
            "cost": {"halving": hcost["total"],
                     "fixed1000": uniform_cost(1000),
                     f"fixed{NEAR_COST_CONTROL}": uniform_cost(NEAR_COST_CONTROL),
                     "fixed3000": uniform_cost(DECISION_TRIALS)},
            "constraints": {"eliminated_candidates_later_trials_never_read":
                            constraint_post_elimination},
        })

    artifact = {
        "script": Path(__file__).name, "code_version": sbc.code_version(),
        "run_command": "python " + " ".join([str(Path(__file__).relative_to(ROOT)), *sys.argv[1:]]),
        "status": "DEVELOPMENT DIAGNOSTIC on four configurations that are now development data.  "
                  "Not an independently validated rule, even if the numbers look good.",
        "not_novel": "progressively eliminating trailing candidates is the established successive-"
                     "halving / racing idea; budget-allocation methods of this kind are well "
                     "documented.  Nothing here is claimed as a new algorithm.",
        "question": "can simulation be withdrawn from trailing candidates without losing the "
                    "candidate that would have won?",
        "schedule": {"stages": [{"stage": i + 1, "candidates": CANDIDATES_SCREENED // (2 ** i),
                                 "cumulative_trials": s[0], "keep": s[1]}
                                for i, s in enumerate(STAGES)],
                     "ties": "ascending node id", "revival": "none", "tuning": "none"},
        "cost_model": {
            "halving": hcost,
            "uniform": {f"fixed{t}": uniform_cost(t)
                        for t in (1000, NEAR_COST_CONTROL, DECISION_TRIALS)},
            "halving_vs_fixed3000_reduction":
                1.0 - hcost["total"] / uniform_cost(DECISION_TRIALS),
            "near_cost_control_difference": hcost["total"] - uniform_cost(NEAR_COST_CONTROL),
            "note": "the saving is guaranteed by the schedule, so cost is not what is being tested; "
                    "the open question is quality, and fixed1888 is the near-equal-cost control that "
                    "decides whether any gain is due to allocating the budget cleverly",
        },
        "replay": {"decision_prefix": [0, DECISION_TRIALS], "evaluation_segment": [EVAL_START, EVAL_END],
                   "new_simulation": "none"},
        "summary": {
            "configurations": len(blocks),
            "fixed3000_candidate_eliminated_early":
                sum(1 for b in blocks if b["fixed3000_candidate_eliminated_early"]),
            "halving_equals_fixed1888": sum(
                1 for b in blocks
                if b["evaluation"]["halving_minus_fixed1888"]["same_choice"]),
            "halving_equals_fixed3000": sum(
                1 for b in blocks
                if b["evaluation"]["halving_minus_fixed3000"]["same_choice"]),
        },
        "configurations": blocks,
        "complete": True,
    }
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ trajectories
    print("=" * 116)
    print("  SUCCESSIVE HALVING 8->4->2->1 -- development diagnostic, zero new simulation")
    print("=" * 116)
    print(f"  cost: halving {hcost['total']:,} cascades "
          f"({hcost['candidate_runs']:,} candidate + {hcost['base_runs']:,} base) versus "
          f"fixed3000 {uniform_cost(DECISION_TRIALS):,}  ->  "
          f"{artifact['cost_model']['halving_vs_fixed3000_reduction']*100:.1f}% fewer")
    print(f"  near-cost control fixed{NEAR_COST_CONTROL} = "
          f"{uniform_cost(NEAR_COST_CONTROL):,} cascades "
          f"({artifact['cost_model']['near_cost_control_difference']:+d} versus halving)")
    print()
    for b in blocks:
        print(f"  cfg{b['configuration']}  degree top-8 {b['degree_top8']}")
        for s in b["halving"]["trail"]:
            kept = " ".join(f"{c}:{s['cumulative_mean'][str(c)]:.3f}" for c in s["candidates_computed"])
            print(f"    stage {s['stage']}  trials {s['new_trials_read']}  "
                  f"kept {s['survivors']}  out {s['eliminated']}")
            print(f"        means {kept}")
        print(f"    winner {b['halving']['chosen']}   fixed3000 picked "
              f"{b['chosen']['fixed3000']}   "
              f"eliminated early: {b['fixed3000_candidate_eliminated_early']}"
              + (f" (at trial {b['fixed3000_candidate_eliminated_at_trial']})"
                 if b["fixed3000_candidate_eliminated_early"] else ""))
        print()
    print("  COST AND GAIN")
    print(f"    {'cfg':>3}{'method':>13}{'cascades':>10}{'candidate':>11}{'gain':>9}")
    for b in blocks:
        for method in ("fixed1000", f"fixed{NEAR_COST_CONTROL}", "halving", "fixed3000"):
            print(f"    {b['configuration']:>3}{method:>13}{b['cost'][method]:>10,}"
                  f"{b['chosen'][method]:>11}"
                  f"{b['evaluation']['gains'][method]['mean']:>9.3f}")
    print()
    print("  HALVING minus FIXED1888 (the near-equal-cost control)")
    for b in blocks:
        c = b["evaluation"]["halving_minus_fixed1888"]
        if c["same_choice"]:
            print(f"    cfg{b['configuration']}  same choice ({c['chosen']})")
        else:
            print(f"    cfg{b['configuration']}  {c['mean']:+.3f} +-{c['se']:.3f} "
                  f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  p={c['p_two_sided']:.3f}")
    print("  HALVING minus FIXED3000")
    for b in blocks:
        c = b["evaluation"]["halving_minus_fixed3000"]
        if c["same_choice"]:
            print(f"    cfg{b['configuration']}  same choice ({c['chosen']})")
        else:
            print(f"    cfg{b['configuration']}  {c['mean']:+.3f} +-{c['se']:.3f} "
                  f"[{c['ci95_low']:+.3f}, {c['ci95_high']:+.3f}]  p={c['p_two_sided']:.3f}")
    print()
    print("  CONSTRAINT: eliminated candidates' later trials never read")
    for b in blocks:
        print(f"    cfg{b['configuration']}  {b['constraints']['eliminated_candidates_later_trials_never_read']}")
    print()
    print(f"  fixed3000's candidate eliminated early in "
          f"{artifact['summary']['fixed3000_candidate_eliminated_early']}/{len(blocks)} configurations")
    print(f"  wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
