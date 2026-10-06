"""Reconcile the recorded query counts against the pipeline's structure, using existing logs only.

Two things the Gate 2 report left ambiguous and must now be answered from the recorded data alone:

1. the cap was 36,369 but every run reports 36,378 -- where do the 9 extra queries come from?
2. what episode count was the "same episode count" comparison actually taken at?

The reconciliation is arithmetic on fields already stored in ``results/gate2_shaping.json``; no
training is re-run.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "results" / "gate2_shaping.json").read_text(encoding="utf-8"))

CAP = 36_369
VALIDATE_EVERY = 100
COMMON_EPISODES = 2000
T = 8


def predicted(episodes: int, omega: float) -> dict:
    """Queries the pipeline must issue, decomposed by stage.

    Per training episode:
      reset                                1
      T environment steps                  T            (one sigma each)
      final-value re-query when omega != 0 1            (cached, but still ISSUED and counted)
      T leave-one-out sets when omega != 0 T
    Plus, outside the cap check:
      validation rollouts                  floor(E/100) * (1 + T)
      the common-episode snapshot          (1 + T)      -- runs in ADDITION to that episode's
                                                         validation, because 2000 % 100 == 0
      the final policy rollout             (1 + T)      -- after the loop, never charged to the cap
    """
    per_episode = 1 + T + ((1 + T) if omega else 0)
    training = episodes * per_episode
    validation = (episodes // VALIDATE_EVERY) * (1 + T)
    snapshot = (1 + T) if episodes >= COMMON_EPISODES else 0
    final = 1 + T
    return {"per_episode": per_episode, "training": training, "validation": validation,
            "common_snapshot": snapshot, "final_rollout": final,
            "total": training + validation + snapshot + final,
            "over_cap": training + validation + snapshot + final - CAP}


def main() -> int:
    print(f"cap = {CAP}, validate_every = {VALIDATE_EVERY}, common_episodes = {COMMON_EPISODES}")
    print()
    ok = True
    for name, entry in DATA["graphs"].items():
        for key in ("omega_0.0", "omega_1.0"):
            arm = entry["arms"][key]
            omega = arm["omega"]
            for run in arm["runs"]:
                pred = predicted(run["episodes"], omega)
                match = pred["total"] == run["queries_issued"]
                ok &= match
                print(f"  {name:<9} omega={omega:.0f} seed={run['train_seed']}  "
                      f"E={run['episodes']:>5}  per-ep={pred['per_episode']:>2}  "
                      f"train={pred['training']:>6}  valid={pred['validation']:>4}  "
                      f"snap={pred['common_snapshot']:>2}  final={pred['final_rollout']:>2}  "
                      f"predicted={pred['total']:>6}  recorded={run['queries_issued']:>6}  "
                      f"{'OK' if match else 'MISMATCH'}")
    print()
    print(f"every run reconciled: {ok}")
    print()

    # what the two arms actually pay per episode, and the resulting episode counts
    for name, entry in DATA["graphs"].items():
        a, b = entry["arms"]["omega_0.0"], entry["arms"]["omega_1.0"]
        print(f"  {name}: omega=0 -> {a['mean_episodes']:.0f} episodes at "
              f"{predicted(int(a['mean_episodes']), 0.0)['per_episode']} queries each; "
              f"omega=1 -> {b['mean_episodes']:.0f} episodes at "
              f"{predicted(int(b['mean_episodes']), 1.0)['per_episode']} queries each")
    print()
    print("  the common-episode comparison is taken at the snapshot episode "
          f"= {COMMON_EPISODES}, for BOTH arms;")
    print("  omega=1 then continues to 2010, so its 'final' column is 10 episodes later than its "
          "'common' column.")
    print()
    for name, entry in DATA["graphs"].items():
        b = entry["arms"]["omega_1.0"]
        diffs = [abs(r["final_normalized"] - r["common_episode_normalized"])
                 for r in b["runs"] if r["common_episode_normalized"] is not None]
        print(f"  {name} omega=1: |final - common| per seed = "
              f"{[round(d, 4) for d in diffs]}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
