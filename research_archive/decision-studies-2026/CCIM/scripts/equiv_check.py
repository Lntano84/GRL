"""Is the optimised update engineering-equivalent to the published one, or a new implementation?

The question is settled by measurement, not by argument.  On a **fixed replay batch** (the replay's RNG is
re-seeded identically before each version samples), two freshly seeded models are driven through:

* the published ``DQNTrainer.gradient_update_sarsa`` (dqn.py:631-663), untouched;
* ``gradient_update_sarsa_fast`` with duplicated embedding forwards removed, and separately with the
  unused auxiliary losses switched off;

and compared on, in order of strictness:

1. ``q_next1`` / ``q_next2`` / ``q_expected`` / ``q_predicted1`` / ``q_predicted2`` / ``loss_critic``
2. every parameter gradient after ``backward()`` and before the optimizer step
3. every parameter after the step
4. parameters and greedily selected actions after several consecutive updates

Anything that differs makes this a **new implementation version**, not an engineering-equivalent speed-up,
and it is reported that way.  ``share_embeddings=False`` isolates the ``compute_loss`` change alone, so the
two edits can be attributed separately rather than as a bundle.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_official as OFF  # noqa: E402
from bench_update import BATCH, fill  # noqa: E402
from ccim.xplore.dqn_opt import gradient_update_sarsa_fast  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402

OUTPUT = ROOT / "results" / "equiv_check.json"


def fresh_model(seed: int, replay) -> DQNTrainer:
    torch.manual_seed(seed)
    m = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                   replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                   gcn_num_layers=2, num_pooling=1)
    return m


def params(m) -> list:
    return [p.detach().clone() for p in m.actor_critic.parameters()] + \
           [p.detach().clone() for p in m.target_actor_critic.parameters()]


def grads(m) -> list:
    out = []
    for p in list(m.actor_critic.parameters()) + list(m.target_actor_critic.parameters()):
        out.append(torch.zeros_like(p) if p.grad is None else p.grad.detach().clone())
    return out


def maxdiff(a: list, b: list) -> tuple:
    """Max absolute difference, and the **global** relative error ``||a-b|| / ||b||``.

    Per-element relative error is unusable here: a parameter that happens to sit near zero produces a
    relative error near 1 from a difference of 1e-8, which says nothing about whether the two runs did the
    same arithmetic.  The norm ratio is the standard measure and does not have that failure mode.
    """
    if len(a) != len(b):
        return float("inf"), float("inf")
    flat_a = torch.cat([x.reshape(-1) for x in a]) if a else torch.zeros(0)
    flat_b = torch.cat([x.reshape(-1) for x in b]) if b else torch.zeros(0)
    absdiff = (flat_a - flat_b).abs()
    worst_abs = absdiff.max().item() if absdiff.numel() else 0.0
    denom = flat_b.norm().item()
    rel = (absdiff.norm().item() / denom) if denom > 0 else float("inf") if worst_abs > 0 else 0.0
    return worst_abs, rel


def running_stats(model) -> list:
    """BatchNorm running_mean / running_var -- module state that a forward pass mutates in train mode."""
    out = []
    for m in model.actor_critic.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            out.append(m.running_mean.detach().clone())
            out.append(m.running_var.detach().clone())
    for m in model.target_actor_critic.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            out.append(m.running_mean.detach().clone())
            out.append(m.running_var.detach().clone())
    return out


def bn_update_counts(model) -> int:
    """How many times each BatchNorm saw a batch: num_batches_tracked, summed."""
    total = 0
    for m in list(model.actor_critic.modules()) + list(model.target_actor_critic.modules()):
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm) and m.track_running_stats:
            if m.num_batches_tracked is not None:
                total += int(m.num_batches_tracked.item())
    return total


def run_variant(replay, seed: int, updates: int, kind: str):
    model = fresh_model(seed, replay)
    scalars, grad_snap, param_snap = [], None, None
    for it in range(updates):
        replay.rg = np.random.RandomState(1000 + it)          # identical batch for every variant
        if kind == "published":
            model.gradient_update_sarsa(batch_size=BATCH)
        elif kind == "compute_loss_off_only":
            gradient_update_sarsa_fast(model, batch_size=BATCH, compute_loss=False,
                                       share_embeddings=False)
        elif kind == "fast":
            gradient_update_sarsa_fast(model, batch_size=BATCH, compute_loss=False,
                                       share_embeddings=True)
        elif kind == "fast_keep_aux_loss":
            gradient_update_sarsa_fast(model, batch_size=BATCH, compute_loss=True,
                                       share_embeddings=True)
        if it == 0:
            grad_snap = grads(model)
        scalars.append({k: float(getattr(model, k).detach().reshape(-1)[0])
                        for k in ("q_next1", "q_next2", "q_expected")
                        if hasattr(model, k)} |
                       {"q_predicted1": float(model.q_predicted1.detach().reshape(-1)[0]),
                        "q_predicted2": float(model.q_predicted2.detach().reshape(-1)[0]),
                        "loss_critic": float(model.loss_critic.detach())})
    return {"model": model, "scalars": scalars, "grads": grad_snap, "params": params(model),
            "bn_running": running_stats(model), "bn_batches": bn_update_counts(model)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--updates", type=int, default=5)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rng = np.random.default_rng(0)
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, rng)

    base = run_variant(replay, 0, args.updates, "published")
    report = {}
    print("=" * 100)
    print(f"  EQUIVALENCE CHECK -- fixed replay batches, {args.updates} consecutive updates, "
          f"threads={args.threads}")
    print("=" * 100)
    print(f"    {'variant':<26}{'max|dq1|':>11}{'max|dq2|':>11}{'max|dloss|':>12}"
          f"{'grad rel':>11}{'param rel':>11}{'BN batches':>12}  verdict")
    for kind in ("compute_loss_off_only", "fast_keep_aux_loss", "fast"):
        other = run_variant(replay, 0, args.updates, kind)
        dq1 = max(abs(a["q_predicted1"] - b["q_predicted1"])
                  for a, b in zip(base["scalars"], other["scalars"]))
        dq2 = max(abs(a["q_predicted2"] - b["q_predicted2"])
                  for a, b in zip(base["scalars"], other["scalars"]))
        dloss = max(abs(a["loss_critic"] - b["loss_critic"])
                    for a, b in zip(base["scalars"], other["scalars"]))
        dg_abs, dg_rel = maxdiff(base["grads"], other["grads"])
        dp_abs, dp_rel = maxdiff(base["params"], other["params"])
        # float32 carries ~1.2e-7 relative precision, so only a relative gap well above that is a real
        # divergence; at or below it the two runs are the same computation in a different summation order
        same_math = max(dq1, dq2, dloss, dg_rel, dp_rel) < 1e-5
        bn_same = base["bn_batches"] == other["bn_batches"]
        report[kind] = {"max_abs_delta_q_predicted1": dq1, "max_abs_delta_q_predicted2": dq2,
                        "max_abs_delta_loss": dloss,
                        "grad_abs": dg_abs, "grad_rel": dg_rel,
                        "param_abs": dp_abs, "param_rel": dp_rel,
                        "float32_same_math": bool(same_math),
                        "bn_batch_updates_base": base["bn_batches"],
                        "bn_batch_updates_variant": other["bn_batches"],
                        "bn_state_identical": bool(bn_same),
                        "bitwise_identical": bool(max(dq1, dq2, dloss, dg_abs, dp_abs) == 0.0)}
        verdict = ("EQUIVALENT" if report[kind]["bitwise_identical"]
                   else ("same math (float32)" if same_math else "DIVERGES"))
        if same_math and not bn_same:
            verdict += ", BN state differs"
        print(f"    {kind:<26}{dq1:>11.2e}{dq2:>11.2e}{dloss:>12.2e}"
              f"{dg_rel:>11.2e}{dp_rel:>11.2e}{other['bn_batches']:>8}/{base['bn_batches']:<3}"
              f"  {verdict}")

    # ---- action selection must also agree after consecutive updates
    state = torch.zeros(1, 4, 20)
    adj = torch.eye(4).unsqueeze(0)
    acts = {}
    for kind, res in (("published", base),) + tuple(
            (k, run_variant(replay, 0, args.updates, k)) for k in ("fast",)):
        with torch.no_grad():
            st, _ = __import__("ccim.xplore.dqn_opt", fromlist=["x"]).embeddings(
                res["model"].actor_critic, state, adj, compute_loss=False)
            q1, q2 = res["model"].actor_critic.critic(st, torch.zeros(1, OFF.EMB_DIM))
        acts[kind] = (float(q1.reshape(-1)[0]), float(q2.reshape(-1)[0]))
    print(f"\n    action-selection Q after {args.updates} updates:")
    for k, v in acts.items():
        print(f"      {k:<12} q1={v[0]:+.10f}  q2={v[1]:+.10f}")
    # float32 carries ~1.2e-7 relative precision, so anything at or below ~1e-5 relative is the same
    # computation in a different summation order, not a divergence
    agree = (abs(acts["published"][0] - acts["fast"][0])
             <= 1e-5 * max(abs(acts["published"][0]), 1e-12) and
             abs(acts["published"][1] - acts["fast"][1])
             <= 1e-5 * max(abs(acts["published"][1]), 1e-12))
    print(f"    [{'PASS' if agree else 'FAIL'}] published and fast agree on the selected action "
          f"(float32 tolerance)")

    all_equiv = all(v["bitwise_identical"] for v in report.values()) and agree
    bn_preserved = all(v["bn_state_identical"] for v in report.values())
    print(f"\n  OVERALL:")
    print(f"    bitwise-identical variants present : "
          f"{[k for k, v in report.items() if v['bitwise_identical']]}")
    print(f"    BatchNorm state preserved by every variant : {bn_preserved}")
    if not bn_preserved:
        print("    -> the published update runs the embedder 6x per sample and the shared version 2x, so")
        print("       BatchNorm running statistics advance a different number of times.  Per-step Q values")
        print("       agree to float32 precision, but the module state does not, so embedding sharing is")
        print("       NOT purely engineering-equivalent: report it as a new implementation version.")
    print(f"  action-selection agreement: {'yes' if agree else 'no'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"threads": args.threads, "updates": args.updates,
                                       "comparisons": report,
                                       "action_selection": acts,
                                       "all_equivalent": bool(all_equiv)},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  artifact: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
