"""Where does the first-step parameter difference come from?

Both versions are confirmed to start from **bit-identical weights and bit-identical Adam state**, then take
one step on the **same** replay batch.  The script reports, per parameter tensor: the gradient in each
version, the Adam update each produced, and the resulting absolute parameter difference.

The hypothesis under test is that the overall gradient relative error is tiny while individual gradients
sit near zero, where Adam's denominator ``sqrt(v_hat) + eps`` is dominated by ``eps`` and the update becomes
almost ``lr * m_hat / eps`` -- a large gain on an arbitrarily small gradient difference.  This is a
localisation, not an inference about long-run behaviour.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import xplore_official as OFF  # noqa: E402
from bench_update import BATCH, fill  # noqa: E402
from ccim.xplore.dqn_batched import gradient_update_sarsa_batched, swap_in_batched_encoder  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402
from equiv_check import maxdiff  # noqa: E402

OUTPUT = ROOT / "results" / "first_step_localisation.json"


def build(seed, replay, replay_seed, batched=False):
    torch.manual_seed(seed)
    m = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                   replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                   gcn_num_layers=2, num_pooling=1)
    if batched:
        swap_in_batched_encoder(m)
    opt_state = m.critic_opt.state_dict()
    return m, opt_state


def named_params(m):
    return dict(m.actor_critic.named_parameters())


def adam_state(m):
    out = {}
    for i, (name, p) in enumerate(m.actor_critic.named_parameters()):
        st = m.critic_opt.state.get(p)
        if st is not None:
            out[name] = {"exp_avg": st["exp_avg"].detach().clone(),
                         "exp_avg_sq": st["exp_avg_sq"].detach().clone(),
                         "step": st["step"]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    rng = np.random.default_rng(0)
    replay = PriortizedReplay(OFF.BUFF, 10, beta=0.6)
    fill(replay, rng)
    # the replay's own sampler is re-seeded so both versions consume the identical batch
    replay.rg = np.random.RandomState(1000)
    batch = replay.sample_(BATCH)

    ref, _ = build(0, replay, 1000, batched=False)
    ref_before = {n: p.detach().clone() for n, p in named_params(ref).items()}

    bat, _ = build(0, replay, 1000, batched=True)
    bat_before = {n: p.detach().clone() for n, p in named_params(bat).items()}

    same_init = all(torch.equal(ref_before[n], bat_before[n]) for n in ref_before)
    # Adam state is empty before the first step; confirm that too
    empty_ref = len(ref.critic_opt.state) == 0
    empty_bat = len(bat.critic_opt.state) == 0

    # one step each, on the identical batch
    replay.rg = np.random.RandomState(1000)
    grad_ref, upd_ref = step_with_grads(ref)
    replay.rg = np.random.RandomState(1000)
    grad_bat, upd_bat = step_with_grads(bat)

    rows = []
    for name in ref_before:
        g1, g2 = grad_ref.get(name), grad_bat.get(name)
        if g1 is None or g2 is None:
            continue
        u1, u2 = upd_ref.get(name), upd_bat.get(name)
        d_theta = (named_params(ref)[name].detach() - named_params(bat)[name].detach())
        rows.append({
            "param": name,
            "param_absmax": float(ref_before[name].abs().max()),
            "grad_norm_ref": float(g1.norm()),
            "grad_norm_bat": float(g2.norm()),
            "grad_absdiff_max": float((g1 - g2).abs().max()),
            "grad_reldiff": float((g1 - g2).norm() / max(float(g1.norm()), 1e-30)),
            "update_absmax_ref": float(u1.abs().max()) if u1 is not None else None,
            "update_absmax_bat": float(u2.abs().max()) if u2 is not None else None,
            "param_absdiff_max": float(d_theta.abs().max()),
            "param_reldiff": float(d_theta.norm() / max(float(ref_before[name].norm()), 1e-30)),
            "grad_is_near_zero": bool(float(g1.abs().max()) < 1e-6),
        })
    rows.sort(key=lambda r: -r["param_absdiff_max"])

    print("=" * 100)
    print(f"  FIRST-STEP LOCALISATION -- threads={args.threads}")
    print("=" * 100)
    print(f"    weights bit-identical before the step : {same_init}")
    print(f"    Adam state empty before the step     : ref={empty_ref} bat={empty_bat} "
          f"(both start from the same, empty, optimiser state)")
    tot_abs, tot_rel = maxdiff(list(ref_before.values()), list(named_params(bat).values()))
    print(f"    overall parameter absdiff {tot_abs:.3e}  reldiff {tot_rel:.3e}")
    print(f"\n    {'parameter tensor':<44}{'|grad|':>10}{'grad reldiff':>13}"
          f"{'max|dtheta|':>13}{'near-zero g':>12}")
    for r in rows[:10]:
        print(f"    {r['param'][:43]:<44}{r['grad_norm_ref']:>10.2e}{r['grad_reldiff']:>13.2e}"
              f"{r['param_absdiff_max']:>13.3e}{str(r['grad_is_near_zero']):>12}")

    nz = [r for r in rows if r["grad_is_near_zero"]]
    print(f"\n    tensors with near-zero gradients (max|g| < 1e-6): {len(nz)} of {len(rows)}")
    if nz:
        worst = max(nz, key=lambda r: r["param_absdiff_max"])
        print(f"    largest absolute parameter move among them: {worst['param']} "
              f"-> {worst['param_absdiff_max']:.3e}")

    args.output.write_text(json.dumps({"same_initial_weights": bool(same_init),
                                       "adam_state_empty_both": bool(empty_ref and empty_bat),
                                       "overall_param_absdiff": tot_abs,
                                       "overall_param_reldiff": tot_rel,
                                       "per_parameter": rows[:30]},
                                      ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"    artifact: {args.output}")
    return 0


def step_with_grads(model):
    """Run one batched/fast step and capture the gradients and the resulting Adam update per tensor."""
    names = [n for n, _ in model.actor_critic.named_parameters()]
    before = {n: p.detach().clone() for n, p in model.actor_critic.named_parameters()}
    is_batched = type(model.actor_critic.graph_embedder).__name__ == "BatchedSoftPoolingGcnEncoder"
    if is_batched:
        gradient_update_sarsa_batched(model, batch_size=BATCH)
    else:
        from ccim.xplore.dqn_opt import gradient_update_sarsa_fast
        gradient_update_sarsa_fast(model, batch_size=BATCH, compute_loss=False, share_embeddings=True)
    grads = {}
    updates = {}
    for n, p in model.actor_critic.named_parameters():
        if p.grad is not None:
            grads[n] = p.grad.detach().clone()
        updates[n] = (p.detach() - before[n])
    return grads, updates


if __name__ == "__main__":
    raise SystemExit(main())
