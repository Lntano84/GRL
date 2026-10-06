"""Gate for the batched encoder: does one padded forward reproduce 600 per-sample forwards?

Compares, on the **same fixed replay batch** covering the real observed-subgraph range (16-63 nodes):

1. the encoder's ``state_embed`` for each sample, and its ``node_embeddings``
2. the two Q values, the loss, every parameter gradient, and the parameters after consecutive updates

against the current best reference -- the ``fast`` (shared-embedding) update -- and, for the encoder
itself, against the authors' unmodified per-sample encoder.  A failure here means the padding changed the
computation and the batched path is a new implementation, not a checked optimisation of the same one.
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
from ccim.xplore.dqn_batched import (gradient_update_sarsa_batched, pad_batch,  # noqa: E402
                                     swap_in_batched_encoder, batched_embeddings)
from ccim.xplore.dqn_opt import gradient_update_sarsa_fast  # noqa: E402
from rl_alg.dqn import DQNTrainer  # noqa: E402
from rl_alg.replay import PriortizedReplay  # noqa: E402
from equiv_check import grads, maxdiff, params  # noqa: E402

OUTPUT = ROOT / "results" / "batched_check.json"


def build(seed, replay, batched: bool):
    torch.manual_seed(seed)
    m = DQNTrainer(input_dim=OFF.INPUT_DIM, state_dim=OFF.EMB_DIM, action_dim=OFF.EMB_DIM,
                   replayBuff=replay, lr=OFF.LR, use_cuda=False, gamma=OFF.GAMMA,
                   gcn_num_layers=2, num_pooling=1)
    if batched:
        swap_in_batched_encoder(m)
    return m


def encoder_compare(replay, seed=0, n=12):
    """Authors' per-sample encoder vs one padded batched forward, on identical samples."""
    m_per = build(seed, replay, batched=False)
    m_bat = build(seed, replay, batched=True)

    rng = np.random.RandomState(7)
    idx = rng.choice(replay.size, n, replace=False)
    entries = [replay.buffer[int(i)] for i in idx]
    xsb = [(torch.from_numpy(np.array(e[0][0], dtype=np.float32)).reshape((1,) + np.array(e[0][0]).shape),
            torch.from_numpy(np.array(e[0][1], dtype=np.float32)).reshape((1,) + np.array(e[0][1]).shape))
           for e in entries]
    xs, adjs, sizes = pad_batch(xsb, "cpu")
    print(f"    mixed batch sizes: min {min(sizes)}  max {max(sizes)}  (n={n})")

    ref_state, ref_node = [], []
    enc = m_per.actor_critic.graph_embedder
    with torch.no_grad():
        for xb, ab in xsb:
            out = enc.forward(xb, ab, compute_loss=False)
            ref_state.append(out[:, -m_per.actor_critic.graph_embedding_dim:])
            ref_node.append(enc.node_embeddings.clone())
    ref_state = torch.cat(ref_state, dim=0)
    b_state, _ = batched_embeddings(m_bat.actor_critic, xs, adjs, sizes)
    bat_node = m_bat.actor_critic.graph_embedder.node_embeddings

    d_state_abs, d_state_rel = maxdiff([ref_state], [b_state])
    worst_node_abs, worst_node_rel = 0.0, 0.0
    for b, m in enumerate(sizes):
        a, r_ = bat_node[b:b + 1, :m, :], ref_node[b][:, :m, :]
        x_, y_ = maxdiff([a.reshape(-1)], [r_.reshape(-1)])
        worst_node_abs = max(worst_node_abs, float(torch.abs(a - r_).max()))
        worst_node_rel = max(worst_node_rel,
                             float(torch.abs(a - r_).norm() / max(float(r_.norm()), 1e-12)))
    return {"sizes": sizes, "state_embed_abs": d_state_abs, "state_embed_rel": d_state_rel,
            "node_embed_abs": worst_node_abs, "node_embed_rel": worst_node_rel,
            "state_embed_rel_ok": d_state_rel < 1e-5, "node_embed_rel_ok": worst_node_rel < 1e-5}


def update_compare(replay, updates: int, seed=0):
    ref = build(seed, replay, batched=False)
    bat = build(seed, replay, batched=True)
    out = {"ref_scalars": [], "bat_scalars": [], "grad_rel": None, "param_rel": None}
    for it in range(updates):
        replay.rg = np.random.RandomState(1000 + it)
        gradient_update_sarsa_fast(ref, batch_size=BATCH, compute_loss=False, share_embeddings=True)
        if it == 0:
            out["ref_grads"] = grads(ref)
        out["ref_scalars"].append(float(ref.loss_critic.detach()))
        replay.rg = np.random.RandomState(1000 + it)
        gradient_update_sarsa_batched(bat, batch_size=BATCH)
        if it == 0:
            out["bat_grads"] = grads(bat)
        out["bat_scalars"].append(float(bat.loss_critic.detach()))
    _, out["grad_rel"] = maxdiff(out["ref_grads"], out["bat_grads"])
    _, out["param_rel"] = maxdiff(params(ref), params(bat))
    out["loss_max_abs"] = max(abs(a - b) for a, b in zip(out["ref_scalars"], out["bat_scalars"]))
    return out


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
    report = {}
    print("=" * 100)
    print(f"  BATCHED ENCODER CHECK -- threads={args.threads}, replay={replay.size}")
    print("=" * 100)

    e = encoder_compare(replay)
    report["encoder"] = e
    print(f"    state_embed   abs {e['state_embed_abs']:.3e}  rel {e['state_embed_rel']:.3e}"
          f"  [{'PASS' if e['state_embed_rel_ok'] else 'FAIL'}]")
    print(f"    node_embeds   abs {e['node_embed_abs']:.3e}  rel {e['node_embed_rel']:.3e}"
          f"  [{'PASS' if e['node_embed_rel_ok'] else 'FAIL'}]")

    u = update_compare(replay, args.updates)
    report["update"] = {k: v for k, v in u.items() if not k.endswith(("grads", "scalars"))}
    report["update"]["ref_losses"] = u["ref_scalars"]
    report["update"]["bat_losses"] = u["bat_scalars"]
    print(f"    loss over {args.updates} updates: max abs diff {u['loss_max_abs']:.3e}")
    print(f"    gradient rel {u['grad_rel']:.3e}   parameter rel {u['param_rel']:.3e}")
    ok = u["grad_rel"] < 1e-5 and u["param_rel"] < 1e-5 and u["loss_max_abs"] < 1e-5
    print(f"    [{'PASS' if ok else 'FAIL'}] batched update tracks the shared-embedding reference")
    report["overall_pass"] = bool(e["state_embed_rel_ok"] and e["node_embed_rel_ok"] and ok)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  overall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"  artifact: {args.output}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
