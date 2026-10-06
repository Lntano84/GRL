"""Atomic, complete training checkpoints: everything the loop consumes, so a restart continues it exactly.

What has to be in a resume point
--------------------------------
Weights alone are not a resume point.  Restarting an RL loop mid-training also needs the replay buffer
*and its priorities*, the optimiser moments, the episode counter, epsilon, and every random stream the loop
actually draws from.  In this driver those streams are:

===========================  ==========================================================
``rng``                      ``np.random.Generator``: training graph choice, initial
                             seed choice, epsilon draws, exploration draws, and the
                             per-``get_embeds`` embedding seeds
``seed_rng``                 ``np.random.Generator``: the 5 free survey nodes per episode
``replay.rg``                ``np.random.RandomState``: which transitions a batch samples
``random`` (stdlib)          used inside the authors' ``ge/walker.py`` random walks and
                             ``icm.sample_live_icm`` live-edge sampling
``torch``                    model initialisation and any stochastic module behaviour
``np.random`` (global)       legacy global state, saved defensively
===========================  ==========================================================

Atomicity
---------
Each save writes to a temporary file and then ``os.replace``s it over the target.  ``os.replace`` is
atomic on Windows and POSIX, so a crash during a save can never leave a half-written checkpoint where a
valid one used to be -- which matters precisely because the machine is expected to be switched off
mid-run.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SCHEMA = 2


def capture(acmodel, replay, episode: int, epsilon: float, rng, seed_rng,
            val_curve: list, selected_episode, extra: dict | None = None) -> dict:
    """Everything needed to continue: networks, optimiser, replay, counters and all RNG states."""
    return {
        "schema": SCHEMA,
        "episode": int(episode),
        "epsilon": float(epsilon),
        "actor_critic": {k: v.clone() for k, v in acmodel.actor_critic.state_dict().items()},
        "target_actor_critic": {k: v.clone()
                                for k, v in acmodel.target_actor_critic.state_dict().items()},
        "critic_opt": acmodel.critic_opt.state_dict(),
        "replay_buffer": list(replay.buffer),
        "replay_probs": list(replay.probs),
        "replay_max_size": replay.max_size,
        "replay_rg": replay.rg.get_state(),
        "replay_beta": replay.beta,
        "replay_eps": replay.eps,
        "rng": rng.bit_generator.state,
        "seed_rng": seed_rng.bit_generator.state,
        "python_random": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "numpy_global": np.random.get_state(),
        "val_curve": list(val_curve),
        "selected_episode": selected_episode,
        "extra": extra or {},
    }


def save_atomic(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)          # atomic on Windows and POSIX


def load(path: Path) -> dict:
    obj = torch.load(Path(path), weights_only=False)
    if obj.get("schema") != SCHEMA:
        raise ValueError(f"checkpoint schema {obj.get('schema')} != {SCHEMA}; refusing to resume")
    return obj


def restore(obj: dict, acmodel, replay, rng, seed_rng) -> int:
    """Put the whole state back and return the episode to resume **after**."""
    acmodel.actor_critic.load_state_dict(obj["actor_critic"])
    acmodel.target_actor_critic.load_state_dict(obj["target_actor_critic"])
    acmodel.critic_opt.load_state_dict(obj["critic_opt"])
    replay.buffer.clear()
    replay.buffer.extend(obj["replay_buffer"])
    replay.probs.clear()
    replay.probs.extend(obj["replay_probs"])
    replay.rg.set_state(obj["replay_rg"])
    rng.bit_generator.state = obj["rng"]
    seed_rng.bit_generator.state = obj["seed_rng"]
    random.setstate(obj["python_random"])
    torch.set_rng_state(obj["torch_rng"])
    np.random.set_state(obj["numpy_global"])
    return int(obj["episode"])


def describe(obj: dict) -> dict:
    """A short fingerprint, for asserting that two checkpoints really are the same state."""
    ac = obj["actor_critic"]
    return {
        "episode": obj["episode"],
        "epsilon": obj["epsilon"],
        "replay_size": len(obj["replay_buffer"]),
        "replay_probs_sum": float(sum(obj["replay_probs"])),
        "weight_checksum": float(sum(float(v.double().sum()) for v in ac.values())),
        "n_tensors": len(ac),
    }
