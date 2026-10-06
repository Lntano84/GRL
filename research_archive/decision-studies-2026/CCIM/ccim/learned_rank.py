"""The one frozen model for the learned swap ranker, plus the dataset builder and the split rules.

One configuration, no sweep
---------------------------
The user's instruction was *exactly one config, no hyperparameter search*.  Everything that could be
tuned lives in :data:`CONFIG` and is written into the checkpoint, so a later run can be shown to have
used the same numbers rather than "the best numbers".

The target
----------
``delta(u,v|S) = sigma(S \\ {u} u {v}) - sigma(S)`` -- the real, full-cascade gain of the swap.  Within
one state ``sigma(S)`` is constant, so ranking by ``delta`` and ranking by the post-swap ``sigma`` are
the same ordering.  The model predicts **only** this number.  It never sees the query position, never
sees whether the swap was accepted, and never decides acceptance: the online loop still verifies with
the true diffuser and still requires a strict improvement.

Honest limitation, restated
---------------------------
The replayed trajectories stop at the first improving candidate, so the negatives are drawn from *the
visited candidates*, not uniformly from the candidate pool.  The rows are a real sample of what the
search actually asks, and a biased sample of what it could ask.  Offline accuracy on them is therefore
not evidence that online search improves; only the online three-arm comparison is.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import numpy as np

from .swap_features import FEATURE_NAMES


@dataclass(frozen=True)
class FrozenConfig:
    feature_names: tuple = FEATURE_NAMES
    hidden: tuple = (32, 32)
    activation: str = "relu"
    loss: str = "mse_on_delta"
    optimizer: str = "adam"
    learning_rate: float = 1e-2
    weight_decay: float = 0.0
    batch_size: int = 256
    max_epochs: int = 40
    training_seed: int = 20240117
    checkpoint_metric: str = "val_pairwise_accuracy"
    checkpoint_rule: str = "max_metric_ties_earliest_epoch"
    standardise_inputs: bool = True
    standardise_target: bool = False


CONFIG = FrozenConfig()


# --------------------------------------------------------------------------------------- dataset
def build_dataset(records: list[dict]) -> dict:
    """Turn replayed ``(state_id, u, v, before, after)`` rows into grouped features and labels.

    ``before``/``after`` come from the recorded rows, so the label is the *recorded* diffusion result;
    features are recomputed from the state's ``S``.

    Two identity rules, both of which matter:

    * ``state_id`` is an **opaque key**, not an integer.  Every trajectory numbers its states from 1, so
      raw ids collide across seeds; the caller must namespace them (e.g. ``"0:17"``).  Getting this wrong
      silently scores one trajectory's rows against another trajectory's seed set, which is why unknown
      keys raise instead of being tolerated, and why it is covered by
      ``tests/test_learned_rank.py::test_states_from_two_seeds_do_not_collide``.
    * **a state is its seed set.**  Distinct keys with the same ``S`` are merged into one group, because
      the degree-52 initial set begins every trajectory and would otherwise be counted three times.  Rows
      are de-duplicated on ``(S, u, v)``, so the same swap observed on two visits to the same state
      contributes one observation, while two *different* swaps observed at that state are both kept.

    ``keys_of_group`` reports which original keys were merged, so the merge is auditable.
    """
    from .swap_features import build_context

    graph = records["graph"]
    states = {str(k): v for k, v in records["state_sets"].items()}
    ctx_cache: dict[int, object] = {}
    group_of_S: dict[frozenset, int] = {}
    keys_of_group: dict[int, list[str]] = {}
    seen = set()
    X, y, group = [], [], []
    for row in records["rows"]:
        key = str(row["state_id"])
        if key not in states:
            raise KeyError(f"row refers to unknown state {key!r}")
        S = states[key]
        Skey = frozenset(int(x) for x in S)
        if Skey not in group_of_S:
            group_of_S[Skey] = len(group_of_S) + 1
            keys_of_group[group_of_S[Skey]] = []
        g = group_of_S[Skey]
        if key not in keys_of_group[g]:
            keys_of_group[g].append(key)
        dedup = (g, int(row["u"]), int(row["v"]))
        if dedup in seen:
            continue
        seen.add(dedup)
        if g not in ctx_cache:
            ctx_cache[g] = build_context(graph, S, records["K"])
        X.append(ctx_cache[g].features(int(row["u"]), int(row["v"])))
        y.append(float(row["after"]) - float(row["before"]))
        group.append(g)
    if not X:
        raise ValueError("no rows in the supplied records")
    return {"X": np.asarray(X, dtype=np.float64),
            "y": np.asarray(y, dtype=np.float64),
            "group": np.asarray(group, dtype=np.int64),
            "S_of_group": {g: states[keys_of_group[g][0]] for g in keys_of_group},
            "keys_of_group": keys_of_group,
            "graph": graph}


def surviving_keys(ds: dict) -> set:
    """The original state keys that survived into the dataset (after any split repair)."""
    out = set()
    for g in np.unique(ds["group"]):
        out.update(ds["keys_of_group"][int(g)])
    return out


def state_keys(ds: dict) -> dict:
    """``group -> frozenset(S)``; the split is defined on the seed set, not on a state id."""
    return {int(g): frozenset(ds["S_of_group"][int(g)]) for g in np.unique(ds["group"])}

def check_split_disjoint(datasets: dict[str, dict]) -> dict:
    """The same ``S`` must never appear in two splits.  Checked on the seed set itself, not the id.

    This is not a formality: all three replayed trajectories start from the *same* degree-52 initial
    set, so the initial state is shared unless it is removed.  The caller must call
    :func:`drop_shared_states` (or otherwise resolve the clash) before training, and this function
    reports the clash rather than silently tolerating it.
    """
    seen: dict[frozenset, str] = {}
    clashes = []
    for name, ds in datasets.items():
        for g, key in state_keys(ds).items():
            if key in seen and seen[key] != name:
                clashes.append({"split_a": seen[key], "split_b": name,
                                "seed_set_size": len(key), "state": sorted(key)})
            seen[key] = name
    return {"states_total": len(seen), "clashes": clashes, "disjoint": not clashes}


def drop_shared_states(keep: str, others: list[str], datasets: dict[str, dict]) -> dict:
    """Remove from ``others`` every state whose ``S`` also occurs in ``keep``.

    Returns a description of what was removed, so the report can state the exact cost of the fix rather
    than claiming the split was clean from the start.
    """
    keep_keys = set(state_keys(datasets[keep]).values())
    removed = {}
    for name in others:
        ds = datasets[name]
        keys = state_keys(ds)
        drop = {g for g, key in keys.items() if key in keep_keys}
        if not drop:
            removed[name] = {"states_dropped": 0, "rows_dropped": 0, "positives_dropped": 0}
            continue
        mask = ~np.isin(ds["group"], list(drop))
        removed[name] = {
            "states_dropped": len(drop),
            "rows_dropped": int((~mask).sum()),
            "positives_dropped": int((ds["y"][~mask] > 0).sum()),
            "shared_state_sizes": sorted(len(keys[g]) for g in drop),
        }
        for field in ("X", "y", "group"):
            ds[field] = ds[field][mask]
        for g in drop:
            ds["S_of_group"].pop(g, None)
    return removed


# --------------------------------------------------------------------------------------- model
def _torch():
    import torch
    return torch


class RankModel:
    """Frozen MLP plus the training-split standardisation it was fitted with."""

    def __init__(self, net, mean, std, config=CONFIG, train_summary=None):
        self.net = net
        self.mean = mean
        self.std = std
        self.config = config
        self.train_summary = train_summary or {}

    def predict(self, features: np.ndarray) -> np.ndarray:
        torch = _torch()
        with torch.no_grad():
            x = torch.tensor(np.atleast_2d(features), dtype=torch.float32)
            x = (x - self.mean) / self.std
            return self.net(x).squeeze(-1).numpy()

    @staticmethod
    def fresh(config: FrozenConfig = CONFIG) -> "RankModel":
        """An untrained model with the frozen shape -- used by tests that only exercise the plumbing."""
        torch = _torch()
        torch.manual_seed(config.training_seed)
        net = _build_net(len(config.feature_names), config)
        net.eval()
        d = len(config.feature_names)
        return RankModel(net, torch.zeros(d), torch.ones(d), config)

    def save(self, path: str) -> None:
        torch = _torch()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "mean": self.mean, "std": self.std,
                    "config": asdict(self.config), "train_summary": self.train_summary,
                    "feature_names": list(self.config.feature_names)}, path)

    @staticmethod
    def load(path: str) -> "RankModel":
        torch = _torch()
        blob = torch.load(path, weights_only=False)
        cfg = FrozenConfig(**{**asdict(CONFIG), **{k: (tuple(v) if k == "hidden" else v)
                                                  for k, v in blob["config"].items()}})
        net = _build_net(len(blob["feature_names"]), cfg)
        net.load_state_dict(blob["state_dict"])
        net.eval()
        return RankModel(net, blob["mean"], blob["std"], cfg, blob.get("train_summary"))

    def numpy_scores(self, features) -> np.ndarray:
        """The frozen network evaluated in numpy, with no torch in the loop.

        Same weights, float32 throughout, same standardisation.  Used for the fixed-wall-clock
        comparison because a torch forward pass measurably inflates the timing of the *surrounding*
        regions without doing more work (see ``results/diffusion_cost_attribution.json``), which would
        otherwise be charged to the learned arm.  ``tests/test_learned_rank.py`` checks that this
        reproduces the torch ordering exactly, so this is an implementation detail and not a change of
        method.
        """
        layers = [m for m in self.net if isinstance(m, _torch().nn.Linear)]
        h = np.asarray(features, dtype=np.float32)
        h = (h - self.mean.numpy().astype(np.float32)) / self.std.numpy().astype(np.float32)
        for i, lin in enumerate(layers):
            h = h @ lin.weight.detach().numpy().astype(np.float32).T \
                + lin.bias.detach().numpy().astype(np.float32)
            if i < len(layers) - 1:
                h = np.maximum(h, 0.0)
        return h.reshape(-1)

    def numpy_ranker(self):
        """A ranker callable for :func:`ccim.batched_search.batched_search`'s ``custom`` hook."""
        def ranker(ctx, pairs):
            scores = self.numpy_scores([ctx.features(u, v) for (u, v) in pairs])
            order = np.argsort(-scores, kind="stable")
            return [pairs[int(i)] for i in order]
        return ranker


def _build_net(d_in: int, config: FrozenConfig):
    torch = _torch()
    layers, prev = [], d_in
    for h in config.hidden:
        layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
        prev = h
    layers += [torch.nn.Linear(prev, 1)]
    return torch.nn.Sequential(*layers)


# --------------------------------------------------------------------------------------- metrics
def pairwise_accuracy(scores: np.ndarray, y: np.ndarray, group: np.ndarray) -> dict:
    """Within-state ordering accuracy: P(score(positive) > score(negative)), ties counted as 0.5.

    States with no positive or no negative candidate are skipped -- they carry no ordering information.
    """
    correct = total = 0.0
    usable = 0
    for g in np.unique(group):
        m = group == g
        s, yy = scores[m], y[m]
        pos, neg = s[yy > 0], s[yy <= 0]
        if pos.size == 0 or neg.size == 0:
            continue
        usable += 1
        for p in pos:
            correct += float(np.sum(p > neg)) + 0.5 * float(np.sum(p == neg))
            total += neg.size
    return {"pairwise_accuracy": (correct / total) if total else None,
            "states_usable": usable, "pairs": int(total)}


# --------------------------------------------------------------------------------------- training
def train(records_train: dict, records_val: dict, config: FrozenConfig = CONFIG,
          verbose: bool = False) -> tuple[RankModel, dict]:
    """Train the frozen configuration; select the checkpoint on validation pairwise accuracy only.

    Returns ``(model, history)``.  The selected epoch, the selection metric and the per-epoch curve are
    all reported, so the checkpoint choice can be audited rather than trusted.
    """
    torch = _torch()
    torch.manual_seed(config.training_seed)
    np.random.seed(config.training_seed)

    tr, va = build_dataset(records_train), build_dataset(records_val)
    Xtr, ytr, gtr = tr["X"], tr["y"], tr["group"]
    Xva, yva, gva = va["X"], va["y"], va["group"]

    mean = torch.tensor(Xtr.mean(axis=0), dtype=torch.float32)
    std = torch.tensor(Xtr.std(axis=0), dtype=torch.float32)
    std[std == 0] = 1.0

    def to_t(X):
        x = torch.tensor(X, dtype=torch.float32)
        return (x - mean) / std if config.standardise_inputs else x

    xtr, ytr_t = to_t(Xtr), torch.tensor(ytr, dtype=torch.float32).unsqueeze(-1)
    xva = to_t(Xva)

    net = _build_net(len(config.feature_names), config)
    opt = torch.optim.Adam(net.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    lossfn = torch.nn.MSELoss()

    n = xtr.shape[0]
    g = torch.Generator().manual_seed(config.training_seed)
    history = []
    best = {"epoch": None, "metric": -1.0, "state_dict": None}

    for epoch in range(1, config.max_epochs + 1):
        net.train()
        perm = torch.randperm(n, generator=g)
        epoch_loss = 0.0
        for i in range(0, n, config.batch_size):
            idx = perm[i:i + config.batch_size]
            opt.zero_grad()
            loss = lossfn(net(xtr[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
            epoch_loss += float(loss.detach()) * idx.numel()
        net.eval()
        with torch.no_grad():
            val_scores = net(xva).squeeze(-1).numpy()
        pv = pairwise_accuracy(val_scores, yva, gva)
        metric = pv["pairwise_accuracy"] if pv["pairwise_accuracy"] is not None else -1.0
        history.append({"epoch": epoch, "train_mse": epoch_loss / n,
                        "val_pairwise_accuracy": pv["pairwise_accuracy"],
                        "val_states_usable": pv["states_usable"]})
        if verbose:
            print(f"  epoch {epoch:3d}  mse {epoch_loss / n:.4f}  val_pw {metric:.4f}")
        if metric > best["metric"]:
            best = {"epoch": epoch, "metric": metric,
                    "state_dict": {k: v.clone() for k, v in net.state_dict().items()}}

    net.load_state_dict(best["state_dict"])
    net.eval()
    model = RankModel(net, mean, std, config,
                      train_summary={"selected_epoch": best["epoch"],
                                     "selected_metric": best["metric"],
                                     "epochs_run": config.max_epochs,
                                     "train_rows": int(len(ytr)),
                                     "train_states": int(len(np.unique(gtr))),
                                     "val_rows": int(len(yva)),
                                     "val_states": int(len(np.unique(gva))),
                                     "train_positive_rate": float((ytr > 0).mean()),
                                     "val_positive_rate": float((yva > 0).mean())})
    return model, {"history": history, "selected": dict(model.train_summary),
                   "config": asdict(config)}


def save_history(path: str, history: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(history, fh, ensure_ascii=False, indent=2)
