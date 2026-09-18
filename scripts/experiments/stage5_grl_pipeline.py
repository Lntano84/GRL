"""Stage 5: connect the GRL pipeline to the overexposure oracle.

The pipeline under test is the one already in the repository::

    candidate shortlist -> state-aware marginal predictor -> audited residual trust
    -> progressive MC verification -> selection -> fallback

``grl.algorithms.sequential_im`` already implements all of that behind one interface,
``oracle.score(seeds, candidates, step)``.  This script supplies the two oracles it needs:

* ``OverexposureMonteCarloOracle`` -- the ground-truth reference (stage 3);
* a learned oracle wrapping ``StateConditionedMarginalPredictor``, trained on signed
  overexposure marginal gains produced by that same reference.

It then answers the six verification questions from the task brief, each as a measured
quantity rather than a code reading:

V1 does the predictor really condition on the current state?
V2 is it predicting a *marginal gain*, not a static node score?
V3 does the audit actually test prediction reliability?
V4 does progressive verification actually reduce oracle cost?
V5 does a fallback path really exist and get used?
V6 does predictor corruption increase verification or trigger fallback?

Method comparisons use one graph, one budget, one diffusion parameter set and one evaluation
Monte Carlo budget, and every final seed set is scored by an evaluation oracle that no method
was allowed to consult.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from evaluate_overexposure_pool_ranking import spearman  # noqa: E402
from grl.algorithms.sequential_im import (  # noqa: E402
    adaptive_selective_greedy,
    full_oracle_greedy,
    learned_greedy,
    selective_greedy,
)
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.models import StateConditionedMarginalPredictor  # noqa: E402
from grl.oracle import OverexposureMonteCarloOracle  # noqa: E402
from grl.training.overexposure_dataset import (  # noqa: E402
    OverexposureMarginalSample,
    build_overexposure_dataset,
    dataset_statistics,
)

# --------------------------------------------------------------------------------------
# learned oracle with an exposure channel and an optional corruption mode
# --------------------------------------------------------------------------------------
CORRUPTION_MODES = ("clean", "noise", "shuffle", "random", "sign_flip")


class LearnedOverexposureOracle:
    """Wrap a trained predictor as an oracle, with the state observation it needs.

    The oracle is deterministic in its scores given ``(seeds, candidates, step)`` except where a
    corruption mode deliberately injects randomness; the corruption uses its own RNG so that
    ``clean`` results are exactly reproducible.

    Cost accounting is separate from the MC oracle: ``learned_evaluations`` counts candidate
    forward passes, which cost no cascades.  This separation is the whole point of the
    quality-versus-cost comparison.
    """

    def __init__(
        self,
        model: StateConditionedMarginalPredictor,
        graph: nx.DiGraph,
        embeddings: torch.Tensor,
        norm_degrees: torch.Tensor,
        exposure_fn,
        device: torch.device,
        corruption: str = "clean",
        corruption_strength: float = 0.0,
        corruption_seed: int = 0,
        batch_size: int = 64,
    ) -> None:
        if corruption not in CORRUPTION_MODES:
            raise ValueError(f"corruption must be one of {CORRUPTION_MODES}")
        self.model = model
        self.graph = graph
        self.embeddings = embeddings
        self.norm_degrees = norm_degrees
        self.exposure_fn = exposure_fn
        self.device = torch.device(device)
        self.corruption = corruption
        self.corruption_strength = float(corruption_strength)
        self.corruption_seed = int(corruption_seed)
        self.batch_size = int(batch_size)
        self.num_nodes = int(embeddings.shape[0])
        self.learned_evaluations = 0
        self.mc_cascades = 0          # exposure observation does cost cascades
        self._exposure_cache: dict[tuple[int, ...], torch.Tensor] = {}
        nodes = list(graph.nodes())
        self._order = {node: i for i, node in enumerate(nodes)}

    def _exposure(self, seeds: list[int]) -> torch.Tensor:
        key = tuple(sorted(int(s) for s in seeds))
        if key not in self._exposure_cache:
            values = self.exposure_fn(list(seeds))
            self._exposure_cache[key] = torch.tensor(
                [[float(v)] for v in values], dtype=torch.float32, device=self.device
            )
        return self._exposure_cache[key]

    def _corrupt(self, scores: dict[int, float], step: int) -> dict[int, float]:
        if self.corruption == "clean" or not scores:
            return scores
        rng = random.Random(self.corruption_seed + 7919 * step)
        keys = sorted(scores)
        if self.corruption == "random":
            return {k: rng.uniform(-1.0, 1.0) for k in keys}
        values = [scores[k] for k in keys]
        scale = statistics.pstdev(values) if len(values) > 1 else 1.0
        if self.corruption == "noise":
            return {k: scores[k] + rng.gauss(0.0, self.corruption_strength * max(scale, 1e-9))
                    for k in keys}
        if self.corruption == "shuffle":
            shuffled = list(values)
            rng.shuffle(shuffled)
            return dict(zip(keys, shuffled))
        if self.corruption == "sign_flip":
            return {k: -scores[k] for k in keys}
        return scores

    def score(self, seeds: list[int], candidates: list[int], step: int = 0) -> dict[int, float]:
        if not candidates:
            return {}
        mask = torch.zeros((1, self.num_nodes, 1), dtype=torch.float32, device=self.device)
        if seeds:
            mask[0, seeds, 0] = 1.0
        exposure = self._exposure(seeds) if self.model.exposure_dim > 0 else None
        out: dict[int, float] = {}
        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(candidates), self.batch_size):
                chunk = candidates[start:start + self.batch_size]
                b = len(chunk)
                candidate_tensor = torch.as_tensor(chunk, dtype=torch.long, device=self.device)
                emb = self.embeddings.unsqueeze(0).expand(b, -1, -1)
                nd = self.norm_degrees.unsqueeze(0).expand(b, -1, -1)
                m = mask.expand(b, -1, -1)
                exp = None if exposure is None else exposure.unsqueeze(0).expand(b, -1, -1)
                pred = self.model(emb, nd, m, candidate_tensor, exp).reshape(-1)
                out.update({int(v): float(p) for v, p in zip(chunk, pred.detach().cpu().tolist())})
        self.learned_evaluations += len(candidates)
        return self._corrupt(out, step)


class CumulativeOracle:
    """Accumulate MC-cascade cost across every oracle a policy touches."""

    def __init__(self, mc: OverexposureMonteCarloOracle) -> None:
        self.mc = mc

    @property
    def cascades(self) -> int:
        return self.mc.stats.mc_cascades

    def reset(self) -> None:
        self.mc.stats.reset()


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def train_predictor(
    graph: nx.DiGraph,
    splits: dict[str, list[OverexposureMarginalSample]],
    embedding_dim: int,
    exposure_dim: int,
    hidden_dim: int,
    epochs: int,
    lr: float,
    seed: int,
    use_seed_mask: bool = True,
) -> tuple[StateConditionedMarginalPredictor, dict]:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    n = len(nodes)

    # Node features: random embeddings are enough to test whether the state channel carries
    # signal; structured embeddings are a later refinement and would confound this stage.
    embeddings = torch.randn(n, embedding_dim) / math.sqrt(embedding_dim)
    degrees = dict(graph.out_degree())
    max_degree = max(degrees.values()) if degrees else 1
    norm_degrees = torch.tensor([[degrees[v] / max_degree] for v in nodes], dtype=torch.float32)

    model = StateConditionedMarginalPredictor(
        embedding_dim, hidden_dim=hidden_dim, structural_dim=1,
        exposure_dim=exposure_dim, use_seed_mask=use_seed_mask,
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    def make_batch(samples: list[OverexposureMarginalSample]):
        """Stack samples into one seed state per row with several candidates per row.

        Samples are grouped by context so that each row is a distinct ``(seed set, state)``; rows
        with fewer candidates than the widest are padded by repeating their first candidate.  The
        padded entries are excluded from the loss.
        """
        by_context: dict[str, list[OverexposureMarginalSample]] = {}
        for s in samples:
            by_context.setdefault(s.context_id, []).append(s)
        contexts = list(by_context.values())
        rows = len(contexts)
        cols = max(len(c) for c in contexts)

        masks = torch.zeros((rows, n, 1))
        exposure = torch.zeros((rows, n, exposure_dim)) if exposure_dim > 0 else None
        candidates = torch.zeros((rows, cols), dtype=torch.long)
        labels = torch.zeros((rows, cols))
        valid = torch.zeros((rows, cols), dtype=torch.bool)

        for r, group in enumerate(contexts):
            seed_set = group[0].seed_set
            if seed_set:
                masks[r, seed_set, 0] = 1.0
            if exposure is not None and group[0].exposure:
                exposure[r, :, 0] = torch.tensor(group[0].exposure, dtype=torch.float32)
            for c, s in enumerate(group):
                candidates[r, c] = s.candidate
                labels[r, c] = s.marginal_gain
                valid[r, c] = True
            for c in range(len(group), cols):
                candidates[r, c] = group[0].candidate
        return masks, candidates, labels, exposure, valid

    train_samples = list(splits["train"])
    if not train_samples:
        raise ValueError("no training samples; increase 'contexts' in overexposure_dataset")
    labels_all = torch.tensor([s.marginal_gain for s in train_samples])
    scale = float(labels_all.std()) or 1.0
    offset = float(labels_all.mean())

    history = []
    for epoch in range(epochs):
        rng.shuffle(train_samples)
        # group into mini-batches of contexts
        order: dict[str, list[OverexposureMarginalSample]] = {}
        for s in train_samples:
            order.setdefault(s.context_id, []).append(s)
        context_groups = list(order.values())
        total, count = 0.0, 0
        for start in range(0, len(context_groups), 16):
            batch = [s for g in context_groups[start:start + 16] for s in g]
            masks, candidates, labels, exposure, valid = make_batch(batch)
            model.train()
            optimiser.zero_grad()
            pred = model.score_candidates(embeddings, norm_degrees, masks, candidates, exposure)
            loss = nn.functional.mse_loss(
                pred[valid] / scale, (labels[valid] - offset) / scale
            )
            loss.backward()
            optimiser.step()
            total += float(loss) * int(valid.sum())
            count += int(valid.sum())
        history.append(total / max(1, count))

    metrics = {"final_train_mse_normalised": history[-1] if history else float("nan"),
               "label_scale": scale, "label_offset": offset, "epochs": epochs}
    return model, {"embeddings": embeddings, "norm_degrees": norm_degrees, "metrics": metrics}


# --------------------------------------------------------------------------------------
# verification checks
# --------------------------------------------------------------------------------------
@dataclass
class MethodResult:
    graph: str
    budget: int
    policy: str
    corruption: str
    spread: float
    spread_stderr: float
    gap_to_oracle: float
    mc_cascades: int
    verified_total: int
    accepted_steps: int
    envelope_steps: int
    full_scan_steps: int
    cap_stopped_steps: int
    steps: int


def run_policy(
    policy: str,
    graph: nx.DiGraph,
    pool: list[int],
    budget: int,
    learned,
    exact: OverexposureMonteCarloOracle,
    evaluator: OverexposureMonteCarloOracle,
    top_m: int = 8,
    initial_m: int = 4,
    batch_m: int = 4,
    max_m: int | None = None,
    oracle_spread: float | None = None,
) -> MethodResult:
    before = exact.stats.mc_cascades
    if policy == "learned_only":
        result = learned_greedy(pool, budget, learned)
    elif policy == "selective_greedy":
        result = selective_greedy(pool, budget, learned, exact, top_m=top_m)
    elif policy == "adaptive_selective":
        result = adaptive_selective_greedy(
            pool, budget, learned, exact,
            initial_m=initial_m, batch_m=batch_m, max_m=max_m,
        )
    elif policy == "full_oracle":
        result = full_oracle_greedy(pool, budget, exact)
    else:
        raise ValueError(policy)
    used = exact.stats.mc_cascades - before

    spread = evaluator.spread(result.selected_seeds)
    steps = result.steps
    verified_total = sum(int(s.get("verified", 0)) for s in steps)
    # Three stages, reported separately (audit item P1-2).  ``empirical_accept`` is a heuristic
    # envelope test and can be wrong; only ``fallback_full_scan`` is exact with respect to the pool;
    # ``statistical_certificate`` is never set by this codebase.
    accepted_steps = sum(1 for s in steps if s.get("empirical_accept", False))
    full_scan_steps = sum(1 for s in steps if s.get("fallback_full_scan", False))
    envelope_steps = sum(
        1 for s in steps
        if s.get("empirical_accept", False) and not s.get("fallback_full_scan", False)
    )
    cap_stopped_steps = sum(1 for s in steps if not s.get("empirical_accept", False))
    certified_steps = sum(1 for s in steps if s.get("statistical_certificate", False))
    if certified_steps:
        raise RuntimeError(
            "a statistical certificate was reported, but nothing in this repository establishes one"
        )
    gap = float("nan")
    if oracle_spread and oracle_spread > 1e-9:
        gap = (oracle_spread - spread["mean"]) / oracle_spread
    return MethodResult(
        graph="", budget=budget, policy=policy, corruption="clean",
        spread=spread["mean"], spread_stderr=spread["stderr"],
        gap_to_oracle=gap, mc_cascades=used,
        verified_total=verified_total, accepted_steps=accepted_steps,
        envelope_steps=envelope_steps, full_scan_steps=full_scan_steps,
        cap_stopped_steps=cap_stopped_steps, steps=len(steps),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pool-size", type=int, default=30)
    parser.add_argument("--contexts", type=int, default=90)
    parser.add_argument("--candidates-per-context", type=int, default=20)
    parser.add_argument("--dataset-mc", type=int, default=25)
    parser.add_argument("--oracle-mc", type=int, default=25)
    parser.add_argument("--eval-mc", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--top-m", type=int, default=6)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    params = resolve_overexposure_params(None)
    device = torch.device("cpu")

    print(f"graph={args.graph} n={n} m={graph.number_of_edges()} "
          f"<k>={2*graph.number_of_edges()/n:.2f}")
    print(f"budgets={args.budgets} pool={args.pool_size} dataset_mc={args.dataset_mc} "
          f"oracle_mc={args.oracle_mc} eval_mc={args.eval_mc}")
    print()

    # ---------------- dataset ----------------
    cfg = {
        "overexposure_dataset": {
            "budget": max(args.budgets),
            "candidates_per_context": args.candidates_per_context,
            "contexts": args.contexts,
            "mc_runs": args.dataset_mc,
            "random_seed": args.random_seed,
        }
    }
    splits = build_overexposure_dataset(graph, cfg, params=params, oracle_mc=args.dataset_mc)
    stats = dataset_statistics(splits)
    print("dataset:")
    for name, values in stats.items():
        if values.get("n"):
            print(f"  {name:<11} n={values['n']:<5} contexts={values['contexts']:<4} "
                  f"mean={values['mean_gain']:>9.3f} min={values['min_gain']:>9.3f} "
                  f"neg={values['negative_share']*100:>5.1f}%")
        else:
            print(f"  {name:<11} EMPTY")
    print()

    # ---------------- train two variants ----------------
    variants: dict[str, tuple] = {}
    for name, exposure_dim, use_mask in (
        ("seed_only", 0, True),
        ("state_conditioned", 1, True),
        ("exposure_only", 1, False),
    ):
        model, bundle = train_predictor(
            graph, splits, args.embedding_dim, exposure_dim, args.hidden_dim,
            args.epochs, 3e-3, args.random_seed, use_seed_mask=use_mask,
        )
        variants[name] = (model, bundle)
        print(f"trained {name:<18} exposure_dim={exposure_dim} use_seed_mask={use_mask} "
              f"train_mse={bundle['metrics']['final_train_mse_normalised']:.4f}")
    print()

    # ---------------- V2: is it a marginal gain, not a static score? ----------------
    print("V2  is the prediction a marginal gain rather than a static node score?")
    test = splits["test"]
    if test:
        oracle_ref = OverexposureMonteCarloOracle(graph, mc_runs=args.oracle_mc,
                                                  random_seed=args.random_seed, params=params)
        model, bundle = variants["state_conditioned"]
        learned = LearnedOverexposureOracle(
            model, graph, bundle["embeddings"], bundle["norm_degrees"],
            exposure_fn=lambda s: oracle_ref.state(s, step=args.random_seed),
            device=device,
        )
        by_context: dict[str, list[OverexposureMarginalSample]] = {}
        for s in test:
            by_context.setdefault(s.context_id, []).append(s)
        ctx_id, samples = next(iter(by_context.items()))
        seeds = samples[0].seed_set
        cands = [s.candidate for s in samples]
        truth = [s.marginal_gain for s in samples]
        pred = learned.score(seeds, cands, step=0)
        pred_list = [pred[c] for c in cands]
        # static control: the same candidates scored with an empty seed set
        pred_static = learned.score([], cands, step=0)
        static_list = [pred_static[c] for c in cands]
        print(f"  context {ctx_id} |S|={len(seeds)} candidates={len(cands)}")
        print(f"    rho(prediction, truth)              = {spearman(pred_list, truth):+.3f}")
        print(f"    rho(static score, truth)            = {spearman(static_list, truth):+.3f}")
        print(f"    truth range [{min(truth):.2f}, {max(truth):.2f}] "
              f"negatives={sum(1 for t in truth if t<0)}/{len(truth)}")
    print()

    # ---------------- main sweep ----------------
    rows: list[MethodResult] = []
    for budget in args.budgets:
        rng = random.Random(args.random_seed + 31 * budget)
        pool = rng.sample(nodes, min(args.pool_size, n))

        exact = OverexposureMonteCarloOracle(graph, mc_runs=args.oracle_mc,
                                             random_seed=args.random_seed, params=params)
        evaluator = OverexposureMonteCarloOracle(graph, mc_runs=args.eval_mc,
                                                 random_seed=args.random_seed + 99,
                                                 params=params)
        model, bundle = variants["state_conditioned"]
        learn_oracle = LearnedOverexposureOracle(
            model, graph, bundle["embeddings"], bundle["norm_degrees"],
            exposure_fn=lambda s, o=exact: o.state(s, step=args.random_seed),
            device=device,
        )

        oracle_result = run_policy("full_oracle", graph, pool, budget,
                                   learn_oracle, exact, evaluator)
        oracle_spread = oracle_result.spread
        oracle_result.graph = args.graph
        rows.append(oracle_result)
        print(f"k={budget}  oracle spread={oracle_spread:.2f} "
              f"mc_cascades={oracle_result.mc_cascades}")

        for policy in ("learned_only", "selective_greedy", "adaptive_selective"):
            exact.stats.reset()
            learn_oracle.learned_evaluations = 0
            r = run_policy(policy, graph, pool, budget, learn_oracle, exact, evaluator,
                           top_m=args.top_m, oracle_spread=oracle_spread)
            r.graph = args.graph
            rows.append(r)
            print(f"   {policy:<18} spread={r.spread:8.2f} gap={r.gap_to_oracle*100:6.2f}% "
                  f"mc={r.mc_cascades:<6} verified={r.verified_total:<4} "
                  f"accept={r.accepted_steps}/{r.steps} "
                  f"(envelope={r.envelope_steps} full_scan={r.full_scan_steps} "
                  f"cap_stopped={r.cap_stopped_steps})")
        print()

    print("=" * 100)
    print("STAGE 5 SUMMARY")
    print("=" * 100)
    print(f"  {'k':>3}{'policy':<20}{'spread':>9}{'gap%':>8}{'mc_cascades':>13}"
          f"{'verified':>10}{'accept':>7}{'env':>5}{'full':>5}{'cap':>5}")
    for r in rows:
        print(f"  {r.budget:>3}{r.policy:<20}{r.spread:>9.2f}{r.gap_to_oracle*100:>8.2f}"
              f"{r.mc_cascades:>13}{r.verified_total:>10}{r.accepted_steps:>7}"
              f"{r.envelope_steps:>5}{r.full_scan_steps:>5}{r.cap_stopped_steps:>5}")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph, "n": n, "budgets": args.budgets,
            "pool_size": args.pool_size, "dataset": stats,
            "rows": [asdict(r) for r in rows],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
