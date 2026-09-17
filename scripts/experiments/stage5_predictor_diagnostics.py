"""Stage 5 diagnostics: is the predictor actually learning anything useful?

Stage 5's first run raised three suspicions that must be resolved before any claim:

S1. ``adaptive_selective`` spent MORE Monte-Carlo cascades than the full oracle it is supposed to
    economise on.  Feasible explanations: the audit needs more verification than the budget
    allows; or the exposure observation (which costs cascades) is being charged to the policy.
S2. ``exposure_only`` (no seed mask) had the LOWEST training MSE.  A model that cannot see the
    seed set should be worse, so a low loss suggests it is regressing each candidate's
    context-average and the other variants are overfitting noise.
S3. The stage-5 "is it marginal?" check reported identical Spearman for the state-conditioned
    prediction and a static score, because it measured at ``|S| = 0`` where there is no state to
    condition on.  The check must be run at a non-empty state.

Underlying risk for all three: node features are random embeddings, so two nodes are nearly
indistinguishable and the model cannot express "this candidate is better" except through the
seed mask.  This script measures that directly by comparing, within a context:

  * the predictor's ranking of candidates against the truth,
  * a static degree ranking,
  * a random ranking,

and by reporting how much of the within-context label variance the predictor explains.  If the
predictor cannot beat random inside a context, the learnt part is not working, whatever the
aggregate spread says.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts" / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "experiments"))

from evaluate_density_degree_signflip import GRAPHS, load as load_graph  # noqa: E402
from evaluate_overexposure_pool_ranking import spearman  # noqa: E402
from grl.diffusion.params import resolve_overexposure_params  # noqa: E402
from grl.models import StateConditionedMarginalPredictor  # noqa: E402
from grl.oracle import OverexposureMonteCarloOracle  # noqa: E402
from grl.training.overexposure_dataset import (  # noqa: E402
    build_overexposure_dataset,
    dataset_statistics,
)


@dataclass
class ContextScore:
    context_id: str
    seed_size: int
    n_candidates: int
    negative_share: float
    rho_model: float
    rho_degree: float
    rho_random: float
    top1_model_ok: bool
    top1_degree_ok: bool


def feature_bundle(graph: nx.DiGraph, n: int, embedding_dim: int, mode: str, seed: int):
    """Node features under three regimes, to isolate whether identity is the bottleneck."""
    g = torch.Generator().manual_seed(seed)
    nodes = list(graph.nodes())
    if mode == "random":
        emb = torch.randn(n, embedding_dim, generator=g) / math.sqrt(embedding_dim)
    elif mode == "onehot":
        # a unique identifiable code per node, projected down; lets the model tell nodes apart
        emb = torch.zeros(n, embedding_dim)
        for i in range(n):
            emb[i, i % embedding_dim] = 1.0 + i / max(1, n)
        emb = emb + 0.01 * torch.randn(n, embedding_dim, generator=g)
    elif mode == "structural":
        out_deg = dict(graph.out_degree())
        in_deg = dict(graph.in_degree())
        out_sum = {v: 0.0 for v in nodes}
        for u, v, d in graph.edges(data=True):
            out_sum[u] += float(d.get("weight", 0.0))
        cols = []
        for v in nodes:
            cols.append([out_deg[v], in_deg[v], out_sum[v], out_deg[v] * out_sum[v]])
        base = torch.tensor(cols, dtype=torch.float32)
        mean = base.mean(0, keepdim=True)
        std = base.std(0, keepdim=True).clamp_min(1e-6)
        base = (base - mean) / std
        emb = base.repeat(1, max(1, embedding_dim // base.shape[1] + 1))[:, :embedding_dim]
        if emb.shape[1] < embedding_dim:
            pad = torch.randn(n, embedding_dim - emb.shape[1], generator=g) * 0.01
            emb = torch.cat([emb, pad], dim=1)
    else:
        raise ValueError(mode)
    degrees = dict(graph.out_degree())
    mx = max(degrees.values()) if degrees else 1
    norm_degrees = torch.tensor([[degrees[v] / mx] for v in nodes], dtype=torch.float32)
    return emb, norm_degrees


def train(
    graph, splits, embedding_dim, exposure_dim, hidden_dim, epochs, lr, seed, features,
    ranking_weight: float = 0.0,
):
    """Train the predictor.

    ``ranking_weight > 0`` adds a per-row listwise ranking term to the regression loss.  This
    matters: the label scale differs by orders of magnitude between contexts (a candidate can be
    worth 0 or 368 depending on how saturated the state is), so a pure MSE objective is dominated
    by getting the magnitude right and can ignore the within-context ordering that seed selection
    actually uses.  The diagnostic showed exactly that failure, so the term is switchable and
    measured rather than assumed.
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    nodes = list(graph.nodes())
    n = len(nodes)
    emb, nd = features
    model = StateConditionedMarginalPredictor(
        embedding_dim, hidden_dim=hidden_dim, structural_dim=1, exposure_dim=exposure_dim
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    groups: dict[str, list] = {}
    for s in splits["train"]:
        groups.setdefault(s.context_id, []).append(s)
    labels = torch.tensor([s.marginal_gain for s in splits["train"]])
    scale = float(labels.std()) or 1.0
    offset = float(labels.mean())

    def batch_of(context_groups):
        rows = len(context_groups)
        cols = max(len(g) for g in context_groups)
        masks = torch.zeros((rows, n, 1))
        expo = torch.zeros((rows, n, exposure_dim)) if exposure_dim > 0 else None
        cand = torch.zeros((rows, cols), dtype=torch.long)
        lab = torch.zeros((rows, cols))
        val = torch.zeros((rows, cols), dtype=torch.bool)
        for r, g in enumerate(context_groups):
            if g[0].seed_set:
                masks[r, g[0].seed_set, 0] = 1.0
            if expo is not None and g[0].exposure:
                expo[r, :, 0] = torch.tensor(g[0].exposure, dtype=torch.float32)
            for c, s in enumerate(g):
                cand[r, c] = s.candidate
                lab[r, c] = s.marginal_gain
                val[r, c] = True
            for c in range(len(g), cols):
                cand[r, c] = g[0].candidate
        return masks, cand, lab, expo, val

    for _ in range(epochs):
        order = list(groups.values())
        rng.shuffle(order)
        for start in range(0, len(order), 16):
            masks, cand, lab, expo, val = batch_of(order[start:start + 16])
            model.train()
            opt.zero_grad()
            pred = model.score_candidates(emb, nd, masks, cand, expo)
            regression = torch.nn.functional.mse_loss(
                pred[val] / scale, (lab[val] - offset) / scale
            )
            loss = regression
            if ranking_weight > 0.0:
                # Listwise ranking within each row: the softmax over predicted scores should put
                # mass on the candidates with the largest true gain.  Padded entries are masked.
                masked = pred.masked_fill(~val, float("-inf"))
                target = torch.softmax(lab.masked_fill(~val, float("-inf")) / scale, dim=1)
                logp = torch.log_softmax(masked, dim=1)
                loss = regression + ranking_weight * (-(target * logp).sum(dim=1)).mean()
            loss.backward()
            opt.step()
    return model, emb, nd


def score_contexts(model, emb, nd, oracle, graph, splits, split_name, exposure_dim, seed):
    groups: dict[str, list] = {}
    for s in splits[split_name]:
        groups.setdefault(s.context_id, []).append(s)
    degree = dict(graph.out_degree())
    out: list[ContextScore] = []
    rng = random.Random(seed)
    model.eval()
    for cid, g in groups.items():
        seeds = g[0].seed_set
        cands = [s.candidate for s in g]
        truth = [s.marginal_gain for s in g]
        masks = torch.zeros((1, len(list(graph.nodes())), 1))
        if seeds:
            masks[0, seeds, 0] = 1.0
        expo = None
        if exposure_dim > 0:
            expo = torch.tensor([[v] for v in g[0].exposure], dtype=torch.float32).unsqueeze(0)
        cand_t = torch.tensor([cands], dtype=torch.long)
        with torch.no_grad():
            pred = model.score_candidates(emb, nd, masks, cand_t, expo).reshape(-1).tolist()
        deg = [float(degree[c]) for c in cands]
        rnd = [rng.random() for _ in cands]
        from evaluate_overexposure_pool_ranking import _top_k_indices  # local import
        best = max(range(len(truth)), key=lambda i: truth[i])
        out.append(ContextScore(
            context_id=cid, seed_size=len(seeds), n_candidates=len(cands),
            negative_share=sum(1 for t in truth if t < 0) / len(truth),
            rho_model=spearman(pred, truth), rho_degree=spearman(deg, truth),
            rho_random=spearman(rnd, truth),
            top1_model_ok=_top_k_indices(pred, 1)[0] == best,
            top1_degree_ok=_top_k_indices(deg, 1)[0] == best,
        ))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="congress_twitter", choices=list(GRAPHS))
    parser.add_argument("--feature-modes", nargs="+",
                        default=["random", "onehot", "structural"])
    parser.add_argument("--ranking-weights", type=float, nargs="+", default=[0.0, 5.0],
                        help="listwise ranking term weight; 0.0 is pure regression")
    parser.add_argument("--contexts", type=int, default=60)
    parser.add_argument("--candidates-per-context", type=int, default=12)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--dataset-mc", type=int, default=25)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--random-seed", type=int, default=20260917)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    graph = load_graph(args.graph)
    n = graph.number_of_nodes()
    params = resolve_overexposure_params(None)
    cfg = {"overexposure_dataset": {
        "budget": args.budget, "candidates_per_context": args.candidates_per_context,
        "contexts": args.contexts, "mc_runs": args.dataset_mc,
        "random_seed": args.random_seed}}
    splits = build_overexposure_dataset(graph, cfg, params=params, oracle_mc=args.dataset_mc)
    stats = dataset_statistics(splits)
    print(f"graph={args.graph} n={n}")
    for k, v in stats.items():
        if v.get("n"):
            print(f"  {k:<11} n={v['n']:<5} ctx={v['contexts']:<4} "
                  f"neg={v['negative_share']*100:>5.1f}%  mean={v['mean_gain']:>9.3f}")
    print()

    oracle = OverexposureMonteCarloOracle(graph, mc_runs=args.dataset_mc,
                                          random_seed=args.random_seed, params=params)
    results: dict[str, dict] = {}
    per_context_rows: list[dict] = []

    for mode in args.feature_modes:
        features = feature_bundle(graph, n, args.embedding_dim, mode, args.random_seed)
        for dim in (0, 1):
            for rw in args.ranking_weights:
                model, emb, nd = train(graph, splits, args.embedding_dim, dim, args.hidden_dim,
                                       args.epochs, 3e-3, args.random_seed, features,
                                       ranking_weight=rw)
                rows = score_contexts(model, emb, nd, oracle, graph, splits, "test", dim,
                                      args.random_seed)
                label = f"{mode}/exp{dim}/rw{rw:g}"
                agg = {
                    "n_contexts": len(rows),
                    "rho_model": statistics.fmean([r.rho_model for r in rows if r.rho_model == r.rho_model]),
                    "rho_degree": statistics.fmean([r.rho_degree for r in rows if r.rho_degree == r.rho_degree]),
                    "rho_random": statistics.fmean([r.rho_random for r in rows if r.rho_random == r.rho_random]),
                    "top1_model": statistics.fmean([1.0 if r.top1_model_ok else 0.0 for r in rows]),
                    "top1_degree": statistics.fmean([1.0 if r.top1_degree_ok else 0.0 for r in rows]),
                    "negative_share": statistics.fmean([r.negative_share for r in rows]),
                }
                results[label] = agg
                for r in rows:
                    d = asdict(r)
                    d["variant"] = label
                    per_context_rows.append(d)
                print(f"{label:<28} rho_model={agg['rho_model']:+.3f}  "
                      f"rho_degree={agg['rho_degree']:+.3f}  rho_random={agg['rho_random']:+.3f}  "
                      f"top1_model={agg['top1_model']:.2f} top1_degree={agg['top1_degree']:.2f}")
        print()

    print("=" * 96)
    print("DIAGNOSTIC VERDICT")
    print("=" * 96)
    best = max(results.items(), key=lambda kv: kv[1]["rho_model"])
    print(f"  best variant by within-context rho: {best[0]} ({best[1]['rho_model']:+.3f})")
    print(f"  degree baseline: {best[1]['rho_degree']:+.3f}   "
          f"random baseline: {best[1]['rho_random']:+.3f}")
    if best[1]["rho_model"] <= best[1]["rho_random"] + 0.05:
        print("  -> the learnt ranking does NOT beat random within a context.")
        print("     The aggregate spread numbers in stage 5 cannot be attributed to learning.")
    elif best[1]["rho_model"] < best[1]["rho_degree"]:
        print("  -> learning beats random but loses to plain degree within a context.")
    else:
        print("  -> learning beats both random and degree within a context.")
    print()

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "graph": args.graph, "n": n, "dataset": stats, "aggregate": results,
            "per_context": per_context_rows,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
