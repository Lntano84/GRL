"""The one component of the published pipeline that cannot be executed here, re-implemented.

What is missing, and what is not
--------------------------------
``ge/models/deepwalk.py`` is the authors' DeepWalk.  Everything in it except one line runs here: the
random walks come from ``ge/walker.py`` (their ``RandomWalker``, p=q=1), which needs only pandas, joblib
and tqdm -- all installed.  The single unavailable dependency is ``from gensim.models import Word2Vec``.

This module therefore supplies **only the embedding trainer**, and calls the authors' walker for the walks.
It is registered as a reproducibility deviation, not presented as their code.

Deviation, stated precisely
---------------------------
The authors instantiate gensim's Word2Vec with ``sg=1`` (skip-gram), ``hs=1`` (hierarchical softmax),
``window=5``, ``iter=50``, ``size=60``, ``min_count=0``.  This trainer keeps skip-gram, window, epochs,
dimension and min_count, and replaces **hierarchical softmax with negative sampling**.  That is an
algorithmic difference in the objective, so the result may not be described as a fully faithful
reproduction of the published training procedure.

Their own configuration, from ``train.py`` argparse defaults:
``--num_walks 80 --walk_len 10 --win 5 --emb_iters 50 --actiondim 60``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# the authors' assembled package (built by scripts/official_assembly.py) must be importable so that
# this module can call their RandomWalker rather than re-implementing the walk stage
_OFFICIAL = Path(__file__).resolve().parents[2] / "_official"
if str(_OFFICIAL) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL))


def train_skipgram(sentences, embed_size: int = 60, window_size: int = 5, iterations: int = 50,
                   negatives: int = 5, lr: float = 0.025, seed: int = 0, batch: int = 4096,
                   device: str = "cpu") -> dict:
    """Skip-gram with negative sampling over integer-encoded walks.

    ``sentences`` is a list of lists of string node ids, exactly what the authors' walker returns.
    Returns ``{node_id: np.ndarray}``.  Nodes that appear in no walk get a zero vector, matching
    gensim's behaviour for ``min_count=0`` corpora that never saw them.
    """
    vocab: dict[str, int] = {}
    for s in sentences:
        for tok in s:
            if tok not in vocab:
                vocab[tok] = len(vocab)
    if not vocab:
        return {}
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    pairs_i, pairs_j = [], []
    for s in sentences:
        ids = [vocab[t] for t in s]
        for a in range(len(ids)):
            lo, hi = max(0, a - window_size), min(len(ids), a + window_size + 1)
            for b in range(lo, hi):
                if a != b:
                    pairs_i.append(ids[a])
                    pairs_j.append(ids[b])
    if not pairs_i:
        return {k: np.zeros(embed_size, dtype=np.float32) for k in vocab}
    wi = torch.tensor(pairs_i, dtype=torch.long)
    wj = torch.tensor(pairs_j, dtype=torch.long)
    n_vocab, n_pairs = len(vocab), len(wi)

    # subsample frequent tokens, as gensim does by default
    counts = np.bincount(np.concatenate([wi.numpy(), wj.numpy()]), minlength=n_vocab).astype(np.float64)
    freq = counts / counts.sum()
    keep_p = np.minimum(1.0, (np.sqrt(freq / 1e-4) + 1.0) * 1e-4 / np.maximum(freq, 1e-12))
    keep = rng.random(n_pairs) < keep_p[wi.numpy()]
    wi, wj = wi[keep], wj[keep]
    if len(wi) == 0:
        wi, wj = torch.tensor(pairs_i, dtype=torch.long), torch.tensor(pairs_j, dtype=torch.long)

    noise = counts ** 0.75
    noise = noise / noise.sum()
    v_in = (torch.rand(n_vocab, embed_size) - 0.5) / embed_size
    v_out = torch.zeros(n_vocab, embed_size)
    v_in.requires_grad_(True)
    v_out.requires_grad_(True)
    opt = torch.optim.Adam([v_in, v_out], lr=lr)

    n = len(wi)
    for _ in range(iterations):
        perm = torch.randperm(n)
        for start in range(0, n, batch):
            idx = perm[start:start + batch]
            if len(idx) < 2:
                continue
            c = wi[idx]
            o = wj[idx]
            neg = torch.from_numpy(rng.choice(n_vocab, size=(len(idx), negatives), p=noise))
            h = v_in[c]                                     # (b, d)
            pos = (h * v_out[o]).sum(1)
            negs = torch.bmm(v_out[neg], h.unsqueeze(2)).squeeze(2)   # (b, k)
            loss = -(torch.nn.functional.logsigmoid(pos).mean()
                     + torch.nn.functional.logsigmoid(-negs).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
    emb = v_in.detach().numpy()
    return {node: emb[i] for node, i in vocab.items()}


def get_embeds(graph, num_walks: int = 80, walk_length: int = 10, window: int = 5,
               iterations: int = 50, embed_size: int = 60, seed: int = 0) -> dict:
    """The published ``get_embeds``: relabel to strings, walk with the authors' walker, then train.

    Mirrors ``train.py``, which builds ``DeepWalk(g1, num_walks=args.num_walks,
    walk_length=args.walk_len)`` and calls ``graph_model.train(window_size=args.win,
    iter=args.emb_iters, embed_size=action_dim)``, then keys the result by node:
    ``embs[int(n)] = emb1[n]``.  Returns ``{node_id: vector}``.
    """
    import networkx as nx
    from ge.walker import RandomWalker

    g1 = nx.relabel_nodes(graph, {n: str(n) for n in graph.nodes()})
    walker = RandomWalker(g1, p=1, q=1)
    sentences = walker.simulate_walks(num_walks=num_walks, walk_length=walk_length,
                                      workers=1, verbose=0)
    raw = train_skipgram(sentences, embed_size=embed_size, window_size=window,
                         iterations=iterations, seed=seed)
    zero = np.zeros(embed_size, dtype=np.float32)
    return {u: np.asarray(raw.get(str(u), zero), dtype=np.float32) for u in graph.nodes()}
