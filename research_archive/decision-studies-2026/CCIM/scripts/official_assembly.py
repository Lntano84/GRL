"""Assemble the authors' package tree from the downloaded sources and probe what actually runs.

Every edit applied to the published sources is listed in :data:`PATCHES` and written to the artifact, so
"what did you change to make it run" is answerable without diffing by hand.  A patch that would change
*algorithmic* behaviour is not allowed here -- those are recorded as reproduction deviations instead.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "_ref"
PKG = ROOT / "_official"

# source file -> destination inside the package
LAYOUT = {
    "gsrl_diffpool_utils.py": "diffpool/utils.py",
    "gsrl_diffpool_set2set.py": "diffpool/set2set.py",
    "gsrl_diffpool_encoders.py": "diffpool/encoders.py",
    "gsrl_rl_alg_utils.py": "rl_alg/utils.py",
    "gsrl_rl_alg_replay.py": "rl_alg/replay.py",
    "gsrl_rl_alg_dqn.py": "rl_alg/dqn.py",
    "gsrl_icm.py": "icm.py",
    "gsrl_utils.py": "utils.py",
    "gsrl_expts_net_env.py": "expts/net_env.py",
    "gsrl_expts_influence.py": "expts/influence.py",
    "gsrl_expts_change_baseline.py": "expts/change_baseline.py",
    "gsrl_expts_gengraph.py": "expts/gengraph.py",
    "gsrl_expts_gendata.py": "expts/gendata.py",
    "gsrl_expts_graphembed.py": "expts/graphembed.py",
    "gsrl_train.py": "train.py",
    # the authors' random-walk generator (needs pandas/joblib/tqdm, all present).  Only the Word2Vec
    # trainer inside ge/models/deepwalk.py needs gensim, which is not installable in this environment.
    "gsrl_ge_alias.py": "ge/alias.py",
    "gsrl_ge_utils.py": "ge/utils.py",
    "gsrl_ge_walker.py": "ge/walker.py",
    "gsrl_ge_models_deepwalk.py": "ge/models/deepwalk.py",
}

# Compatibility patches only: names that moved or were removed between library versions.
PATCHES = [
    ("numba import", "from numba import jit", "def jit(*a, **k):\n"
                                              "    def deco(f): return f\n"
                                              "    return deco if not a or not callable(a[0]) else a[0]"),
    ("np.int removed in numpy 2", "dtype=np.int)", "dtype=int)"),
    ("init.xavier_uniform -> _", "init.xavier_uniform(", "init.xavier_uniform_("),
    ("init.constant -> _", "init.constant(", "init.constant_("),
    ("nx.to_numpy_matrix removed", "nx.to_numpy_matrix(", "nx.to_numpy_array("),
    ("matplotlib/PIL/tensorboard not installed", "import matplotlib.pyplot as plt", "plt = None"),
    # Python >= 3.11 removed random.sample over a set; the authors call it on env.possible_actions
    ("random.sample on a set removed", "random.sample(self.possible_actions,1)[0]",
     "random.sample(sorted(self.possible_actions),1)[0]"),
    ("random.sample on a set removed", "random.sample(s1,1)[0]", "random.sample(sorted(s1),1)[0]"),
    ("random.sample on a set removed", "random.sample(list(self.fullgraph.neighbors(u)),1)[0]",
     "random.sample(sorted(self.fullgraph.neighbors(u)),1)[0]"),
]


def main() -> int:
    if PKG.exists():
        shutil.rmtree(PKG)
    for sub in ("diffpool", "rl_alg", "expts", "ge", "ge/models"):
        (PKG / sub).mkdir(parents=True, exist_ok=True)
    (PKG / "__init__.py").write_text("", encoding="utf-8")
    for sub in ("diffpool", "rl_alg", "expts", "ge", "ge/models"):
        (PKG / sub / "__init__.py").write_text("", encoding="utf-8")

    applied = []
    for src, dst in LAYOUT.items():
        text = (REF / src).read_text(encoding="utf-8", errors="ignore")
        for name, old, new in PATCHES:
            if old in text:
                text = text.replace(old, new)
                applied.append({"patch": name, "file": dst, "from": old, "to": new})
        (PKG / dst).write_text(text, encoding="utf-8")
    for name, old, _ in PATCHES:
        pass
    # imports of the authors' helper modules must resolve inside the assembled package
    for path in PKG.rglob("*.py"):
        t = path.read_text(encoding="utf-8")
        t = re.sub(r"^import utils as utls", "import utils as utls", t, flags=re.M)
        path.write_text(t, encoding="utf-8")

    print("=" * 100)
    print("  ASSEMBLED THE AUTHORS' PACKAGE")
    print("=" * 100)
    for dst in LAYOUT.values():
        p = PKG / dst
        print(f"    {dst:<32} {p.stat().st_size:>7} bytes")
    print(f"\n  compatibility patches applied ({len(applied)} sites):")
    seen = set()
    for a in applied:
        key = (a["patch"], a["file"])
        if key in seen:
            continue
        seen.add(key)
        print(f"    - {a['patch']:<42} in {a['file']}")

    # ---------------------------------------------------------------- probes
    # the authors' modules use absolute imports (`import diffpool.utils`, `from icm import ...`),
    # so the assembled package root itself has to be importable, not just the repository root
    sys.path.insert(0, str(PKG))
    import numpy as np
    import torch
    probes = {}

    def probe(label, fn):
        try:
            detail = fn()
            probes[label] = {"ok": True, "detail": detail}
            print(f"\n  [PASS] {label}\n         {detail}")
        except Exception as exc:                       # noqa: BLE001
            probes[label] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                             "trace": traceback.format_exc()[-900:]}
            print(f"\n  [FAIL] {label}\n         {type(exc).__name__}: {exc}")

    def p_diffpool():
        from diffpool.encoders import GcnEncoderGraph
        import inspect
        sig = str(inspect.signature(GcnEncoderGraph.__init__))
        n, dim = 12, 16
        adj = np.eye(n, dtype=np.float32)
        for i in range(n - 1):
            adj[i, i + 1] = adj[i + 1, i] = 1.0
        x = np.random.randn(n, dim).astype(np.float32)
        # the authors' own call sites reshape to a batch of one before the encoder (dqn.py lines 525-526)
        enc = GcnEncoderGraph(dim, dim, dim, 1, 2, use_cuda=False)
        out = enc(torch.from_numpy(x).reshape((1,) + x.shape),
                  torch.from_numpy(adj).reshape((1,) + adj.shape))
        return f"signature {sig} | forward output shape {tuple(out.shape)}"

    def p_dqn():
        from rl_alg.dqn import DQNTrainer
        import inspect
        return f"signature {str(inspect.signature(DQNTrainer.__init__))[:180]}"

    def p_icm():
        import icm
        import networkx as nx
        g = nx.karate_club_graph()
        for u, v in g.edges():
            g[u][v]["p"] = 0.1
        live = icm.sample_live_icm(g, 20)
        return f"sampled {len(live)} live-edge graphs from karate"

    def p_netenv():
        import networkx as nx
        from expts.net_env import NetworkEnv
        g = nx.gnp_random_graph(40, 0.1, seed=1)
        env = NetworkEnv(g, seeds=[0, 1], max_T=3, opt_reward=0)
        a = env.sample_action()
        env.step(a)
        return f"NetworkEnv ran one step, frontier size {len(env.possible_actions)}"

    def p_walker():
        import time
        import networkx as nx
        from ge.walker import RandomWalker
        g = nx.gnp_random_graph(120, 0.06, seed=3)
        g = nx.relabel_nodes(g, {n: str(n) for n in g.nodes()})
        t = time.perf_counter()
        w = RandomWalker(g, p=1, q=1)
        sents = w.simulate_walks(num_walks=80, walk_length=10, workers=1, verbose=0)
        dt = time.perf_counter() - t
        return (f"{len(sents)} walks / {sum(len(s) for s in sents)} tokens on a 120-node graph "
                f"in {dt:.2f} s")

    probe("diffpool.encoders.GcnEncoderGraph (authors' graph representation)", p_diffpool)
    probe("rl_alg.dqn.DQNTrainer (authors' Q-network)", p_dqn)
    probe("icm.py (authors' IC sampling)", p_icm)
    probe("expts/net_env.py (authors' environment)", p_netenv)
    probe("ge.walker.RandomWalker (authors' random walks, DeepWalk's walk stage)", p_walker)

    artifact = {"layout": LAYOUT, "patches_applied": applied, "probes": probes,
                "note": ("compatibility patches only; none of them changes an algorithm.  Anything that "
                         "would is recorded as a reproduction deviation instead.")}
    (ROOT / "results" / "official_assembly.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  artifact: results/official_assembly.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
