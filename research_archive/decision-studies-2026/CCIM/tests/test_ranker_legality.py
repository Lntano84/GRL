"""Gates for the legal-information ranker.

The load-bearing one is :func:`test_features_scores_and_choice_ignore_the_hidden_graph`.  Two graphs agree on
everything the agent has surveyed or can see and differ only in edges among nodes it has never touched.  The
candidate pool, the features, the ranker's scores and its final choice must all be identical.  If they are
not, the ranker is reading structure it could not legally have, and the held-out result means nothing.

Twins are built by copying a **real** graph and rewiring only the unrevealed part, rather than by inventing
small graphs: the state generator draws seed indices with ``choice(number_of_nodes())``, so a graph whose
labels are not ``0..n-1`` is not a valid input to it.

A relabelling test is deliberately **not** included: the survey walk breaks ties by node id
(``sorted(frontier)``), so a relabelling changes the trajectory itself and is not an
information-preserving transformation.
"""

from __future__ import annotations

import itertools
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "_official"))

import legal_ranker as LR  # noqa: E402
import xplore_learn as SIMP  # noqa: E402

NS, I = "gate", 0


def _real_graph():
    return pickle.loads((SIMP.DATA / f"{LR.GRAPH}.pkl").read_bytes())


def _state(g) -> dict:
    """A state carrying the fields ``collect`` would produce, so the ranker can be exercised."""
    st = LR.build_state(g, NS, I)
    st["key"] = LR.visible_key(st["env"])
    rows = []
    for c in LR.candidates(st["env"], NS, I):
        f = LR.features(st["env"], st, c)
        # deterministic stand-in label: these gates test features/scores, not learning quality
        rows.append({"cand": int(c), "feat": f, "sim": float(-f[0]), "chosen": [], "conf": None})
    st["rows"] = rows
    return st


def _hidden_twin(g, st):
    """Same graph, plus edges among nodes that the state has never revealed."""
    visible = set(st["env"].active) | set(st["env"].possible_actions)
    hidden = [n for n in g.nodes() if n not in visible]
    g2 = g.copy()
    for a, b in itertools.islice(itertools.combinations(hidden, 2), 4000):
        g2.add_edge(a, b)
    return g2, len(hidden)


def test_features_scores_and_choice_ignore_the_hidden_graph():
    g = _real_graph()
    s1 = _state(g)
    g2, n_hidden = _hidden_twin(g, s1)
    s2 = _state(g2)
    assert n_hidden > 5, "the twin has almost no hidden nodes, so the test would be vacuous"
    assert s1["key"] == s2["key"], "the twins must share a visible state"
    assert s1["rows"] and s2["rows"], "no candidates were produced"
    assert [r["cand"] for r in s1["rows"]] == [r["cand"] for r in s2["rows"]], \
        "the candidate pool depends on hidden structure"
    assert [r["feat"] for r in s1["rows"]] == [r["feat"] for r in s2["rows"]], \
        "features differ although the visible state is identical"

    rng = np.random.default_rng(0)
    w, b = rng.normal(size=len(LR.FEATURES)), 0.37
    mean, std = np.zeros(len(LR.FEATURES)), np.ones(len(LR.FEATURES))
    sc1 = LR.ranker_scores(w, b, mean, std, [r["feat"] for r in s1["rows"]])
    sc2 = LR.ranker_scores(w, b, mean, std, [r["feat"] for r in s2["rows"]])
    assert np.allclose(sc1, sc2), "ranker scores depend on structure the agent never observed"
    assert int(np.argmax(sc1)) == int(np.argmax(sc2)), "the final choice depends on hidden structure"


def test_training_on_twins_gives_identical_weights():
    g = _real_graph()
    s1 = _state(g)
    g2, _ = _hidden_twin(g, s1)
    s2 = _state(g2)
    w1, b1, m1, d1 = LR.ranker_train([s1])
    w2, b2, m2, d2 = LR.ranker_train([s2])
    assert np.allclose(w1, w2) and b1 == b2 and np.allclose(m1, m2) and np.allclose(d1, d2), \
        "training on twin graphs produced different models"


def test_ranker_training_has_no_access_to_confirmation_labels():
    import inspect
    assert list(inspect.signature(LR.ranker_train).parameters) == ["states"]
    assert "conf" not in inspect.getsource(LR.ranker_train), "training touched a confirmation field"


def test_standardisation_is_fitted_on_the_supplied_states_only():
    st = _state(_real_graph())
    w, b, mean, std = LR.ranker_train([st])
    X = np.array([r["feat"] for r in st["rows"]])
    assert np.allclose(mean, X.mean(axis=0))
    assert np.all(std > 0)
