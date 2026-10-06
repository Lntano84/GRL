"""Small independent checks of the new global-survey probe's information boundary."""
import sys
import unittest
from pathlib import Path

import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import attribute_voi_probe as probe  # noqa: E402


class AttributeSurveyProbeTest(unittest.TestCase):
    def test_data_contract_and_day(self):
        graph, ids, classes, i, j, same, full, pair, info = probe.load_graph()
        self.assertEqual((info["n"], info["m"], info["rows"]), (329, 2573, 47338))
        self.assertEqual(len(set(ids)), 329)
        self.assertEqual(len(i), 329 * 328 // 2)
        self.assertEqual(int(full.sum()), 2573)
        self.assertEqual(graph.number_of_nodes(), 329)
        self.assertEqual(pair.shape, (329, 329))

    def test_each_pair_is_counted_once_and_hidden_changes_are_invisible(self):
        n = 12
        i, j = np.triu_indices(n, 1)
        pair = np.zeros((n, n), dtype=np.int32)
        pair[i, j] = np.arange(len(i))
        pair[j, i] = np.arange(len(i))
        classes = np.array(["a"] * 6 + ["b"] * 6)
        same = classes[i] == classes[j]
        queried = np.zeros(n, dtype=bool)
        queried[:9] = True
        known = probe.known_pairs(queried, i, j)
        first = np.zeros(len(i), dtype=bool)
        first[pair[0, 1]] = True
        first[pair[0, 9]] = True
        twin = first.copy()
        twin[pair[9, 10]] = True  # Neither endpoint has been queried.
        q1, rates1 = probe.posterior(known, first, same)
        q2, rates2 = probe.posterior(known, twin, same)
        self.assertTrue(np.array_equal(q1, q2))
        self.assertEqual(rates1, rates2)
        self.assertEqual(rates1["same"]["edges"] + rates1["different"]["edges"], 2)

        state = {"split": "twin", "number": 0, "queried": queried, "known": known, "q": q1}
        base1 = probe.solve(n, i, j, q1, probe.seed("same-solver"))
        base2 = probe.solve(n, i, j, q2, probe.seed("same-solver"))
        self.assertEqual(base1, base2)
        rules1 = probe.simple_rules(state, n, i, j, first, pair, base1)
        rules2 = probe.simple_rules(state, n, i, j, twin, pair, base2)
        self.assertEqual(rules1, rules2)
        pool = probe.candidate_pool(rules1, ~queried, "twin", 0)
        self.assertEqual(pool, probe.candidate_pool(rules2, ~queried, "twin", 0))
        old_draws, old_eval = probe.N_DRAW, probe.N_PLAN_EVAL_MC
        try:
            probe.N_DRAW, probe.N_PLAN_EVAL_MC = 2, 4
            picked1, scores1 = probe.plan_choice(state, pool, base1, n, i, j, same, first)
            picked2, scores2 = probe.plan_choice(state, pool, base2, n, i, j, same, twin)
        finally:
            probe.N_DRAW, probe.N_PLAN_EVAL_MC = old_draws, old_eval
        self.assertEqual((picked1, scores1), (picked2, scores2))
        # After querying 9, the formerly hidden pair (9, 10) is a legitimate observation.
        self.assertFalse(np.array_equal(probe.after_query(state, 9, first, i, j, same),
                                        probe.after_query(state, 9, twin, i, j, same)))

    def test_confirmed_edge_and_nonedge_update(self):
        i, j = np.triu_indices(4, 1)
        same = np.array([True, False, False, False, False, True])
        known = np.array([True, True, False, False, False, False])
        edges = np.array([True, False, False, False, False, False])
        q, rates = probe.posterior(known, edges, same)
        self.assertEqual(rates["same"]["rate"], 2 / 3)
        self.assertEqual(rates["different"]["rate"], 1 / 3)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[1], 0.0)

    def test_weighted_greedy_covers_components_once(self):
        objective = object.__new__(probe.WeightedObjective)
        objective.n, objective.samples = 4, 2
        objective.labels = np.array([[0, 0, 1, 1], [0, 1, 1, 2]], dtype=np.int32)
        objective.sizes = np.array([[2, 2, 2, 2], [1, 2, 2, 1]], dtype=np.int32)
        self.assertEqual(objective.greedy(k=2), [1, 3])


if __name__ == "__main__":
    unittest.main()
