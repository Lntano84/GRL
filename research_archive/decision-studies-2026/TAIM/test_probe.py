import unittest

import numpy as np

from probe import IC, State, cheap_choice, ff_choice, rng_for


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.model = IC("football", 0.1)

    def test_world_reproducible(self):
        self.assertEqual(self.model.world(rng_for("test", 1)), self.model.world(rng_for("test", 1)))

    def test_no_retransmission_from_spent_active(self):
        model = self.model
        world = [0] * model.n
        world[0] = 1 << 1
        world[1] = 1 << 2
        final, wave = model.spread(1 << 0, 0, [], world, 2)
        self.assertEqual(final, 1 << 0)
        self.assertEqual(wave, 0)
        final, wave = model.spread(1 << 0, 1 << 0, [], world, 2)
        self.assertEqual(final & 7, 7)
        self.assertEqual(wave, 1 << 2)

    def test_last_round_expectation(self):
        model = self.model
        active = 1 << 0
        frontier = active
        actual = model.expected_last(active, frontier, [1])
        samples = []
        rng = rng_for("expected_last")
        for _ in range(4000):
            final, _ = model.spread(active, frontier, [1], model.world(rng), 1)
            samples.append(final.bit_count())
        self.assertLess(abs(np.mean(samples) - actual), 0.12)

    def test_budget_conditioned_and_ff_order(self):
        s = State(1, 1, "s", "football", 0.1)
        self.assertEqual(cheap_choice("balanced", 1, s, []), 1)
        self.assertEqual(cheap_choice("balanced", 2, s, []), 1)
        self.assertEqual(cheap_choice("frontier_0.2_1.0", 1, s, []), 0)
        self.assertEqual(cheap_choice("frontier_0.2_1.0", 2, s, []), 2)
        self.assertEqual(ff_choice([0.7, 0.3], 0.5), 1)
        self.assertEqual(ff_choice([0.7, 0.6], 0.5), 2)

    def test_prefix_shared_across_budgets(self):
        model = self.model
        active, frontier = model.spread(0, 0, [0], model.world(rng_for("state")), 1)
        s = State(active, frontier, "s", "football", 0.1)
        worlds = [model.world(rng_for("greedy", s.sid, j)) for j in range(8)]
        one = model.greedy_now(s, 1, worlds)
        two = model.greedy_now(s, 2, worlds)
        self.assertEqual(one, two[:1])

    def test_hand_example_budget_flip(self):
        # Two 60-node regions and two external active nodes. The four worlds
        # enumerate the independent 0.5/0.5 external transmissions exactly.
        arcs = []
        for region in range(2):
            center = 2 + 60 * region
            relay = center + 49
            arcs.append((region, center, 0.5))
            arcs.extend((center, center + i, 1.0) for i in range(1, 49))
            arcs.append((center, relay, 1.0))
            arcs.extend((relay, relay + i, 1.0) for i in range(1, 11))
        model = IC.from_arcs(122, arcs)
        s = State(3, 3, "hand", "hand_example", -1.0)
        worlds = []
        for first in (False, True):
            for second in (False, True):
                rows = [0] * 122
                for u, v, p in arcs:
                    if p == 1.0 or (u == 0 and first) or (u == 1 and second):
                        rows[u] |= 1 << v
                worlds.append(rows)
        prefix = model.greedy_now(s, 2, worlds)
        self.assertEqual(set(prefix), {2, 62})
        choice1, values1 = model.sof(s, prefix[:1], 1, worlds)
        choice2, values2 = model.sof(s, prefix, 2, worlds)
        self.assertEqual(choice1, 0)
        self.assertEqual(choice2, 2)
        self.assertEqual(values1[0], 92.0)  # includes the two external nodes
        self.assertEqual(values1[1], 87.0)
        self.assertEqual(values2[0], 112.0)
        self.assertEqual(values2[2], 122.0)


if __name__ == "__main__":
    unittest.main()
