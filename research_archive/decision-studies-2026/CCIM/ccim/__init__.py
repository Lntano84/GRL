"""CCIM (complex contagion influence maximization) utilities for the deterministic-setting probe.

Modules
-------
``model``      the K-complex-contagion cascade from Chen et al., IJCAI 2023, in the deterministic
               ``p0 = 1, p1 = 0`` setting the paper's basic experiments use
``baselines``  degree and true-reward greedy, with sigma-evaluation accounting
``search``     a budget-controlled non-learning combinatorial search
``dqn``        the original double-DQN with terminal handling and legal action masking fixed
"""
