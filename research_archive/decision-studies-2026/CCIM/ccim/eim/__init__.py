"""EIM -- exploratory influence maximization with *attributes* (a separate research line).

This package is deliberately independent of ``ccim`` (the known-graph CCIM line) and of
``ccim.xplore`` (the frontier-only adapted Geometric-DQN line).  Nothing here imports their
objective, their environment or their checkpoints: the scenario below changes the access privilege,
so reusing their numbers would silently mix two different problems.

The scenario, in one paragraph.  An organisation holds a roster of members and a pre-registered
attribute for each of them (here: the class group of a high-school student).  It does **not** hold the
contact network.  It may contact any member on the roster and ask them who they interacted with,
which reveals *all* of that member's neighbours at once.  Contacting people is expensive, so only a
handful of surveys can be bought.  When surveying stops, a fixed seed-selection routine is run and the
chosen seeds are activated; the payoff is the influence those seeds achieve on the real network.
Every method -- learned or heuristic -- gets the same roster, the same attributes, the same survey
actions and the same seed solver.  Nothing is restricted to a frontier.

What is being tested is therefore *not* "can a network be reconstructed" but "is a survey's value to
the final seed decision computable, and does exploiting it beat cheap rules that use the same
information".
"""

from __future__ import annotations

__all__ = ["graphdata", "gm", "ic", "rules", "lookahead", "states"]
