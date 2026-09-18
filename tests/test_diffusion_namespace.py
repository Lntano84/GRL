"""Guard against silent symbol collisions between the diffusion submodules.

A real bug motivated this file: ``grl/diffusion/__init__.py`` re-exported the
overexposure module's ``estimate_spread_over_configs`` under the same name as the
independent-cascade estimator.  Because the later import wins, ``from grl.diffusion
import estimate_spread_over_configs`` silently returned the overexposure function, and
every IC-vs-overexposure comparison ended up evaluating overexposure against itself
(means agreed to three decimals while single realizations differed).

These tests fail loudly if that class of collision ever returns.
"""

from __future__ import annotations

import grl.diffusion as diffusion
from grl.diffusion import estimate_spread_over_configs as ic_paired
from grl.diffusion import overexposure as oe


def test_ic_and_overexposure_estimators_are_distinct_objects():
    assert ic_paired is not oe.estimate_overexposure_spread_over_configs
    assert ic_paired.__module__ == "grl.diffusion.independent_cascade"
    assert oe.estimate_overexposure_spread_over_configs.__module__ == "grl.diffusion.overexposure"


def test_ic_estimators_resolve_to_the_ic_module():
    assert diffusion.estimate_spread.__module__ == "grl.diffusion.independent_cascade"
    assert diffusion.estimate_spread_over_configs.__module__ == "grl.diffusion.independent_cascade"
    assert diffusion.estimate_marginal_gain.__module__ == "grl.diffusion.independent_cascade"
    assert diffusion.estimate_marginal_gains.__module__ == "grl.diffusion.independent_cascade"
    assert diffusion.run_independent_cascade.__module__ == "grl.diffusion.independent_cascade"


def test_overexposure_estimators_resolve_to_the_overexposure_module():
    assert diffusion.estimate_overexposure_spread.__module__ == "grl.diffusion.overexposure"
    assert (
        diffusion.estimate_overexposure_spread_over_configs.__module__
        == "grl.diffusion.overexposure"
    )
    assert (
        diffusion.estimate_overexposure_marginal_gain.__module__
        == "grl.diffusion.overexposure"
    )
    assert (
        diffusion.estimate_overexposure_marginal_gains.__module__
        == "grl.diffusion.overexposure"
    )
    assert diffusion.run_overexposure.__module__ == "grl.diffusion.overexposure"


OVEREXPOSURE_NAMES = {
    "run_overexposure", "sample_threshold_windows", "positive_activation_probability",
    "marginal_positive_probability", "OverexposureRun",
    "INACTIVE", "POSITIVE", "NEGATIVE", "DETERMINISTIC", "STOCHASTIC", "ACTIVATION_MODES",
}
CONTRACT_NAMES = {
    "ContractViolation", "ObjectiveContract", "ModelContract", "TargetedObjective",
    "build_contract", "OverexposureParams", "resolve_overexposure_params",
}
IC_NAMES = {
    "run_independent_cascade",
    "estimate_spread",
    "estimate_spread_over_configs",
    "estimate_marginal_gain",
    "estimate_marginal_gains",
}


def test_no_public_name_is_claimed_by_both_modules():
    """Every exported name must map to exactly one module's definition.

    The sets are listed explicitly rather than inferred, so adding a new module to the package
    forces this test to be updated instead of silently widening what counts as "IC".
    """
    covered = OVEREXPOSURE_NAMES | CONTRACT_NAMES | IC_NAMES
    unclassified = set(diffusion.__all__) - covered - {
        name for name in diffusion.__all__ if name.startswith("estimate_overexposure")
    }
    assert not unclassified, (
        f"these exported names are not classified by module: {sorted(unclassified)}; "
        f"add them to OVEREXPOSURE_NAMES, CONTRACT_NAMES or IC_NAMES"
    )
    for name in set(diffusion.__all__) - OVEREXPOSURE_NAMES - CONTRACT_NAMES - IC_NAMES:
        if name.startswith("estimate_overexposure"):
            continue
        obj = getattr(diffusion, name)
        module = getattr(obj, "__module__", "")
        assert module == "grl.diffusion.independent_cascade", (name, module)


def test_contract_names_come_from_the_contract_module():
    for name in CONTRACT_NAMES:
        assert name in diffusion.__all__, name
        obj = getattr(diffusion, name)
        module = getattr(obj, "__module__", None)
        # OverexposureParams is defined in params.py and re-exported through contract; either is a
        # single definition site, which is what matters.
        assert module in {"grl.diffusion.contract", "grl.diffusion.params"}, (name, module)


def test_models_actually_disagree_on_a_toy_graph():
    """End-to-end guard: the two models must not be numerically identical."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edge(0, 1, weight=0.5)
    graph.add_edge(1, 2, weight=0.5)

    ic = ic_paired(graph, [[0]], 300, 1)[0]
    oe_est = oe.estimate_overexposure_spread_over_configs(graph, [[0]], 300, 1)[0]
    assert ic["mean"] != oe_est["mean"], (ic, oe_est)
