from .independent_cascade import (
    estimate_marginal_gain,
    estimate_marginal_gains,
    estimate_spread,
    estimate_spread_over_configs,
    run_independent_cascade,
)
from .contract import (
    ContractViolation,
    ModelContract,
    ObjectiveContract,
    TargetedObjective,
    build_contract,
)
from .params import OverexposureParams, resolve_overexposure_params
from .overexposure import (
    ACTIVATION_MODES,
    DETERMINISTIC,
    INACTIVE,
    NEGATIVE,
    POSITIVE,
    STOCHASTIC,
    OverexposureRun,
    estimate_marginal_gain as estimate_overexposure_marginal_gain,
    estimate_marginal_gains as estimate_overexposure_marginal_gains,
    estimate_overexposure_spread_over_configs,
    estimate_spread as estimate_overexposure_spread,
    marginal_positive_probability,
    positive_activation_probability,
    run_overexposure,
    sample_threshold_windows,
)

__all__ = [
    # Independent cascade.  ``estimate_spread`` / ``estimate_spread_over_configs`` /
    # ``estimate_marginal_gain(s)`` are the IC estimators; every overexposure analogue is
    # prefixed with ``overexposure`` so the two can never shadow each other.
    "run_independent_cascade",
    "estimate_spread",
    "estimate_spread_over_configs",
    "estimate_marginal_gain",
    "estimate_marginal_gains",
    # Overexposure threshold-window model
    "INACTIVE",
    "POSITIVE",
    "NEGATIVE",
    "DETERMINISTIC",
    "STOCHASTIC",
    "ACTIVATION_MODES",
    "OverexposureRun",
    "positive_activation_probability",
    "marginal_positive_probability",
    "sample_threshold_windows",
    "run_overexposure",
    "estimate_overexposure_spread",
    "estimate_overexposure_spread_over_configs",
    "estimate_overexposure_marginal_gain",
    "estimate_overexposure_marginal_gains",
    # Frozen model contract: target set, seed eligibility, budget semantics, threshold law.
    "ContractViolation",
    "ObjectiveContract",
    "ModelContract",
    "TargetedObjective",
    "build_contract",
    "OverexposureParams",
    "resolve_overexposure_params",
]
