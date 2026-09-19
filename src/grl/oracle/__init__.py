from .marginal import BatchedMonteCarloMarginalOracle, LearnedMarginalOracle, OracleStats
from .overexposure_mc import (
    OverexposureMonteCarloOracle,
    OverexposureOracleStats,
)
from .targeted_mc import (
    TargetedMonteCarloOracle,
    TargetedObjectiveError,
    TargetedOracleStats,
    trial_seeds,
)

__all__ = [
    # Independent-cascade oracle (live-edge sampling; NOT valid for overexposure).
    "OracleStats",
    "BatchedMonteCarloMarginalOracle",
    "LearnedMarginalOracle",
    # Overexposure threshold-window oracle over the WHOLE graph.  Reports run.spread, which is not
    # the contracted objective when a target set D is in play; use TargetedMonteCarloOracle there.
    "OverexposureMonteCarloOracle",
    "OverexposureOracleStats",
    # Contracted objective: every number is |positive_at_end ∩ D|.
    "TargetedMonteCarloOracle",
    "TargetedObjectiveError",
    "TargetedOracleStats",
    "trial_seeds",
]
