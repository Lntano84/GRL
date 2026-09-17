from .marginal import BatchedMonteCarloMarginalOracle, LearnedMarginalOracle, OracleStats
from .overexposure_mc import (
    OverexposureMonteCarloOracle,
    OverexposureOracleStats,
)

__all__ = [
    # Independent-cascade oracle (live-edge sampling; NOT valid for overexposure).
    "OracleStats",
    "BatchedMonteCarloMarginalOracle",
    "LearnedMarginalOracle",
    # Overexposure threshold-window oracle (state-tracking cascade simulation).
    "OverexposureMonteCarloOracle",
    "OverexposureOracleStats",
]
