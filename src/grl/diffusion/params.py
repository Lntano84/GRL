"""Config-driven resolution of overexposure diffusion parameters.

The task brief requires that diffusion parameters be controllable from the experiment config and
never hard-coded in the simulation.  :func:`grl.diffusion.overexposure.run_overexposure` takes an
explicit ``activation_mode`` and pre-sampled windows, so this module sits one level above it and
turns a config dict into a validated, explicit parameter set.

Config schema (all keys optional; defaults reproduce the source model's setting)::

    overexposure:
      activation_mode: deterministic   # or stochastic
      overexposure_free: false         # clamp theta_tau to 1 (no-overexposure limit)
      window_lo: 0.0                   # lower bound of the tau support, see note below
      mc_runs: 200                     # evaluation budget, used by oracles/runners
      random_seed: 20260917

``window_lo`` note
------------------
``theta_tau`` is drawn as ``kappa + U[0, 1 - kappa]``, i.e. supported on ``[kappa, 1]``.
``window_lo`` compresses that support into ``[max(kappa, window_lo), 1]``, which makes
overexposure easier to trigger.  It exists because the audit showed the process's percolation
scale is highly sensitive to the window support, and experiments need to vary it explicitly
rather than by editing the sampler.  ``window_lo = 0.0`` is the source model's setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .overexposure import ACTIVATION_MODES, DETERMINISTIC

DEFAULT_MC_RUNS = 200
DEFAULT_RANDOM_SEED = 20260917


@dataclass(frozen=True)
class OverexposureParams:
    """Validated overexposure diffusion parameters."""

    activation_mode: str = DETERMINISTIC
    overexposure_free: bool = False
    window_lo: float = 0.0
    mc_runs: int = DEFAULT_MC_RUNS
    random_seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        if self.activation_mode not in ACTIVATION_MODES:
            raise ValueError(
                f"activation_mode must be one of {ACTIVATION_MODES}, "
                f"got {self.activation_mode!r}"
            )
        if not 0.0 <= self.window_lo < 1.0:
            raise ValueError(f"window_lo must lie in [0, 1), got {self.window_lo}")
        if self.mc_runs <= 0:
            raise ValueError(f"mc_runs must be positive, got {self.mc_runs}")
        if self.overexposure_free and self.window_lo != 0.0:
            # Clamping tau to 1 and compressing the tau support are contradictory requests;
            # silently honouring one of them would make results unexplainable.
            raise ValueError(
                "overexposure_free=True clamps theta_tau to 1, so window_lo must be 0.0"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "activation_mode": self.activation_mode,
            "overexposure_free": self.overexposure_free,
            "window_lo": self.window_lo,
            "mc_runs": self.mc_runs,
            "random_seed": self.random_seed,
        }


def resolve_overexposure_params(config: Mapping[str, Any] | None) -> OverexposureParams:
    """Read the ``overexposure`` block of a loaded config, with validated defaults.

    ``None`` or a missing block yields the default (source-model) parameters rather than an
    error, so existing configs keep working unchanged.
    """
    if not config:
        return OverexposureParams()
    block = config.get("overexposure") or {}
    if not isinstance(block, Mapping):
        raise TypeError(f"config['overexposure'] must be a mapping, got {type(block).__name__}")

    diffusion = config.get("diffusion") or {}
    if not isinstance(diffusion, Mapping):
        diffusion = {}

    # ``mc_runs`` is read from the overexposure block first, then the diffusion block, so an
    # existing config that only sets ``diffusion.mc_runs_eval`` still controls the budget.
    mc_runs = block.get("mc_runs", diffusion.get("mc_runs_eval", DEFAULT_MC_RUNS))
    random_seed = block.get(
        "random_seed",
        (config.get("experiment") or {}).get("random_seed", DEFAULT_RANDOM_SEED),
    )

    return OverexposureParams(
        activation_mode=str(block.get("activation_mode", DETERMINISTIC)),
        overexposure_free=bool(block.get("overexposure_free", False)),
        window_lo=float(block.get("window_lo", 0.0)),
        mc_runs=int(mc_runs),
        random_seed=int(random_seed),
    )
