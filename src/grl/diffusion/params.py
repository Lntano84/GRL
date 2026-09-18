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

#: The source model's law: ``(kappa, tau)`` uniform on the 2-simplex
#: ``{0 <= kappa <= tau <= 1}``, i.e. the order statistics of two independent uniforms.  Marginals
#: are ``F_kappa(x) = 2x - x^2`` and ``F_tau(x) = x^2``.
LAW_SIMPLEX = "simplex"

#: The source model's law with the upper threshold clamped to 1, keeping kappa's marginal at
#: ``2x - x^2``.  This removes the overexposure branch while leaving the lower-threshold
#: distribution untouched, so it isolates "overexposure off" from "different threshold law".
LAW_SIMPLEX_TAU_CLAMPED = "simplex_tau_clamped_to_one"

#: ``kappa ~ U[0, 1]`` with ``tau = 1``.  This is the genuine uniform-threshold linear-threshold
#: reading, and it is a DIFFERENT process from the clamped-simplex law above because kappa's
#: marginal differs.  It exists solely as an explicit control; the audit required the two not to be
#: conflated, and the original ``overexposure_free`` flag conflated them.
LAW_UNIFORM_LT = "uniform_lt_tau_one"

#: The source model's law with the lower end of tau's support raised, which makes overexposure
#: easier to trigger.  An intervention knob, not a degeneracy path.
LAW_SIMPLEX_TAU_RAISED = "simplex_tau_support_raised"

THRESHOLD_LAWS = (LAW_SIMPLEX, LAW_SIMPLEX_TAU_CLAMPED, LAW_UNIFORM_LT, LAW_SIMPLEX_TAU_RAISED)


@dataclass(frozen=True)
class OverexposureParams:
    """Validated overexposure diffusion parameters.

    This fixes the *diffusion* half of the frozen model contract.  The *objective* half -- target
    set and seed eligibility -- lives in :class:`grl.diffusion.contract.ObjectiveContract`, because
    it decides which nodes are counted rather than how the process runs.
    """

    activation_mode: str = DETERMINISTIC
    threshold_law: str = LAW_SIMPLEX
    window_lo: float = 0.0
    mc_runs: int = DEFAULT_MC_RUNS
    random_seed: int = DEFAULT_RANDOM_SEED

    def __post_init__(self) -> None:
        if self.activation_mode not in ACTIVATION_MODES:
            raise ValueError(
                f"activation_mode must be one of {ACTIVATION_MODES}, "
                f"got {self.activation_mode!r}"
            )
        if self.threshold_law not in THRESHOLD_LAWS:
            raise ValueError(
                f"threshold_law must be one of {THRESHOLD_LAWS}, got {self.threshold_law!r}"
            )
        if not 0.0 <= self.window_lo < 1.0:
            raise ValueError(f"window_lo must lie in [0, 1), got {self.window_lo}")
        if self.mc_runs <= 0:
            raise ValueError(f"mc_runs must be positive, got {self.mc_runs}")
        if self.threshold_law == LAW_SIMPLEX_TAU_RAISED and self.window_lo <= 0.0:
            raise ValueError(
                f"threshold_law={LAW_SIMPLEX_TAU_RAISED} needs a positive window_lo"
            )
        if self.threshold_law != LAW_SIMPLEX_TAU_RAISED and self.window_lo != 0.0:
            raise ValueError(
                f"window_lo only applies to threshold_law={LAW_SIMPLEX_TAU_RAISED}; "
                f"got law={self.threshold_law} with window_lo={self.window_lo}"
            )

    @property
    def overexposure_free(self) -> bool:
        """True when the upper threshold cannot be crossed.

        Kept as a derived property rather than a settable field so that the three degeneracy paths
        cannot be collapsed into one boolean again.  Both ``simplex_tau_clamped_to_one`` and
        ``uniform_lt_tau_one`` disable overexposure, but they use different kappa marginals.
        """
        return self.threshold_law in (LAW_SIMPLEX_TAU_CLAMPED, LAW_UNIFORM_LT)

    def as_dict(self) -> dict[str, Any]:
        return {
            "activation_mode": self.activation_mode,
            "threshold_law": self.threshold_law,
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

    # ``threshold_law`` is preferred.  ``overexposure_free: true`` is still accepted for backward
    # compatibility and maps to the clamped-simplex law, which is the one that flag always meant.
    if "threshold_law" in block:
        law = str(block["threshold_law"])
    elif block.get("overexposure_free"):
        law = LAW_SIMPLEX_TAU_CLAMPED
    elif float(block.get("window_lo", 0.0)) > 0.0:
        law = LAW_SIMPLEX_TAU_RAISED
    else:
        law = LAW_SIMPLEX

    return OverexposureParams(
        activation_mode=str(block.get("activation_mode", DETERMINISTIC)),
        threshold_law=law,
        window_lo=float(block.get("window_lo", 0.0)),
        mc_runs=int(mc_runs),
        random_seed=int(random_seed),
    )
