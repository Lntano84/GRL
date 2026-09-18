"""The Go/No-Go criterion, stated in a form that can actually be evaluated.

Why the criterion had to be restated
------------------------------------
The plan proposed: *quality loss <= 1% AND online-cascade saving >= 30%*.  The **cascade** half is
fine.  The **quality** half is not evaluable in this regime, and this module exists so that the
problem is visible in code rather than discovered after the sweep.

At ``|S|/n >= 20%`` on Congress-Twitter the measured marginal of a candidate is ``+0.70`` target
nodes with a paired standard error of ``0.05``--``0.39``
(``docs/results/signflip_fixedmodel_mc300.json``).  One percent of ``0.70`` is ``0.007`` target
nodes -- one to two orders of magnitude *below* the error of measuring it.  So a relative gate would
pass a method that lost 1% and be statistically blind to one that lost 20%; symmetrically, a method
that is genuinely equivalent could fail it on noise.

What replaces it
----------------
An **absolute tolerance in target-count units**, with a paired confidence interval:

    PASS  iff  the upper bound of the paired CI on (reference - method) is <= ``abs_tolerance``
               AND the method's online cascade cost is <= ``(1 - saving) * reference cost``
    FAIL  otherwise
    UNDECIDED  iff the paired CI is wider than ``abs_tolerance``

``UNDECIDED`` is a first-class outcome, not a cop-out.  A measurement that cannot resolve the
tolerance has not answered the question, and reporting it as a pass or a fail would be the same
error as the withdrawn negative-share column: reading a number that the noise produced.

Choosing ``abs_tolerance``
--------------------------
It must be set from the *use case*, not from the data, or it becomes a fitted parameter.  Two
defensible anchors are provided and the caller must pick one explicitly:

``tolerance_from_target_fraction``
    A tolerance of "one node in a thousand of the target set", i.e. ``|D| / 1000``.  This scales with
    the problem and is stated in the same units as the objective.
``tolerance_from_reference_se``
    A tolerance of one paired standard error of the reference measurement.  This is the *smallest*
    difference the experiment can resolve, so a gate set here is asking "is the method
    indistinguishable from the reference?", which is a different and weaker question than "is it
    within 1%".  It is offered because it is honest about what is measurable, and it is labelled so
    that a reader can see which question was asked.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: The cascade half of the plan's criterion, unchanged: it is measurable and was never the problem.
DEFAULT_CASCADE_SAVING = 0.30

#: The relative quality half, retained ONLY for reporting alongside the absolute verdict.
REPORT_ONLY_RELATIVE_TOLERANCE = 0.01

PASS = "PASS"
FAIL = "FAIL"
UNDECIDED = "UNDECIDED"


def tolerance_from_target_fraction(target_size: int, *, per_thousand: float = 1.0) -> float:
    """Absolute tolerance in target-count units, scaled to the target set.

    ``per_thousand=1.0`` means "one node per thousand target nodes", so on a target set of 95 the
    tolerance is ``0.095`` nodes.  Stated this way it is a property of the problem rather than of the
    measurement.
    """
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if per_thousand <= 0:
        raise ValueError(f"per_thousand must be positive, got {per_thousand}")
    return target_size * per_thousand / 1000.0


def tolerance_from_reference_se(paired_se: float, *, multiple: float = 1.0) -> float:
    """Absolute tolerance of ``multiple`` paired standard errors of the reference.

    This is the resolution floor of the experiment.  Setting the tolerance here asks a weaker
    question than a relative gate does, and :class:`GateVerdict` records which anchor was used so the
    two cannot be confused.
    """
    if not math.isfinite(paired_se):
        raise ValueError(f"paired_se must be finite, got {paired_se}")
    if paired_se < 0.0:
        raise ValueError(f"paired_se must be non-negative, got {paired_se}")
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return paired_se * multiple


@dataclass(frozen=True)
class GateVerdict:
    """One cell's verdict, with everything needed to re-derive it."""

    graph: str
    budget: int
    reference: str
    method: str
    # paired quantity: reference spread minus method spread, same windows
    paired_mean_gain: float
    paired_se: float
    ci_low: float
    ci_high: float
    reference_spread: float
    method_spread: float
    abs_tolerance: float
    tolerance_anchor: str
    cascade_saving: float
    required_saving: float
    quality_ok: bool
    cost_ok: bool
    verdict: str
    relative_loss: float
    n_pairs: int

    def as_dict(self) -> dict:
        return {
            "graph": self.graph,
            "budget": self.budget,
            "reference": self.reference,
            "method": self.method,
            "paired_mean_gain": self.paired_mean_gain,
            "paired_se": self.paired_se,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "reference_spread": self.reference_spread,
            "method_spread": self.method_spread,
            "abs_tolerance": self.abs_tolerance,
            "tolerance_anchor": self.tolerance_anchor,
            "cascade_saving": self.cascade_saving,
            "required_saving": self.required_saving,
            "quality_ok": self.quality_ok,
            "cost_ok": self.cost_ok,
            "verdict": self.verdict,
            "relative_loss": self.relative_loss,
            "n_pairs": self.n_pairs,
        }


def evaluate_gate(
    *,
    graph: str,
    budget: int,
    reference: str,
    method: str,
    paired_gains: list[float],
    reference_spread: float,
    method_spread: float,
    reference_cascades: float,
    method_cascades: float,
    abs_tolerance: float,
    tolerance_anchor: str,
    required_saving: float = DEFAULT_CASCADE_SAVING,
    z: float = 1.96,
) -> GateVerdict:
    """Decide one (reference, method) cell.

    ``paired_gains`` are per-trial differences ``reference - method`` on **shared threshold windows**,
    which is what makes the comparison sensitive enough to be worth making: window-draw variance
    cancels, so the paired standard error is far below the unpaired one.

    A positive mean means the reference is better.  The quality test asks whether we can *exclude* a
    loss larger than ``abs_tolerance``, i.e. whether the upper confidence bound on the loss is within
    tolerance.
    """
    if not paired_gains:
        raise ValueError("paired_gains must not be empty")
    n = len(paired_gains)
    mean = sum(paired_gains) / n
    if n > 1:
        var = sum((g - mean) ** 2 for g in paired_gains) / n
        se = math.sqrt(var / n)
    else:
        se = float("inf")

    half = z * se
    ci_low, ci_high = mean - half, mean + half

    if reference_spread <= 1e-12:
        relative_loss = float("nan")
    else:
        relative_loss = max(0.0, mean) / reference_spread

    if reference_cascades <= 0:
        saving = 0.0 if method_cascades > 0 else float("nan")
    else:
        saving = (reference_cascades - method_cascades) / reference_cascades

    cost_ok = method_cascades <= (1.0 - required_saving) * reference_cascades + 1e-12

    # The quality test needs the CI to be tighter than the tolerance: if the interval is wider than
    # what we are trying to resolve, the measurement has not answered the question.
    if not math.isfinite(half) or (ci_high - ci_low) > abs_tolerance:
        quality_ok = False
        verdict = UNDECIDED
    else:
        quality_ok = ci_high <= abs_tolerance
        verdict = PASS if (quality_ok and cost_ok) else FAIL

    return GateVerdict(
        graph=graph, budget=budget, reference=reference, method=method,
        paired_mean_gain=mean, paired_se=se, ci_low=ci_low, ci_high=ci_high,
        reference_spread=reference_spread, method_spread=method_spread,
        abs_tolerance=abs_tolerance, tolerance_anchor=tolerance_anchor,
        cascade_saving=saving, required_saving=required_saving,
        quality_ok=quality_ok, cost_ok=cost_ok, verdict=verdict,
        relative_loss=relative_loss, n_pairs=n,
    )


def explain(verdict: GateVerdict) -> str:
    """A one-line human reading, so a table cannot be misread."""
    if verdict.verdict == UNDECIDED:
        return (
            f"{verdict.method} vs {verdict.reference}: UNDECIDED — the paired 95% CI on the loss is "
            f"[{verdict.ci_low:+.3f}, {verdict.ci_high:+.3f}] target nodes, wider than the tolerance "
            f"{verdict.abs_tolerance:.3f}. The measurement cannot resolve the question."
        )
    if verdict.verdict == PASS:
        return (
            f"{verdict.method} vs {verdict.reference}: PASS — loss <= {verdict.abs_tolerance:.3f} "
            f"nodes (CI high {verdict.ci_high:+.3f}) with {verdict.cascade_saving*100:.0f}% fewer "
            f"online cascades (needed {verdict.required_saving*100:.0f}%)."
        )
    if not verdict.quality_ok and verdict.cost_ok:
        return (
            f"{verdict.method} vs {verdict.reference}: FAIL on quality — CI high "
            f"{verdict.ci_high:+.3f} exceeds the tolerance {verdict.abs_tolerance:.3f} "
            f"(relative loss {verdict.relative_loss*100:.1f}%, reported for context only)."
        )
    if verdict.quality_ok and not verdict.cost_ok:
        return (
            f"{verdict.method} vs {verdict.reference}: FAIL on cost — saved "
            f"{verdict.cascade_saving*100:.0f}% of cascades, needed "
            f"{verdict.required_saving*100:.0f}%."
        )
    return (
        f"{verdict.method} vs {verdict.reference}: FAIL on both — quality CI high "
        f"{verdict.ci_high:+.3f} > {verdict.abs_tolerance:.3f} and cascade saving "
        f"{verdict.cascade_saving*100:.0f}% < {verdict.required_saving*100:.0f}%."
    )
