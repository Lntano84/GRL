"""Tests for the Go/No-Go gate.

The gate exists because the proposed criterion --- "quality loss <= 1% and >= 30% fewer online
cascades" --- cannot be evaluated in this regime.  At ``|S|/n >= 20%`` on Congress-Twitter a 1%
quality loss is ``0.007`` target nodes while the paired standard error of a marginal is
``0.05``--``0.39``, so the threshold sits one to two orders of magnitude below the noise.

These tests pin the three things that make the replacement honest:

1. a measurement that cannot resolve the tolerance returns ``UNDECIDED``, never a pass or a fail;
2. the tolerance is an **absolute** count of target nodes, not a percentage;
3. which anchor produced the tolerance is recorded on the verdict, so a weak question cannot be
   read as a strong one.
"""

from __future__ import annotations

import math

import pytest

from grl.evaluation.gate import (
    FAIL,
    PASS,
    UNDECIDED,
    evaluate_gate,
    explain,
    tolerance_from_reference_se,
    tolerance_from_target_fraction,
)


def gate(**overrides):
    """A cell with a resolvable difference, then overridden per test."""
    params = dict(
        graph="g", budget=10, reference="ref", method="m",
        paired_gains=[0.4, 0.5, 0.6, 0.45, 0.55],   # mean 0.5, se ~0.035
        reference_spread=100.0, method_spread=99.5,
        reference_cascades=1000, method_cascades=100,
        abs_tolerance=1.0, tolerance_anchor="reference_se",
    )
    params.update(overrides)
    return evaluate_gate(**params)


# ---------------------------------------------------------------------------------------------
# the three verdicts
# ---------------------------------------------------------------------------------------------
def test_pass_when_the_ci_fits_inside_the_tolerance_and_cost_is_saved():
    verdict = gate()
    assert verdict.verdict == PASS
    assert verdict.quality_ok and verdict.cost_ok
    assert verdict.ci_high <= verdict.abs_tolerance


def test_fail_on_quality_when_the_ci_upper_bound_exceeds_the_tolerance():
    verdict = gate(paired_gains=[5.0, 5.2, 4.8, 5.1, 4.9])
    assert verdict.verdict == FAIL
    assert not verdict.quality_ok


def test_fail_on_cost_when_the_cascade_saving_is_too_small():
    verdict = gate(method_cascades=900)   # 10% saved, 30% required
    assert verdict.verdict == FAIL
    assert verdict.quality_ok and not verdict.cost_ok


def test_undecided_when_the_ci_is_wider_than_the_tolerance():
    """The core claim: an uninformative measurement must not be reported as a verdict."""
    verdict = gate(abs_tolerance=0.05)
    assert verdict.verdict == UNDECIDED
    assert not verdict.quality_ok
    assert (verdict.ci_high - verdict.ci_low) > verdict.abs_tolerance


def test_undecided_takes_precedence_over_a_cost_success():
    """Cheap but unmeasurable is still unknown, not a pass."""
    verdict = gate(abs_tolerance=0.05, method_cascades=1)
    assert verdict.cost_ok is True
    assert verdict.verdict == UNDECIDED


def test_a_single_trial_is_undecided_because_the_se_is_infinite():
    verdict = gate(paired_gains=[0.5])
    assert not math.isfinite(verdict.paired_se)
    assert verdict.verdict == UNDECIDED


# ---------------------------------------------------------------------------------------------
# the relative form is report-only
# ---------------------------------------------------------------------------------------------
def test_the_relative_loss_is_recorded_but_does_not_decide():
    lenient = gate(abs_tolerance=1.0)
    strict = gate(abs_tolerance=0.05)
    assert lenient.relative_loss == pytest.approx(strict.relative_loss)
    assert lenient.verdict == PASS
    assert strict.verdict == UNDECIDED, (
        "the same relative loss must not decide the verdict; only the absolute tolerance may"
    )


def test_a_relative_loss_below_one_percent_can_still_be_undecided():
    """Exactly the trap the plan's '<= 1%' wording walks into."""
    verdict = gate(paired_gains=[0.5, 0.7, 0.3, 0.6, 0.4], abs_tolerance=0.05)
    assert verdict.relative_loss < 0.01
    assert verdict.verdict == UNDECIDED


# ---------------------------------------------------------------------------------------------
# the anchors
# ---------------------------------------------------------------------------------------------
def test_target_fraction_anchor_scales_with_the_target_set():
    assert tolerance_from_target_fraction(1000, per_thousand=1.0) == pytest.approx(1.0)
    assert tolerance_from_target_fraction(95, per_thousand=1.0) == pytest.approx(0.095)
    assert tolerance_from_target_fraction(95, per_thousand=10.0) == pytest.approx(0.95)


def test_target_fraction_anchor_rejects_degenerate_inputs():
    with pytest.raises(ValueError, match="target_size must be positive"):
        tolerance_from_target_fraction(0)
    with pytest.raises(ValueError, match="per_thousand must be positive"):
        tolerance_from_target_fraction(100, per_thousand=0.0)


def test_resolution_anchor_is_a_multiple_of_the_paired_se():
    assert tolerance_from_reference_se(0.25, multiple=1.0) == pytest.approx(0.25)
    assert tolerance_from_reference_se(0.25, multiple=2.0) == pytest.approx(0.50)


def test_resolution_anchor_rejects_non_finite_input():
    with pytest.raises(ValueError, match="must be finite"):
        tolerance_from_reference_se(float("inf"))


def test_the_anchor_is_recorded_on_the_verdict():
    """A weak question must not be readable as a strong one."""
    weak = gate(abs_tolerance=1.0, tolerance_anchor="reference_se")
    strong = gate(abs_tolerance=1.0, tolerance_anchor="target_fraction:1.0")
    assert weak.tolerance_anchor == "reference_se"
    assert strong.tolerance_anchor == "target_fraction:1.0"
    assert weak.as_dict()["tolerance_anchor"] != strong.as_dict()["tolerance_anchor"]


# ---------------------------------------------------------------------------------------------
# accounting
# ---------------------------------------------------------------------------------------------
def test_cascade_saving_is_measured_against_the_reference():
    verdict = gate(reference_cascades=1000, method_cascades=250)
    assert verdict.cascade_saving == pytest.approx(0.75)


def test_a_method_that_costs_more_than_the_reference_saves_nothing():
    verdict = gate(reference_cascades=1000, method_cascades=1500)
    assert verdict.cascade_saving < 0
    assert verdict.cost_ok is False


def test_the_pair_count_is_reported():
    assert gate(paired_gains=[0.4, 0.5, 0.6]).n_pairs == 3


def test_empty_paired_gains_are_refused():
    with pytest.raises(ValueError, match="paired_gains must not be empty"):
        gate(paired_gains=[])


# ---------------------------------------------------------------------------------------------
# the human reading
# ---------------------------------------------------------------------------------------------
def test_explain_names_the_outcome_and_the_number_that_decided_it():
    assert "PASS" in explain(gate())
    assert "UNDECIDED" in explain(gate(abs_tolerance=0.01))
    assert "FAIL" in explain(gate(paired_gains=[5.0, 5.1, 4.9, 5.05, 4.95]))
