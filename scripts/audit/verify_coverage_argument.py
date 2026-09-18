"""Verify the narrow argument that replaces the retracted "RR has no domain" claim.

What is being replaced
----------------------
The old claim was: "every reverse-reachability set is empty, therefore RR sampling has no domain
here."  That probe was invalid -- it required a predecessor's window to contain 0 and excluded the
root, and it counted whether the returned set was non-empty rather than checking ``S ∩ R``.  A
correct RR set for a root answers "which seeds can reach this root", which does not require a node
to self-activate under zero influence.  So the empty-set observation was an artefact of a
mis-specified probe, and the claim, the paper section and the paper title are withdrawn.

What replaces it
----------------
A strictly narrower and checkable statement, established by exact arithmetic on a one-hop instance
rather than by simulation:

  Any coverage function of the form  c * Pr[S ∩ R ≠ ∅]  with R independent of S is monotone
  non-decreasing and submodular in S.  If the target F_D is shown to be non-monotone on a concrete
  instance, then F_D cannot equal any such coverage function on that instance.

The verification below is exact: it computes the probability that a single target node ends
positive, under the model's own window distribution, for the empty set, each singleton and the
pair.  Two threshold distributions are reported separately, because the source paper's simplex
sampling and a uniform-threshold LT reading are different and must not be conflated:

  simplex   (kappa, tau) uniform on {0 <= kappa <= tau <= 1}, giving P(positive | delta) = 2d(1-d)
  uniform   kappa ~ U[0,1], tau = 1 -- a standard uniform-threshold LT reading

The argument only needs ONE distribution on which F_D is non-monotone, and the simplex one is the
model's own.  The uniform-threshold reading is reported because the audit requires the distinction.
"""

from __future__ import annotations

import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def simplex_probability(delta: float) -> float:
    """P(kappa <= delta <= tau) for (kappa, tau) uniform on the 2-D simplex.

    Derived, not assumed: with density 2 on {0 <= k <= t <= 1}, integrating over the region
    {k <= d} INTERSECT {t >= d} gives area of {0 <= k <= min(d, ...)} ... the closed form is
    2*d*(1-d), checked below against direct numerical integration.
    """
    if delta <= 0.0 or delta >= 1.0:
        return 0.0
    return 2.0 * delta * (1.0 - delta)


def simplex_probability_numeric(delta: float, steps: int = 4000) -> float:
    """Direct 2-D integration over the simplex, to validate the closed form."""
    total = 0.0
    # (k, t) uniform on the simplex with density 2; integrate the indicator {k <= delta <= t}
    for i in range(steps):
        k = i / steps
        # conditional on k, t is uniform on [k, 1]
        if k > delta:
            break
        length = 1.0 - k
        if length <= 0:
            continue
        # fraction of t in [max(k, delta), 1]
        lo = max(k, delta)
        total += 2.0 * (1.0 - k) * ((1.0 - lo) / length) * (1.0 / steps)
    return total


def uniform_lt_probability(delta: float) -> float:
    """P(activate) for a standard uniform-threshold LT node when the sum of active in-weights is
    delta and the threshold is kappa ~ U[0,1] with no upper threshold: P(kappa <= delta) = delta.
    """
    return min(max(delta, 0.0), 1.0)


def main() -> int:
    print("=" * 100)
    print("PART 1  validate the closed form P(positive | delta) = 2*delta*(1-delta)")
    print("=" * 100)
    worst = 0.0
    for d in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        closed = simplex_probability(d)
        numeric = simplex_probability_numeric(d)
        worst = max(worst, abs(closed - numeric))
        print(f"  delta={d:.1f}  closed={closed:.6f}  numeric={numeric:.6f}  diff={closed-numeric:+.6f}")
    print(f"  worst discrepancy {worst:.6f}")
    assert worst < 5e-3, "closed form does not match direct integration"
    print()

    print("=" * 100)
    print("PART 2  the one-hop instance:  a -> t weight 0.4,  b -> t weight 0.4,  D = {t}")
    print("=" * 100)
    w = 0.4
    cases = {
        "S = {}": 0.0,
        "S = {a}": w,
        "S = {b}": w,
        "S = {a,b}": w + w,
    }
    print()
    print("  simplex-sampled window (the model's own distribution)")
    print(f"  {'S':<12}{'delta_t':>10}{'F_D(S)':>12}")
    simplex_values = {}
    for label, delta in cases.items():
        value = simplex_probability(delta)
        simplex_values[label] = value
        print(f"  {label:<12}{delta:>10.2f}{value:>12.4f}")

    print()
    print("  uniform-threshold reading (kappa ~ U[0,1], tau = 1) -- reported for contrast only")
    print(f"  {'S':<12}{'delta_t':>10}{'F_D(S)':>12}")
    uniform_values = {}
    for label, delta in cases.items():
        value = uniform_lt_probability(delta)
        uniform_values[label] = value
        print(f"  {label:<12}{delta:>10.2f}{value:>12.4f}")

    print()
    print("=" * 100)
    print("PART 3  monotonicity and the coverage-representation argument")
    print("=" * 100)
    s_empty = simplex_values["S = {}"]
    s_a = simplex_values["S = {a}"]
    s_ab = simplex_values["S = {a,b}"]
    non_monotone = s_ab < s_a
    print(f"  simplex:    F({{}})={s_empty:.4f}  F({{a}})={s_a:.4f}  F({{a,b}})={s_ab:.4f}")
    print(f"              F({{a,b}}) < F({{a}}) ? {non_monotone}")
    print(f"  uniform LT: F({{a}})={uniform_values['S = {a}']:.4f}  "
          f"F({{a,b}})={uniform_values['S = {a,b}']:.4f}  (monotone: "
          f"{uniform_values['S = {a,b}'] >= uniform_values['S = {a}']})")
    print()
    if non_monotone:
        print("  Under the model's OWN window distribution the target is non-monotone on this")
        print("  instance.  Any coverage function c*Pr[S cap R != 0] with R independent of S is")
        print("  monotone non-decreasing in S, so F_D cannot equal such a function here.")
        print()
        print("  Scope of the claim, stated so it cannot be over-read:")
        print("    * it excludes the STANDARD, non-negative, seed-independent coverage form;")
        print("    * it does NOT exclude signed decompositions, or other sampling structures,")
        print("      or algorithms specialised to restricted instances;")
        print("    * it does NOT show Monte-Carlo is the only available estimator -- a plain MC")
        print("      sample mean is itself unbiased;")
        print("    * it does NOT imply learning must help.")
    else:
        print("  the instance is monotone under the simplex distribution too; the argument needs")
        print("  a different instance.")
    print()

    if len(sys.argv) > 1 and sys.argv[1] != "--":
        out = Path(sys.argv[1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "instance": {"edges": {"a->t": w, "b->t": w}, "target_set": ["t"],
                         "seeds_disjoint_from_targets": True},
            "simplex_values": simplex_values,
            "uniform_lt_values": uniform_values,
            "non_monotone_under_simplex": bool(non_monotone),
            "closed_form_check_worst_discrepancy": worst,
        }, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
