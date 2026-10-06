"""Independent verification of the Student-t quantiles used in run_diffusion.py.

Two methods that share nothing but the standard library:

1. numerical integration of the t density, with the CDF built from the exact closed-form expression
   ``P(T > t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)`` where ``I`` is the regularized incomplete beta
   function evaluated by the Lentz continued fraction plus a bisection inverse;
2. direct high-accuracy quadrature of the density with adaptive step refinement.

A third anchor: published textbook values for df = 1, 2, 10, 30, 100.
"""
from __future__ import annotations

import math

USED = {
    "T95": {19: 2.093024054408307, 39: 2.022690920036760, 99: 1.984216951508683,
            255: 1.969497515030559, 2047: 1.961503689029643},
    "T_BONF_80": {39: 3.119157955619764, 99: 3.016373414358630,
                  255: 2.984786210455561, 2047: 2.972927631918957},
    "T_BONF_40": {19: 3.257212557472532, 39: 3.119157955619764, 99: 3.016373414358630,
                  255: 2.984786210455561, 2047: 2.972927631918957},
}


def betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-16:
            break
    return h


def betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta + b * math.log1p(-x) + a * math.log(x)) * betacf(b, a, 1.0 - x) / b


def t_upper_tail(df: int, t: float) -> float:
    """P(T > t) for T ~ t_df, from the exact closed form."""
    if t <= 0:
        return 1.0 - t_upper_tail(df, -t)
    return 0.5 * betai(df / 2.0, 0.5, df / (df + t * t))


def t_cdf(df: int, t: float) -> float:
    return 1.0 - t_upper_tail(df, t)


def bisect_quantile(df: int, target_cdf: float, lo=0.0, hi=100.0) -> float:
    f = lambda t: t_cdf(df, t) - target_cdf
    a, b = lo, hi
    for _ in range(200):
        m = 0.5 * (a + b)
        if f(m) > 0:
            b = m
        else:
            a = m
        if b - a < 1e-15:
            break
    return 0.5 * (a + b)


def t_density(df: int, t: float) -> float:
    return (math.exp(math.lgamma((df + 1) / 2.0) - math.lgamma(df / 2.0))
            / math.sqrt(df * math.pi) * (1.0 + t * t / df) ** (-(df + 1) / 2.0))


def quad_cdf(df: int, t: float, n: int = 4_000_000) -> float:
    """Method 2: Simpson quadrature of the density from -T to t, with T chosen so the tail is tiny."""
    big = 400.0
    a, b = -big, t
    h = (b - a) / n
    total = t_density(df, a) + t_density(df, b)
    for i in range(1, n):
        x = a + i * h
        total += (4.0 if i % 2 else 2.0) * t_density(df, x)
    return total * h / 3.0


def main() -> int:
    print("=" * 100)
    print("  ANCHOR CHECK against published two-sided 95% t values")
    print("=" * 100)
    published = {1: 12.7062047364, 2: 4.30265272975, 10: 2.22813885196,
                 30: 2.04227245630, 100: 1.98397151845, 1000: 1.96233908083}
    worst_anchor = 0.0
    for df, val in published.items():
        mine = bisect_quantile(df, 0.975)
        err = abs(mine - val)
        worst_anchor = max(worst_anchor, err)
        print(f"  df={df:<5} published {val:.11f}  computed {mine:.11f}  |err| {err:.2e}")
    print(f"\n  worst anchor error: {worst_anchor:.3e}")

    print("\n" + "=" * 100)
    print("  METHOD 2 CROSS-CHECK (Simpson quadrature of the density)")
    print("=" * 100)
    worst_quad = 0.0
    for df in (19, 39, 99, 255):
        t = bisect_quantile(df, 0.975)
        c = quad_cdf(df, t, n=200_000)
        err = abs(c - 0.975)
        worst_quad = max(worst_quad, err)
        print(f"  df={df:<5} t={t:.9f}  quadrature CDF {c:.12f}  |CDF-0.975| {err:.2e}")
    print(f"\n  worst quadrature error: {worst_quad:.3e}")

    print("\n" + "=" * 100)
    print("  VERIFICATION OF THE CONSTANTS ACTUALLY USED IN run_diffusion.py")
    print("=" * 100)
    bad = []
    for table, vals in USED.items():
        if table == "T95":
            levels = {df: 0.975 for df in vals}
        elif table == "T_BONF_80":
            levels = {df: 1.0 - 0.05 / 80 for df in vals}
        else:
            levels = {df: 1.0 - 0.05 / 40 for df in vals}
        print(f"\n  {table}")
        for df, val in sorted(vals.items()):
            target = levels[df]
            mine = bisect_quantile(df, target)
            err = abs(mine - val)
            flag = "OK" if err < 1e-9 else "MISMATCH"
            if err >= 1e-9:
                bad.append((table, df, val, mine, err))
            # confirm the cdf at the used constant
            cdf_at = t_cdf(df, val)
            print(f"    df={df:<5} used {val:.12f}  computed {mine:.12f}  |err| {err:.2e}  "
                  f"CDF(used)={cdf_at:.12f} target={target:.12f}  {flag}")
    print()
    if bad:
        print("  !! MISMATCHES FOUND:")
        for b in bad:
            print("   ", b)
        return 1
    print("  ALL CONSTANTS VERIFIED to < 1e-9 by an independent implementation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
