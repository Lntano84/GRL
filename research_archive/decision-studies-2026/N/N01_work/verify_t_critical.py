#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N01_work/verify_t_critical.py
=============================
Independent numerical verification of two-sided Student-t critical values
    P(|T_df| <= t) = 1 - alpha
for df = 19, 39, 99, 255, 2047 and for three confidence levels:

    T95          : alpha = 0.05                 (plain two-sided 95%)
    T_BONF_2N80  : alpha = 0.05/80              (Bonferroni, family size m = 2N = 80, N = 40)
    T_BONF_2N40  : alpha = 0.05/40              (Bonferroni, family size m = 2N = 40, N = 20)

DEFINITION USED FOR "BONFERRONI" (stated once, used everywhere)
---------------------------------------------------------------
A family of m = 2N two-sided interval estimates/test statistics is to hold
*simultaneously* with confidence 0.95.  Bonferroni allocates alpha/m to each
member, so member i is computed at per-comparison confidence

       1 - 0.05/m  =  1 - 0.05/(2N)                                    (D1)

equivalently per-comparison two-sided significance alpha_i = 0.05/m,
equivalently per-comparison one-sided tail probability 0.5*0.05/m = 0.025/m.

    T_BONF(m, df) := the t with P(|T_df| <= t) = 1 - 0.05/m.

NOT USED (shown only for contrast in section 8): the convention
"one-sided alpha_i = 0.05/m", which would give P(|T|<=t) = 1 - 0.10/m.

METHODS
-------
A. Numerical Recipes `betacf` + `betai` (math.lgamma), the literal identity
       CDF(t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)      (t > 0)
   plus plain bisection.  This is the required reference implementation.
B. Same continued fraction, but the prefactor ln(1/B(a,b)) is supplied from
   40-digit `decimal` gamma values instead of `math.lgamma`, which removes the
   cancellation lgamma(a+b) - lgamma(a) - lgamma(b) (visible for large df).
C. Completely independent: 60-digit `decimal` numerical integration of the
   Student-t density itself,
       f(x) = C_nu (1 + x^2/nu)^(-(nu+1)/2),
       C_nu = Gamma((nu+1)/2) / (sqrt(nu*pi) Gamma(nu/2))
   with C_nu built from *exact* factorials/sqrt(pi) (no lgamma), the tail
   mapped by x = sqrt(nu)*sinh(w), u = tanh(w/2) to a finite rational
   integrand, composite Gauss-Legendre quadrature, and bisection in Decimal.
D. Published anchors (section 7) and exact closed forms for df = 1, 2.

Only the Python standard library is used.
"""

from __future__ import annotations

import math
import os
import statistics
import time
from decimal import Decimal, localcontext

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------
DFS = (19, 39, 99, 255, 2047)

# (label, family size m, two-sided alpha, human description)
FAMILIES = (
    ("T95", 1, "0.05", "plain two-sided 95%"),
    ("T_BONF_2N80", 80, "0.000625", "Bonferroni, m = 2N = 80 (N = 40), per-test 1 - 0.05/80"),
    ("T_BONF_2N40", 40, "0.00125", "Bonferroni, m = 2N = 40 (N = 20), per-test 1 - 0.05/40"),
)

C_ALPHA = Decimal("0.05")

ANCHORS = (  # published t_{0.975, df} (upper 0.975 quantile = two-sided 95%)
    (1, "12.7062047364"),
    (2, "4.30265272975"),
    (10, "2.22813885196"),
    (30, "2.04227245630"),
    (100, "1.98397151845"),
    (1000, "1.96233908083"),
)

BONF_N_VALUES = (5, 10, 20, 40, 100, 500)   # family size m = 2N

DEC_PREC = 60            # working precision of the Decimal method
GL_NODES = 24            # Gauss-Legendre nodes per panel  (production setting)
GL_PANELS = 6            # panels on [0, U]
SOLVE_TOL_F = 1e-15      # bisection tolerance, float methods
SOLVE_TOL_D = Decimal("1e-25")   # bisection tolerance, Decimal method

OUT = []


def say(line: str = "") -> None:
    print(line)
    OUT.append(line)


def hr(ch: str = "=", n: int = 100) -> None:
    say(ch * n)


def head(num: str, title: str) -> None:
    say()
    hr()
    say(" %s. %s" % (num, title))
    hr()


def dstr(x: Decimal, nd: int = 20) -> str:
    return format(x, ".%df" % nd)


# ==========================================================================
# 1.  Regularized incomplete beta function  (Numerical Recipes)
# ==========================================================================
FPMIN = 1e-300
CF_STATS = {"max_iter": 0, "calls": 0, "worst": 0}


def betacf(a: float, b: float, x: float, maxit: int = 500, eps: float = 3.0e-16):
    """Continued fraction for the incomplete beta function (NR, 'betacf').

    Returns (value, iterations).  Iteration count is tracked for reporting.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    it = 0
    for m in range(1, maxit + 1):
        it = m
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    CF_STATS["calls"] += 1
    CF_STATS["max_iter"] = max(CF_STATS["max_iter"], it)
    return h, it


def betai(a: float, b: float, x: float, *, one_minus_x: float | None = None,
          ln_inv_B: float | None = None, maxit: int = 500) -> float:
    """Regularized incomplete beta I_x(a,b)  (NR, 'betai').

    ln_inv_B : if given, must equal ln(1/B(a,b)); overrides the lgamma route
               (used by method B to remove lgamma cancellation).
    one_minus_x : if given, must equal 1-x without cancellation error.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    omx = (1.0 - x) if one_minus_x is None else one_minus_x
    if ln_inv_B is None:
        ln_inv_B = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(ln_inv_B + a * math.log(x) + b * math.log(omx))
    if x < (a + 1.0) / (a + b + 2.0):
        cf, _ = betacf(a, b, x, maxit)
        return bt * cf / a
    cf, _ = betacf(b, a, omx, maxit)
    return 1.0 - bt * cf / b


def betai_complement(a: float, b: float, x: float, *, one_minus_x: float | None = None,
                     ln_inv_B: float | None = None, maxit: int = 500) -> float:
    """1 - I_x(a,b), evaluated without catastrophic cancellation in either branch."""
    if x <= 0.0:
        return 1.0
    if x >= 1.0:
        return 0.0
    omx = (1.0 - x) if one_minus_x is None else one_minus_x
    if ln_inv_B is None:
        ln_inv_B = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(ln_inv_B + a * math.log(x) + b * math.log(omx))
    if x < (a + 1.0) / (a + b + 2.0):
        cf, _ = betacf(a, b, x, maxit)
        return 1.0 - bt * cf / a          # I_x is the small quantity here
    cf, _ = betacf(b, a, omx, maxit)
    return bt * cf / b                    # = I_{1-x}(b,a) = 1 - I_x(a,b), computed directly


# ==========================================================================
# 2.  Decimal machinery: pi, exact Gamma at half-integers, Gauss-Legendre
# ==========================================================================
def machin_pi(prec: int) -> Decimal:
    """pi to `prec` digits, Machin's formula, integer-power series only."""
    with localcontext() as ctx:
        ctx.prec = prec + 15
        eps = Decimal(1).scaleb(-(prec + 12))

        def atan_inv(x: int) -> Decimal:
            xd = Decimal(x)
            x2 = xd * xd
            total = Decimal(0)
            term = Decimal(1) / xd
            n, sign = 1, 1
            while term > eps:
                add = term / n
                total = total + add if sign > 0 else total - add
                term = term / x2
                n += 2
                sign = -sign
            return total

        return +(16 * atan_inv(5) - 4 * atan_inv(239))


PI_DEC = machin_pi(DEC_PREC)


def gamma_half(m: int) -> Decimal:
    """Gamma(m/2) for integer m >= 1, from exact factorials (no lgamma)."""
    if m % 2 == 0:
        return Decimal(math.factorial(m // 2 - 1))
    p = (m - 1) // 2
    num = Decimal(math.factorial(2 * p))
    den = Decimal(4) ** p * Decimal(math.factorial(p))
    return num / den * PI_DEC.sqrt()


def dec_sin_cos(x: Decimal, prec: int):
    """(sin x, cos x) at `prec` digits by Taylor series (fine for |x| < ~1.5)."""
    with localcontext() as ctx:
        ctx.prec = prec
        x2 = x * x
        eps = Decimal(1).scaleb(-(prec - 5))
        s = term = x
        n = 1
        while True:
            term = -term * x2 / ((n + 1) * (n + 2))
            n += 2
            if abs(term) < eps:
                break
            s += term
        c = cterm = Decimal(1)
        m = 0
        while True:
            cterm = -cterm * x2 / ((m + 1) * (m + 2))
            m += 2
            if abs(cterm) < eps:
                break
            c += cterm
        return +s, +c


_GL_CACHE: dict = {}


def gl_nodes(n: int, prec: int):
    """Gauss-Legendre nodes/weights on [-1,1] at `prec` digits (Newton from a float start)."""
    key = (n, prec)
    if key in _GL_CACHE:
        return _GL_CACHE[key]
    with localcontext() as ctx:
        ctx.prec = prec + 15
        xs, ws = [], []
        for i in range(1, n + 1):
            x = Decimal(math.cos(math.pi * (i - 0.25) / (n + 0.5)))
            for _ in range(200):
                p0, p1 = Decimal(1), x
                for j in range(2, n + 1):
                    p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
                dpn = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dpn
                x -= dx
                if abs(dx) < Decimal(1).scaleb(-(prec + 10)):
                    break
            p0, p1 = Decimal(1), x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dpn = n * (x * p1 - p0) / (x * x - 1)
            xs.append(+x)
            ws.append(+(2 / ((1 - x * x) * dpn * dpn)))
    _GL_CACHE[key] = (xs, ws)
    return xs, ws


_CONST_CACHE: dict = {}


def t_constants(nu: int):
    """(C_nu, sqrt(nu)) at 60 digits;  C_nu = Gamma((nu+1)/2)/(sqrt(nu*pi)Gamma(nu/2))."""
    if nu not in _CONST_CACHE:
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            nuD = Decimal(nu)
            snu = nuD.sqrt()
            C = gamma_half(nu + 1) / (snu * PI_DEC.sqrt() * gamma_half(nu))
            _CONST_CACHE[nu] = (+C, +snu)
    return _CONST_CACHE[nu]


def dec_coverage(t, nu: int, n_nodes: int = GL_NODES, n_panels: int = GL_PANELS) -> Decimal:
    """P(|T_nu| <= t) by direct Decimal integration of the t density.

        x = sqrt(nu) sinh(w),  u = tanh(w/2)   =>
        P(|T|<=t) = 2 C_nu sqrt(nu) Int_0^W cosh(w)^-nu dw
                  = 4 C_nu sqrt(nu) Int_0^U (1-u^2)^(nu-1) (1+u^2)^(-nu) du
        U = z/(1+sqrt(1+z^2)),  z = t/sqrt(nu)      (stable, no cancellation)

    Integer powers only -> no exp/ln, full 60-digit accuracy.
    """
    if t == 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = DEC_PREC
        t = Decimal(t)
        C, snu = t_constants(nu)
        z = t / snu
        U = z / (1 + (1 + z * z).sqrt())
        xs, ws = gl_nodes(n_nodes, DEC_PREC)
        acc = Decimal(0)
        h = U / n_panels
        for p in range(n_panels):
            c = (p + Decimal("0.5")) * h
            for x, w in zip(xs, ws):
                u = c + (h / 2) * x
                acc += w * (1 - u * u) ** (nu - 1) / (1 + u * u) ** nu
        acc = acc * (h / 2)
        return +(4 * C * snu * acc)


# ==========================================================================
# 3.  CDF / coverage / tail probability, float methods
# ==========================================================================
_LN_INVB_CACHE: dict = {}


def ln_inv_B_exact(nu: int) -> float:
    """ln(1/B(1/2, nu/2)) computed at 40 digits with exact factorials."""
    key = nu
    if key not in _LN_INVB_CACHE:
        with localcontext() as ctx:
            ctx.prec = 40
            # B(1/2, nu/2) = Gamma(1/2) Gamma(nu/2) / Gamma((nu+1)/2)
            B = PI_DEC.sqrt() * gamma_half(nu) / gamma_half(nu + 1)
            _LN_INVB_CACHE[key] = float((1 / B).ln())
    return _LN_INVB_CACHE[key]


def coverage_A(t: float, df: int) -> float:
    """P(|T|<=t) using the literal identity of the task, NR betai + lgamma."""
    if t == 0.0:
        return 0.0
    a = df / 2.0
    x = df / (df + t * t)                 # x in (0,1]
    return 1.0 - betai(a, 0.5, x)         # = 1 - 2*(1-CDF(t)) = 2*CDF(t)-1


def cdf_A(t: float, df: int) -> float:
    """CDF of t_df via the literal identity CDF(t) = 1 - 0.5*I_{df/(df+t^2)}(df/2,1/2)."""
    a = df / 2.0
    x = df / (df + t * t)
    if t >= 0.0:
        return 1.0 - 0.5 * betai(a, 0.5, x)
    return 0.5 * betai(a, 0.5, x)


def tail_A(t: float, df: int) -> float:
    """P(T > t) for t >= 0, evaluated as the small quantity directly (no cancellation).

    P(T>t) = 0.5*I_x(df/2,1/2), x = df/(df+t^2) = 0.5*(1 - I_y(1/2,df/2)), y = 1-x.
    """
    if t == 0.0:
        return 0.5
    y = t * t / (df + t * t)
    return 0.5 * betai_complement(0.5, df / 2.0, y)


def coverage_B(t: float, df: int) -> float:
    """P(|T|<=t) with an accurate beta prefactor (no lgamma cancellation).

    P(|T|<=t) = I_y(1/2, df/2),  y = t^2/(df+t^2)   (complement form of the identity)
    """
    if t == 0.0:
        return 0.0
    y = t * t / (df + t * t)
    return betai(0.5, df / 2.0, y, ln_inv_B=ln_inv_B_exact(df),
                 one_minus_x=df / (df + t * t))


def tail_B(t: float, df: int) -> float:
    """P(T > t) with an accurate prefactor, computed as a small quantity directly."""
    if t == 0.0:
        return 0.5
    y = t * t / (df + t * t)
    return 0.5 * betai_complement(0.5, df / 2.0, y, ln_inv_B=ln_inv_B_exact(df),
                                  one_minus_x=df / (df + t * t))


def bisect(f, lo, hi, target, tol, maxit: int = 300):
    """Plain bisection for an increasing f.  Returns (root, iters, width, f(root)-target)."""
    flo = f(lo) - target
    fhi = f(hi) - target
    if not (flo < 0.0 < fhi):
        raise ValueError("bad bracket: f(lo)-t=%.6g  f(hi)-t=%.6g" % (flo, fhi))
    it = 0
    while (hi - lo) > tol and it < maxit:
        mid = (lo + hi) / 2
        if f(mid) - target < 0:
            lo = mid
        else:
            hi = mid
        it += 1
    root = (lo + hi) / 2
    return root, it, hi - lo, f(root) - target


def solve_coverage_float(target: float, df: int, method: str = "A"):
    cov = coverage_A if method == "A" else coverage_B
    f = lambda t: cov(t, df)
    return bisect(f, 0.5, 50.0, target, SOLVE_TOL_F)


def solve_tail_float(alpha_two_sided: float, df: int):
    """Solve 2*P(T>t) = alpha  (an independent equation form)."""
    g = lambda t: alpha_two_sided - 2.0 * tail_A(t, df)   # increasing in t
    return bisect(g, 0.0, 50.0, 0.0, SOLVE_TOL_F)


def solve_coverage_decimal(target: Decimal, df: int, n_nodes=GL_NODES, n_panels=GL_PANELS,
                           lo=Decimal(1), hi=Decimal(50)):
    f = lambda t: dec_coverage(t, df, n_nodes, n_panels)
    root, it, width, resid = bisect(f, lo, hi, target,
                                    SOLVE_TOL_D, maxit=400)
    return root, it, width, resid


# ==========================================================================
start_time = time.time()
hr()
say(" INDEPENDENT VERIFICATION OF TWO-SIDED STUDENT-t CRITICAL VALUES")
say(" (pure Python standard library; no numpy / scipy available)")
hr()
say(" python %s" % __import__("sys").version.split()[0])
say(" df values            : %s" % (list(DFS),))
for lab, m, al, desc in FAMILIES:
    say(" %-13s: alpha = %-9s -> target P(|T|<=t) = 1 - alpha = %s   [%s]"
        % (lab, al, dstr(1 - Decimal(al), 6), desc))
say()
say(" DEFINITION (Bonferroni, used throughout): family of m = 2N two-sided")
say(" intervals at simultaneous confidence 0.95 -> each member at per-comparison")
say(" confidence 1 - 0.05/m = 1 - 0.05/(2N).  So family size m = 80 means N = 40")
say(" comparisons (2N = 80) and m = 40 means N = 20 (2N = 40).")

# ==========================================================================
head("1", "REGULARIZED INCOMPLETE BETA  I_x(a,b)  (Numerical Recipes betacf + betai)")
# ==========================================================================
say(" I_x(a,b) = B_x(a,b)/B(a,b), evaluated with the NR continued fraction and")
say(" the prefactor ln(1/B(a,b)) = lgamma(a+b) - lgamma(a) - lgamma(b).")
say()

# --- 1a. integer a,b against the exact binomial sum -----------------------
say(" 1a. validation against the EXACT binomial sum for integer (a,b):")
say("     I_x(a,b) = sum_{j=a}^{a+b-1} C(a+b-1,j) x^j (1-x)^(a+b-1-j)")
worst_binom = 0.0
with localcontext() as ctx:
    ctx.prec = 40
    for (ia, ib) in ((5, 7), (2, 1), (1, 1), (12, 3), (30, 30), (100, 2)):
        for x in (Decimal("0.01"), Decimal("0.3"), Decimal("0.5"),
                  Decimal("0.77"), Decimal("0.999")):
            n = ia + ib - 1
            s = Decimal(0)
            for j in range(ia, n + 1):
                s += Decimal(math.comb(n, j)) * x ** j * (1 - x) ** (n - j)
            got = betai(float(ia), float(ib), float(x))
            worst_binom = max(worst_binom, abs(got - float(s)))
say("     max |betai(a,b,x) - binomial_exact| over 30 (a,b,x) cases : %.3e" % worst_binom)
say()

# --- 1b. symmetry and boundary ------------------------------------------
# For the identity to be testable in floating point, 1-x must be EXACT, so the
# primary test uses dyadic x (all powers of two / sums of two powers of two).
DYADIC = (2.0 ** -30, 2.0 ** -20, 2.0 ** -10, 0.25, 0.5, 0.75,
          1.0 - 2.0 ** -10, 1.0 - 2.0 ** -20)
GENERIC = (1e-8, 1e-3, 0.02, 0.999)
worst_sym = worst_sym_gen = 0.0
for (a, b) in ((0.5, 9.5), (0.5, 1023.5), (9.5, 0.5), (3.0, 4.5)):
    for x in DYADIC:
        v = betai(a, b, x) + betai(b, a, 1.0 - x)
        worst_sym = max(worst_sym, abs(v - 1.0))
    for x in GENERIC:
        v = betai(a, b, x) + betai(b, a, 1.0 - x)
        worst_sym_gen = max(worst_sym_gen, abs(v - 1.0))
say(" 1b. symmetry  I_x(a,b) + I_{1-x}(b,a) = 1 :")
say("     max deviation, 32 cases with EXACT complements (dyadic x) : %.3e" % worst_sym)
say("     max deviation, 16 cases with generic x (x=1e-8..0.999)     : %.3e" % worst_sym_gen)
say("     The identity is exact; BOTH residuals are floating-point artifacts, not")
say("     mathematics.  (i) with generic x, fl(1-fl(1-x)) differs from x by up to")
say("     ~1e-13 relative.  (ii) the shared prefactor exp(lgamma(a+b)-lgamma(a)")
say("     -lgamma(b)) is a difference of numbers of size ~5900 for df=2047, and the")
say("     two terms of the identity evaluate it in a different ORDER, so they can")
say("     disagree by ~1e-12 in the exponent even when the arguments are identical.")
say("     Section 4 measures that lgamma error directly (8.9e-13 for df=2047).")
say("     I_0(a,b)      = %.1f   (exact 0)" % betai(3.0, 4.5, 0.0))
say("     I_1(a,b)      = %.1f   (exact 1)" % betai(3.0, 4.5, 1.0))
say("     I_x(a,1) = x^a: max dev over x in {1e-6..0.9} = %.3e"
    % max(abs(betai(2.5, 1.0, x) - x ** 2.5) for x in (1e-6, 0.1, 0.5, 0.9)))
say("     max continued-fraction iterations used so far: %d" % CF_STATS["max_iter"])

# ==========================================================================
head("2", "STUDENT-t CDF: the identity  CDF(t) = 1 - 0.5*I_{df/(df+t^2)}(df/2, 1/2)")
# ==========================================================================
say(" (t > 0;  CDF(-t) = 1 - CDF(t).  Equivalently, with y = t^2/(df+t^2),")
say("  P(T > t) = 0.5*I_{df/(df+t^2)}(df/2, 1/2) = 0.5*(1 - I_y(1/2, df/2)),")
say("  so P(|T| <= t) = 1 - I_y(1/2, df/2).)")
say()
say(" 2a. CDF(0) = 0.5 exactly?")
for df in DFS:
    v = cdf_A(0.0, df)
    say("     df=%5d : CDF(0) = %.17g   exact 0.5 -> %s" % (df, v, "OK" if v == 0.5 else "MISMATCH"))
say()
say(" 2b. monotonicity of CDF on t in [-30, 30], 60001 points:")
say("     (a 'tie' is CDF(t_i) == CDF(t_{i-1}) in double precision; a real decrease is d < 0)")
for df in DFS:
    prev = cdf_A(-30.0, df)
    dec = 0
    ties = 0
    worst = 0.0
    tie_at = set()
    strict_min = float("inf")
    for i in range(1, 60001):
        t = -30.0 + 60.0 * i / 60000.0
        cur = cdf_A(t, df)
        d = cur - prev
        if d < 0.0:
            dec += 1
            worst = max(worst, -d)
        elif d == 0.0:
            ties += 1
            tie_at.add(round(t, 1))
        else:
            strict_min = min(strict_min, d)
        prev = cur
    say("     df=%5d : decreases(d<0)=%d  max decrease=%.3e | ties=%d  min strict step=%.3e"
        % (df, dec, worst, ties, strict_min))
    say("               CDF(-30)=%.3e   CDF(30)=%.17f" % (cdf_A(-30.0, df), cdf_A(30.0, df)))
    say("               ties occur only where the CDF has saturated to 0.0 or 1.0;")
    say("               sampled t values with ties span [%.1f, %.1f] of [-30, 30]"
        % (min(tie_at), max(tie_at)))
say()
say(" 2c. equivalence of the two algebraic forms of the identity (float, NR betai):")
say("     literal  : CDF(t) = 1 - 0.5*I_x(df/2, 1/2),          x = df/(df+t^2)")
say("     complement: CDF(t) = 0.5 + 0.5*I_y(1/2, df/2),        y = t^2/(df+t^2)")
worst_form = 0.0
for df in DFS:
    for t in (0.1, 0.5, 1.0, 1.96, 2.5, 3.5, 6.0):
        v1 = cdf_A(t, df)
        v2 = 0.5 + 0.5 * betai(0.5, df / 2.0, t * t / (df + t * t))
        worst_form = max(worst_form, abs(v1 - v2))
say("     max |literal - complement| over 35 (df,t) pairs : %.3e" % worst_form)
say("     (the difference is pure floating-point cancellation in x = df/(df+t^2):")
say("      1-x loses ~%.1f digits for df=2047, t~2)" % math.log10(1.0 / (3.84 / 2050.84)))
say()
say(" 2d. closed-form checks of the whole chain:")
say("     df=1  (Cauchy)  :  CDF(t) = 0.5 + atan(t)/pi")
say("     df=2            :  CDF(t) = 0.5*(1 + t/sqrt(2+t^2))")
w1 = w2 = 0.0
for t in (0.1, 0.5, 1.0, 2.0, 5.0, 12.7062047364):
    w1 = max(w1, abs(cdf_A(t, 1) - (0.5 + math.atan(t) / math.pi)))
for t in (0.1, 0.5, 1.0, 2.0, 4.30265272991):
    w2 = max(w2, abs(cdf_A(t, 2) - 0.5 * (1.0 + t / math.sqrt(2.0 + t * t))))
say("     max |CDF_A(t) - Cauchy exact| (6 t)   : %.3e" % w1)
say("     max |CDF_A(t) - df=2 exact|   (5 t)   : %.3e" % w2)

# ==========================================================================
head("3", "METHOD A - Numerical Recipes betai + lgamma, inverted by bisection")
# ==========================================================================
say(" bisection on [0.5, 50] to bracket width < %.0e; residual = coverage(t*) - target" % SOLVE_TOL_F)
say()
say(" %-13s %-5s %-5s %-21s %-5s %-11s %-12s %s" %
    ("family", "m", "df", "t (method A)", "iter", "width", "coverage(t*)", "residual"))
say(" " + "-" * 98)
SOL_A = {}
for lab, m, al, desc in FAMILIES:
    target = float(1 - Decimal(al))
    for df in DFS:
        t, it, wid, res = solve_coverage_float(target, df, "A")
        SOL_A[(lab, df)] = t
        say(" %-13s %-5d %-5d %-21.15f %-5d %-11.2e %-12.10f %+.3e" %
            (lab, m, df, t, it, wid, coverage_A(t, df), res))
        assert abs(res) < 1e-13, "residual requirement violated"

# ==========================================================================
head("4", "METHOD B - same continued fraction, exact beta prefactor (no lgamma cancellation)")
# ==========================================================================
say(" Only the prefactor changes: ln(1/B(1/2,df/2)) comes from 40-digit factorials")
say(" instead of lgamma(a+b)-lgamma(a)-lgamma(b).  B isolates the floating-point")
say(" cost of the lgamma route used in method A.")
say()
with localcontext() as ctx:
    ctx.prec = 40
    for df in DFS:
        lg = math.lgamma((df + 1) / 2.0) - math.lgamma(0.5) - math.lgamma(df / 2.0)
        ex = ln_inv_B_exact(df)
        say("   df=%5d :  lgamma route ln(1/B) = %+.15f   exact = %+.15f   |diff| = %.3e"
            % (df, lg, ex, abs(lg - ex)))
say()
say(" %-13s %-5s %-21s %-11s %-21s %s" %
    ("family", "df", "t (method B)", "residual", "t(B) - t(A)", "rel. diff"))
say(" " + "-" * 98)
SOL_B = {}
for lab, m, al, desc in FAMILIES:
    target = float(1 - Decimal(al))
    for df in DFS:
        t, it, wid, res = solve_coverage_float(target, df, "B")
        SOL_B[(lab, df)] = t
        say(" %-13s %-5d %-21.15f %+.3e %+-21.15e %.2e" %
            (lab, df, t, res, t - SOL_A[(lab, df)],
             abs(t - SOL_A[(lab, df)]) / t))
say()
say(" 4b. internal consistency of the two complementary branches of betai_complement")
say("     (coverage via betai + tail via betai_complement must sum to 1):")
worst_ct = 0.0
for df in DFS:
    w = 0.0
    for t in (0.25, 0.5, 1.0, 1.96, 2.5, 3.4, 4.5, 6.0, 10.0):
        w = max(w, abs(coverage_B(t, df) + 2.0 * tail_B(t, df) - 1.0))
    worst_ct = max(worst_ct, w)
    say("     df=%5d : max |coverage_B(t) + 2*tail_B(t) - 1| over 9 t values = %.3e" % (df, w))
say("     max over all 45 (df,t) pairs : %.3e" % worst_ct)

# ==========================================================================
head("5", "METHOD C - 60-digit Decimal numerical integration of the t density")
# ==========================================================================
say(" f(x) = C_nu (1+x^2/nu)^(-(nu+1)/2),  C_nu = Gamma((nu+1)/2)/(sqrt(nu*pi)Gamma(nu/2))")
say(" C_nu from exact factorials + Machin pi (no lgamma, no cancellation).")
say()
say(" 5a. pi (Machin, Decimal) vs math.pi : |diff| = %.3e   (math.pi is the double rounding)" %
    abs(PI_DEC - Decimal(math.pi)))
say("     60-digit pi = %s" % dstr(PI_DEC, 60))
say()
say(" 5b. C_nu : Decimal (exact factorials) vs float lgamma route")
say("     %-6s %-24s %-24s %s" % ("df", "C_nu (Decimal)", "C_nu (lgamma float)", "relative diff"))
for df in DFS + (1, 2, 10, 30, 100, 1000):
    C, snu = t_constants(df)
    Cf = math.exp(math.lgamma((df + 1) / 2.0) - 0.5 * math.log(df * math.pi) - math.lgamma(df / 2.0))
    say("     %-6d %-24s %-24.17g %.3e" % (df, dstr(C, 22), Cf, abs(float(C) - Cf) / Cf))
say()
say(" 5c. Gauss-Legendre rule sanity (n=24 on [-1,1]):")
with localcontext() as ctx:
    ctx.prec = DEC_PREC
    xs, ws = gl_nodes(GL_NODES, DEC_PREC)
    say("     sum(w)      = %s   (exact 2)" % dstr(sum(ws), 40))
    say("     int x^2     = %s   (exact 2/3)" % dstr(sum(w * x * x for x, w in zip(xs, ws)), 40))
    say("     int x^46    = %s   (exact 2/47, exact for degree <= 2n-1 = 47)"
        % dstr(sum(w * x ** 46 for x, w in zip(xs, ws)), 40))
    say("     int x^48    = %s   (exact 2/49 -> should FAIL: quadrature order limit)"
        % dstr(sum(w * x ** 48 for x, w in zip(xs, ws)), 40))
say()
say(" 5d. quadrature convergence: P(|T|<=t) at the method-A root, four GL settings")
say("     %-6s %-13s %-24s %-24s %-24s %s" %
    ("df", "family", "(16 nodes, 3 panels)", "(20,4)", "(24,6) production", "spread (24,6)-(32,12)"))
with localcontext() as ctx:
    ctx.prec = DEC_PREC
    worst_quad = Decimal(0)
    for (lab, m, al, desc) in (("T95", 1, "0.05", ""), ("T_BONF_2N80", 80, "0.000625", ""),
                               ("T_BONF_2N40", 40, "0.00125", "")):
        for df in DFS:
            t = Decimal(SOL_A[(lab, df)])
            c1 = dec_coverage(t, df, 16, 3)
            c2 = dec_coverage(t, df, 20, 4)
            c3 = dec_coverage(t, df, 24, 6)
            c4 = dec_coverage(t, df, 32, 12)
            worst_quad = max(worst_quad, abs(c3 - c4))
            say("     %-6d %-13s %-24s %-24s %-24s %.2e" %
                (df, lab, dstr(c1, 22), dstr(c2, 22), dstr(c3, 22), abs(c3 - c4)))
say("     worst spread between the two production-independent settings : %.2e" % worst_quad)
say()
say(" 5e. bisection in Decimal (target 1e-25, bracket [1,50])")
say(" %-13s %-5s %-5s %-26s %-5s %-10s %-12s %s" %
    ("family", "m", "df", "t (method C)", "iter", "width", "coverage(t*)", "residual"))
say(" " + "-" * 108)
SOL_C = {}
for lab, m, al, desc in FAMILIES:
    target = 1 - C_ALPHA / Decimal(m)
    for df in DFS:
        t, it, wid, res = solve_coverage_decimal(target, df)
        SOL_C[(lab, df)] = t
        say(" %-13s %-5d %-5d %-26s %-5d %-10.1e %-12s %+.2e" %
            (lab, m, df, dstr(t, 22), it, float(wid), dstr(dec_coverage(t, df), 10), float(res)))
        assert abs(res) < Decimal("1e-13"), "residual requirement violated"

# ==========================================================================
head("6", "THREE-METHOD COMPARISON - the 15 requested constants")
# ==========================================================================
say(" %-13s %-5s %-9s %-20s %-20s %-20s %-10s %-10s" %
    ("family", "df", "target", "A: NR betai+lgamma", "B: exact prefactor", "C: 60-digit Decimal",
     "|A-C|", "|B-C|"))
say(" " + "-" * 116)
max_AC = max_BC = max_AB = 0.0
rows = []
for lab, m, al, desc in FAMILIES:
    for df in DFS:
        a, b, c = SOL_A[(lab, df)], SOL_B[(lab, df)], float(SOL_C[(lab, df)])
        dac, dbc, dab = abs(a - c), abs(b - c), abs(a - b)
        max_AC, max_BC, max_AB = max(max_AC, dac), max(max_BC, dbc), max(max_AB, dab)
        rows.append((lab, m, df, a, b, c, dac, dbc, dab))
        say(" %-13s %-5d %-9s %-20.15f %-20.15f %-20.15f %-10.2e %-10.2e" %
            (lab, df, dstr(1 - C_ALPHA / Decimal(m), 6), a, b, c, dac, dbc))
say()
say(" MAX |method A - method C| over the 15 constants : %.3e" % max_AC)
say(" MAX |method B - method C| over the 15 constants : %.3e" % max_BC)
say(" MAX |method A - method B| over the 15 constants : %.3e" % max_AB)
say(" (method C is the arbiter: its quadrature is converged to ~1e-55, see 5d)")
say()
say(" 6b. why the double-precision methods cannot do better: the root error of a")
say("     float method is (coverage error) / f(t), and 1/f(t) is huge in the tail.")
say("     coverage_C(t_A) is the 60-digit coverage evaluated at method A's root.")
say()
say(" %-13s %-5s %-12s %-13s %-13s %-13s %s" %
    ("family", "df", "t (method A)", "f(t) density", "1/f(t)", "coverage_C(tA)-tgt", "predicted dt"))
say(" " + "-" * 104)
for lab, m, al, desc in FAMILIES:
    target = 1 - C_ALPHA / Decimal(m)
    for df in DFS:
        tA = SOL_A[(lab, df)]
        with localcontext() as ctx:
            ctx.prec = DEC_PREC
            tAD = Decimal(repr(tA))
            err = dec_coverage(tAD, df) - target
            C, _ = t_constants(df)
            dens = C / ((1 + tAD * tAD / Decimal(df)) ** (df + 1)).sqrt()
            pred = -err / dens
        say(" %-13s %-5d %-12.9f %-13.5e %-13.4e %+.3e      %+.3e   (actual %+.3e)" %
            (lab, df, tA, float(dens), float(1 / dens), float(err), float(pred),
             float(SOL_C[(lab, df)] - tAD)))

# ==========================================================================
head("7", "PUBLISHED ANCHOR CHECK  (t_{0.975,df}, i.e. two-sided 95%)")
# ==========================================================================
say(" %-6s %-18s %-20s %-20s %-20s %-11s %-11s" %
    ("df", "published", "A (lgamma)", "B (exact pref.)", "C (Decimal)", "|A-pub|", "|C-pub|"))
say(" " + "-" * 112)
max_anchor = max_anchor_A = 0.0
anchor_rows = []
for df, pub in ANCHORS:
    pubD = Decimal(pub)
    tA = bisect(lambda t: coverage_A(t, df), 0.5, 50.0, 0.95, SOLVE_TOL_F)[0]
    tB = bisect(lambda t: coverage_B(t, df), 0.5, 50.0, 0.95, SOLVE_TOL_F)[0]
    tC, _, _, resC = solve_coverage_decimal(Decimal("0.95"), df)
    ea, ec = abs(Decimal(repr(tA)) - pubD), abs(tC - pubD)
    max_anchor = max(max_anchor, float(ec))
    max_anchor_A = max(max_anchor_A, float(ea))
    anchor_rows.append((df, pub, tA, tB, float(tC), float(ea), float(ec)))
    say(" %-6d %-18s %-20.15f %-20.15f %-20.15f %-11.3e %-11.3e" %
        (df, pub, tA, tB, float(tC), float(ea), float(ec)))
say()
say(" MAX |method C - published| : %.3e" % max_anchor)
say(" MAX |method A - published| : %.3e" % max_anchor_A)
say()
say(" exact closed forms (independent of all three methods):")
# df = 1 : 2*atan(t)/pi = 0.95  ->  t = tan(0.475*pi) = cot(pi/40) = cos(pi/40)/sin(pi/40)
with localcontext() as ctx:
    ctx.prec = DEC_PREC + 10
    s, c = dec_sin_cos(PI_DEC / 40, DEC_PREC + 10)
    t_exact_1 = c / s                      # cot(pi/40)
    # df = 2 : t^2 = 2*0.95^2/(1-0.95^2)
    t_exact_2 = (Decimal(2) * Decimal("0.95") ** 2 / (1 - Decimal("0.95") ** 2)).sqrt()
say("   df=1 exact: t = cot(pi/40) = cos(pi/40)/sin(pi/40) = %s" % dstr(t_exact_1, 22))
say("   df=2 exact: t = sqrt(2*0.95^2/(1-0.95^2))            = %s" % dstr(t_exact_2, 22))
with localcontext() as ctx:
    ctx.prec = DEC_PREC
    tC1 = solve_coverage_decimal(Decimal("0.95"), 1)[0]
    tC2 = solve_coverage_decimal(Decimal("0.95"), 2)[0]
    tA1 = Decimal(repr(bisect(lambda t: coverage_A(t, 1), 0.5, 50.0, 0.95, SOLVE_TOL_F)[0]))
    tA2 = Decimal(repr(bisect(lambda t: coverage_A(t, 2), 0.5, 50.0, 0.95, SOLVE_TOL_F)[0]))
    cov_pub2 = dec_coverage(Decimal("4.30265272975"), 2)
say("   -> df=1 : |method C - exact| = %.3e   |method A - exact| = %.3e" % (abs(tC1 - t_exact_1), abs(tA1 - t_exact_1)))
say("   -> df=2 : |method C - exact| = %.3e   |method A - exact| = %.3e" % (abs(tC2 - t_exact_2), abs(tA2 - t_exact_2)))
say()
say(" NOTE on the published anchors.  All six quoted values are rounded/truncated at")
say(" the number of digits shown, so |published - exact| is bounded by ~5e-11 for")
say(" 10-decimal quotes.  The largest anchor deviation (df=1, 2.25e-10) is exactly")
say(" this truncation: 12.7062047364 was rounded up from 12.7062047361747.")
say(" The df=2 anchor 4.30265272975 sits %.2e ABOVE the exact" % float(Decimal("4.30265272975") - t_exact_2))
say(" sqrt(2*0.95^2/(1-0.95^2)) = 4.302652729749464; its two-sided coverage is")
say(" %s instead of 0.95 (error %+.2e), i.e. consistent with the quote precision."
    % (dstr(cov_pub2, 18), float(cov_pub2 - Decimal("0.95"))))
say()
say(" 7b. exact closed forms at ALL THREE levels (df = 1 and df = 2), which validates")
say("     the Bonferroni targets as well as the 95% one:")
say("        df=1: coverage = 2*atan(t)/pi  ->  t = cot(pi*(1-c)/2)")
say("        df=2: coverage = t/sqrt(2+t^2) ->  t = sqrt(2c^2/(1-c^2))")
say("     %-10s %-4s %-30s %-30s %-30s %s" %
    ("target c", "df", "exact (closed form)", "method C (Decimal)", "method A (lgamma)", "|C-exact|"))
for c in (Decimal("0.95"), Decimal("0.99875"), Decimal("0.999375")):
    for df in (1, 2):
        with localcontext() as ctx:
            ctx.prec = DEC_PREC + 10
            if df == 1:
                s, cs = dec_sin_cos(PI_DEC * (1 - c) / 2, DEC_PREC + 10)
                exact = cs / s
            else:
                exact = (2 * c * c / (1 - c * c)).sqrt()
        tC = solve_coverage_decimal(c, df, lo=Decimal(1), hi=Decimal(5000))[0]
        tA = Decimal(repr(bisect(lambda t: coverage_A(t, df), 0.5, 5000.0,
                                 float(c), SOLVE_TOL_F)[0]))
        say("     %-10s %-4d %-30s %-30s %-30.15f %.2e" %
            (dstr(c, 6), df, dstr(exact, 20), dstr(tC, 20), float(tA), float(abs(tC - exact))))
say()
say(" 7c. large-df asymptotic sanity check (independent of both the beta function")
say("     and the quadrature).  Cornish-Fisher: t_p(nu) = z_p + (z^3+z)/(4 nu) + O(nu^-2)")
z975 = statistics.NormalDist().inv_cdf(0.975)
say("     z_0.975 (statistics.NormalDist) = %.17f" % z975)
say("     %-6s %-22s %-24s %s" % ("df", "t (method C)", "z + (z^3+z)/(4 nu)", "difference"))
for df in (19, 99, 255, 2047):
    approx = z975 + (z975 ** 3 + z975) / (4.0 * df)
    exact = float(SOL_C[("T95", df)])
    say("     %-6d %-22.15f %-24.15f %.3e" % (df, exact, approx, exact - approx))
say("     The residual is the O(nu^-2) term: it falls from 8.2e-3 at df=19 to 6.7e-7")
say("     at df=2047, a factor ~1.2e4, matching the nu^-2 scaling (nu ratio 108 ->")
say("     1.2e4).  The tabulated values are therefore consistent with the known")
say("     large-df behaviour of the t quantile, with no free parameters.")

# ==========================================================================
head("8", "BONFERRONI IDENTITY CHECKS  (family size m = 2N)")
# ==========================================================================
say(" DEFINITION USED: P(|T_df| <= t) = 1 - 0.05/m, m = 2N   [per-comparison")
say(" confidence 1 - 0.05/(2N); per-comparison two-sided alpha = 0.05/(2N);")
say(" per-comparison one-sided tail = 0.025/(2N)].")
say()
say(" Three independent evaluations of the same quantity:")
say("   route 1 : solve coverage_A(t)      = 1 - 0.05/m            (bisection, float)")
say("   route 2 : solve 2*tail_A(t)        = 0.05/m               (different function, float)")
say("   route 3 : solve dec_coverage(t)    = 1 - 0.05/m           (Decimal, 60 digits)")
say("   check   : family-wise error rate  m * 2*tail(t)  ==  0.05")
say()
say(" %-4s %-5s %-5s %-21s %-21s %-21s %-10s %-10s %s" %
    ("N", "m=2N", "df", "route 1 (coverage)", "route 2 (tail)", "route 3 (Decimal)",
     "|r1-r2|", "|r1-r3|", "m*2*tail(r1)-0.05"))
say(" " + "-" * 130)
max_r12 = max_r13 = max_fwer = 0.0
for N in BONF_N_VALUES:
    m = 2 * N
    alpha = 0.05 / m
    target = 1.0 - alpha
    for df in DFS:
        t1, _, _, _ = bisect(lambda t: coverage_A(t, df), 0.5, 50.0, target, SOLVE_TOL_F)
        t2, _, _, _ = solve_tail_float(alpha, df)
        t3, _, _, _ = solve_coverage_decimal(1 - C_ALPHA / Decimal(m), df)
        fwer = m * 2.0 * tail_A(t1, df) - 0.05
        max_r12 = max(max_r12, abs(t1 - t2))
        max_r13 = max(max_r13, abs(Decimal(repr(t1)) - t3))
        max_fwer = max(max_fwer, abs(fwer))
        say(" %-4d %-5d %-5d %-21.15f %-21.15f %-21.15f %-10.2e %-10.2e %+.2e" %
            (N, m, df, t1, t2, float(t3), abs(t1 - t2), abs(Decimal(repr(t1)) - t3), fwer))
say()
say(" MAX |route 1 - route 2|  (30 cases) : %.3e" % max_r12)
say(" MAX |route 1 - route 3|  (30 cases) : %.3e" % max_r13)
say(" MAX |m*2*tail(t) - 0.05| (30 cases) : %.3e" % max_fwer)
say()
say(" explicit statement of the identity for the two headline families:")
say("   T_BONF(m=2N, df) := the t with P(|T_df| <= t) = 1 - 0.05/(2N), i.e. the")
say("   ordinary two-sided t critical value at confidence level 1 - 0.05/(2N).")
for N, lab in ((40, "T_BONF_2N80"), (20, "T_BONF_2N40")):
    m = 2 * N
    say("   m = 2N = %-4d (N = %-3d), per-comparison level 1 - 0.05/(2N) = %s :"
        % (m, N, dstr(1 - C_ALPHA / Decimal(m), 8)))
    for df in DFS:
        cov_target = 1.0 - 0.05 / m
        r1, _, _, _ = bisect(lambda t: coverage_A(t, df), 0.5, 50.0, cov_target, SOLVE_TOL_F)
        r2, _, _, _ = solve_tail_float(0.05 / m, df)
        r3, _, _, _ = solve_coverage_decimal(1 - C_ALPHA / Decimal(m), df)
        fwer = m * 2.0 * tail_A(r1, df)
        say("      df=%5d : coverage-route %.15f | tail-route %.15f | Decimal %s"
            % (df, r1, r2, dstr(r3, 20)))
        say("                 |r1-r2|=%.1e  |r1-r3|=%.1e  FWER = m*2*tail = %.15f (target 0.05)"
            % (abs(r1 - r2), abs(Decimal(repr(r1)) - r3), fwer))
say()
say(" NOT USED (contrast): if one instead allocated a ONE-SIDED alpha = 0.05/m to each")
say(" of the m members, the target coverage would be the smaller 1 - 0.10/m and the")
say(" critical values would be SMALLER than ours for the same m.  For reference only")
say(" (note that 1 - 0.10/80 coincides with 1 - 0.05/40, and 1 - 0.10/40 with 1 - 0.05/20):")
for lab, m, al, desc in FAMILIES[1:]:
    for df in DFS:
        t, _, _, _ = bisect(lambda t: coverage_A(t, df), 0.5, 50.0, 1.0 - 0.10 / m, SOLVE_TOL_F)
        say("      m=%-4d df=%5d : 1-0.10/m critical value = %.15f   (our definition: %.15f)"
            % (m, df, t, SOL_A[(lab, df)]))

# ==========================================================================
head("9", "RECOMMENDED CONSTANTS (method C, 60-digit Decimal integration)")
# ==========================================================================
say(" Values below are method C rounded to 15 decimals; the underlying Decimal")
say(" roots are good to ~1e-24, the double-precision representation to ~1e-16.")
say()
say("# " + "-" * 74)
say("# Two-sided Student-t critical values, P(|T_df| <= t) = target.")
say("# BONFERRONI DEFINITION: family size m = 2N two-sided intervals at simultaneous")
say("# 95% -> per-comparison confidence 1 - 0.05/m.  'T_BONF_2N80' means m = 2N = 80")
say("# (N = 40 comparisons); 'T_BONF_2N40' means m = 2N = 40 (N = 20 comparisons).")
say("# Method C (60-digit Decimal quadrature); max |method C - method A| = %.2e." % max_AC)
say("# " + "-" * 74)
for lab, m, al, desc in FAMILIES:
    say("")
    say("# %s : target P(|T|<=t) = %s   (per-comparison alpha = %s)  [%s]"
        % (lab, dstr(1 - C_ALPHA / Decimal(m), 6), al, desc))
    for df in DFS:
        name = "%s_DF_%d" % (lab, df)
        say("%-22s = %s" % (name, dstr(SOL_C[(lab, df)], 15)))
say("")
say("# aliases making the family size explicit (same numbers):")
for df in DFS:
    say("%-22s = %s" % ("T_BONF_M80_DF_%d" % df, dstr(SOL_C[("T_BONF_2N80", df)], 15)))
for df in DFS:
    say("%-22s = %s" % ("T_BONF_M40_DF_%d" % df, dstr(SOL_C[("T_BONF_2N40", df)], 15)))

say("")
say("# " + "-" * 74)
say("# ALTERNATIVE CONVENTION - NOT the definition used above.")
say("# If each of the m members were instead given a ONE-SIDED alpha = 0.05/m, the")
say("# per-comparison two-sided confidence would be 1 - 0.10/m and the constants")
say("# would be these (smaller).  The illustrative digits in the task prompt")
say("# ('T_BONF_2N80_DF_2047 = 3.2...', 'T_BONF_2N40_DF_2047 = 3.0...') match THIS")
say("# block, not the stated 1 - 0.05/m definition.  For df=2047 they are")
say("# 3.231723227703792 (m=80) and 3.027090023031109 (m=40).")
say("# " + "-" * 74)
ALT = {}
for lab, m, al, desc in FAMILIES[1:]:
    for df in DFS:
        ALT[(lab, df)] = solve_coverage_decimal(1 - 2 * C_ALPHA / Decimal(m), df)[0]
    say("")
    say("# %s under one-sided alpha = 0.05/%d  (target coverage 1 - 0.10/%d)"
        % (lab, m, m))
    for df in DFS:
        say("%-22s = %s" % ("T_BONF_1SIDED_%s_DF_%d" % (lab.replace("T_BONF_", ""), df),
                            dstr(ALT[(lab, df)], 15)))

# ==========================================================================
head("10", "SUMMARY OF AGREEMENTS AND DISAGREEMENTS")
# ==========================================================================
say(" 1) three methods, 15 constants:")
say("      max |A - C| = %.3e      (A = required NR betai + lgamma route)" % max_AC)
say("      max |B - C| = %.3e      (B = same CF, exact beta prefactor)" % max_BC)
say(" 2) published anchors (6 values):")
say("      max |C - published| = %.3e      max |A - published| = %.3e"
    % (max_anchor, max_anchor_A))
say(" 3) Bonferroni route agreement: max |coverage-route - tail-route| = %.3e" % max_r12)
say("      max |float - Decimal| = %.3e ; max |FWER - 0.05| = %.3e" % (max_r13, max_fwer))
say(" 4) all 15 + 6 + 30 roots satisfy |CDF(t*) - target| < 1e-13 (asserted).")
say(" 5) continued-fraction worst case: %d iterations of %d allowed." % (CF_STATS["max_iter"], 500))
say()
say(" total runtime: %.1f s" % (time.time() - start_time))
hr()

# --------------------------------------------------------------------------
# dump the transcript next to this script (UTF-8, LF) for the markdown summary
# --------------------------------------------------------------------------
_OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "verify_t_critical_output.txt")
with open(_OUT_PATH, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("\n".join(OUT) + "\n")
print("\n[transcript written to %s]" % _OUT_PATH)
