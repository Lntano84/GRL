#!/usr/bin/env python3
"""Scratch probe: validate the Decimal high-precision machinery before the real script."""
import math
import time
from decimal import Decimal, localcontext


def machin_pi(prec):
    """pi to `prec` digits via Machin's formula (pure Decimal, integer powers only)."""
    with localcontext() as ctx:
        ctx.prec = prec + 15
        eps = Decimal(1).scaleb(-(prec + 12))

        def atan_inv(x):
            xd = Decimal(x)
            x2 = xd * xd
            total = Decimal(0)
            term = Decimal(1) / xd
            n = 1
            sign = 1
            while term > eps:
                add = term / n
                total = total + add if sign > 0 else total - add
                term = term / x2
                n += 2
                sign = -sign
            return total

        pi = 16 * atan_inv(5) - 4 * atan_inv(239)
    return +pi


def gamma_half(m, PI):
    """Gamma(m/2) for integer m >= 1 in Decimal (exact factorials)."""
    if m % 2 == 0:
        return Decimal(math.factorial(m // 2 - 1))
    p = (m - 1) // 2
    num = Decimal(math.factorial(2 * p))
    den = Decimal(4) ** p * Decimal(math.factorial(p))
    return num / den * PI.sqrt()


def gl_nodes(n, prec):
    with localcontext() as ctx:
        ctx.prec = prec + 15
        xs, ws = [], []
        for i in range(1, n + 1):
            x = Decimal(math.cos(math.pi * (i - 0.25) / (n + 0.5)))
            for _ in range(200):
                p0, p1 = Decimal(1), x
                for j in range(2, n + 1):
                    p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
                pn = p1
                dpn = n * (x * pn - p0) / (x * x - 1)
                dx = pn / dpn
                x -= dx
                if abs(dx) < Decimal(1).scaleb(-(prec + 10)):
                    break
            p0, p1 = Decimal(1), x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            pn = p1
            dpn = n * (x * pn - p0) / (x * x - 1)
            w = 2 / ((1 - x * x) * dpn * dpn)
            xs.append(+x)
            ws.append(+w)
    return xs, ws


t0 = time.time()
PI = machin_pi(60)
print("pi 60 =", PI)
print("math.pi=", Decimal(math.pi))
print("pi match to 16 digits:", str(PI)[:18] == str(Decimal(math.pi))[:18])
print("|pi - math.pi| =", abs(PI - Decimal(math.pi)))
print("t=%.2fs" % (time.time() - t0))

print()
print("Gamma checks:")
print("  G(1)   =", gamma_half(2, PI))
print("  G(1/2) =", gamma_half(1, PI), " sqrt(pi)=", PI.sqrt())
print("  G(3/2) =", gamma_half(3, PI), " sqrt(pi)/2=", PI.sqrt() / 2)
print("  G(21/2)/G(19/2) =", gamma_half(21, PI) / gamma_half(19, PI), "(expect 9.5)")
print("  G(1024)/G(1023) =", gamma_half(2048, PI) / gamma_half(2046, PI), "(expect 1023)")

t0 = time.time()
xs, ws = gl_nodes(20, 60)
print()
print("GL nodes n=20 built in %.2fs" % (time.time() - t0))
with localcontext() as ctx:
    ctx.prec = 60
    print("  sum w        =", sum(ws), "(expect 2)")
    print("  int x^2      =", sum(w * x * x for x, w in zip(xs, ws)), "(expect", Decimal(2) / 3, ")")
    print("  int x^38     =", sum(w * x ** 38 for x, w in zip(xs, ws)), "(expect", Decimal(2) / 39, ")")
    print("  int x^40     =", sum(w * x ** 40 for x, w in zip(xs, ws)), "(expect", Decimal(2) / 41, ")")
    print("  min/max node =", xs[0], xs[-1])

# --- Student-t density constant and Decimal coverage -------------------------
print()
with localcontext() as ctx:
    ctx.prec = 60
    for nu in (1, 2, 19, 39, 99, 255, 2047):
        k = (nu + 1) / 2
        kint = (nu + 1) // 2
        g1 = gamma_half(nu + 1, PI)
        g2 = gamma_half(nu, PI)
        C = g1 / ((Decimal(nu) * PI).sqrt() * g2)
        # float lgamma reference
        Cf = math.exp(math.lgamma((nu + 1) / 2) - 0.5 *
                      math.log(nu * math.pi) - math.lgamma(nu / 2))
        print("nu=%5d  C_dec=%s  C_float=%.17g  reldiff=%.3e  k_int=%d" %
              (nu, str(C)[:22], Cf, abs(float(C) - Cf) / Cf, kint))


def dec_coverage(t, nu, C, kint, n_nodes, n_panels):
    """2*C*int_0^t (1+x^2/nu)^-k dx by composite Gauss-Legendre."""
    if t == 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 60
        xs, ws = gl_nodes(n_nodes, 60)
        total = Decimal(0)
        h = t / n_panels
        for p in range(n_panels):
            a = p * h
            c = a + h / 2
            for x, w in zip(xs, ws):
                pt = c + (h / 2) * x
                y = 1 + pt * pt / Decimal(nu)
                total += w * y ** (-kint)
        total = total * (h / 2)
        return 2 * C * total


with localcontext() as ctx:
    ctx.prec = 60
    for nu, t in ((19, Decimal("2.0930240544083")), (2047, Decimal("1.9611")), (1, Decimal("12.7062047361747"))):
        kint = (nu + 1) // 2
        C = gamma_half(nu + 1, PI) / ((Decimal(nu) * PI).sqrt() * gamma_half(nu, PI))
        t0 = time.time()
        cov = dec_coverage(t, nu, C, kint, 20, 4)
        dt1 = time.time() - t0
        t0 = time.time()
        cov2 = dec_coverage(t, nu, C, kint, 30, 8)
        dt2 = time.time() - t0
        print("nu=%5d t=%s cov(20,4)=%s cov(30,8)=%s  |diff|=%.3e  (%.3fs/%.3fs)" %
              (nu, t, str(cov)[:24], str(cov2)[:24], abs(cov - cov2), dt1, dt2))
