#!/usr/bin/env python3
"""Probe 2: sinh substitution x = sqrt(nu)*sinh(w) for the Decimal t-density integral."""
import math
import time
from decimal import Decimal, localcontext


def machin_pi(prec):
    with localcontext() as ctx:
        ctx.prec = prec + 15
        eps = Decimal(1).scaleb(-(prec + 12))

        def atan_inv(x):
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

        pi = 16 * atan_inv(5) - 4 * atan_inv(239)
    return +pi


def gamma_half(m, PI):
    if m % 2 == 0:
        return Decimal(math.factorial(m // 2 - 1))
    p = (m - 1) // 2
    return Decimal(math.factorial(2 * p)) / (Decimal(4) ** p * Decimal(math.factorial(p))) * PI.sqrt()


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
                dpn = n * (x * p1 - p0) / (x * x - 1)
                dx = p1 / dpn
                x -= dx
                if abs(dx) < Decimal(1).scaleb(-(prec + 10)):
                    break
            p0, p1 = Decimal(1), x
            for j in range(2, n + 1):
                p0, p1 = p1, ((2 * j - 1) * x * p1 - (j - 1) * p0) / j
            dpn = n * (x * p1 - p0) / (x * x - 1)
            ws.append(+ (2 / ((1 - x * x) * dpn * dpn)))
            xs.append(+x)
    return xs, ws


PI = machin_pi(60)
XS, WS = gl_nodes(24, 60)


def coverage(t, nu, n_nodes=24, n_panels=6):
    """P(|T|<=t) for T~t_nu, by Decimal integration of the density.

    x = sqrt(nu) sinh w  =>  (1+x^2/nu)^-k dx = sqrt(nu) cosh(w)^(1-2k) dw, 1-2k = -nu
    """
    if t == 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = 60
        t = Decimal(t)
        nuD = Decimal(nu)
        C = gamma_half(nu + 1, PI) / ((nuD * PI).sqrt() * gamma_half(nu, PI))
        W = (t / nuD.sqrt() + (t * t / nuD + 1).sqrt()).ln()
        xs, ws = XS[:n_nodes], WS[:n_nodes]
        total = Decimal(0)
        h = W / n_panels
        for p in range(n_panels):
            c = (p + Decimal("0.5")) * h
            for x, w in zip(xs, ws):
                pt = c + (h / 2) * x
                ch = (pt.exp() + (-pt).exp()) / 2
                total += w / (ch ** nu)
        total = total * (h / 2) * nuD.sqrt()
        return 2 * C * total


cases = [
    (1, Decimal("12.7062047361747")),
    (2, Decimal("4.302652729911275")),
    (10, Decimal("2.2281388519649385")),
    (19, Decimal("2.0930240544083")),
    (19, Decimal("3.6")),
    (39, Decimal("2.0227")),
    (99, Decimal("1.9842")),
    (255, Decimal("1.9690")),
    (2047, Decimal("1.9611")),
    (2047, Decimal("3.63")),
]
with localcontext() as ctx:
    ctx.prec = 60
    for nu, t in cases:
        t0 = time.time()
        a = coverage(t, nu, 24, 6)
        d1 = time.time() - t0
        t0 = time.time()
        b = coverage(t, nu, 32, 10)
        d2 = time.time() - t0
        t0 = time.time()
        c = coverage(t, nu, 20, 4)
        d3 = time.time() - t0
        print("nu=%5d t=%-22s cov=%s\n             |(24,6)-(32,10)|=%.3e  |(20,4)-(32,10)|=%.3e  [%.3f/%.3f/%.3fs]" %
              (nu, t, str(b)[:26], abs(a - b), abs(c - b), d1, d2, d3))
