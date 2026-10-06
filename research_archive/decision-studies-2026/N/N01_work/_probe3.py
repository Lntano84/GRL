#!/usr/bin/env python3
"""Probe 3: rational substitution u = tanh(w/2) -> integer powers only, no exp/ln.

coverage(t) = 4*C_nu*sqrt(nu) * int_0^U (1-u^2)^(nu-1) (1+u^2)^(-nu) du,
U = z/(1+sqrt(1+z^2)),  z = t/sqrt(nu).
"""
import math
import time
from decimal import Decimal, localcontext


def machin_pi(prec):
    with localcontext() as ctx:
        ctx.prec = prec + 15
        eps = Decimal(1).scaleb(-(prec + 12))

        def atan_inv(x):
            xd = Decimal(x); x2 = xd * xd
            total = Decimal(0); term = Decimal(1) / xd
            n, sign = 1, 1
            while term > eps:
                add = term / n
                total = total + add if sign > 0 else total - add
                term = term / x2; n += 2; sign = -sign
            return total

        pi = 16 * atan_inv(5) - 4 * atan_inv(239)
    return +pi


def gamma_half(m, PI):
    if m % 2 == 0:
        return Decimal(math.factorial(m // 2 - 1))
    p = (m - 1) // 2
    return Decimal(math.factorial(2 * p)) / (Decimal(4) ** p * Decimal(math.factorial(p))) * PI.sqrt()


_cache = {}


def gl_nodes(n, prec):
    if (n, prec) in _cache:
        return _cache[(n, prec)]
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
            ws.append(+(2 / ((1 - x * x) * dpn * dpn)))
            xs.append(+x)
        _cache[(n, prec)] = (xs, ws)
        return xs, ws


PI = machin_pi(60)


def coverage_u(t, nu, n_nodes=24, n_panels=6, prec=60):
    if t == 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = prec
        t = Decimal(t)
        nuD = Decimal(nu)
        C = gamma_half(nu + 1, PI) / ((nuD * PI).sqrt() * gamma_half(nu, PI))
        z = t / nuD.sqrt()
        U = z / (1 + (1 + z * z).sqrt())
        xs, ws = gl_nodes(n_nodes, prec)[0][:n_nodes], gl_nodes(n_nodes, prec)[1][:n_nodes]
        acc = Decimal(0)
        h = U / n_panels
        for p in range(n_panels):
            c = (p + Decimal("0.5")) * h
            for x, w in zip(xs, ws):
                u = c + (h / 2) * x
                g = (1 - u * u) ** (nu - 1) / (1 + u * u) ** nu
                acc += w * g
        acc = acc * (h / 2)
        return 4 * C * nuD.sqrt() * acc


# ---- cross-check against the sinh/exp formulation (independent substitution) ----
def coverage_sinh(t, nu, n_nodes=24, n_panels=6, prec=60):
    if t == 0:
        return Decimal(0)
    with localcontext() as ctx:
        ctx.prec = prec
        t = Decimal(t); nuD = Decimal(nu)
        C = gamma_half(nu + 1, PI) / ((nuD * PI).sqrt() * gamma_half(nu, PI))
        W = (t / nuD.sqrt() + (t * t / nuD + 1).sqrt()).ln()
        xs, ws = gl_nodes(n_nodes, prec)
        acc = Decimal(0)
        h = W / n_panels
        for p in range(n_panels):
            c = (p + Decimal("0.5")) * h
            for x, w in zip(xs, ws):
                pt = c + (h / 2) * x
                ch = (pt.exp() + (-pt).exp()) / 2
                acc += w / (ch ** nu)
        acc = acc * (h / 2)
        return 2 * C * nuD.sqrt() * acc


cases = [(1, "12.7062047361747"), (2, "4.302652729911275"), (10, "2.2281388519649385"),
         (19, "2.0930240544083"), (19, "3.6"), (39, "2.0227"), (99, "1.9842"),
         (255, "1.9690"), (2047, "1.9611"), (2047, "3.63")]

with localcontext() as ctx:
    ctx.prec = 60
    print("%-6s %-20s %-28s %-10s %-10s %-10s" % ("nu", "t", "cov(24,6)", "conv", "u-vs-sinh", "time"))
    for nu, ts in cases:
        t0 = time.time(); a = coverage_u(ts, nu, 24, 6); d1 = time.time() - t0
        b = coverage_u(ts, nu, 32, 12)
        c = coverage_u(ts, nu, 16, 3)
        s = coverage_sinh(ts, nu, 24, 6)
        print("%-6d %-20s %-28s %.2e %.2e %.3fs" %
              (nu, ts, str(a)[:26], abs(a - b), abs(a - s), d1))
        print("        (16,3) vs (32,12) diff = %.2e ; (24,6) vs (32,12) = %.2e ; (24,6) vs (16,3) = %.2e" %
              (abs(c - b), abs(a - b), abs(a - c)))
