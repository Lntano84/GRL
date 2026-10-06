"""
Stage 04 instance generator: three scales x two production loads x two seeds = 12 nominal
instances, each with two disruptions = 24 disruption states.

These are OUR CALIBRATION INSTANCES, not a reproduction of the authors' instances. The paper's
Set 1/Set 2 use M in {2,3,4}, N in {30,35,40}, T = 30, capacities in {3000,3500,4000} and
time-decreasing lost-sales costs (arXiv:2605.27339v1, Sec. 4.1.2). We deliberately scale down
and hold the lost-sales penalties constant.

Frozen generation rule (do not adjust after seeing results)
-----------------------------------------------------------
  c_jt = 100 for every machine and period; every item compatible with every machine
  b_i = 1 ; p_i = 0 ; I_i0 = 0
  s_i ~ U[10, 20]                       (setup time)
  f_i = 0.1 * s_i                       (setup cost)
  h_i ~ U[0.05, 0.35]                   (inventory cost)
  v_i ~ U[0.6, 1.4]                     (minimum-lot multiplier)
  items split into high / medium / low with counts
      ceil(N/4), ceil(N/3), remainder
  group weights 0.45 / 0.40 / 0.15, shared equally inside a group, so sum_i w_i = 1
  d_it = rho * 100 * M * w_i * u_it ,  u_it ~ U[0.8, 1.2]
  lost-sales penalties 10 / 4 / 1 per group, constant over time
  m_i = v_i * mean_t(d_it)

rho is a PRODUCTION LOAD parameter and does NOT account for setup time; it is therefore not
an achieved capacity utilization and must not be described as one.

Sampling order is fixed and recorded: per item, s_i then h_i then v_i in item order; then
u_it in item-major, period-minor order. Everything is written to JSON BEFORE any solve.

Disruptions
-----------
  D1 : machine 0 down for periods 1-2
  D2 : every machine down for period 1
No re-sampling is permitted because of poor improvement, hard solves or large losses.
"""

from __future__ import annotations

import json
import math
import random
import zlib
from typing import Any, Dict, List, Tuple

from lsp_model import Instance

SCALES: Dict[str, Dict[str, int]] = {
    "small": {"N": 6, "M": 2, "T": 6, "tau": 3},
    "medium": {"N": 12, "M": 3, "T": 12, "tau": 6},
    "large": {"N": 24, "M": 3, "T": 18, "tau": 6},
}
RHOS = (0.75, 1.10)
SEEDS = (0, 1)
CAPACITY = 100.0
GROUP_WEIGHTS = (0.45, 0.40, 0.15)
GROUP_LOST_SALES = (10.0, 4.0, 1.0)


def group_sizes(N: int) -> Tuple[int, int, int]:
    n_high = math.ceil(N / 4)
    n_medium = math.ceil(N / 3)
    n_low = N - n_high - n_medium
    assert n_low >= 0, (N, n_high, n_medium, n_low)
    return n_high, n_medium, n_low


def instance_seed(scale: str, rho: float, seed: int) -> int:
    """Deterministic across runs and Python versions.

    `hash()` is NOT usable here: it is salted for str/bytes (PYTHONHASHSEED) and its
    tuple/float behaviour is not a documented contract. crc32 over an explicit string is.
    """
    key = "stage04|%s|%.6f|%d" % (scale, rho, seed)
    return zlib.crc32(key.encode("ascii")) & 0xFFFFFFFF


def build_instance(scale: str, rho: float, seed: int) -> Tuple[Instance, Dict[str, Any]]:
    spec = SCALES[scale]
    N, M, T = spec["N"], spec["M"], spec["T"]
    rng_seed = instance_seed(scale, rho, seed)
    rng = random.Random(rng_seed)

    name = "%s_rho%.2f_s%d" % (scale, rho, seed)

    # ---- sampling order is part of the frozen definition
    order: List[str] = []
    s = []
    for i in range(N):
        s.append(rng.uniform(10.0, 20.0))
        order.append("s[%d]" % i)
    h = []
    for i in range(N):
        h.append(rng.uniform(0.05, 0.35))
        order.append("h[%d]" % i)
    v = []
    for i in range(N):
        v.append(rng.uniform(0.6, 1.4))
        order.append("v[%d]" % i)

    n_high, n_medium, n_low = group_sizes(N)
    groups = ([0] * n_high) + ([1] * n_medium) + ([2] * n_low)
    assert len(groups) == N
    w = []
    for g, (cnt, wsum) in enumerate(zip((n_high, n_medium, n_low), GROUP_WEIGHTS)):
        w.extend([wsum / cnt] * cnt)
    assert abs(sum(w) - 1.0) < 1e-12, sum(w)

    d: List[List[float]] = []
    for i in range(N):
        row = []
        for t in range(T):
            u = rng.uniform(0.8, 1.2)
            order.append("u[%d][%d]" % (i, t))
            row.append(rho * CAPACITY * M * w[i] * u)
        d.append(row)

    f = [0.1 * s[i] for i in range(N)]
    p = [0.0] * N
    b = [1.0] * N
    l = [GROUP_LOST_SALES[groups[i]] for i in range(N)]
    m = [v[i] * (sum(d[i]) / T) for i in range(N)]
    c = [[CAPACITY] * T for _ in range(M)]
    compat = [[1] * M for _ in range(N)]

    inst = Instance(name=name, N=N, M=M, T=T, f=f, p=p, h=h, l=l, s=s, b=b, m=m,
                    d=d, c=c, w=compat, I0=[0.0] * N,
                    note="stage04 calibration instance, scale=%s rho=%.2f seed=%d"
                         % (scale, rho, seed))
    meta = {
        "name": name, "scale": scale, "rho": rho, "seed": seed,
        "N": N, "M": M, "T": T, "tau": spec["tau"],
        "capacity_per_machine_period": CAPACITY,
        "group_sizes": {"high": n_high, "medium": n_medium, "low": n_low},
        "group_weights": list(GROUP_WEIGHTS),
        "group_lost_sales": list(GROUP_LOST_SALES),
        "parameter_arrays": {"f": f, "p": p, "h": h, "l": l, "s": s, "b": b, "m": m,
                             "d": d, "c": c, "w": compat, "I0": [0.0] * N,
                             "groups": groups, "weights": w},
        "sampling_order": "s[i] for i=0..N-1; h[i]; v[i]; then u[i][t] item-major, "
                          "period-minor.",
        "rng_seed": rng_seed,
        "rng_seed_rule": "zlib.crc32('stage04|<scale>|<rho:%.6f>|<seed>') & 0xFFFFFFFF",
        "derived": {
            "total_demand": sum(sum(row) for row in d),
            "total_capacity": CAPACITY * M * T,
            "demand_over_capacity": sum(sum(row) for row in d) / (CAPACITY * M * T),
            "max_setup_time_share_of_capacity": max(s) / CAPACITY,
            "mean_minlot_over_mean_demand": sum(m) / sum(sum(row) / T for row in d),
        },
        "note": "rho is a production-load parameter and does NOT include setup time; it is "
                "not an achieved capacity utilization.",
    }
    return inst, meta


def all_instances() -> List[Tuple[Instance, Dict[str, Any]]]:
    out = []
    for scale in ("small", "medium", "large"):
        for rho in RHOS:
            for seed in SEEDS:
                out.append(build_instance(scale, rho, seed))
    return out


# --------------------------------------------------------------------------------------
# Disruptions
# --------------------------------------------------------------------------------------


class Disruption:
    def __init__(self, name: str, down: Dict[int, List[int]], note: str = ""):
        self.name = name
        self.down = {int(j): sorted(int(t) for t in ts) for j, ts in down.items()}
        self.note = note

    def delta_for(self, j: int) -> int:
        ts = self.down.get(j, [])
        d = 0
        while (d + 1) in ts:
            d += 1
        return d

    def is_down(self, j: int, t: int) -> bool:
        return t in self.down.get(j, [])

    def describe(self) -> str:
        if not self.down:
            return "no disruption"
        return "; ".join("machine %d down during %s" % (j, ts)
                         for j, ts in sorted(self.down.items()))

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "down": {str(j): ts for j, ts in self.down.items()},
                "note": self.note, "description": self.describe()}


def disruptions_for(inst: Instance) -> List[Disruption]:
    return [
        Disruption("D1_m0_2p", {0: [1, 2]}, "machine 0 down for periods 1-2"),
        Disruption("D2_all_1p", {j: [1] for j in range(inst.M)},
                   "every machine down for period 1"),
    ]


def apply_disruption(inst: Instance, dis: Disruption) -> Instance:
    """Only the affected capacities change. Everything else is carried over, so the derived
    activation bound c'_jt / b_i automatically becomes 0 in a downed period."""
    c = [[float(inst.c[j][t - 1]) for t in range(1, inst.T + 1)] for j in range(inst.M)]
    for j, ts in dis.down.items():
        for t in ts:
            if 1 <= t <= inst.T:
                c[j][t - 1] = 0.0
    return Instance(name="%s|%s" % (inst.name, dis.name), N=inst.N, M=inst.M, T=inst.T,
                    f=list(inst.f), p=list(inst.p), h=list(inst.h), l=list(inst.l),
                    s=list(inst.s), b=list(inst.b), m=list(inst.m),
                    d=[list(r) for r in inst.d], c=c,
                    w=[list(r) for r in inst.w], I0=list(inst.I0),
                    note="%s :: %s" % (inst.note, dis.describe()))


def write_instances(path: str) -> List[Dict[str, Any]]:
    """Write every instance and its full parameter arrays BEFORE any solve."""
    payload = [meta for _, meta in all_instances()]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"scales": SCALES, "rhos": list(RHOS), "seeds": list(SEEDS),
                   "instances": payload}, fh, indent=2)
    return payload


if __name__ == "__main__":
    metas = write_instances("stage04_instances.json")
    print("%-22s %8s %8s %8s %10s %10s" %
          ("instance", "N", "M", "T", "d/cap", "max s/cap"))
    for m in metas:
        print("%-22s %8d %8d %8d %10.4f %10.4f" %
              (m["name"], m["N"], m["M"], m["T"],
               m["derived"]["demand_over_capacity"],
               m["derived"]["max_setup_time_share_of_capacity"]))
