"""Standalone AC snapshot test of the unchanged FRAME search rule."""
from collections import defaultdict
from copy import deepcopy
from hashlib import sha256
from itertools import combinations
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.networks import case118, case300

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
OUT = ROOT / "outputs/grid340_pandapower_transfer"
POLICY = ROOT / "work/grid91_mechanism/policies.py"
CASES = {"ieee118": case118, "ieee300": case300}
LOADS = (1.0, 1.05)
WALL_CAP = 600
SOLVE_CAP = 1804
POLICIES = ("NO_LOCAL", "GLOBAL_MATCH", "FRAME_GUARD")
VOLTAGE = (0.90, 1.10)

def candidates(net):
    by_station = defaultdict(list)
    for line, row in net.line.iterrows():
        if row.in_service:
            by_station[int(row.from_bus)].append((int(line), "from_bus"))
            by_station[int(row.to_bus)].append((int(line), "to_bus"))
    groups = {}
    for bus, terminals in sorted(by_station.items()):
        terminals = sorted(terminals)
        options = [tuple(bundle) for size in (2, 3) if len(terminals) - size >= 2
                   for bundle in combinations(terminals[1:], size)]
        if options:
            groups[bus] = options
    # Fixed round-robin station interleave. The first terminal stays
    # on busbar A, removing complementary duplicates in this restricted library.
    return [(bus, groups[bus][i]) for i in range(max(map(len, groups.values())))
            for bus in groups if i < len(groups[bus])]

def solve(net):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pp.runpp(net, algorithm="nr", calculate_voltage_angles=True, init="flat",
                     max_iteration=20, tolerance_mva=1e-8, numba=False, lightsim2grid=False)
    except LoadflowNotConverged:
        return {"converged": False, "voltage_ok": False, "rho": None}
    result = net.res_bus.loc[net.bus.in_service, "vm_pu"].to_numpy()
    branch = np.concatenate([
        net.res_line.loc[net.line.in_service, "loading_percent"].to_numpy(),
        net.res_trafo.loc[net.trafo.in_service, "loading_percent"].to_numpy(),
    ])
    if not np.isfinite(result).all() or not np.isfinite(branch).all():
        return {"converged": True, "voltage_ok": False, "rho": None}
    return dict(converged=True, voltage_ok=bool(((VOLTAGE[0] <= result) & (result <= VOLTAGE[1])).all()),
                rho=float(branch.max() / 100), min_v=float(result.min()), max_v=float(result.max()))

def load_policy():
    fixed = json.loads((HERE / "protocol.json").read_bytes())
    digest = sha256(POLICY.read_bytes()).hexdigest()
    if digest != fixed["policy_sha256"] or version("pandapower") != fixed["pandapower"]:
        raise RuntimeError("Frozen source or simulator version differs")
    spec = importlib.util.spec_from_file_location("frozen_frame", POLICY)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    assert policy.DEFAULT == dict(p=128, h=64, patience=2, safe=.9)
    return policy, fixed

def summarize(policy, library, feedback):
    order = list(range(len(library)))
    buses = [bus for bus, _ in library]
    aliases = dict(zip(order, order))
    best = max((row for row in feedback if row["strict_admissible"]), key=policy.key, default=None)
    result = {"FULL": {"queries": len(order), "winner": None if best is None else best["pool_id"],
                       "rho": None if best is None else best["rho"]}}
    for name in POLICIES:
        visited = []

        def observe(cid, phase):
            visited.append([cid, phase])
            return feedback[cid]

        record = policy.run(order, observe, name, buses, aliases)
        cid = record["chosen_pool_id"]
        result[name] = dict(queries=record["public_queries"], winner=cid,
                            rho=None if cid is None else feedback[cid]["rho"],
                            prefix_winner=record["pre_extra_choice"],
                            lock=record["local_substation"], stop_reason=record["stop_reason"],
                            local_queries=record["local_queries"], visited=visited)
    assert result["FRAME_GUARD"]["queries"] == result["GLOBAL_MATCH"]["queries"]
    return result

def main():
    policy, fixed = load_policy()
    start = time.monotonic()
    OUT.mkdir(exist_ok=False)
    count = 0
    results = []

    def checked_solve(net):
        nonlocal count
        if count >= SOLVE_CAP or time.monotonic() - start >= WALL_CAP:
            raise TimeoutError("Budget exhausted")
        count += 1
        return solve(net)

    with (OUT / "observations.jsonl").open("x", encoding="utf8") as stream:
        for name, make in CASES.items():
            grid = make()
            library = candidates(grid)
            assert len(library) == fixed["candidate_counts"][name], name
            serial = [(bus, [list(terminal) for terminal in moved]) for bus, moved in library]
            (OUT / f"{name}_candidates.json").write_text(json.dumps(serial) + "\n", encoding="utf8")
            for scale in LOADS:
                net = deepcopy(grid)
                net.load["scaling"] *= scale
                base = checked_solve(net)
                assert base["converged"] and base["rho"] is not None, (name, scale, base)
                stream.write(json.dumps(dict(case=name, scale=scale, pool_id=None, **base)) + "\n")
                feedback = []
                for index, (bus, moved) in enumerate(library):
                    trial = deepcopy(net)
                    bar = pp.create_bus(trial, vn_kv=float(net.bus.at[bus, "vn_kv"]),
                                        name=f"Split bus {bus}")
                    for line, terminal in moved:
                        assert trial.line.at[line, terminal] == bus
                        trial.line.at[line, terminal] = bar
                    row = checked_solve(trial)
                    rho = row["rho"]
                    feedback.append(dict(pool_id=index, reward=0.0 if rho is None else 2.0-rho,
                                         strict_admissible=row["voltage_ok"] and rho is not None
                                         and 0 < rho < min(1.0, base["rho"]), **row))
                    stream.write(json.dumps(dict(case=name, scale=scale, **feedback[-1])) + "\n")
                    stream.flush()
                outcomes = summarize(policy, library, feedback)
                result = dict(case=name, scale=scale, base=base,
                              admissible=sum(r["strict_admissible"] for r in feedback),
                              nonconverged=sum(not r["converged"] for r in feedback), arms=outcomes)
                results.append(result)
                print(json.dumps(dict(case=name, scale=scale, count=count,
                      outcomes={k: {f:v for f,v in a.items() if f != "visited"}
                                for k,a in outcomes.items()})), flush=True)
    assert count == SOLVE_CAP
    summary = dict(cases=results, solves=count, wall_s=time.monotonic()-start,
                   new_fits=0, grid2op_steps=0, pandapower=version("pandapower"))
    (OUT / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
                                      encoding="utf8")

if __name__ == "__main__":
    main()
