"""Fixed PGLib candidate comparison with immutable saved baselines."""
from collections import Counter
from copy import deepcopy
from importlib import util
from importlib.metadata import version
import json
from pathlib import Path
import sys
import time

import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
OUT = ROOT / "outputs/grid342_pglib_frame_comparison"
BASE = HERE.parent / "GRID341_pglib_baseline"
NAMES = ("pglib_opf_case118_ieee.m", "pglib_opf_case118_ieee__api.m")
LIMIT = 1284
WALL = 600

def module(name, path):
    spec = util.spec_from_file_location(name, path)
    value = util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

baseline = module("grid341", BASE / "baseline.py")
previous = module("grid340", baseline.PREVIOUS)

def frozen_inputs():
    freeze = json.loads((HERE / "references.json").read_bytes())
    for name, expected in freeze["hashes"].items():
        assert baseline.digest(ROOT / name) == expected, name
    assert version("pandapower") == "3.5.6"
    receipt = json.loads((BASE / "result.json").read_bytes())
    qualified = [row for row in receipt["rows"] if row["qualified"]]
    assert receipt["baseline_attempts"] == 4 and receipt["candidate_solves"] == 0
    assert [row["case"] for row in qualified] == list(NAMES)
    policy, _ = previous.load_policy()
    return qualified, policy

def imported(receipt):
    source = baseline.OUT / "inputs" / receipt["case"]
    assert baseline.digest(source) == receipt["sha256"]
    case = baseline.parse(source)
    net = baseline.from_ppc(case, f_hz=60)
    lookup, active = baseline.check_import(case, net)
    library = previous.candidates(net)
    assert len(library) == receipt["candidates"] == 642
    assert len(net.bus) == receipt["bus_count"] == 118
    assert int(active.sum()) == receipt["active_branch_count"] == 186
    return case, net, lookup, active, library

def connected(net, lookup, active):
    """Count all active buses using the unchanged branch indices and service flags."""
    adjacency = {int(bus): set() for bus in net.bus.index[net.bus.in_service]}
    for kind, (ends, _) in baseline.COLS.items():
        selected = (lookup.element_type == kind).to_numpy() & active
        frame = net[kind].loc[lookup.loc[selected, "element"].astype(int)]
        assert frame.in_service.all()
        for origin, end in frame.loc[:, list(ends)].itertuples(index=False, name=None):
            adjacency[origin].add(end)
            adjacency[end].add(origin)
    seen = {next(iter(adjacency))}
    frontier = list(seen)
    while frontier:
        for neighbor in adjacency[frontier.pop()] - seen:
            frontier.append(neighbor)
            seen.add(neighbor)
    return len(seen) == len(adjacency)

def split(net, lookup, active, station, terminals):
    trial = deepcopy(net)
    new_bus = pp.create_bus(trial, vn_kv=float(trial.bus.at[station, "vn_kv"]),
                           name=f"Split bus {station}")
    for line, side in terminals:
        assert trial.line.at[line, side] == station
        trial.line.at[line, side] = new_bus
    assert len(trial.bus) == len(net.bus) + 1
    for kind in baseline.COLS:
        assert trial[kind].index.equals(net[kind].index)
    return trial, connected(trial, lookup, active)

def qualification():
    receipts, _ = frozen_inputs()
    assert not OUT.exists()
    OUT.mkdir()
    records = []
    for receipt in receipts:
        _, net, lookup, active, library = imported(receipt)
        assert connected(net, lookup, active)
        path = OUT / (Path(receipt["case"]).stem + "_library.json")
        path.write_text(json.dumps(library) + "\n", encoding="utf8")
        islands = []
        for index, (station, terminals) in enumerate(library):
            _, continuous = split(net, lookup, active, station, terminals)
            if not continuous:
                islands.append(index)
        records.append(dict(case=receipt["case"], paths=path.name, sha256=baseline.digest(path),
                            candidates=len(library), disconnected_ids=islands))
    freeze = dict(passed=True, new_solves=0, membership=records)
    (OUT / "preflight.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf8")
    print(json.dumps(freeze))

def candidate_run():
    receipts, policy = frozen_inputs()
    preflight = json.loads((HERE / "preflight.json").read_bytes())
    assert baseline.digest(OUT / "preflight.json") == baseline.digest(HERE / "preflight.json")
    results = []
    start = time.monotonic()
    attempts = 0
    with (OUT / "observations.jsonl").open("x", encoding="utf8") as log:
        for receipt, identity in zip(receipts, preflight["membership"], strict=True):
            case, net, lookup, active, library = imported(receipt)
            path = OUT / identity["paths"]
            assert baseline.digest(path) == identity["sha256"]
            assert json.loads(path.read_bytes()) == json.loads(json.dumps(library))
            assert identity["case"] == receipt["case"]
            assert connected(net, lookup, active)
            feedback = []
            for index, (station, terminals) in enumerate(library):
                if time.monotonic() - start >= WALL:
                    raise TimeoutError("GRID342 wall cap")
                trial, continuous = split(net, lookup, active, station, terminals)
                assert continuous == (index not in identity["disconnected_ids"])
                record = dict(pool_id=index, station=station, attempted=False, converged=False,
                              strict_admissible=False, rho=None, reward=0.0)
                if continuous:
                    if attempts >= LIMIT:
                        raise TimeoutError("GRID342 solve cap")
                    attempts += 1
                    record["attempted"] = True
                    try:
                        baseline.load_native(trial)
                    except LoadflowNotConverged:
                        record["reason"] = "nonconverged"
                    else:
                        record["converged"] = True
                        values = baseline.rating_readout(case, trial, lookup, active)
                        record.update(values)
                        rho = values.get("rho")
                        admissible = values["voltage_ok"] and rho is not None and 0 < rho < min(1.0, receipt["rho"])
                        if admissible:
                            reason = "admissible"
                        elif not values["voltage_ok"]:
                            reason = "voltage_or_nonfinite"
                        else:
                            reason = "thermal_or_no_improvement"
                        record.update(strict_admissible=admissible, reason=reason,
                                      reward=0.0 if rho is None else 2-rho)
                else:
                    record["reason"] = "disconnected"
                feedback.append(record)
                log.write(json.dumps(dict(case=receipt["case"], **record), allow_nan=False) + "\n")
                log.flush()
            arms = previous.summarize(policy, library, feedback)
            voltage_ok = [record for record in feedback if record.get("voltage_ok")]
            best = min(voltage_ok, key=lambda row: (row["rho"], row["pool_id"]), default=None)
            row = dict(case=receipt["case"], baseline_rho=receipt["rho"], candidates=len(library),
                       statuses=dict(Counter(record["reason"] for record in feedback)),
                       best_voltage_qualified=best, arms=arms)
            results.append(row)
            print(json.dumps({**row, "arms": {name: {k:v for k,v in a.items() if k != "visited"}
                                             for name, a in arms.items()}}), flush=True)
    output = dict(candidate_solves=attempts, baseline_solves=0, fits=0,
                  seconds=time.monotonic()-start, results=results)
    (OUT / "result.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf8")

if __name__ == "__main__":
    commands = {"preflight": qualification, "run": candidate_run}
    commands[sys.argv[1]]()
