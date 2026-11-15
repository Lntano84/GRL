"""One AC qualification per pinned PGLib case, without candidate evaluations."""
from collections import Counter
from hashlib import sha256
from importlib import util
from importlib.metadata import version
import json
from pathlib import Path
import re
import time
import warnings

import numpy as np
import pandapower as pp
from pandapower.auxiliary import LoadflowNotConverged
from pandapower.converter.pypower import from_ppc
from pandapower.pypower.idx_brch import BR_STATUS, RATE_A, F_BUS, T_BUS
from pandapower.pypower.idx_bus import BUS_I, PD

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[5] / "outputs/grid341_pglib_baseline"
PREVIOUS = HERE.parent / "GRID340_pandapower_transfer/pilot.py"
COLS = {
    "line": (("from_bus", "to_bus"), ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar")),
    "impedance": (("from_bus", "to_bus"), ("p_from_mw", "q_from_mvar", "p_to_mw", "q_to_mvar")),
    "trafo": (("hv_bus", "lv_bus"), ("p_hv_mw", "q_hv_mvar", "p_lv_mw", "q_lv_mvar")),
}
DIMENSIONS = {"bus": 13, "gen": 10, "branch": 13, "gencost": 7}
NUMERIC = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def parse(path):
    """Read the four numeric tables; no MATLAB evaluation."""
    text = "\n".join(line.split("%", 1)[0] for line in path.read_text(encoding="utf8").splitlines())
    ver, = re.findall(r"mpc\.version\s*=\s*'(\d+)'\s*;", text)
    base, = re.findall(r"mpc\.baseMVA\s*=\s*([0-9.]+)\s*;", text)
    assert ver == "2" and float(base) > 0
    case = {"version": ver, "baseMVA": float(base)}
    for name, columns in DIMENSIONS.items():
        block, = re.findall(rf"mpc\.{name}\s*=\s*\[([^\]]*)\]\s*;", text)
        rows = [row.split() for row in block.split(";") if row.strip()]
        assert rows and all(len(row) == columns and all(NUMERIC.fullmatch(v) for v in row) for row in rows), name
        matrix = np.asarray(rows, dtype=np.float64)
        assert np.isfinite(matrix).all()
        case[name] = matrix
    assert len(case["gencost"]) == len(case["gen"])
    return case

def check_import(case, net):
    branch = case["branch"]
    lookup = net._from_ppc_lookups["branch"]
    if len(lookup) != len(branch) or len(net.bus) != len(case["bus"]):
        raise ValueError("Imported topology count differs")
    assert set(net.bus.index) == set(case["bus"][:, BUS_I].astype(int))
    assert set(lookup["element_type"]) <= COLS.keys()
    assert not lookup.duplicated(["element_type", "element"]).any()
    active = branch[:, BR_STATUS] > 0
    assert np.all(branch[active, RATE_A] > 0)
    for kind, (sides, _) in COLS.items():
        subset = lookup["element_type"] == kind
        rows = net[kind].loc[lookup.loc[subset, "element"].astype(int)]
        got = rows.loc[:, list(sides)].to_numpy()
        intended = branch[subset][:, [F_BUS,T_BUS]]
        assert np.all(np.all(got == intended, axis=1) | np.all(got[:, ::-1] == intended, axis=1))
        assert np.array_equal(rows.in_service.to_numpy(), active[subset])
    return lookup, active

def load_native(net):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pp.runpp(net, algorithm="nr", init="flat", calculate_voltage_angles=True,
                 max_iteration=20, tolerance_mva=1e-8, numba=False, lightsim2grid=False)

def rating_readout(case, net, lookup, active):
    ratios = np.full(len(lookup), np.nan)
    for kind, (_, result_columns) in COLS.items():
        subset = (lookup["element_type"] == kind).to_numpy() & active
        flow = net[f"res_{kind}"].loc[lookup.loc[subset, "element"].astype(int), list(result_columns)].to_numpy()
        ratios[subset] = np.maximum(np.hypot(flow[:, 0], flow[:, 1]), np.hypot(flow[:, 2], flow[:, 3])) / case["branch"][subset, RATE_A]
    voltage = net.res_bus.loc[net.bus.in_service, "vm_pu"].to_numpy()
    if not np.isfinite(ratios[active]).all() or not np.isfinite(voltage).all():
        return dict(finite=False, voltage_ok=False, thermal_pressure=False)
    rho = float(np.max(ratios[active]))
    min_bus, max_bus = float(voltage.min()), float(voltage.max())
    return dict(finite=True, rho=rho, min_v=min_bus, max_v=max_bus,
                voltage_ok=bool(.9 <= min_bus and max_bus <= 1.1), thermal_pressure=bool(rho >= .9),
                max_branch_index=int(np.nanargmax(ratios)), branches_at_or_above_09=int((ratios >= .9).sum()))

def main():
    freeze = json.loads((HERE / "source_freeze.json").read_bytes())
    assert version("pandapower") == freeze["pandapower"]
    assert digest(PREVIOUS) == freeze["grid340_pilot_sha256"]
    spec = util.spec_from_file_location("fixed_grid340", PREVIOUS)
    previous = util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    assert len(freeze["inputs"]) == 4
    start = time.monotonic()
    rows = []
    with (OUT / "baseline_rows.jsonl").open("x", encoding="utf8") as log:
        attempts = 0
        for entry in freeze["inputs"]:
            path = OUT / "inputs" / entry["filename"]
            assert digest(path) == entry["sha256"]
            case = parse(path)
            net = from_ppc(case, f_hz=60)
            lookup, active = check_import(case, net)
            library = previous.candidates(net)
            record = dict(case=entry["filename"], sha256=entry["sha256"], bus_count=len(net.bus),
                          active_branch_count=int(active.sum()), branch_kinds=dict(Counter(lookup["element_type"])),
                          total_source_demand_mw=float(case["bus"][:, PD].sum()),
                          candidates=len(library), stations=len({bus for bus, _ in library}),
                          converged=False, candidate_solves=0, qualified=False)
            assert len(net.bus) == entry["bus_count"]
            if time.monotonic() - start >= 120:
                raise TimeoutError("GRID341 source qualification deadline")
            attempts += 1
            try:
                load_native(net)
            except LoadflowNotConverged:
                record["reason"] = "base_nonconverged"
            else:
                record["converged"] = True
                record.update(rating_readout(case, net, lookup, active))
                record["qualified"] = record["voltage_ok"] and record["thermal_pressure"] and len(library) > 256
                record["reason"] = "basis_available" if record["qualified"] else "fixed_gate_not_met"
            rows.append(record)
            line = json.dumps(record, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)
    result = dict(baseline_attempts=attempts, candidate_solves=0, pglib_release=freeze["pglib_release"],
                  qualified=sum(row["qualified"] for row in rows), rows=rows)
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")

if __name__ == "__main__":
    main()
