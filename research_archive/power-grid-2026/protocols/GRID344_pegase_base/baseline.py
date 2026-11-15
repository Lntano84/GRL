"""One power-flow qualification on the original, pinned PEGASE benchmark point."""
from hashlib import sha256
from importlib import util
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
OUT = ROOT / "outputs/grid344_pegase_base"
SOURCE = ROOT / "outputs/grid343_public_operating_point/inputs/case1354pegase.m"
OLD = HERE.parent / "GRID341_pglib_baseline/baseline.py"

def digest(path):
    return sha256(path.read_bytes()).hexdigest()

def ppc():
    text = "\n".join(line.split("%", 1)[0] for line in SOURCE.read_text(encoding="utf8").splitlines())
    case = dict(version="2", baseMVA=100.0)
    for key, shape in {"bus":(1354,13), "gen":(260,21), "branch":(1991,13), "gencost":(260,7)}.items():
        body, = re.findall(rf"mpc\.{key}\s*=\s*\[([^\]]*)\]\s*;",text)
        mat = np.asarray([r.split() for r in body.split(";") if r.strip()], dtype=np.float64)
        assert mat.shape == shape
        case[key] = mat
    return case

def imported():
    frozen = json.loads((HERE / "source_freeze.json").read_bytes())
    assert digest(SOURCE) == frozen["input_sha256"] and digest(OLD) == frozen["adapter_sha256"]
    assert version("pandapower") == frozen["pandapower"]
    spec = util.spec_from_file_location("base341", OLD)
    old = util.module_from_spec(spec)
    spec.loader.exec_module(old)
    prev_spec = util.spec_from_file_location("pilot340", old.PREVIOUS)
    prev = util.module_from_spec(prev_spec)
    prev_spec.loader.exec_module(prev)
    assert digest(old.PREVIOUS) == frozen["candidates_sha256"]
    case = ppc()
    assert all(np.isfinite(case[key]).all() for key in ("bus", "branch","gencost"))
    q = case["gen"][:,3:5]
    assert q.shape == (260,2) and int((~np.isfinite(q)).sum()) == 4
    assert not np.isnan(q).any()
    original = {k: v.copy() for k,v in case.items() if isinstance(v, np.ndarray)}
    net = old.from_ppc(case, f_hz=50)
    assert all(np.array_equal(case[k], original[k]) for k in original)
    assert set(net.bus.index) == set(case["bus"][:, old.BUS_I].astype(int))
    lookup = net._from_ppc_lookups["branch"]
    source = case["branch"]
    active = source[:, old.BR_STATUS] > 0
    rated = active & (source[:,old.RATE_A] > 0)
    assert len(lookup) == len(source) == int(active.sum()) == 1991
    assert int(rated.sum()) == 1432 and int((active & ~rated).sum()) == 559
    assert (source[:, old.RATE_A] >= 0).all()
    assert np.array_equal(lookup.index.to_numpy(), np.arange(len(lookup)))
    assert set(lookup.element_type) <= old.COLS.keys()
    assert not lookup.duplicated(["element_type", "element"]).any()
    for kind, (sides, _) in old.COLS.items():
        subset = lookup["element_type"] == kind
        rows = net[kind].loc[lookup.loc[subset,"element"].astype(int)]
        ends = rows.loc[:,list(sides)].to_numpy()
        expected = source[subset][:,[old.F_BUS,old.T_BUS]]
        assert (np.all(ends == expected,axis=1) | np.all(ends[:,::-1] == expected,axis=1)).all()
        assert np.array_equal(rows.in_service.to_numpy(),active[subset])
    library = prev.candidates(net)
    structure = dict(buses=len(net.bus), branches=len(lookup), rated=int(rated.sum()),
                     unlimited=int((active & ~rated).sum()), candidates=len(library),
                     candidate_stations=len({station for station,_ in library}), ac_attempts=0)
    return old, case, net, lookup, active, rated, structure

def write(path, obj):
    with path.open("x",encoding="utf8") as stream:
        json.dump(obj,stream,indent=2,allow_nan=False)

def preflight():
    *_, structure = imported()
    OUT.mkdir(exist_ok=False)
    write(OUT / "preflight.json",structure)
    print(structure)

def run():
    old, case, net, lookup, active, rated, record = imported()
    assert record == json.loads((HERE / "preflight.json").read_bytes())
    write(OUT / "started.json", dict(ac_attempt_cap=1, source=SOURCE.name))
    start = time.monotonic()
    record.update(ac_attempts=1,candidate_ac=0,opf_calls=0,converged=False,qualified=False)
    try:
        old.load_native(net)
    except old.LoadflowNotConverged:
        record["reason"] = "base_nonconverged"
    else:
        record["converged"] = True
        # All active branches remain in AC. Only thermal ratios use the rated mask.
        record.update(old.rating_readout(case,net,lookup,rated))
        if record["finite"]:
            voltage = net.res_bus.loc[net.bus.in_service,"vm_pu"]
            record.update(above_1p10=int((voltage > 1.10).sum()),below_0p90=int((voltage < .90).sum()))
        record["qualified"] = record["voltage_ok"] and record["thermal_pressure"] and record["candidates"] > 256
        record["reason"] = "gate_passed" if record["qualified"] else "fixed_gate_not_met"
    record["wall_s"] = time.monotonic()-start
    write(OUT / "result.json",record)
    print(json.dumps(record))

if __name__ == "__main__":
    {"preflight":preflight,"run":run}[sys.argv[1]]()
