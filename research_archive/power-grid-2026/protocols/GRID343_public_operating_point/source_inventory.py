"""Read metadata from the pinned PEGASE benchmark without executing MATLAB or AC."""
from collections import Counter
from hashlib import sha256
from math import comb
import json
from pathlib import Path
import re

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[5] / "outputs/grid343_public_operating_point"
SOURCE = OUT / "inputs/case1354pegase.m"
DIGEST = "1b08b25a2f6c1d540d090009dfaff41ff2b05784a2d8d302a7ad695821557b89"

def inspect_source():
    data = SOURCE.read_bytes()
    assert sha256(data).hexdigest() == DIGEST
    text = "\n".join(line.split("%", 1)[0] for line in data.decode("utf8").splitlines())
    assert re.findall(r"mpc\.version\s*=\s*'(\d+)'\s*;", text) == ["2"]
    assert re.findall(r"mpc\.baseMVA\s*=\s*(\d+)\s*;", text) == ["100"]

    def read_table(key, shape):
        block, = re.findall(rf"mpc\.{key}\s*=\s*\[([^\]]*)\]\s*;", text)
        rows = [row.split() for row in block.split(";") if row.strip()]
        table = np.asarray(rows, dtype=np.float64)
        assert table.shape == shape
        return table

    bus = read_table("bus", (1354, 13))
    gen = read_table("gen", (260, 21))
    branch = read_table("branch", (1991, 13))
    cost = read_table("gencost", (260, 7))
    assert all(np.isfinite(table).all() for table in (bus, branch, cost))
    mask = ~np.isfinite(gen)
    assert not mask[:, [0, 1, 2, *range(5, 21)]].any()
    assert np.isposinf(gen[mask[:, 3], 3]).all() and np.isneginf(gen[mask[:, 4], 4]).all()
    assert np.array_equal(mask[:, 3], mask[:, 4]) and int(mask.sum()) == 4
    assert set(branch[:, :2].ravel()) <= set(bus[:, 0])
    assert len(set(bus[:, 0])) == len(bus)
    active = branch[branch[:, 10] > 0]
    assert (active[:, 5] >= 0).all()
    rated = active[active[:, 5] > 0, 5]
    degree = Counter([*active[:, 0], *active[:, 1]])
    bound = sum(sum(comb(n-1, k) for k in (2, 3) if n-k >= 2) for n in degree.values())
    return dict(source_sha256=DIGEST, buses=len(bus), generators=len(gen),
                branches=len(branch), active_branches=len(active),
                rated_active_branches=len(rated), unlimited_active_branches=len(active)-len(rated),
                rate_a_mva_interval=[float(rated.min()),
                      float(rated.max())], source_vm_range=[float(bus[:, 7].min()),
                      float(bus[:, 7].max())], source_bus_types={int(k):v for k,v in Counter(bus[:, 1]).items()},
                source_generator_p_mw=float(gen[gen[:, 7] > 0, 1].sum()),
                source_load_mw=float(bus[:, 2].sum()), no_q_limit_generator_rows=int(mask[:, 3].sum()),
                branch_degree_candidate_upper_bound=bound, baseline_solves=0, candidate_solves=0)

if __name__ == "__main__":
    result = inspect_source()
    path = OUT / "source_inventory.json"
    with path.open("x", encoding="utf8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2, allow_nan=False))
