"""Check stored graphs against physical vector fields without reusing converter."""
import hashlib
import json
import os
import sys
from pathlib import Path
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ[k] = '1'
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid40'
sys.path.insert(0, str(ROOT / 'work/grid08'))
from common import np, make_env
def read(p): return json.loads(p.read_text(encoding='utf-8'))
env = make_env(); env.seed(0); env.reset(); records = read(OUT / 'inputs_manifest.json')['records']
max_error = 0.; graphs = set()
for r in records:
    source = ROOT / r['source']
    with np.load(source.with_name(source.stem + '_state.npz')) as z: obs = env.observation_space.from_vect(z['observation'])
    with np.load(OUT / 'inputs' / f"{r['id']}.npz") as z: x = z['x'].copy(); edges = z['edges'].copy()
    ids = np.concatenate([obs.get_obj_substations(substation_id=s)[:, 1:] for s in range(obs.n_sub)])
    lm = ids[:, 3].copy(); lm[ids[:, 2] != -1] = ids[ids[:, 2] != -1, 2]
    expected = {
        0: obs.topo_vect, 2: np.append(obs.load_p, 0.)[ids[:, 0]],
        5: np.append(obs.gen_p, 0.)[ids[:, 1]],
        10: np.append(obs.p_or, 0.)[ids[:, 2]], 14: np.append(obs.p_ex, 0.)[ids[:, 3]],
        18: np.append(obs.timestep_overflow, 0.)[lm], 19: np.append(obs.line_status, 0.)[lm],
        20: np.append(obs.rho, 0.)[lm], 23: np.append(obs.time_before_cooldown_line, 0.)[lm]}
    for col, values in expected.items():
        error = float(np.max(np.abs(x[:, col] - np.asarray(values, dtype=np.float32))))
        assert error == 0., (r['id'], col, error); max_error = max(max_error, error)
    pairs = set(); start = 0
    for count in obs.sub_info:
        positions = range(start, start + int(count))
        for i in positions:
            for j in positions:
                if obs.topo_vect[i] > 0 and obs.topo_vect[i] == obs.topo_vect[j]: pairs.add((i, j))
        start += int(count)
    for line in np.flatnonzero(obs.line_status):
        a, b = int(obs.line_or_pos_topo_vect[line]), int(obs.line_ex_pos_topo_vect[line])
        pairs.add((a, b)); pairs.add((b, a))
    actual = set(zip(edges[0].tolist(), edges[1].tolist()))
    assert pairs == actual and len(actual) == edges.shape[1], r['id']
    graphs.add(hashlib.sha256(np.ascontiguousarray(edges).tobytes()).hexdigest())
env.close()
R = {'passed': True, 'states': len(records), 'distinct_topology_graphs': len(graphs),
     'independent_edge_sets': len(records), 'direct_physical_columns_checked': 9, 'max_physical_feature_error': max_error,
     'new_forecasts': 0, 'new_refits': 0, 'reserved_evaluations': 0,
     'warning': 'Stored features versus fresh from-vector physical fields and independent edge construction; no shared official converter in validation.'}
(OUT / 'graph_input_audit.json').write_text(json.dumps(R, indent=2), encoding='utf-8')
print(json.dumps(R, indent=2))
