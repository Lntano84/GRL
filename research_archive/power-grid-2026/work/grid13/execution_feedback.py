"""Replay the exact archived prefix to reconcile public forecast with feedback.

No counterfactual branch, no hidden future used for an action. This is retrospective
failure diagnosis of an already-executed training prefix.
"""
import gzip
import sys
import traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, np, json, hashlib, time, make_env, write_json, digest, flags
from trace_agent import action_vector

OUT = ROOT / 'outputs/grid13/feedback_replay'; assert not OUT.exists(); OUT.mkdir()
parent = json.loads((OUT.parent / 'design.json').read_text()); source_item = parent['source']
source = ROOT / source_item['path']
with gzip.open(source / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f: rows = list(map(json.loads, f))
with np.load(source / 'vectors.npz') as f: vectors = f['action'].copy()
write_json(OUT / 'design.json', {'source': source_item, 'physical_cap': 1718, 'wall_cap_s': 180,
    'scope': 'Exact already-observed training prefix only; extract native feedback after the archived reconnection. Not a policy input or a future counterfactual.'})
write_json(OUT / 'code_freeze.json', {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
start = time.perf_counter(); steps = 0; records = []; env = None
try:
    env = make_env(); env.seed(0); env.set_id(source_item['scenario']); obs = env.reset()
    for i in range(1718):
        assert time.perf_counter() - start < 180
        assert digest(obs.to_vect()) == rows[i]['before_hash']
        action = env.action_space(); action.from_vect(vectors[i])
        before = {'state': int(obs.current_step), 'rho': float(obs.rho.max()),
            'line_81_status': bool(obs.line_status[81]), 'line_81_cooldown': int(obs.time_before_cooldown_line[81])}
        obs, _, done, info = env.step(action); steps += 1
        assert not done and digest(obs.to_vect()) == rows[i]['after_hash']
        if i >= 1713:
            record = {'action_step': i + 1, 'before': before,
                'action_hash': digest(action_vector(action)),
                'requested_line_81_set_status': int(action.line_set_status[81]),
                'after': {'state': int(obs.current_step), 'rho': float(obs.rho.max()),
                    'line_81_status': bool(obs.line_status[81]), 'line_81_cooldown': int(obs.time_before_cooldown_line[81])},
                'feedback_keys': sorted(info), 'flags': flags(info),
                'opponent_attack_line': info.get('opponent_attack_line'),
                'opponent_attack_sub': info.get('opponent_attack_sub'),
                'opponent_attack_duration': info.get('opponent_attack_duration'),
                'cascade_disconnections': info.get('disc_lines')}
            records.append(record)
    write_json(OUT / 'results.json', {'passed': True, 'physical_steps': steps, 'forecast_steps': 0,
        'wall_s': time.perf_counter() - start, 'records': records, 'scope': 'Retrospective native feedback, kept outside decision features.'})
    print(json.dumps({'passed': True, 'physical_steps': steps, 'last_feedback': records[-1]}, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else int(x)), flush=True)
except Exception:
    write_json(OUT / 'failure.json', {'physical_steps': steps, 'wall_s': time.perf_counter() - start, 'traceback': traceback.format_exc()})
    raise
finally:
    if env is not None: env.close()
