"""Saved-vector qualification and old paired-timing decomposition; no grid replay."""
import csv, gzip, hashlib, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'): os.environ[key] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid29b'


def load(path): return json.loads(path.read_text(encoding='utf-8'))
def rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as f: return [json.loads(line) for line in f]
def save(path, data): path.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


P = load(OUT/'profile.json')
assert P['passed']
for file, expected in load(OUT/'code_freeze.json').items():
    assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest() == expected, file
vectors = np.load(OUT/'dc_vectors.npz')['flows']
assert vectors.shape[0] == len(P['pairs']) and vectors.shape[1] == 3
maximum = 0.
for index, pair in enumerate(P['pairs']):
    for slot, name in ((1, 'ORIGINAL'), (2, 'CACHED')):
        reference, probe = vectors[index, 0], vectors[index, slot]
        assert np.array_equal(np.isfinite(reference), np.isfinite(probe))
        finite = np.isfinite(reference)
        error = float(np.max(np.abs(reference[finite]-probe[finite]))) if finite.any() else 0.
        assert error <= 1e-6 and error == pair['flow_error_MW'][name]
        maximum = max(maximum, error)
    assert pair['order'] == ([0, 1] if index % 2 == 0 else [1, 0])
run = OUT/'runs/r0__2035-03-05_0__LIVE_DC_PROFILE'
baseline = ROOT/'outputs/grid29/pilot/runs/r0__2035-03-05_0__ORIGINAL'
new_rows, old_rows = rows(run/'steps.jsonl.gz'), rows(baseline/'steps.jsonl.gz')
assert len(new_rows) == len(old_rows) == 2017
for a, b in zip(new_rows, old_rows):
    for key in ('step', 'before_hash', 'action_hash', 'after_hash', 'raw_cost', 'done', 'complete', 'simulations'):
        assert a[key] == b[key], (a['step'], key)
    assert not a['illegal'] and not a['ambiguous'] and not a['exceptions']
with np.load(run/'vectors.npz') as a, np.load(baseline/'vectors.npz') as b:
    for key in ('action', 'observation'): assert np.array_equal(a[key], b[key], equal_nan=True)
active = [p for p in P['pairs'] if p['in_control']]
timing = {k: math.fsum(p['shadow_s'][k] for p in active) for k in ('ORIGINAL', 'CACHED')}
for k in timing: assert abs(timing[k]-P['controlled_shadow_s'][k]) < 1e-7
groups = defaultdict(list)
for record in P['solver_records']:
    # Explicit solver invocations only: recursive solver=None contains children.
    if record['in_control'] and record['solver'] is not None:
        groups[(record['phase'], record['owner'], record['solver'])].append(record)
stats = []
for (phase, owner, solver), group in sorted(groups.items()):
    item = {'phase': phase, 'owner': owner, 'solver': solver, 'calls': len(group),
            'statuses': dict(Counter(r['status'] for r in group)),
            'wall_s': math.fsum(r['wall_s'] for r in group)}
    for source, dest in (('compilation_s', 'compile_s'), ('backend_s', 'backend_s'), ('iterations', 'iterations')):
        valid = [r[source] for r in group if r[source] is not None]
        item[dest] = math.fsum(valid) if valid else None
        item[dest+'_missing'] = len(group)-len(valid)
    stats.append(item)
audit = {'passed': True, 'saved_dc_triples': len(P['pairs']), 'in_control_dc_pairs': len(active),
         'max_flow_difference_MW': maximum, 'physical_rows': len(new_rows), 'same_delivery': True,
         'new_ac_replays': 0, 'new_fits': 0, 'reserved_evaluations': 0,
         'qualification': 'Saved vectors, hashes, physical ledger, explicit-solver grouping; no independent solve.'}
save(OUT/'audit.json', audit)

# Diagnose the original two pairs without reclassifying old results.
old_pairs = []
for repeat in range(2):
    paths = {method: ROOT/'outputs/grid29/pilot/runs'/f'r{repeat}__2035-03-05_0__{method}'
             for method in ('ORIGINAL', 'CACHED')}
    source = {m: rows(p/'steps.jsonl.gz') for m, p in paths.items()}
    summaries = {m: load(p/'summary.json') for m, p in paths.items()}
    changes = []
    by_restore = defaultdict(lambda: {'steps': 0, 'original_s': 0., 'cached_s': 0., 'saved_s': 0.})
    for a, b in zip(source['ORIGINAL'], source['CACHED']):
        assert a['action_hash'] == b['action_hash'] and a['restore'] == b['restore']
        change = {'repeat': repeat, 'step': a['step'], 'restore': a['restore'],
                  'original_s': a['decision_s'], 'cached_s': b['decision_s'],
                  'saved_s': a['decision_s']-b['decision_s'],
                  'forecast_saved_s': a['all_forecast_s']-b['all_forecast_s']}
        changes.append(change)
        dest = by_restore[str(a['restore'])]
        dest['steps'] += 1
        for key in ('original_s', 'cached_s', 'saved_s'): dest[key] += change[key]
    expected = summaries['ORIGINAL']['controller_s']-summaries['CACHED']['controller_s']
    assert abs(math.fsum(r['saved_s'] for r in changes)-expected) < 1e-6
    sorted_changes = sorted(changes, key=lambda r: -r['saved_s'])
    old_pairs.append({'repeat': repeat, 'saved_s': expected, 'by_restore': dict(by_restore),
                      'forecast_saved_s': math.fsum(r['forecast_saved_s'] for r in changes),
                      'non_forecast_saved_s': expected-math.fsum(r['forecast_saved_s'] for r in changes),
                      'top_saved_steps': sorted_changes[:12],
                      'top_regressed_steps': sorted_changes[-12:]})
    with (OUT/f'old_pair_{repeat}_steps.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(changes[0])); writer.writeheader(); writer.writerows(changes)
result = {'audit': audit, 'live_dc_shadow_s': timing,
          'live_dc_reduction': 1-timing['CACHED']/timing['ORIGINAL'],
          'solver_components': stats, 'old_pilot_pairs': old_pairs,
          'scope': 'Live-memory isolated DC pairs and saved old-step accounting. Instrumented whole-control time is not an acceleration score.',
          'causality_limit': 'Old pilot has per-step and forecast timing but no per-solver timing. Component shadow result cannot identify the old pairs individual DC savings.'}
save(OUT/'review.json', result)
lines = ['# GRID29B 实际状态配对诊断', '',
         f"保存数据核验通过：{len(P['pairs'])}组DC三向量，{len(active)}组发生于实际控制中；流量最大差{maximum:g}MW，2017步交付/成本/查询与原轨迹相同。", '',
         f"同状态/同实际存储记忆隔离探针：原DC {timing['ORIGINAL']:.3f}s，缓存DC {timing['CACHED']:.3f}s，减少{100*result['live_dc_reduction']:.2f}%。探针输出未派发；整个诊断回放含探针开销，不能作为整体提速。", '',
         '| 阶段 | 对象 | solver | 次数 | 调用时间 | 编译时间 | 后端时间 | 迭代 |',
         '|---|---|---|---:|---:|---:|---:|---:|']
for s in stats:
    lines.append(f"| {s['phase']} | {s['owner']} | {s['solver']} | {s['calls']} | {s['wall_s']:.3f} | {s['compile_s']} | {s['backend_s']} | {s['iterations']} |")
lines += ['', '缺失统计保持缺失，recursive solver=None调用不叠加到explicit子调用。', '',
          '旧两组是平衡顺序测量，10.93%和0.39%均保留。下面只分解已测总时间；restore=True是实际交付恢复动作的物理步标记，并不代表所有尝试恢复/DC调用。', '',
          '| 旧重复 | 整体节省 | 预测节省 | 非预测节省 | restore=True步节省 | 其余步节省 |',
          '|---|---:|---:|---:|---:|---:|']
for p in old_pairs:
    lines.append(f"| {p['repeat']} | {p['saved_s']:.3f} | {p['forecast_saved_s']:.3f} | {p['non_forecast_saved_s']:.3f} | {p['by_restore']['True']['saved_s']:.3f} | {p['by_restore']['False']['saved_s']:.3f} |")
lines += ['', result['causality_limit'], '',
          '下一步若需要解释幅度差异，应在完整原/缓存两臂中同时记录每次DC和连续QP的时间、编译/后端/迭代，保持实际动作完全一致，而不是把探针时间从整体时间机械相减。']
(OUT/'GRID29B_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND SAVED-DATA AUDITED: actual-memory DC pairs; diagnostic whole time is not a speed score.\n', encoding='utf-8')
print(json.dumps({'audit': audit, 'live_dc_shadow_s': timing, 'live_dc_reduction': result['live_dc_reduction'],
                  'solver_components': stats, 'old_pilot_pairs': [{k:v for k,v in p.items() if not k.startswith('top_')} for p in old_pairs]}, indent=2))
