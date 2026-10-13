"""Validate new repetition against both saved repetitions, then reuse exact GRID55 statistics."""
import copy
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid55b'
OLD = ROOT / 'outputs/grid55'
read = lambda p: json.loads(p.read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p, x: p.write_text(json.dumps(x, indent=2, allow_nan=False), encoding='utf-8')


def rows(p):
    with gzip.open(p, 'rt', encoding='utf-8') as f:
        return [json.loads(s) for s in f]


def main():
    F = read(OUT / 'finished.json'); D = read(OUT / 'design.json')
    assert F['passed_engineering'] and len(F['summaries']) == 8
    for p, h in read(OUT / 'code_freeze.json').items():
        assert sha(ROOT / p) == h, p
    # System power evidence, not a wall/CPU replacement score.
    command = "$s=[datetime]::Parse('" + F['started_utc'] + "').ToLocalTime(); $e=[datetime]::Parse('" + F['ended_utc'] + "').ToLocalTime(); $a=@(Get-WinEvent -FilterHashtable @{LogName='System';ProviderName='Microsoft-Windows-Kernel-Power';Id=42,107,506,507;StartTime=$s;EndTime=$e} -ErrorAction SilentlyContinue | ForEach-Object { @{time_utc=$_.TimeCreated.ToUniversalTime().ToString('o');event_id=$_.Id;message=$_.Message} }); ConvertTo-Json -InputObject $a -Depth 4"
    proc = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
        capture_output=True, encoding='utf-8', errors='replace')
    # UTF8 JSON extraction is robust even if localized message bytes are replaced;
    # event IDs and ISO timestamps are the qualification fields.
    assert proc.returncode == 0 or not proc.stdout.strip(), proc.stderr
    events = json.loads(proc.stdout or '[]')
    save(OUT / 'windows_power_events.json', events)
    # Also verify System log can be read; empty filtered result alone is insufficient.
    probe = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
        "(Get-WinEvent -LogName System -MaxEvents 1 -ErrorAction Stop).RecordId"], capture_output=True, text=True)
    assert probe.returncode == 0 and probe.stdout.strip().isdigit(), probe.stderr
    save(OUT / 'power_log_access.json', {'readable': True, 'latest_record_id': int(probe.stdout.strip())})
    assert not events, 'Power event during supplemental run; do not choose timing view.'
    assert read(OUT / 'awake_lease_release.json')['returned'] != 0
    comparisons = []
    fields = ['step', 'before_hash', 'action_hash', 'after_hash', 'raw_cost', 'done', 'complete',
              'simulations', 'illegal', 'ambiguous', 'exceptions']
    call_fields = ['step', 'state_hash', 'input_hash', 'prior_hash', 'returned_action_hash',
                   'representatives', 'aliases', 'chosen_pool_id', 'public_queries', 'order', 'shortlist', 'masked_ids']
    outcome_fields = ['pool_id', 'rho', 'reward', 'done', 'illegal', 'ambiguous', 'exceptions',
                      'strict_admissible', 'phase', 'combined_hash']
    for s in F['summaries']:
        folder = OUT / 'runs' / f"r2__{s['week']}__{s['method']}"
        a = rows(folder / 'steps.jsonl.gz')
        for rep in [0, 1]:
            other = OLD / 'runs' / f"r{rep}__{s['week']}__{s['method']}"
            b = rows(other / 'steps.jsonl.gz')
            assert len(a) == len(b)
            assert [[r[k] for k in fields] for r in a] == [[r[k] for k in fields] for r in b]
            with np.load(folder / 'vectors.npz') as x, np.load(other / 'vectors.npz') as y:
                for name in ['action', 'observation']:
                    assert np.array_equal(x[name], y[name], equal_nan=True), (s['method'], name)
            ac = sorted(folder.glob('call[0-9][0-9][0-9].json'))
            bc = sorted(other.glob('call[0-9][0-9][0-9].json'))
            assert len(ac) == len(bc) == s['n1_calls']
            for p, q in zip(ac, bc):
                x, y = read(p), read(q)
                assert {k: x[k] for k in call_fields} == {k: y[k] for k in call_fields}, (p, q)
                assert [[r[k] for k in outcome_fields] for r in x['outcomes']] == [[r[k] for k in outcome_fields] for r in y['outcomes']]
            comparisons.append({'method': s['method'], 'historical_repeat': rep,
                'steps_vectors_cost_feedback_queries_exact': True})
    save(OUT / 'supplement_audit.json', {'passed': True, 'new_runs': 8, 'comparisons': comparisons,
         'power_events': 0, 'raw_hashes_preserved': True, 'no_independent_ac_resolve': True})
    view = OUT / 'qualified_view'
    assert not view.exists(), 'Qualified report already exists; do not overwrite.'
    view.mkdir()
    d = read(OLD / 'design.json'); source = read(OLD / 'finished.json')
    items = []
    run_paths = {}; provenance = []
    new = {s['method']: s for s in F['summaries']}
    for s in source['summaries']:
        replace = s['week'] == D['weeks'][0] and s['repeat'] == 1
        actual = new[s['method']] if replace else s
        path = (OUT if replace else OLD) / 'runs' / f"r{actual['repeat']}__{actual['week']}__{actual['method']}"
        items.append({**actual, 'repeat': s['repeat']})
        key = f"{s['repeat']}|{s['week']}|{s['method']}"
        run_paths[key] = str(path)
        provenance.append({'logical_slot': key, 'physical_repeat': actual['repeat'],
            'source_folder': path.relative_to(ROOT).as_posix(), 'summary_sha256': sha(path / 'summary.json'),
            'new_supplement': replace})
    assert sum(p['new_supplement'] for p in provenance) == 8
    d['run_paths'] = run_paths
    save(view / 'design.json', d)
    save(view / 'finished.json', {**source, 'summaries': items})
    save(view / 'provenance.json', provenance)
    save(view / 'code_freeze.json', read(OUT / 'code_freeze.json'))
    # Statistics and selection rule are the original frozen code; only storage
    # resolution is changed. Logical slot1 references physical repeat2 for ALL8.
    code = (ROOT / 'work/grid55/audit.py').read_text(encoding='utf-8')
    old_out = "OUT=ROOT/'outputs/grid55'"
    old_path = "folder=OUT/'runs'/f'r{ident[0]}__{ident[1]}__{ident[2]}'"
    assert code.count(old_out) == code.count(old_path) == 1
    code = code.replace(old_out, "OUT=ROOT/'outputs/grid55b/qualified_view'")
    code = code.replace(old_path, "folder=Path(D['run_paths'][f'{ident[0]}|{ident[1]}|{ident[2]}'])")
    namespace = {'__file__': str(ROOT / 'work/grid55/audit.py'), '__name__': 'grid55b_saved_audit'}
    exec(compile(code, str(ROOT / 'work/grid55/audit.py'), 'exec'), namespace)
    namespace['main']()
    report = read(view / 'review.json')
    assert report['passed_saved_data'] and all(r['same_path'] for r in report['repeat_checks'])
    text = (view / 'GRID55_report.md').read_text(encoding='utf-8')
    prefix = '# GRID55B 待机修复后的同日期版本比较\n\n原128条与RAW报告保持不变。本表使用七日期原重复0/1、受影响日期原重复0/新重复2，对全部八方法同样替换计时槽位。新八条与原两次路径、成本、合法反馈和查询完全一致，无运行期电源事件。不是新独立确认。\n\n'
    (OUT / 'GRID55B_report.md').write_text(prefix + text + '\n原始受待机影响表：outputs/grid55/GRID55_report.md；逐槽位来源：qualified_view/provenance.json。公共先验/DC/安全QP缓存提供给适用对照，不能归为GNN独有。\n', encoding='utf-8')
    save(OUT / 'review.json', {'passed': True, 'new_runs': 8, 'qualified_view': 'qualified_view/review.json',
        'raw_preserved': True, 'power_events': 0, 'physical_path_comparisons': 16,
        'version_selection': report['version_selection'], 'own_seed_average': report['own_seed_average']})
    for p, h in read(OUT / 'raw_preservation.json').items():
        assert sha(ROOT / p) == h
    (OUT / 'RUN_STATE.md').write_text('COMPLETE AND AUDITED: eight supplemental runs and qualified128 timing view. Original raw128 immutable.\n', encoding='utf-8')


if __name__ == '__main__':
    main()
