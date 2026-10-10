"""Independent stored-graph/ledger review; no power flow or training."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid11_failure_context'
def read(path):
    return json.loads((OUT / path).read_text())

context = read('results.json')
bridge = read('bridge_check/results.json')
probe = read('emergency_probe/results.json')
failed = read('emergency_probe_invalid_loop_01/failure.json')
for folder in ['', 'bridge_check', 'emergency_probe']:
    for path, expected in read(str(Path(folder) / 'code_freeze.json')).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
for path, expected in read('design.json')['source_hashes'].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path

# Union-find is independent of the diagnostic's depth-first traversal.
vertices = set(x for row in bridge['components_before'] for x in row['nodes'])
def partition(excluded):
    parents = {x: x for x in vertices}
    def root(x):
        while x != parents[x]:
            x = parents[x]
        return x
    for edge in bridge['public_line_endpoints']:
        if edge['line'] == excluded:
            continue
        a, b = root(edge['u']), root(edge['v'])
        parents[b] = a
    groups = {}
    for x in vertices:
        groups.setdefault(root(x), []).append(x)
    return sorted(sorted(group) for group in groups.values())
assert partition(None) == sorted(sorted(c['nodes']) for c in bridge['components_before'])
assert partition(11) == sorted(sorted(c['nodes']) for c in bridge['components_without_line_11'])
assert len(partition(None)) == 1 and len(partition(11)) == 2
last = context['contexts'][-1]
trips = [i for i, cascade in enumerate(last['cascade_disconnections']) if cascade >= 0]
assert trips == [11]
assert last['before']['current_step'] == bridge['current_step'] == 1718
assert all(not r['before']['maintenance_within_hour'] for r in context['contexts'])
assert all(r['done'] and r['exceptions'] for r in bridge['forecasts'])
assert sum(probe['pool_sizes'].values()) == 1330 and probe['unique_pool_size'] == 1024
for row in probe['records']:
    details = read(f"emergency_probe/state_{row['state_step']}_candidates.json")
    assert len(details) == row['unique_candidates'] == row['native_highres_count']
    assert len({x['action_hash'] for x in details}) == len(details)
    assert sum(x['valid'] for x in details) == row['valid']
    assert sum(x['valid'] and x['max_rho'] < 1 for x in details) == row['valid_rho_lt_1']
    for candidate in details:
        valid = not candidate['done'] and not candidate['illegal'] and not candidate['ambiguous'] and not candidate['exceptions'] and candidate['max_rho'] > 0
        assert candidate['valid'] == valid
    viable = [x for x in details if x['valid']]
    best = min(viable, key=lambda x: (x['max_rho'], x['cost'])) if viable else None
    assert row['best'] == best
assert probe['forecasts'] == sum(x['unique_candidates'] for x in probe['records'])
resources = {
    'physical_steps_including_failed_attempt': context['physical_steps'] + bridge['physical_steps'] + failed['physical_steps'] + probe['physical_steps'],
    'forecast_calls_including_failed_attempt': context['forecast_calls'] + bridge['native_highres_count'] + failed['forecasts'] + probe['forecasts'],
    'measured_simulator_phase_wall_s': context['wall_s'] + bridge['wall_s'] + failed['wall_s'] + probe['wall_s'],
    'failed_loop_attempt_preserved': True, 'training_runs': 0,
}
result = {'passed': True, 'graph_crosscheck': True, 'trip_crosscheck': True,
          'candidate_records_recounted': probe['forecasts'], 'resources': resources,
          'scope': 'Stored graph connectivity and scalar recount; not independent power-flow re-simulation or proof about the full action space.'}
(OUT / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
lines = [
    '# GRID11：一月训练故障诊断', '',
    '这是对已见训练失败的条件诊断，不是封存测试，也没有新增学习或实际未来分支。', '',
    '冻结动作重放的 1,719 步观测哈希全部匹配。第 1,719 步之前，线路 11 的负载率为 '
    '1.014742，连续过载计数为 3；下一步反馈记录该线路切除并出现断网潮流错误。'
    '最后 25 步的公开观测未显示一小时内计划维护，不能据此排除所有其他扰动或早期决策影响。', '',
    '独立并查集复核：当前母线图连通，删除线路 11 后分成两部分。小岛包含发电机 40'
    '（当前输出 0 MW）及负荷 60、98（合计约 68.9 MW）。这是图连通性证据；'
    '不是 AC 可行性或策略因果证明。终止后清零的观测不能被解释为全部线路真实跳闸。', '',
    '| 公共状态步 | 当前最大负载率 | 去重后拓扑候选 | 一步合法非终止 | 一步最大负载率 < 1 |',
    '|---|---:|---:|---:|---:|',
]
for row in probe['records']:
    lines.append(f"| {row['state_step']} | {row['current_rho']:.6f} | {row['unique_candidates']} | {row['valid']} | {row['valid_rho_lt_1']} |")
lines += [
    '', '在 1,714 步状态，最好的已测拓扑候选预测最大负载率仍为 1.022310；'
    '在 1,718 步状态，1,018 个候选全部预测终止。候选来自作者两个动作池的合并，'
    '并非全动作空间；该结果不能证明没有连续控制、组合拓扑或更早干预能够救回。', '',
    '当前二选一恢复策略在这两个状态都没有被提供动作：它仅在低于 0.9 的安全状态、'
    '规定的时间间隔内介入。因此，该故障不能单纯通过改善这个二选一策略的分类能力来补救；'
    '还需要检查更早控制、候选范围或统一安全模块。此前作者控制器在已有断线时已经搜索 N1 '
    '动作池，不能把这里的问题简化成 NN20 只看 20 个动作。', '',
    '第一次探针循环边界遗漏了 1,718 步诊断并触发断言；其脚本、日志和费用全部保留。'
    '仅修正循环覆盖后重跑，未改状态、动作池或筛选规则。', '',
    f"资源：含失败尝试共 {resources['physical_steps_including_failed_attempt']} 个物理步、"
    f"{resources['forecast_calls_including_failed_attempt']} 次公共预测；不计入 GRID10 的训练预算，独立列账。", '',
    '下一项小诊断：在相同训练前缀的较早风险状态，调用现成应急 QP 并用公共多步预测检查。'
    '不先训练新安全网络；预测可行也不等于真实整周控制成功。',
]
(OUT / 'GRID11_review_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('COMPLETE. Diagnostics and stored-data audit passed; no training or deployed-controller claim. Failed attempt retained and charged.\n', encoding='utf-8')
files = [p for directory in [OUT, ROOT / 'work/grid11'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json']
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(OUT / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({**result, 'sealed_files': len(manifest)}, indent=2))
