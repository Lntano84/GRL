"""Stored-action/observation and retrospective attack-feedback reconciliation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, hashlib, make_env, write_json
from trace_agent import action_vector, digest

OUT = ROOT / 'outputs/grid13'
def read(path): return json.loads((OUT / path).read_text())
result = read('results.json'); feedback = read('feedback_replay/results.json'); design = read('design.json')
assert result['passed'] and feedback['passed']
for folder in ['', 'feedback_replay']:
    for path, expected in read(str(Path(folder) / 'code_freeze.json')).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
assert result['physical_steps'] == feedback['physical_steps'] == 1718
assert result['forecast_steps'] == sum(len(branch) for r in result['records'] for branch in r['branches'].values())
assert result['forecast_steps'] <= design['forecast_cap'] and result['wall_s'] < design['wall_cap_s']
last = feedback['records'][-1]
assert last['action_step'] == 1718 and last['before']['state'] == 1717
assert last['requested_line_81_set_status'] == 1
assert not any(last['flags'].values())
assert np.flatnonzero(last['opponent_attack_line']).tolist() == [81]
assert last['opponent_attack_duration'] == 1
assert not last['after']['line_81_status'] and last['after']['line_81_cooldown'] == 3

env = make_env(); env.seed(0); env.set_id(design['source']['scenario']); initial = env.reset()
checked = 0
try:
    with np.load(OUT / 'public_vectors.npz') as data:
        for record in result['records']:
            i = record['state']; obs = initial.copy(); obs.from_vect(data[f's{i}_public_observation'])
            assert float(obs.rho.max()) == record['rho']
            assert bool(obs.line_status[81]) == record['line_81_status']
            assert int(obs.time_before_cooldown_line[81]) == record['line_81_cooldown']
            for name, branch in record['branches'].items():
                a = env.action_space(); a.from_vect(data[f's{i}_{name}_action'])
                assert np.array_equal(action_vector(a), data[f's{i}_{name}_action'])
                if i == 1717 and name == 'saved_base':
                    assert digest(action_vector(a)) == last['action_hash']
                    assert int(a.line_set_status[81]) == 1
                for h, row in enumerate(branch, 1):
                    future = initial.copy(); future.from_vect(data[f's{i}_{name}_forecast{h}'])
                    assert row['horizon_step'] == h
                    assert float(future.rho.max()) == row['rho']
                    assert float(future.rho[11]) == row['line_11_rho']
                    assert bool(future.line_status[81]) == row['line_81_status']
                    if row['done']: assert h == len(branch)
                    checked += 1
finally:
    env.close()
first = result['records'][0]
assert first['state'] == 1717 and first['line_81_cooldown'] == 0
base = first['branches']['saved_base']
assert len(base) == 6 and all(not x['done'] and not x['illegal'] and not x['ambiguous'] and not x['exceptions'] for x in base)
assert all(x['line_81_status'] for x in base)
resources = {'physical_steps_including_feedback_replay': result['physical_steps'] + feedback['physical_steps'],
             'public_forecast_steps': result['forecast_steps'], 'training_runs': 0,
             'measured_simulator_phase_wall_s': result['wall_s'] + feedback['wall_s']}
audit = {'passed': True, 'public_forecast_vectors_recounted': checked, 'action_roundtrips': 12,
         'native_attack_reconciliation': True, 'resources': resources,
         'scope': 'Stored-vector decoding and native feedback reconciliation; no independent AC solve, new policy or training.'}
write_json(OUT / 'audit.json', audit)
lines = [
    '# GRID13：重连与真实扰动的区分', '',
    '**本轮避免了一项错误归因：原控制器确实重连了线路 81；真实执行受到该线路的新攻击。**', '',
    '在已见一月训练轨迹的状态 1,717，线路 81 冷却为 0。归档的原动作包含该线路重连；'
    f"公共预测第一步的最大负载率为 {base[0]['rho']:.6f}，线路 81 变为连通，"
    '随后的六步固定 no-op 预测分支均非终止。', '',
    '精确重放同一个归档动作及其真实前缀后，原生反馈记录 opponent_attack_line=[81]、'
    '攻击持续期 1；真实状态 1,718 的线路 81 仍断开，冷却重置为 3，最大负载率为 '
    f"{last['after']['rho']:.6f}。动作本身没有非法、歧义或异常标记。"
    '反馈与既有动作、观测哈希逐项一致，没有更换实际动作或新开真实反事实分支。', '',
    '攻击记录只用于事后诊断，未输入候选选择、GNN、奖励或新训练。'
    '不能把固定种子下事后知道的下一次攻击告诉在线控制器。公共预测与真实过程'
    '的区别不能被解读为神经预测误差：该公共预测是物理 forecast，并未提前提供这次未知攻击。', '',
    '| 状态 | 原动作公共预测 | 默认母线重连 | 四种显式母线组合 |',
    '|---|---|---|---|',
    '| 1,717，冷却 0 | 六步均非终止 | 六步均非终止 | 1/1 六步非终止；1/2、2/1 在第二步终止；2/2 在第一步终止 |',
    '| 1,718，冷却 3 | 下一步终止 | 非法重连、下一步终止 | 全部非法重连、下一步终止 |', '',
    '所以本轮并没有发现“显式指定母线能纠正作者遗漏重连”的收益。'
    'GRID11 最后状态的动作池无解与 GRID12 单次提前 QP 无效仍是各自成立的限定观察，'
    '但不能据此倒推原控制器在 1,717 步错过了一个已知安全重连动作。'
    '公共六步非终止也不表示真实整周一定安全。', '',
    '## 论文依据与后续角色', '',
    '[RL2Grid](https://arxiv.org/html/2503.23101v2)将 opponent 扰动与公开计划维护区分。'
    '[电网代理故障分析](https://arxiv.org/html/2406.16426v1)也指出，临近失败的稳定表象'
    '可能被突然攻击改变，并分析了具体关键线路及不同故障类型。'
    '这些工作支持先分类故障，不能把任何终止都归为训练不足。', '',
    '保留已经降低费用的连续恢复模块；当前 GNN 二选一任务尚无额外收益。'
    '随后应将可控的多步费用差与未预知扰动分别建模，优先研究候选动作的空间影响和'
    '危险拓扑的风险。若使用 N1/多步筛选，所有非学习对照必须有相同信息和计算预算。'
    '本轮尚未证明这种信息能改善控制，也没有新算法贡献；不直接启动故障分类或风险网络训练。', '',
    'GRID07 已经留下了一次立即较便宜、续接却更昂贵的拓扑干预见证。'
    '在保留恢复模块后，该空间是否仍存在要重新检查，不能直接沿用恢复前的差距。', '',
    f"资源：含真实前缀反馈重放共 {resources['physical_steps_including_feedback_replay']} 个物理步，"
    f"{resources['public_forecast_steps']} 个公共预测步；没有模型拟合。"
    '独立列账；保存了全部候选动作、预测状态、真实反馈和失败边界。',
]
(OUT / 'GRID13_review_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('COMPLETE. Public reconnection candidates and exact archived-feedback replay audited; one new opponent attack identified retrospectively. No actual counterfactual, policy fit or deployment claim.\n', encoding='utf-8')
verified = {}
for name in ['grid08', 'grid09', 'grid10', 'grid11_failure_context', 'grid12']:
    manifest = json.loads((ROOT / 'outputs' / name / 'delivery_manifest.json').read_text())
    for path, expected in manifest.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    verified[name] = len(manifest)
write_json(OUT / 'previous_deliveries_verified.json', verified)
files = [p for directory in [OUT, ROOT / 'work/grid13'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json']
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
write_json(OUT / 'delivery_manifest.json', manifest)
print(json.dumps({**audit, 'previous_deliveries_verified': verified, 'sealed_files': len(manifest)}, indent=2))
