"""Persisted-vector forecast audit and public-state decoding, no new transitions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, hashlib, make_env, write_json
from trace_agent import action_vector, digest

OUT = ROOT / 'outputs/grid12'
result = json.loads((OUT / 'results.json').read_text()); assert result['passed']
design = json.loads((OUT / 'design.json').read_text())
for path, expected in json.loads((OUT / 'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
assert result['physical_steps'] == design['physical_cap']
assert result['qp_solves'] == len(design['states']) == 2
assert result['forecast_steps_including_QP'] == 2 + sum(len(branch) for record in result['records'] for branch in record['branches'].values())
assert result['forecast_steps_including_QP'] <= design['forecast_step_cap']
assert result['wall_s'] < design['wall_cap_s']

env = make_env(); env.seed(0); env.set_id(design['source']['scenario']); initial = env.reset()
contexts = []; verified_vectors = 0
try:
    with np.load(OUT / 'public_vectors.npz') as data:
        for record in result['records']:
            i = record['state_step']; obs = initial.copy(); obs.from_vect(data[f's{i}_public_observation'])
            assert abs(float(obs.rho.max()) - record['current_rho']) < 1e-8
            a = data[f's{i}_saved_base_action']; b = data[f's{i}_default_unsafe_QP_action']
            assert record['qp_differs'] == (not np.array_equal(a, b))
            for name, branch in record['branches'].items():
                action = env.action_space(); action.from_vect(data[f's{i}_{name}_action'])
                assert np.array_equal(action_vector(action), data[f's{i}_{name}_action'])
                for h, row in enumerate(branch, 1):
                    future = initial.copy(); future.from_vect(data[f's{i}_{name}_forecast{h}'])
                    assert row['horizon_step'] == h
                    assert abs(float(future.rho.max()) - row['rho']) < 1e-8
                    assert abs(float(future.rho[11]) - row['line_11_rho']) < 1e-8
                    assert int(future.timestep_overflow[11]) == row['line_11_overflow']
                    if row['done']: assert h == len(branch)
                    verified_vectors += 1
            contexts.append({'step': i, 'observation_hash': digest(obs.to_vect()),
                'line_81': {'status': bool(obs.line_status[81]), 'cooldown': int(obs.time_before_cooldown_line[81]),
                            'next_maintenance': int(obs.time_next_maintenance[81]), 'maintenance_remaining': int(obs.duration_next_maintenance[81])},
                'generator_40': {'p': float(obs.gen_p[40]), 'type': str(env.gen_type[40]),
                                 'redispatchable': bool(env.gen_redispatchable[40]),
                                 'margin_up': float(obs.gen_margin_up[40]), 'margin_down': float(obs.gen_margin_down[40])},
                'loads_60_98_total_mw': float(obs.load_p[[60, 98]].sum()),
                'scope': 'Decoded archived public state; no additional simulate/step call.'})
finally:
    env.close()
write_json(OUT / 'public_context_decoded.json', contexts)
audit = {'passed': True, 'forecast_vectors_recomputed': verified_vectors, 'action_vectors_roundtrip': 4,
         'additional_physical_steps': 0, 'additional_forecast_steps': 0, 'additional_fits': 0,
         'environment_initialization_for_decoding': 1,
         'scope': 'Archived observation/action decoding and scalar recount with shared Grid2Op vector schema; not independent forecast power-flow replay.'}
write_json(OUT / 'audit.json', audit)
lines = [
    '# GRID12：提前调用现有应急 QP 的公共预测探针', '',
    '**结果：两次现成应急 QP 都优化成功并产生不同动作，但没有降低关键线路的第一步预测过载，也没有避免该固定预测分支终止。**', '',
    '| 已见训练状态 | 现有动作分支 | 一次应急 QP 分支 | QP 状态 |',
    '|---|---|---|---|',
]
for r in result['records']:
    branches = r['branches']; base = branches['saved_base']; qp = branches['default_unsafe_QP']
    lines.append(f"| {r['state_step']} | 第 {len(base)} 个预测步终止；第一步 ρ={base[0]['rho']:.6f} | 第 {len(qp)} 个预测步终止；第一步 ρ={qp[0]['rho']:.6f} | {', '.join(r['qp_solve_status'])} |")
lines += [
    '', '控制变量：冻结同一已见训练失败前缀，两个预先固定状态；使用作者默认 unsafe QP，'
    '从当前公开观测独立 reset，不改参数、不改原 agent 的记忆。每条预测分支仅第一步'
    '采用其候选，后续均 no-op，最多六步并在终止时停止。', '',
    '因此本轮只排除了“在这两个状态提前单次调用当前默认 QP、然后不再控制，'
    '就能消除该公共预测故障”的设想。它不是闭环 MPC，也没有测试更早的拓扑准备、'
    '每步重新优化、全拓扑空间或真实整周存活，不能推出这些方法无效。', '',
    '第一步最大负载率相同，不代表两份控制完全相同；动作向量已确认不同，'
    '其他线路预测也可能不同。优化成功只表示 QP 子问题求解状态，不能当作'
    '原系统保证安全的证书。该优化含软过载惩罚，与完整 AC 约束不同。', '',
    '## 与电网方法的关系', '',
    '[RL2Grid](https://arxiv.org/html/2503.23101v2)明确区分平稳时的规则与风险时的控制，'
    '并按完整运行与约束评价 RL。当前模块是其思路下的受控诊断，不是它的复现。'
    '[SMAAC](https://github.com/sunghoonhong/SMAAC)提供高层拓扑目标、afterstate 和低层执行的参考；'
    '不能据此保证加入动作后特征会在我们的二选一任务中产生收益。', '',
    '保留连续恢复的已测收益、作者拓扑候选及公共预测/独立账目。'
    '暂不继续在当前二选一接收器上换网络；后续应先检查关键连接风险的控制权限与更早候选，'
    '再决定学习承担动作提议、排序还是搜索辅助。必须让同信息的非学习规则拥有相同候选。', '',
    f"资源：{result['physical_steps']} 个冻结前缀物理步、{result['forecast_steps_including_QP']} 个公共预测步"
    f"（含 QP 内部预测）、{result['qp_solves']} 次 QP，仿真阶段 {result['wall_s']:.2f} 秒。"
    '本轮独立列账，不计入 GRID10 训练额度；没有新模型拟合。', '',
    '已存公共状态、动作与各步预测向量；用相同 Grid2Op 向量结构重读核对，'
    '不声称独立电力潮流复现。全部既有模型与试验记录保留。',
]
(OUT / 'GRID12_review_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('COMPLETE. Two fixed-state QP forecast branches and vector audit passed. No model fit or actual-future branch; no closed-loop MPC claim.\n', encoding='utf-8')
for name in ['grid08', 'grid09', 'grid10', 'grid11_failure_context']:
    for path, expected in json.loads((ROOT / 'outputs' / name / 'delivery_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
files = [p for directory in [OUT, ROOT / 'work/grid12'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json']
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
write_json(OUT / 'delivery_manifest.json', manifest)
print(json.dumps({**audit, 'public_contexts': contexts, 'sealed_files': len(manifest)}, indent=2))
