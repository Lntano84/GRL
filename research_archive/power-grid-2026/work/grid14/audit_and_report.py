"""Independent stored-vector arithmetic and common-prefix review, no new physics or fit."""
import sys, math, gzip, csv, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'grid08'))
from common import ROOT, json, np, hashlib, write_json, make_env
from trace_agent import action_vector, digest

OUT = ROOT / 'outputs/grid14'
START = time.perf_counter()
D = json.loads((OUT / 'design.json').read_text())
finished = json.loads((OUT / 'finished.json').read_text()); assert finished['passed_engineering']
for file, expected in json.loads((OUT / 'code_freeze.json').read_text()).items():
    assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected, file

def read_rows(path):
    with gzip.open(path / 'steps.jsonl.gz', 'rt', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def fsum(a): return math.fsum(float(x) for x in np.asarray(a).ravel())

schema = json.loads((OUT / 'forecast_schema.json').read_text())
assert schema['physical_steps'] == 0 and schema['public_forecasts'] == 1
def decode(v, layout):
    assert len(v) == sum(field['size'] for field in layout)
    fields = {}; pos = 0
    for field in layout:
        fields[field['name']] = np.asarray(v[pos:pos+field['size']]).reshape(field['shape'])
        pos += field['size']
    return fields
forecast_cost_checks = 0

steps = action_checks = observation_checks = prefixes = 0
max_ledger_residual = 0.; comparisons = []; gates = []; results = {}
env = make_env(); env.seed(0); env.set_id(D['training_weeks'][0]); initial = env.reset()
try:
    for summary in finished['summaries']:
        key = (summary['scenario'], summary['rule']); path = OUT / 'runs' / f'{key[0]}__{key[1]}'
        rows = read_rows(path); meta = json.loads((path / 'metadata.json').read_text())
        assert len(rows) == summary['steps']; assert rows[-1]['complete'] == summary['complete']
        assert abs(math.fsum(r['raw_cost'] for r in rows) - summary['cost']) < 1e-6
        assert sum(r['simulations'] for r in rows) == summary['simulations']
        with np.load(path / 'vectors.npz') as compressed:
            # NPZ is lazy: indexing each array in the row loop re-decompresses the
            # whole trajectory each time. Materialize once, retain all-row checks.
            vectors = {name: compressed[name] for name in compressed.files}
            for i, row in enumerate(rows):
                a = env.action_space(); a.from_vect(vectors['action'][i])
                assert digest(action_vector(a)) == row['action_hash']; action_checks += 1
                obs = initial.copy(); obs.from_vect(vectors['observation'][i])
                assert digest(obs.to_vect()) == row['after_hash']; observation_checks += 1
                assert float(np.max(vectors['rho'][i])) == row['rho']
                assert np.array_equal(obs.rho, vectors['rho'][i])
                if row['ledger'] is not None:
                    price = max(float(p) for p, power in zip(meta['cost_per_MW'], vectors['gen_p'][i]) if power > 0)
                    losses = fsum(vectors['gen_p'][i]) - fsum(vectors['load_p'][i])
                    redisp = fsum(np.abs(vectors['actual_dispatch'][i]))
                    storage = fsum(np.abs(vectors['storage_power'][i]))
                    current_curt = fsum(vectors['curtailment_mw'][i])
                    old_curt = row['ledger']['previous_curtailed_mw']
                    if i > 0: assert old_curt == fsum(vectors['curtailment_mw'][i-1])
                    recomputed = price * meta['dt_hours'] * (losses + redisp + storage + current_curt - old_curt)
                    residual = abs(recomputed - row['raw_cost']); max_ledger_residual = max(max_ledger_residual, residual)
                    assert residual <= row['ledger_tolerance']
                steps += 1
        results[key] = {'summary': summary, 'rows': rows, 'path': path}
        if summary['probe_found']:
            gate_rows = json.loads((path / 'delivered_gates.json').read_text())
            assert len(gate_rows) == 1; g = gate_rows[0]; gates.append(g)
            assert g['discrete_exact']
            assert max(g['continuous_differences'].values()) <= D['continuous_gate_atol']
            assert g['rho_gap'] <= D['forecast_rho_gate_atol'] and g['cost_gap'] <= g['cost_tolerance']
            assert summary['actual_gate']
    for week in D['training_weeks']:
        baseline = results[(week, 'AUTHOR')]
        if not baseline['summary']['probe_found']:
            comparisons.append({'scenario': week, 'rule': 'NONE', 'status': 'NO_QUALIFYING_EVENT'})
            continue
        probe = json.loads((baseline['path'] / 'probe.json').read_text()); k = probe['action_step']
        meta = json.loads((baseline['path'] / 'metadata.json').read_text())
        with np.load(baseline['path'] / 'probe_vectors.npz') as saved:
            assert digest(saved['before_observation']) == probe['before_hash']
            before = decode(saved['before_observation'], schema['native_schema'])
            for c in probe['candidates']:
                i = c['source_index']
                a = env.action_space(); a.from_vect(saved[f'c{i}_delivered'])
                assert digest(action_vector(a)) == c['delivered_hash']
                topo = env.action_space(); topo.from_vect(saved[f'c{i}_topology'])
                assert digest(action_vector(topo)) == c['topology_hash']
                # Grid2Op's forecast schema has gen_p_delta and protection fields
                # beyond the physical observation. Decode with its actual layout.
                future = decode(saved[f'c{i}_forecast'], schema['forecast_schema'])
                assert float(future['rho'].max()) == c['final_forecast_rho']
                if c['qualified']:
                    price = max(float(p) for p, power in zip(meta['cost_per_MW'], future['gen_p']) if power > 0)
                    value = price * meta['dt_hours'] * (fsum(future['gen_p']) - fsum(future['load_p']) +
                        fsum(np.abs(future['actual_dispatch'])) + fsum(np.abs(future['storage_power'])) +
                        fsum(future['curtailment'] * env.gen_pmax) - fsum(before['curtailment'] * env.gen_pmax))
                    assert abs(value-c['final_forecast_cost']) < 1e-8
                    forecast_cost_checks += 1
            q = [c for c in probe['candidates'] if c['qualified']]
            assert len({c['delivered_hash'] for c in q}) >= 2
            assert probe['choices']['FINAL_COST'] == min(q, key=lambda c: (c['final_forecast_cost'], c['source_index']))
            assert probe['choices']['FINAL_RHO'] == min(q, key=lambda c: (c['final_forecast_rho'], c['source_index']))
        hash_to_result = {probe['choices']['AUTHOR']['delivered_hash']: baseline}
        for rule in ['FINAL_COST', 'FINAL_RHO']:
            choice = probe['choices'][rule]; data = results.get((week, rule))
            identical = False
            if data is None:
                assert (baseline['path'] / f'{rule}_identical.json').exists()
                data = hash_to_result[choice['delivered_hash']]; identical = True
            else:
                hash_to_result[choice['delivered_hash']] = data
                for ref, r in zip(baseline['rows'][:k-1], data['rows'][:k-1]):
                    for field in ['before_hash', 'action_hash', 'after_hash', 'raw_cost']:
                        assert ref[field] == r[field], (week, rule, r['step'], field)
                    prefixes += 1
                assert data['rows'][k-1]['before_hash'] == probe['before_hash']
                assert data['summary']['prefix_exact_steps'] == k-1
            b = baseline['summary']; s = data['summary']; joint = b['complete'] and s['complete']
            c24_b = math.fsum(r['raw_cost'] for r in baseline['rows'][k-1:k+23])
            c24_s = math.fsum(r['raw_cost'] for r in data['rows'][k-1:k+23])
            c288_b = math.fsum(r['raw_cost'] for r in baseline['rows'][k-1:k+287])
            c288_s = math.fsum(r['raw_cost'] for r in data['rows'][k-1:k+287])
            same_length = len(baseline['rows']) == len(data['rows'])
            suffix_start = None
            if same_length:
                for pos in range(len(baseline['rows'])-1, k-2, -1):
                    ref, row = baseline['rows'][pos], data['rows'][pos]
                    equal = all(ref[field] == row[field] for field in ['before_hash', 'action_hash', 'after_hash', 'raw_cost'])
                    if not equal: break
                    suffix_start = pos+1
            comparisons.append({'scenario': week, 'rule': rule, 'action_step': k, 'module': probe['module'],
                'identical_intervention': identical, 'baseline_complete': b['complete'], 'alternative_complete': s['complete'],
                'baseline_steps': b['steps'], 'alternative_steps': s['steps'],
                'forecast_cost_delta': choice['final_forecast_cost'] - probe['choices']['AUTHOR']['final_forecast_cost'],
                'forecast_rho_delta': choice['final_forecast_rho'] - probe['choices']['AUTHOR']['final_forecast_rho'],
                'cost24_baseline': c24_b, 'cost24_alternative': c24_s,
                'cost288_baseline': c288_b, 'cost288_alternative': c288_s,
                'whole_cost_baseline': b['cost'] if joint else None, 'whole_cost_alternative': s['cost'] if joint else None,
                'whole_gain': (b['cost'] - s['cost']) / abs(b['cost']) if joint else None,
                'cost_rank_eligible': joint,
                'identical_suffix_first_step': suffix_start,
                'identical_suffix_steps': len(baseline['rows'])-suffix_start+1 if suffix_start else 0,
                'first24_available': min(b['steps'], s['steps']) >= k+23,
                'first288_available': min(b['steps'], s['steps']) >= k+287})
finally:
    env.close()
assert steps == sum(s['steps'] for s in finished['summaries']) == finished['resources']['physical_steps']
assert sum(s['simulations'] for s in finished['summaries']) == finished['resources']['public_forecasts']

# Both interrupted attempts occurred at the same state; recheck prefix before explaining copy gates.
valid = results[(D['training_weeks'][0], 'AUTHOR')]
invalid_records = []
for folder in ['invalid_attempt_01', 'invalid_attempt_02']:
    path = OUT / folder / 'runs' / f"{D['training_weeks'][0]}__AUTHOR"
    rows = read_rows(path); failure = json.loads((path / 'failure.json').read_text())
    assert len(rows) == 200
    for r, ref in zip(rows, valid['rows']):
        for field in ['before_hash', 'action_hash', 'after_hash', 'raw_cost']: assert r[field] == ref[field]
    invalid_records.append({'attempt': folder, 'physical_steps': failure['counts']['physical_steps'],
                            'public_forecasts_committed_or_recorded': failure['counts']['public_forecasts'],
                            'shadow_qp_calls': failure['counts']['shadow_qp_calls'], 'wall_s': failure['wall_s']})
diag = json.loads((OUT / 'shadow_delivery_diagnosis/result.json').read_text())
assert diag['physical_steps'] == 200
clone = json.loads((OUT / 'clone_copy_diagnosis.json').read_text())
assert clone['all_numerically_equal']
# V0 missed the uncommitted call counter. Bound it from the exact source path;
# all lines connected => one outer reconnection forecast, <=n_sub recovery candidates,
# 20 neural shortlist forecasts, and one combined forecast before the copy assertion.
uncommitted_lower = 22; uncommitted_upper = 1 + 118 + 20 + 1
engineering_physical = sum(x['physical_steps'] for x in invalid_records) + diag['physical_steps']
engineering_forecast_known = sum(x['public_forecasts_committed_or_recorded'] for x in invalid_records) + diag['public_forecasts'] + schema['public_forecasts']
resources = {'formal_physical_steps': steps, 'engineering_physical_steps': engineering_physical,
             'physical_steps_total': steps + engineering_physical,
             'formal_forecasts': finished['resources']['public_forecasts'],
             'engineering_forecasts_lower': engineering_forecast_known + uncommitted_lower,
             'engineering_forecasts_upper': engineering_forecast_known + uncommitted_upper,
             'all_forecasts_lower': finished['resources']['public_forecasts'] + engineering_forecast_known + uncommitted_lower,
             'all_forecasts_upper': finished['resources']['public_forecasts'] + engineering_forecast_known + uncommitted_upper,
             'formal_shadow_qp_calls': finished['resources']['shadow_qp_calls'],
             'formal_phase_wall_s': finished['wall_s'], 'training_runs': 0,
             'engineering_note': 'V0 failed-decision counter was not logged; exact total is unavailable, bounded explicitly. V1 and diagnosis counters include their stopped decisions. Import/zero-step setup wall excluded from simulator phase.'}
# Counter arithmetic does not by itself establish a latency saving: expensive
# forecasts coincide with other modules and CPU timing was not split per module.
query_profile = []
for (week, rule), data in results.items():
    if rule != 'AUTHOR': continue
    heavy = [r for r in data['rows'] if r['simulations'] >= 500]
    query_profile.append({'scenario': week, 'total_forecasts': data['summary']['simulations'],
                          'decisions_with_at_least500_forecasts': len(heavy),
                          'forecasts_in_those_decisions': sum(r['simulations'] for r in heavy),
                          'fraction_of_all_forecasts': sum(r['simulations'] for r in heavy)/data['summary']['simulations']})
write_json(OUT / 'query_profile.json', query_profile)
verified = {}
for name in ['grid08', 'grid09', 'grid10', 'grid11_failure_context', 'grid12', 'grid13']:
    old = json.loads((ROOT / 'outputs' / name / 'delivery_manifest.json').read_text())
    for file, expected in old.items():
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == expected, file
    verified[name] = len(old)
joint = [c for c in comparisons if c.get('cost_rank_eligible') and not c['identical_intervention']]
material_weeks = sorted({c['scenario'] for c in joint if abs(c['whole_gain']) >= .01})
simple_positive = {}
for rule in ['FINAL_COST', 'FINAL_RHO']:
    eligible = [c for c in comparisons if c.get('cost_rank_eligible') and c['rule'] == rule]
    simple_positive[rule] = bool(eligible and all(c['whole_gain'] >= 0 for c in eligible))
regrets = {rule: [] for rule in ['AUTHOR', 'FINAL_COST', 'FINAL_RHO']}
joint_weeks = []
for week in D['training_weeks']:
    eligible = [c for c in comparisons if c['scenario'] == week and c.get('cost_rank_eligible')]
    if len(eligible) != 2: continue
    baseline_cost = eligible[0]['whole_cost_baseline']
    costs = {'AUTHOR': baseline_cost, **{c['rule']: c['whole_cost_alternative'] for c in eligible}}
    best = min(costs.values()); joint_weeks.append(week)
    for rule in regrets:
        regrets[rule].append({'scenario': week, 'gap_to_best_observed': (costs[rule]-best)/abs(baseline_cost)})
near_best = {rule: bool(values and max(x['gap_to_best_observed'] for x in values) < .01)
             for rule, values in regrets.items()}
verdict = {'material_joint_complete_weeks': material_weeks, 'nonnegative_simple_rules_all_joint_complete': simple_positive,
           'joint_complete_weeks_all_three': joint_weeks, 'observed_rule_regrets': regrets,
           'simple_rule_within_1pct_of_best_observed_every_joint_week': near_best,
           'action_evaluator_gate': len(material_weeks) >= 2 and bool(joint_weeks) and not any(near_best.values()),
           'interpretation': 'One event per already exposed training week and one seed. No statistical generalization or trained policy result.'}
audit = {'passed': True, 'physical_step_rows': steps, 'action_roundtrips': action_checks,
         'forecast_cost_scalar_checks': forecast_cost_checks, 'schema_qualification_forecasts': schema['public_forecasts'],
         'audit_wall_s': time.perf_counter() - START,
         'observation_roundtrips': observation_checks, 'common_prefix_exact_step_checks': prefixes,
         'delivery_gates': len(gates), 'max_independent_ledger_residual': max_ledger_residual,
         'preserved_failed_attempts': invalid_records, 'prior_deliveries_verified': verified,
         'scope': 'Stored-vector decoding, math.fsum ledger calculation, candidate rule and prefix recomputation; one public forecast for schema qualification only, no independent AC solver or model fit.'}
write_json(OUT / 'audit.json', audit); write_json(OUT / 'comparison.json', comparisons)
write_json(OUT / 'resources_audited.json', resources); write_json(OUT / 'interpretation.json', verdict)
fields = sorted({key for c in comparisons for key in c})
with (OUT / 'comparison.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(comparisons)

lines = ['# GRID14：加入恢复后的拓扑候选与后续费用', '',
         '本轮实际完成受控诊断，没有拟合新模型。作者拓扑、危险状态 QP、ALWAYS_RESTORE 均保留。', '',
         '| 训练周 | 干预步 | 规则 | 完成：原/改 | 一步预测费用差 | 整周费用降幅 |',
         '|---|---:|---|---|---:|---:|']
for c in comparisons:
    if c['rule'] == 'NONE':
        lines.append(f"| {c['scenario']} | — | 无合格事件 | — | — | 不可判定 |")
        continue
    gain = f"{100*c['whole_gain']:+.4f}%" if c['cost_rank_eligible'] else '不对失败分支作费用排名'
    lines.append(f"| {c['scenario']} | {c['action_step']} | {c['rule']} | {c['baseline_complete']}/{c['alternative_complete']} | {c['forecast_cost_delta']:+.4f} | {gain} |")
lines += ['', '## 如何解读', '',
          f"至少达到原整周费用 1% 的绝对差异，在 {len(material_weeks)} 个共同完成的训练周出现。"
          '这是一次性干预与固定续接的结果，不是最优 oracle、可部署完整策略或学习优势。', '',
          '事件由固定时钟窗口和公共有效候选条件确定，未按后续费用或攻击位置挑选。'
          '候选来自原控制器实际短名单中的前八个合格拓扑；没有证明其他候选或其他时点没有空间。'
          '三个简单规则能获得相同候选描述；真实后续费用仅作事后诊断，不进入动作选择。', '',
          '危险 QP 使用作者的既有数值求解协议。其中存在 user_limit；它可以产生通过物理预测屏幕的动作，'
          '但本轮不把该状态称为最优解。每个实际候选的离散动作、连续误差和一步预测均重新核验。'
          '候选额外预测/QP集中在诊断控制臂，费用分开记录；本轮不宣称同延迟部署优势。', '',
          '整周完成优先于费用。失败分支的低累计费用不能写成改进。24/288 步费用仅在对应完整片段上解释。'
          '原 GRID07 恢复前的单次反例没有被当作本轮收益证据。', '',
          '## 工程修正与封存', '',
          '两次门槛失败均保留：V0 将数值相同的 int64/float64 参数按原始字节比较；V1 要求近似 QP 连续输出逐字节相同。'
          '零物理步参数检查确认全部数值相同；单独 200 步诊断测得连续动作最大差约 1.29e-5、离散拓扑一致、预测最大 rho 一致。'
          'V2 在继续正式结果前冻结：离散动作精确；连续 atol=1e-4；预测 rho atol=1e-5；费用用独立舍入误差界。'
          '没有用它放宽线路/拓扑合法性或改训练结果。共同前缀仍按实际动作、观测哈希和费用逐项一致检查。', '',
          'V0 未存失败决策的预测计数，无法伪造精确总数，资源表明确保留下上界；物理步全部计入。'
          '六个旧阶段封存哈希复核通过。封存最终测试未评价，当前四周是已见训练材料。', '',
          '审核记录另保留两个尝试：首次全向量核验因逐行重复解压完整 NPZ 耗时过高而中止；'
          '改为每条轨迹一次解压，仍核验每一行。第二次错误地把 forecast 向量按物理观测 schema 解码，'
          '由于长度 4,712 与 4,460 不同而中止。单独一次公共预测确认实际 schema，'
          '随后按布局独立解码并重算候选费用；没有重跑正式控制轨迹。', '',
          '第三次候选复算把 curtailment_mw 当作序列化字段，触发 KeyError；'
          '它实际上是 curtailment×gen_pmax 的派生量。修正后从储存比例和公开固定容量恢复，'
          '该失败脚本与日志也已留档。以上审核失败均没有发布有效判决或改动实验数据。', '',
          '## 可借鉴的已有方法', '',
          '[SMAAC 作者实现](https://github.com/sunghoonhong/SMAAC)已有 afterstate 与高层目标/低层执行。'
          '使用候选后的图状态本身不足以构成创新。'
          '[Applied Energy 图表示研究](https://arxiv.org/html/2501.07186v3)讨论当前与潜在母线连接，'
          '因此若后续训练动作评价器，应让表示明确包含候选连接和控制量，不能只更换当前状态图的网络名称。', '',
          '后续模型拟合必须由可重复的候选差异和简单规则残余损失支持；本轮没有开始新的 GNN/RL 训练。', '',
          '资源和审计：见 resources_audited.json、audit.json、comparison.csv。']
(OUT / 'GRID14_review_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('COMPLETE. Formal rollouts and two preserved engineering failures audited. No training or pending job.\n', encoding='utf-8')
print(json.dumps({'audit': audit, 'interpretation': verdict, 'resources': resources}, indent=2))
