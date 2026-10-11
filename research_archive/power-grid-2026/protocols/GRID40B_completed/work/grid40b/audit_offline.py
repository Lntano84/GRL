"""Independent stopped-query/choice reconstruction for both adapted seeds."""
import hashlib
import json
import struct
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid40b'
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def key(r): return struct.unpack('f', struct.pack('f', r['reward']))[0], -r['pool_id']
subs = read(ROOT / 'outputs/grid28/offline.json')['public_subids']
def independent(order, truth, aliases):
    seen = set(); best = None
    def acquire(ids):
        nonlocal best
        previous = None if best is None else key(best)
        for i in ids:
            assert i not in seen; seen.add(i); r = truth[i]
            if r['strict_admissible'] and (best is None or key(r) > key(best)): best = r
        return best is not None and (previous is None or key(best) > previous)
    acquire(order[:128]); fallback = len(order) > 128 and (best is None or best['rho'] >= .9)
    if fallback:
        stale = 0
        for offset in range(128, len(order), 64):
            stale = 0 if acquire(order[offset:offset + 64]) else stale + 1
            if best is not None and best['rho'] < .9: break
            if best is not None and len(seen) >= 256 and stale >= 2 and len(seen) < len(order): break
    if best is not None and fallback and len(seen) < len(order):
        sub = subs[best['pool_id']]
        targets = {int(rep) for i, rep in aliases.items() if subs[int(i)] == sub}
        acquire([i for i in order if i in targets and i not in seen])
    return len(seen), None if best is None else best['pool_id']
R = read(OUT / 'offline_review.json'); assert R['passed_execution'] and len(R['rows']) == 52
for p, h in R['source_hashes'].items(): assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
for r in R['rows']:
    full = read(ROOT / 'outputs/grid26/runs' / f"r0__{r['week']}__FULL" / r['call'])
    truth = {t['pool_id']: t for t in full['outcomes']}
    assert set(r['order']) == set(full['representatives']) and len(set(r['order'])) == len(r['order'])
    assert independent(r['order'], truth, full['aliases']) == (r['queries'], r['choice'])
    winner = full['chosen_pool_id']; assert winner == r['full_choice']
    gap = None if r['choice'] is None or winner is None else truth[r['choice']]['rho'] - truth[winner]['rho']
    assert gap == r['rho_regret'] and (winner == r['choice']) == r['same_full_choice']
audit = {'passed': True, 'independent_stopping_checks': len(R['rows']), 'same_source_hashes': True,
         'new_forecasts': 0, 'new_fits': 0, 'reserved_evaluations': 0,
         'warning': 'Independent saved feedback/stop/choice reconstruction,not an independent model refit or powerflow solve.'}
(OUT / 'offline_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
fits = read(OUT / 'audit.json')['fits']
lines = ['# GRID40B SoftGNN核心适配及训练优化检查', '',
    '官方GAT和27维输入转换代码未修改。IDF2023、909动作、公开base连续动作上下文与共同搜索mask是明确适配；不是作者WCCI2022完整复现。', '',
    '原论文学习率配置在148个小样本/批量8上训练进入平台；一版学习率0.001的修改只依据训练记录，原初始化/64轮/两种子/模型/dropout/数据不变，所有旧新模型保留。', '',
    '| 种子 | 原训练KL | 修订训练KL | 静态频率KL |', '|---|---:|---:|---:|']
for r in fits: lines.append(f"| {r['seed']} | {r['old_final_training_kl']:.4f} | {r['new_final_training_kl']:.4f} | {r['frequency_kl']:.4f} |")
lines += ['', '| 模型 | 26状态查询 | 原GNN查询 | 同FULL选择 | 最大rho差 | 关键1733匹配 |', '|---|---:|---:|---:|---:|---|']
for r in R['summaries']: lines.append(f"| {r['ranker']} | {r['queries']} | {r['original_queries']} | {r['same_full_choice']}/26 | {r['max_rho_regret']:.7f} | {r['known_critical_choice_match']} |")
lines += ['', R['warning'], '', '训练拟合改善不是闭环质量提升；下一步以同一控制器和共有优化测完整质量—时间，不把公开方法适配版的有限结果外推到原论文。']
(OUT / 'GRID40B_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
print(json.dumps(audit))
