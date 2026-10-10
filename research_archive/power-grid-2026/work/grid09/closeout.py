"""Finalize the bounded comparison; preserve and verify the previous delivery."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid09'
assert json.loads((OUT / 'finished.json').read_text())['passed']
result = json.loads((OUT / 'results.json').read_text())
comparisons = {(r['version'], r['reference']): r for r in result['comparisons']}
previous = json.loads((ROOT / 'outputs/grid08/delivery_manifest.json').read_text())
for path, expected in previous.items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path

lines = [
    '# GRID09：回报归因对照与审计', '',
    '两臂均从同一份 V2 GNN 权重开始，重置 Adam，并按相同顺序完成两轮四个训练周。'
    '普通续训每个物理步应用 λ；改动臂仅在连接下一个实际可选状态时应用 λ，'
    '物理折扣 γ、奖励、安全筛选、候选、特征和模型结构不变。', '',
    '| 方法 | 完成评价周 | 对 NN20 平均成本降幅 | 对始终恢复平均成本降幅 | 与始终恢复逐项相同 |',
    '|---|---:|---:|---:|---:|',
]
for version, name in [('physical', '普通 PPO 续训'), ('decision', '决策间归因')]:
    records = [r for r in result['records'] if r['version'] == version]
    complete = sum(r['complete'] for r in records)
    nn, always = comparisons[version, 'NN20'], comparisons[version, 'ALWAYS_RESTORE']
    def gain(item):
        return '不比较未完成轨迹' if item['mean_gain'] is None else f"{100 * item['mean_gain']:.4f}%"
    lines.append(f"| {name} | {complete}/4 | {gain(nn)} | {gain(always)} | {always['identical']}/4 |")
direct = comparisons['decision', 'PHYSICAL_CONTINUE']
lines += ['', f"两臂之间：{direct['identical']}/4 周动作与观测哈希序列相同；"
          f"共同完成 {direct['joint_complete']}/4 周。"]
if all(comparisons[v, 'ALWAYS_RESTORE']['identical'] == 4 for v in ['physical', 'decision']):
    lines += ['', '**本轮没有建立学习器的额外决策价值。** 两臂和始终恢复执行完全相同，'
              '因此相对 NN20 的改善属于新增恢复模块；不能归因于 GNN、RL 或本轮归因改动。']
lines += [
    '', '训练中不完整轨迹保留，不用其较低累计成本宣称改善。训练后的描述性诊断'
    '仍发现价值预测与所经历的回报差距较大，但这些回报来自变化中的随机策略，'
    '不能据此认定价值误差就是失败根因。', '',
    '本轮保留 256 步 rollout 的截断，不能称为完整半马尔可夫或 SMAAC 复现。'
    '只有一个训练种子、四个已用过的开发评价周，未使用封存最终测试集进行训练、'
    '调参或成绩选择，不能声称独立泛化确认。', '',
    f"独立保存向量的成本与计量审计通过；本轮总物理步 {result['resources']['total_physical']}，"
    '未超 50,000 步上限。该审计不是独立电力潮流重跑或独立训练。', '',
    f"GRID08 上一轮封存清单 {len(previous)} 个文件逐一哈希不变；"
    '所有原模型、试过的改动、失败记录和检查点保留。', '',
    '下一项可检验假设是价值目标的尺度与损失处理是否妨碍学习；已有仅训练数据的'
    '拟合探针显示标准化 MSE 可降低训练 RMSE，但尚没有控制器收益证据。'
    '不能把价值拟合改善当作行动改善，也不能直接把这项常用技巧作为论文创新。',
]
(OUT / 'GRID09_review_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text(
    '# GRID09\n\nCOMPLETE; TWO TRAINING AND TWO EVALUATION PHASES AUDITED.\n\n'
    'Fixed final checkpoints; no final-test use. All originals retained. '
    'Training-only diagnostic uses changing-policy realized returns, not unbiased policy-value targets.\n',
    encoding='utf-8')
delivery = {'passed': True, 'previous_grid08_files_verified': len(previous),
            'resources': result['resources'], 'final_test_used': False}
(OUT / 'delivery.json').write_text(json.dumps(delivery, indent=2), encoding='utf-8')
files = [p for directory in [OUT, ROOT / 'work/grid09'] for p in directory.rglob('*')
         if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json']
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(OUT / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({**delivery, 'sealed_files': len(manifest)}, indent=2))
