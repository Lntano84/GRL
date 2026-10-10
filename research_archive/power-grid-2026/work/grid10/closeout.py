"""Seal the controlled critic experiment and verify original versions unchanged."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / 'outputs/grid10'
assert json.loads((OUT / 'finished.json').read_text())['passed']
neural = json.loads((OUT / 'neural_audit.json').read_text()); assert neural['passed']
result = json.loads((OUT / 'results.json').read_text())
comparisons = {(x['version'], x['reference']): x for x in result['comparisons']}
previous = {}
for name in ['grid08', 'grid09']:
    manifest = json.loads((ROOT / 'outputs' / name / 'delivery_manifest.json').read_text())
    for path, expected in manifest.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    previous[name] = len(manifest)
assert all(comparisons[v, 'ALWAYS_RESTORE']['identical'] == 4 for v in ['raw', 'normalized'])
assert comparisons['normalized', 'RAW_HUBER']['identical'] == 4
lines = [
    '# GRID10：价值训练处理的控制实验', '',
    '**结果：两臂均完成 4/4 开发评价周，与“始终恢复”逐项相同；没有建立额外学习收益。**', '',
    '| 方法 | 完成评价周 | 相对 NN20 成本降幅（逐周宏平均） | 相对始终恢复 | 相同动作＋观测轨迹 |',
    '|---|---:|---:|---:|---:|',
]
for v, label in [('raw', '原尺度 Huber'), ('normalized', '固定标准化 MSE')]:
    nn, always = comparisons[v, 'NN20'], comparisons[v, 'ALWAYS_RESTORE']
    lines.append(f"| {label} | {nn['joint_complete']}/4 | {100 * nn['mean_gain']:.4f}% | {100 * always['mean_gain']:.4f}% | {always['identical']}/4 |")
lines += [
    '', '相对 NN20 的约 63.86% 成本改善是连续恢复模块及其使用规则的效果，'
    '不能归因于 GNN、RL 或本轮价值训练改动。四个开发周此前已反复使用，'
    '一个训练种子；封存最终测试集没有用于训练、调参或选择 checkpoint。', '',
    '两臂的 actor 初始权重完全相同，均来自 V2；critic 分别采用已审计、'
    '同预算训练数据拟合的原尺度 Huber 与固定标准化 MSE 权重，因此完整初始网络并不相同。'
    '两臂都重置 Adam，训练顺序、物理步 GAE、奖励、模型结构、候选和安全筛选相同。'
    '固定尺度和 MSE 损失同时改变，仅检验整个训练处理，不能分别归因，也不是自适应 PopArt。', '',
    '保存特征上的最终模型复预测核对了 1,748 个实际可选动作，全部与记录中的决策相符。'
    '这是使用共享 forward 实现的核对，不是独立实现或独立训练。保存向量的成本重算与冻结代码审计也通过。', '',
    '训练中的策略不同：标准化臂第二轮十月在 133 步失败，而对照完成该周；'
    '这些是随训练变化的随机策略轨迹，不能直接作为最终策略的因果比较，'
    '更不能把提前结束后的低累计成本当作节约。最终开发评价两臂都完成。', '',
    '仅训练数据的末轮描述性 critic RMSE：Huber 为约 105.35，标准化为约 161.28；'
    '两者来自各自不同的轨迹和样本，不能用这个差值证明哪种损失拟合更差。'
    '两臂最终均选择始终恢复，说明本轮修补没有改变最终决策。', '',
    '运行器报告阶段发现旧版本参数残留：finish.py 向审计程序传入 physical/decision，'
    '参数解析立即拒绝。训练与评价此前均已成功；旧脚本和错误日志归档，'
    '修正为 raw/normalized 后只继续审计和报告，没有重跑训练或评价。', '',
    f"物理步数：Huber {result['resources']['raw']}，标准化 {result['resources']['normalized']}，"
    f"合计 {result['resources']['total_physical']}，低于预定 50,000 步上限。"
    '已有 critic 拟合费用另行披露，不能把先前诊断权重当作免费生成。', '',
    f"原封存验证：GRID08 {previous['grid08']} 个文件、GRID09 {previous['grid09']} 个文件哈希均不变。"
    '原模型、检查点、失败尝试及改动记录全部保留。', '',
    '## 对后续修补的判断', '',
    '保留作者拓扑策略、危险状态 QP、独立连续恢复模块和公共预测接口。'
    '当前低风险二选一任务中，学习没有超过简单始终恢复规则；不继续围绕它叠加网络。'
    '已单独诊断一月训练失败：关键线路过载后切除，且最后风险状态的现有拓扑池全部失败。'
    '下一项工作检查更早的控制时机和候选能力；不能把这些诊断直接宣称为学习优势或论文创新。', '',
    '电网依据及与已有工作的差异见 [literature_review.md](literature_review.md)。',
]
(OUT / 'GRID10_review_zh.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('COMPLETE: training, fixed final-checkpoint evaluation, stored-vector audit and neural re-prediction audit passed. No final-test use; all originals retained. Reporting-only orchestration error archived.\n', encoding='utf-8')
delivery = {'passed': True, 'resources': result['resources'], 'previous_files_verified': previous,
            'neural_choices_recomputed': neural['total_choices_recomputed'], 'final_test_used': False,
            'extra_learning_gain_vs_always_restore': 0.0}
(OUT / 'delivery.json').write_text(json.dumps(delivery, indent=2), encoding='utf-8')
files = [p for directory in [OUT, ROOT / 'work/grid10'] for p in directory.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name != 'delivery_manifest.json']
manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(OUT / 'delivery_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps({**delivery, 'sealed_files': len(manifest)}, indent=2))
