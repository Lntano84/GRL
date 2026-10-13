"""Read-only interpretation update after GRID59; no model fitting or rollout."""
import csv
import gzip
import hashlib
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/grid59_closeout'
REPO = ROOT / 'work/GRL-results-integration'
ARCH = REPO / 'research_archive/power-grid-2026/protocols/GRID59_closeout'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values) / len(values)


def steps(folder):
    with gzip.open(folder / 'steps.jsonl.gz', 'rt', encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def main():
    assert not OUT.exists() and not ARCH.exists()
    pilot = read(ROOT / 'outputs/grid59/review.json')
    design = read(ROOT / 'outputs/grid59/design.json')
    finished = read(ROOT / 'outputs/grid59/finished.json')
    final = read(ROOT / 'outputs/grid55b/qualified_view/review.json')
    static_review = read(ROOT / 'outputs/grid57_prior_review/review.json')
    assert pilot['passed_saved_data'] and final['passed_saved_data']
    assert pilot['verdict'] == 'WEAK_OR_MIXED_STOP_REVISION'
    assert static_review['library_order_matches_all8_runs']
    ix = {(r['week'], r['method']): r for r in finished['summaries']}
    result_rows = []
    for week in design['weeks']:
        baseline = ix[week, 'STATIC_ID']
        full = ix[week, 'FULL']
        base_steps = steps(ROOT / 'outputs/grid59/runs' / f'r0__{week}__STATIC_ID')
        for seed in (0, 1):
            method = f'FREQUENCY_GNN{seed}'
            actual = ix[week, method]
            actual_steps = steps(ROOT / 'outputs/grid59/runs' / f'r0__{week}__{method}')
            assert actual['steps'] == baseline['steps'] == full['steps']
            assert len(actual_steps) == len(base_steps)
            saved = next(r for r in pilot['pairs'] if r['week'] == week and r['method'] == method)
            query_reduction = 1 - actual['n1_forecasts'] / baseline['n1_forecasts']
            control_reduction = 1 - actual['controller_s'] / baseline['controller_s']
            assert abs(query_reduction - saved['query_reduction_static']) < 1e-14
            assert abs(control_reduction - saved['control_reduction_static']) < 1e-14
            n1_delta = actual['n1_source_s'] - baseline['n1_source_s']
            other_delta = (actual['controller_s'] - actual['n1_source_s']) - (baseline['controller_s'] - baseline['n1_source_s'])
            total_delta = actual['controller_s'] - baseline['controller_s']
            assert abs(total_delta - n1_delta - other_delta) < 1e-10
            differing_actions = sum(a['action_hash'] != b['action_hash'] for a, b in zip(actual_steps, base_steps))
            differing_observations = sum(a['after_hash'] != b['after_hash'] for a, b in zip(actual_steps, base_steps))
            model_s = actual['prior_s'] + actual['feature_s'] + actual['forward_rank_s']
            result_rows.append({
                'week': week, 'seed': seed, 'steps': actual['steps'], 'complete': actual['complete'],
                'query_reduction_static': query_reduction, 'control_reduction_static': control_reduction,
                'control_delta_s': total_delta, 'n1_source_delta_s': n1_delta,
                'outside_n1_delta_s': other_delta, 'graph_prior_forward_s': model_s,
                'action_hash_differences': differing_actions, 'observation_hash_differences': differing_observations,
                'cost_gain_static_of_full': saved['cost_gain_static_of_full'],
            })
    macro_query = mean([r['query_reduction_static'] for r in result_rows])
    macro_time = mean([r['control_reduction_static'] for r in result_rows])
    assert abs(macro_query - pilot['seed_average_macro_query_reduction_static']) < 1e-14
    assert macro_query < .05
    assert all(sum(r['query_reduction_static'] > 0 for r in result_rows if r['seed'] == s) == 2 for s in (0, 1))
    assert all(r['quality_qualified'] for r in pilot['pairs'] if r['method'].startswith('FREQUENCY'))
    final_summaries = {r['method']: r for r in final['summaries']}
    framework_speed = mean([final_summaries[f'COMPRESSED_GNN{s}']['macro_time_reduction_full'] for s in (0, 1)])
    framework_worst_cost = max(final_summaries[f'COMPRESSED_GNN{s}']['max_complete_week_cost_regret'] for s in (0, 1))
    claims = [
        {'claim': '原软压缩候选相对同共有缓存FULL减少完整控制时间', 'status': 'SUPPORTED_WITH_SCOPE',
         'value': framework_speed, 'scope': 'GRID55B八个已暴露日期的明确计时视图；不是新独立确认；不把查询减少等同完整提速。'},
        {'claim': '原软压缩候选守住预设0.5%完整周成本尺度', 'status': 'SUPPORTED_SCREEN_ONLY',
         'value': framework_worst_cost, 'scope': '两种子16/16完成，最坏成本变化约+0.08045%；不是等价检验。'},
        {'claim': '上述框架收益证明GNN有增量价值', 'status': 'NOT_ESTABLISHED',
         'scope': 'GRID57强STATIC为作者历史教师频次顺序；GNN查询宏平均仅少1.90%，汇总多6.69%，成本1好/6差/1平。未作跨批控制速度比较。'},
        {'claim': '换成频次先验已解决GNN增量问题', 'status': 'FAILED_FROZEN_CONTINUATION_SCREEN',
         'value': macro_query, 'scope': 'GRID59平均查询少2.901%，每种子2/4日期正向；实际控制多约0.570%；仅旧开发配对单次计时。'},
        {'claim': '频次先验GNN质量明显失败', 'status': 'REJECTED',
         'scope': '两种子质量4/4通过；三周完成，六月四方法同在1901步失败；不是方法特有的提前失败。'},
        {'claim': 'STATIC是不含学习经验的随机动作顺序', 'status': 'REJECTED',
         'scope': '公开动作库已按作者教师频次降序；作者教师日期未知，所有对照共有，不能指控泄漏。'},
        {'claim': '本轮建立了新的RL算法贡献', 'status': 'REJECTED',
         'scope': 'GRID58是监督残差排序拟合；GRID59未训练；原PPO只是既有控制器组成。'},
    ]
    source_paths = [
        Path(__file__), ROOT / 'outputs/grid59/design.json', ROOT / 'outputs/grid59/code_freeze.json',
        ROOT / 'outputs/grid59/finished.json', ROOT / 'outputs/grid59/review.json',
        ROOT / 'outputs/grid55b/qualified_view/review.json', ROOT / 'outputs/grid57_prior_review/review.json',
        ROOT / 'outputs/grid58/qualification.json',
    ]
    result = {
        'passed_arithmetic': True, 'new_simulations': 0, 'new_fits': 0,
        'framework_speed_full': framework_speed, 'framework_worst_complete_cost': framework_worst_cost,
        'frequency_macro_query_reduction_static': macro_query, 'frequency_macro_control_reduction_static': macro_time,
        'frequency_complete_cost_gain_static_of_full': mean([r['cost_gain_static_of_full'] for r in result_rows if r['cost_gain_static_of_full'] is not None]),
        'frozen_prior_revision_stopped': True,
        'interpretation': '限时框架提速保留，但当前两个先验下的GNN都未证明超过强频次顺序的实用增量。计时分解是账目恒等式，非独立因果分解。',
        'sources': [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p)} for p in source_paths],
    }
    OUT.mkdir()
    for name, payload in [('review.json', result), ('claim_evidence_current.json', claims)]:
        (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    with (OUT / 'paired_diagnostics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(result_rows[0])); writer.writeheader(); writer.writerows(result_rows)
    text = f'''# GRID59后续证据整理：框架收益与学习增量

本文件只读现有结果，没有新增回放、训练、教师标签、参数或数据选择。GRID59原审计及冻结判据未改。

## 本轮完成结果

- GRID59正式16条：FULL、强STATIC及两种子均完成三周；六月均在1901步失败。两GNN种子均无新增提前失败，质量4/4通过，完成周最大成本变化+0.37200% FULL。
- GNN相对STATIC查询宏平均减少{macro_query:.3%}，两种子各只有2/4正向；低于5%且至少3/4正向的继续条件。
- 相对STATIC实际控制时间宏平均减少{macro_time:.3%}，即略慢。仅一次平衡计时，不能宣称精确的稳定速度退化。
- GRID59裁决WEAK_OR_MIXED_STOP_REVISION，停止这次先验改动，不继续改常数或挑种子。

## 仍保留什么

GRID55B原软压缩候选相对FULL控制时间宏平均减少{framework_speed:.3%}，两种子最坏完整周成本变化{framework_worst_cost:+.5%}，两种子均16/16完成。它保留了整体框架的工程价值。PPO和SoftGNN略快，版本选择依赖预设质量尺度，不能写成全面支配。

GRID57的STATIC已经通过8/8质量尺度，现有GNN成本相对它1好/6差/1平，平均高0.509个FULL归一化百分点。其查询宏平均少1.90%，但汇总多6.69%。STATIC是作者教师频次先验，并非任意或无经验顺序。GRID57没有同批公平速度主张。

因此，不能再把“相对FULL节省约70%查询、约15.75%控制时间”直接用作图消息传递的独立贡献。GRID56历史方法稿和贡献表须与此更新一起阅读；历史原件不覆盖。

## 现有分解能说明什么

paired_diagnostics.csv按日期和种子保存查询、控制时间、N1搜索时间、其余控制时间、图/先验/前向时间及动作/状态hash差异。控制差=搜索差+其余差逐项通过。

这只是实际账目分解。轨迹改变会同时影响其余控制，不把它解释成两个独立原因，不把全周差异都归因于模型推理开销。控制时间保留共同失败周的同长度比较；完整周成本只使用双方完成的三周。

## 下一步的研究边界

1. 当前审计、有限先验改动及结果归档已结束，无未结求解作业。
2. 若以整体有界拓扑搜索框架为论文主线，须重新核对其与已有方法的区别，把强频次基线放入主证据，GNN暂列可替换候选排序组件。现在不能把“GNN核心贡献已经完成”当作事实。
3. 若仍要求GNN作为核心贡献，需要另立具体的学习假设，重新冻结评价；当前结果不足以支持扩大RL、扩大数据或继续先验/尺度试探。
4. 这两条路线涉及论文目标选择；先把证据交给用户判断，再安排新的耗时研究。不自动恢复任何跟进或训练。

没有宣称GNN普遍无用，也没有宣称这份工程结果已经足以录用论文。主对照仍是FULL/作者PPO/SoftGNN，STATIC只补一项必要的强先验消融，不再扩对照菜单。
'''
    (OUT / 'GRID59_closeout.md').write_text(text, encoding='utf-8')
    (OUT / 'RUN_STATE.md').write_text('COMPLETE: offline evidence update; no running jobs or next experiment scheduled.\n', encoding='utf-8')
    ARCH.mkdir()
    receipts = []
    for p in [Path(__file__)] + list(OUT.iterdir()):
        dest = ARCH / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        assert sha(dest) == sha(p)
        receipts.append({'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size})
    (ARCH / 'manifest.json').write_text(json.dumps({'files': receipts}, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=True))


if __name__ == '__main__':
    main()
