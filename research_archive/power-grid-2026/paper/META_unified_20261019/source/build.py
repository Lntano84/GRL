"""Unify existing META configurations and verify saved decisions, without simulation."""
from pathlib import Path
from collections import Counter, defaultdict
import datetime as dt
import functools
import hashlib
import importlib.util
import json
import math
import os
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/meta_unified_20261019'
WORK = Path(__file__).parent
SOURCES = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(relative):
    path = ROOT / relative
    SOURCES[relative] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                      allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def source_dependencies():
    # Complete local policy/feature dependency chain, plus the explicit entry point.
    names = ['work/grid77/metacontrol.py', 'work/grid75/feature_gate.py',
             'work/grid72/feature_gate.py', 'work/grid64/metacontrol.py',
             'work/grid70/policies.py', 'work/grid67/policies.py',
             'work/grid63/policies.py', 'work/meta_unified_20261019/runtime.py']
    pending = names.copy()
    import re
    while pending:
        name = pending.pop()
        path = ROOT / name
        SOURCES[name] = sha(path)
        text = path.read_text(encoding='utf-8')
        for found in re.findall(r"work/[A-Za-z0-9_/]+\.py", text):
            if (ROOT / found).is_file() and found not in names:
                names.append(found)
                pending.append(found)
    return {name: SOURCES[name] for name in sorted(names)}


def normalize_row(batch, r):
    return dict(batch=batch, date=r['date'], method=r['method'], repeat=r['repeat'],
                steps=r['steps'], complete=r['complete'], quality_pass=r['quality_pass'],
                cost=r['cost'], cost_delta_full=r['cost_delta_full'], queries=r['queries'],
                controller_s=r['controller_s'],
                all_p95_s=r.get('decision_p95_s', r.get('all_p95')),
                search_p95_s=r.get('n1_decision_p95_s', r.get('search_p95')),
                same_horizon_full=r.get('equal_horizon_full', r.get('same_horizon_full')))


def summarize(batch, rows, methods, bindings):
    by_group = defaultdict(list)
    for row in rows:
        by_group[(row['method'], row['date'])].append(row)
    groups = {}
    for key, own in by_group.items():
        # Repeats are timings, not independent quality observations.
        assert len({(r['steps'], r['complete'], r['cost'], r['queries']) for r in own}) == 1
        groups[key] = dict(own[0], quality_pass=all(r['quality_pass'] for r in own),
                           controller_s=statistics.mean(r['controller_s'] for r in own),
                           all_p95_s=statistics.mean(r['all_p95_s'] for r in own),
                           search_p95_s=statistics.mean(r['search_p95_s'] for r in own
                                                       if r['search_p95_s'] is not None)
                           if any(r['search_p95_s'] is not None for r in own) else None)
    dates = sorted({r['date'] for r in rows})
    result = []
    for method in methods:
        own = [groups[(method, date)] for date in dates]
        time_changes, query_changes, p95_changes, search_p95_changes = [], [], [], []
        all_same_horizon = True
        for date in dates:
            a, b = groups[('FRAME_GUARD', date)], groups[(method, date)]
            equal = a['steps'] == b['steps'] and a['complete'] == b['complete']
            all_same_horizon &= equal
            time_changes.append(1 - a['controller_s'] / b['controller_s'])
            if equal and b['queries'] > 0:
                query_changes.append(1 - a['queries'] / b['queries'])
            p95_changes.append(1 - a['all_p95_s'] / b['all_p95_s'])
            if equal and a['search_p95_s'] is not None and b['search_p95_s']:
                search_p95_changes.append(1 - a['search_p95_s'] / b['search_p95_s'])
        complete_costs = [r['cost_delta_full'] for r in own if r['cost_delta_full'] is not None]
        result.append(dict(batch=batch, method=method, **bindings.get(method, {}),
                           dates=len(dates), repeats=len(by_group[(method, dates[0])]),
                           quality_dates=sum(r['quality_pass'] for r in own),
                           complete_dates=sum(r['complete'] for r in own),
                           worst_comparable_cost_delta_full=max(complete_costs) if complete_costs else None,
                           queries_total_per_repeat=sum(r['queries'] for r in own),
                           mean_controller_s=statistics.mean(r['controller_s'] for r in own),
                           frame_controller_reduction=statistics.mean(time_changes),
                           frame_query_reduction_matching_active_dates=statistics.mean(query_changes)
                           if query_changes else None,
                           matching_active_query_dates=len(query_changes),
                           frame_all_p95_reduction=statistics.mean(p95_changes),
                           frame_search_p95_reduction=statistics.mean(search_p95_changes)
                           if search_p95_changes else None,
                           equal_horizon_all_dates=all_same_horizon))
    return result


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    old = read('outputs/grid77/design.json')
    selected = read('outputs/grid77/selected.json')
    d84 = read('outputs/grid84/design.json')
    d89 = read('outputs/grid89_review/meta/design.json')
    train89 = read('outputs/grid89_review/meta_training/design.json')
    receipt89 = read('outputs/grid89_review/meta_training/training_finished.json')
    r84 = read('outputs/grid84/review.json')
    r89 = read('outputs/grid89_review/review.json')
    decision84 = read('outputs/grid84/decision.json')
    decision89 = read('outputs/grid89_review/decision.json')
    read('outputs/grid84/behavior.json')
    evidence = read('outputs/grid89_review/evidence/meta_interface_and_split.json')
    for design in [d84, train89]:
        for field in ['episodes', 'min_explore_steps', 'epsilon_anneal_steps', 'epsilon_final',
                      'lr', 'gamma', 'batch_size', 'replay_capacity', 'target_retention']:
            assert design[field] == old[field], field
    assert d84['weeks'] == d89['weeks'] and d84['timing'] == d89['timing']
    assert selected['cost_level'] == 0 and selected['calibration_quality_matched'] is False
    assert receipt89['costindex'] == 1 and receipt89['cost_level'] == .0025
    models = {}
    for variant, cost, index in [('C0', 0., 2), ('C0025', .0025, 1)]:
        for seed in range(5):
            folder = f'outputs/grid77/models/cost{index}_seed{seed}' if seed < 2 else (
                f'outputs/grid84/models/cost2_seed{seed}' if index == 2 else
                f'outputs/grid89_review/meta_training/models/cost1_seed{seed}')
            summary = read(folder + '/summary.json')
            checkpoint = folder + '/final.pt'
            SOURCES[checkpoint] = sha(ROOT / checkpoint)
            assert summary['final_sha256'] == SOURCES[checkpoint]
            assert summary['seed'] == seed and summary['cost_level'] == cost
            assert summary['beta'] == math.log1p(cost) and summary['updates'] > 0
            curve = read(folder + '/curve.json')
            assert curve[-1]['episode'] == 30000
            models[f'{variant}_S{seed}'] = dict(variant=variant, seed=seed, cost_level=cost,
                beta=summary['beta'], checkpoint=checkpoint, sha256=SOURCES[checkpoint],
                summary=folder+'/summary.json', curve=folder+'/curve.json',
                episodes=30000, updates=summary['updates'],
                calibration_mean_regret=summary['calibration_mean_regret'],
                calibration_max_regret=summary['calibration_max_regret'])
    dependencies = source_dependencies()
    cfg = dict(schema_version=1, name='META_CONTEXT_V1', created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        purpose='Explicit mapping of existing configurations, not a new preregistered experiment.',
        status='HISTORICAL_CONFIGURATION_UNIFICATION', learning_core='ICAPS2022-style DoubleDQN grid adaptation',
        original_source_commit=old['source_commit'], historical_primary='C0',
        primary_reason='Retain the pre-evaluation GRID77 calibration choice; no test-based reselection.',
        variants={'C0': dict(cost_level=0., beta=0., role='historical primary; no explicit query cost in utility'),
                  'C0025': dict(cost_level=.0025, beta=math.log1p(.0025), role='fixed nonzero-cost sensitivity')},
        seeds=list(range(5)), selection=selected, features=evidence['features'],
        network=dict(input=25, hidden=[64,32], output=3, activation='ReLU'),
        actions=dict(RETURN='Only if an admissible incumbent exists', GLOBAL64='Reveal next min(64, remaining)',
                     DELEGATE_FRAME='Initial decision only; full frozen FRAME_GUARD with cached prefix reused'),
        common=dict(initial_queries=128, batch_queries=64, pool='eligible deduplicated representatives',
                    order='author frequency descending, retained through representative IDs',
                    admissible='0 < predicted_rho < min(root_rho, controller_threshold), no done/exception/illegal/ambiguous',
                    ranking='float32 predicted reward descending, lower representative ID breaks ties',
                    controller='GRID28 fast controller with common DC/safeQP caches',
                    frame_reference='work/grid70/policies.py:FRAME_GUARD',
                    context='same root operating context computed in both FRAME and META adapters',
                    environment='work/grid80/fresh_env', environment_seed=0,
                    dates=d84['weeks'], torch_threads=1, benchmark_processes='serial fresh worker per trajectory'),
        training={**{k:old[k] for k in ['episodes','lr','gamma','batch_size','replay_capacity',
                    'target_retention','min_explore_steps','epsilon_anneal_steps','epsilon_final']},
                  'network_initialization':old['initialization'], 'double_q_target':old['double_q_target'],
                  'deploy':old['deploy'], 'records':'outputs/grid77/records.json',
                  'train_records':129, 'calibration_records':22,
                  'train_families':old['training_families'], 'calibration_families':old['calibration_families'],
                  'checkpoint_rule':'final at 30000 episodes; all seeds retained'},
        utility=dict(quality='clip(best admissible predicted reward / 2, 0, 1), or 0 without incumbent',
                     value='q - expm1(log1p(cost_level) * queries/M)',
                     transition='U_next - U_current; RETURN reward 0',
                     cost_scope='training checkpoint identity, not a runtime switch; zero cost has no explicit compute penalty'),
        evaluation=dict(quality_tolerance=.005, quality_reference='GRID81 FULL for survival/cost only',
                        timing=d84['timing'], primary_batch='GRID84', primary_repeats=2,
                        sensitivity_batch='GRID89/meta', sensitivity_repeats=1,
                        aggregate='average timings within date, then date-macro relative reduction, then arithmetic mean over all five seeds',
                        forbid='No cross-batch paired timing; no early-failure speed claim; no seed or test-cost selection'),
        models=models, runtime_source_hashes=dependencies,
        limits=['Includes FRAME macro action; not an independent topology controller.',
                'Original calibration quality match failed and remains false.',
                'Source A*/RRT* tasks are not reproduced.',
                'Saved decision re-execution does not create new AC simulations or timing evidence.'])
    save('configuration.json', cfg)
    bindings84 = {f'META_CONTEXT{s}':dict(variant='C0',seed=s,checkpoint=models[f'C0_S{s}']['checkpoint']) for s in range(5)}
    bindings89 = {f'META_COST{c}_SEED{s}':dict(variant='C0' if c==0 else 'C0025',seed=s,
                    checkpoint=models[f'{"C0" if c==0 else "C0025"}_S{s}']['checkpoint']) for c in [0,1] for s in range(5)}
    rows84 = [normalize_row('GRID84', r) for r in r84['rows']]
    rows89 = [normalize_row('GRID89_META', r) for r in r89['rows'] if r['phase']=='meta']
    summary = summarize('GRID84', rows84, [m['method'] for m in d84['methods']], bindings84)
    summary += summarize('GRID89_META', rows89, [m['method'] for m in d89['methods']], bindings89)
    save('results_by_configuration.json', dict(rows=rows84+rows89, summary=summary))
    # Check each past manifest against the explicit model binding and saved metadata/summary.
    run_count=0
    for batch, design, binding, base in [('GRID84', d84, bindings84, 'outputs/grid84'),
                                       ('GRID89_META', d89, bindings89, 'outputs/grid89_review/meta')]:
        for item in design['execution_order']:
            folder=f"{base}/runs/r{item['repeat']}__{item['week']}__{item['method']}"
            meta=read(folder+'/metadata.json'); stored=read(folder+'/summary.json')
            assert meta['timing']==d84['timing']
            for k in ['rule','budget','cache_dc','cache_safe','week','repeat','method','policy']:
                assert item[k]==meta[k]==stored[k], (folder,k)
            assert meta['torch_threads']==1
            if item['method']!='FRAME_GUARD':
                b=binding[item['method']]
                assert item['meta_model']==meta['meta_model']==stored['meta_model']==b['checkpoint']
                assert (models[f"{b['variant']}_S{b['seed']}"]['sha256']==sha(ROOT/b['checkpoint']))
            run_count+=1
    # Real saved search calls: reproduce exact query order, chosen action and learned states.
    import numpy as np
    import torch
    torch.set_num_threads(1)
    U=module('meta_unified_runtime_check',WORK/'runtime.py')
    subs=read('outputs/grid28/offline.json')['public_subids']
    call_count=0; paid_query_records=0; behavior=defaultdict(Counter)
    for batch, design, binding, base in [('GRID84',d84,bindings84,'outputs/grid84'),
                                       ('GRID89_META',d89,bindings89,'outputs/grid89_review/meta')]:
        for item in design['execution_order']:
            if item['method']=='FRAME_GUARD':continue
            folder=ROOT/base/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}"
            b=binding[item['method']]
            for path in sorted(folder.glob('call[0-9][0-9][0-9].json')):
                c=read(path.relative_to(ROOT).as_posix())
                inputs=path.with_name(path.stem+'_inputs.npz')
                SOURCES[inputs.relative_to(ROOT).as_posix()]=sha(inputs)
                with np.load(inputs) as z:context={k:z[k] for k in ['x','edges','glob']}
                truth={r['pool_id']:r for r in c['outcomes']};seen=[]
                def reveal(i,phase):
                    assert i in truth and i not in seen, (str(path),i)
                    assert truth[i]['phase']==phase
                    seen.append(i)
                    return truth[i]
                got=U.run(c['order'],reveal,subs,{int(k):int(v) for k,v in c['aliases'].items()},
                          limit=c['rho_limit'],context=context,variant=b['variant'],seed=b['seed'])
                for key in ['chosen_pool_id','public_queries','stop_reason','local_queries',
                            'completion_budget','extra_selected_ids','meta_events']:
                    assert got[key]==c[key], (str(path),key)
                assert seen==[r['pool_id'] for r in c['outcomes']]
                for event in got['meta_events']:
                    state=event['state'];action=event['action']
                    assert len(state)==25 and all(math.isfinite(x) for x in state)
                    assert action!=0 or state[2]>=.5
                    assert action!=2 or state[24]>=.5
                if c['chosen_pool_id'] is not None:
                    best=max((r for r in c['outcomes'] if r['strict_admissible']),key=U.core().key)
                    assert best['pool_id']==c['chosen_pool_id']
                if got['meta_events']:
                    behavior[(batch,item['method'])]['search_calls_with_decision']+=1
                    behavior[(batch,item['method'])]['delegate_first']+=int(got['meta_events'][0]['action']==2)
                behavior[(batch,item['method'])]['search_calls']+=1
                call_count+=1;paid_query_records+=len(seen)
    # Feature legality/split inspection uses historical records only, no simulator is imported.
    records=read('outputs/grid77/records.json')
    counts=Counter(r['split'] for r in records)
    assert counts=={'train':129,'calibration':22}
    families={s:{r['family'] for r in records if r['split']==s} for s in counts}
    assert not families['train'] & families['calibration']
    assert families['train']==set(old['training_families'])
    assert families['calibration']==set(old['calibration_families'])
    assert not (families['train']|families['calibration']) & {w.rsplit('_',1)[0] for w in d84['weeks']}
    frozen_maps=[read(p) for p in ['outputs/grid84/code_freeze.json',
                                  'outputs/grid89_review/code_freeze.json',
                                  'outputs/grid89_review/meta_training/training_code_freeze.json']]
    freeze_checked=0
    for relative, expected in {**dependencies,
            **{m['checkpoint']:m['sha256'] for m in models.values()},
            'outputs/grid77/records.json':SOURCES['outputs/grid77/records.json']}.items():
        if relative=='work/meta_unified_20261019/runtime.py':
            continue
        matches=[frozen[relative] for frozen in frozen_maps if relative in frozen]
        assert matches and all(value==expected for value in matches), relative
        freeze_checked+=1
    # Verify recomputed numbers against the frozen reports rather than inventing a pooled score.
    primary=[r for r in summary if r['batch']=='GRID84' and r['method']!='FRAME_GUARD']
    mean_primary=statistics.mean(r['frame_controller_reduction'] for r in primary)
    assert abs(mean_primary-decision84['overall']['mean_seed_macro_controller_reduction'])<1e-12
    for row in summary:
        if row['batch']=='GRID89_META' and row['method']!='FRAME_GUARD':
            ref=next(c for c in decision89['comparisons'] if c['phase']=='meta' and c['baseline']==row['method'])
            assert abs(row['frame_controller_reduction']-ref['macro_controller_reduction'])<1e-12
    save('behavior.json',[dict(batch=b,method=m,**v,
                               delegate_share=v['delegate_first']/v['search_calls_with_decision']
                               if v['search_calls_with_decision'] else None)
                          for (b,m),v in behavior.items()])
    # Correct inherited descriptive labels in the new contract; never rewrite frozen history.
    issues=[dict(id='INHERITED_POSITIVE_TRAINING_LABELS',historical_file='outputs/grid89_review/meta_training/design.json',
                 observed='Inherited stage GRID84, selected_cost2, zero-cost method list and two repeats remain in a positive-cost training design.',
                 effective='train_positive.py and training_finished.json specify costindex1=.0025, seeds2/3/4; evaluation meta/design.json specifies one repeat.',
                 resolution='Separate immutable training recipe, checkpoint bindings and evaluation batches in configuration.json.'),
            dict(id='STALE_STOPPING_ONLY_DESCRIPTION',historical_file='outputs/grid77/design.json:source',
                 observed='An inherited source sentence says stopping-only while source_mapping/actions and executed code include DELEGATE_FRAME.',
                 resolution='Use explicit 25-feature, three-action META_CONTEXT_V1 naming; exclude earlier 12-feature two-action results from its primary mean.'),
            dict(id='COST_INDEX_ALIAS',observed='Historical cost2 means coefficient0; META_COST0 also means coefficient0.',
                 resolution='C0/C0025 identifiers record the actual coefficient and exact checkpoint SHA256.'),
            dict(id='NO_NEW_COMMON_TIMING_BATCH',observed='GRID84 has two repetitions; GRID89_META has one. They cannot become one common three-repeat timing result.',
                 resolution='Retain two result blocks, pair FRAME within each batch only; additional balanced repetitions would be new experiments.'),
            dict(id='CALIBRATION_LIMIT',observed='Original shared calibration quality match failed.',
                 resolution='Keep false; do not convert engineering verification or test quality6/6 into calibration success.')]
    save('configuration_issues.json',issues)
    validation=dict(passed=True,checkpoint_hashes=10,historical_runs_checked=run_count,
        bindings_matched_original_freezes=freeze_checked,
        saved_search_calls_reexecuted=call_count,saved_query_records_checked=paid_query_records,
        feature_dimensions=25,training_records=counts['train'],calibration_records=counts['calibration'],
        split_families_disjoint=True,existing_result_arithmetic_matched=True,
        new_training=0,new_simulator_calls=0,new_closed_loop_runs=0,
        historical_sources_unchanged=True,independent_ac_resolve=False,
        limitation='Policy execution from saved feedback only; not a new end-to-end simulation or timing qualification.')
    # All source hashes must still match after the operation.
    assert all(sha(ROOT/p)==h for p,h in SOURCES.items())
    save('validation.json',validation)
    save('source_receipts.json',SOURCES)
    write_report(cfg,summary,validation,mean_primary,issues)
    print(json.dumps(validation,ensure_ascii=False),flush=True)


def write_report(cfg,summary,validation,mean_primary,issues):
    lines=['# META 统一配置与结果对应说明','',
      '本次统一已有学习方法的命名、权重、费用、执行接口和结果出处。未训练、未调用新仿真、未重跑闭环。历史冻结文件和论文原稿保留。','',
      '## 一、统一的方法定义','',
      '论文统一称为“META：ICAPS 2022 元推理方法的电网适配版”。具体实现名为 META_CONTEXT_V1，采用25维可见搜索/电网上下文、25–64–32–3 ReLU网络及原DoubleDQN式训练内核。',
      '动作固定为：返回当前最佳可用动作；沿共有顺序继续查询64个候选；仅首次决策可委托完整FRAME_GUARD，复用已查询前缀。它包含FRAME子程序。早期12维、两动作的停止型META另作历史版本，不混入主比较。','',
      '## 二、统一配置','',
      '| 项目 | 固定定义 |','|---|---|',
      '| 初始与批次 | 前128个代表候选；后续每批64个；候选不足时取剩余全部 |',
      '| 候选顺序 | 原作者教师频次顺序，经筛选和去重后保留；STATIC_ID不是随机编号优先级 |',
      '| 动作可接纳与返回规则 | 相同合法性/预测负载率限制；float32奖励最大，低ID破并列 |',
      '| 基础控制器和缓存 | 原GRID28快控制、DC/safeQP缓存；两臂都构造相同公开根状态上下文 |',
      '| 网络及训练 | 25–64–32–3 ReLU；30000回合；Adam学习率0.0001；折扣0.99；批量256；回放容量1000000；目标保留0.995 |',
      '| 训练/校准 | 129/22份记录；日期家族分离；全部五种子0–4；固定最终权重 |',
      '| 费用命名 | C0表示实际费用系数0；C0025表示0.0025；路径中的cost1/cost2不作为论文名称 |',
      '| 主比较 | GRID84：C0五种子、六场景、两次反向计时 |',
      '| 费用补充 | GRID89/meta：C0和C0025各五种子、同六场景、一次计时 |',
      '| 时间 | 同批完整控制decision_s；包含特征、首次模型加载/推理、仿真和连续优化；初始化、物理推进和记录另计 |','',
      '效用为 U=q−[exp(log(1+c)·n/M)−1]，q为最佳可接纳候选的裁剪归一化预测奖励，n/M为已查询比例，c为训练费用系数。转移奖励为效用增量，返回奖励为0。C0没有显式查询费用惩罚，不能写成非零计算费用训练；改变配置中的费用数字不能改变已训练权重。','',
      '历史主配置继续采用原校准阶段选定的C0。原三费用校准未达到质量匹配，calibration_quality_matched=false原样保留；此处不重新选择费用，也不将后来的测试结果用于选费用。C0025单列说明非零费用下的质量/计算权衡。','',
      '## 三、每个结果明确对应一套配置','',
      '正号表示FRAME相对对应META减少时间或查询；负号表示FRAME更多。先平均同日期计时，再平均逐日期相对变化；五种子最后等权平均。质量达标数与完整周完成数分别统计。查询比例只使用两方法时域相同且META查询非零的日期。','',
      '| 批次 | 费用 | 种子 | 质量 | 完整周 | META查询合计/重复 | META控制秒/日期 | FRAME控制减少 | FRAME查询减少 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['method']=='FRAME_GUARD':continue
        q=r['frame_query_reduction_matching_active_dates']
        lines.append(f"| {r['batch']}（{r['repeats']}重复） | {cfg['variants'][r['variant']]['cost_level']:g} | {r['seed']} | {r['quality_dates']}/6 | {r['complete_dates']}/6 | {r['queries_total_per_repeat']} | {r['mean_controller_s']:.3f} | {100*r['frame_controller_reduction']:+.3f}% | {'不适用' if q is None else f'{100*q:+.3f}%'} |")
    lines += ['',f'GRID84原主比较：五种子完整控制时间宏平均减少{100*mean_primary:.3f}%，与原2.23%对应。全部五种子质量6/6；原加强五种子提速判据未通过，不改写成显著或普遍优势。',
              'GRID89/meta的一次计时为费用复查，不能与GRID84拼成三次重复。两个批次均只与各自同期FRAME计时配对；不混入旧FULL时间。C0025种子0和4质量5/6，其余种子6/6，失败结果保留。','',
              '## 四、已消除的配置歧义','']
    lines += [
      '- 正费用训练目录继承旧标签：原design.json仍含旧阶段名、零费用方法表、selected_cost=2和两重复描述。实际训练脚本及收据为费用0.0025，评价方案为一次重复；新清单分别记录这三层信息。',
      '- 早期“仅停止”描述残留：当前实际执行包含FRAME委托，统一使用25维、三动作的META_CONTEXT_V1名称。12维、两动作的早期结果单列。',
      '- 费用索引含义混乱：历史cost2和META_COST0都表示实际系数0。统一名称C0/C0025直接对应系数及准确权重。',
      '- 两次计时与一次复查分开：不拼成三次重复，不跨批配对FRAME时间。',
      '- 校准未达质量匹配：保留原False，不把工程核对通过或评价质量达标改写为校准成功。']
    lines += ['','最主要的实际问题是：正费用训练目录的design.json继承了旧批次的阶段名、零费用方法列表、selected_cost及重复次数；真正训练脚本和训练收据记录的是0.0025。新清单将训练配方、权重和评价批次拆开，旧冻结文件不改。','',
              '## 五、核对范围与可执行入口','',
              f"十份权重SHA256和{validation['historical_runs_checked']}条历史运行元数据通过；统一入口重执行{validation['saved_search_calls_reexecuted']}次已有搜索调用、核对{validation['saved_query_records_checked']}条已保存查询反馈。查询顺序、选择动作、停止原因、25维状态和网络动作与原日志一致。训练/校准家族分离及主结果算术一致。",
              '这是已保存数据的一致性核对，没有新AC求解或闭环计时。模型已完成固定训练，不据此声称训练已证明收敛。',
              '统一调用入口：work/meta_unified_20261019/runtime.py。每次调用显式给出variant=C0或C0025、seed=0..4；入口按SHA256绑定准确权重和源码，不再隐式读取selected.json来决定模型。调用者提供共同observe接口，入口本身不启动仿真。',
              '配置：configuration.json；逐运行和逐配置指标：results_by_configuration.json；行为：behavior.json；问题记录：configuration_issues.json；核验：validation.json；来源：source_receipts.json。','',
              '## 六、论文可直接使用的表述','',
              '我们采用基于Bhatia等人ICAPS 2022元推理方法构建的电网适配对照META。该方法使用25维搜索与运行状态特征，在返回当前动作、继续评估64个候选以及首次决策时委托FRAME之间选择。网络结构为25–64–32–3，所有种子使用相同的129条训练记录、22条校准记录与30000回合训练。主比较沿用校准阶段选定的零费用配置，报告全部五个训练种子；非零费用0.0025的结果单列为敏感性分析。两方法采用相同候选顺序、基础控制器、仿真接口和公共缓存，完整控制时间在各批次内配对比较。',
              '在六个已暴露生成场景、两次反向计时的主比较中，FRAME与五个META种子均达到全部场景的预设质量标准，FRAME的完整控制时间在五种子间平均减少2.23%。该结果表明固定反馈规则具有竞争力；META适配包含FRAME委托，当前结果不构成原论文任务的完整复现。','',
              '## 七、后续约束','',
              '今后的META结果均使用本清单中的特征、动作和准确权重标识。若更换费用、去掉FRAME委托、调整状态或训练参数，需另立版本。若需要将C0/C0025做成同一套多重复比较，必须另做完整同批计时；当前不补造结果，也不自动启动新实验。']
    (OUT/'META_统一配置说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    main()
