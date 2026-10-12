"""Freeze compressed-candidate confirmation using unaccessed date metadata only."""
import hashlib, json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'outputs/grid52'
SELECTION_SEED = 'GRID52_FRESH_FAMILY_CONFIRMATION_20261012_V1'

def read(p):
    return json.loads(p.read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def selection_key(value):
    return hashlib.sha256((SELECTION_SEED + '|' + value).encode()).hexdigest()

assert not OUT.exists(), 'Never replace an existing freeze or outcome.'
assert read(ROOT / 'outputs/grid51/review.json')['passed_saved_data']
assert read(ROOT / 'outputs/grid50/audit_fit.json')['passed']
old = read(ROOT / 'outputs/grid44/design.json')
train = read(ROOT / 'outputs/grid21/design.json')
exposure = read(ROOT / 'outputs/grid_research_audit/split_exposure.json')
extracted = sorted(p.name for p in (ROOT / 'work/grid03/formal_env_initialized/chronics').iterdir() if p.is_dir())
excluded = set(exposure['exposed_dates'])
excluded.update(w[:10] for w in train['training_weeks'] + train['development_weeks'])
excluded.update(w[:10] for w in old['weeks'])
excluded.update(w[:10] for w in extracted)
index = read(ROOT / 'outputs/grid03/archive_index.json')
bydate = {}
for r in index['entries']:
    parts = PurePosixPath(r['name']).parts
    if len(parts) >= 3 and parts[1] == 'chronics' and parts[2][:4] == '2035':
        episode = parts[2]
        bydate.setdefault(episode[:10], set()).add(episode)
available = {q: sorted(d for d in bydate if d not in excluded and (int(d[5:7])-1)//3+1 == q) for q in range(1, 5)}
families = []
for quarter, dates in available.items():
    assert len(dates) >= 2, (quarter, dates)
    for date in sorted(dates, key=selection_key)[:2]:
        variants = sorted(bydate[date], key=lambda s: int(s.split('_')[1]))
        assert len(variants) == 16
        families.append({'quarter': quarter, 'date': date,
                         'primary_episode': min(variants, key=selection_key),
                         'reserved_all_variants': variants})
weeks = [f['primary_episode'] for f in families]
assert len(weeks) == len(set(weeks)) == 8
assert not set(w[:10] for w in weeks) & excluded
methods = [dict(m) for m in old['methods'] if m['method'] in ['FULL', 'SOFTGNN0', 'SOFTGNN1', 'LOCAL_PPO']]
for seed in [0, 1]:
    methods.append({'method': f'COMPRESSED_GNN{seed}', 'rule': 'COMPRESSEDGNN', 'budget': 128,
                    'checkpoint': f'outputs/grid50/models/gnn_seed{seed}/final.pt',
                    'policy': 'PATIENCE64', 'cache_dc': True, 'cache_safe': True})
assert len(methods) == 6
order = []
for repeat in [0, 1]:
    for n, week in enumerate(weeks):
        offset = (n + 3*repeat) % len(methods)
        sequence = methods[offset:] + methods[:offset]
        if repeat:
            sequence = list(reversed(sequence))
        order.extend({**m, 'week': week, 'repeat': repeat, 'stage_phase': 'pilot'} for m in sequence)
reservation = {'status': 'FROZEN_BEFORE_EXTRACTION_AND_EVALUATION', 'metadata_only': True,
               'selection_seed': SELECTION_SEED, 'selection_rule': 'Two dates per quarter by sha256(seed|date); primary variant by sha256(seed|episode). No scores, environments, or inference.',
               'available_dates_by_quarter': available, 'excluded_dates': sorted(excluded),
               'existing_extracted_episode_names': extracted, 'date_families': families,
               'limitations': 'One grid; only two remaining unaccessed dates in quarters1/3, so those choices are forced. Three unused date families remain after this freeze. Author PPO original training exposure unknown.',
               'all_variants_training_exclusion': True}
D = {**old, 'stage': SELECTION_SEED, 'weeks': weeks, 'families': families, 'methods': methods,
     'execution_order': order,
     'choice_before_access': 'GRID50 soft-compressed scoring candidate, both frozen seeds, original GRID28 fast PATIENCE64/local completion. Original GRID21 weights remain preserved. GRID47 strict completion and GRID43 tie completion not adopted. No new fits or teachers.',
     'scope': 'Second fresh eight-date-family confirmation:6implementations,3baseline families(FULL,publicPPO,recentSoftGNN2seeds),candidate2seeds. No outcome-dependent date/model choice. GRID44 is already exposed and is excluded here.',
     'statistics': 'Fixed10000 date-family bootstrap resamples,seed20261011; same estimator as GRID44. Average two timing repeats and report both model seeds plus their equal-weight family average. Runs are not independent grids.',
     'sealed_use': 'All8date families/all16variants excluded from subsequent training/tuning. Once evaluated these are exposed confirmation. No retuning on these results or claiming old GRID44 April outlier repaired.',
     'model_status': 'Candidate qualified on old development only; this confirmation cannot be used to choose the better seed. All outcomes retained. No automatic default-model change.',
     'caps': {'episode_s':1500, 'wall_s':18000, 'physical_steps':200000,
              'public_forecasts':2000000, 'new_fits':0, 'reserved_evaluations':96}}

for name in ['worker.py', 'extract.py', 'preflight.py', 'audit.py', 'run.py']:
    source = (ROOT / 'work/grid44' / name).read_text(encoding='utf-8')
    source = source.replace('grid44', 'grid52').replace('GRID44', 'GRID52')
    if name == 'run.py':
        target = '    install_search_adapter(engine.Search)'
        assert source.count(target) == 1
        source = source.replace(target, target + "\n    sys.path.insert(0,str(ROOT/'work/grid50'))\n    from online_adapter import install_free_adapter\n    install_free_adapter(engine.Search)")
    if name == 'audit.py':
        source = source.replace('LOCAL_GNN0', 'COMPRESSED_GNN0').replace('LOCAL_GNN1', 'COMPRESSED_GNN1')
        source = source.replace('First reserved date-family confirmation', 'Fresh second date-family confirmation of soft-compressed candidate')
        source = source.replace('两原模型种子', '两软压缩模型种子')
    (HERE / name).write_text(source, encoding='utf-8')

OUT.mkdir()
(OUT / 'reservation.json').write_text(json.dumps(reservation, indent=2), encoding='utf-8')
D['reservation_sha256'] = sha(OUT / 'reservation.json')
(OUT / 'reservation.sha256').write_text(D['reservation_sha256']+'\n', encoding='utf-8')
(OUT / 'design.json').write_text(json.dumps(D, indent=2), encoding='utf-8')
files = list(HERE.glob('*.py')) + [OUT / 'reservation.json', OUT / 'design.json',
        ROOT / 'outputs/grid50/audit_fit.json', ROOT / 'outputs/grid51/review.json',
        ROOT / 'work/grid50/compressed_residual.py', ROOT / 'work/grid50/online_adapter.py']
files += [ROOT / p for p in read(ROOT / 'outputs/grid44/code_freeze.json')]
files += [ROOT / m['checkpoint'] for m in methods if m['checkpoint']]
(OUT / 'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix(): sha(p) for p in dict.fromkeys(files)}, indent=2), encoding='utf-8')
(OUT / 'RUN_STATE.md').write_text('FROZEN BEFORE EXTRACTION:96fresh serial confirmation runs;candidate2seeds;three baseline families. No fits.\n', encoding='utf-8')
(OUT / 'PAPER_EVIDENCE_PLAN.md').write_text('''# 当前论文证据链与下一步

核心主张：以图评分辅助既有电网控制器的拓扑候选搜索，保持闭环质量并降低实际控制时间。

已完成：原版八日期确认；一版软压缩训练和四个旧开发日期配对；共有缓存公平提供给全部方法；原版、失败版本与权重保存。

尚未完成：新版封存日期同批FULL/PPO/SoftGNN比较；最终版的小规模PPO先验/图评分归因消融；质量—速度关系汇总；训练/教师成本、完整方法说明与可运行复现材料。

GRID51新版对FULL控制时间减少14.3%–15.1%、查询减少66%；三个完成周成本相同，六月共同失败保留。此为旧开发数据，不能替代独立确认。GRID44原版对PPO快约2.5%–3.2%，与SoftGNN各有胜负；不能转用为新版已经胜过它们。

本轮只执行固定新版与三类主对照的96条新日期确认，不增加模型/基线或以新评价选择种子。SoftGNN为明确资源匹配适配；作者PPO训练范围未知；单电网日期泛化不等于跨电网或生产验证。GNN+RL组合本身不作为新意，当前监督GNN排序与已有PPO的角色须如实定义。
''', encoding='utf-8')
print(json.dumps({'frozen': True, 'runs': 96, 'weeks': weeks, 'available_by_quarter': available, 'new_fits': 0}))
