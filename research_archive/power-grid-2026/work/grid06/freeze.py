"""Seal a fixed rule on the pre-existing calendar/group validation split."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid06'
load=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not (OUT/'design_freeze.json').exists()
assert load(ROOT/'outputs/grid05/GRID05_postrun_review.json')['passed']
splitfile=ROOT/'outputs/grid03/scenario_split_frozen.json';split=load(splitfile)
assert sha(splitfile)==(splitfile.with_suffix('.sha256')).read_text().strip()
months=[1,4,7,10]
selected=[s for s in split['extracted_subsets']['validation'] if int(s[5:7]) in months]
assert len(selected)==4 and [int(s[5:7]) for s in selected]==months
assert not set(selected)&set(split['extracted_subsets']['test'])
history=[]
for p in sorted((ROOT/'outputs').glob('grid*/run_manifest*.json')):
    for r in load(p):
        if isinstance(r,dict) and isinstance(r.get('scenario'),str): history.append(r['scenario'])
assert not set(history)&set(selected), 'Do not use previously evaluated weeks'
orders=[['NN20','FALLBACK90','NN352'],['FALLBACK90','NN352','NN20'],['NN352','NN20','FALLBACK90'],['NN352','FALLBACK90','NN20']]
plan=[dict(scenario=s,policy=p) for s,order in zip(selected,orders) for p in order]
reused=load(ROOT/'outputs/grid05/reused_assets_frozen.json')
for p,h in reused.items():assert sha(ROOT/p)==h,p
for name in ['work/grid05/fallback.py','work/grid05/run_one.py','outputs/grid05/GRID05_summary.json',
    'outputs/grid05/GRID05_audit.json','outputs/grid05/GRID05_postrun_review.json','outputs/grid05/preflight.json',
    'outputs/grid05/execution_code_frozen.json','outputs/grid03/scenario_split_frozen.json',
    'outputs/grid03/extraction_manifest.json','outputs/grid03/official_initialization_manifest.json']:
    reused[name]=sha(ROOT/name)
design=dict(stage='GRID06',environment_seed=0,data_root='work/grid03/formal_env_initialized',
    split='validation',selected_weeks=selected,selection='Pre-existing GRID03 monthly validation subset; take January, April, July, October. Naming only; no outcome/initial-load selection.',
    policies=['NN20','FALLBACK90','NN352'],plan=plan,initial_top_k=20,expanded_top_k=352,
    rule='Exact frozen GRID05 fallback.py reused; no candidate or selected forecast max rho >= original rho_safe=0.9 expands to remaining332. Cache first20 original float32 rewards; preserve full neural tie order and all downstream modules.',
    quality_screen=dict(no_earlier_end_than_NN20_any_week=True,maximum_complete_week_relative_cost_regression=.02,
        meaningful_benefit_vs_NN20='A new full native completion, or >=5% lower raw cost on a jointly complete week; both-incomplete end-step gains are reported but do not meet this benefit screen.',
        meaningful_benefit_weeks_required=2,
        full_search_quality_retention='No earlier end than concurrent NN352 on any week; no >2% raw cost regression on jointly complete pairs.'),
    verdict_order=['Any earlier end or >2% complete-week cost regression vs NN20: DO_NOT_PROMOTE_RULE_AS_GENERAL_IMPROVEMENT.',
        'Otherwise, >=2 meaningful benefit weeks vs NN20 and full-search quality retention: VALIDATION_QUALITY_SCREEN_SUPPORTED.',
        'Otherwise, >=2 meaningful benefit weeks but NN352 retention fails: BENEFIT_VS_NN20_WITH_FULLSEARCH_QUALITY_TRADEOFF.',
        'Otherwise, zero meaningful benefits: NO_PRACTICAL_INCREMENTAL_BENEFIT_CONFIRMED.',
        'Otherwise: LIMITED_SIGNAL_NOT_CONFIRMED.'],
    cost_order='Completion/end-step first; no cost ranking of incomplete episodes. Raw costs only; competition normalized score not certified.',
    compute_order='Record all simulations and end-to-end worker wall; compare equal-prior-state prefixes. Do not infer speed from incomparable different-length trajectories.',
    qualification='Pure file/hash/AST checks only. Reuse unchanged GRID05 zero-physical preflight, including its source full-greedy equality test. No new preflight simulations or physical steps.',
    caps=dict(main_runs=12,physical_steps=24300,per_run_after_import_s=900,import_s=180,total_runner_including_qualification_s=2700,output_bytes=2147483648),
    restrictions=['No training or model changes','No threshold/seed/pool tuning','No replacement weeks or automated retries','Other validation and all test outcomes remain sealed',
        'No automatic next-stage or GRL branch'],
    limitations=['Four previously unrun validation weeks, one environment seed: not a population guarantee.',
        'Suffix groups are a partition convention, not proven statistical independence.',
        'Pretrained LJN training identities unknown: cannot certify these weeks unseen by the released model.',
        'Synthetic public France2035 modified IEEE118 benchmark, not RTE production logs.',
        'Existing author neural shortlist plus physics search/continuous control is not our novel architecture.'])
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'design_freeze.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
(OUT/'design_freeze.sha256').write_text(sha(OUT/'design_freeze.json')+'\n',encoding='ascii')
(OUT/'reused_assets_frozen.json').write_text(json.dumps(reused,indent=2),encoding='utf-8')
old=(ROOT/'work/grid05/run_one.py').read_text(encoding='utf-8');new=(ROOT/'work/grid06/run_one.py').read_text(encoding='utf-8')
(OUT/'worker_derivation.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='GRID05 frozen worker',tofile='GRID06 validation worker')),encoding='utf-8')
(OUT/'GRID06_protocol.md').write_text('''# GRID06 固定回退规则的封存验证

原GRID03划分中一、四、七、十月各一个验证周。只按月份和预先抽取清单选择，先冻结，后运行。此前未由我们评估，不代表作者发布模型未见。其他8个验证周及全部12个测试周继续封存。

NN20、原样复用的FALLBACK90、同库NN352三臂，种子0，共12条交错轨迹。连续优化、恢复、重连、N−1动作库及原生规则不改。没有模型/阈值/种子搜索；复用GRID05已通过的语义/真实零推进预检，新资格检查只读文件和源码，不产生新仿真。

先看结束步和完整运行，不排名提前结束周的成本。回退若任一周比NN20更早结束，或任一共同完整周成本劣于NN20超过2%，不推广为普遍改善。否则至少2/4周出现实际增益（新增完整运行，或共同完整周成本降低至少5%），并对同时运行的NN352无更早结束、无共同完整周>2%成本退化，才通过验证质量筛查。相对NN20通过而相对NN352失败，单列质量权衡；零实际增益记未确认；只有一周记有限信号。严格使用原始逐周数据，不调门槛/换周挽救结果。筛查不等同统计显著性或等价检验。

完整记录模拟数、动作时间、求解状态与全部交付向量。同状态前缀比较开销；不把不同长度整周墙钟解释为加速。此阶段不设必须加速的附加筛查；质量与开销分别报告。原生五分钟步长不是已认证的生产计算时限。

包络：12条轨迹、24300次实际推进、每条导入后900秒、含资格检查总2700秒、结果2GiB。错误/超限保留并停止，不自动重跑。计量沿用已审计原始成本与物理账本；不报未认证的归一化比赛分数。不训练，不自动解封其余数据或启动下一阶段。
''',encoding='utf-8')
print(json.dumps(dict(passed=True,selected=selected,runs=len(plan),design_sha256=sha(OUT/'design_freeze.json')),indent=2))
