"""All exposed-week paths, known terminal failure and component conservation."""
import gzip,hashlib,json,math,os,csv
from pathlib import Path
from collections import Counter,defaultdict
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid32'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
F=load(OUT/'finished.json');D=load(OUT/'design.json');assert F['passed_engineering']
for name,h in load(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
configs=[];index={};solver_records={};physical=0;forecasts=0
for summary in F['summaries']:
    week=summary['week'];method=summary['method']
    folder=OUT/'runs'/f'r0__{week}__{method}'
    ref=ROOT/'outputs/grid26/runs'/f'r0__{week}__FULL'
    actual,original=rows(folder/'steps.jsonl.gz'),rows(ref/'steps.jsonl.gz');r=load(ref/'summary.json')
    assert len(actual)==len(original)==summary['steps'] and summary['complete']==r['complete']
    assert summary['cost']==r['cost']
    for a,b in zip(actual,original):
        for key in ['step','before_hash','action_hash','after_hash','raw_cost','done','complete']:
            assert a[key]==b[key],(week,method,a['step'],key)
        assert a['illegal']==b['illegal'] and a['ambiguous']==b['ambiguous'] and bool(a['exceptions'])==bool(b['exceptions'])
    with np.load(folder/'vectors.npz') as x,np.load(ref/'vectors.npz') as y:
        for key in ['action','observation']:assert np.array_equal(x[key],y[key],equal_nan=True)
    assert abs(math.fsum(a['decision_s'] for a in actual)-summary['controller_s'])<1e-6
    profile=load(folder/'components.json');active=[x for x in profile['calls'] if x['in_control']]
    phases=['run_dc','compute_optimum_safe','compute_optimum_unsafe']
    times={key:math.fsum(c['wall_s'] for c in active if c['phase']==key) for key in phases}
    counts=dict(Counter(x['phase'] for x in active))
    remaining=summary['controller_s']-sum(times.values())-summary['all_forecast_s'];assert remaining>=0
    explicit=[x for x in profile['solver_records'] if x['in_control'] and x['solver'] is not None]
    groups=defaultdict(list)
    for x in explicit:groups[(x['phase'],x['owner'],x['solver'])].append(x)
    stats=[]
    for (phase,owner,solver),items in sorted(groups.items()):
        value={'phase':phase,'owner':owner,'solver':solver,'calls':len(items),
            'wall_s':math.fsum(x['wall_s'] for x in items),'statuses':dict(Counter(x['status'] for x in items))}
        for key in ['compilation_s','backend_s','iterations']:
            valid=[x[key] for x in items if x[key] is not None]
            value[key]=math.fsum(valid) if valid else None;value[key+'_missing']=len(items)-len(valid)
        stats.append(value)
    config={'week':week,'method':method,'steps':summary['steps'],'complete':summary['complete'],
        'cost':summary['cost'],'controller_s':summary['controller_s'],**times,'remaining_s':remaining,
        'all_forecast_s':summary['all_forecast_s'],'public_forecasts':summary['public_forecasts'],
        'n1_forecasts':summary['n1_forecasts'],'n1_source_s':summary['n1_source_s'],'component_calls':counts,
        'dc_cache_counts':profile['cache_counts'],'safe_cache_counts':profile['safe_cache_counts'],'solver_stats':stats}
    configs.append(config);index[(week,method)]=config;solver_records[(week,method)]=explicit
    physical+=len(actual);forecasts+=summary['public_forecasts']
pairs=[]
for week in D['weeks']:
    a,b=index[(week,'ORIGINAL')],index[(week,'DC_AND_SAFE')]
    assert a['public_forecasts']==b['public_forecasts'] and a['n1_forecasts']==b['n1_forecasts']
    assert a['component_calls']==b['component_calls']
    saved={key:a[key]-b[key] for key in ['controller_s','run_dc','compute_optimum_safe',
        'compute_optimum_unsafe','all_forecast_s','remaining_s']}
    assert abs(saved['controller_s']-math.fsum(v for k,v in saved.items() if k!='controller_s'))<1e-7
    left,right=solver_records[(week,'ORIGINAL')],solver_records[(week,'DC_AND_SAFE')]
    assert len(left)==len(right)
    mismatches=[]
    for x,y in zip(left,right):
        for key in ['step','phase','owner','solver']:assert x[key]==y[key]
        if (x['status'],x['iterations'])!=(y['status'],y['iterations']):
            mismatches.append({'step':x['step'],'phase':x['phase'],'original_status':x['status'],
                'cached_status':y['status'],'original_iterations':x['iterations'],'cached_iterations':y['iterations']})
    pairs.append({'week':week,'original_s':a['controller_s'],'cached_s':b['controller_s'],
        'control_reduction':saved['controller_s']/a['controller_s'],'saved_s':saved,'steps':a['steps'],
        'complete':a['complete'],'same_path':True,'solver_path_mismatches':mismatches})
o=math.fsum(p['original_s'] for p in pairs);c=math.fsum(p['cached_s'] for p in pairs)
R={'passed':True,'runs':6,'paired_date_families':3,'physical_rows':physical,'public_forecasts':forecasts,
    'all_paths_identical_to_full':True,'new_or_earlier_failure':0,
    'completed_pairs':sum(p['complete'] for p in pairs),'pairs':pairs,'configurations':configs,
    'pooled_control_reduction':1-c/o,'new_fits':0,'reserved_evaluations':0,
    'warning':'3already-exposed development date pairs,one timed repeat each. Known June failure retained. No independent final generalization/GNN attribution or universal runtime guarantee.'}
(OUT/'review.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
with (OUT/'pair_results.csv').open('w',newline='',encoding='utf-8') as f:
    keys=[k for k in pairs[0] if k not in ['saved_s','solver_path_mismatches']]
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:p[k] for k in keys} for p in pairs)
lines=['# GRID32 其余开发周共享编译复用兼容性', '',
       f'6条回放、{physical}个物理步骤、{forecasts}次公开预测均核验。动作/观测/成本逐步等同原FULL，未新增或提前失败；六月原有1901步失败保留，不能说全周完成。', '',
       '| 日期 | 原控制 | 双复用控制 | 减少 | 步数 | 完成 | 数值路径状态/迭代差别数 |',
       '|---|---:|---:|---:|---:|---|---:|']
for p in pairs:
    lines.append(f"| {p['week']} | {p['original_s']:.3f}s | {p['cached_s']:.3f}s | {100*p['control_reduction']:.2f}% | {p['steps']} | {p['complete']} | {len(p['solver_path_mismatches'])} |")
lines+=['',f"本批汇总控制时间减少{100*R['pooled_control_reduction']:.2f}%，仅为3条单次配对的点估计。三月22.42%来自另2重复，不在这里当作更多独立日期。", '',R['warning'], '',
        '共有DPP/程序复用应提供给所有强基线。GNN架构、RL增量和辅助搜索的独有提速尚需另行公平比较；原模型、质量取舍、全部失败与快慢计时保留。']
(OUT/'GRID32_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:3exposed-week pairs,known June failure same1901,no new fit/final-date evaluation.\n',encoding='utf-8')
print(json.dumps({k:v for k,v in R.items() if k!='configurations'},indent=2))
