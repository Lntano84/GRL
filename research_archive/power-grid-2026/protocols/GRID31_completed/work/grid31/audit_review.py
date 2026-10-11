"""Six-arm saved-data accounting; keep original/DC-only/shared-safe comparisons."""
import csv,gzip,hashlib,json,math,os
from collections import Counter,defaultdict
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid31'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
F=load(OUT/'finished.json');D=load(OUT/'design.json');assert F['passed_engineering']
for name,h in load(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
reference=ROOT/'outputs/grid29c/runs/r0__2035-03-05_0__ORIGINAL'
refrows=rows(reference/'steps.jsonl.gz');refsummary=load(reference/'summary.json')
configs=[];index={};records={}
for summary in F['summaries']:
    folder=OUT/'runs'/f"r{summary['repeat']}__{summary['week']}__{summary['method']}"
    profile=load(folder/'components.json');trajectory=rows(folder/'steps.jsonl.gz')
    assert len(trajectory)==len(refrows)==2017 and summary['complete']
    assert summary['cost']==refsummary['cost'] and summary['public_forecasts']==11325 and summary['n1_forecasts']==6559
    for a,b in zip(trajectory,refrows):
        for key in ['step','before_hash','action_hash','after_hash','raw_cost','simulations','done','complete']:
            assert a[key]==b[key],(summary['method'],a['step'],key)
        assert not a['illegal'] and not a['ambiguous'] and not a['exceptions']
    with np.load(folder/'vectors.npz') as actual,np.load(reference/'vectors.npz') as ref:
        for key in ['action','observation']:assert np.array_equal(actual[key],ref[key],equal_nan=True)
    assert abs(math.fsum(r['decision_s'] for r in trajectory)-summary['controller_s'])<1e-6
    calls=[r for r in profile['calls'] if r['in_control']]
    phases=['run_dc','compute_optimum_safe','compute_optimum_unsafe']
    assert dict(Counter(r['phase'] for r in calls))=={'run_dc':153,'compute_optimum_safe':153,'compute_optimum_unsafe':34}
    times={name:math.fsum(r['wall_s'] for r in calls if r['phase']==name) for name in phases}
    groups=defaultdict(list)
    explicit=[r for r in profile['solver_records'] if r['in_control'] and r['solver'] is not None]
    for r in explicit:groups[(r['phase'],r['owner'],r['solver'])].append(r)
    solvers=[]
    for (phase,owner,solver),values in sorted(groups.items()):
        data={'phase':phase,'owner':owner,'solver':solver,'calls':len(values),
            'wall_s':math.fsum(r['wall_s'] for r in values),'statuses':dict(Counter(r['status'] for r in values))}
        for key in ['compilation_s','backend_s','iterations']:
            available=[r[key] for r in values if r[key] is not None]
            data[key]=math.fsum(available) if available else None;data[key+'_missing']=len(values)-len(available)
        solvers.append(data)
    remaining=summary['controller_s']-sum(times.values())-summary['all_forecast_s'];assert remaining>=0
    config={'repeat':summary['repeat'],'method':summary['method'],'controller_s':summary['controller_s'],
        'cpu_s':summary['controller_cpu_s'],**times,'all_forecast_s':summary['all_forecast_s'],
        'remaining_s':remaining,'n1_source_s':summary['n1_source_s'],'setup_s':summary['setup_s'],
        'dc_cache_counts':profile['cache_counts'],'safe_cache_counts':profile['safe_cache_counts'],'solvers':solvers}
    configs.append(config);index[(summary['repeat'],summary['method'])]=config
    records[(summary['repeat'],summary['method'])]=explicit
pairs=[]
for repeat in range(2):
    for source,target in [('ORIGINAL','DC_ONLY'),('DC_ONLY','DC_AND_SAFE'),('ORIGINAL','DC_AND_SAFE')]:
        a=index[(repeat,source)];b=index[(repeat,target)]
        saved={key:a[key]-b[key] for key in ['controller_s','run_dc','compute_optimum_safe',
            'compute_optimum_unsafe','all_forecast_s','remaining_s']}
        assert abs(saved['controller_s']-math.fsum(v for k,v in saved.items() if k!='controller_s'))<1e-7
        left,right=records[(repeat,source)],records[(repeat,target)];assert len(left)==len(right)==340
        mismatches=[]
        for x,y in zip(left,right):
            for key in ['step','phase','owner','solver']:assert x[key]==y[key]
            if (x['status'],x['iterations'])!=(y['status'],y['iterations']):
                mismatches.append({'step':x['step'],'phase':x['phase'],'original_status':x['status'],
                    'target_status':y['status'],'original_iterations':x['iterations'],'target_iterations':y['iterations']})
        pairs.append({'repeat':repeat,'source':source,'target':target,'original_s':a['controller_s'],
            'target_s':b['controller_s'],'control_reduction':saved['controller_s']/a['controller_s'],
            'saved_s':saved,'numeric_solver_path_mismatches':mismatches})
pooled=[]
for source,target in [('ORIGINAL','DC_ONLY'),('DC_ONLY','DC_AND_SAFE'),('ORIGINAL','DC_AND_SAFE')]:
    group=[p for p in pairs if p['source']==source and p['target']==target]
    a=math.fsum(p['original_s'] for p in group);b=math.fsum(p['target_s'] for p in group)
    pooled.append({'source':source,'target':target,'control_reduction':1-b/a,
        'both_pairs_faster':all(p['control_reduction']>0 for p in group),
        'pair_reductions':[p['control_reduction'] for p in group]})
result={'passed':True,'runs':6,'physical_rows':12102,'public_forecasts':67950,'cost_delta':0,
    'all_delivered_vectors_identical':True,'configurations':configs,'pairs':pairs,'pooled':pooled,
    'new_fits':0,'reserved_evaluations':0,'independent_new_ac_or_qp_solves':0,
    'warning':'One exposed March/2balanced repeats. Shared conventional engineering for all baselines; not GNN novelty, final heldout result or guaranteed universal speed.'}
(OUT/'review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
with (OUT/'component_totals.csv').open('w',newline='',encoding='utf-8') as f:
    keys=[k for k in configs[0] if k not in ['dc_cache_counts','safe_cache_counts','solvers']]
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in configs)
lines=['# GRID31 共有DC/安全QP编译复用完整控制试点', '',
    '6条完整回放：12102个物理步骤和67950次预测，逐步交付/成本均与原受保护FULL相同；所有新拟合/RL/封存评价为0。', '',
    '| 重复 | 原控制 | DC复用 | DC+安全QP复用 |', '|---|---:|---:|---:|']
for repeat in range(2):
    lines.append(f"| {repeat} | {index[(repeat,'ORIGINAL')]['controller_s']:.3f}s | {index[(repeat,'DC_ONLY')]['controller_s']:.3f}s | {index[(repeat,'DC_AND_SAFE')]['controller_s']:.3f}s |")
lines += ['', '| 比较 | 重复0减少 | 重复1减少 | 汇总点估计 |', '|---|---:|---:|---:|']
for p in pooled:
    lines.append(f"| {p['source']}→{p['target']} | {100*p['pair_reductions'][0]:.2f}% | {100*p['pair_reductions'][1]:.2f}% | {100*p['control_reduction']:.2f}% |")
lines += ['', '设置/物理步/归档单列；控制时间包括实际模型、查询、求解、恢复、缓存和相同组件计时开销。真实params始终刷新；原solver/资格/warm_start=False保留。不得相加局部影子百分比或择最快重复。', '',
    '原作者连续QP可能触及迭代上限，其规则原样保留；恢复QP资格未放松。V0非DPP、复制器负值setter失败与预冻结准备错误全部保留，不等于方法价值否定。', '',result['warning']]
(OUT/'GRID31_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:6exact shared-cache pilot runs; no pending workers,fits or final reserved evaluations.\n',encoding='utf-8')
print(json.dumps({'passed':True,'pooled':pooled,'whole_times':[{'repeat':r['repeat'],'method':r['method'],'controller_s':r['controller_s']} for r in configs],
    'numeric_path_mismatch_counts':[len(p['numeric_solver_path_mismatches']) for p in pairs]},indent=2))
