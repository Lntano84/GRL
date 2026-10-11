"""Complete-arm exactness, timing conservation and solver-path diagnosis."""
import csv, gzip, hashlib, json, math, os
from collections import Counter, defaultdict
from pathlib import Path
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid29c'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def read_rows(p):
    with gzip.open(p,'rt',encoding='utf-8') as f:return [json.loads(line) for line in f]
F=load(OUT/'finished.json');D=load(OUT/'design.json');assert F['passed_engineering']
for name,h in load(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
old=ROOT/'outputs/grid29/pilot/runs/r0__2035-03-05_0__ORIGINAL'
old_rows=read_rows(old/'steps.jsonl.gz');configs=[];paths={};component_paths={}
for summary in F['summaries']:
    folder=OUT/'runs'/f"r{summary['repeat']}__{summary['week']}__{summary['method']}"
    rows=read_rows(folder/'steps.jsonl.gz');profile=load(folder/'components.json')
    assert len(rows)==len(old_rows)==2017 and summary['cost']==load(old/'summary.json')['cost']
    for a,b in zip(rows,old_rows):
        for key in ['step','before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']:
            assert a[key]==b[key],(summary['repeat'],summary['method'],a['step'],key)
        assert not a['illegal'] and not a['ambiguous'] and not a['exceptions']
    with np.load(old/'vectors.npz') as ref,np.load(folder/'vectors.npz') as got:
        for key in ['action','observation']:assert np.array_equal(ref[key],got[key],equal_nan=True)
    assert abs(math.fsum(r['decision_s'] for r in rows)-summary['controller_s'])<1e-6
    assert abs(math.fsum(r['all_forecast_s'] for r in rows)-summary['all_forecast_s'])<1e-6
    group=defaultdict(list)
    calls=[r for r in profile['calls'] if r['in_control']]
    for r in profile['solver_records']:
        if r['in_control'] and r['solver'] is not None:group[(r['phase'],r['owner'],r['solver'])].append(r)
    stats=[]
    for (phase,owner,solver),items in sorted(group.items()):
        val={'phase':phase,'owner':owner,'solver':solver,'calls':len(items),
             'wall_s':math.fsum(r['wall_s'] for r in items),'statuses':dict(Counter(r['status'] for r in items))}
        for key in ['compilation_s','backend_s','iterations']:
            available=[r[key] for r in items if r[key] is not None]
            val[key]=math.fsum(available) if available else None
            val[key+'_missing']=len(items)-len(available)
        stats.append(val)
    totals={name:math.fsum(r['wall_s'] for r in calls if r['phase']==name)
            for name in ['run_dc','compute_optimum_safe','compute_optimum_unsafe']}
    assert sum(1 for r in calls if r['phase']=='run_dc')==153
    assert sum(1 for r in calls if r['phase']=='compute_optimum_safe')==153
    assert sum(1 for r in calls if r['phase']=='compute_optimum_unsafe')==34
    rest=summary['controller_s']-math.fsum(totals.values())-summary['all_forecast_s']
    assert rest>=0
    record={'repeat':summary['repeat'],'method':summary['method'],'controller_s':summary['controller_s'],
            'controller_cpu_s':summary['controller_cpu_s'],**totals,'all_forecast_s':summary['all_forecast_s'],
            'remaining_control_s':rest,'solver_groups':stats,'cache_counts':profile['cache_counts']}
    configs.append(record);paths[(summary['repeat'],summary['method'])]=record
    component_paths[(summary['repeat'],summary['method'])]=profile

pairs=[];numeric_path=[]
for repeat in range(2):
    original,cached=paths[(repeat,'ORIGINAL')],paths[(repeat,'CACHED')]
    saved={key:original[key]-cached[key] for key in ['controller_s','run_dc','compute_optimum_safe',
            'compute_optimum_unsafe','all_forecast_s','remaining_control_s']}
    assert abs(saved['controller_s']-math.fsum(v for k,v in saved.items() if k!='controller_s'))<1e-7
    # Matching delivered state, component and solver calls identifies numerical paths.
    indexed=[]
    for method in ['ORIGINAL','CACHED']:
        indexed.append([r for r in component_paths[(repeat,method)]['solver_records'] if r['in_control'] and r['solver'] is not None])
    assert len(indexed[0])==len(indexed[1])
    mismatches=[]
    for a,b in zip(*indexed):
        for key in ['step','phase','owner','solver']:assert a[key]==b[key],(repeat,key)
        if (a['status'],a['iterations'])!=(b['status'],b['iterations']):
            mismatches.append({'step':a['step'],'phase':a['phase'],'original_status':a['status'],
                               'cached_status':b['status'],'original_iterations':a['iterations'],
                               'cached_iterations':b['iterations']})
    numeric_path.append({'repeat':repeat,'matched_solver_calls':len(indexed[0]),'status_iteration_mismatches':mismatches})
    pairs.append({'repeat':repeat,'controller_reduction':saved['controller_s']/original['controller_s'],
                  'dc_reduction':saved['run_dc']/original['run_dc'],'saved_s':saved,
                  'dc_saving_as_baseline_control_pp':100*saved['run_dc']/original['controller_s'],
                  'non_dc_delta_s':saved['controller_s']-saved['run_dc']})
o=math.fsum(c['controller_s'] for c in configs if c['method']=='ORIGINAL')
c=math.fsum(c['controller_s'] for c in configs if c['method']=='CACHED')
result={'passed':True,'runs':len(configs),'same_physical_rows':len(configs)*len(old_rows),'configurations':configs,
        'pairs':pairs,'pooled_control_reduction':1-c/o,'solver_path_comparisons':numeric_path,
        'new_fits':0,'reserved_evaluations':0,'independent_ac_replays':0,
        'warning':'Single exposed March,2balanced pairs with identical instrumentation. Component accounting identifies measured savings, not thermal/OS-specific causes or independent generalization.'}
save(OUT/'review.json',result)
with (OUT/'component_totals.csv').open('w',encoding='utf-8',newline='') as f:
    keys=[k for k in configs[0] if k not in ['solver_groups','cache_counts']]
    w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in configs])
lines=['# GRID29C 完整两臂组件计时', '',
       '每条2017步动作、观测、成本及11325次预测与旧受保护FULL相同。组件调用：153DC/153安全QP/34原作者连续QP。所有计时在真实派发两臂中取得，无隔离探针插入。', '',
       '| 重复 | 原控制 | 缓存控制 | 整体减少 | DC节省 | 安全QP节省 | 原作者QP节省 | 预测节省 | 其余节省 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for p in pairs:
    s=p['saved_s'];a=paths[(p['repeat'],'ORIGINAL')];b=paths[(p['repeat'],'CACHED')]
    lines.append(f"| {p['repeat']} | {a['controller_s']:.3f} | {b['controller_s']:.3f} | {100*p['controller_reduction']:.2f}% | {s['run_dc']:.3f} | {s['compute_optimum_safe']:.3f} | {s['compute_optimum_unsafe']:.3f} | {s['all_forecast_s']:.3f} | {s['remaining_control_s']:.3f} |")
lines+=['',f"汇总控制减少{100*result['pooled_control_reduction']:.2f}%是该两组点估计，不替换旧10.93%/0.39%。按组件相加严格重建总差，正数为节省，负数为变慢。", '',
        '| 重复 | DC局部减少 | DC节省/原完整控制 | 非DC额外节省 | solver状态/迭代改变数 |',
        '|---|---:|---:|---:|---:|']
for p,n in zip(pairs,numeric_path):
    lines.append(f"| {p['repeat']} | {100*p['dc_reduction']:.2f}% | {p['dc_saving_as_baseline_control_pp']:.2f}个百分点 | {p['non_dc_delta_s']:.3f}s | {len(n['status_iteration_mismatches'])} |")
lines += ['',result['warning'], '',
          '原作者允许迭代上限等状态的路径仍保留；恢复QP资格规则未放松。复用不等于warm_start，后者仍False。具体compile/backend/iterations记录见review.json和各run/components.json，null不能填0。']
(OUT/'GRID29C_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:4exact full-arm component profiles; no pending workers, fits or reserved evaluation.\n',encoding='utf-8')
print(json.dumps({'passed':True,'pairs':pairs,'pooled_control_reduction':result['pooled_control_reduction'],
                  'solver_mismatch_counts':[len(n['status_iteration_mismatches']) for n in numeric_path],
                  'configurations':configs},indent=2))
