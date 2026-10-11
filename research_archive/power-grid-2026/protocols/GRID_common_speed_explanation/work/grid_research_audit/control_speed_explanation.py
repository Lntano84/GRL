"""Saved-only answer to historical10.93%vs0.39% and new cross-week magnitude."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid_speed_review';OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'facts.json').exists()
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
recent=load('outputs/grid32/review.json');pilot=load('outputs/grid31/review.json')
assert recent['passed'] and pilot['passed']
per_date=[]
sources=['outputs/grid32/review.json','outputs/grid31/review.json','outputs/grid29b/review.json','outputs/grid29c/review.json']
for week in ['2035-03-05_0']+list(x['week'] for x in recent['pairs']):
    if week=='2035-03-05_0':
        own=[];cached=[]
        for repeat in [0,1]:
            for method,target in [('ORIGINAL',own),('DC_AND_SAFE',cached)]:
                file=f'outputs/grid31/runs/r{repeat}__{week}__{method}/components.json';sources.append(file)
                prof=load(file);summ=load(f'outputs/grid31/runs/r{repeat}__{week}__{method}/summary.json')
                active=[x for x in prof['calls'] if x['in_control']]
                target.append({'controller_s':summ['controller_s'],
                    'run_dc':math.fsum(x['wall_s'] for x in active if x['phase']=='run_dc'),
                    'compute_optimum_safe':math.fsum(x['wall_s'] for x in active if x['phase']=='compute_optimum_safe'),
                    'compute_optimum_unsafe':math.fsum(x['wall_s'] for x in active if x['phase']=='compute_optimum_unsafe'),
                    'safe_calls':sum(x['phase']=='compute_optimum_safe' for x in active),
                    'safe_counts':next(x for x in prof['safe_cache_counts'] if x) if method=='DC_AND_SAFE' else None})
        a={k:math.fsum(x[k] for x in own)/2 for k in ['controller_s','run_dc','compute_optimum_safe','compute_optimum_unsafe','safe_calls']}
        b={k:math.fsum(x[k] for x in cached)/2 for k in ['controller_s','run_dc','compute_optimum_safe','compute_optimum_unsafe','safe_calls']}
        hit=math.fsum(x['safe_counts']['hit'] for x in cached)/math.fsum(x['safe_counts']['calls'] for x in cached);repeats=2
    else:
        a=next(x for x in recent['configurations'] if x['week']==week and x['method']=='ORIGINAL')
        b=next(x for x in recent['configurations'] if x['week']==week and x['method']=='DC_AND_SAFE')
        a={**a,'safe_calls':a['component_calls']['compute_optimum_safe']}
        counts=next(x for x in b['safe_cache_counts'] if x);hit=counts['hit']/counts['calls'];repeats=1
    per_date.append({'week':week,'paired_repeats':repeats,'original_control_s':a['controller_s'],'cached_control_s':b['controller_s'],
       'reduction':1-b['controller_s']/a['controller_s'],'safe_calls':a['safe_calls'],'safe_cache_hit_rate':hit,
       'original_dc_safe_fraction':(a['run_dc']+a['compute_optimum_safe'])/a['controller_s'],
       'original_uncached_unsafe_fraction':a['compute_optimum_unsafe']/a['controller_s'],
       'saved_dc_safe_s':a['run_dc']+a['compute_optimum_safe']-b['run_dc']-b['compute_optimum_safe']})
R={'passed_saved_source_check':True,'dates':per_date,
   'history':'GRID29old10.93%/0.39% had no per-solver timings. Restored-delivery-step differences are observed, but cannot recover individual DC contributions or attribute physical/thermal/runtime causes retrospectively.',
   'measured_mechanism':'GRID29Bisolatedsame-input DC vectors identical; repeated compilation dominates DC. GRID29Cactual DC saves17.47/18.14s while disjoint other components offset5.74/6.27s. GRID31/32cachebothDCandsafeQP; frequency,reuseanduncachedunsafeproportion vary.',
   'qualification':'Exact delivered trajectories and original solver-status/iteration paths preserved. Shared conventional program reuse granted to all opponents,not GNN novelty.',
   'metadata_correction':'GRID32design inherited repeats=2 and older four-run wording, but its frozen execution_order has6runs,3singlepairedweeks. All reports/audits useactual6runs/onepairperweek. Original frozen JSON retained,not rewritten.',
   'input_hashes':{x:hashlib.sha256((ROOT/x).read_bytes()).hexdigest() for x in sources},
   'new_fits':0,'new_solves':0,'reserved_evaluations':0}
(OUT/'facts.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
lines=['# 为什么10%有效，而部分运行几乎看不到整体提速', '',
 '结论：DC重复编译的节省是真实的；总控制百分比由这部分在总时间中的占比、复用率，以及其他部分的耗时变化共同决定。旧10.93%/0.39%缺逐solver计时，不能事后确定精确波动根因。', '',
 '旧两组中，交付恢复的步骤分别省25.42秒与10.32秒；其他步骤分别多耗1.75秒与9.62秒。后续同输入DC诊断保持流量完全相同，原编译29.17秒、后端约0.07秒。实际闭环组件复测DC省17.47/18.14秒，却被其他部分多耗5.74/6.27秒抵消，整体仍省6.45%/6.51%。这些是不同批次的证据，不替代旧缺失计时。', '',
 '## 为什么加入安全QP复用后，开发周之间仍差很多', '',
 '| 日期 | 配对重复 | 原控制/双复用(s) | 时间减少 | 安全QP调用 | 安全QP缓存命中率 | 原DC+安全QP占比 | 未复用unsafe优化占比 |',
 '|---|---:|---:|---:|---:|---:|---:|---:|']
for x in per_date:lines.append(f"| {x['week']} | {x['paired_repeats']} | {x['original_control_s']:.2f}/{x['cached_control_s']:.2f} | {100*x['reduction']:.2f}% | {x['safe_calls']:.0f} | {100*x['safe_cache_hit_rate']:.2f}% | {100*x['original_dc_safe_fraction']:.2f}% | {100*x['original_uncached_unsafe_fraction']:.2f}% |")
lines+=['', '其他三周恢复调用更多，缓存命中率更高，未复用unsafe求解占比更低；对应的可节省组成占比也更大。这里有直接组件账目和相同路径支持，不需推测温度、后台负载或训练好坏。各组成百分比不能简单相加为架构收益。', '',
 '这些是已暴露开发日期，三月两重复、其余单重复；六月仍在1901步共同失败，没有改善安全完成率。下一步公平比较已固定：FULL、原PPO、两GNN和两MLP均使用相同共享优化，单独检验排序/停止的增量。', '',R['metadata_correction']]
(OUT/'WHY_10_PERCENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'dates':per_date,'new_solves':0},indent=2))
