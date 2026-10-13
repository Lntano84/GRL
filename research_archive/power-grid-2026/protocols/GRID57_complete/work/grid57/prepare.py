"""One final static-order ablation; no new baseline families, fitting or tuning."""
import copy,hashlib,json,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid57'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
save=lambda p,x:p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
assert not OUT.exists();OUT.mkdir()
assert read(ROOT/'outputs/grid56/FINAL_AUDIT.json')['passed']
original=read(ROOT/'outputs/grid55/design.json');static=next(r for r in read(ROOT/'outputs/grid37/design.json')['methods'] if r['method']=='STATIC_ID')
D=copy.deepcopy(original)
D.update(stage='GRID57_FINAL_STATIC_ORDER_ABLATION',methods=[static],repeats=1,
    scope='Eight already-exposed final dates, one frozen STATIC_ID rollout each. Existing final GNN/FULL cost and query evidence reused; new batch wall times descriptive only. No source model or input tuning.',
    primary='No earlier failure than FULL; completed-week cost relative FULL; same causal-public query semantics. Compare both final GNN seeds equally, never best seed.',
    timing='Single serial worker, same original controller/timers/shared caches. No cross-batch actual-time superiority claim. Scoped awake lease; first construction charged as before.',
    statistics='Eight date families are paired development ablation units; single timing repeat. Not new independent confirmation, significance or equivalence.',
    decision_rule='If STATIC_ID meets the0.5%cost/no-earlier-failure screen on all8dates and mean_i(Q_STATIC_i/Q_GNN_seed_average_i)<=1.05, the necessity of learned ranking is not established. If STATIC_ID fails quality where both final GNNs pass, report that concrete quality-tail difference. If all qualify and mean_i(1-Q_GNN_seed_average_i/Q_STATIC_i)>=.05 with>=6/8positive families, report extra query evidence only,not end-to-end speed. These are different denominators explicitly;other outcomes mixed;no new ranking/tuning/repeat loop.',
    prohibited='No new training, dates, weights, scalar choices, additional cheap rankers, relabeling or outcome filtering; STATIC_ID is the exact prior GRID37 ablation,not a newly optimized baseline.')
D['execution_order']=[{**static,'week':w,'repeat':0,'stage_phase':'pilot'} for w in original['weeks']]
D['caps'].update(wall_s=7200,physical_steps=17000,public_forecasts=500000,new_fits=0)
save(OUT/'design.json',D)
# Reuse the safe eight-worker launcher, and the exact GRID55 worker branch.
source=(ROOT/'work/grid55b/run.py').read_text(encoding='utf-8').replace('outputs/grid55b','outputs/grid57')
source=source.replace('One eight-arm supplemental timing repetition; original GRID55 remains immutable.','Eight static-order ablation episodes; original results remain immutable.')
needle="        sys.argv = [str(source_path), '--worker', str(args.worker)]"
assert source.count(needle)==1
injection="""        marker = '    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
        assert source.count(marker) == 1
        patch = \"    sys.path.insert(0,str(ROOT/'work/grid36'))\\n    from cheap_rank import rank as static_rank\\n    engine.Search.rank=static_rank\\n\"
        source = source.replace(marker, patch + marker)
"""
source=source.replace(needle,injection+needle)
assert not (HERE/'run.py').exists();(HERE/'run.py').write_text(source,encoding='utf-8')
freeze=read(ROOT/'outputs/grid55b/code_freeze.json')
for p,h in freeze.items():assert sha(ROOT/p)==h,p
for p in [ROOT/'work/grid36/cheap_rank.py',ROOT/'work/grid35b/physics_order.py',ROOT/'work/grid37/design.json',ROOT/'work/grid37/run.py',OUT/'design.json']+list(HERE.glob('*.py')):
    freeze[p.relative_to(ROOT).as_posix()]=sha(p)
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
save(OUT/'code_freeze.json',freeze)
# Directly exercise the frozen cheap rank without any environmental feedback.
import importlib.util,sys
spec=importlib.util.spec_from_file_location('grid57_static_probe',ROOT/'work/grid36/cheap_rank.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
obj=type('Probe',(),{'rule':'STATIC_ID'})()
for reps in [[3,1,9],[902,4,8,10],[]]:
    order,prior,features,times=module.rank(obj,None,None,reps,reps,{i:i for i in reps})
    assert order==sorted(reps) and prior is None and features is None
save(OUT/'preflight.json',{'passed':True,'runs':8,'exact_existing_static_config':True,'sorted_representative_checks':3,'new_fits':0,'new_baseline_families':0,'cross_batch_timing_claim':False})
(OUT/'RUN_STATE.md').write_text('FROZEN: eight static-order quality/query ablation episodes; no fitting.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':8,'rule':'STATIC_ID','source_hashes':len(freeze)}))
