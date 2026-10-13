"""One paired olddevelopment pilot, all seeds and strong prior baseline retained."""
import copy,hashlib,json,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid59'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(ROOT/'outputs/grid58/qualification.json')['old_development_closedloop_eligible']
assert not OUT.exists();OUT.mkdir()
old=read(ROOT/'outputs/grid55/design.json');static=next(m for m in read(ROOT/'outputs/grid37/design.json')['methods'] if m['method']=='STATIC_ID')
full=old['methods'][0]
methods=[full,static]+[{**static,'method':f'FREQUENCY_GNN{s}','rule':'FREQUENCYGNN','checkpoint':f'outputs/grid58/models/gnn_seed{s}/final.pt'} for s in [0,1]]
weeks=read(ROOT/'outputs/grid58/design.json')['development_weeks']
D=copy.deepcopy(old);D.update(stage='GRID59_ONE_FREQUENCY_PRIOR_CLOSEDLOOP_PILOT',weeks=weeks,methods=methods,repeats=1,
    scope='Four olddevelopment weeks,including prior common-failure week;FULL/STATIC_ID/frequencyGNN0/1,16serial fresh runs. No final8date evaluation,new fit/teacher,threshold change or extra baseline families.',
    timing='Rotate four implementations over four dates,one single-thread worker each;same GRID55 control/DC/safe caches and component timers. Single paired timing repeat is exploratory,no reliable speed guarantee.',
    primary='Bothseeds noearlierfailure than same-batch FULL;completedweek cost<=.5%ofFULL. Then report paired quality/control/queries against same-batch STATIC_ID and FULL. Unequal survival is not a speed win.',
    investment='Olddevelopment engineering screen only:both seeds pass survival/.5%cost. Extra learning signal requires seed-average macro query reduction>=5%againstSTATIC_ID and>=3/4date directions positive with bothseed macro directions positive. Actual controltime shown separately. Other outcomes mixed/no further prior/constant/model sweep;do not select a seed.',
    decision_after='If quality fails,mixedorweak,retain strong frequency baseline and stop this prior revision. If bothquality andquery conditions pass,consider one separately frozen confirmation,not immediate claim of paper contribution.',
    old_models='Original PPO-prior graph,tanh/soft-compressed/free variants and all failures stay unchanged/local;STATIC_ID remains strong reference.')
D['execution_order']=[]
for i,w in enumerate(weeks):
    ordered=methods[i:]+methods[:i]
    D['execution_order'] += [{**m,'week':w,'repeat':0,'stage_phase':'pilot'} for m in ordered]
D['caps'].update(wall_s=8000,physical_steps=35000,public_forecasts=650000,new_fits=0)
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
source=(ROOT/'work/grid55b/run.py').read_text(encoding='utf-8').replace('outputs/grid55b','outputs/grid59')
source=source.replace("'planned': 8","'planned': 16").replace("'planned': 8,","'planned': 16,").replace('len(summaries) == 8','len(summaries) == 16')
source=source.replace('eight-arm','sixteen-run').replace('One eight-arm supplemental timing repetition','One sixteen-run olddevelopment pilot')
needle="        sys.argv = [str(source_path), '--worker', str(args.worker)]";assert source.count(needle)==1
injection="""        marker = '    old_init=ResidualControl.__init__;old_propose=ResidualControl.propose'
        assert source.count(marker)==1
        patch = \"    sys.path.insert(0,str(ROOT/'work/grid59'))\\n    from adapter import install_frequency_adapter\\n    install_frequency_adapter(engine.Search)\\n\"
        source=source.replace(marker,patch+marker)
"""
source=source.replace(needle,injection+needle)
# The old runner already imports a module named adapter from grid40. Load our
# distinct adapter under a unique name to avoid Python module-cache collision.
source=source.replace('from adapter import install_frequency_adapter','from frequency_adapter import install_frequency_adapter')
assert not (HERE/'run.py').exists();(HERE/'run.py').write_text(source,encoding='utf-8')
(HERE/'frequency_adapter.py').write_text((HERE/'adapter.py').read_text(encoding='utf-8'),encoding='utf-8')
freeze=read(ROOT/'outputs/grid55b/code_freeze.json')
for p,h in freeze.items():assert sha(ROOT/p)==h,p
for p in list(HERE.glob('*.py'))+list((ROOT/'work/grid58').glob('*.py'))+[OUT/'design.json',ROOT/'outputs/grid58/qualification.json',ROOT/'outputs/grid58/models/gnn_seed0/final.pt',ROOT/'outputs/grid58/models/gnn_seed1/final.pt',ROOT/'work/grid36/cheap_rank.py',ROOT/'work/grid35b/physics_order.py']:
    freeze[p.relative_to(ROOT).as_posix()]=sha(p)
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(OUT/'preflight.json').write_text(json.dumps({'passed':True,'runs':16,'methods':4,'dates':4,'new_fits':0,'all_seeds_retained':True,'source_worker_patch':'Outputdirectory and frequency-adapter install only;exact frozen GRID55worker/timers.'},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:16same-batch olddevelopment runs;quality/query/time against strong frequency and FULL.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':16,'dates':4,'new_fits':0}))

