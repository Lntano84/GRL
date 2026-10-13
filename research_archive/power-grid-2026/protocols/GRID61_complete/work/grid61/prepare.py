"""Freeze one bounded live mechanism comparison, no model selection."""
import copy,hashlib,json,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid61'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
Q=read(ROOT/'outputs/grid60/qualification.json');assert Q['passed_engineering'] and Q['permission_checks']==104 and Q['existing_frame_replay_equivalent']==90
assert not OUT.exists();OUT.mkdir()
D=copy.deepcopy(read(ROOT/'outputs/grid59/design.json'))
methods=[{'method':p,'rule':'STATIC_ID','budget':128,'checkpoint':None,'policy':p,'cache_dc':True,'cache_safe':True} for p in ['FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME']]
D.update(stage='GRID61_FRAMEWORK_COMMON_ORDER_MECHANISM',methods=methods,repeats=2,
    scope='Four existing development weeks,32 fresh serial paired runs,all policies use same author-frequency order. No new ranking/training/teacher/data/parameter selection or new external baseline families.',
    timing='Two balanced repetitions;rotate four policies over weeks,reverse within each date for repeat1. Original GRID55worker/control/timers/shared DCsafeQP preserved;common ordering overhead included.',
    primary='Whole-week cost,survival,N1queries and actual whole-controller time. FRAME quality relative same-batch FULL: noearlierfailure,completeweek cost<=.5%. Common failedweeks retained;no speed ratios on unequal horizons.',
    investment='Mechanism study,not winner search:report all paired effects. Local completion supported in this matrix only when FRAME removes a NO_LOCAL quality violation/earlierfailure,or reduces complete-week cost by>=.1%ofFULL in at leastonefamily while retaining overallquality. Report any adverse families. Do not infer global optimum or safetycertificate.',
    continuation='If FIRST_SAFE_BEST passes allsamequality and has macro control time<=102%ofFRAME,accept it as competitive and do not claim adaptive structure necessary. If FRAME failsquality,stop current deployment claim and analyze saved traces before changes. Mixed/small effects remain unresolved;no parameter/prior/model sweep. No RL.',
    first_safe_reference='SoftGNN2025 sequential first strict-admissible rho<.9;strengthened exhausted fallback keeps best admissible. Resource-matched search-policy adaptation,not full published agent reproduction.',
    no_local_ablation='Same128initial/64batch/256minimum/two-stale/safe.9,only disables incumbent-substation completion.',
    permission='Current observation,candidate library and actually queried feedback only;historical FULL results used only by offline audits,never runner.',
    old_models='All prior GNN variants and failedresults remain;none loaded or trained in this matrix.')
D['execution_order']=[]
for rep in [0,1]:
    for i,w in enumerate(D['weeks']):
        order=methods[i:]+methods[:i]
        if rep:order=order[::-1]
        D['execution_order'] += [{**m,'week':w,'repeat':rep,'stage_phase':'mechanism'} for m in order]
D['caps'].update(wall_s=6000,episode_s=1200,physical_steps=65000,public_forecasts=500000,new_fits=0)
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
source=(ROOT/'work/grid59/run.py').read_text(encoding='utf-8').replace('outputs/grid59','outputs/grid61')
source=source.replace("ROOT/'work/grid59'","ROOT/'work/grid61'").replace('from frequency_adapter import install_frequency_adapter','from mechanism_adapter import install_mechanism_adapter').replace('install_frequency_adapter(engine.Search)','install_mechanism_adapter(engine.Search)')
source=source.replace("'planned': 16","'planned': 32").replace('len(summaries) == 16','len(summaries) == 32').replace('sixteen-run','32-run').replace('EIGHT RUNS COMPLETE','32 RUNS COMPLETE')
assert 'frequency_adapter' not in source and "'planned': 32" in source and 'len(summaries) == 32' in source
(HERE/'run.py').write_text(source,encoding='utf-8')
freeze={}
for name in ['outputs/grid59/code_freeze.json','outputs/grid60/code_freeze.json']:
    for p,h in read(ROOT/name).items():assert sha(ROOT/p)==h,p;freeze[p]=h
for p in list(HERE.glob('*.py'))+[OUT/'design.json',ROOT/'outputs/grid60/qualification.json']:
    freeze[p.relative_to(ROOT).as_posix()]=sha(p)
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(OUT/'preflight.json').write_text(json.dumps({'passed':True,'runs':32,'methods':4,'dates':4,'repeats':2,'model_fits':0,'same_ranking':True,'engine':'Original GRID55 worker with policy-only adapter;all common caches preserved.','qualified_saved_state_checks':104,'qualified_original_frame_calls':90},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:32serial common-order mechanism runs;all4policies/2repeats retained,no training.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'runs':32,'fits':0,'cap_s':6000}))
