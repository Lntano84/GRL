"""Freeze a mechanism audit before new policy-replay outcomes."""
import hashlib,json,py_compile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid60'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert not OUT.exists();OUT.mkdir()
weeks=read(ROOT/'outputs/grid59/design.json')['weeks']
sources=[]
for week in weeks:sources+=sorted((ROOT/'outputs/grid59/runs'/f'r0__{week}__FULL').glob('call[0-9][0-9][0-9].json'))
assert len(sources)==26
frames=[]
for stage,dates in [('grid59',weeks),('grid57',read(ROOT/'outputs/grid57/design.json')['weeks'])]:
    for week in dates:frames+=sorted((ROOT/f'outputs/{stage}/runs'/f'r0__{week}__STATIC_ID').glob('call[0-9][0-9][0-9].json'))
D={'stage':'GRID60_FRAMEWORK_MECHANISM_QUALIFICATION','authorization':'User 2026-10-13 selects framework as paper focus;GNN optional. Preserve all variants.',
   'scope':'Read-only code/literature/evidence audit and policy replay on 26 olddevelopment FULL states;existing FRAME traces. No fitting,new physical simulation,date selection,threshold sweep or new external baseline families.',
   'policies':['FULL','FIRST_SAFE_BEST','NO_LOCAL','FRAME'],'ranking':'Same original author teacher-frequency order,after same public legal mask/canonicalization.',
   'first_safe':'Stop at first strict-admissible rho<.9;when none exists,use best strict-admissible from exhaustive search. This is a strengthened adaptation of SoftGNN first-safe logic,not complete published-agent reproduction.',
   'no_local':'Same frozen128initial/64batch/256minimum/2stale/safe.9;only disable incumbent-substation completion.',
   'frame':'Original GRID28 semantic completion,parameters unchanged;no global stopping certificate or wall-clock guarantee.',
   'primary':'Separate query reduction,observed local improvements,same-state reward/choice and archived real whole-controller timing. No cached-state regret interpreted as whole-week loss.',
   'continuation':'Engineering/replay equivalence and hidden-feedback perturbation must pass before any live mechanism matrix. Then one separately frozen bounded same-order policy comparison;no training or parameter change.',
   'sources':[p.relative_to(ROOT).as_posix() for p in sources], 'frame_traces':[p.relative_to(ROOT).as_posix() for p in frames],
   'substation_metadata':'outputs/grid28/offline.json',
   'papers':[{'url':'https://github.com/lajavaness/l2rpn-2023-ljn-agent','coverage':'Official author hybrid continuous/topology modules and NN top-k search;components already exist.'},
     {'url':'https://arxiv.org/html/2503.15190v2#S4.SS4','coverage':'SoftGNN2025 first-safe sequential simulation,disconnected-line masking,and top10N-1 extension.'},
     {'url':'https://arxiv.org/html/2501.07186v1#S4.SS9','coverage':'Verify+greedy and Verify+N-1 hybrids;verification/fallback is established.'}],
   'caps':{'wall_s':120,'new_fits':0,'new_physical_rollouts':0,'new_forecasts':0}}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=list(HERE.glob('*.py'))+sources+frames+[OUT/'design.json',ROOT/'outputs/grid28/offline.json',ROOT/'work/grid27/policy.py',ROOT/'work/grid28/policy.py',ROOT/'work/grid02/ljn_nn/modules/topology_heuristic.py',ROOT/'work/grid02/ljn_nn/modules/base_module.py']
freeze={p.relative_to(ROOT).as_posix():sha(p) for p in files}
for p in HERE.glob('*.py'):py_compile.compile(str(p),doraise=True)
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN: read-only framework mechanism qualification;no new fit or physical replay.\n',encoding='utf-8')
print(json.dumps({'full_states':len(sources),'existing_frame_calls':len(frames),'frozen':True}))
