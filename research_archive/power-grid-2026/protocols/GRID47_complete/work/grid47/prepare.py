"""Freeze common existing-author-buffer fallback on old development episodes only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid47'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT/'outputs/grid46/audit.json')['passed']
assert read(ROOT/'outputs/grid46/audit.json')['all_critical1733']
previous=read(ROOT/'outputs/grid43/design.json');methods=previous['methods'];order=[]
for n,week in enumerate(previous['weeks']):
    shift=n%len(methods)
    for m in methods[shift:]+methods[:shift]:order.append({**m,'week':week,'repeat':0,'stage_phase':'pilot' if n==0 else 'remaining'})
D={**previous,'stage':'GRID47_COMMON_AUTHOR_THRESHOLD_COMPLETION','methods':methods,'execution_order':order,
   'changes':'Only restore author existing .9 buffer check after original GRID28 local-completion:if still >=.9,exhaust remaining canonical candidates. Identical shared cache/masks/basecontrol/models to all rankers. No tie completion,new model or threshold. Keep original GRID44 results.',
   'scope':'4oldexposed development weeks,6implementations/3baseline families,single timing repetition. No GRID44episode inference or training.',
   'pilot_gate':'6Marchruns. Continue18only if BOTH originalGNNseeds survive/complete likeFULL,cost<=.5%andpositive control-time reduction. Baseline quality failures recorded,not grounds for discarding their methods or artificially weakening them.',
   'development_screen':'Across4oldweeks:bothownseeds noearlierfailure;cost<=.5%onFULL-completed weeks;mean paired controller reduction>=5%,positive>=3/4dates. Mixedslightchanges are not absolute research failure. Do not choose bestseed.',
   'interpretation':'Conservative quality-time endpoint reusing author buffer,not new safety certificate or GNN-specific contribution. Single timing repeat and exposed dates;later confirmation must use freshly frozen date families,not GRID44 again.',
   'caps':{'episode_s':1500,'wall_s':9000,'physical_steps':50000,'public_forecasts':600000,'new_fits':0,'reserved_evaluations':0}}
src=(ROOT/'work/grid43/run.py').read_text(encoding='utf-8')
changes={
 '"""GRID43 common observed-tie completion; all existing rankers, unchanged models."""':'"""GRID47 common original-author-buffer completion; original models unchanged."""',
 "OUT=ROOT/'outputs/grid43'":"OUT=ROOT/'outputs/grid47'",
 "str(ROOT/'work/grid42b')":"str(ROOT/'work/grid46')",
 'from tied_policy import tied_complete':'from safe_completion import safe_complete',
 'engine.P.local_complete=tied_complete':'engine.P.local_complete=safe_complete',
 'all6arms shareDC+safe cache and common tied completion':'all6arms shareDC+safe cache and common author threshold completion'}
for a,b in changes.items():assert src.count(a)==1,(a,src.count(a));src=src.replace(a,b)
(HERE/'run.py').write_text(src,encoding='utf-8')
a=(ROOT/'work/grid43/audit.py').read_text(encoding='utf-8')
changes={"OUT=ROOT/'outputs/grid43'":"OUT=ROOT/'outputs/grid47'",
 '# GRID43 共同并列补查的质量与速度':'# GRID47 既有作者阈值回补的质量与速度',
 "OUT/'GRID43_report.md'":"OUT/'GRID47_report.md'",
 'common observed-tie completion/controller/caches.':'common original-author-threshold completion/controller/caches.',
 '24 common tied-completion comparisons.':'24 common author-buffer completion comparisons.',
 "regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if comparable else None":"regret=(s['cost']-full['cost'])/max(1,abs(full['cost'])) if full['complete'] and s['complete'] else None",
 "bool(survival and (regret is not None and regret<=.005))":"bool(survival and (not full['complete'] or (regret is not None and regret<=.005)))"}
for x,y in changes.items():assert a.count(x)==1,(x,a.count(x));a=a.replace(x,y)
(HERE/'audit.py').write_text(a,encoding='utf-8')
OUT.mkdir(exist_ok=False);(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),HERE/'run.py',HERE/'audit.py',HERE/'audit_pilot.py',ROOT/'work/grid46/safe_completion.py',ROOT/'outputs/grid46/audit.json']
files += [ROOT/p for p in read(ROOT/'outputs/grid41/code_freeze.json')]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in dict.fromkeys(files)},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:6oldMarchpilots,18olddevelopment only if own2seeds quality/positive-speedgate. No new fits or GRID44reuse.\n',encoding='utf-8')
print(json.dumps({'planned':24,'pilot':6,'new_fits':0,'GRID44_reuse':0}))
