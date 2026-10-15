"""Freeze the second, branch-isolated repair; retain GRID69's regression."""
from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid70';OUT.mkdir(parents=True,exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
assert not (OUT/'design.json').exists()
assert read(ROOT/'outputs/grid69/pilot_review.json')['passed_saved_data']
assert not read(ROOT/'outputs/grid69/pilot_review.json')['continue_regression']
assert read(OUT/'preflight.json')['passed']
D=read(ROOT/'outputs/grid69/design.json')
D.update(stage='GRID70',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    repair='Only initial-safe branch: minimum256 queries and two stale64 batches; same-station completion only if verification replaces the initially safe incumbent. Otherwise retain original incumbent. Original unsafe-prefix fallback/completion is byte-for-byte delegated, with prefix cache reused once.',
    scope='Second finite known-date repair after GRID69 March regression. All13 dates retained. FULL references reused solely for quality and query counts. Single new timing measurements reported raw without historical speed ratios; no independent confirmation or fitting.',
    timing='One serial repair worker per date. Same GRID55/28/cache instrumentation; no paired fresh FULL timing comparison in this development stage.',
    decision='Pilot July,December,March all complete, every complete cost<=FULL*1.005 and macro query saving>=50%; only then evaluate remaining10 dates without tuning. Preserve all outcomes. Stop if pilot fails; no date-wise policy selection.',
    methods=[dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,method=p,policy=p) for p in ['FULL','FRAME_GUARD']])
base=D['methods'][1]
D['execution_order']=[dict(base,week=w,repeat=0,stage_phase='pilot' if w in D['pilot'] else 'remaining') for w in D['pilot']+[x for x in D['weeks'] if x not in D['pilot']]]
save(OUT/'design.json',D)
frozen=read(ROOT/'outputs/grid69/code_freeze.json')
for p,h in frozen.items():assert sha(ROOT/p)==h,p
for p in [*sorted((ROOT/'work/grid70').glob('*.py')),OUT/'design.json']:
    frozen[p.relative_to(ROOT).as_posix()]=sha(p)
save(OUT/'code_freeze.json',frozen)
(OUT/'RUN_STATE.md').write_text('FROZEN: branch-isolated guard v2; GRID69 retained.\n',encoding='utf-8')
print(json.dumps(dict(runs=len(D['execution_order']),pilot_runs=3,frozen=len(frozen))))
