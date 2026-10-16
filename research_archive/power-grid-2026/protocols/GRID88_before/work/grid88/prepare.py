"""Freeze the final original13 matched timing comparison: guard vs literal512."""
from pathlib import Path
import datetime as dt,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;O=ROOT/'outputs/grid88'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
 assert not O.exists();assert read(ROOT/'outputs/grid87/review.json')['passed_saved_data']
 for n in ['run.py','mechanism_adapter.py','environment.py','pipeline.py']:
  t=(ROOT/f'work/grid87/{n}').read_text(encoding='utf-8-sig').replace('grid87','grid88').replace('GRID87','GRID88').replace('serial_benchmark_13','serial_benchmark_52').replace('fixed256 original13 quality regression','guard vs fixed512 original13 matched timing')
  if n=='run.py':t=t.replace('one serial balanced timing experiment','52-run serial balanced timing comparison')
  (W/n).write_text(t,encoding='utf8')
 (W/'policies.py').write_text('''"""Delegate both arms to the unchanged GRID86 policy implementation."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('grid88_policy_reference',ROOT/'work/grid86/policies.py')
REF=importlib.util.module_from_spec(s);s.loader.exec_module(REF)
POLICIES=('FRAME_GUARD','FIXED_512')
key=REF.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES
    return REF.run(order,observe,policy,subids,aliases,safe,limit,context)
''',encoding='utf8')
 old=read(ROOT/'outputs/grid87/design.json');base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True)
 methods=[dict(base,method=n,policy=n) for n in ['FRAME_GUARD','FIXED_512']];weeks=old['weeks'];order=[]
 for rep in [0,1]:
  for w in weeks if rep==0 else weeks[::-1]:
   j=weeks.index(w);block=methods[j%2:]+methods[:j%2]
   if rep:block=block[::-1]
   order.extend(dict(m,week=w,repeat=rep,stage_phase='all') for m in block)
 D=dict(stage='GRID88_FINAL_ORIGINAL13_GUARD_VS_FIXED512',authorization='2026-10-16 user 开始吧 after proposed13dates x2arms x2reverse timing repeats; final fixed-budget comparison before manuscript consolidation.',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),weeks=weeks,methods=methods,repeats=2,pilot=[],execution_order=order,
 caps=dict(episode_s=1200,wall_s=10800,physical_steps=110000,public_forecasts=2000000,new_fits=0),quality_threshold=.005,timing=old['timing'],fixed_policy=old['fixed_policy'].replace('cap','512'),guard_policy='Exact existing GRID70 final FRAME_GUARD,128 prefix,64batches,256verification,two stale batches,.9 control threshold,alias-based local completion; no change.',environment=old['environment'],full_reference=old['full_reference'],guard_reference=old['guard_reference'],
 fairness='Same original physical model, neural/base controller, static-ID candidate order/aliases, public forecast interface, strict admissibility, float32 reward/low-ID tie, graph-context construction and shared DC/safeQP/prior caches. Fresh process per replay; first construction and per-step work charged. Opposite method order per date in repeat1; reverse date order; no concurrent benchmark/training/install/audit.',
 scope='All13 previously exposed original2035 dates, two reverse timing repeats,52new serial replays. Development mechanism regression, not independent confirmation or second grid. No new training, tuning, budgets, dates, models or checkpoints.',
 decision='Quality: no earlier termination than frozen FULL; cost<=FULL*1.005 when both complete. Whole13 same-quality computational advantage requires both methods13/13 quality, identical horizons/completion on13dates, date-macro control reduction>=5%,>=9/13positive dates,both repeat macro reductions>0,and matched active-date macro query reduction>=5%. If FIXED512 fails quality while GUARD passes, report bounded quality evidence and descriptive computing, not whole13 same-quality speed. If both pass same horizons and fixed date-macro control extra<=2% and matched active-date macro query ratio<=1.02, accept competitive cheap baseline. All other outcomes mixed/incomplete; keep every date and both timing repeats; no new cap/model/seed/date rescue.',
 contribution='Test whether feedback-based compute allocation improves upon uniform512 querying. Existing GRID86 six-date fixed256 competitiveness and GRID87 quality failures remain unchanged. Structural local completion and GNN are not individually established as universal sources of benefit.',
 timing_scope='Only this batch FRAME vs FIXED512 yields paired timing comparisons. FULL65/68 supplies quality, survival and query reference only; no old FULL wall-clock speed claim. Per-date latency P50/P95/P99,max and disjoint optimizer/forecast/search/remainder accounting reported; repeats are timing replication, not new dates.',
 closure='After finite comparison, consolidate mechanism/evidence and Chinese manuscript. No automatic new sampler/model/budget experiments. All historical adverse results and user drafts retained; saved-data audit does not independently re-solve AC.')
 O.mkdir();save(O/'design.json',D);shutil.copy2(ROOT/'outputs/grid87/data_freeze.json',O/'data_freeze.json')
 f=read(ROOT/'outputs/grid87/code_freeze.json')
 for p,h in f.items():assert sha(ROOT/p)==h,p
 for p in [*W.glob('*.py'),O/'design.json',O/'data_freeze.json']:f[p.relative_to(ROOT).as_posix()]=sha(p)
 save(O/'code_freeze.json',f)
 (O/'RUN_STATE.md').write_text('FROZEN:original13 dates x FRAME_GUARD/FIXED_512 x2reverse timing=52serial replays. Preflight pending; no training or tuning.\n',encoding='utf8')
 print(json.dumps(dict(runs=len(order),dates=13,repeats=2,code_items=len(f))))
if __name__=='__main__':main()
