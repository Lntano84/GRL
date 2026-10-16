"""Freeze 13-date literal FIXED_256 quality regression; no tuning or training."""
from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;O=ROOT/'outputs/grid87'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def main():
 assert not O.exists();assert read(ROOT/'outputs/grid86/review.json')['passed_saved_data']
 for n in ['run.py','mechanism_adapter.py','pipeline.py']:
  t=(ROOT/f'work/grid86/{n}').read_text(encoding='utf8').replace('grid86','grid87').replace('GRID86','GRID87').replace('serial_benchmark_36','serial_benchmark_13')
  if n=='run.py':
   t=t.replace("    assert read(ROOT/'outputs/grid80/generation.json')['passed']\n    assert read(ROOT/'outputs/grid80/qualification.json')['passed']\n",'')
   t=t.replace("        if len(summaries)==len(D['execution_order']):save", "        assert sum(v['steps'] for v in summaries)<=D['caps']['physical_steps']\n        assert sum(v['public_forecasts'] for v in summaries)<=D['caps']['public_forecasts']\n        if len(summaries)==len(D['execution_order']):save")
  if n=='pipeline.py':t=t.replace('fixed-budget closed-loop mechanism comparison','fixed256 original13 quality regression')
  (W/n).write_text(t,encoding='utf8')
 (W/'policies.py').write_text('''"""Exact previously frozen fixed256 policy; no new rule."""
from pathlib import Path
import importlib.util
ROOT=Path(__file__).resolve().parents[2]
s=importlib.util.spec_from_file_location('grid87_fixed_reference',ROOT/'work/grid86/policies.py')
REF=importlib.util.module_from_spec(s);s.loader.exec_module(REF)
POLICIES=('FIXED_256',)
key=REF.key
def run(order,observe,policy,subids,aliases,safe=.9,limit=None,context=None):
    assert policy in POLICIES
    return REF.run(order,observe,policy,subids,aliases,safe,limit,context)
''',encoding='utf8')
 t=(ROOT/'work/grid86/environment.py').read_text(encoding='utf8').replace('work/grid80/fresh_env','work/grid03/formal_env_initialized')
 (W/'environment.py').write_text(t,encoding='utf8')
 old=read(ROOT/'outputs/grid70/design.json');base=dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,method='FIXED_256',policy='FIXED_256')
 weeks=old['weeks'];assert len(weeks)==13
 D=dict(stage='GRID87_ORIGINAL13_FIXED256_QUALITY_REGRESSION',authorization='User 开始吧 after proposal to test only fixed256 on all13 original dates.',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),weeks=weeks,methods=[base],repeats=1,pilot=[],execution_order=[dict(base,week=w,repeat=0,stage_phase='all') for w in weeks],caps=dict(episode_s=1200,wall_s=7200,physical_steps=40000,public_forecasts=1000000,new_fits=0),quality_threshold=.005,timing=read(ROOT/'outputs/grid86/design.json')['timing'],
 fixed_policy=read(ROOT/'outputs/grid86/design.json')['fixed_policy'],environment='Exact original work/grid03/formal_env_initialized model, rewards and chronics; seed0. Common GRID28 controller and shared DC/safeQP/prior caches.',
 full_reference='Frozen FULL repeat0 from GRID65 first7 and GRID68 last6; quality, full-week cost, survival and queries only.',guard_reference='Frozen GRID70 FRAME_GUARD repeat0 all13, quality/survival/cost/queries and first divergence only.',scope='13 already exposed original2035 dates, one new FIXED_256 replay per date. Development regression; no new training, model, cap, date, ordering, threshold or timing repeats. All dates and adverse results retained.',
 decision='Quality: no earlier termination than FULL; if both full weeks completed, cost<=FULL*1.005. If FIXED_256 passes13/13 and on all identical guard horizons its active-date macro query ratio<=1.02, accept cheap fixed baseline and close current extra-verification-superiority claim. If fixed quality fails and guard passes, locate first saved action divergence and revealed-feedback difference; interpret as bounded quality evidence, never general necessity. Otherwise mixed/inconclusive. No cap/model/seed/date changes or rescue experiments.',
 timing_scope='New control time, P95 and disjoint components are descriptive only. No speed ratio against previous-batch FULL/GUARD timings; one repeat cannot establish timing variance.',
 provenance='Previously exposed benchmark dates, not independent confirmation. Saved vector/cost/causal feedback audit; no independent AC re-solve.')
 O.mkdir();save(O/'design.json',D)
 data={};env=ROOT/'work/grid03/formal_env_initialized'
 for p in env.iterdir():
  if p.is_file():data[p.relative_to(ROOT).as_posix()]=sha(p)
 for w in weeks:
  files=[p for p in (env/'chronics'/w).rglob('*') if p.is_file() and '__pycache__' not in p.parts];assert files,w
  for p in files:data[p.relative_to(ROOT).as_posix()]=sha(p)
 for stage in [65,68,70]:
  p=ROOT/f'outputs/grid{stage}/finished.json';data[p.relative_to(ROOT).as_posix()]=sha(p)
 for w in weeks:
  folder=ROOT/'outputs/grid70/runs'/f'r0__{w}__FRAME_GUARD'
  for p in [folder/'steps.jsonl.gz',folder/'summary.json',*folder.glob('call*.json'),*folder.glob('call*_state.npz')]:data[p.relative_to(ROOT).as_posix()]=sha(p)
 save(O/'data_freeze.json',data)
 # Preserve all previously qualified dependencies; no altered historical files.
 f={}
 for stage in [70,86]:
  for p,h in read(ROOT/f'outputs/grid{stage}/code_freeze.json').items():
   assert sha(ROOT/p)==h,p
   if p in f:assert f[p]==h,p
   f[p]=h
 for p in [*W.glob('*.py'),O/'design.json',O/'data_freeze.json']:f[p.relative_to(ROOT).as_posix()]=sha(p)
 save(O/'code_freeze.json',f)
 (O/'RUN_STATE.md').write_text('FROZEN:13 original dates; literal FIXED_256 only. Preflight pending; no training or timing comparison with historical batches.\n',encoding='utf8')
 print(json.dumps(dict(runs=13,data_files=len(data),code_files=len(f))))
if __name__=='__main__':main()
