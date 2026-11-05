"""A new, explicitly approved scope. Old starts never create a new budget."""
from pathlib import Path
import datetime, functools, hashlib, importlib.util, json, os, subprocess, time
R=Path('D:/Codex/GRL研究'); W=Path(__file__).parent
O=R/'outputs/grid301_remaining_collection'; OLD=R/'outputs/grid300_new_training_dates'
I=R/'work/GRL-results-integration'; A=I/'research_archive/power-grid-2026/protocols/GRID301_remaining_collection'
SCOPE='GRID301_REMAINING_BASELINES_AND_OPPOSITES'
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp')
 t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8'); t.replace(p)
def git(*args): return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*args])
@functools.lru_cache(maxsize=1)
def boot_id():
 return subprocess.check_output(['powershell','-NoProfile','-Command',"(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')"]).decode('utf-8-sig').strip()
def expected_fields(design_hash,freeze_hash):
 return dict(scope_id=SCOPE,generation_new=0,generation_reused=16,reused_baseline_receipts=35,skipped_partial_index=35,new_baseline_runs=108,max_new_opposite_runs=572,max_new_runs=680,wall_cap_s=86400,worker_cap_s=1200,physical_steps_cap=1371560,forecasts_cap=22500000,design_sha256=design_hash,source_freeze_sha256=freeze_hash)
def validate_receipt(a,design_hash,freeze_hash):
 if a.get('approved_by_user') is not True or not a.get('user_instruction') or not a.get('user_message_reference'): raise PermissionError('new GRID301 physical scope is not approved')
 if any(type(a.get(k)) is not type(v) or a.get(k)!=v for k,v in expected_fields(design_hash,freeze_hash).items()): raise PermissionError('new scope or budget differs')
def validate_clock(ctx,launch,now_mono,now_utc,current_boot):
 if current_boot!=launch['boot_id'] or ctx['boot_id']!=launch['boot_id']: raise PermissionError('different OS boot; no restart')
 for k in ['start_monotonic','deadline_monotonic','start_utc_epoch','deadline_utc_epoch','boot_id']:
  if ctx[k]!=launch[k]: raise PermissionError('runtime clock identity differs')
 if launch['deadline_monotonic']-launch['start_monotonic']!=86400 or launch['deadline_utc_epoch']-launch['start_utc_epoch']!=86400: raise PermissionError('budget extension or reset')
 if now_mono<launch['start_monotonic'] or now_utc<launch['start_utc_epoch']-2: raise PermissionError('clock moved backwards')
 if now_mono>=min(launch['deadline_monotonic'],ctx.get('worker_deadline_monotonic',launch['deadline_monotonic'])) or now_utc>=launch['deadline_utc_epoch']: raise TimeoutError('new approved window reached')
def check(ctx):
 if (O/'STOP_REQUEST.json').exists() or (OLD/'STOP_REQUEST.json').exists(): raise RuntimeError('stop requested; preserve partial')
 launch=read(O/'launch_context.json'); validate_clock(ctx,launch,time.perf_counter(),time.time(),boot_id())
def require_authorization(clock=True):
 p=O/'physical_authorization.json'
 if not p.exists(): raise PermissionError('GRID301 authorization absent; no live imports')
 a=read(p); validate_receipt(a,sha(O/'design.json'),sha(O/'source_freeze.json'))
 head=git('rev-parse','HEAD').decode().strip()
 for local,rel in [(p,'authorization/physical_authorization.json'),(O/'source_freeze.json','source/source_freeze.json'),(O/'preparation_review.json','results/preparation_review.json')]:
  if git('show',head+':'+(A/rel).relative_to(I).as_posix())!=local.read_bytes(): raise PermissionError('exact approval/source/qualification not published')
 if git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head or git('status','--porcelain').strip(): raise PermissionError('HEAD/remote/worktree publication barrier')
 if read(O/'preparation_review.json').get('passed') is not True: raise PermissionError('preparation incomplete')
 if clock:
  c=read(O/'launch_context.json'); check(c)
  if c['authorization_sha256']!=sha(p): raise PermissionError('launch approval changed')
 return a
def verify_frozen():
 for p,h in read(O/'source_freeze.json').items():
  if sha(R/p)!=h: raise PermissionError('frozen dependency changed: '+p)
 for name in ['model_freeze.json','static_freeze.json']:
  for p,h in read(OLD/name).items():
   if sha(R/p)!=h: raise PermissionError('original payload changed: '+p)
def ancestor(pid):
 spec=importlib.util.spec_from_file_location('g301_pid',R/'work/grid291_control_preparation/guard.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
 return m.manager_ancestor_matches(pid)
def claim(kind,index):
 require_authorization(); verify_frozen(); ctx=read(O/'run_context.json'); c=read(O/'claims'/f'{kind}_{index:03d}.json')
 if kind!='replay' or index==35 or any(c[k]!=ctx[k] for k in ['kind','index','manager_pid','token']) or ctx['kind']!=kind or ctx['index']!=index or os.environ.get('GRL_GRID300_TOKEN')!=ctx['token'] or not ancestor(ctx['manager_pid']): raise PermissionError('not a fresh exact manager claim')
 check(ctx)
 with (O/'claims'/f'{kind}_{index:03d}.started').open('x',encoding='utf8') as f: f.write(str(os.getpid()))
 return ctx
def verify_generated(week):
 prefix=read(O/'design.json')['environment']+'/chronics/'+week+'/'
 chosen={p:h for p,h in read(OLD/'generated_freeze.json').items() if p.startswith(prefix)}
 if not chosen: raise PermissionError('no previously generated scenario')
 for p,h in chosen.items():
  if sha(R/p)!=h: raise PermissionError('scenario bytes changed')
def verify_stage(name):
 if name!='baseline': raise PermissionError('no new generation stage')
 ready=read(O/'baseline_ready.json'); release=read(O/'baseline_publication.json')
 if release['ready_sha256']!=sha(O/'baseline_ready.json'): raise PermissionError('baseline release differs')
 head=git('rev-parse','HEAD').decode().strip()
 if git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head: raise PermissionError('stage remote differs')
 for n,h in ready['files'].items():
  if sha(O/n)!=h or sha(O/'stage_payload/baseline'/n)!=h or hashlib.sha256(git('show',head+':'+(A/'baseline_public'/n).relative_to(I).as_posix())).hexdigest()!=h: raise PermissionError('source publication differs')
 return ready
