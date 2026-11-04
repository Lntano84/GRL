"""Stdlib-only guards. Old physical approvals never authorize this new scope."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid300_new_training_dates';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates'
SCOPE='GRID300_NEW_TRAIN_DATES_SUFFIX_COLLECTION'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def validate_receipt(a,design_hash,freeze_hash,calendar_hash):
 fields={'scope_id':SCOPE,'generation_scenarios':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'wall_cap_s':86400,'generation_cap_s':7200,'generation_worker_cap_s':1800,'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000,'design_sha256':design_hash,'source_freeze_sha256':freeze_hash,'calendar_sha256':calendar_hash}
 if a.get('approved_by_user') is not True or not a.get('user_instruction') or not a.get('user_message_reference'):raise PermissionError('explicit new generation+replay scope approval missing')
 if any(a.get(k)!=v for k,v in fields.items()):raise PermissionError('new scope/source/calendar/resource receipt differs')
def require_authorization():
 p=O/'physical_authorization.json'
 if not p.exists():raise PermissionError('new physical authorization absent: zero model/environment imports')
 a=read(p);validate_receipt(a,sha(O/'design.json'),sha(O/'source_freeze.json'),sha(O/'calendar.json'))
 head=git('rev-parse','HEAD').decode().strip()
 for local,rel in [(p,'authorization/physical_authorization.json'),(O/'source_freeze.json','source/source_freeze.json'),(O/'preparation_review.json','results/preparation_review.json')]:
  if git('show',head+':'+(A/rel).relative_to(I).as_posix())!=local.read_bytes():raise PermissionError('exact authorization/source/preparation blob is not published')
 if git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head:raise PermissionError('local/remote HEAD differ')
 if not read(O/'preparation_review.json')['passed']:raise PermissionError('finite preparation incomplete')
 return a
def verify_frozen():
 for name in ['source_freeze.json','model_freeze.json','static_freeze.json']:
  for rel,h in read(O/name).items():
   if sha(R/rel)!=h:raise PermissionError('frozen bytes changed: '+rel)
def check(ctx):
 if (O/'STOP_REQUEST.json').exists():raise RuntimeError('user stop requested; preserve partial, no retry')
 if time.perf_counter()>=min(ctx['deadline_monotonic'],ctx.get('worker_deadline_monotonic',ctx['deadline_monotonic'])):raise TimeoutError('original scope deadline reached')
def ancestor(pid):
 spec=importlib.util.spec_from_file_location('g300_original_pid_guard',R/'work/grid291_control_preparation/guard.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.manager_ancestor_matches(pid)
def claim(kind,index):
 require_authorization();verify_frozen();ctx=read(O/'run_context.json');c=read(O/'claims'/f'{kind}_{index:03d}.json')
 if ctx['kind']!=kind or ctx['index']!=index or c['token']!=ctx['token'] or os.environ.get('GRL_GRID300_TOKEN')!=ctx['token'] or not ancestor(ctx['manager_pid']):raise PermissionError('worker is not claimed by exact live manager')
 check(ctx)
 with (O/'claims'/f'{kind}_{index:03d}.started').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
 return ctx
def verify_stage(name):
 ready=read(O/f'{name}_ready.json');release=read(O/f'{name}_publication.json')
 if release['ready_sha256']!=sha(O/f'{name}_ready.json'):raise PermissionError('stage release does not match manager-ready source')
 head=git('rev-parse','HEAD').decode().strip()
 if git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head:raise PermissionError('stage remote differs')
 for file,h in ready['files'].items():
  p=O/'stage_payload'/name/file;rel=(A/(name+'_public')/file).relative_to(I).as_posix()
  if sha(p)!=h or sha(O/file)!=h or hashlib.sha256(git('show',head+':'+rel)).hexdigest()!=h:raise PermissionError('stage exact local/published blob differs: '+file)
 return ready
def verify_generated(week):
 verify_stage('generation');freeze=read(O/'generated_freeze.json');prefix=read(O/'design.json')['environment']+'/chronics/'+week+'/'
 selected={p:h for p,h in freeze.items() if p.startswith(prefix)}
 if not selected:raise PermissionError('missing generated scenario freeze')
 for p,h in selected.items():
  if sha(R/p)!=h:raise PermissionError('generated scenario changed: '+p)
