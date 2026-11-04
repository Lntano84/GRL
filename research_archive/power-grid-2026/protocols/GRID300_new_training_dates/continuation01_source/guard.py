"""New explicit post-generation scope, same original absolute total deadline."""
from pathlib import Path
import hashlib,importlib.util,json,os,subprocess,time
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid300_new_training_dates';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';SCOPE='GRID300_POST_GENERATION_CONTINUATION01'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');t.replace(p)
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def validate_receipt(a,design_hash,freeze_hash):
 old=read(O/'collection_started.json');fields={'scope_id':SCOPE,'generation_new':0,'generation_reused':16,'baseline_runs':144,'max_opposite_runs':576,'max_runs':720,'original_start_monotonic':old['monotonic_start'],'original_deadline_monotonic':old['deadline'],'worker_cap_s':1200,'physical_steps_cap':1452240,'forecasts_cap':22500000,'design_sha256':design_hash,'source_freeze_sha256':freeze_hash,'original_authorization_sha256':sha(O/'physical_authorization.json'),'failed_generation_gate_preserved':True}
 if a.get('approved_by_user') is not True or not a.get('user_instruction') or not a.get('user_message_reference'):raise PermissionError('explicit one-time post-generation continuation approval missing')
 if any(a.get(k)!=v for k,v in fields.items()):raise PermissionError('continuation scope differs or original clock was reset')
def check(ctx):
 if (O/'STOP_REQUEST.json').exists():raise RuntimeError('user stop; no automatic retry')
 original=read(O/'collection_started.json')['deadline']
 if ctx['deadline_monotonic']>original:raise PermissionError('cannot extend original absolute deadline')
 if time.perf_counter()>=min(original,ctx['deadline_monotonic'],ctx.get('worker_deadline_monotonic',original)):raise TimeoutError('original total window expired')
def require_authorization():
 p=O/'continuation01_authorization.json'
 if not p.exists():raise PermissionError('continuation01 authorization absent; no real model/environment imports')
 a=read(p);validate_receipt(a,sha(O/'continuation01_design.json'),sha(O/'continuation01_source_freeze.json'));check({'deadline_monotonic':read(O/'collection_started.json')['deadline']})
 head=git('rev-parse','HEAD').decode().strip()
 for local,rel in [(p,'continuation01_authorization/continuation01_authorization.json'),(O/'continuation01_source_freeze.json','continuation01_source/source_freeze.json'),(O/'continuation01_preparation_review.json','continuation01_results/preparation_review.json')]:
  if git('show',head+':'+(A/rel).relative_to(I).as_posix())!=local.read_bytes():raise PermissionError('exact continuation source/approval/control check not published')
 if git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head:raise PermissionError('remote HEAD differs')
 if not read(O/'continuation01_preparation_review.json')['passed']:raise PermissionError('continuation control check incomplete')
 return a
def verify_frozen():
 for name in ['continuation01_source_freeze.json','model_freeze.json','static_freeze.json']:
  for p,h in read(O/name).items():
   if sha(R/p)!=h:raise PermissionError('frozen continuation dependency changed: '+p)
def ancestor(pid):
 s=importlib.util.spec_from_file_location('g300c_original_pid',R/'work/grid291_control_preparation/guard.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.manager_ancestor_matches(pid)
def claim(kind,index):
 require_authorization();verify_frozen();ctx=read(O/'run_context.json');c=read(O/'claims_continuation01'/f'{kind}_{index:03d}.json')
 if kind!='replay' or ctx['kind']!=kind or ctx['index']!=index or c['kind']!=kind or c['index']!=index or c['manager_pid']!=ctx['manager_pid'] or c['token']!=ctx['token'] or os.environ.get('GRL_GRID300_TOKEN')!=ctx['token'] or not ancestor(ctx['manager_pid']):raise PermissionError('continuation worker not claimed by exact manager')
 check(ctx)
 with (O/'claims_continuation01'/f'{kind}_{index:03d}.started').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
 return ctx
def verify_stage(name):
 head=git('rev-parse','HEAD').decode().strip()
 if git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head:raise PermissionError('stage remote differs')
 if name=='generation':
  # Independent publication of reused sources, NOT release of the failed2h gate.
  for n in ['generated_freeze.json','generation_manifest.json','collection_error.json','collection_started.json']:
   if git('show',head+':'+(A/'continuation01_source'/n).relative_to(I).as_posix())!=(O/n).read_bytes():raise PermissionError('reused generation provenance differs')
  return {'reused_generation':16,'old_gate_released':False}
 ready=read(O/'baseline_ready.json');release=read(O/'baseline_publication.json')
 if name!='baseline' or release['ready_sha256']!=sha(O/'baseline_ready.json'):raise PermissionError('baseline release differs')
 for n,h in ready['files'].items():
  if sha(O/n)!=h or sha(O/'stage_payload/baseline'/n)!=h or hashlib.sha256(git('show',head+':'+(A/'baseline_public'/n).relative_to(I).as_posix())).hexdigest()!=h:raise PermissionError('baseline source identity differs')
 return ready
def verify_generated(week):
 verify_stage('generation');prefix=read(O/'design.json')['environment']+'/chronics/'+week+'/';chosen={p:h for p,h in read(O/'generated_freeze.json').items() if p.startswith(prefix)}
 if not chosen:raise PermissionError('no reused scenario')
 for p,h in chosen.items():
  if sha(R/p)!=h:raise PermissionError('reused scenario bytes changed')
