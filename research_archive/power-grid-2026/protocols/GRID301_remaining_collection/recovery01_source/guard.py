"""Same original GRID301 dual clock; explicit repair source approval first."""
import importlib.util,os
from pathlib import Path
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent
s=importlib.util.spec_from_file_location('g301_original_guard',W.parent/'guard.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
O=b.O;OLD=b.OLD;I=b.I;A=b.A;read=b.read;save=b.save;sha=b.sha;git=b.git;check=b.check;boot_id=b.boot_id;ancestor=b.ancestor
verify_generated=b.verify_generated;verify_stage=b.verify_stage
SCOPE='GRID301_ZERO_MODEL_METADATA_RECOVERY01'
def fields():
 return dict(scope_id=SCOPE,original_physical_authorization_sha256=sha(O/'physical_authorization.json'),original_launch_context_sha256=sha(O/'launch_context.json'),recovery_source_freeze_sha256=sha(O/'recovery01_source_freeze.json'),metadata_code_freeze_sha256=sha(O/'code_freeze.json'),original_clock_unchanged=True,new_generation=0,new_fits=0,worker_cap_s=1200,max_new_runs=680)
def validate(a):
 if a.get('approved_by_user') is not True or not a.get('user_instruction') or not a.get('user_message_reference'):raise PermissionError('explicit recovery approval absent')
 if any(type(a.get(k)) is not type(v) or a.get(k)!=v for k,v in fields().items()):raise PermissionError('recovery source or original window differs')
def require_authorization(clock=True):
 p=O/'recovery01_authorization.json'
 if not p.exists():raise PermissionError('GRID301 recovery01 authorization absent; no live imports')
 a=read(p);validate(a);b.require_authorization(clock=True)
 head=git('rev-parse','HEAD').decode().strip()
 for local,rel in [(p,'recovery01_authorization/recovery01_authorization.json'),(O/'recovery01_source_freeze.json','recovery01_source/recovery01_source_freeze.json'),(O/'recovery01_preparation_review.json','recovery01_results/recovery01_preparation_review.json')]:
  if git('show',head+':'+(A/rel).relative_to(I).as_posix())!=local.read_bytes():raise PermissionError('recovery publication barrier')
 if read(O/'recovery01_preparation_review.json').get('passed') is not True:raise PermissionError('repair qualification missing')
 return a
def verify_frozen():
 b.verify_frozen()
 for p,h in read(O/'recovery01_source_freeze.json').items():
  if sha(R/p)!=h:raise PermissionError('recovery dependency changed: '+p)
def claim(kind,index):
 require_authorization();verify_frozen();ctx=read(O/'run_context.json');c=read(O/'claims_recovery01'/f'{kind}_{index:03d}.json')
 if kind!='replay' or index==35 or index<36 or any(c[k]!=ctx[k] for k in ['kind','index','manager_pid','token']) or ctx['kind']!=kind or ctx['index']!=index or os.environ.get('GRL_GRID300_TOKEN')!=ctx['token'] or not ancestor(ctx['manager_pid']):raise PermissionError('not exact recovery claim')
 check(ctx)
 with (O/'claims_recovery01'/f'{kind}_{index:03d}.started').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
 return ctx
