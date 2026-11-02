"""Fail closed before model/simulator imports; old291 approval is inapplicable."""
from pathlib import Path
import hashlib,json,subprocess,importlib.util
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid297_paired_collection_preparation';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def validate_receipt(auth,design_hash,freeze_hash):
 required={'scope_id':'GRID297_PAIRED_SUFFIX_COLLECTION','runs':288,'pairs':144,'wall_cap_s':28800,'worker_cap_s':1200,'physical_steps_cap':580896,'forecasts_cap':9000000,'design_sha256':design_hash,'source_freeze_sha256':freeze_hash}
 if auth.get('approved_by_user') is not True or not auth.get('user_instruction') or not auth.get('user_message_reference'):raise PermissionError('new scoped physical approval missing')
 if any(auth.get(k)!=v for k,v in required.items()):raise PermissionError('new paired scope/source or physical budget differs')
def require_authorization(remote=True):
 p=O/'physical_authorization.json'
 if not p.exists():raise PermissionError('new physical authorization receipt missing; zero model/environment imports')
 auth=read(p);validate_receipt(auth,sha(O/'design.json'),sha(O/'source_freeze.json'))
 head=git('rev-parse','HEAD').decode().strip()
 for local,rel in [(p,'authorization/physical_authorization.json'),(O/'source_freeze.json','source/source_freeze.json'),(O/'preparation_review.json','results/preparation_review.json')]:
  if git('show',head+':'+(A/rel).relative_to(I).as_posix())!=local.read_bytes():raise PermissionError('unpublished exact source/authorization/qualification blob')
 if remote and git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]!=head:raise PermissionError('authorization not published remotely')
 if not read(O/'preparation_review.json')['passed']:raise PermissionError('preparation incomplete')
 return auth
def verify_frozen():
 for name in ['source_freeze.json','model_freeze.json','data_freeze.json']:
  for rel,h in read(O/name).items():
   if sha(R/rel)!=h:raise PermissionError('frozen source/model/input changed: '+rel)
def manager_ancestor_matches(pid):
 s=importlib.util.spec_from_file_location('g297_pid_guard',R/'work/grid291_control_preparation/guard.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m.manager_ancestor_matches(pid)
