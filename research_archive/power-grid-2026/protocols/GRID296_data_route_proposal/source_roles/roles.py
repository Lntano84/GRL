"""Tenth and last document: exact pre-existing own12/fit8/cal4 date roles."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys
R=Path('D:/Codex/GRL研究'); I=R/'work/GRL-results-integration'; O=R/'outputs/grid296_data_route_proposal'; A=I/'research_archive/power-grid-2026/protocols/GRID296_data_route_proposal'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def freeze():
 p=R/'outputs/grid235_calibration_ancestry/fixed_probe_pools.json'; q=O/'source_documents/fixed_pool_roles.json'; assert not q.exists();shutil.copy2(p,q)
 d=dict(document_number=10,max_documents=10,source=p.relative_to(R).as_posix(),snapshot=q.relative_to(O).as_posix(),sha256=sha(q),purpose='Resolve all12 own12/fit8/cal4 date memberships; no labels or model arrays.',no_previous_document_reinspection=True)
 save(O/'role_source_addendum.json',d); S=A/'source_roles';S.mkdir(exist_ok=False);shutil.copy2(q,S/q.name);shutil.copy2(O/'role_source_addendum.json',S/'role_source_addendum.json');shutil.copy2(Path(__file__),S/'roles.py')
 save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
def inspect():
 assert not (O/'fixed_pool_roles_read.json').exists(); head=subprocess.check_output(['git','-C',str(I),'rev-parse','HEAD']).decode().strip()
 assert subprocess.check_output(['git','-C',str(I),'ls-remote','lntano-src','refs/heads/main']).decode().split()[0]==head
 data=json.loads((O/'role_source_addendum.json').read_text(encoding='utf8'));p=O/data['snapshot']; assert sha(p)==data['sha256']
 assert subprocess.check_output(['git','-C',str(I),'show',head+':'+(A/'source_roles/fixed_pool_roles.json').relative_to(I).as_posix()])==p.read_bytes()
 save(O/'fixed_pool_roles_read.json',dict(source_published=head,documents_total=10,data=json.loads(p.read_text(encoding='utf8')))); print(p.read_text(encoding='utf8'))
if __name__=='__main__':globals()[sys.argv[1]]()
