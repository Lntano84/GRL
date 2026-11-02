"""Saved preparation delivery review only; no repeated qualification or models."""
from pathlib import Path
import json,hashlib,subprocess,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid297_paired_collection_preparation';A=I/'research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation';C=R/'论文协作资料/FRAME_配对采集准备_GRID297'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
assert not (O/'publication_review.json').exists();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
assert not git('status','--porcelain').strip();blobs={}
for p in A.rglob('*'):
 if p.is_file():
  rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes(),rel;blobs[rel]=sha(p)
for p in A.rglob('archive_manifest.json'):
 for rel,h in read(p).items():assert sha(p.parent/rel)==h,(p,rel)
for rel,h in read(C/'delivery_hashes.json').items():assert sha(C/rel)==h and sha(O/rel)==h,rel
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
assert read(O/'preparation_review.json')['passed'] and read(O/'complete.json')['preparation_only']
assert not any((O/x).exists() for x in ['physical_authorization.json','collection_started.json','runs'])
q=dict(passed=True,result_sha=head,remote_verified=True,exact_git_blobs=blobs,exact_blob_count=len(blobs),archive_manifest_verified=True,D_copy_verified=True,preparation_qualified=True,models_published_byte_checks=84,physical_authorized=False,physical_started=False,new_model_loads=0,new_predictions=0,new_fits=0,new_labels=0,new_physics=0,previous_upload_review_rejection_resolved_by_explicit_user_approval=True)
save(O/'publication_review.json',q);S=A/'final_receipts';S.mkdir(exist_ok=False)
for p in [O/'publication_review.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',Path(__file__)]:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()});shutil.copy2(O/'publication_review.json',C/'publication_review.json')
print('GRID297_DELIVERY_PASSED',len(blobs),head)
