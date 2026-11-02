"""Verify saved proposal delivery only, never repeat scientific inspection."""
from pathlib import Path
import hashlib,json,subprocess,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid296_data_route_proposal';A=I/'research_archive/power-grid-2026/protocols/GRID296_data_route_proposal';C=R/'论文协作资料/FRAME_后续数据方案_GRID296'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
assert not (O/'publication_review.json').exists()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
assert not git('status','--porcelain').strip()
blobs={}
for p in A.rglob('*'):
 if p.is_file():
  rel=p.relative_to(I).as_posix(); assert git('show',head+':'+rel)==p.read_bytes(),rel;blobs[rel]=sha(p)
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h
for rel,h in read(C/'delivery_hashes.json').items():assert sha(C/rel)==h and sha(O/rel)==h
for d in read(O/'design.json')['documents']:assert sha(R/d['source'])==d['sha256']
d=read(O/'role_source_addendum.json');assert sha(R/d['source'])==d['sha256']
plan=read(O/'data_collection_proposal.json');assert plan['episodes']==2*plan['pairs']==288
assert plan['fit_role_pairs']+plan['cal_role_pairs']==plan['pairs']==144
assert plan['maximum_physical_steps']==plan['episodes']*2017
assert len(read(O/'pool_role_counts.json'))==12
assert read(O/'complete.json')['new_physics']==0 and read(O/'review.json')['static_documents']==10
receipt=dict(passed=True,result_sha=head,remote_verified=True,exact_git_blobs=blobs,exact_blob_count=len(blobs),D_copy_verified=True,archive_manifests_verified=True,new_fits=0,new_labels=0,new_predictions=0,new_physics=0,automation_status='PAUSED_AUTO_REVIEW_REJECTED_REACTIVATION',automation_rejection='Review did not recognize the current direct user resume instruction after shutdown pause. Two tool rejections preserved in chat; no file or alternate scheduler workaround. Explicit async confirmation pending.',scope='This is proposal delivery, not collection authorization, scientific gain or simulator execution.')
save(O/'publication_review.json',receipt)
S=A/'final_receipts';S.mkdir(exist_ok=False)
for p in [O/'publication_review.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',Path(__file__)]:shutil.copy2(p,S/p.name)
shutil.copy2(O/'publication_review.json',C/'publication_review.json');save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID296_PUBLICATION_REVIEW_PASSED',len(blobs),head)
