"""Verify this new shutdown delivery once; no raw archive/trajectory reread."""
from pathlib import Path
import hashlib,json,subprocess,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid297_paired_collection_preparation';A=I/'research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation';C=R/'论文协作资料/FRAME_配对采集准备_GRID297'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'publication_closeout.json').exists();assert not git('status','--porcelain').strip()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
blobs={}
for folder in ['runtime_start','runtime_collection']:
 for p in (A/folder).rglob('*'):
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
for rel,h in read(A/'runtime_collection/archive_manifest.json').items():assert sha(A/'runtime_collection'/rel)==h
for rel,h in read(C/'runtime_delivery_hashes.json').items():assert sha(C/rel)==h==sha(O/rel)
r=read(O/'runtime_complete.json');assert r['accepted_episodes']==288 and r['pairs']==144 and r['new_fits']==r['new_labels_derived']==0
B=Path(r['raw_archive']);done=read(B/'archive_complete.json');assert done['passed'] and done['manifest_sha256']==sha(B/'raw_records_manifest.json')==r['raw_archive_manifest_sha256']
q=dict(passed=True,result_sha=head,remote_verified=True,exact_new_Git_blobs=blobs,exact_blob_count=len(blobs),D_runtime_copy_verified=True,raw_archive_readback_receipt_verified_without_repeat=True,source_collection_288_complete=True,user_shutdown_pause=True,grl_status='PAUSED',new_models=0,new_labels=0,new_predictions=0,new_physics=0)
save(O/'publication_closeout.json',q);S=A/'runtime_final_receipts';S.mkdir(exist_ok=False)
for p in [O/'publication_closeout.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',Path(__file__)]:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()});shutil.copy2(O/'publication_closeout.json',C/'publication_closeout.json');print('GRID297_SHUTDOWN_PUBLICATION_PASSED',len(blobs),head)
