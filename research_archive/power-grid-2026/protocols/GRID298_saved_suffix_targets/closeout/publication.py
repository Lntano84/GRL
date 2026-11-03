"""One new delivery review, saved outputs only; no trajectory/model reread."""
from pathlib import Path
import hashlib,json,subprocess,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid298_saved_suffix_targets';A=I/'research_archive/power-grid-2026/protocols/GRID298_saved_suffix_targets';C=R/'论文协作资料/FRAME_后缀费用目标_GRID298'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def save(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'publication_review.json').exists() and not git('status','--porcelain').strip()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
blobs={}
for sub in ['source','results']:
 for p in (A/sub).rglob('*'):
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 for rel,h in read(A/sub/'archive_manifest.json').items():assert sha(A/sub/rel)==h
old={}
for sub,name in [('source','design.json'),('source','pool_roles.json'),('runtime_collection','raw_records_manifest.json'),('runtime_collection','runtime_complete.json'),('runtime_collection','finished.json')]:
 rel='research_archive/power-grid-2026/protocols/GRID297_paired_collection_preparation/'+sub+'/'+name;p=R/'outputs/grid297_paired_collection_preparation'/name;assert git('show',head+':'+rel)==p.read_bytes();old[rel]=sha(p)
for name,h in read(C/'delivery_hashes.json').items():assert sha(C/name)==h==sha(O/name)
s=read(O/'summary.json');d=read(O/'decision.json');assert read(O/'complete.json')['passed'] and s['physical_rows_checked']==566046 and s['cost_targets_derived']==124 and not d['new_training_qualified']
review=dict(passed=True,result_sha=head,remote_verified=True,exact_new_blob_count=len(blobs),exact_new_blobs=blobs,exact_prior_provenance_blobs=old,D_collaboration_hashes_verified=True,original_raw_archive_not_repeated=True,new_physics=0,new_predictions=0,new_fits=0,cost_targets_derived=124,training_qualified=False,decision=d['status'])
save(O/'publication_review.json',review);shutil.copy2(O/'publication_review.json',C/'publication_review.json')
p=O/'CONTINUOUS_HANDOFF.md';p.write_text('# 发布核验完成\n\n源844c7fd1、结果'+head[:8]+'，'+str(len(blobs))+'个新具体Gitblob及5个旧来源blob、D协作哈希核对通过。grl ACTIVE，无获资格的新训练/物理；支线结束，状态无新信息安静。\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
shutil.copy2(p,C/p.name);shutil.copy2(p,R/'outputs/grid297_paired_collection_preparation/CONTINUOUS_HANDOFF.md')
nextfile=I/'docs/NEXT_STEPS.md';t=nextfile.read_text(encoding='utf8');t=t.replace('1. 完成必要派生结果上传与具体Gitblob/D协作核验，不重跑readout/采集/旧归档。','1. 源844c7fd1/结果'+head[:8]+'已上传，'+str(len(blobs))+'新blob及5旧来源blob、D协作核验通过；不重跑readout/采集/旧归档。');nextfile.write_text(t,encoding='utf8')
for doc in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/doc;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-11-03 — GRID298发布核验\n\n源844c7fd1/结果'+head[:8]+'；'+str(len(blobs))+'新具体Gitblob、5旧来源blob及D协作核对通过，无新拟合/预测/物理。费用头支线不资格，原模型保留。\n',encoding='utf8')
S=A/'closeout';S.mkdir(exist_ok=False)
for p in [O/'publication_review.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',Path(__file__)]:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID298_PUBLICATION_REVIEW_PASSED',len(blobs),len(old),head)
