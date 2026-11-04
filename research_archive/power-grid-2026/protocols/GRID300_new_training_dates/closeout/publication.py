"""One exact preparation delivery review, no model/environment requalification."""
from pathlib import Path
import hashlib,json,subprocess,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid300_new_training_dates';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';C=R/'论文协作资料/FRAME_新日期采集实施_GRID300'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
assert not (O/'publication_review.json').exists() and not git('status','--porcelain').strip()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
blobs={}
for sub in ['source','results']:
 for p in (A/sub).rglob('*'):
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 for n,h in read(A/sub/'archive_manifest.json').items():assert sha(A/sub/n)==h
prior={}
for archived,local in [('new_calendar_manifest.json','calendar.json'),('new_pool_roles.json','pool_roles.json'),('frozen_model_references.json','model_references.json')]:
 rel='research_archive/power-grid-2026/protocols/GRID299_public_state_coverage/results/'+archived;p=O/local;assert git('show',head+':'+rel)==p.read_bytes();prior[rel]=sha(p)
for n,h in read(C/'delivery_hashes.json').items():assert sha(C/n)==h==sha(O/n)
q=read(O/'preparation_review.json');assert q['passed'] and q['new_model_loads']==q['new_predictions']==q['new_generation']==q['new_physics']==0
assert not (O/'physical_authorization.json').exists() and not (O/'collection_started.json').exists() and not any((R/'work/grid300_new_training_dates/train_env/chronics').iterdir())
review={'passed':True,'result_sha':head,'remote_verified':True,'exact_new_blob_count':len(blobs),'exact_new_blobs':blobs,'exact_prior_plan_blobs':prior,'payload84_prior_exact_blob_receipt':q['source_sha'],'D_collaboration_verified':True,'live_qualification_not_repeated':True,'new_generation':0,'new_physics':0,'new_predictions':0,'new_model_loads':0,'new_fits':0,'new_labels':0,'physical_scope_approved':False,'waiting_for_user':'16generation + up to720replays /24h exact new scope'}
save(O/'publication_review.json',review);shutil.copy2(O/'publication_review.json',C/'publication_review.json')
msg='GRID300源b9ac57f6/准备结果'+head[:8]+'，'+str(len(blobs))+'新具体Gitblob及3个299方案绑定/D协作核验通过。0模型加载/生成/预测/物理/训练，准备不是720条完成或资源通过；仅等待用户16生成+至多720回放/24h新范围。'
p=O/'CONTINUOUS_HANDOFF.md';p.write_text('# 交付核验已通过\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');shutil.copy2(p,C/p.name);shutil.copy2(p,R/'outputs/grid299_public_state_coverage/CONTINUOUS_HANDOFF.md')
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-11-04 — GRID300准备发布核验\n\n'+msg+'\n',encoding='utf8')
p=I/'docs/NEXT_STEPS.md';t=p.read_text(encoding='utf8').replace('1. 完成必要准备结果/D协作/具体Gitblob核对后，只询问一次16生成+最多720条/24h新范围。','1. '+msg+' 只询问一次新范围，等待答复不重复打扰。');p.write_text(t,encoding='utf8')
names=list(read(C/'delivery_hashes.json'))+['publication_review.json'];assert all(sha(C/n)==sha(O/n) for n in names);save(C/'delivery_hashes.json',{n:sha(C/n) for n in names})
S=A/'closeout';S.mkdir(exist_ok=False)
for p in [O/'publication_review.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',C/'delivery_hashes.json',Path(__file__)]:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID300_PREPARATION_PUBLICATION_PASSED',len(blobs),head)
