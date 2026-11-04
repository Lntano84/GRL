"""Verify this coverage/design delivery once; no call/trajectory/model reread."""
from pathlib import Path
import hashlib,json,subprocess,shutil
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid299_public_state_coverage';A=I/'research_archive/power-grid-2026/protocols/GRID299_public_state_coverage';C=R/'论文协作资料/FRAME_新增训练日期方案_GRID299'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
assert not (O/'publication_review.json').exists() and not git('status','--porcelain').strip()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
blobs={}
for sub in ['source','results']:
 for p in (A/sub).rglob('*'):
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 for rel,h in read(A/sub/'archive_manifest.json').items():assert sha(A/sub/rel)==h
prior=[('GRID100_data_foundation/frozen/split_manifest.json','outputs/grid100_data_foundation/split_manifest.json'),('GRID100_data_foundation/frozen/generate.py','work/grid100_data_foundation/generate.py'),('GRID297_paired_collection_preparation/source/pool_roles.json','outputs/grid297_paired_collection_preparation/pool_roles.json'),('GRID297_paired_collection_preparation/source/model_references.json','outputs/grid297_paired_collection_preparation/model_references.json'),('GRID297_paired_collection_preparation/source/published_model_bindings.json','outputs/grid297_paired_collection_preparation/published_model_bindings.json'),('GRID297_paired_collection_preparation/runtime_collection/raw_records_manifest.json','outputs/grid297_paired_collection_preparation/raw_records_manifest.json'),('GRID298_saved_suffix_targets/results/decision.json','outputs/grid298_saved_suffix_targets/decision.json')]
old={}
for ar,local in prior:
 rel='research_archive/power-grid-2026/protocols/'+ar;p=R/local;assert git('show',head+':'+rel)==p.read_bytes(),rel;old[rel]=sha(p)
for n,h in read(C/'delivery_hashes.json').items():assert sha(C/n)==h==sha(O/n)
c=read(O/'complete.json');assert c['passed'] and c['new_fits']==c['new_predictions']==c['new_physics']==c['new_cost_labels']==0
assert read(O/'scope_choice.json')['authorized_now']=='design_only' and not read(O/'new_training_date_proposal.json').get('new_physical_authorization',False)
review={'passed':True,'result_sha':head,'remote_verified':True,'exact_new_blob_count':len(blobs),'exact_new_blobs':blobs,'exact_prior_provenance_blobs':old,'D_collaboration_verified':True,'new_generation':0,'new_physics':0,'new_predictions':0,'new_fits':0,'new_cost_labels':0,'new_collection_authorized':False,'user_scope':'new_training_date_design_only'}
save(O/'publication_review.json',review);shutil.copy2(O/'publication_review.json',C/'publication_review.json')
msg='GRID299源2b455fc4/结果'+head[:8]+'，'+str(len(blobs))+'新具体Gitblob和7旧来源blob/D协作核对通过。用户仅授权新增日期设计，0生成/回放/预测/标签/训练；16新日期/144基线/最多576相反动作的24h方案已交付，实施未启动、预算未批准。'
p=O/'CONTINUOUS_HANDOFF.md';p.write_text('# 发布核验完成\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');shutil.copy2(p,C/p.name);shutil.copy2(p,R/'outputs/grid298_saved_suffix_targets/CONTINUOUS_HANDOFF.md')
for name in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/name;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-11-04 — GRID299发布核验\n\n'+msg+'\n',encoding='utf8')
p=I/'docs/NEXT_STEPS.md';t=p.read_text(encoding='utf8').replace('1. 完成报告与具体Gitblob/D协作核验，不重复公开盘点。','1. '+msg+' 不重复公开盘点。');p.write_text(t,encoding='utf8')
names=list(read(C/'delivery_hashes.json'))+['publication_review.json'];assert all(sha(C/n)==sha(O/n) for n in names);save(C/'delivery_hashes.json',{n:sha(C/n) for n in names})
S=A/'closeout';S.mkdir(exist_ok=False)
for p in [O/'publication_review.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',C/'delivery_hashes.json',Path(__file__)]:shutil.copy2(p,S/p.name)
save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID299_PUBLICATION_PASSED',len(blobs),len(old),head)
