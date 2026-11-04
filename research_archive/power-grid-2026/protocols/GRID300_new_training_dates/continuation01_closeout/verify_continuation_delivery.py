"""New stop/remedy delivery verification only; do not reread raw ZIP or models."""
from pathlib import Path
import hashlib,json,shutil,subprocess
R=Path('D:/Codex/GRL研究');I=R/'work/GRL-results-integration';O=R/'outputs/grid300_new_training_dates';A=I/'research_archive/power-grid-2026/protocols/GRID300_new_training_dates';C=R/'论文协作资料/FRAME_新日期采集实施_GRID300'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def git(*a):return subprocess.check_output(['git','--no-optional-locks','-C',str(I),*a])
assert not (O/'continuation01_publication_review.json').exists() and not git('status','--porcelain').strip()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
blobs={}
for sub in ['stopped_generation','continuation01_source','continuation01_results']:
 for p in (A/sub).rglob('*'):
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 for n,h in read(A/sub/'archive_manifest.json').items():assert sha(A/sub/n)==h
for sub in ['stopped_generation','continuation01']:
 for n,h in read(C/sub/'delivery_hashes.json').items():assert sha(C/sub/n)==h
stop=read(O/'generation_stop_complete.json');B=Path(read(O/'generation_stop_raw_manifest.json')['local_archive']);receipt=read(B/'archive_complete.json');assert receipt['passed'] and receipt['raw_manifest_sha256']==sha(B/'raw_records_manifest.json')==stop['archive_manifest_sha256']
assert stop['accepted_baseline_runs']==0 and read(O/'continuation01_preparation_review.json')['passed']
assert not (O/'continuation01_authorization.json').exists() and not (O/'continuation01_started.json').exists() and not (O/'runs').exists()
review={'passed':True,'result_sha':head,'remote_verified':True,'exact_new_Git_blobs':blobs,'exact_blob_count':len(blobs),'D_collaboration_verified':True,'raw_archive_readback_receipt_verified_not_repeated':True,'generated_scenarios_preserved':16,'new_replays':0,'new_model_predictions':0,'new_fits':0,'new_labels':0,'continuation_authorized':False,'old_generation_gate_failed_preserved':True}
save(O/'continuation01_publication_review.json',review);shutil.copy2(O/'continuation01_publication_review.json',C/'continuation01/continuation01_publication_review.json')
msg='停机04870cf0、受控衔接源48caca85、控制结果'+head[:8]+'；'+str(len(blobs))+'具体Gitblob、D协作与本地归档成功收据核验通过（未重读ZIP）。旧2h门失败保留；16生成源可复用，0回放/新预测/训练。一次原截止内衔接尚需用户明确批准，不自动执行。'
p=O/'CONTINUOUS_HANDOFF.md';p.write_text('# 停机与受控衔接交付核验通过\n\n'+msg+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8');shutil.copy2(p,C/'continuation01/CONTINUOUS_HANDOFF.md')
for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
 p=I/'docs'/n;p.write_text(p.read_text(encoding='utf8')+'\n\n## 2026-11-04 — 停机/衔接交付核验\n\n'+msg+'\n',encoding='utf8')
p=I/'docs/NEXT_STEPS.md';t=p.read_text(encoding='utf8').replace('1. 完成受控衔接源码/收据发布核验后，只询问一次原截止内的恢复，不给新24h。','1. '+msg+' 只询问一次新批准，不重复打扰。');p.write_text(t,encoding='utf8')
shutil.copy2(O/'decision.json',O/'decision_before_generation_stop.json');save(O/'decision.json',{'status':'GENERATION_STOPPED_WAITING_EXPLICIT_CONTINUATION01','generated_scenarios':16,'new_replays':0,'new_runtime_authorized':False,'new_training_qualified':False,'original_deadline_monotonic':read(O/'collection_started.json')['deadline'],'source_control_review_passed':True})
S=A/'continuation01_closeout';S.mkdir(exist_ok=False)
for p in [O/'continuation01_publication_review.json',O/'CONTINUOUS_HANDOFF.md',O/'NEXT_PROTOCOL_PLAN.json',O/'decision.json',O/'decision_before_generation_stop.json',Path(__file__)]:shutil.copy2(p,S/p.name)
names=list(read(C/'continuation01/delivery_hashes.json'))+['continuation01_publication_review.json'];save(C/'continuation01/delivery_hashes.json',{n:sha(C/'continuation01'/n) for n in names});shutil.copy2(C/'continuation01/delivery_hashes.json',S/'delivery_hashes.json');save(S/'archive_manifest.json',{p.name:sha(p) for p in S.iterdir() if p.is_file()})
print('GRID300_STOP_CONTINUATION_PUBLICATION_PASSED',len(blobs),head)
