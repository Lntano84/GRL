"""Administrative checkpoint only; preserve frozen scientific code and completed runs.

Run after dispatcher is stopped and the current child exits. Reconcile the last
orphaned worker result by its predeclared identity; do not run any new episode.
"""
import gzip,hashlib,json,math,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid44'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
D=read(OUT/'design.json')
for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
old=read(OUT/'manifest.json')['summaries'];oldids={(s['repeat'],s['week'],s['method']) for s in old}
summaries=[];artifacts=[]
for i,item in enumerate(D['execution_order']):
    p=OUT/f'worker_result_{i:02d}.json'
    if not p.exists():continue
    s=read(p)
    assert all(s[k]==item[k] for k in ['repeat','week','method','rule','policy','checkpoint'])
    folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
    assert read(folder/'summary.json')==s
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf-8') as f:r=[json.loads(l) for l in f]
    assert len(r)==s['steps'] and abs(math.fsum(x['raw_cost'] for x in r)-s['cost'])<1e-6
    assert abs(math.fsum(x['decision_s'] for x in r)-s['controller_s'])<1e-6
    assert sum(x['simulations'] for x in r)==s['public_forecasts']
    assert (folder/'vectors.npz').exists() and (folder/'components.json').exists()
    for name in ['summary.json','metadata.json','steps.jsonl.gz','vectors.npz','components.json']:
        a=folder/name;artifacts.append({'path':a.relative_to(ROOT).as_posix(),'bytes':a.stat().st_size,'sha256':sha(a)})
    summaries.append(s)
ids={(s['repeat'],s['week'],s['method']) for s in summaries};assert oldids<=ids
assert len(ids)==len(summaries) and summaries==[read(OUT/f'worker_result_{i:02d}.json') for i in range(len(summaries))]
save(OUT/'manifest.json',{'summaries':summaries,'planned':96})
R={'status':'STOPPED_FOR_USER_SHUTDOWN','completed_runs':len(summaries),'remaining_runs':96-len(summaries),
   'reconciled_last_worker_count':len(summaries)-len(old),'scientific_code_hashes_unchanged':True,
   'physical_rows_preserved':sum(s['steps'] for s in summaries),'public_forecasts_preserved':sum(s['public_forecasts'] for s in summaries),
   'resume_command':'.\\work\\grid02\\.venv\\Scripts\\python.exe work/grid44/run.py --phase all',
   'resume_semantics':'Skip every completed predeclared identity. Same frozen weights/code/config; no refit or restart of completed episodes.',
   'interpretation':'Partial execution checkpoint only;no confirmation verdict or test-based model change.',
   'artifact_hashes':artifacts}
save(OUT/'shutdown_checkpoint.json',R)
(OUT/'RUN_STATE.md').write_text(f"STOPPED FOR USER SHUTDOWN:{len(summaries)}/96 complete,{96-len(summaries)}pending. No training/model change. Resume original --phase all;completed identities skipped. Final audit pending.\n",encoding='utf-8')
(OUT/'RESUME.md').write_text('# 关机续跑\n\n已完成 '+str(len(summaries))+'/96 条，剩余 '+str(96-len(summaries))+' 条。原模型、配置、数据和已完成轨迹保留；整批尚未完成，不作确认裁决。\n\n在项目根目录运行：\n\n```powershell\n'+R['resume_command']+'\n```\n\n完成后运行 `work/grid44/audit.py`；不要重跑 prepare/extract，不改日期、种子或模型。LG01仍暂停。\n',encoding='utf-8')
print(json.dumps({k:R[k] for k in ['status','completed_runs','remaining_runs','reconciled_last_worker_count','physical_rows_preserved','public_forecasts_preserved']}))
