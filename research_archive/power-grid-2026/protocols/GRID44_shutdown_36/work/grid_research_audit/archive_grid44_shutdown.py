"""Archive shutdown checkpoint without presenting unfinished confirmation as a result."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid44'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
R=read(OUT/'shutdown_checkpoint.json');assert R['status']=='STOPPED_FOR_USER_SHUTDOWN'
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID44_shutdown_'+str(R['completed_runs']));assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid44').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv']]
for s in read(OUT/'manifest.json')['summaries']:
    folder=OUT/'runs'/f"r{s['repeat']}__{s['week']}__{s['method']}"
    files += [folder/'summary.json',folder/'metadata.json']
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),
    'scope':'User shutdown checkpoint,unfinished confirmation,no scientific verdict. Large local vectors/logs hashed in checkpoint.'},indent=2),encoding='utf-8')
note=f"用户准备关机，GRID44在{R['completed_runs']}/96条后保存停止，剩余{R['remaining_runs']}条。全部已完成轨迹、计量与哈希保留，调度器不再派发任务；未重训/调参、尚无整批确认裁决。下次按原 --phase all 跳过已完成身份续跑，不重新准备/解压。"
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — confirmation saved for shutdown\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 用户下次继续时，从GRID44关机检查点按原run.py --phase all续跑剩余轨迹；不重跑prepare/extract。\n2. 96条完成后才跑冻结audit.py，分别报告存活、完成周成本、相同长度提速。\n3. 原模型和全部对照保留，LG01自动跟进暂停；本次停止因关机，不是研究失败。\n',encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — GRID44 user shutdown\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID44关机检查点\n\n'+note+' [续跑说明](protocols/'+dest.name+'/outputs/grid44/RESUME.md)。\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'archived_files':len(entries),'completed':R['completed_runs'],'remaining':R['remaining_runs'],'historical_preserved':len(old)}))
