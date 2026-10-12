"""Archive an existing-timing diagnosis; preserve every prior scientific artifact."""
import hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid53'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OUT/'timing_review.json')['passed'] and read(OUT/'prior_profile.json')['passed']
for r in read(OUT/'timing_inputs_manifest.json')['files']:assert sha(ROOT/r['path'])==r['sha256']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols/GRID53_complete';assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid53').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),
    'scope':'212saved-run timing reconciliation and one old visible-state prior profile. No new candidate forecasts/training; old/new dates not conflated.'},indent=2),encoding='utf-8')
note='GRID53已有212份计时/1272项逐搜索账目核查完成：GRID44原版对PPO快2.48%–3.21%、对SoftGNN1快2.25%–2.93%，但慢于SoftGNN0；GRID52新版在不同日期上对这两项均略慢，不能据跨日期比较认定模型退步。GRID51同批新版seed0快5.606%、seed1慢0.678%，结果混合。新版前向仅约2.3ms/搜索、共同PPO先验约233ms；一份旧可见状态剖析确认909次as_dict重复静态拓扑展开占先验剖析约85%。相对SoftGNN额外先验及不同路径连续优化共同形成时间差；相对PPO主要差值在连续优化，不能只归于神经网络或用公共缓存冒充图贡献。未训练、未新候选simulate、未改模型或停止规则；原新全部结果保留。'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — old/new comparison and timing diagnosis\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 对共同PPO先验的静态元数据重复展开做输出等价资格验证；所有使用相同先验的方法公平共享。\n2. 旧版、新版、失败版本全保留，不跨日期挑赢家；资格后再冻结必要的同日期配对，不追加训练或扫参。\n3. 分清GNN增量与共有工程收益，继续整理论文证据；LG01与GRID52通知暂停。\n',encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID53 timing diagnosis\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID53 timing diagnosis\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'archived':True,'files':len(entries),'historical_preserved':len(old)}))
