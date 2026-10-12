"""Preserve the one soft-compression followup and original fast default."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid50'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','complete'],required=True);phase=parser.parse_args().phase
if phase=='before':assert not (OUT/'training_finished.json').exists()
else:assert read(OUT/'audit_fit.json')['passed'] and read(OUT/'review.json')['passed_saved_screen']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID50_'+phase);assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid50').glob('*.py'))+[Path(__file__).resolve(),ROOT/'outputs/grid49/closed_after_pilot.json']
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
if phase=='complete':
    for folder in (OUT/'models').iterdir():files += [folder/'curve.json',folder/'summary.json']
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'scope':'One final activation-onlyfollowup,originalscale/data/structure/faststop retained. No GRID44training/evaluation;localweights with hashes;allseeds and oldmodels kept.'},indent=2),encoding='utf-8')
if phase=='before':
    note='按用户保留原版优势：GRID49自由残差种子0成本零且约10.69%提速，种子1成本+1.374%故不替换原版、15条未跑。GRID50只做一版软压缩2*asinh(raw/2)，尺度仍原2、局部斜率1、先验KL0.1保留；同结构/初始化/数据/目标/64轮/两种子，原快停止不变。若仍混合，停止激活形式变更，转向训练覆盖；不在GRID44调参或择种子。'
    nxt='1. 完成GRID50两次固定拟合和旧26状态复核，全部版本保留。\n2. 按两种子1733及输入资格决定有限闭环，不因训练目标降低宣布提升。\n3. 此后不继续扫激活/尺度；原版仍默认，GRID44排除训练，LG01暂停。\n'
else:
    R=read(OUT/'review.json');note='GRID50一版软压缩两次拟合及旧26状态复核完成；'+'；'.join(f"{s['ranker']}查询{s['old_queries']}→{s['queries']}、同FULL选择{s['old_same_full_choice']}→{s['same_full_choice']}/26、1733{s['known_critical_choice_match']}" for s in R['summaries'])+'。不挑种子，不重训/评价GRID44，原快模型仍默认；缓存表现不等于闭环质量/提速。'
    nxt='1. 依据两种子缓存回归安排有限旧开发闭环或保留本版；不继续扫激活/尺度。\n2. 若不稳先调查训练覆盖，不再次叠加保护吃掉速度；原模型仍默认。\n3. GRID44排除训练，LG01暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — retain fast advantage,one soft-compression followup\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID50 '+phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID50 '+phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'phase':phase,'files':len(entries),'historical_preserved':len(old)}))
