"""Archive a single scoring-change development check without choosing a seed."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid49'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','pilot','complete'],required=True);phase=parser.parse_args().phase
if phase=='before':assert not (OUT/'manifest.json').exists()
elif phase=='pilot':assert len(read(OUT/'manifest.json')['summaries'])==5
else:assert read(OUT/'review.json')['passed_saved_data']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID49_'+phase);assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid49').glob('*.py'))+[Path(__file__).resolve(),ROOT/'work/grid48/online_adapter.py',ROOT/'work/grid48/free_residual.py']
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
if phase!='before':
    for folder in (OUT/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json']
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'scope':'One unboundedresidual scoring change,originalfaststop,alloriginal/newseeds retained,no GRID44reuse. Large localtrajectory records/weights stay localwith hashes.'},indent=2),encoding='utf-8')
if phase=='before':
    note='GRID49冻结原快速控制下的一次评分改动比较：原GNN两种子、新无硬界残差两种子、FULL；先5旧三月试点，两个新种子质量/存活合格才15其他旧开发回放。停止/保护/特征/缓存/控制均沿用原快版，不采用GRID47严格回补。没有GRID44训练/重评或新模型拟合；近期强基线原比较完整保留，不扩baseline菜单。'
    nxt='1. 完成GRID49五条旧三月试点，核验两个新种子质量，不按最快/最好的种子择优。\n2. 两者符合预定质量条件才15条旧开发比较；混合/略差保留诊断，原模型仍默认。\n3. GRID44排除训练，不继续以已看确认日期找赢家；LG01暂停。\n'
elif phase=='pilot':
    A=read(OUT/'pilot_audit.json');note='GRID49原快控制五条评分改动试点完成；两新种子质量='+str(A['both_new_seeds_qualified'])+'；'+'；'.join(f"{r['method']}成本差{r['cost_regret']:.6f}、查询{r['queries']}、控制{r['controller_s']:.2f}s" if r['cost_regret'] is not None else f"{r['method']}不可比" for r in A['results'])+'。全部原/新权重、结果保留，不选种子。'
    nxt='1. '+('按冻结协议执行15条其他旧开发回放。' if A['both_new_seeds_qualified'] else '保留5条失败/混合试点，不自动15条，原快模型仍默认；分析训练覆盖或具体失误。')+'\n2. 不改GRID44裁决、不训确认数据，不扩大规则菜单；LG01暂停。\n'
else:
    R=read(OUT/'review.json');note='GRID49原快速控制20条评分改动比较与保存数据核验完成；'+'；'.join(f"{r['method']}质量{r['quality_qualified_pairs']}/4、查询{r['queries']}、控制{r['control_s']:.2f}s" for r in R['summaries'])+'。全部种子保留；旧开发非独立确认，实际改动不自动成为默认。'
    nxt='1. 按原/新两种子实际质量—时间决定是否保留这版评分；不择最快种子。\n2. 原确认结果不改写；新模型如需确认另冻结日期，不训GRID44。\n3. LG01暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — preserve original fast-search advantage\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID49 '+phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID49 '+phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'phase':phase,'files':len(entries),'historical_preserved':len(old)}))
