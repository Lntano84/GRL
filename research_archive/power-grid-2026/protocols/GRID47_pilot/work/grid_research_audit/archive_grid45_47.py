"""Preserve post-confirmation diagnosis and a bounded common guard endpoint."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','pilot','complete'],required=True);phase=parser.parse_args().phase
assert read(ROOT/'outputs/grid45/review.json')['passed_independent_scalar_audit'] and read(ROOT/'outputs/grid46/audit.json')['passed']
if phase=='before':assert not (ROOT/'outputs/grid47/manifest.json').exists()
elif phase=='pilot':assert len(read(ROOT/'outputs/grid47/manifest.json')['summaries'])==6
else:assert read(ROOT/'outputs/grid47/review.json')['passed_saved_data']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
counts={}
for stage in ([45,46,47] if phase=='before' else [47]):
    src=ROOT/f'work/grid{stage}';output=ROOT/f'outputs/grid{stage}';dest=ARCH/'protocols'/f'GRID{stage}_{phase}';assert not dest.exists();dest.mkdir(parents=True)
    files=list(src.glob('*.py'))+[Path(__file__).resolve()]
    files += [p for p in output.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv']]
    if stage==47 and phase!='before':
        for folder in (output/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json']
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
        entries.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    (dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'scope':'GRID44unchanged;no new training/confirmation labels. Commonauthorbuffer restore,olddevelopmentonly,all rankers/seeds retained.'},indent=2),encoding='utf-8');counts[stage]=len(entries)
if phase=='before':
    note='GRID45独立复算96条确认并定位4月23日540步：两GNN与PPO均在rho0.92207平台停查、漏掉FULL844（真实奖励差0.00686），不能单动作因果归整周成本。GRID46只恢复作者原0.9缓冲要求，旧26状态×5排序器130项序列/未观测反馈核验通过，原GNN/PPO同FULL选择26/26；查询14893。GRID47冻结24条旧开发回放，先6三月，原两GNN质量及正提速通过才18其他旧日期；所有基线共用保护，不重新训练/复用GRID44确认当未见评价。回补为已有保护的质量—时间端点，不包装新安全理论。'
    nxt='1. 执行GRID47六条旧三月试点，按自身两模型质量/正提速门槛决定18条旧开发回放。\n2. 原模型/GRID44质量缺口保留；全部排序器共用保护，对照失败不删。\n3. 不调已使用确认日期，不添baseline菜单；LG01暂停。\n'
elif phase=='pilot':
    A=read(ROOT/'outputs/grid47/pilot_audit.json');note='GRID47六条共同作者阈值回补试点完成；原两GNN门槛='+str(A['own_rankers_qualified'])+'；'+'；'.join(f"{r['method']}成本差{r['cost_regret']:.6f}、时间减少{r['time_reduction']:.4f}" if r['cost_regret'] is not None else f"{r['method']}不可比" for r in A['results'])+'。所有基线与种子保留，单次开发计时。'
    nxt='1. '+('按冻结协议执行其他18条旧开发回放。' if A['own_rankers_qualified'] else '保留6条试点，分析缺口，不自动启动18条。')+'\n2. GRID44原确认及0.885%缺口不改写；LG01暂停。\n'
else:
    R=read(ROOT/'outputs/grid47/review.json');note='GRID47共同作者阈值回补24条旧开发比较与保存数据核验完成；'+'；'.join(f"{r['method']}质量{r['quality_qualified_pairs']}/4、查询{r['queries']}、控制{r['control_s']:.2f}秒" for r in R['summaries'])+'。保护是已有作者缓冲的回补，不是新安全理论或图专属；没有训练或GRID44重评价。'
    nxt='1. 汇总共同作者保护下的质量—时间端点，与原版分开保留；评价实际提速的可用范围。\n2. 若需要独立确认，另冻结新日期；GRID44永远排除训练且不再叫封存。\n3. 三类baseline范围不扩大，LG01暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — common quality guard development\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID45/47 '+phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID45–47 '+phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'phase':phase,'files':counts,'historical_preserved':len(old)}))
