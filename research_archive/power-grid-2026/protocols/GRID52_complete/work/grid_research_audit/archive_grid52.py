"""Archive each fresh-candidate confirmation milestone, preserving historical bytes."""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration'
ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid52'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','acquired','complete'],required=True);phase=parser.parse_args().phase
D=read(OUT/'design.json');assert read(OUT/'preflight.json')['passed']
if phase=='before':assert not (OUT/'extraction_complete.json').exists() and not (OUT/'manifest.json').exists()
elif phase=='acquired':assert read(OUT/'extraction_complete.json')['passed'] and not (OUT/'manifest.json').exists()
else:assert read(OUT/'review.json')['passed_saved_data']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID52_'+phase);assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid52').glob('*.py'))+[Path(__file__).resolve(),ROOT/'work/grid50/online_adapter.py',ROOT/'work/grid50/compressed_residual.py']
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv','.sha256']]
if phase=='complete':
    for folder in (OUT/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json']
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),
    'scope':'Fresh8date-family confirmation,candidate2seeds,3baseline families,96serialruns. Original/failed versions retained. Large data/weights/vectors local with hashes.'},indent=2),encoding='utf-8')
if phase=='before':
    note='GRID51软压缩候选保留：旧开发完成周成本与FULL相同、控制减少14.3%–15.1%，不声称已胜PPO/SoftGNN。GRID52新版新八日期确认已在解压前冻结：两模型种子、三类主对照、两次平衡计时共96条；全部先前暴露/训练/开发/已提取日期排除，按季度固定哈希选日期和变体。Q1/Q3剩余各两日期，选择受限如实披露。无新训练/教师，原GRID44结果及质量损失完整保留。'
    nxt='1. 只解压冻结8首选序列并验字节/哈希，不按数据换日期。\n2. 执行96条新版同条件确认；全部方法共有原快控制、缓存、合法信息，计时串行。\n3. 最终版仅补必要归因与论文材料，主对照不扩；LG01暂停。\n'
elif phase=='acquired':
    x=read(OUT/'extraction_complete.json');note=f"GRID52八冻结首选序列解压核验完成：{len(x['files'])}文件、{x['payload_bytes']}字节。尚无控制器评价/新拟合，96条固定确认待运行。"
    nxt='1. 执行96条冻结串行确认，逐条落盘续跑，不并行训练/安装。\n2. 完成后统一报告质量、存活、实际时间、查询及PPO/SoftGNN增量，不择种子。\n3. LG01暂停。\n'
else:
    R=read(OUT/'review.json');note='GRID52新版八新日期96条确认及保存数据核验完成；'+'；'.join(f"{r['method']}完成{r['completed_runs']}/16、质量{r['quality_qualified_pairs']}/16、控制{r['control_s']:.2f}s" for r in R['summaries'])+'。新版两种子均通过原质量—时间筛查，完整周最大成本增幅均0.08045%，对FULL宏平均控制时间减少14.465%/14.748%、查询少70.00%/70.62%；平均两种子时间减少14.606%（八家族bootstrap95%[7.240%,22.175%]）。但控制时间比PPO约长3.24%–3.57%、比SoftGNN1约长4.34%–4.60%，不能主张全面领先；PPO最大成本增幅12.434%，SoftGNN两种子最大增幅0.177%/0.169%且均通过质量筛查。48/48计时重复路径一致。全部种子/失败保留；两次重复不是独立家族；SoftGNN为资源匹配适配；GNN独有归因未自动建立，不把旧GRID44失误视为已经修复。'
    nxt='1. 保留新版通过质量—时间确认的事实及未普遍快于学习对照的事实，不在已暴露GRID52上择种子调参。\n2. 优先以已有计时做轻量组件分解，查少查询却比SoftGNN慢的开销，不新增训练/全面扫参。\n3. 固定最终版必要的PPO先验/图评分归因范围，整理质量—速度、训练成本、方法主表和可运行入口；LG01暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — fresh compressed-candidate confirmation\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID52 '+phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID52 '+phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'phase':phase,'files':len(entries),'historical_preserved':len(old)}))
