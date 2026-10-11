"""Freeze/archive each meaningful reserved-confirmation unit; preserve old hashes."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid44'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','acquired','complete'],required=True);phase=parser.parse_args().phase
D=read(OUT/'design.json');assert read(OUT/'preflight.json')['passed']
if phase=='before':assert not (OUT/'extraction_complete.json').exists() and not (OUT/'manifest.json').exists()
if phase=='acquired':assert read(OUT/'extraction_complete.json')['passed'] and not (OUT/'manifest.json').exists()
if phase=='complete':assert read(OUT/'review.json')['passed_saved_data']
old={}
for manifest in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(manifest).get('files',[]):
        p=manifest.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID44_'+phase);assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid44').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.md','.json','.csv']]
files += [ROOT/'outputs/grid43/closed_after_pilot.json',ROOT/'outputs/grid43/RUN_STATE.md',ROOT/'outputs/grid_final_holdout/reserved_date_families.json',ROOT/'outputs/grid_final_holdout/reservation.sha256']
if phase=='complete':
    for folder in (OUT/'runs').iterdir():
        files += [folder/name for name in ['summary.json','metadata.json']]
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),
    'scope':'First8reserved families,frozen original2GNNseeds,3baseline method families,96serialruns. No refit. Large chronics/weights/physical vectors/raw logs local;all code/model hashes frozen.'},indent=2),encoding='utf-8')
if phase=='before':
    note='GRID43并列补查试点关闭：6条保留、18条未运行；不改善原模型成本且增加查询，当前仍用原GRID21权重和GRID28规则。GRID44首次8预留日期家族确认已冻结：96串行回放、两计时重复、原GNN/SoftGNN各两种子、作者PPO及FULL；预处理/缓存/控制/停止一致，先冻结后解压。不重新训练、不按确认结果调参；全部家族及变体持续排除训练。'
    nxt='1. 只解压冻结8首选序列，核对字节/哈希；不按数据替换日期。\n2. 执行96条串行确认并保存全部种子/失败，结束后核验完成周成本与相同存活长度提速。\n3. 三类主对照不扩菜单；LG01自动跟进暂停。\n'
elif phase=='acquired':
    x=read(OUT/'extraction_complete.json')
    note=f"GRID44冻结8个首选序列已按官方索引安全解压并逐文件哈希，{len(x['files'])}文件、{x['payload_bytes']}字节；尚未运行控制器或读取评价成绩。96条冻结串行确认待运行，不重新训练。"
    nxt='1. 运行96条冻结串行确认，按每条落盘支持续跑；不在正式计时并行训练/安装。\n2. 结束后按冻结审计报告成本、存活、控制提速与各对照；保留失败/较差种子。\n3. LG01保持暂停。\n'
else:
    R=read(OUT/'review.json')
    note='GRID44八预留日期家族96条串行确认及保存数据核验完成；'+ '；'.join(f"{r['method']}完成{r['completed_runs']}/16、无提前失败{r['no_earlier_failure_pairs']}/16、质量筛查{r['quality_qualified_pairs']}/16" for r in R['summaries'])+'。两原模型种子投资筛查均通过='+str(R['own_seed_average']['both_model_seeds_screen_passed'])+'；两次计时/两种子不是独立日期，SoftGNN为资源匹配适配。全部日期使用后不再叫封存，禁止据其择模型赢家。'
    nxt='1. 按GRID44共同完成周质量与相同存活长度提速定位可支持的贡献；不在已使用确认日期上择优调参。\n2. 保留全部种子/对照/失败，主对照限三类；补论文所需的可复现方法说明与已有局限。\n3. LG01保持暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — reserved-date confirmation\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['EXPERIMENT_LOG.md','DECISIONS.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-11 — GRID44 '+phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';t=p.read_text(encoding='utf-8');p.write_text(t+'\n## GRID44 '+phase+'\n\n'+note+' [协议与结果](protocols/GRID44_'+phase+'/outputs/grid44/design.json)。\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'phase':phase,'new_files':len(entries),'historical_preserved':len(old)}))
