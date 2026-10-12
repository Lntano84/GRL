"""Preserve equivalent-cache qualification and one bounded old/new comparison."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['54','55'],required=True);parser.add_argument('--phase',choices=['before','complete'],required=True);a=parser.parse_args()
OUT=ROOT/('outputs/grid'+a.stage);D=read(OUT/'design.json')
if a.stage=='54':
    if a.phase=='before':assert not (OUT/'qualification.json').exists()
    else:assert read(OUT/'qualification.json')['passed']
else:
    assert read(ROOT/'outputs/grid54/qualification.json')['passed']
    if a.phase=='before':assert not (OUT/'manifest.json').exists() and read(OUT/'preflight.json')['passed']
    else:assert read(OUT/'review.json')['passed_saved_data']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID'+a.stage+'_'+a.phase);assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/('work/grid'+a.stage)).glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
if a.stage=='55':files += list((ROOT/'work/grid54').glob('*.py'))+[ROOT/'outputs/grid54/qualification.json']
if a.stage=='55' and a.phase=='complete':
    for folder in (OUT/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json']
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),
    'scope':'Equivalent public-prior metadata cache shared by old/new/PPO;one fixed-range matched comparison;all weights/failed models retained;large local data not copied.'},indent=2),encoding='utf-8')
if a.stage=='54' and a.phase=='before':
    note='GRID54先验静态元数据缓存资格已冻结：只替换909候选拓扑字典与动作映射的重复展开；每次仍读取新鲜动作字段验证快照，替换/原地变化即重建，不用可能陈旧的to_vect。130个旧开发状态精确比较先验分数/排序/最终模型排名与特征哈希，原新版与PPO公平共享；冷启动计入先验时间，无训练/候选simulate/保护改动。'
    nxt='1. 执行GRID54的130项输出等价、原地突变重建和局部计时资格。\n2. 通过后才冻结一次有限的同日期旧新版及现有强对照比较，不扩模型或baseline。\n3. 旧新权重和历史裁决全保留；LG01与GRID52通知暂停。\n'
elif a.stage=='54':
    Q=read(OUT/'qualification.json');note=f"GRID54等价先验缓存资格通过：{Q['cases']}状态分数/排序/最终排名/特征逐项一致，四项替换/原地修改/恢复重建通过；局部先验累计{Q['original_prior_s']:.3f}→{Q['cached_prior_s']:.3f}s（减少{100*Q['local_prior_time_reduction']:.2f}%），含首次冷构造与每次动作新鲜字段校验。无新训练/候选simulate；不是闭环速度或GNN独有贡献，公平给旧新PPO。"
    nxt='1. 冻结一次有限的同日期旧新版/三个既有主对照比较，全部用相同公共缓存，所有模型种子保留。\n2. 不按已暴露确认结果改模型，不新增训练或baseline。\n3. 质量—时间共同定版；LG01与GRID52通知暂停。\n'
elif a.phase=='before':
    note=f"GRID55一次有限同日期旧新版及既有主对照比较已冻结：{len(D['execution_order'])}条串行回放、{len(D['weeks'])}个既有暴露日期、两个平衡计时；原GNN/软压缩GNN/SoftGNN各两种子、作者PPO、FULL。共有GRID54先验缓存含首次构造和逐次内容核验，原GRID28快控制/缓存/合法信息不变。无新训练/参数/日期替换；此为配对选型与工程归因，不重新宣称独立确认。"
    nxt='1. 执行GRID55冻结的有限串行矩阵，逐条落盘续跑，不并行训练/安装。\n2. 按共同完整周质量、无提前失败、实际控制时间及全部两种子判断旧新版本；不只择最快种子。\n3. 共有工程收益不冒充GNN贡献；LG01与GRID52通知暂停。\n'
else:
    R=read(OUT/'review.json');note='GRID55有限同日期旧新版及既有强对照比较与保存数据核验完成；'+'；'.join(f"{r['method']}质量{r['quality_qualified_pairs']}/{r['runs']}、控制{r['control_s']:.2f}s" for r in R['summaries'])+'。各版本/全部种子/失败保留，实际质量—时间及配对差见报告；已暴露日期不当独立确认，共有缓存不作为图模型独有。'
    nxt='1. 依据GRID55同条件的成本/存活/时间与两种子稳定性固定论文主版本，保留全部对照与失败。\n2. 汇总必要的图评分/PPO先验归因、训练和教师成本及质量—速度，不新增规则/模型菜单。\n3. 整理论文和可运行复现材料；LG01与GRID52通知暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — equivalent prior cache and matched version selection\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID'+a.stage+' '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID'+a.stage+' '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'stage':a.stage,'phase':a.phase,'files':len(entries),'historical_preserved':len(old)}))
