"""Preserve one bound-removal trial and every original checkpoint identity."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid48'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['before','complete'],required=True);phase=parser.parse_args().phase
assert read(OUT/'qualification.json')['passed_mathematical_checks']
if phase=='before':assert not (OUT/'training_finished.json').exists()
else:assert read(OUT/'review.json')['passed_saved_screen']
old={}
for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
    for r in read(m).get('files',[]):
        p=m.parent/r['path']
        if not p.exists():p=ARCH/r['path']
        assert sha(p)==r['sha256'];old[str(p)]=r['sha256']
dest=ARCH/'protocols'/('GRID48_'+phase);assert not dest.exists();dest.mkdir(parents=True)
files=list((ROOT/'work/grid48').glob('*.py'))+[Path(__file__).resolve()]
files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
if phase=='complete':
    for folder in (OUT/'models').iterdir():files += [folder/'curve.json',folder/'summary.json']
entries=[]
for p in sorted(set(files)):
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'historical_preserved':len(old),'scope':'One free-residual2seed fit,originaltraininginputs/initialweights preserved,originalfaststop kept. GRID44excluded from training/evaluation. Local checkpoints recorded by hashes,not claimed pretrained author weights.'},indent=2),encoding='utf-8')
if phase=='before':
    note='按用户要求保留原快速结构。GRID48训练侧资格：±2残差导致72/148最佳目标无法排第一、70/148峰值概率质量超硬界；但没有任何目标数学上被排除128候选，不能把硬界当成闭环失败原因。仅移除末端tanh硬界，保留PPO先验KL0.1、原GNN结构、同初始权重/数据/目标/64轮/两种子；不采用严格GRID47保护，不训练或重评GRID44日期。序列化NumPy布尔缺陷原版保留，资格数值按相同规则修复。两次拟合尚未开始。'
    nxt='1. 完成GRID48两个固定64轮拟合及旧26状态排序复核，全部旧模型/种子保留。\n2. 已知1733回归和合法输入检查通过才安排有限闭环；不按训练损失宣称质量/提速。\n3. 原版快速控制保留、严格保护仅质量端点；GRID44排除训练，LG01暂停。\n'
else:
    R=read(OUT/'review.json');note='GRID48移除残差硬界的两次固定拟合及旧26状态复核完成；'+'；'.join(f"{s['ranker']}查询{s['old_queries']}→{s['queries']}、同FULL选择{s['old_same_full_choice']}→{s['same_full_choice']}/26、1733匹配{s['known_critical_choice_match']}" for s in R['summaries'])+'。原模型全部保留，未训练/评价GRID44日期；仅缓存开发比较，实际质量和提速尚未证明。'
    nxt='1. 按两种子缓存复核与1733回归结果决定有限旧开发闭环，保留负向或混合结果。\n2. 不改停止/阈值，不扩大baseline菜单；原快版与严格质量端点分开保留。\n3. GRID44不训练、不再叫封存；LG01暂停。\n'
p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text('# Research state — preserve fast model and test one scoring change\n\n- **最新：** '+note+'\n\n'+t.split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n'+nxt,encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-12 — GRID48 '+phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID48 '+phase+'\n\n'+note+'\n',encoding='utf-8')
for p,h in old.items():assert sha(Path(p))==h
print(json.dumps({'phase':phase,'files':len(entries),'historical_preserved':len(old)}))
