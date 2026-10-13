"""Preserve the single prior revision and all original versions."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid58';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args()
if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'training_finished.json').exists()
else:assert read(OUT/'qualification.json')['passed_engineering']
dest=ARCH/'protocols'/('GRID58_'+a.phase);assert not dest.exists();dest.mkdir();entries=[]
files=list((ROOT/'work/grid58').glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
if a.phase=='complete':files += list((OUT/'models').glob('*/summary.json'))+list((OUT/'models').glob('*/curve.json'))
for p in files:
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'scope':'One prior-only revision;no model/menu/constant sweep;large checkpoints retained locally with hashes.'},indent=2),encoding='utf-8')
if a.phase=='before':note='GRID58冻结一次先验替换：保留作者教师频次STATIC_ID为零残差基准，原软压缩GNN/尺度2/初始化/167状态148目标/64轮/两种子/优化器/停止/守卫均不变。仅两个拟合与26旧开发缓存状态资格；不训练最终八日期，不调参或新增教师。两种子须找回既知1733才另设有限旧开发闭环。'
else:
    q=read(OUT/'qualification.json');note='GRID58两个固定频次先验GNN拟合及52项旧开发缓存核验完成；'+'；'.join(f"seed{s['seed']}查询{s['queries']}、STATIC{s['static_queries']}、同FULL{s['same_full_choices']}/26、1733={s['known1733match']}" for s in q['summaries'])+f"。旧开发闭环资格={q['old_development_closedloop_eligible']}，尚未运行闭环，无新教师；缓存结果不作质量/提速证明。"
p=REPO/'docs/RESEARCH_STATE.md';p.write_text('# Research state — frequency-prior residual revision\n\n- **最新：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — GRID58 '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID58 '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 只完成GRID58两固定拟合与原26开发状态缓存资格。\n2. 两种子均找回1733才另冻结一次有限旧开发闭环，不追加先验变体/温度/常数。\n3. 质量通过后才判断实际查询/控制收益；旧频次基准与原新PPO先验版全部保留。\n',encoding='utf-8')
print(json.dumps({'phase':a.phase,'files':len(entries)}))
