import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid59';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args()
if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
else:assert read(OUT/'review.json')['passed_saved_data']
dest=ARCH/'protocols'/('GRID59_'+a.phase);assert not dest.exists();dest.mkdir();entries=[]
files=list((ROOT/'work/grid59').glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
if a.phase=='complete':
    for f in (OUT/'runs').iterdir():files += [f/'summary.json',f/'metadata.json']
for p in files:
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries},indent=2),encoding='utf-8')
if a.phase=='before':note='GRID59冻结16条四旧开发周有限配对试点：频次先验GNN两种子、强STATIC_ID、FULL；原快控制/停止/合法信息/DC安全QP缓存不变，单次平衡串行计时。无新拟合或最终八日期评价；先验证两种子质量，再按冻结查询条件看额外学习信号，不新增先验/常数变体。'
else:
    R=read(OUT/'review.json');note='GRID59有限频次先验闭环及保存数据审计完成；'+'；'.join(f"{s['method']}质量{s['quality_pass']}/4、对STATIC查询少{100*s['mean_query_reduction_static']:.3f}%、对STATIC控制少{100*s['mean_control_reduction_static']:.3f}%" for s in R['summaries'])+f"；裁决{R['verdict']}。单次旧开发配对，不是封存确认。原新先验/模型/失败全部保留。"
p=REPO/'docs/RESEARCH_STATE.md';p.write_text('# Research state — frequency-prior pilot\n\n- **最新：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — GRID59 '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID59 '+a.phase+'\n\n'+note+'\n',encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 完成GRID59冻结16条旧开发配对试点及保存数据审计。\n2. 不挑种子，不按这些日期改参数；质量失败或查询信号弱则停止该先验修订。\n3. 只有两种子质量/额外查询条件均通过才另定确认；当前材料与全部反例保留。\n',encoding='utf-8')
print(json.dumps({'phase':a.phase,'files':len(entries)}))
