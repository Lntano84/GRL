import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid61';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args()
if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
else:assert read(OUT/'review.json')['passed_saved_data']
dest=ARCH/'protocols'/('GRID61_'+a.phase);assert not dest.exists();dest.mkdir();receipts=[]
files=list((ROOT/'work/grid61').glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.csv','.md']]
if a.phase=='complete':
    for folder in (OUT/'runs').iterdir():files += [folder/'summary.json',folder/'metadata.json']
for src in files:
    target=dest/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);assert sha(src)==sha(target)
    receipts.append({'path':src.relative_to(ROOT).as_posix(),'sha256':sha(src),'bytes':src.stat().st_size})
(dest/'manifest.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
note='GRID61同频次顺序的框架机制矩阵：FULL/加强首个安全候选/去局部补全/现有FRAME，四旧开发日期、两个平衡重复32条；仅改变停止/补全策略。原GRID28基础控制、合法反馈和共有DC/safeQP缓存不变；无GNN拟合、阈值调节或外部基线扩散。'
if a.phase=='complete':
    R=read(OUT/'review.json');note+='裁决'+R['verdict']+'；'+ ';'.join(f"{s['method']}质量{s['quality_pass']}/8、控制减少{s['macro_control_reduction_full']:.3%}" for s in R['summaries'])+'。保留全部共同失败、较慢和较差结果；同排序机制比较不是新独立确认。'
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    f=REPO/'docs'/name;f.write_text(f.read_text(encoding='utf-8')+f'\n## 2026-10-13 — GRID61 {a.phase}\n\n'+note+'\n',encoding='utf-8')
f=REPO/'docs/RESEARCH_STATE.md';f.write_text('# Research state — framework mechanism\n\n- **最新：** '+note+'\n\n'+f.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
f=ARCH/'README.md';f.write_text(f.read_text(encoding='utf-8')+f'\n## GRID61 {a.phase}\n\n'+note+'\n',encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 完成GRID61冻结32条同排序闭环机制比较及保存向量/反馈/成本/计时审计。\n2. 重点检验渐进补查与变电站补全的质量—计算作用；若首个安全候选强参照已足够则接受，不调参数救结果。\n3. GNN是可替换辅助，暂不训练；论文区分既有混合控制/Top-k/验证部件、共有编译复用与本框架的具体差别。\n',encoding='utf-8')
print(json.dumps({'phase':a.phase,'files':len(receipts)}))
