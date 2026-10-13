import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid60';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['before','complete'],required=True);a=p.parse_args()
if a.phase=='before':assert not (OUT/'qualification.json').exists()
else:assert read(OUT/'qualification.json')['passed_engineering']
dest=ARCH/'protocols'/('GRID60_'+a.phase);assert not dest.exists();dest.mkdir();receipts=[]
for src in list((ROOT/'work/grid60').glob('*.py'))+[p for p in OUT.iterdir() if p.suffix in ['.json','.csv','.md']]:
    target=dest/src.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target)
    assert sha(src)==sha(target);receipts.append({'path':src.relative_to(ROOT).as_posix(),'sha256':sha(src),'bytes':src.stat().st_size})
(dest/'manifest.json').write_text(json.dumps(receipts,indent=2),encoding='utf-8')
note='用户已选择以框架作用为论文主线、GNN可替换辅助。GRID60仅做机制/近邻方法资格核对：共同频次排序下FULL、首个安全候选加增强回退、无变电站补全、现有完整框架；不新增训练/参数。'
if a.phase=='complete':
    R=read(OUT/'qualification.json');note+=f"104项合法反馈扰动与{R['existing_frame_replay_equivalent']}项原FRAME复算通过；实际局部补查改变{R['existing_local_changed_calls']}次选择。缓存状态数量/选择尚不是完整周机制因果证据。"
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    f=REPO/'docs'/name;f.write_text(f.read_text(encoding='utf-8')+f'\n## 2026-10-13 — GRID60 {a.phase}\n\n'+note+'\n',encoding='utf-8')
f=REPO/'docs/RESEARCH_STATE.md';f.write_text('# Research state — framework mechanism\n\n- **最新：** '+note+'\n\n'+f.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
f=ARCH/'README.md';f.write_text(f.read_text(encoding='utf-8')+f'\n## GRID60 {a.phase}\n\n'+note+'\n',encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 用户已选框架主线，GNN不要求成为核心；停止GRID59先验修订，保留所有版本。\n2. 完成GRID60同排序机制资格后，冻结一次有限闭环，分辨渐进停止和变电站补全的质量—控制成本作用；不新增训练、常数或外部基线家族。\n3. 共有DC/safeQP复用沿已有同路径证据单列，不将历史百分比相加；论文明确区分已知部件与待验证的组合贡献。\n',encoding='utf-8')
print(json.dumps({'phase':a.phase,'files':len(receipts)}))
