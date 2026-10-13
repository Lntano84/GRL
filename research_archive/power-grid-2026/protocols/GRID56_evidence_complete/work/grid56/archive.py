"""Archive paper evidence closure once; leave older manifests untouched."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';OUT=ROOT/'outputs/grid56';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(OUT/'FINAL_AUDIT.json')['passed']
dest=ARCH/'protocols/GRID56_evidence_complete';assert not dest.exists();dest.mkdir()
files=list((ROOT/'work/grid56').glob('*.py'))+[p for p in OUT.iterdir() if p.is_file()]
entries=[]
for p in files:
    q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
    entries.append({'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size})
(dest/'manifest.json').write_text(json.dumps({'files':entries,'scope':'Audit closure and offline paper evidence only; no new fit/simulation.'},indent=2),encoding='utf-8')
note='GRID56最终审计收口通过：84项冻结哈希、9项RAW锚点、174项新补测文件收据、3842项历史归档条目、128槽位来源与21项统计/bootstrap/选型公式。无新仿真或训练，不宣称独立AC重求解。完成质量—速度主表、科学图、方法结果工作稿、贡献边界与复现入口；训练教师费用按20训练周分离为50.68分钟/146521个N1查询，最终两拟合调用89.99秒，已有PPO成本仍未知。现成STATIC_ID开发消融接近GNN，下一项仅补最终配置的廉价顺序消融，不扩主对照。'
p=REPO/'docs/RESEARCH_STATE.md';p.write_text('# Research state — paper evidence closure\n\n- **最新：** '+note+'\n\n'+p.read_text(encoding='utf-8').split('\n',1)[1],encoding='utf-8')
(REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 仅补STATIC_ID在既有八日期上的完整周质量/查询消融，不训练模型、不扩主对照家族。\n2. 单独批次控制墙钟不和历史批次形成公平提速主张；保留全部失败/较差结果。\n3. 完成后用结果收紧论文图模型归因与主表，不按这些日期改模型。\n',encoding='utf-8')
for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
    p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-13 — GRID56 evidence closure\n\n'+note+'\n',encoding='utf-8')
p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+'\n## GRID56 论文证据与审计收口\n\n'+note+'\n\n[报告](protocols/GRID56_evidence_complete/outputs/grid56/GRID56_report.md) · [方法草稿](protocols/GRID56_evidence_complete/outputs/grid56/method_and_results_draft.md)\n',encoding='utf-8')
(REPO/'docs/GRID_PAPER_EVIDENCE.md').write_text('# 当前电网论文证据\n\n'+note+'\n\n完整材料：[证据报告](../research_archive/power-grid-2026/protocols/GRID56_evidence_complete/outputs/grid56/GRID56_report.md)、[方法草稿](../research_archive/power-grid-2026/protocols/GRID56_evidence_complete/outputs/grid56/method_and_results_draft.md)、[贡献检查](../research_archive/power-grid-2026/protocols/GRID56_evidence_complete/outputs/grid56/claim_evidence.json)。\n\n历史关机交接仅记录当时状态，不表示当前仍有未完成审计。\n',encoding='utf-8')
print(json.dumps({'archived':True,'files':len(entries)}))
