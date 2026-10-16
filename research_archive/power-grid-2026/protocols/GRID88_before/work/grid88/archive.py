"""Append-only publication of original13 paired FRAME/fixed512 evidence."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid88';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['before','complete']);a=ap.parse_args()
 assert read(OUT/'preflight.json')['passed']
 if a.phase=='complete':assert read(OUT/'review.json')['passed_saved_data']
 dest=ARCH/'protocols'/f'GRID88_{a.phase}';assert not dest.exists()
 files=list(Path(__file__).parent.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']];entries=[]
 for p in files:
  q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p);entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
 (dest/'archive_receipt.json').write_text(json.dumps(dict(phase=a.phase,files=entries),ensure_ascii=False,indent=2),encoding='utf8')
 note='GRID88最后一组固定预算对照：原13个已暴露2035日期，FRAME_GUARD与固定512各两次反向计时，共52条串行回放。同原物理环境、控制器、候选顺序与公共缓存，无新增训练/调参/预算/日期。质量门槛沿用0.5%成本与不早于FULL终止；双方13/13质量且时域相同时，至少5%控制/查询减少、9/13日期同向和两重复同向才支持计算优势。固定512与FRAME的实际时间仅本批配对；FULL65/68只作质量/存活/查询参照。完整保留GRID86、GRID87裁决与所有不利结果。完成后转中文稿和机制证据整合，不自动追加模型或预算实验。'
 if a.phase=='complete':note+=' 保存数据审计通过，冻结裁决：'+read(OUT/'decision.json')['overall']+'。'
 for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+f'\n## 2026-10-16 — GRID88 {a.phase}\n\n'+note+'\n',encoding='utf8')
 p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **最新：** '+note+'\n'+''.join(lines[1:]),encoding='utf8')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+('阅读GRID88全部13日期质量、同批计时和查询，整理贡献证据与中文稿。' if a.phase=='complete' else '由GRID88已有pipeline完成52条串行回放、审计和归档；不启动第二运行器。')+'\n2. 保留固定256在旧13日期质量9/13、新6日期6/6且更便宜的完整边界，不泛称所有固定预算均劣。\n3. 完成后聚焦前缀压缩、反馈核查、结构补查及实际控制开销的证据链；不继续新增采样器/模型/预算寻找赢家。\n',encoding='utf8')
 p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf8')+f'\n## GRID88 {a.phase}\n\n'+note+f'\n\n[方案与结果](protocols/GRID88_{a.phase}/outputs/grid88/)\n',encoding='utf8')
 if a.phase=='complete':
  folder=ROOT/'论文协作资料/FRAME_固定512最终对照_20261016';folder.mkdir(exist_ok=False)
  for p in OUT.iterdir():
   if p.is_file() and (p.suffix in ['.md','.csv'] or p.name in ['decision.json','design.json']):shutil.copy2(p,folder/p.name)
 print(json.dumps(dict(phase=a.phase,files=len(entries))))
if __name__=='__main__':main()
