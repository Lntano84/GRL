"""Append-only publication of all original-date fixed256 regression evidence."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid87';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['before','complete']);a=ap.parse_args()
 assert read(OUT/'preflight.json')['passed']
 if a.phase=='complete':assert read(OUT/'review.json')['passed_saved_data']
 dest=ARCH/'protocols'/f'GRID87_{a.phase}';assert not dest.exists()
 files=list(Path(__file__).parent.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']];entries=[]
 for p in files:
  q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p);entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
 (dest/'archive_receipt.json').write_text(json.dumps(dict(phase=a.phase,files=entries),ensure_ascii=False,indent=2),encoding='utf8')
 note='GRID87固定256旧日期质量回归：全部13个原2035日期各新增1条FIXED_256，逐字复用GRID86规则与原物理环境，沿用模型/控制器/缓存/候选顺序；不训练、不调参或换预算。FULL65/68和GUARD70仅作质量/存活/完整周成本/查询参照，不混历史计时。逐日期保存所有结果，并定位相同前缀的首次动作分歧；已暴露开发回归，不称独立确认。'
 if a.phase=='complete':note+=' 保存数据审计通过，冻结裁决：'+read(OUT/'decision.json')['overall']+'。'
 for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
  p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf8')+f'\n## 2026-10-16 — GRID87 {a.phase}\n\n'+note+'\n',encoding='utf8')
 p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **最新：** '+note+'\n'+''.join(lines[1:]),encoding='utf8')
 (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+('按GRID87全部13日期结果及首次动作分歧更新贡献证据，接受固定强基线或明确额外核查的适用边界。' if a.phase=='complete' else '由GRID87已有pipeline完成13条串行回放、审计和归档；不启动第二运行器。')+'\n2. 保留GRID86固定256在六生成日期已具竞争力及全部历史裁决。\n3. 本轮不调查询预算，不以旧批计时计算提速，不训练新模型。\n',encoding='utf8')
 p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf8')+f'\n## GRID87 {a.phase}\n\n'+note+f'\n\n[方案与结果](protocols/GRID87_{a.phase}/outputs/grid87/)\n',encoding='utf8')
 if a.phase=='complete':
  folder=ROOT/'论文协作资料/FRAME_固定256旧日期回归_20261016';folder.mkdir(exist_ok=False)
  for p in OUT.iterdir():
   if p.is_file() and (p.suffix in ['.md','.csv'] or p.name in ['decision.json','design.json','first_divergences.json']):shutil.copy2(p,folder/p.name)
 print(json.dumps(dict(phase=a.phase,files=len(entries))))
if __name__=='__main__':main()
