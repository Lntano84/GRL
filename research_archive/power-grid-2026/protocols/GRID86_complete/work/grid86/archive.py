"""Append-only protocol/result publication for the finite mechanism trial."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid86';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['before','complete']);a=ap.parse_args()
    assert read(OUT/'preflight.json')['passed']
    if a.phase=='complete':assert read(OUT/'review.json')['passed_saved_data']
    dest=ARCH/'protocols'/f'GRID86_{a.phase}';assert not dest.exists()
    files=list(Path(__file__).parent.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
    entries=[]
    for p in files:
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    (dest/'archive_receipt.json').write_text(json.dumps(dict(phase=a.phase,files=entries),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID86固定预算机制闭环：最终FRAME_GUARD对固定256/512查询，同六个已暴露2036生成日期、两次反向计时共36条串行回放。固定版无同站补查或安全反馈触发扩展，无可接纳候选时返回空给共同控制器；不新增训练/调参/日期。质量与成本门槛不变，完整同质量提速另要求六日期相同时域，避免把早失败当省时。FULL仅作质量/时域参照，不混旧计时。全部预算和不利结果保留。'
    if a.phase=='complete':note+=' 保存数据审计完成，总体判读：'+read(OUT/'decision.json')['overall']+'。'
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-16 — GRID86 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **最新：** '+note+'\n'+''.join(lines[1:]),encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+('阅读GRID86_report.md与两个固定预算结果，按机制证据更新中文稿；不再调预算找赢家。' if a.phase=='complete' else '按GRID86冻结方案完成36条串行机制比较，已有pipeline负责；不启动第二运行器。')+'\n2. FRAME主线为反馈核查与计算分配，结构/GNN独立优势尚未建立；保留所有历史裁决。\n3. 固定预算与反馈方法的质量、存活和计算分别报告；六日期已暴露，不称独立确认。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID86 {a.phase}\n\n'+note+f'\n\n[方案与结果](protocols/GRID86_{a.phase}/outputs/grid86/)\n',encoding='utf-8')
    if a.phase=='complete':
        folder=ROOT/'论文协作资料/FRAME_反馈核查与固定预算_20261016';folder.mkdir(exist_ok=False)
        for p in OUT.iterdir():
            if p.is_file() and (p.suffix in ['.md','.csv'] or p.name in ['decision.json','design.json']):shutil.copy2(p,folder/p.name)
    print(json.dumps(dict(phase=a.phase,files=len(entries))))
if __name__=='__main__':main()
