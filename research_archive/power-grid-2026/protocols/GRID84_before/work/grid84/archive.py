"""Append-only staged publication; excludes raw physics traces/data packages."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid84';REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['before','models','complete']);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID84_{a.phase}';assert not dest.exists()
    if a.phase=='complete':assert read(OUT/'review.json')['passed_saved_data']
    if a.phase=='models':assert read(OUT/'preflight.json')['passed']
    files=list(Path(__file__).parent.glob('*.py'))+[p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv'] and p.name not in ['pipeline_error.json']]
    if a.phase!='before':files+=list((OUT/'models').rglob('*.json'))+list((OUT/'models').rglob('*.pt'))
    entries=[]
    for p in files:
        target=dest/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        assert sha(p)==sha(target);entries.append(dict(source=p.relative_to(ROOT).as_posix(),sha256=sha(p)))
    (dest/'archive_receipt.json').write_text(json.dumps(dict(phase=a.phase,files=entries),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID84固定费用种子扩展：保留0/1，仅新增2/3/4各一次拟合，原129/22数据、网络、训练循环、costindex2=0均不变；不重选校准费用。FRAME与五种子同六个已暴露生成场景、两次反向计时，共72条串行回放。所有种子/失败保留，原校准质量匹配False保留；FULL只作质量/时域参照，不混旧计时。'
    if a.phase=='models':note+=' 三次拟合与训练反馈预检已完成，尚无新闭环结果。'
    if a.phase=='complete':
        d=read(OUT/'decision.json');o=d['overall'];note+=f" 已完成保存数据审计；五种子平均控制变化{o['mean_seed_macro_controller_reduction']*100:+.3f}%，逐种子提速筛查通过{o['per_seed_speed_passes']}/5，整体加强判据{o['stronger_five_seed_support']}。"
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-16 — GRID84 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **最新：** '+note+'\n'+''.join(lines[1:]),encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+('阅读GRID84_report.md与五种子完整分布，更新中文实验稿，不挑种子或按结果调参。' if a.phase=='complete' else '按冻结GRID84方案完成三新种子及72条串行比较；已有pipeline负责，不启动第二运行器。')+'\n2. 主方法FRAME_GUARD；旧稿/旧模型全部保留。原会议适配的FRAME委托、费用0和校准未匹配如实披露。\n3. 六日期已暴露；本轮只检验现有适配训练波动，不称新的独立确认或原SOTA完整复现。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID84 {a.phase}\n\n'+note+f'\n\n[方案与结果](protocols/GRID84_{a.phase}/outputs/grid84/)\n',encoding='utf-8')
    if a.phase=='complete':
        collab=ROOT/'论文协作资料/FRAME_五种子会议对照_20261016';collab.mkdir(exist_ok=False)
        for p in OUT.iterdir():
            if p.is_file() and p.suffix in ['.md','.csv'] or p.name in ['decision.json','behavior.json','design.json']:shutil.copy2(p,collab/p.name)
    print(json.dumps(dict(phase=a.phase,files=len(entries))),flush=True)
if __name__=='__main__':main()
