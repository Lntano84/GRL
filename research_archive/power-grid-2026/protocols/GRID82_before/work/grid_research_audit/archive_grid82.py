"""Preserve final-method materials and one finite external-method comparison."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid82'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True,choices=['before','complete']);a=ap.parse_args();dest=ARCH/'protocols'/('GRID82_'+a.phase);assert not dest.exists()
    assert read(OUT/'preflight.json')['passed']
    if a.phase=='complete':assert read(OUT/'review.json')['passed_saved_data']
    files=list((ROOT/'work/grid82').glob('*.py'))+list((ROOT/'outputs/grid82_method').iterdir())+[Path(__file__).resolve()]
    files.extend(p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv'])
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(q.read_bytes()).hexdigest()==h
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=h,bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,scope='Chinese final fixed-guard method/pseudocode/evidence, plus36 same-six-scenario matched comparison runs; two frozen ICAPS2022-core models, no new fits or tuning; calibration-quality limitation retained.'),ensure_ascii=False,indent=2),encoding='utf-8')
    note='按用户先完成工作1和2：已形成固定核查版FRAME_GUARD的中文方法、完整伪代码和贡献证据对应表，旧稿不覆盖；学习门控保留扩展消融。同六个已暴露生成场景比较最终固定核查版与既有ICAPS2022内核电网适配两个冻结种子，两次反向计时共36条。全部沿用模型/参数/控制底座/排序/缓存与0.5%质量条件，原共同校准质量匹配False显式保留，质量不合格不称同质量加速。'
    if a.phase=='complete':
        rr=read(OUT/'decision.json');note+=' 全部回放及保存数据审计完成；'+ '；'.join(f"{d['baseline']}:{d['category']},控制时间宏平均减少{100*d['macro_controller_reduction']:+.3f}%" for d in rr['decisions'])+'。'
    else:note+=' 453项既有训练/校准付费反馈策略一致性预检通过，禁止新增训练/调参/择种子。'
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-16 — GRID82 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+('阅读GRID82_report.md，依据最终版本比较更新中文主稿；不继续在同一场景调参。' if a.phase=='complete' else '完成冻结GRID82的36条串行回放、保存数据审计和归档；正常计时不训练、不安装、不启动第二运行器。')+'\n2. 最终方法与证据见outputs/grid82_method，原用户手写稿及旧版本不覆盖。\n3. 两个会议模型的校准质量缺口保留；六场景已暴露，不称再次独立确认；不自动扩对照或训练。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID82 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    print(json.dumps(dict(phase=a.phase,files=len(entries))))
if __name__=='__main__':main()
