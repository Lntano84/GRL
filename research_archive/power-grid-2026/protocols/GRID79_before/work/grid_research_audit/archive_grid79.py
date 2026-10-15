"""Append-only balanced timing materials; no large vectors copied into Git."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid79'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID79_{a.phase}';assert not dest.exists();assert read(OUT/'preflight.json')['passed']
    if a.phase=='before':assert not (OUT/'manifest.json').exists()
    else:assert read(OUT/'review.json')['passed_saved_data']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid79').glob('*.py'))+list((ROOT/'work/grid79_review').glob('*.py'))+[Path(__file__).resolve()]
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
    if a.phase=='complete':
        for folder in (OUT/'runs').iterdir():files.extend(folder/f for f in ['metadata.json','summary.json','components.json'] if (folder/f).exists())
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),scope='Same-host reversed matched timing, unchanged policies, exposed seven dates; two timing repeats not independent dates. No independent AC re-solve.'),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID79同批平衡计时：FULL、普通上下文门控、会议适配两种子；七个原日期×两次反向重复，共56条。模型/日期/质量阈值不改，完整控制计时含图上下文、首载模型、优化与预测；P95及不重叠计算分解同时报告。604个既有付费反馈原策略一致性检查通过。'
    if a.phase=='complete':
        r=read(OUT/'review.json');note+=f" 全部56条与保存数据审计完成，门控质量{r['context_quality_dates']}/7；"+'；'.join(f"相对{x['baseline']}控制宏平均减少{x['macro_controller_reduction']*100:+.3f}%，正日期{x['positive_dates']}/7，5%稳定筛查{x['stable_5percent_screen']}" for x in r['comparisons'])+'。查询和控制时间分开，所有负面结果保留。'
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID79 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    nextstep='计时及核验已完成，见GRID79_report.md；按结果收紧或保留完整控制提速主张，不改阈值补计时。' if a.phase=='complete' else '完成GRID79固定56条串行回放，期间不训练、安装、第二运行器或重型审计；保存错误、按原冻结哈希续跑。'
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. '+nextstep+'\n2. 结构特征额外收益未建立；论文围绕条件核查流程与资源/质量结果组织，会议方法是电网适配。\n3. 全部七日期为开发，时钟重复不算独立场景；用户旧稿不覆盖，所有差版本保留。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID79 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(phase=a.phase,files=len(entries),historical_preserved=len(previous))))
if __name__=='__main__':main()
