"""Finite same-information conference-adaptation comparison, append-only."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid78'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID78_{a.phase}';assert not dest.exists();assert read(OUT/'preflight.json')['passed']
    if a.phase=='before':assert not (OUT/'manifest.json').exists()
    else:assert read(OUT/'review.json')['passed_saved_data']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid78').glob('*.py'))+[Path(__file__).resolve()]
    files+=list((ROOT/'work/grid78_review').glob('*.py'))
    files += [p for p in OUT.iterdir() if p.is_file() and p.suffix in ['.json','.md','.csv']]
    if a.phase=='complete':
        for folder in (OUT/'runs').iterdir():files.extend(folder/f for f in ['metadata.json','summary.json','components.json'] if (folder/f).exists())
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),scope='Same-context grid-mode source adaptation, seven exposed dates, no independent AC re-solve or new paired timing superiority.'),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID78有限14条回放：旧七日期×两个共同校准模型，会议适配具有相同公开上下文与完整核查宏动作；不改排序/控制/日期/模型，按质量及查询比较。单次时间只作描述。'
    if a.phase=='complete':
        r=read(OUT/'review.json');note+=' 已完成并核验；'+'；'.join(f"{s['method']}质量{s['quality_dates']}/7、查询{s['queries']}，普通门控相对省查询{s['context_query_reduction_vs_meta']*100:+.3f}%" for s in r['by_seed'])+'。不据此冒充原任务SOTA复现或结构新颖性证明。'
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID78 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    next_text=('# Next steps\n\n1. GRID76十三日期及GRID78同信息/核查权限会议适配比较完成并核验，完整结果见GRID78_report.md。\n2. 本轮不再扫模型/费用或按日期择种子。下一次若主张实际控制时间优势，须单独冻结同批平衡计时；不能复用旧计时比值替代。\n3. 结构特征增量尚未建立。论文贡献围绕减少候选预测的条件核查框架，成熟停止/参数控制或普通RF不冒充原创；所有坏结果保留，用户旧稿不覆盖。\n' if a.phase=='complete' else '# Next steps\n\n1. 完成GRID78固定14条回放及公开上下文/反馈/账目审计，两个种子及不利日期全部保留。\n2. 根据质量与查询结果判断廉价门控是否超过相同信息/权限会议适配；没有新配对时间前不宣称实际控制加速。\n3. 结构特征增量尚未成立；贡献围绕条件核查框架，不能把成熟停止/参数控制或普通RF当作原创。\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_text,encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(phase=a.phase,files=len(entries),historical_preserved=len(previous))))
if __name__=='__main__':main()
