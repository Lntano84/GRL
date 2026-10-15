"""Append-only repair records. Preserve every historical artifact manifest."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid69'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','pilot','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID69_{a.phase}';assert not dest.exists()
    if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
    else:assert read(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'))['passed_saved_data']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid69').glob('*.py'))+[Path(__file__).resolve()]
    files+=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.csv')]
    if a.phase!='before':
        for folder in (OUT/'runs').iterdir():files+=[folder/f for f in ['metadata.json','summary.json','components.json'] if (folder/f).exists()]
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        assert sha(p)==sha(q);entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),scope='Finite exposed-date repair development. Original models/results preserved; no new training or independent AC re-solve.'),ensure_ascii=False,indent=2),encoding='utf-8')
    if a.phase=='before':note='GRID69单一修复冻结：取消rho安全时128查询直接返回；所有压缩返回先完成至少256查询与连续两批无改进，再同站补全。先七月、十二月及旧三月三日期FULL/修复版6条；仅质量与查询门槛通过后运行其余10已暴露日期。8项手算/权限检查、308条旧搜索复现通过。原版不改，0.5%旧筛查不改，0.64%仍可连续报告为小幅增加，不删不利日期；无训练。'
    else:
        r=read(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'))
        note=f"GRID69 {a.phase}完成保存数据核验，{r['runs']}条、质量{r['quality_dates']}/{len(r['dates'])}、完成{r['complete_dates']}/{len(r['dates'])}；裁决{r['verdict']}。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。"
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID69 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text(t.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+t.split('\n',1)[1],encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n工作区 D:/Codex/GRL研究。\n\n1. 完成GRID69单一提前返回修复及三日期试验，审计后按冻结门槛决定是否继续其余10日期；不扫参数。\n2. 保留GRID68和原FRAME所有失败；新稿突出查询/计算减少，完整质量结果留正文表或附录。\n3. 修复结果明确后更新单独中文修订稿，旧稿不覆盖；框架贡献优先，GNN创新不强行主张。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID69 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(phase=a.phase,files=len(entries),historical_preserved=len(previous))),flush=True)
if __name__=='__main__':main()
