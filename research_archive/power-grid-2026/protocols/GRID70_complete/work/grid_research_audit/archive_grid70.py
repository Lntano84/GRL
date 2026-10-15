"""Append-only repair records. Preserve every historical artifact manifest."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid70'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','pilot','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID70_{a.phase}';assert not dest.exists()
    if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
    else:assert read(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'))['passed_saved_data']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid70').glob('*.py'))+[Path(__file__).resolve()]
    files += list((ROOT/'work/grid70_review').glob('*.py'))
    files+=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.csv')]
    if a.phase!='before':
        for folder in (OUT/'runs').iterdir():files+=[folder/f for f in ['metadata.json','summary.json','components.json'] if (folder/f).exists()]
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        assert sha(p)==sha(q);entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),scope='Finite exposed-date repair development. Original models/results preserved; no new training or independent AC re-solve.'),ensure_ascii=False,indent=2),encoding='utf-8')
    if a.phase=='before':note='GRID70分支隔离修复v2冻结：只核查原128安全前缀提前返回；256最低核查和两批停滞后，仅当发现更好incumbent时同站补全。原不安全前缀回退/补全精确委托旧版，前缀缓存不重复计费。GRID69旧三月回退代价1.374%及其全部记录保留；诊断发现首次退步来自未改进安全incumbent也新增触发局部补全。先三问题/回归日期修复版3条，质量门槛通过才跑其余10已暴露日期。无训练，不改日期/阈值，不以历史计时宣称稳定加速。'
    else:
        r=read(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'))
        note=f"GRID70 {a.phase}完成保存数据核验，{r['runs']}条、质量{r['quality_dates']}/{len(r['dates'])}、完成{r['complete_dates']}/{len(r['dates'])}；裁决{r['verdict']}。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。"
        if a.phase=='complete':
            z=read(OUT/'supplementary_review.json');assert z['passed']
            note+=f" 全部13日期统一FRAME_GUARD：原版质量11/13→13/13，完整周10/13→11/13（FULL11/13）；最坏完整周成本+{100*z['max_complete_cost_delta']:.3f}%，12个有搜索等时域日期查询少{100*z['macro_query_saving']:.3f}%。跨历史FULL同状态70个/24627反馈逐项相同。0.64%旧成本增幅修复至-0.031%，七月1901步恢复2017；没有据历史时间比更新整套加速结论。"
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID70 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text(t.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+t.split('\n',1)[1],encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n工作区 D:/Codex/GRL研究。\n\n1. 完成GRID70分支隔离修复及三日期试验，审计后按冻结门槛决定是否继续其余10日期；不扫参数。\n2. 保留GRID68和原FRAME所有失败；新稿突出查询/计算减少，完整质量结果留正文表或附录。\n3. 修复结果明确后更新单独中文修订稿，旧稿不覆盖；框架贡献优先，GNN创新不强行主张。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID70 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(phase=a.phase,files=len(entries),historical_preserved=len(previous))),flush=True)
if __name__=='__main__':main()
