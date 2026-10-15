"""Append-only repair records. Preserve every historical artifact manifest."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid76'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','pilot','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID76_{a.phase}';assert not dest.exists()
    if a.phase=='before':assert read(OUT/'preflight.json')['passed'] and not (OUT/'manifest.json').exists()
    else:assert read(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'))['passed_saved_data']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid76').glob('*.py'))+[Path(__file__).resolve()]
    files+=list((ROOT/'work/grid76_review').glob('*.py'))
    files+=[p for p in OUT.iterdir() if p.is_file() and p.suffix in ('.json','.md','.csv')]
    if a.phase!='before':
        for folder in (OUT/'runs').iterdir():files+=[folder/f for f in ['metadata.json','summary.json','components.json'] if (folder/f).exists()]
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
        assert sha(p)==sha(q);entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),scope='Finite exposed-date repair development. Original models/results preserved; no new training or independent AC re-solve.'),ensure_ascii=False,indent=2),encoding='utf-8')
    if a.phase=='before':note='GRID76廉价对照闭环冻结：仅普通电网上下文模型通过11.735%校准查询门槛，结构主候选未通过，不冒充结构创新。初始安全128可按固定普通模型提前返回，否则精确复用GRID70；特征构造、首个JSON加载和推理均计入控制。77合法轨迹重放、14/14校准教师动作复现及2实际观测图上下文重建通过。先七月/十二月/三月3条质量压力试验，质量通过再跑其余10日期；压力日期不要求省查询。未优于现有强对照且质量合格前不重训会议对照。'
    else:
        r=read(OUT/('pilot_review.json' if a.phase=='pilot' else 'review.json'))
        note=f"GRID76 {a.phase}完成保存数据核验，{r['runs']}条、质量{r['quality_dates']}/{len(r['dates'])}、完成{r['complete_dates']}/{len(r['dates'])}；裁决{r['verdict']}。所有不利结果保留，修复开发不作为独立确认；单次计时仅描述。"
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID76 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';t=p.read_text(encoding='utf-8');p.write_text(t.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+t.split('\n',1)[1],encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n工作区 D:/Codex/GRL研究。\n\n1. GRID76全部13日期及上下文核验完成，查询宏平均减少66.468%，质量13/13。普通上下文是廉价对照，结构特征增量尚未成立。\n2. GRID77只补齐会议适配基线的相同24维上下文和完整核查模式，训练/校准排除全部13评价日期；保留6个固定拟合。不能把旧会议对照信息不足造成的差异作为创新。\n3. 固定模型后进行有限同日期闭环比较，不扩日期或调参救结果；没有新配对计时前不宣称新端到端加速。旧稿与所有失败结果不覆盖。\n',encoding='utf-8')
    p=ARCH/'README.md';p.write_text(p.read_text(encoding='utf-8')+f'\n## GRID76 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(phase=a.phase,files=len(entries),historical_preserved=len(previous))),flush=True)
if __name__=='__main__':main()
