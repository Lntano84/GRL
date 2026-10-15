"""Archive fixed same-context source-style baseline fits; exclude large paid traces."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid77'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID77_{a.phase}';assert not dest.exists()
    assert read(OUT/'learning_preflight.json')['passed']
    if a.phase=='before':assert not (OUT/'training_finished.json').exists()
    else:assert read(OUT/'fit_audit.json')['passed']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid77').glob('*.py'))+[Path(__file__).resolve()]
    files += [p for p in OUT.rglob('*') if p.is_file() and p.name!='records.json' and p.suffix in ['.json','.md','.pt']]
    entries=[]
    for p in sorted(set(files)):
        assert p.stat().st_size<8_000_000,str(p)
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    external=OUT/'records.json'
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),external_paid_trace=dict(path=str(external),sha256=sha(external),bytes=external.stat().st_size),scope='Six fixed baseline fits, no new teacher queries, no original A*/RRT* reproduction claim.'),ensure_ascii=False,indent=2),encoding='utf-8')
    note='GRID77相同信息与核查权限会议适配：25维(共享24维+模式位)、返回/继续64/完整FRAME_GUARD，保留ICAPS2022式DoubleDQN内核；2种子×3既定费用共6拟合，129训练22校准，排除13评价家族。不把停止/参数控制作为新颖性，不把普通上下文结果冒充结构优势。'
    if a.phase=='complete':
        r=read(OUT/'training_finished.json');note+=f" 拟合完成，校准质量匹配{r['calibration_quality_matched']}，共享costindex={r['selected']['costindex']}；尚无新闭环成绩。"
    for n in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/n;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID77 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps\n\n1. 完成GRID77六个固定会议适配拟合和复算，全部记录保留；不按测试结果选种子。\n2. 模型冻结后，在旧七日期有限比较两种子与GRID76；查询与质量可比，新配对计时前不宣称实际提速胜过会议对照。\n3. 全部日期为开发，不做独立确认主张；结构增量未建立，旧稿与失败版本保留。\n',encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(phase=a.phase,files=len(entries),historical_preserved=len(previous))))
if __name__=='__main__':main()
