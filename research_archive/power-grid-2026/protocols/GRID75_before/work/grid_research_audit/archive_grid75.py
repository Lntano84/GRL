"""Append-only conditional-verification qualification receipts."""
from pathlib import Path
import argparse,hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];REPO=ROOT/'work/GRL-results-integration';ARCH=REPO/'research_archive/power-grid-2026';OUT=ROOT/'outputs/grid75'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['before','complete'],required=True);a=ap.parse_args()
    dest=ARCH/'protocols'/f'GRID75_{a.phase}';assert not dest.exists()
    assert read(OUT/'preflight.json')['passed']
    if a.phase=='before':assert not (OUT/'training_finished.json').exists()
    else:assert read(OUT/'training_finished.json')['passed_engineering']
    previous={}
    for m in list(ARCH.glob('*manifest.json'))+list((ARCH/'protocols').glob('*/manifest.json')):
        c=read(m)
        for e in c if isinstance(c,list) else c.get('files',[]):
            p=m.parent/e['path']
            if not p.exists():p=ARCH/e['path']
            assert p.exists() and sha(p)==e['sha256'],str(p);previous[str(p)]=e['sha256']
    files=list((ROOT/'work/grid75').glob('*.py'))+[Path(__file__).resolve()]
    files+=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.md','.npz']]
    entries=[]
    for p in sorted(set(files)):
        q=dest/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
        entries.append(dict(path=p.relative_to(ROOT).as_posix(),sha256=sha(p),bytes=p.stat().st_size))
    (dest/'manifest.json').write_text(json.dumps(dict(files=entries,historical_preserved=len(previous),scope='Two fixed light-forest gate fits, teacher-action-change target only; no closed-loop/quality/novelty claim.'),ensure_ascii=False,indent=2),encoding='utf-8')
    if a.phase=='before':note='GRID75最后固定输入资格：在原搜索反馈上加入已审计快照的真实电网全局/局部物理上下文，普通24/结构38维；同RF参数、63/14安全状态、相同教师动作改变标签与10%门槛。77后缀权限检查通过，排除13评价日期家族；保留GRID72/73/74，不扫模型参数，不启动物理回放。'
    else:
        r=read(OUT/'training_finished.json');note='GRID75两固定轻量模型拟合完成，'+f"墙钟{r['wall_s']:.3f}s，结构资格{r['structural_qualified']}；"+'；'.join(f"{x['name']}校准省查询{x['calibration_query_saving']*100:.3f}%、略过{x['calibration_skipped']}/14、漏改变{x['calibration_false_stop']}" for x in r['models'])+'。14校准状态不是闭环质量证明；尚无新物理回放。'
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+f'\n## 2026-10-15 — GRID75 {a.phase}\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');p.write_text(s.split('\n',1)[0]+'\n\n- **最新：** '+note+'\n'+s.split('\n',1)[1],encoding='utf-8')
    for p,h in previous.items():assert sha(Path(p))==h
    print(json.dumps(dict(files=len(entries),preserved=len(previous),phase=a.phase)),flush=True)
if __name__=='__main__':main()
