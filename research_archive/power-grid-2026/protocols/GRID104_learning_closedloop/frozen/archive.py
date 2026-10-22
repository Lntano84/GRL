"""Publish source/saved records and hashes; retain large physical vectors on D:."""
import shutil,zipfile
from registry import ROOT,W,OUT,REPO,ARCHIVE,read,save,sha

def main():
    assert read(OUT/'review.json')['passed_saved_data']
    dest=ARCHIVE/'complete';dest.mkdir(parents=True,exist_ok=False)
    files=list(W.glob('*.py'));excluded=[]
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        if p.suffix in ['.json','.md','.log','.txt','.csv'] or p.name.endswith('.jsonl.gz'):files.append(p)
        else:excluded.append(dict(path=p.relative_to(ROOT).as_posix(),bytes=p.stat().st_size,sha256=sha(p),reason='Large saved physical/input vectors retained locally on D:, hashes published.'))
    batches=[];batch=[];size=0
    for p in sorted(set(files)):
        if batch and size+p.stat().st_size>24*1024*1024:batches.append(batch);batch=[];size=0
        batch.append(p);size+=p.stat().st_size
    if batch:batches.append(batch)
    entries=[];archives=[]
    for i,batch in enumerate(batches):
        path=dest/f'records-{i+1:03d}.zip'
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for p in batch:
                member=p.relative_to(ROOT).as_posix();z.write(p,member)
                entries.append(dict(path=member,bytes=p.stat().st_size,sha256=sha(p),archive=path.name))
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            for e in [e for e in entries if e['archive']==path.name]:assert __import__('hashlib').sha256(z.read(e['path'])).hexdigest()==e['sha256']
        archives.append(dict(path=path.name,sha256=sha(path),bytes=path.stat().st_size))
    for name in ['GRID104_report.md','review.json','decision.json','design.json','preflight.json','run_metrics.csv']:shutil.copy2(OUT/name,dest/name)
    save(dest/'files.json',entries);save(dest/'archive_receipt.json',dict(archives=archives,files=len(entries),crc_and_hashes_verified=True,excluded_large_local_vectors=excluded))
    decision=read(OUT/'decision.json')
    text='\n\n## 2026-10-22 — GRID104 fixed learning closed-loop development completed\n\n48 serial runs on four already exposed validation dates (replicate100), FRAME/BATCH1/RF/all3 gamma0 DQN seeds, two reverse timings; no fitting/new scenarios or final-test access. Saved vectors/feedback/policy/context/cost/repeat/timing audit passed. Learning projection and inference counted in actual controller time. Ready-to-scale exploratory gate='+str(decision['ready_to_scale'])+'. All adverse outcomes retained, no best-seed selection or early-termination speed claim. Report: research_archive/power-grid-2026/protocols/GRID104_learning_closedloop/complete/GRID104_report.md. Large physical vectors retained on D: with published hash manifests. No automatic expansion.\n'
    for name in ['RESEARCH_STATE.md','EXPERIMENT_LOG.md','DECISIONS.md']:
        with (REPO/'docs'/name).open('a',encoding='utf8') as f:f.write(text)
    (REPO/'docs/NEXT_STEPS.md').write_text('# Next steps — GRID104完成\n\n1. 阅读GRID104完整开发闭环结果与所有种子，扩大资格='+str(decision['ready_to_scale'])+'。\n2. 扩规模前通知用户并交付具体方案；不自动增加日期、训练、预算或第二电网。\n3. 最终8测试场景保持封存，不恢复旧自动化。\n',encoding='utf8')
    with (REPO/'research_archive/power-grid-2026/README.md').open('a',encoding='utf8') as f:f.write('\n- [GRID104学习补查闭环](protocols/GRID104_learning_closedloop/complete/GRID104_report.md)：四个已暴露验证日期，六臂两次计时，原模型固定，无扩大或最终测试。\n')
    print('Complete source/record archives verified',flush=True)
if __name__=='__main__':main()
