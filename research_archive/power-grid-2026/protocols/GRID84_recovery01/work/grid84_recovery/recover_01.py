"""Restore an omitted identical environment import before first episode starts."""
from pathlib import Path
import datetime as dt,hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid84'
REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (OUT/'recovery_01.json').exists()
    assert not (OUT/'RUN.lock').exists() and not (OUT/'pipeline.lock').exists()
    assert not (OUT/'finished.json').exists()
    assert not (OUT/'runs').exists() or not any((OUT/'runs').iterdir())
    assert not list(OUT.glob('worker_result_*.json'))
    assert not (OUT/'manifest.json').exists() or not read(OUT/'manifest.json')['summaries']
    assert read(OUT/'training_finished.json')['fits']==3 and read(OUT/'preflight.json')['passed']
    verified=0
    for name in ['training_code_freeze.json','code_freeze.json','data_freeze.json']:
        for p,h in read(OUT/name).items():assert sha(ROOT/p)==h,p;verified+=1
    for v in read(OUT/'model_freeze.json').values():assert sha(ROOT/v['path'])==v['sha256']
    log=(OUT/'worker_00.log').read_text(encoding='utf-8')
    assert "ModuleNotFoundError: No module named 'environment'" in log
    assert read(OUT/'runner_error.json')['runs']==0
    dest=OUT/'interrupted_attempts/attempt01';dest.mkdir(parents=True,exist_ok=False)
    original=[]
    names=['worker_00.log','runner_error.json','pipeline_error.json','pipeline_stderr.log','pipeline_stdout.log','run_pipeline.log','pipeline_state.json','runtime.json','RUN_STATE.md','code_freeze.json']
    for name in names:
        p=OUT/name
        if p.exists():shutil.copy2(p,dest/name);original.append(dict(path=name,sha256=sha(p)))
    # Identical source factory used by GRID82; no change to protocol/model/controller.
    src=ROOT/'work/grid82/environment.py';target=ROOT/'work/grid84/environment.py'
    assert not target.exists();assert read(OUT/'code_freeze.json')['work/grid82/environment.py']==sha(src)
    shutil.copy2(src,target);assert sha(src)==sha(target)
    frozen=read(OUT/'code_freeze.json');frozen[target.relative_to(ROOT).as_posix()]=sha(target)
    (OUT/'code_freeze.json').write_text(json.dumps(frozen,ensure_ascii=False,indent=2),encoding='utf-8')
    # Preserve original errors; remove only their top-level names so retry is unambiguous.
    for name in ['worker_00.log','runner_error.json','pipeline_error.json']:
        assert sha(OUT/name)==sha(dest/name);(OUT/name).unlink()
    receipt=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),reason='Missing environment.py module in new stage directory; failure during import, before any episode directory or observations.',
                 completed_items=0,training_repeated=False,model_hashes_unchanged=True,verified_frozen_entries=verified,
                 added_dependency=dict(source=src.relative_to(ROOT).as_posix(),target=target.relative_to(ROOT).as_posix(),sha256=sha(src)),
                 protocol_changed=False,original_files=original,original_code_freeze=sha(dest/'code_freeze.json'),recovered_code_freeze=sha(OUT/'code_freeze.json'))
    (OUT/'recovery_01.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('RECOVERED ENGINEERING IMPORT: identical GRID82 environment factory added; zero completed episodes, models and protocol unchanged. Original failed startup preserved in interrupted_attempts/attempt01. Resume original pipeline without refitting.\n',encoding='utf-8')
    archive=REPO/'research_archive/power-grid-2026/protocols/GRID84_recovery01';assert not archive.exists()
    files=[OUT/'recovery_01.json',OUT/'code_freeze.json',target,Path(__file__).resolve()]+[p for p in dest.iterdir() if p.is_file()]
    for p in files:
        q=archive/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(q)==sha(p)
    note='GRID84首次worker在导入environment前退出，0条完成且未产生回放目录；训练三模型与全部原冻结哈希核对一致。补入与GRID82逐字节相同的environment.py依赖，仅新增该文件哈希；原错误/日志/冻结清单完整留档。不改协议、不重训、不重复完成项，原pipeline续跑。'
    for name in ['DECISIONS.md','EXPERIMENT_LOG.md']:
        p=REPO/'docs'/name;p.write_text(p.read_text(encoding='utf-8')+'\n## 2026-10-16 — GRID84 recovery01\n\n'+note+'\n',encoding='utf-8')
    p=REPO/'docs/RESEARCH_STATE.md';s=p.read_text(encoding='utf-8');lines=s.splitlines(keepends=True);p.write_text(lines[0]+'\n- **工程恢复：** '+note+'\n'+''.join(lines[1:]),encoding='utf-8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
