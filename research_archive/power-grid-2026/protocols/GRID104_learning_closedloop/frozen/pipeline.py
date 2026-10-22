"""One finite serial runner, saved-data audit and publication. No automatic expansion."""
import ctypes,datetime as dt,gzip,json,os,subprocess,sys,time,traceback
from registry import ROOT,W,OUT,REPO,read,save,verify,utc

def validate(folder,result):
    s=read(folder/'summary.json');assert s==read(result)
    for name in ['components.json','vectors.npz','metadata.json']:assert (folder/name).exists()
    with gzip.open(folder/'steps.jsonl.gz','rt',encoding='utf8') as f:assert sum(1 for _ in f)==s['steps']
    return s

def main():
    assert not (OUT/'complete.json').exists()
    kernel=ctypes.windll.kernel32;kernel.CreateMutexW.restype=ctypes.c_void_p
    mutex=kernel.CreateMutexW(None,True,'GRL_GRID104_SINGLE_SERIAL_RUNNER')
    assert kernel.GetLastError()!=183,'A GRID104 runner already exists'
    kernel.SetThreadExecutionState(0x80000001)
    start=time.perf_counter()
    try:
        verify();D=read(OUT/'design.json');summaries=read(OUT/'manifest.json')['summaries']
        existing={(s['repeat'],s['week'],s['method']) for s in summaries}
        for index,item in enumerate(D['execution_order']):
            key=(item['repeat'],item['week'],item['method'])
            if key in existing:continue
            assert time.perf_counter()-start<D['caps']['wall_s'],'Finite benchmark window exhausted'
            folder=OUT/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}";result=OUT/f'worker_result_{index:02d}.json'
            if folder.exists():
                assert result.exists(),f'Partial output retained; requires recovery audit: {folder}'
                s=validate(folder,result)
            else:
                save(OUT/'pipeline_state.json',dict(phase='benchmark',pid=os.getpid(),completed=len(summaries),planned=48,index=index,item=item,utc=utc()))
                with (OUT/f'worker_{index:02d}.log').open('x',encoding='utf8') as log:
                    child=subprocess.Popen([sys.executable,'-u',str(W/'worker.py'),'--index',str(index)],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    save(OUT/'active_worker.json',dict(pid=child.pid,parent=os.getpid(),index=index,item=item))
                    try:ret=child.wait(timeout=min(D['caps']['episode_s']+120,D['caps']['wall_s']-(time.perf_counter()-start)))
                    except subprocess.TimeoutExpired:
                        subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait();raise
                assert ret==0,(index,ret)
                s=validate(folder,result)
            summaries.append(s);existing.add(key)
            assert sum(x['steps'] for x in summaries)<=D['caps']['physical_steps']
            assert sum(x['public_forecasts'] for x in summaries)<=D['caps']['public_forecasts']
            save(OUT/'manifest.json',dict(planned=48,summaries=summaries))
            print(json.dumps(dict(completed=len(summaries),planned=48,week=item['week'],method=item['method'],repeat=item['repeat'])),flush=True)
        save(OUT/'finished.json',dict(passed_engineering=True,runs=48,summaries=summaries,benchmark_wall_s=time.perf_counter()-start))
        for phase,script in [('audit','audit.py'),('finalize','finalize.py'),('archive','archive.py')]:
            save(OUT/'pipeline_state.json',dict(phase=phase,pid=os.getpid(),completed=48,planned=48,utc=utc()))
            with (OUT/f'{phase}_pipeline.log').open('x',encoding='utf8') as log:
                subprocess.run([sys.executable,'-u',str(W/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        status=subprocess.check_output(['git','-C',str(REPO),'status','--porcelain'],text=True)
        allowed=['docs/','research_archive/power-grid-2026/protocols/GRID104_learning_closedloop/','research_archive/power-grid-2026/README.md']
        assert all(any(line[3:].startswith(p) for p in allowed) for line in status.splitlines()),'Unrelated Git changes; publication requires review'
        environment=os.environ.copy();environment['TEMP']=str(ROOT/'tmp/grid100');environment['TMP']=environment['TEMP']
        with (OUT/'push_pipeline.log').open('x',encoding='utf8') as log:
            subprocess.run([str(REPO/'scripts/git_push.cmd'),'Publish GRID104 fixed learning closed-loop development comparison'],cwd=REPO,
                stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW,env=environment)
        head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
        remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','lntano-src','refs/heads/main'],text=True).split()[0];assert head==remote
        decision=read(OUT/'decision.json')
        save(OUT/'complete.json',dict(passed=True,git_sha=head,remote_verified=True,ready_to_scale=decision['ready_to_scale'],
            report='outputs/grid104_learning_closedloop/GRID104_report.md',final_test_materialized=False,new_model_fits=0,utc=utc()))
        save(OUT/'pipeline_state.json',dict(phase='complete',pid=os.getpid(),completed=48,planned=48,ready_to_scale=decision['ready_to_scale'],git_sha=head,utc=utc()))
        (OUT/'RUN_STATE.md').write_text('COMPLETE:48 development rollouts, saved-data audit and publication. No scaling or final-test access.\n',encoding='utf8')
    finally:
        kernel.SetThreadExecutionState(0x80000000)
        kernel.ReleaseMutex.argtypes=[ctypes.c_void_p];kernel.CloseHandle.argtypes=[ctypes.c_void_p]
        kernel.ReleaseMutex(mutex);kernel.CloseHandle(mutex)

if __name__=='__main__':
    try:main()
    except BaseException:
        save(OUT/'pipeline_error.json',dict(utc=utc(),traceback=traceback.format_exc(),scope='Preserve failure and partial outputs; no automatic restart or new protocol.'));raise
