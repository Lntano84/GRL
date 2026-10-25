"""Single finite benchmark, guarded by a separate human authorization receipt."""
import ctypes,datetime,json,os,subprocess,sys,time,traceback
from run import R,W,O,REPO,read,save,verify,sha,archive
def main():
    verify();auth=read(O/'physical_authorization.json');assert auth['approved_by_user'] is True and auth['design_sha256']==sha(O/'design.json')
    assert read(O/'offline_review.json')['passed'];assert not (O/'benchmark_started.json').exists(),'No automatic restart; inspect original partials and remaining window'
    kernel=ctypes.windll.kernel32;kernel.CreateMutexW.restype=ctypes.c_void_p;mutex=kernel.CreateMutexW(None,True,'GRL_GRID184_SINGLE_BENCHMARK');assert kernel.GetLastError()!=183
    kernel.SetThreadExecutionState(0x80000001);start=time.perf_counter();D=read(O/'design.json');summaries=[]
    save('benchmark_started.json',dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_wall_s=D['caps']['wall_s']))
    try:
        for index,item in enumerate(D['execution_order']):
            verify();remaining=D['caps']['wall_s']-(time.perf_counter()-start);assert remaining>0,'Original benchmark window exhausted'
            save('pipeline_state.json',dict(phase='benchmark',pid=os.getpid(),completed=len(summaries),planned=64,index=index,item=item))
            folder=O/'runs'/f"r{item['repeat']}__{item['week']}__{item['method']}";assert not folder.exists(),'Do not overwrite partial or complete results'
            with (O/f'worker_{index:02d}.log').open('x',encoding='utf8') as log:
                child=subprocess.Popen([sys.executable,'-X','utf8','-u',str(W/'worker.py'),'--index',str(index)],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                save('active_worker.json',dict(pid=child.pid,parent=os.getpid(),index=index,item=item))
                try:ret=child.wait(timeout=min(remaining,D['caps']['episode_s']+120))
                except subprocess.TimeoutExpired:
                    subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait();raise
            assert ret==0,(index,ret)
            s=read(folder/'summary.json');assert s==read(O/f'worker_result_{index:02d}.json')
            for name in ['components.json','vectors.npz','steps.jsonl.gz','metadata.json']:assert (folder/name).exists()
            summaries.append(s);assert sum(v['steps'] for v in summaries)<=D['caps']['physical_steps'];assert sum(v['public_forecasts'] for v in summaries)<=D['caps']['public_forecasts']
            save('manifest.json',dict(planned=64,summaries=summaries));print('COMPLETE',index+1,64,flush=True)
        save('finished.json',dict(passed_engineering=True,runs=64,summaries=summaries,benchmark_wall_s=time.perf_counter()-start));save('pipeline_state.json',dict(phase='audit',completed=64,planned=64,pid=os.getpid()))
        with (O/'audit_pipeline.log').open('x',encoding='utf8') as log:subprocess.run([sys.executable,'-X','utf8','-u',str(W/'audit.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        assert read(O/'review.json')['passed_saved_data'];archive('benchmark_complete')
        env=os.environ.copy();env['TEMP']=str(R/'tmp/grid100');env['TMP']=env['TEMP']
        with (O/'push_pipeline.log').open('x',encoding='utf8') as log:subprocess.run([str(REPO/'scripts/git_push.cmd'),'GRID184 publish all frozen consensus closed-loop results and saved-data audit'],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,check=True,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0];assert local==remote
        save('complete.json',dict(passed=True,git_sha=local,remote_verified=True,new_fits=0,report='outputs/grid184_search_adapter/GRID184_report.md'))
        save('pipeline_state.json',dict(phase='complete',completed=64,planned=64,git_sha=local))
    finally:
        kernel.SetThreadExecutionState(0x80000000);kernel.ReleaseMutex.argtypes=[ctypes.c_void_p];kernel.CloseHandle.argtypes=[ctypes.c_void_p];kernel.ReleaseMutex(mutex);kernel.CloseHandle(mutex)
if __name__=='__main__':
    try:main()
    except BaseException:
        save('pipeline_error.json',dict(traceback=traceback.format_exc(),preserve_partials=True,no_automatic_restart=True));raise
