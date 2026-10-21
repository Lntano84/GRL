"""Wait for frozen training generation, collect one teacher week, audit, stop."""
from pathlib import Path
import datetime as dt,json,os,subprocess,sys,time,traceback,ctypes
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
def save(n,d):
    p=OUT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def main():
    assert not (OUT/'pilot_collection_finished.json').exists()
    mutex=ctypes.windll.kernel32.CreateMutexW(None,True,'GRL_GRID100_PILOT_COLLECTOR');assert ctypes.windll.kernel32.GetLastError()!=183,'Existing collector'
    try:
        start=time.monotonic();save('collection_state.json',dict(stage='waiting_generation_pilot',pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat()))
        while not (OUT/'generation_pilot_finished.json').exists():
            assert not (OUT/'generation_error.json').exists() and not (OUT/'pipeline_error.json').exists(),'Generation error preserved; no collection'
            assert time.monotonic()-start<2040,'Generation wait budget exhausted';time.sleep(5)
        assert read(OUT/'generation_pilot_finished.json')['passed']
        runtime=OUT/'teacher_runtime';D=read(OUT/'design.json');folder=runtime/'runs'/f"r0__{D['pilot']['scenario']}__FULL_TEACHER";assert not folder.exists(),'Partial teacher output requires inspection, never overwrite'
        with (OUT/'teacher_pilot_worker.log').open('x',encoding='utf8') as log:
            child=subprocess.Popen([sys.executable,'-u','-X','utf8',str(W/'teacher_worker.py'),'--index','0'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            save('collection_state.json',dict(stage='teacher_pilot',pid=os.getpid(),child_pid=child.pid,scenario=D['pilot']['scenario'],utc=dt.datetime.now(dt.timezone.utc).isoformat()))
            try:ret=child.wait(timeout=1320)
            except subprocess.TimeoutExpired:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait();raise
            assert ret==0,ret
        with (OUT/'pilot_extraction.log').open('x',encoding='utf8') as log:subprocess.run([sys.executable,'-u','-X','utf8',str(W/'extract_pilot.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        save('collection_state.json',dict(stage='pilot_data_qualified',pid=os.getpid(),new_teacher_episodes=1,model_fits=0,final_test_materialized=False,utc=dt.datetime.now(dt.timezone.utc).isoformat()))
    finally:ctypes.windll.kernel32.ReleaseMutex(mutex);ctypes.windll.kernel32.CloseHandle(mutex)
if __name__=='__main__':
    try:main()
    except BaseException:save('collection_error.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
