"""Persistent hidden pilot with actual process-tree timeout; no mass replay."""
from pathlib import Path
import json,os,subprocess,sys,datetime as dt,traceback
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid100_data_foundation'
def save(n,d):
    p=OUT/n;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n',encoding='utf8');t.replace(p)
def main():
    assert not (OUT/'generation_pilot_finished.json').exists()
    with (OUT/'generation_pilot_worker.log').open('x',encoding='utf8') as log:
        child=subprocess.Popen([sys.executable,'-u','-X','utf8',str(W/'generate.py'),'--mode','pilot'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        save('pipeline_state.json',dict(stage='train_generation_pilot',pid=os.getpid(),child_pid=child.pid,utc=dt.datetime.now(dt.timezone.utc).isoformat(),final_test_access=False))
        try:ret=child.wait(timeout=1920)
        except subprocess.TimeoutExpired:
            subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait();raise
        assert ret==0,ret
    assert json.loads((OUT/'generation_pilot_finished.json').read_text(encoding='utf8'))['passed']
    save('pipeline_state.json',dict(stage='generation_pilot_qualified',pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),new_teacher_replays=0,new_model_fits=0,final_test_access=False,next_stage='Train/validation generation entrypoint ready; teacher runtime collection integration still to qualify before replay.'))
if __name__=='__main__':
    try:main()
    except BaseException:save('pipeline_error.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
