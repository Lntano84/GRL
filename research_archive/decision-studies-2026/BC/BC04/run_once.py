"""Exactly one STATIC-BASE replay attempt, capped at 30 minutes."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import *

def main():
    assert json.loads((WORK/'preflight.json').read_text())['passed']
    assert not (WORK/'run_status.json').exists(),'Do not launch additional replay attempts'
    env=dict(os.environ)
    env['PATH']='C:/Program Files/Git/usr/bin;'+env['PATH']
    env['PYTHONHASHSEED']='0'
    env['PYTHONIOENCODING']='utf-8'
    started=time.monotonic()
    status={'status':'running','budget_seconds':1800,'started_unix':time.time(),'attempts':1}
    (WORK/'run_status.json').write_text(json.dumps(status,indent=2))
    with (WORK/'console.log').open('w',encoding='utf-8') as f:
        process=subprocess.Popen([str(PYTHON),'-B',str(WORK/'replay.py')],cwd=str(REPO),env=env,stdout=f,stderr=subprocess.STDOUT)
        try:
            code=process.wait(timeout=1800)
            status.update(status='completed' if code==0 else 'failed',exit_code=code)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            status['status']='budget_interrupted'
    status['wall_seconds']=time.monotonic()-started
    (WORK/'run_status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status))

if __name__=='__main__':main()
