"""Wait for all physical workers to exit, then finish authorized audit/learning/publication."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid96_value_learning';REPO=ROOT/'work/GRL-results-integration'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def physical_alive():
    cmd="@(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'grid96_value_learning[\\/](run|worker)\\.py(?: |\"|$)' }).Count"
    p=subprocess.run(['powershell.exe','-NoProfile','-Command',cmd],capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return int(p.stdout.strip())>0
def main():
    assert not (OUT/'complete.json').exists();save('postprocess_state.json',dict(phase='waiting_for_physical_collection',pid=os.getpid()))
    while not (OUT/'finished.json').exists():
        if (OUT/'pipeline_error.json').exists():raise RuntimeError('Physical collection stopped; inspect preserved pipeline_error.json. No automatic retry.')
        time.sleep(20)
    assert read(OUT/'finished.json')['runs']==74
    while physical_alive():time.sleep(5)
    for name in ['audit.py','learn.py','audit_learning.py','finalize.py']:
        assert not physical_alive();save('postprocess_state.json',dict(phase=name,pid=os.getpid()))
        with (OUT/(name.removesuffix('.py')+'_stdout_stderr.log')).open('x',encoding='utf-8') as f:
            subprocess.run([sys.executable,'-X','utf8','-u',str(W/name)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    save('postprocess_state.json',dict(phase='publishing',pid=os.getpid()))
    env=dict(os.environ,TEMP=str(ROOT/'tmp/grid96'));Path(env['TEMP']).mkdir(parents=True,exist_ok=True)
    with (OUT/'git_push.log').open('x',encoding='utf-8') as f:
        subprocess.run([str(REPO/'scripts/git_push.cmd'),'Complete GRID96 single-omission labels and grouped simple learning'],cwd=REPO,stdout=f,stderr=subprocess.STDOUT,env=env,shell=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    local=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','lntano-src','refs/heads/main'],cwd=REPO,text=True).split()[0]
    assert local==remote and not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip()
    save('publication.json',dict(passed=True,local_sha=local,remote_sha=remote,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_sha256=hashlib.sha256((OUT/'GRID96_report.md').read_bytes()).hexdigest()))
    save('postprocess_state.json',dict(phase='complete_published',local_sha=local,remote_sha=remote))
if __name__=='__main__':
    try:main()
    except BaseException:save('postprocess_error.json',dict(traceback=traceback.format_exc()));raise
