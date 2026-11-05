"""New approval only; creates a fresh dual-clock, single-boot launch once."""
import os,subprocess,sys,time,uuid
from guard import R,W,O,require_authorization,read,save,sha,boot_id
def main():
 require_authorization(clock=False)
 assert not any((O/n).exists() for n in ['launch_context.json','started.json','claims','runs'])
 observed=subprocess.check_output(['powershell','-NoProfile','-Command',"@(Get-CimInstance Win32_Process | Where-Object {$_.Name -match '^python' -and $_.CommandLine -match '(grid300_new_training_dates[/\\\\]continuation01|grid301_remaining_collection)[/\\\\](manager|worker|publish_stage)\\.py'}).Count"]).decode().strip()
 assert observed=='0','another collection process exists'
 # Boot query and process check precede the new time anchor, no models loaded.
 boot=boot_id(); start=time.perf_counter(); utc=time.time()
 ctx=dict(start_monotonic=start,deadline_monotonic=start+86400,start_utc_epoch=utc,deadline_utc_epoch=utc+86400,boot_id=boot,authorization_sha256=sha(O/'physical_authorization.json'),launch_nonce=uuid.uuid4().hex)
 with (O/'launch_context.json').open('x',encoding='utf8') as f: __import__('json').dump(ctx,f)
 env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','TEMP':str(R/'tmp/grid100'),'TMP':str(R/'tmp/grid100')}
 with (O/'manager_stdout.log').open('x',encoding='utf8') as out,(O/'manager_stderr.log').open('x',encoding='utf8') as err:
  child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(W/'manager.py')],cwd=R,env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
 save(O/'launch_process.json',{'launcher_pid':child.pid,'boot_id':boot,'start_utc_epoch':utc,'deadline_utc_epoch':utc+86400,'no_automatic_recovery':True})
 print('STARTED_ONCE',child.pid)
if __name__=='__main__': main()
