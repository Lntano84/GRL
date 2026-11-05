"""One approved zero-model repair; never creates another24h anchor."""
import os,subprocess,sys
from guard import R,W,O,require_authorization,check,read,save
def main():
 require_authorization();ctx=read(O/'launch_context.json');check(ctx)
 assert not any((O/n).exists() for n in ['started_recovery03.json','claims_recovery03','runs','recovery03_launch_process.json'])
 err=read(O/'collection_error.json');assert err['new_accepted']==err['physical_steps_new']==err['forecasts_new']==0
 assert not (O/'worker_setup_036.json').exists()
 with (O/'manager_recovery03_stdout.log').open('x',encoding='utf8') as out,(O/'manager_recovery03_stderr.log').open('x',encoding='utf8') as error:
  env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','TEMP':str(R/'tmp/grid100'),'TMP':str(R/'tmp/grid100')}
  child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(W/'manager.py')],cwd=R,env=env,stdout=out,stderr=error,creationflags=subprocess.CREATE_NO_WINDOW)
 save(O/'recovery03_launch_process.json',dict(launcher_pid=child.pid,original_launch_context=ctx,original_clock_unchanged=True,no_retry=True))
 print('RECOVERY_STARTED_ONCE',child.pid)
if __name__=='__main__':main()
