"""Exactly12 load-only assembled workers, one per owner pool; zero NN forward/env."""
import os,subprocess,sys,time
from guard import R,W,O,read,save,check
def run_load_only(ctx):
 assert not (O/'load_only_review_recovery02.json').exists()
 pools=[]
 for index in range(0,144,12):
  check(ctx);save(O/'load_only_context.json',{**ctx,'index':index,'kind':'load_only'})
  item=read(O/'design.json')['baseline_runtime']['execution_order'][index]
  with (O/f'load_only_recovery02_{index:03d}.log').open('x',encoding='utf8') as log:
   env={**os.environ,'GRL_GRID300_TOKEN':ctx['token'],'PYTHONDONTWRITEBYTECODE':'1','TEMP':str(R/'tmp/grid100'),'TMP':str(R/'tmp/grid100')}
   child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(W/'load_only.py'),'--index',str(index)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
   save(O/'active_worker.json',{'active':True,'kind':'load_only','index':index,'pid':child.pid,'manager_pid':ctx['manager_pid']})
   try:
    while child.poll() is None:check(ctx);time.sleep(.5)
   except BaseException:
    if child.poll() is None:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT);child.wait()
    raise
   finally:save(O/'active_worker.json',{'active':False,'kind':'load_only','index':index,'manager_pid':ctx['manager_pid']})
   if child.returncode:raise RuntimeError('load-only preflight failed: '+item['pool'])
  r=read(O/f'load_only_recovery02_{index:03d}.json');assert r['pool']==item['pool'] and r['new_forward']==r['new_environment']==r['new_physics']==0;pools.append(r)
 save(O/'load_only_review_recovery02.json',{'passed':True,'pools':pools,'original_clock_unchanged':True,'new_forward':0,'new_physics':0})
