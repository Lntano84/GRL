"""Manager-owned synchronous publication: never wait for a future heartbeat."""
import os,subprocess,sys,time
from guard import R,W,O,read,save,check,verify_stage
def release_baseline(ctx,deadline):
 check({**ctx,'worker_deadline_monotonic':deadline});assert read(O/'active_worker.json').get('active') is False
 save(O/'pipeline_state.json',{'phase':'baseline_publication_pending','no_active_worker':True,'synchronous_publisher':True,'original_deadline':ctx['deadline_monotonic']})
 with (O/'baseline_publication_continuation01.log').open('x',encoding='utf8') as log:
  child=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(W/'publish_stage.py'),'--stage','baseline'],cwd=R,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  try:
   while child.poll() is None:check({**ctx,'worker_deadline_monotonic':deadline});time.sleep(.5)
  except BaseException:
   if child.poll() is None:
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
    else:child.terminate()
    child.wait()
   raise
  if child.returncode:raise RuntimeError('synchronous publication failed; preserve all sources, no retry')
 check({**ctx,'worker_deadline_monotonic':deadline});verify_stage('baseline')
