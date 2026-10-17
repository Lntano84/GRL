"""Persistent generation-wait -> freeze -> serial suite -> audit -> V2 -> publication."""
from pathlib import Path
import datetime as dt,json,os,subprocess,sys,time,traceback
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid89_review';REPO=ROOT/'work/GRL-results-integration'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,x):
 p=OUT/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def state(phase,**kw):save('pipeline_state.json',dict(phase=phase,pid=os.getpid(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),**kw))
def command(name,*args):
 with (OUT/(name+'_pipeline.log')).open('a',encoding='utf-8') as f:subprocess.run([sys.executable,'-u','-X','utf8',str(W/(name+'.py')),*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
def push(message):
 env=dict(os.environ);env['PATH']='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd;'+env['PATH'];env['TEMP']=env['TMP']=str(ROOT/'tmp')
 r=subprocess.run([os.environ['COMSPEC'],'/d','/c',f'scripts\\git_push.cmd "{message}"'],cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace');(OUT/'git_publication.log').open('a',encoding='utf-8').write(r.stdout+'\n')
 assert r.returncode==0 and 'PUSHED OK' in r.stdout,r.stdout[-3000:]
def main():
 import ctypes
 mutex=ctypes.windll.kernel32.CreateMutexW(None,True,'GRL_GRID89_PIPELINE');assert ctypes.windll.kernel32.GetLastError()!=183,'Existing pipeline owns mutex'
 try:
  if (OUT/'complete.json').exists():print('Already complete');return
  state('waiting_generation_and_training')
  while not ((OUT/'generation.json').exists() and (OUT/'meta_training/training_finished.json').exists()):
   if (OUT/'generation_error.json').exists():raise RuntimeError('Independent generation failed; preserve series/error, do not replace dates')
   if (OUT/'generation_progress.json').exists() and not (OUT/'generation.json').exists():
    pid=read(OUT/'generation_progress.json')['pid'];handle=ctypes.windll.kernel32.OpenProcess(0x1000,False,pid)
    alive=False
    if handle:
     code=ctypes.c_ulong();ctypes.windll.kernel32.GetExitCodeProcess(handle,ctypes.byref(code));ctypes.windll.kernel32.CloseHandle(handle);alive=code.value==259
    assert alive,'Generator stopped without final receipt; preserve partial date and audit before resuming'
   time.sleep(10)
  assert read(OUT/'generation.json')['passed'];assert read(OUT/'meta_training/training_finished.json')['fits']==3
  command('preflight');state('sealing_inputs');command('seal');command('archive','--phase','frozen');push('Seal FRAME reviewer checks, positive-cost models and independent data receipts')
  if not (OUT/'benchmark_finished.json').exists():state('benchmark_start');command('run_suite')
  state('saved_data_audit')
  if not (OUT/'review.json').exists():command('audit')
  if not (OUT/'decision.json').exists():command('finalize')
  state('writing_and_rendering_v2');command('manuscript')
  state('awaiting_visual_review',document='outputs/grid89_review/manuscript_v2/FRAME_ICAPS2027_中文合稿_V2.docx',pages=read(OUT/'manuscript_v2/render_complete.json')['pages'])
  while not (OUT/'visual_review.json').exists():time.sleep(10)
  command('publish');state('archiving_complete');command('archive','--phase','complete');push('Publish audited FRAME reviewer validation and Chinese manuscript V2')
  save('complete.json',dict(passed=True,report='outputs/grid89_review/GRID89_report.md',new_replays=246,new_fits=3,utc=dt.datetime.now(dt.timezone.utc).isoformat()));state('complete',report='outputs/grid89_review/GRID89_report.md')
 finally:ctypes.windll.kernel32.ReleaseMutex(mutex);ctypes.windll.kernel32.CloseHandle(mutex)
if __name__=='__main__':
 try:main()
 except BaseException:save('pipeline_error.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat()));raise
