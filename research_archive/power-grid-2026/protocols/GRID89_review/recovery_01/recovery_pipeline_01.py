"""Publication quoting repair only; frozen benchmark and original pipeline unchanged."""
from pathlib import Path
import datetime as dt,hashlib,importlib.util,json,os,subprocess,sys,traceback
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid89_review';REPO=ROOT/'work/GRL-results-integration'
spec=importlib.util.spec_from_file_location('frozen_review_pipeline',Path(__file__).with_name('pipeline.py'));P=importlib.util.module_from_spec(spec);spec.loader.exec_module(P)
def safe_push(message):
 env=dict(os.environ);env['PATH']='C:/Users/windows/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd;'+env['PATH'];env['TEMP']=env['TMP']=str(ROOT/'tmp')
 message_file=ROOT/'tmp/grid89_commit_message.txt';message_file.write_text(message+'\n',encoding='utf-8')
 result=subprocess.run([os.environ['COMSPEC'],'/d','/c','scripts\\git_push.cmd',str(message_file)],cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
 with (OUT/'git_publication_recovery01.log').open('a',encoding='utf-8') as f:f.write(result.stdout+'\n')
 assert result.returncode==0 and 'PUSHED OK' in result.stdout,result.stdout[-3000:]
 print(result.stdout,flush=True)
def main():
 receipt=json.loads((OUT/'recovery_01.json').read_text(encoding='utf-8'))
 for p,h in receipt['frozen_items'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==receipt['recovery_source_sha256']
 P.push=safe_push
 if '--push-only' in sys.argv:safe_push('Seal all FRAME reviewer inputs and retain pre-benchmark publication recovery')
 else:P.main()
if __name__=='__main__':
 try:main()
 except BaseException:
  P.save('pipeline_error.json',dict(traceback=traceback.format_exc(),utc=dt.datetime.now(dt.timezone.utc).isoformat(),recovery=1));raise
