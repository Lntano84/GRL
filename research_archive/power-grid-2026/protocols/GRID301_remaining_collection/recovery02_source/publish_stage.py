"""Only while manager waits without a worker: publish small fixed JSON manifests."""
import argparse,hashlib,json,os,subprocess
from guard import R,O,I,A,read,sha,save,git,require_authorization,check
ALLOWED={'baseline':{'baseline_manifest.json','baseline_raw_manifest.json','selected_markers.json','runtime_design.json'}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=ALLOWED,required=True);name=ap.parse_args().stage
 require_authorization();started=read(O/'launch_context.json');ctx=started.copy();check(ctx)
 active=read(O/'active_worker.json');state=read(O/'pipeline_state.json');assert not active.get('active',True) and state['phase']==name+'_publication_pending'
 ready=read(O/f'{name}_ready.json');assert set(ready['files'])==ALLOWED[name]
 S=A/(name+'_public');S.mkdir(exist_ok=False)
 for n,h in ready['files'].items():
  p=O/'stage_payload'/name/n;assert sha(p)==h and sha(O/n)==h and p.stat().st_size<30_000_000
  (S/n).write_bytes(p.read_bytes())
 allowed={(S/n).relative_to(I).as_posix() for n in ALLOWED[name]}
 def paths(*args):return set(git('-c','core.quotepath=false',*args).decode('utf8').splitlines())
 actual=paths('diff','--name-only')|paths('diff','--cached','--name-only')|paths('ls-files','--others','--exclude-standard');assert actual==allowed,(actual-allowed,allowed-actual)
 env=os.environ.copy();env['TEMP']=env['TMP']=str(R/'tmp/grid100')
 done=subprocess.run(['cmd.exe','/d','/c',str(I/'scripts/git_push.cmd'),'GRID300 publish '+name+' source hashes before next physical stage'],cwd=I,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,shell=False,timeout=max(1,started['deadline_monotonic']-__import__('time').perf_counter()));print(done.stdout.decode('utf8',errors='replace'));assert done.returncode==0
 head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 for n,h in ready['files'].items():assert hashlib.sha256(git('show',head+':'+(S/n).relative_to(I).as_posix())).hexdigest()==h
 save(O/f'{name}_publication.json',{'passed':True,'commit':head,'ready_sha256':sha(O/f'{name}_ready.json'),'exact_published_files':ready['files'],'no_active_worker_at_publication':True});check(ctx)
if __name__=='__main__':main()
