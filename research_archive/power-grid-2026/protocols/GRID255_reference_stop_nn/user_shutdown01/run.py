"""Publish-gated entry to the fixed GRID254-qualified reference training component."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,shutil,time,traceback
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid255_reference_stop_nn';P=R/'outputs/grid254_reference_preflight';CW=R/'work/grid254_reference_preflight';REPO=R/'work/GRL-results-integration';A=REPO/'research_archive/power-grid-2026/protocols/GRID255_reference_stop_nn'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True

def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def verify():
 for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(P/'source_freeze_recovery01.json').items():assert sha(R/rel)==h,rel
 for rel,h in read(P/'qualified_freeze.json').items():assert sha(P/rel)==h,rel

def published(files):
 head=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','-C',str(REPO),'ls-remote','lntano-src','refs/heads/main'],text=True).split()[0];assert head==remote
 for p in files:
  rel=p.relative_to(REPO).as_posix();assert subprocess.check_output(['git','-C',str(REPO),'show',head+':'+rel])==p.read_bytes(),rel
 return head

def prepare():
 assert not O.exists();assert read(P/'complete.json')['passed'] and read(P/'independent_audit.json')['passed'];O.mkdir();A.mkdir(parents=True,exist_ok=False)
 save(O/'design.json',dict(id='GRID255',scope='Fixed18offline referenceNN data foundation ONLY; no new reference held forward, policy or teacher-label join in this stage.',parent_plan='GRID253 fixed_pipeline_proposal',reference_nodes=18,workers=6,torch_threads=1,updates=7250,batch=32,parameters=12980,optimizer='originalAdam lr.0003/.9/.999/1e-8/wd0',loss='originalSmoothL1beta1',gradclip=10,global_window_s=600,inclusive_startup=True,no_extension=True,no_retry_complete=True,seeds=[0,1,2],readout_later_max=22,complete_pipeline_fits_max=40,fixed_control='GRID250; no GAP252',training_code='Unmodified GRID254 qualified worker.py/manager.py/runtime254.py',boundaries='No final8/physical/grid/paper/FRAME/install. Reference source excludes both record/outer date groups, not whole-chain OOF; original parent NN/head ancestry remains. Local utility not full-week/actual-latency.'))
 paths=[W/'run.py',O/'design.json',P/'source_freeze_recovery01.json',P/'qualified_freeze.json',P/'qualification.json',P/'jobs.json',P/'independent_audit.json',R/'outputs/grid253_stop_source_plan/fixed_pipeline_proposal.json']
 paths += [R/rel for rel in read(P/'source_freeze_recovery01.json')]
 paths += [P/rel for rel in read(P/'qualified_freeze.json')]
 save(O/'source_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in set(paths)})
 verify();D=A/'source';D.mkdir();shutil.copy2(W/'run.py',D/'run.py')
 for p in [O/'design.json',O/'source_freeze.json',P/'qualification.json',P/'qualified_freeze.json',P/'jobs.json']:shutil.copy2(p,D/p.name)
 for name in ['worker.py','manager.py','runtime254.py']:shutil.copy2(CW/name,D/name)
 save(D/'archive_manifest.json',{p.name:sha(p) for p in D.iterdir() if p.is_file()})
 print('GRID255_SOURCE_READY_ZERO_FITS')

def authorize():
 assert not (O/'authorization.json').exists() and not (O/'training_started.json').exists();verify()
 source_head=published([p for p in (A/'source').iterdir() if p.is_file()]);save(O/'authorization.json',dict(passed=True,qualification_sha256=sha(P/'qualification.json'),qualified_freeze_sha256=sha(P/'qualified_freeze.json'),jobs_sha256=sha(P/'jobs.json'),all_sources_inputs_published=True,models=18,fit_cap_s=600,source_published_head=source_head,source_freeze_sha256=sha(O/'source_freeze.json'),qualified_code_reused=True))
 d=A/'authorization';d.mkdir();shutil.copy2(O/'authorization.json',d/'authorization.json');print('GRID255_AUTH_READY',source_head)

def start():
 assert not (O/'launch_once.json').exists() and not (O/'training_started.json').exists() and not (O/'run_context.json').exists();verify()
 assert (A/'authorization/authorization.json').read_bytes()==(O/'authorization.json').read_bytes();head=published([A/'authorization/authorization.json',A/'source/source_freeze.json',A/'source/run.py']);assert read(O/'authorization.json')['source_freeze_sha256']==sha(O/'source_freeze.json')
 with (O/'launch_once.json').open('x',encoding='utf8') as f:json.dump(dict(pid=os.getpid(),published_head=head,one_manager_only=True,launch_wall_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f,indent=2)
 env=dict(os.environ,TEMP=str(R/'tmp/grid254'),TMP=str(R/'tmp/grid254'),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1')
 with (O/'manager_stdout.log').open('wb') as fo,(O/'manager_stderr.log').open('wb') as fe:
  p=subprocess.Popen([sys.executable,'-X','utf8','-u',str(CW/'manager.py'),'train'],stdout=fo,stderr=fe,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
  save(O/'manager_process.json',dict(pid=p.pid,parent_pid=os.getpid(),single_manager=True));code=p.wait()
 save(O/'launcher_finished.json',dict(exit_code=code,passed=code==0,reference_finished_exists=(O/'reference_finished.json').exists(),no_scoring=True));print('GRID255_MANAGER_EXIT',code)
 if code:raise RuntimeError('Fixed reference component failed; preserve partial and complete outputs, no automatic retry.')

if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  if O.exists():save(O/('entry_'+sys.argv[1]+'_error.json'),dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True))
  raise
