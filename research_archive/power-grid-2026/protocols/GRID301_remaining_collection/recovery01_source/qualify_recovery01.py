"""Check the previously omitted stdlib metadata header, without live imports."""
from pathlib import Path
import ast,hashlib,importlib.util,json,os,subprocess,sys,time
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent/'recovery01';O=R/'outputs/grid301_remaining_collection';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID301_remaining_collection'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(I),*a])
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def main():
 start=time.perf_counter();assert not (O/'recovery01_preparation_review.json').exists() and not (O/'recovery01_authorization.json').exists() and not (O/'started_recovery01.json').exists()
 assert not git('status','--porcelain').strip();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'recovery01_source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 assert sha(O/'code_freeze.json')==sha(R/'outputs/grid300_new_training_dates/code_freeze.json')
 before=sha(O/'launch_context.json');sys.path.insert(0,str(W));import source_builder
 text=source_builder.build();tree=ast.parse(text)
 boundary=next(i for i,n in enumerate(tree.body) if isinstance(n,ast.If) and 'args.worker' in ast.unparse(n.test))
 header=tree.body[:boundary]
 imports=[n for root in header for n in ast.walk(root) if isinstance(n,(ast.Import,ast.ImportFrom))]
 allowed={'argparse','hashlib','importlib.util','json','os','subprocess','sys','time','pathlib'}
 for n in imports:
  names=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module]
  assert all(x in allowed for x in names),names
 assert 'code_freeze.json' in ast.unparse(ast.Module(body=header,type_ignores=[]))
 oldargv=sys.argv[:];oldenv={n:os.environ.get(n) for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}
 try:
  sys.argv=[str(R/'work/grid55/run.py'),'--worker','36'];scope={'__file__':str(R/'work/grid55/run.py'),'__name__':'g301_static_header_only'}
  exec(compile(ast.Module(body=header,type_ignores=[]),'assembled_static_header','exec'),scope)
  assert scope['args'].worker==36 and scope['OUT']==O
 finally:
  sys.argv=oldargv
  for n,v in oldenv.items():
   if v is None:os.environ.pop(n,None)
   else:os.environ[n]=v
 assert 'common' not in sys.modules and 'torch' not in sys.modules and 'grid2op' not in sys.modules
 copied=['state.py','adapter.py','worker.py','source_builder.py','stages.py','publish_stage.py','auto_publish.py']
 assert all(sha(W/n)==sha(W.parent/n) for n in copied)
 for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 denied=[]
 for n,args in [('launch.py',[]),('manager.py',[]),('worker.py',['--index','36']),('publish_stage.py',['--stage','baseline'])]:
  p=subprocess.run([sys.executable,'-B','-X','utf8',str(W/n),*args],cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
  msg=p.stdout.decode('utf8',errors='replace');assert p.returncode!=0 and 'GRID301 recovery01 authorization absent' in msg
  denied.append(n);(O/(n+'.recovery01_denied.txt')).write_text(msg,encoding='utf8')
 assert before==sha(O/'launch_context.json') and not (O/'runs').exists() and not (O/'started_recovery01.json').exists() and not (O/'claims_recovery01').exists()
 assert time.perf_counter()-start<600
 save(O/'recovery01_preparation_review.json',{'passed':True,'source_commit':head,'exact_source_blobs':blobs,'static_assembled_header_executed_before_live_branch':True,'code_freeze_exact_old_bytes':True,'original_header_source_hash_entries':len(read(O/'code_freeze.json')),'copied_numerical_and_stage_modules_identical':copied,'original_launch_context_sha256':before,'original_clock_unchanged':True,'missing_recovery_approval_entries_denied':denied,'new_model_loads':0,'new_predictions':0,'new_physics':0,'new_labels':0,'new_fits':0,'wall_s':time.perf_counter()-start,'no_old_full_qualification_repeated':True})
 print('RECOVERY_STATIC_HEADER_PASSED',len(blobs),len(denied),time.perf_counter()-start)
if __name__=='__main__':main()
