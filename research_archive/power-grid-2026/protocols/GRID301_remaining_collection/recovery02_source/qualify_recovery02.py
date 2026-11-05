"""One complete namespace JSON/AST check. No model loading, prediction or environment."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent/'recovery02';O=R/'outputs/grid301_remaining_collection';OLD=R/'outputs/grid300_new_training_dates';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID301_remaining_collection'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(I),*a])
def main():
 start=time.perf_counter();assert not (O/'recovery02_preparation_review.json').exists() and not (O/'recovery02_authorization.json').exists()
 assert not git('status','--porcelain').strip();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'recovery02_source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 contract=read(O/'runtime_namespace_contract.json');assert len(contract['immutable_metadata'])==7
 for n,v in contract['immutable_metadata'].items():assert sha(O/n)==sha(R/v['source'])==v['sha256']
 refs=read(O/'model_references.json');frozen=read(O/'model_freeze.json');roles=read(O/'pool_roles.json');assert len(refs)==len(roles)==12
 poolcheck=[]
 for ref in refs:
  role=next(r for r in roles if r['pool']==ref['pool']);assert ref['owner_fold']==int(role['pool'][1]) and ref['seed']==int(role['pool'][-1])
  fields=['checkpoint','normalization','parent_scaler','residual_public41_scaler','residual_model'];paths=[ref[k] for k in fields]+ref['parent_heads'];assert len(paths)==7
  for p in paths:assert p in ref['hashes'] and ref['hashes'][p]==frozen[p] and (R/p).is_file()
  assert len(ref['state_hash'])==64;poolcheck.append(ref['pool'])
 namespace_reads={}
 for p,h in contract['source_inspection'].items():
  assert sha(R/p)==h;tree=ast.parse((R/p).read_text(encoding='utf8'));hits=[]
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['load','read'] and n.args:
    arg=n.args[0]
    if isinstance(arg,ast.BinOp) and isinstance(arg.op,ast.Div) and isinstance(arg.left,ast.Name) and arg.left.id in ['O','OUT'] and isinstance(arg.right,ast.Constant) and isinstance(arg.right.value,str):hits.append(arg.right.value)
  namespace_reads[p]=sorted(set(hits))
 assert namespace_reads['work/grid291_control_preparation/models.py']==['model_references.json']
 assert 'code_freeze.json' in namespace_reads['work/grid55/run.py']
 # Original worker main() preflight/offline reads are not used by our worker() call.
 sys.path.insert(0,str(W));import source_builder
 text=source_builder.build();tree=ast.parse(text);branch=next(n for n in tree.body if isinstance(n,ast.If) and 'args.worker' in ast.unparse(n.test))
 header=tree.body[:tree.body.index(branch)];oldargv=sys.argv[:]
 try:
  sys.argv=[str(R/'work/grid55/run.py'),'--worker','36'];scope={'__file__':str(R/'work/grid55/run.py'),'__name__':'static_header'}
  exec(compile(ast.Module(body=header,type_ignores=[]),'assembled_header','exec'),scope)
 finally:sys.argv=oldargv
 assert scope['OUT']==O and scope['args'].worker==36
 stop=next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Try) and 'summary = worker.worker' in ast.unparse(n));truncated=branch.body[:stop]
 assert 'payload297 = models297.FrozenModels' in ast.unparse(ast.Module(body=truncated,type_ignores=[]))
 assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['worker','simulate','step','make_env','native','extension'] for b in truncated if not isinstance(b,(ast.FunctionDef,ast.ClassDef)) for n in ast.walk(b))
 for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 before=sha(O/'launch_context.json');denied=[]
 for name,args in [('launch.py',[]),('manager.py',[]),('worker.py',['--index','36']),('publish_stage.py',['--stage','baseline']),('load_only.py',['--index','0'])]:
  p=subprocess.run([sys.executable,'-B','-X','utf8',str(W/name),*args],cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20)
  out=p.stdout.decode('utf8',errors='replace');assert p.returncode!=0 and 'GRID301 recovery02 authorization absent' in out;denied.append(name);(O/(name+'.recovery02_denied.txt')).write_text(out,encoding='utf8')
 assert before==sha(O/'launch_context.json') and not (O/'runs').exists() and not (O/'started_recovery02.json').exists()
 assert not any(n in sys.modules for n in ['torch','grid2op','common']) and time.perf_counter()-start<600
 review={'passed_static_preparation':True,'passed':True,'source_commit':head,'exact_source_blobs':blobs,'immutable_metadata_count':7,'owner_pool_references_checked':poolcheck,'payload_paths_per_pool':7,'namespace_literal_reads':namespace_reads,'assembled_header_metadata_passed':True,'load_only_AST_stops_before_worker_and_env':True,'missing_approval_entrypoints_denied':denied,'planned_actual_load_only_pools':12,'actual_model_loads_so_far':0,'actual_new_NN_forward':0,'actual_new_physics':0,'actual_new_fits':0,'original_launch_context_sha256':before,'original_clock_unchanged':True,'wall_s':time.perf_counter()-start,'live_load_only_check_not_yet_run':True}
 (O/'recovery02_preparation_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print('RECOVERY02_STATIC_NAMESPACE_PASSED',len(blobs),len(refs),len(denied),time.perf_counter()-start)
if __name__=='__main__':main()
