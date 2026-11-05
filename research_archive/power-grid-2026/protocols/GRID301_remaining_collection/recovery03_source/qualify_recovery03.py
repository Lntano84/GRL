"""Static boundary and existing evidence only; no further actual model loads."""
from pathlib import Path
import ast,hashlib,json,os,subprocess,sys,time
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent/'recovery03';O=R/'outputs/grid301_remaining_collection';I=R/'work/GRL-results-integration';A=I/'research_archive/power-grid-2026/protocols/GRID301_remaining_collection'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*a):return subprocess.check_output(['git','-C',str(I),*a])
def main():
 start=time.perf_counter();assert not (O/'recovery03_preparation_review.json').exists() and not (O/'recovery03_authorization.json').exists()
 assert not git('status','--porcelain').strip();head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
 blobs={}
 for p in (A/'recovery03_source').iterdir():
  if p.is_file():rel=p.relative_to(I).as_posix();assert git('show',head+':'+rel)==p.read_bytes();blobs[rel]=sha(p)
 proof=read(O/'prior_f0_s0_load_evidence.json');assert proof['payload_loader_completed'] and not proof['whole_assembled_checker_passed'] and proof['pool']=='f0_s0'
 assert sha(R/'work/grid291_control_preparation/models.py')==proof['frozen_models_source_sha256'] and sha(O/'load_only_recovery02_000.log')==proof['failure_log_sha256']
 direct=(W/'load_only.py').read_text(encoding='utf8');tree=ast.parse(direct);calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert sum(isinstance(n.func,ast.Attribute) and n.func.attr=='FrozenModels' for n in calls)==1
 assert not any(isinstance(n.func,ast.Attribute) and n.func.attr in ['native','extension','worker','step','simulate','make_env','predict','forward','predict_proba'] for n in calls)
 assert 'source_builder' not in direct and 'paired_adapter' not in direct
 pre=(W/'preflight.py').read_text(encoding='utf8');assert 'range(12,144,12)' in pre and 'prior_f0_s0_load_evidence.json' in pre and "'new_loaded_pools':11" in pre
 for p in W.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
 unchanged=['state.py','adapter.py','worker.py','source_builder.py','stages.py','publish_stage.py','auto_publish.py']
 assert all(sha(W/n)==sha(W.parent/'recovery02'/n) for n in unchanged)
 before=sha(O/'launch_context.json');denied=[]
 for n,args in [('launch.py',[]),('manager.py',[]),('load_only.py',['--index','12']),('worker.py',['--index','36']),('publish_stage.py',['--stage','baseline'])]:
  p=subprocess.run([sys.executable,'-B','-X','utf8',str(W/n),*args],cwd=R,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=20);out=p.stdout.decode('utf8',errors='replace')
  assert p.returncode!=0 and 'GRID301 recovery03 authorization absent' in out;denied.append(n);(O/(n+'.recovery03_denied.txt')).write_text(out,encoding='utf8')
 assert before==sha(O/'launch_context.json') and not (O/'started_recovery03.json').exists() and not (O/'runs').exists() and time.perf_counter()-start<600
 v={'passed':True,'source_commit':head,'exact_source_blobs':blobs,'prior_completed_payload_load_evidence_retained':True,'whole_old_assembled_checker_passed':False,'direct_FrozenModels_call_count':1,'new_loader_indices':list(range(12,144,12)),'remaining_model_loads':11,'unchanged_numerical_modules':unchanged,'missing_approval_entrypoints_denied':denied,'actual_additional_model_loads_in_check':0,'actual_new_NN_forward':0,'actual_new_physics':0,'actual_new_fits':0,'original_launch_context_sha256':before,'wall_s':time.perf_counter()-start}
 (O/'recovery03_preparation_review.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('DIRECT_LOADER_STATIC_CHECK_PASSED',len(blobs),len(denied),time.perf_counter()-start)
if __name__=='__main__':main()
