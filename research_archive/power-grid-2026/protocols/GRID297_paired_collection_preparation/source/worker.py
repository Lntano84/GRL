"""No simulator/model imports until exact new authorization and manager claim."""
import argparse,os,sys,time
from pathlib import Path
from guard import R,O,read,require_authorization,verify_frozen,manager_ancestor_matches
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);a=ap.parse_args()
 require_authorization();verify_frozen();ctx=read(O/'run_context.json');claim=read(O/'claims'/f'{a.index:03d}.json')
 if os.environ.get('GRL_GRID297_TOKEN')!=ctx['token'] or claim['token']!=ctx['token'] or claim['index']!=a.index or not manager_ancestor_matches(ctx['manager_pid']):raise PermissionError('worker not claimed by exact manager')
 def check():
  if time.perf_counter()>=min(ctx['deadline_monotonic'],ctx['worker_deadline_monotonic']):raise TimeoutError('fixed collection deadline')
 check()
 with (O/'claims'/f'{a.index:03d}.started').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
 from source_builder import build
 # grid55 temporarily prepends another module named adapter; bind ours explicitly.
 import importlib.util
 s=importlib.util.spec_from_file_location('paired_adapter',Path(__file__).with_name('adapter.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);sys.modules['paired_adapter']=m
 sys.argv=[str(R/'work/grid55/run.py'),'--worker',str(a.index)]
 exec(compile(build(),str(R/'work/grid55/run.py'),'exec'),dict(__file__=str(R/'work/grid55/run.py'),__name__='__main__',check_runtime_budget=check,run_ctx=ctx))
if __name__=='__main__':main()
