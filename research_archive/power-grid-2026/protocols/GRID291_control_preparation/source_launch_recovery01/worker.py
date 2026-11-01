"""Authorization checked before physical/model imports. Called only by manager."""
from pathlib import Path
import sys,os,argparse,time,json
from guard import R,O,read,sha,require_authorization,verify_frozen,manager_ancestor_matches
W=Path(__file__).parent
ap=argparse.ArgumentParser();ap.add_argument('--index',type=int,required=True);args=ap.parse_args()
require_authorization();verify_frozen();D=read(O/'design.json');item=D['execution_order'][args.index]
ctx=read(O/'run_context.json');claim=read(O/'claims'/f'{args.index:03d}.json')
if not os.environ.get('GRL_GRID291_TOKEN') or os.environ['GRL_GRID291_TOKEN']!=ctx['token'] or claim['token']!=ctx['token'] or not manager_ancestor_matches(ctx['manager_pid']):raise PermissionError('worker not claimed by manager')
if time.perf_counter()>=ctx['deadline_monotonic']:raise TimeoutError('global deadline before worker')
with (O/'claims'/f'{args.index:03d}.started').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
from source_builder import build
source=R/'work/grid55/run.py';code=build()
sys.argv=[str(source),'--worker',str(args.index)]
exec(compile(code,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
