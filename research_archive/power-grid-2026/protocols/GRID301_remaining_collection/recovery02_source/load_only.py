"""Paid pure-loading gate after explicit recovery02 authorization, never predict."""
import argparse,ast,os,sys
from guard import R,W,O,read,save,require_authorization,verify_frozen,check,ancestor
def main():
 require_authorization();p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);index=p.parse_args().index
 assert index in range(0,144,12);ctx=read(O/'load_only_context.json');assert ctx['index']==index and os.environ.get('GRL_GRID300_TOKEN')==ctx['token'] and ancestor(ctx['manager_pid']);check(ctx);verify_frozen()
 with (O/f'load_only_recovery02_{index:03d}.claimed').open('x',encoding='utf8') as f:f.write(str(os.getpid()))
 from source_builder import build
 text=build();tree=ast.parse(text);branch=next(n for n in tree.body if isinstance(n,ast.If) and 'args.worker' in ast.unparse(n.test))
 stop=next(i for i,n in enumerate(branch.body) if isinstance(n,ast.Try) and 'summary = worker.worker' in ast.unparse(n))
 branch.body=branch.body[:stop];tree.body=tree.body[:tree.body.index(branch)+1]
 sys.argv=[str(R/'work/grid55/run.py'),'--worker',str(index)]
 scope={'__file__':str(R/'work/grid55/run.py'),'__name__':'g301_load_only','check_runtime_budget':lambda:check(ctx),'run_ctx':{**ctx,'forecast_remaining':0,'physical_remaining':0}}
 exec(compile(ast.fix_missing_locations(tree),'assembled_load_only','exec'),scope)
 assert not (O/'runs').exists();assert scope['worker']._intervention.steps==scope['worker']._intervention.forecasts==0
 assert scope['item']['pool']==scope['payload297'].pool
 save(O/f'load_only_recovery02_{index:03d}.json',{'pool':scope['item']['pool'],'model_payload_loaded':True,'new_forward':0,'new_environment':0,'new_physics':0,'stopped_before_worker_worker':True})
 check(ctx)
if __name__=='__main__':main()
