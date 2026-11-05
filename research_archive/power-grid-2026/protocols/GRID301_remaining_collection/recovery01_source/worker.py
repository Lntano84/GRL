"""Fail closed before real model imports; one immutable manager claim."""
import argparse,importlib.util,os,sys
from pathlib import Path
from guard import R,W,O,read,claim,check,verify_stage,verify_generated
def main():
 a=argparse.ArgumentParser();a.add_argument('--index',type=int,required=True);index=a.parse_args().index;ctx=claim('replay',index)
 D=read(O/'design.json');runtime=read(O/'runtime_design.json');item=runtime['execution_order'][index]
 if index<144:
  assert runtime==D['baseline_runtime'] and item==D['baseline_runtime']['execution_order'][index]
 else:
  verify_stage('baseline');assert (O/'runtime_design.json').read_bytes()==(O/'stage_payload/baseline/runtime_design.json').read_bytes()
 role=next(x for x in read(O/'pool_roles.json') if x['pool']==item['pool']);assert item['date'] in role['fit_dates' if item['partition']=='own_training_fit8' else 'cal_dates'] and item['date'] not in role['outer_dates_excluded'] and item['KEEP']==role['KEEP']
 verify_generated(item['week']);check(ctx)
 from source_builder import build
 s=importlib.util.spec_from_file_location('paired_adapter',W/'adapter.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);sys.modules['paired_adapter']=m
 sys.argv=[str(R/'work/grid55/run.py'),'--worker',str(index)]
 exec(compile(build(),str(R/'work/grid55/run.py'),'exec'),{'__file__':str(R/'work/grid55/run.py'),'__name__':'__main__','check_runtime_budget':lambda:check(ctx),'run_ctx':ctx})
if __name__=='__main__':main()
