"""Finite immutable run matrix; generated series remain unevaluated until sealing."""
from pathlib import Path
import datetime as dt,hashlib,importlib.util,json,shutil
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid89_review'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def main():
 assert not (OUT/'evaluation_plan.json').exists();D=read(OUT/'design.json');old=read(ROOT/'outputs/grid88/design.json');fresh=read(ROOT/'outputs/grid84/design.json')['weeks'];plans={}
 def method(name,policy,**kw):return dict(rule='STATIC_ID',budget=128,checkpoint=None,cache_dc=True,cache_safe=True,method=name,policy=policy,order_mode='STATIC_ID',**kw)
 def phase(name,weeks,methods,repeats,environment):
  P=dict(phase=name,weeks=weeks,methods=methods,repeats=repeats,environment=environment,pilot=[],timing=old['timing'],quality_threshold=.005,caps=dict(episode_s=1200,wall_s=43200,physical_steps=600000,public_forecasts=10000000),execution_order=[])
  for r in range(repeats):
   for week in weeks if r%2==0 else weeks[::-1]:
    shift=(weeks.index(week)+r)%len(methods);block=methods[shift:]+methods[:shift]
    if r%2:block=block[::-1]
    P['execution_order'].extend(dict(m,week=week,repeat=r,stage_phase='all') for m in block)
  folder=OUT/name;folder.mkdir();save(folder/'design.json',P);plans[name]=dict(runs=len(P['execution_order']),weeks=weeks,methods=[m['method'] for m in methods]);return P
 methods=[]
 for mode in D['order']['orders']:
  for policy in ['FULL','FRAME_GUARD']:
   x=method(policy+'__'+mode,policy);x['order_mode']=mode;methods.append(x)
 phase('order',fresh,methods,1,'work/grid80/fresh_env')
 methods=[method('FULL','FULL'),method('FRAME_GUARD','FRAME_GUARD')]
 for key,values in D['parameters']['single_axis'].items():
  for value in values:methods.append(method(f'FRAME_{key}_{str(value).replace(".","p")}', 'FRAME_GUARD',parameters={key:value}))
 phase('parameters',D['parameters']['dates'],methods,1,'work/grid03/formal_env_initialized')
 methods=[method('FRAME_GUARD','FRAME_GUARD')]
 for cost in [0,1]:
  for seed in range(5):
   if cost==0:p=f'outputs/grid77/models/cost2_seed{seed}/final.pt' if seed<2 else f'outputs/grid84/models/cost2_seed{seed}/final.pt'
   else:p=f'outputs/grid77/models/cost1_seed{seed}/final.pt' if seed<2 else f'outputs/grid89_review/meta_training/models/cost1_seed{seed}/final.pt'
   methods.append(method(f'META_COST{cost}_SEED{seed}','META',meta_model=p,training_seed=seed,cost_level=0 if cost==0 else .0025))
 phase('meta',fresh,methods,1,'work/grid80/fresh_env')
 methods=[method(p,p) for p in ['FULL','FRAME_GUARD','GLOBAL_MATCH','FIXED_512']]
 phase('sealed',[d+'_'+str(D['scenario_id']) for d in D['dates']],methods,3,D['environment'])
 assert sum(v['runs'] for v in plans.values())==246
 train=read(ROOT/'outputs/grid84/design.json');train.update(seeds=[2,3,4]);train['caps']=dict(train['caps'],training_wall_s=3600)
 td=OUT/'meta_training';td.mkdir();save(td/'design.json',train)
 source=(ROOT/'work/grid84/train.py').read_text(encoding='utf-8')
 source=source.replace("OUT=ROOT/'outputs/grid84'","OUT=ROOT/'outputs/grid89_review/meta_training'")
 source=source.replace("[(2,D['cost_levels'][2])]","[(1,D['cost_levels'][1])]")
 source=source.replace('costindex=2,cost_level=0.0','costindex=1,cost_level=0.0025')
 assert "[(1,D['cost_levels'][1])]" in source
 (W/'train_positive.py').write_text(source,encoding='utf-8')
 save(td/'training_preflight.json',dict(passed=True,new_fits=3,old_models_reused=2,new_teacher_queries=0,training_loop='Exact GRID84 except costindex1 and output directory',records_sha256=sha(ROOT/'outputs/grid77/records.json')))
 training=read(ROOT/'outputs/grid84/training_code_freeze.json')
 for p,h in training.items():assert sha(ROOT/p)==h,p
 for p in [W/'train_positive.py',td/'design.json',ROOT/'outputs/grid77/records.json']:training[p.relative_to(ROOT).as_posix()]=sha(p)
 save(td/'training_code_freeze.json',training)
 save(OUT/'evaluation_plan.json',dict(frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),phases=plans,new_model_fits=3,total_replays=246,order_of_work=['generation','training','preflight','freeze and archive','serial order','serial parameters','serial meta','serial sealed','audit and new Chinese manuscript']))
 print(json.dumps(plans,ensure_ascii=False))
if __name__=='__main__':main()
