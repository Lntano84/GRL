"""Independent policy, hidden-tail and runtime interface qualification before timing."""
from pathlib import Path
import ast,gzip,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid88';sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D=read(OUT/'design.json');assert len(D['execution_order'])==52 and len(set(D['weeks']))==13
 assert P.POLICIES==('FRAME_GUARD','FIXED_512')
 for n in ['code_freeze.json','data_freeze.json']:
  for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
 with gzip.open(ROOT/'outputs/grid85_mechanism/mechanism_feedback.json.gz','rt',encoding='utf8') as f:pack=json.load(f)
 sp=importlib.util.spec_from_file_location('grid88_independent',ROOT/'work/grid85_mechanism/verify.py');V=importlib.util.module_from_spec(sp);sp.loader.exec_module(V)
 cases=checks=tails=0
 for source,q in pack['records'].items():
  if not source.endswith('.json') or not source.split('/')[-2].endswith('__FULL'):continue
  c=dict(q,outcomes=[dict(zip(pack['fields'],v)) for v in q['outcomes']])
  for x in c['outcomes']:x['exceptions']=x.pop('has_exception')
  truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):int(v) for k,v in c['aliases'].items()};cases+=1
  for name in P.POLICIES:
   seen=[]
   def reveal(i,p):assert i not in seen;seen.append(i);return dict(truth[i],phase=p)
   got=P.run(c['order'],reveal,name,pack['subids'],aliases,limit=c['rho_limit']);ref=V.independent(c,name,pack['subids'])
   assert got['chosen_pool_id']==V.pid(V.best(ref)) and seen==[r['pool_id'] for r in ref] and got['public_queries']==len(ref)
   if name=='FIXED_512':assert len(seen)==min(512,len(c['order'])) and got['local_queries']==0
   prior=list(seen);seen=[];unknown=set(truth)-set(prior)
   def perturb(i,p):
    assert i not in seen;seen.append(i);v=dict(truth[i],phase=p)
    if i in unknown:v.update(reward=9999.,rho=.01,done=False,illegal=False,ambiguous=False,exceptions=False,strict_admissible=True)
    return v
   again=P.run(c['order'],perturb,name,pack['subids'],aliases,limit=c['rho_limit'])
   assert got==again and seen==prior;checks+=1;tails+=1
 assert cases==137 and checks==tails==274
 hand=0
 for size,admiss,expect in [(0,None,None),(100,'all',0),(600,'all',0),(600,550,None),(600,511,511),(600,None,None)]:
  seen=[]
  def observe(i,p):
   seen.append(i);return dict(pool_id=i,rho=.8,reward=1.+(1e-9 if i==130 else 0),strict_admissible=admiss=='all' or i==admiss,phase=p)
  v=P.run(list(range(size)),observe,'FIXED_512',[0]*size,{},limit=1.)
  assert v['chosen_pool_id']==expect and len(seen)==min(size,512);hand+=1
 tree=ast.parse((ROOT/'work/grid55/worker.py').read_text(encoding='utf8'));required=set()
 for x in ast.walk(tree):
  if isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id=='design' and isinstance(x.slice,ast.Constant):required.add(x.slice.value)
 assert required=={'caps','execution_order','timing'} and required<=set(D)
 for w in D['weeks']:
  a=[v['method'] for v in D['execution_order'] if v['repeat']==0 and v['week']==w];b=[v['method'] for v in D['execution_order'] if v['repeat']==1 and v['week']==w]
  assert a==b[::-1] and set(a)==set(P.POLICIES)
 assert [v['week'] for v in D['execution_order'] if v['repeat']==1][::2]==D['weeks'][::-1]
 sys.path.insert(0,str(ROOT/'work/grid08'));import environment
 assert Path(environment.__file__).resolve()==(ROOT/'work/grid88/environment.py').resolve()
 assert sha(ROOT/'work/grid88/environment.py')==sha(ROOT/'work/grid87/environment.py')
 env=environment.make_env()
 try:
  assert int(env.nb_highres_called)==0
  for w in D['weeks']:assert (ROOT/'work/grid03/formal_env_initialized/chronics'/w).is_dir()
 finally:env.close()
 for p in Path(__file__).parent.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
 result=dict(passed=True,paid_full_states=cases,independent_policy_checks=checks,hidden_suffix_checks=tails,edge_cases=hand,old_environment_constructed=True,dates=13,planned_runs=52,balanced_reverse_schedule=True,required_worker_design_keys=sorted(required),new_simulations=0,new_fits=0)
 (OUT/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
