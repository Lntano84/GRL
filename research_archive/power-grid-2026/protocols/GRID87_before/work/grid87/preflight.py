"""137 independent fixed-prefix checks, hidden-tail invariance and old-data qualification."""
from pathlib import Path
import ast,gzip,hashlib,importlib.util,json,sys
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid87';sys.path.insert(0,str(Path(__file__).parent));import policies as P
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 D=read(OUT/'design.json');assert len(D['execution_order'])==13 and len(set(D['weeks']))==13
 for n in ['code_freeze.json','data_freeze.json']:
  for p,h in read(OUT/n).items():assert sha(ROOT/p)==h,p
 with gzip.open(ROOT/'outputs/grid85_mechanism/mechanism_feedback.json.gz','rt',encoding='utf8') as f:pack=json.load(f)
 sp=importlib.util.spec_from_file_location('grid87_independent',ROOT/'work/grid85_mechanism/verify.py');V=importlib.util.module_from_spec(sp);sp.loader.exec_module(V)
 checks=tails=0
 for source,q in pack['records'].items():
  if not source.endswith('.json') or not source.split('/')[-2].endswith('__FULL'):continue
  c=dict(q,outcomes=[dict(zip(pack['fields'],v)) for v in q['outcomes']])
  for x in c['outcomes']:x['exceptions']=x.pop('has_exception')
  truth={r['pool_id']:r for r in c['outcomes']};seen=[]
  def reveal(i,p):assert i not in seen;seen.append(i);return dict(truth[i],phase=p)
  got=P.run(c['order'],reveal,'FIXED_256',pack['subids'],{int(k):int(v) for k,v in c['aliases'].items()},limit=c['rho_limit'])
  ref=V.independent(c,'FIXED_256',pack['subids']);assert got['chosen_pool_id']==V.pid(V.best(ref)) and seen==[r['pool_id'] for r in ref]
  assert got['public_queries']==min(256,len(c['order'])) and got['local_queries']==0
  prior=list(seen);seen=[];unknown=set(truth)-set(prior)
  def perturb(i,p):
   assert i not in seen;seen.append(i);v=dict(truth[i],phase=p)
   if i in unknown:v.update(reward=9999,rho=.01,strict_admissible=True)
   return v
  again=P.run(c['order'],perturb,'FIXED_256',pack['subids'],{int(k):int(v) for k,v in c['aliases'].items()},limit=c['rho_limit'])
  assert got==again and seen==prior;checks+=1;tails+=1
 assert checks==tails==137
 hand=0
 for size,admiss,expect in [(0,None,None),(100,'all',0),(600,'all',0),(600,400,None),(600,255,255),(600,None,None)]:
  seen=[]
  def observe(i,p):
   seen.append(i);return dict(pool_id=i,rho=.8,reward=1.+(1e-9 if i==130 else 0),strict_admissible=admiss=='all' or i==admiss,phase=p)
  v=P.run(list(range(size)),observe,'FIXED_256',[0]*size,{},limit=1.)
  assert v['chosen_pool_id']==expect and len(seen)==min(size,256);hand+=1
 tree=ast.parse((ROOT/'work/grid55/worker.py').read_text(encoding='utf8'));required=set()
 for x in ast.walk(tree):
  if isinstance(x,ast.Subscript) and isinstance(x.value,ast.Name) and x.value.id=='design' and isinstance(x.slice,ast.Constant):required.add(x.slice.value)
 assert required<=set(D),required-set(D)
 sys.path.insert(0,str(ROOT/'work/grid08'));import environment
 assert Path(environment.__file__).resolve()==(ROOT/'work/grid87/environment.py').resolve()
 env=environment.make_env()
 try:
  for w in D['weeks']:assert (ROOT/'work/grid03/formal_env_initialized/chronics'/w).is_dir()
  assert int(env.nb_highres_called)==0
 finally:env.close()
 for p in Path(__file__).parent.glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
 result=dict(passed=True,independent_policy_checks=checks,hidden_suffix_checks=tails,edge_cases=hand,old_environment_constructed=True,dates=13,required_worker_design_keys=sorted(required),new_simulations=0,new_fits=0)
 (OUT/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
