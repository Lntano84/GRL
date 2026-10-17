"""Production-default equivalence, independent plans and unrevealed-suffix invariance."""
from pathlib import Path
import csv,hashlib,importlib.util,json,random
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;O=ROOT/'outputs/grid89_review'
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('candidate_policy',W/'policies.py');I=load('independent_review',W/'reference_logic.py')
def main():
 subs=json.loads((ROOT/'outputs/grid28/offline.json').read_text())['public_subids'];rows=list(csv.DictReader((ROOT/'outputs/grid85_mechanism/offline_full_calls.csv').open(encoding='utf-8-sig')))
 cases=[r for r in rows if r['method']=='FULL'];checks=dict(default_equivalence=0,independent_query_plans=0,unrevealed_suffix_perturbations=0,boundaries=0)
 variants=[{}]+[{k:v} for k,vs in json.loads((O/'design.json').read_text(encoding='utf-8'))['parameters']['single_axis'].items() for v in vs]
 for case in cases:
  c=json.loads((ROOT/case['source']).read_text());truth={r['pool_id']:r for r in c['outcomes']};aliases={int(k):int(v) for k,v in c['aliases'].items()};order=c['order']
  def original(i,phase):return dict(truth[i],phase=phase)
  got=P.run(order,original,'FRAME_GUARD',subs,aliases);ref=P.REF.run(order,original,'FRAME_GUARD',subs,aliases)
  assert [r['pool_id'] for r in got['outcomes']]==[r['pool_id'] for r in ref['outcomes']]
  assert got['chosen_pool_id']==ref['chosen_pool_id'] and got['public_queries']==ref['public_queries'];checks['default_equivalence']+=1
  for patch in variants:
   cfg=dict(P.DEFAULT,**patch)
   for policy in ['FRAME_GUARD','GLOBAL_MATCH']:
    selected=P.run(order,original,policy,subs,aliases,patch)
    plan,choice,budget=I.replay(order,truth,subs,aliases,policy,cfg)
    assert plan==[r['pool_id'] for r in selected['outcomes']] and choice==selected['chosen_pool_id'] and budget==selected['completion_budget'];checks['independent_query_plans']+=1
    revealed=set(plan)
    def hidden(i,phase):
     r=dict(truth[i],phase=phase)
     if i not in revealed:r.update(reward=1e6,rho=.01,strict_admissible=True)
     return r
    other=P.run(order,hidden,policy,subs,aliases,patch)
    assert [r['pool_id'] for r in other['outcomes']]==plan and other['chosen_pool_id']==choice;checks['unrevealed_suffix_perturbations']+=1
 for n in [0,1,63,64,127,128,129,255,256,257,512,909]:
  order=list(range(n));a={i:i for i in order};ss=[0]*n
  for valid in [False,True]:
   truth={i:dict(pool_id=i,reward=1.+(1e-9 if i%2 else 0),rho=.8,strict_admissible=valid) for i in order}
   for policy in ['FULL','FIXED_512','FRAME_GUARD','GLOBAL_MATCH']:
    g=P.run(order,lambda i,phase:dict(truth[i],phase=phase),policy,ss,a);plan,choice,b=I.replay(order,truth,ss,a,policy,P.DEFAULT)
    assert plan==[r['pool_id'] for r in g['outcomes']] and choice==g['chosen_pool_id'];checks['boundaries']+=1
 assert checks['default_equivalence']==137
 result=dict(passed=True,checks=checks,new_physical_queries=0,source='137 existing FULL states, all scalar variants, independent logic; no new solver/simulation feedback',source_sha256=hashlib.sha256((W/'policies.py').read_bytes()).hexdigest())
 (O/'preflight.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
