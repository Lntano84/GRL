"""Finite synthetic control-flow and authorization qualification; zero learned predictions."""
from pathlib import Path
import sys,ast,json,importlib.util,subprocess,hashlib,types
sys.dont_write_bytecode=True
import numpy as np
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid291_control_preparation';repo=R/'work/GRL-results-integration'
def rd(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(n,p):
 s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
assert not (O/'preparation_review.json').exists()
for p,h in rd(O/'source_freeze.json').items():assert sha(R/p)==h,p
g=lambda *a:subprocess.check_output(['git','-C',str(repo),*a]);head=g('rev-parse','HEAD').decode().strip();assert g('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
assert g('show',head+':research_archive/power-grid-2026/protocols/GRID291_control_preparation/source_recovery02/source_freeze.json')==(O/'source_freeze.json').read_bytes()
for p in W.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
b=load('g291_source_builder',W/'source_builder.py');compile(b.build(),'generated_worker','exec');compile(b.instrument_worker((R/'work/grid55/worker.py').read_text(encoding='utf8')),'generated_ledger_worker','exec')
P=load('g291_policy',W/'policy.py');G=load('g291_global',R/'work/grid91_mechanism/policies.py')
class Missing(ValueError):pass
def best(rows):return max([r for r in rows if r['strict_admissible']],key=G.key,default=None)
N=types.SimpleNamespace(G=G,best=best,reward=lambda r:np.float32(r['reward']).item(),observe=lambda *a:np.zeros(41,'f'),InputUnavailable=Missing)
class Provider:
 def __init__(self,fail=False):self.fail=fail
 def context(self,*a):
  if self.fail:raise Missing('synthetic_unknown_plan')
  return {}
class Stub:
 def __init__(self,native,gate):self.actions=list(native);self.gate=gate;self.native_calls=self.gate_calls=0
 def native(self,*a):
  cont=self.actions[min(self.native_calls,len(self.actions)-1)];self.native_calls+=1
  return dict(continue_=cont,raw83=None,log={'native_q':[0.,1.] if cont else [1.,0.]})
 def extension(self,*a):self.gate_calls+=1;return dict(continue_=self.gate,log={'parent_CONT':self.gate})
cases=[]
for name,acts,gate,fail,expected,local in [('no_local_bypass',[False],True,False,256,False),('stop_no_extension',[False],False,False,320,True),('stop_one_extension',[False],True,False,384,True),('native_then_one',[True,False],True,False,448,True),('native_exhaust',[True],False,False,600,True),('input_fallback',[False],False,True,600,True)]:
 for arm in ['PARENT208','BOUNDED261']:
  seen=[]
  def reveal(i,phase):
   assert i not in seen;seen.append(i)
   return dict(pool_id=i,reward=2. if local and i==128 else 1.,rho=.5,strict_admissible=True,done=False,illegal=False,ambiguous=False,exceptions=False)
  model=Stub(acts,gate);out=P.CompletionPolicy(arm,model,N).run(list(range(600)),reveal,[0]*600,{i:i for i in range(600)},limit=.9,provider=Provider(fail))
  assert len(seen)==expected and out['chosen_pool_id']==(128 if local else 0) and len(seen)==len(set(seen)) and model.gate_calls<=1,(name,arm,len(seen),expected)
  if fail:assert model.native_calls==model.gate_calls==0 and out['engineering_fallback']=='synthetic_unknown_plan'
  if not local:assert model.native_calls==model.gate_calls==0
  if local and not fail and name!='native_exhaust':assert model.gate_calls==1
  cases.append(dict(case=name,arm=arm,synthetic_paid_records=len(seen),native_stub_calls=model.native_calls,gate_stub_calls=model.gate_calls,passed=True))
D=rd(O/'design.json');items=D['execution_order'];assert len(items)==192
assert [(r['week'],r['pool'],r['arm']) for r in items[96:]]==list(reversed([(r['week'],r['pool'],r['arm']) for r in items[:96]]))
assert len({(r['repeat'],r['week'],r['method']) for r in items})==192
import guard
rejections=0
for receipt in [{},{'approved_by_user':True},{'approved_by_user':True,'user_instruction':'synthetic','user_message_reference':'synthetic','design_sha256':'x','source_freeze_sha256':'y','runs':1,'wall_cap_s':14400,'worker_cap_s':1200}]:
 try:guard.validate_receipt(receipt,'x','y')
 except PermissionError:rejections+=1
 else:raise AssertionError('invalid authorization accepted')
assert not (O/'physical_authorization.json').exists()
blocked=[]
for script,args in [('worker.py',['--index','0']),('manager.py',[])]:
 result=subprocess.run([sys.executable,'-B',str(W/script),*args],cwd=R,capture_output=True,text=True)
 assert result.returncode!=0 and 'physical authorization receipt missing' in result.stderr
 blocked.append(dict(script=script,exit=result.returncode,blocked_before_model_environment_import=True))
assert not (O/'benchmark_started.json').exists() and not (O/'runs').exists()
payload_api=load('g291_payloads',W/'models.py')
payload_checks=[]
for ref in rd(O/'model_references.json'):
 model=payload_api.FrozenModels(ref['pool']);assert model.mean.shape==model.scale.shape==(41,) and (model.scale>0).all();payload_checks.append(dict(pool=ref['pool'],KEEP=model.gate.KEEP,loaded_without_predictions=True))
review=dict(passed=True,source_published=head,payload_checks=payload_checks,model_payloads_loaded_without_forward=12,synthetic_cases=cases,synthetic_learned_predictions=0,real_NN_predictions=0,real_head_predictions=0,real_physics=0,authorization_rejections=rejections,guarded_entry_checks=blocked,compiled_generated_sources=True,manifest192_unique_reverse=True,resource_budget_not_guaranteed=True,physical_launch_authorized=False,live_environment_smoke_test_not_performed=True)
(O/'preparation_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('GRID291 static/generated-source,12stub-flow cases and authorization guards passed;0realpredictions/physics')
