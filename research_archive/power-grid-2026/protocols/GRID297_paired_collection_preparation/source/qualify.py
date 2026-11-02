"""One finite stdlib-only static/stub qualification; no live/model smoke test."""
import ast,types,json,subprocess,sys,importlib.util,hashlib
from pathlib import Path
from guard import R,O,I,A,read,sha,git,validate_receipt
W=Path(__file__).parent;sys.dont_write_bytecode=True
assert not (O/'preparation_review.json').exists()
head=git('rev-parse','HEAD').decode().strip();assert git('ls-remote','lntano-src','refs/heads/main').decode().split()[0]==head
assert git('show',head+':'+(A/'source/source_freeze.json').relative_to(I).as_posix())==(O/'source_freeze.json').read_bytes()
for rel,h in read(O/'source_freeze.json').items():assert sha(R/rel)==h,rel
bindings=read(O/'published_model_bindings.json')
for local,d in bindings.items():assert hashlib.sha256(git('show',head+':'+d['archive_path'])).hexdigest()==d['sha256']==sha(R/local)
for p in W.glob('*.py'):ast.parse(p.read_text(encoding='utf8'))
import source_builder as B
compile(B.build(),'generated_runner','exec');compile(B.instrument_worker((R/'work/grid55/worker.py').read_text(encoding='utf8')),'generated_physical_worker','exec')
def load(name,p):
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
P=load('g297_original_policy_stub',R/'work/grid291_control_preparation/policy.py')
from intervention import EpisodeIntervention,EpisodeModels
class Raw(list):
 def tolist(self):return list(self)
class Missing(ValueError):pass
class G:
 @staticmethod
 def key(r):return (-1.,) if r is None else (r['reward'],-r['pool_id'])
 @staticmethod
 def run(order,reveal,*args,**kwargs):
  rows=[reveal(i,'shortlist') for i in order[:128]]
  return dict(outcomes=rows,completion_candidate_ids=order[128:],local_substation=0,shortlist_choice=0,events=[],chosen_pool_id=0)
def best(rows):return max(rows,key=G.key)
class Payload:
 def __init__(self,native=False,ordinary=True):self.cont=native;self.ordinary=ordinary;self.native_calls=0;self.extension_calls=0
 def native(self,*a):self.native_calls+=1;return dict(continue_=self.cont,raw83=None,log=dict(native_q=[1.,0.] if not self.cont else [0.,1.],NN_input={'synthetic_only':[0.]}))
 def extension(self,*a):self.extension_calls+=1;return dict(continue_=self.ordinary,log={'parent_CONT':self.ordinary,'residual_KEEP':False})
def run_case(arm,native=False,ordinary=True,expected=None,state=None):
 state=state or EpisodeIntervention(arm,expected);payload=Payload(native,ordinary);seen=[]
 def observe(*a):state.observe(*a);return Raw([0.]*41)
 api=types.SimpleNamespace(G=G,best=best,reward=lambda r:r['reward'],observe=observe,InputUnavailable=Missing)
 class Provider:
  def context(self,*a):return {}
 def reveal(i,phase):
  assert i not in seen;seen.append(i);return dict(pool_id=i,rho=.5,reward=float(i>127),strict_admissible=True,done=False,illegal=False,ambiguous=False,exceptions=[])
 state.begin(dict(step=1,state_hash='synthetic_state',base_action_hash='base',order_hash='order',aliases_hash='aliases',rho_limit=.9))
 out=P.CompletionPolicy('BOUNDED261',EpisodeModels(payload,state),api).run(list(range(300)),reveal,[0]*300,{i:i for i in range(300)},limit=.9,provider=Provider())
 return state,payload,seen,out
cases=[]
for arm,paid in [('STOP_NEXT',128),('CONT_NEXT',192)]:
 s,p,seen,out=run_case(arm);assert len(seen)==paid and s.event is not None and p.extension_calls==1
 # Later nativeSTOPs in the same episode revert to261, no second force.
 s2,p2,seen2,out2=run_case(arm,ordinary=True,state=s);assert len(seen2)==192 and s2.event is s.event
 cases.append(dict(case=arm,paid_stub_records=paid,second_call_stub_records=len(seen2),passed=True))
s,p,seen,out=run_case('STOP_NEXT',native=True);assert len(seen)==300 and s.event is None;cases.append(dict(case='nativeCONT_exhaust_no_trigger',passed=True))
s,p,seen,out=run_case('STOP_NEXT');common=s.event['common']
s,p,seen,out=run_case('CONT_NEXT',expected=common);assert s.event['common']==common;cases.append(dict(case='paired_public_prefix_equal',passed=True))
for name,expected in [('mismatch',{'wrong':1}),('first_arm_no_event',{'status':'NO_ELIGIBLE_EVENT'})]:
 try:run_case('CONT_NEXT',expected=expected)
 except RuntimeError:cases.append(dict(case=name,rejected_before_optional_reveal=True))
 else:raise AssertionError(name)
synthetic=dict(approved_by_user=True,user_instruction='SYNTHETIC_NOT_AUTHORIZATION',user_message_reference='synthetic',scope_id='GRID297_PAIRED_SUFFIX_COLLECTION',runs=288,pairs=144,wall_cap_s=28800,worker_cap_s=1200,physical_steps_cap=580896,forecasts_cap=9000000,design_sha256='d',source_freeze_sha256='s')
validate_receipt(synthetic,'d','s');rejections=[]
for key,value in [('approved_by_user',False),('runs',192),('wall_cap_s',14400),('source_freeze_sha256','wrong'),('forecasts_cap',9000001),('scope_id','GRID291')]:
 bad={**synthetic,key:value}
 try:validate_receipt(bad,'d','s')
 except PermissionError:rejections.append(key)
 else:raise AssertionError('invalid authorization')
assert not (O/'physical_authorization.json').exists();blocked=[]
for script,args in [('manager.py',[]),('worker.py',['--index','0'])]:
 result=subprocess.run([sys.executable,'-B','-X','utf8',str(W/script),*args],capture_output=True,cwd=R)
 assert result.returncode and b'new physical authorization receipt missing' in result.stderr
 blocked.append(dict(script=script,rejected_before_model_simulator_import=True))
D=read(O/'design.json');assert len(D['execution_order'])==288 and len(D['pairs'])==144
for pair in D['pairs']:
 a,b=D['execution_order'][pair['pair_index']*2:pair['pair_index']*2+2];assert a['pool']==b['pool']==pair['pool'] and a['week']==b['week']==pair['week'] and {a['arm'],b['arm']}=={'STOP_NEXT','CONT_NEXT'}
assert not any((O/n).exists() for n in ['runs','collection_started.json','claims'])
assert not any(n in sys.modules for n in ['torch','grid2op','common'])
review=dict(passed=True,source_published=head,pools=12,pairs=144,episodes=288,exact_model_Git_blobs=len(bindings),synthetic_cases=cases,authorization_rejections=rejections,guarded_entry_checks=blocked,generated_sources_compiled=True,new_model_loads=0,new_NN_predictions=0,new_head_predictions=0,new_physics=0,new_fits=0,new_labels=0,real_environment_smoke_test=False,resource_passed=False,scope_authorized=False,limitation='Static/stub preparation passed, not a guarantee of simulator/schema compatibility, bit-exact cross-process prefixes or8h completion. Any first live attempt consumes the separately approved window; no automatic retry.')
(O/'preparation_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print('GRID297_PREPARATION_PASSED_0_PREDICTIONS_0_PHYSICS')
