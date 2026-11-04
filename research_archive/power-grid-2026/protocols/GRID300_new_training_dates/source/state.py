"""Public identity and one intervention; no simulator/model dependency."""
import hashlib,json,math
KEYS=['step','before_hash','action_hash','after_hash','raw_cost','done','complete']
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
class State:
 def __init__(self,item,prefix=None):
  self.item=item;self.target=item.get('target');self.prefix_reference=prefix;self.prefix=[];self.prefix_cost=0.;self.steps=0;self.forecasts=0;self.calls=0;self.call=None;self.offer=None;self.candidates=[];self.event=None
 def begin(self,call):self.call={**call,'call_index':self.calls};self.calls+=1
 def observe(self,rows,order,aliases,subids,limit,targets,pending,budget,batch):
  self.offer=dict(next_ids=list(pending[:64]),batch=batch,visible_signature=digest([{k:r[k] for k in ['pool_id','rho','reward','strict_admissible','done','illegal','ambiguous','exceptions'] if k in r} for r in rows]),station_targets=sorted(targets),original_pending=budget)
 def force(self,raw,native,ordinary):
  if self.event is not None:return ordinary
  assert self.offer and 1<=len(self.offer['next_ids'])<=64
  q=native['native_q'];assert len(q)==2 and all(math.isfinite(v) for v in q) and q[1]<=q[0]
  common={**self.call,**self.offer,'prefix_signature':digest(self.prefix),'prefix_cost':self.prefix_cost,'raw41_hash':digest(raw),'native_q':q,'public_NN_input_hash':digest(native['NN_input'])}
  marker={'common':common,'ordinary261_continue':bool(ordinary['continue_'])};self.candidates.append(marker)
  if self.target is None or self.event is not None:return ordinary
  got=(common['step'],common['call_index'],common['batch']);wanted=(self.target['common']['step'],self.target['common']['call_index'],self.target['common']['batch'])
  if got<wanted:return ordinary
  if marker!=self.target:raise RuntimeError('selected baseline public marker/prefix/base/norm identity differs; no intervention')
  action=not marker['ordinary261_continue'];assert action==(self.item['arm']=='CONT_NEXT')
  self.event={'common':common,'ordinary261_continue':marker['ordinary261_continue'],'arm':self.item['arm'],'forced_continue':action,'delivered':None}
  return {'continue_':action,'log':{**ordinary['log'],'paired_intervention':self.item['arm'],'fixed_newdate_marker':True}}
 def before(self,step,before_hash,action_hash):
  if self.target is None:return
  t=self.target['common']['step']
  if step<t:
   r=self.prefix_reference[step-1]
   if (r['step'],r['before_hash'],r['action_hash'])!=(step,before_hash,action_hash):raise RuntimeError('full joint pre-intervention action/observation mismatch before env.step')
  elif step==t:
   if self.event is None or before_hash!=self.target['common']['state_hash']:raise RuntimeError('missing event or wrong common state before env.step')
  elif self.event is None:raise RuntimeError('selected intervention was not reached')
 def accept(self,row):
  if self.target is not None and row['step']<self.target['common']['step']:
   if any(row[k]!=self.prefix_reference[row['step']-1][k] for k in KEYS):raise RuntimeError('post-step prefix differs; preserve partial')
  if self.event is None:self.prefix.append({k:row[k] for k in KEYS});self.prefix_cost+=row['raw_cost']
  elif self.event['delivered'] is None:
   assert row['step']==self.event['common']['step'];self.event['delivered']={k:row[k] for k in KEYS+['restore']}
  self.steps+=1
 def record(self):return {'status':'BASELINE261_SOURCE' if self.target is None else 'OPPOSITE_EVENT_COLLECTED' if self.event else 'TARGET_NOT_REACHED','event':self.event,'candidates':self.candidates,'steps':self.steps,'forecasts':self.forecasts,'prefix_matched':self.target is not None and self.event is not None,'new_cost_labels':0}
