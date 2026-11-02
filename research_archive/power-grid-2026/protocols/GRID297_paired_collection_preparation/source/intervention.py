"""One public, episode-level query intervention. No model/environment imports."""
import hashlib,json
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
class EpisodeIntervention:
 def __init__(self,arm,expected=None):
  if arm not in ['STOP_NEXT','CONT_NEXT']:raise ValueError('arm')
  self.arm=arm;self.expected=expected;self.event=None;self.call=None;self.offer=None;self.calls=0;self.prefix=[];self.prefix_cost=0.;self.forecasts=0;self.steps=0
 def begin(self,call):self.call={**call,'call_index':self.calls};self.calls+=1
 def observe(self,rows,order,aliases,subids,limit,targets,pending,budget,batch):
  self.offer=dict(next_ids=list(pending[:64]),batch=batch,visible_signature=digest([{k:r[k] for k in ['pool_id','rho','reward','strict_admissible','done','illegal','ambiguous','exceptions'] if k in r} for r in rows]),station_targets=sorted(targets),original_pending=budget)
 def force(self,raw,native_log,ordinary):
  if self.event is not None:return ordinary
  if not self.offer or not self.offer['next_ids']:raise ValueError('nativeSTOP without next batch')
  common={**self.call,**self.offer,'prefix_signature':digest(self.prefix),'prefix_cost':self.prefix_cost,'raw41':raw,'native_q':native_log['native_q'],'public_NN_input_hash':digest(native_log['NN_input'])}
  if self.expected is not None and common!=self.expected:raise RuntimeError('paired common prefix or public trigger differs; no intervention performed')
  self.event={'common':common,'arm':self.arm,'ordinary261':ordinary,'forced_continue':self.arm=='CONT_NEXT','delivered':None}
  return dict(continue_=self.arm=='CONT_NEXT',log={**ordinary['log'],'paired_intervention':self.arm,'first_episode_nativeSTOP':True})
 def accept(self,row):
  if self.event is None:
   self.prefix.append({k:row[k] for k in ['step','before_hash','action_hash','after_hash','raw_cost','done','complete']});self.prefix_cost+=row['raw_cost']
  elif self.event['delivered'] is None:
   if row['step']!=self.event['common']['step']:raise RuntimeError('trigger/delivered step differs')
   self.event['delivered']={k:row[k] for k in ['step','before_hash','action_hash','after_hash','restore','raw_cost','done','complete']}
  self.steps+=1
 def record(self):return dict(status='EVENT_COLLECTED' if self.event else 'NO_ELIGIBLE_EVENT',event=self.event,complete_no_event_signature=digest(self.prefix) if self.event is None else None,steps=self.steps,forecasts=self.forecasts)
class EpisodeModels:
 def __init__(self,payload,state):self.payload=payload;self.state=state;self.latest_native=None
 def native(self,*a):self.latest_native=self.payload.native(*a);return self.latest_native
 def extension(self,raw,raw83,arm):
  ordinary=self.payload.extension(raw,raw83,arm)
  return self.state.force(raw.tolist() if hasattr(raw,'tolist') else raw,self.latest_native['log'],ordinary)
