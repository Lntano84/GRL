"""One optional batch at first native STOP; inject public/model/reveal APIs."""
import time
class CompletionPolicy:
 def __init__(self,arm,models,api):
  if arm not in ['PARENT208','BOUNDED261']:raise ValueError('arm')
  self.arm=arm;self.models=models;self.api=api
 def run(self,order,reveal,subids,aliases,limit=None,provider=None,context=None):
  N=self.api;result=N.G.run(order,reveal,'NO_LOCAL',subids,aliases,limit=limit,context=context)
  rows=result['outcomes'];inc=N.best(rows);initial=None if inc is None else N.reward(inc)
  pending=result['completion_candidate_ids'].copy();budget=len(pending);station=result['local_substation'];targets=set() if not budget else {int(rep) for alias,rep in aliases.items() if int(subids[int(alias)])==station}
  purchased=[];decisions=[];batch=0;feature_s=inference_s=0.;fallback=None;gate_used=False
  while pending:
   ids=pending[:64];raw=None;details={};extension=False
   if fallback:action=True;details={'engineering_fallback':fallback}
   else:
    t=time.perf_counter()
    try:
     raw=N.observe(rows,order,aliases,subids,limit,targets,pending,budget,batch)
     public=provider.context(rows,ids,station,targets)
    except N.InputUnavailable as e:fallback=str(e)
    feature_s+=time.perf_counter()-t
    if fallback:action=True;details={'engineering_fallback':fallback}
    else:
     t=time.perf_counter();native=self.models.native(raw,public);details.update(native['log']);action=bool(native['continue_'])
     if not action:
      assert not gate_used;gate_used=True;extension=True
      gate=self.models.extension(raw,native['raw83'],self.arm);action=bool(gate['continue_']);details.update(gate['log'])
     inference_s+=time.perf_counter()-t
   decisions.append(dict(batch=batch,next_ids=list(ids),action=int(action),optional_extension=extension,paid_queries=len(rows),remaining=len(pending),raw_features=None if raw is None else raw.tolist(),**details))
   if not action:break
   before=N.G.key(inc)
   for i in ids:
    assert i not in {r['pool_id'] for r in rows};r=reveal(i,'local_completion');assert r['pool_id']==i
    rows.append(r);purchased.append(i)
    if r['strict_admissible'] and N.G.key(r)>N.G.key(inc):inc=r
   result['events'].append(dict(queries=len(rows),phase='local_completion',chosen=inc['pool_id'],improved=N.G.key(inc)>before));pending=pending[len(ids):];batch+=1
   if extension:break
  result.update(chosen_pool_id=None if inc is None else inc['pool_id'],public_queries=len(rows),local_queries=len(purchased),extra_selected_ids=purchased,stop_reason=result.get('stop_reason','')+('_extension_once' if gate_used else '_native_or_exhausted'),learned_decisions=decisions,local_feature_s=feature_s,local_inference_s=inference_s,learning_method=self.arm,engineering_fallback=fallback,local_reward_gain=0. if initial is None else N.reward(inc)-initial)
  return result
