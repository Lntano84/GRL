"""Bounded reviewer checks; the default is checked against frozen GRID70."""
from pathlib import Path
import functools,importlib.util,struct
ROOT=Path(__file__).resolve().parents[2]
@functools.lru_cache(maxsize=8)
def module(n,p):
 s=importlib.util.spec_from_file_location(n,ROOT/p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
REF=module('review_guard','work/grid70/policies.py');key=REF.key
DEFAULT=dict(p=128,h=64,qmin=256,patience=2,safe=.9)
def run(order,observe,policy,subids,aliases,params=None,context=None,limit=None,model_path=None):
 cfg=dict(DEFAULT);cfg.update(params or {});safe=cfg['safe'];rows=[];best=None;events=[];seen=set()
 assert len(order)==len(set(order))
 if policy=='META':
  M=module('review_meta','work/grid77/metacontrol.py')
  return M.run(order,observe,0,subids,aliases,limit,context,model=load_model(model_path))
 def purchase(ids,phase):
  nonlocal best
  old=None if best is None else key(best)
  for i in ids:
   assert i not in seen;seen.add(i)
   r=observe(i,phase);assert r['pool_id']==i;rows.append(r)
   if r['strict_admissible'] and (best is None or key(r)>key(best)):best=r
  new=None if best is None else key(best)
  improved=new is not None and (old is None or new>old)
  events.append(dict(queries=len(rows),phase=phase,chosen=None if best is None else best['pool_id'],improved=improved))
  return improved
 if policy in ('FULL','FIXED_512'):
  purchase(order if policy=='FULL' else order[:512],'full_search' if policy=='FULL' else 'fixed_prefix')
  prefix=next((e['chosen'] for e in events),None);prefix_safe=False;reason='full_exhaustion' if policy=='FULL' else 'fixed_budget_reached';local=[];candidate_ids=[];sub=None;chosen=None if best is None else best['pool_id'];budget=0
 else:
  assert policy in ('FRAME_GUARD','GLOBAL_MATCH','NO_LOCAL')
  n=min(cfg['p'],len(order));purchase(order[:n],'shortlist');prefix=None if best is None else best['pool_id'];prefix_safe=best is not None and best['rho']<safe
  reason='full_exhaustion';stale=0
  for offset in range(n,len(order),cfg['h']):
   improved=purchase(order[offset:offset+cfg['h']],'verification' if prefix_safe else 'fallback');stale=0 if improved else stale+1
   if not prefix_safe and best is not None and best['rho']<safe:reason='verified_below_author_safe';break
   if best is not None and len(rows)>=cfg['qmin'] and stale>=cfg['patience'] and len(rows)<len(order):reason='guard_verified_plateau' if prefix_safe else 'patience64_experimental_stop';break
  chosen=None if best is None else best['pool_id'];local=[];candidate_ids=[];sub=None;budget=0
  if chosen is not None and len(rows)<len(order) and (not prefix_safe or chosen!=prefix):
   sub=int(subids[chosen]);targets={int(v) for a,v in aliases.items() if int(subids[int(a)])==sub}
   seen={r['pool_id'] for r in rows};remaining=[i for i in order if i not in seen];local=[i for i in remaining if i in targets];candidate_ids=local.copy();budget=len(local)
   if policy=='GLOBAL_MATCH':local=remaining[:budget]
   if policy!='NO_LOCAL':purchase(local,'global_budget_extension' if policy=='GLOBAL_MATCH' else 'local_completion');reason+='_global_budget_complete' if policy=='GLOBAL_MATCH' else '_local_complete'
   else:local=[]
  elif prefix_safe and len(rows)<len(order):reason+='_unchanged_incumbent'
 return dict(outcomes=rows,chosen_pool_id=None if best is None else best['pool_id'],shortlist_choice=prefix,fallback=policy!='FULL' and len(rows)>min(cfg['p'],len(order)),stop_reason=reason,events=events,public_queries=len(rows),local_substation=sub,local_queries=len(local) if policy=='FRAME_GUARD' else 0,completion_budget=budget,global_extra_queries=len(local) if policy=='GLOBAL_MATCH' else 0,pre_extra_choice=chosen,completion_candidate_ids=candidate_ids,extra_selected_ids=local,verification_guard=prefix_safe,prefix_safe_at_entry=prefix_safe,parameters=cfg)
@functools.lru_cache(maxsize=12)
def load_model(path):
 import torch
 M=module('review_model','work/grid77/metacontrol.py');m=M.QModel();m.load_state_dict(torch.load(ROOT/path,map_location='cpu',weights_only=True));m.eval();return m
