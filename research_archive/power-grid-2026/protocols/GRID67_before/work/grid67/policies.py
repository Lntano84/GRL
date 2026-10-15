"""Single-axis FRAME sensitivity. Defaults reproduce the frozen FRAME exactly."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sp=importlib.util.spec_from_file_location('grid67_reference',ROOT/'work/grid63/policies.py')
old=importlib.util.module_from_spec(sp);sp.loader.exec_module(old)
key=old.key
POLICIES=('FULL','FRAME','GLOBAL_MATCH','FRAME_P64','FRAME_P256','FRAME_S1','FRAME_S3')
def run(order,observe,policy,subids,aliases,safe=.9,limit=None):
 if policy in old.POLICIES:return old.run(order,observe,policy,subids,aliases,safe)
 assert policy in POLICIES and len(order)==len(set(order))
 initial={'FRAME_P64':64,'FRAME_P256':256}.get(policy,128)
 patience={'FRAME_S1':1,'FRAME_S3':3}.get(policy,2)
 rows=[];events=[];best=None
 def purchase(ids,phase):
  nonlocal best
  before=None if best is None else key(best)
  for i in ids:
   row=observe(i,phase);assert row['pool_id']==i;rows.append(row)
   if row['strict_admissible'] and (best is None or key(row)>key(best)):best=row
  after=None if best is None else key(best)
  improved=after is not None and (before is None or after>before)
  events.append({'queries':len(rows),'phase':phase,'chosen':None if best is None else best['pool_id'],'improved':improved})
  return improved
 n=min(initial,len(order));purchase(order[:n],'shortlist');prefix=None if best is None else best['pool_id']
 fallback=n<len(order) and (best is None or best['rho']>=safe)
 reason='initial_safe' if not fallback else 'full_exhaustion'
 if fallback:
  stale=0
  for offset in range(n,len(order),64):
   improved=purchase(order[offset:offset+64],'fallback');stale=0 if improved else stale+1
   if best is not None and best['rho']<safe:reason='verified_below_author_safe';break
   if best is not None and len(rows)>=256 and stale>=patience and len(rows)<len(order):reason='patience64_experimental_stop';break
 chosen=None if best is None else best['pool_id'];local=[];sub=None
 if chosen is not None and fallback and len(rows)<len(order):
  sub=int(subids[chosen]);targets={int(rep) for a,rep in aliases.items() if int(subids[int(a)])==sub}
  seen={r['pool_id'] for r in rows};local=[i for i in order if i not in seen and i in targets]
  purchase(local,'local_completion');reason+='_local_complete'
 return {'outcomes':rows,'chosen_pool_id':None if best is None else best['pool_id'],'shortlist_choice':prefix,
 'fallback':fallback,'stop_reason':reason,'events':events,'public_queries':len(rows),'local_substation':sub,'local_queries':len(local),
 'completion_budget':len(local),'global_extra_queries':0,'pre_extra_choice':chosen,'completion_candidate_ids':local,'extra_selected_ids':local,
 'initial_prefix':initial,'stale_batches':patience}
