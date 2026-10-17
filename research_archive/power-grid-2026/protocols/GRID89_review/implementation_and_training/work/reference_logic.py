"""Independent list-based query plan for audit, no production policy imports."""
import struct
def k(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def replay(order,rows,subs,aliases,policy,cfg):
 observations=[];offset=0
 def take(ids):
  for i in ids:assert i in rows;observations.append(rows[i])
 def incumbent():return max((r for r in observations if r['strict_admissible']),key=k,default=None)
 def choice():
  b=incumbent();return None if b is None else b['pool_id']
 if policy in ('FULL','FIXED_512'):
  take(order if policy=='FULL' else order[:512]);return [r['pool_id'] for r in observations],choice(),0
 take(order[:cfg['p']]);offset=len(observations);first=incumbent();first_id=choice();initial_safe=first is not None and first['rho']<cfg['safe'];unchanged=0
 while offset<len(order):
  before=incumbent();before_key=None if before is None else k(before);take(order[offset:offset+cfg['h']]);offset=len(observations);b=incumbent()
  improved=b is not None and (before_key is None or k(b)>before_key);unchanged=0 if improved else unchanged+1
  if not initial_safe and b is not None and b['rho']<cfg['safe']:break
  if b is not None and offset>=cfg['qmin'] and unchanged>=cfg['patience']:break
 b=incumbent();extra=0
 if b is not None and offset<len(order) and (not initial_safe or choice()!=first_id):
  target=subs[b['pool_id']];target_reps={v for a,v in aliases.items() if subs[a]==target};pending=order[offset:];local=[i for i in pending if i in target_reps];extra=len(local)
  if policy=='FRAME_GUARD':take(local)
  elif policy=='GLOBAL_MATCH':take(pending[:extra])
 return [r['pool_id'] for r in observations],choice(),extra
