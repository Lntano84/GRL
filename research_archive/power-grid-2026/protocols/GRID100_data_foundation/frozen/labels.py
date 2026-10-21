"""Local search-gain labels from fully observed teacher calls; never fit a model."""
import struct,hashlib,json
FIELDS=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
def input_at(call,subids,n):
    """Accept only paid prefix rows for features; no suffix reward accepted."""
    order=call['order'];assert 0<n<len(order)
    visible=call['visible'];assert [r['pool_id'] for r in visible]==order[:n]
    best=max((r for r in visible if r['strict_admissible']),key=key,default=None)
    if best is None:return None
    aliases={int(k):v for k,v in call['aliases'].items()};station=subids[best['pool_id']];targets={int(v) for a,v in aliases.items() if subids[a]==station};seen=set(order[:n]);pending=[i for i in order[n:] if i in targets]
    if not pending:return None
    return dict(visible=[{k:r[k] for k in FIELDS} for r in visible],public_order=order,public_aliases=aliases,public_subids=subids,visible_count=n,incumbent_id=best['pool_id'],remaining_same_station_ids=pending,
        rho_limit=call['rho_limit'],candidate_count=len(order),known_completion_queries=len(pending))
def collect(call,subids,identity):
    order=call['order'];truth={r['pool_id']:r for r in call['outcomes']}
    assert len(order)==len(set(order)) and set(order)==set(truth),'Full teacher feedback required'
    assert [r['pool_id'] for r in call['outcomes']]==order
    # Fixed search checkpoints, not outcome-selected examples. Adjacent rows stay grouped.
    offsets=list(range(min(128,len(order)),len(order),64));rows=[];excluded=[]
    for n in offsets:
        visible=[truth[i] for i in order[:n]];public=dict(order=order,aliases=call['aliases'],rho_limit=call['rho_limit'],visible=visible)
        x=input_at(public,subids,n)
        if x is None:excluded.append(dict(visible_count=n,reason='no_admissible_incumbent_or_no_unqueried_station_candidate'));continue
        best=truth[x['incumbent_id']];local=[truth[i] for i in x['remaining_same_station_ids']];eligible=[r for r in local if r['strict_admissible']];after=max([best]+eligible,key=key)
        gain=key(after)[0]-key(best)[0];assert gain>=0
        sid=hashlib.sha256(f"{identity}|{n}".encode()).hexdigest()[:24]
        label=dict(sample_id=sid,local_reward_gain=gain,local_rho_reduction=float(best['rho'])-float(after['rho']),choice_changes=after['pool_id']!=best['pool_id'],before_id=best['pool_id'],after_id=after['pool_id'],known_completion_queries=len(local),
            recorded_teacher_forecast_seconds_sum=sum(r.get('forecast_s',0.) for r in local),scope='Immediate best admissible reward improvement at a fixed saved state; not long-term cost, safety or whole-path value.')
        # Suffix deletion/poison cannot alter prefix input.
        changed={**public,'visible':[dict(r) for r in visible]};assert input_at(changed,subids,n)==x
        rows.append(dict(sample_id=sid,input=x,label=label,visible_count=n))
    return rows,excluded
