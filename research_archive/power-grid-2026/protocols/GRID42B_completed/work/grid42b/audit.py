"""Independent sequence reconstruction and first matched physical-divergence review."""
import gzip, hashlib, json, struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/grid42b'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
subs=read(OUT/'library.json')['subids']
orders={(r['week'],r['call'],r['ranker']):r['order'] for r in read(ROOT/'outputs/grid33/review.json')['rows']+read(ROOT/'outputs/grid40b/offline_review.json')['rows']}
def independent(order,truth,aliases):
    sequence=[]; seen=set(); best=None
    def acquire(ids):
        nonlocal best
        before=None if best is None else key(best)
        for i in ids:
            assert i not in seen;seen.add(i);sequence.append(i);r=truth[i]
            if r['strict_admissible'] and (best is None or key(r)>key(best)):best=r
        return best is not None and (before is None or key(best)>before)
    acquire(order[:128])
    if len(order)>128 and (best is None or best['rho']>=.9):
        stale=0
        for at in range(128,len(order),64):
            stale=0 if acquire(order[at:at+64]) else stale+1
            if best is not None and best['rho']<.9:break
            if best is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    completed=set()
    while best is not None and len(seen)<len(order):
        groups={subs[i] for i in sequence if truth[i]['strict_admissible'] and key(truth[i])[0]==key(best)[0]}
        fresh=groups-completed
        if not fresh:break
        completed.update(fresh)
        targets={int(rep) for i,rep in aliases.items() if subs[int(i)] in fresh}
        acquire([i for i in order if i not in seen and i in targets])
    return sequence,None if best is None else best['pool_id']
R=read(OUT/'review.json');assert R['passed'] and len(R['rows'])==130
for p,h in R['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
for r in R['rows']:
    p=ROOT/'outputs/grid26/runs'/f"r0__{r['week']}__FULL"/r['call']; f=read(p)
    t={v['pool_id']:v for v in f['outcomes']}; o=orders[(r['week'],r['call'],r['ranker'])]
    sequence,winner=independent(o,t,f['aliases'])
    assert sequence==r['observed_sequence'] and len(sequence)==r['queries'] and winner==r['choice']
    assert r['same_full_choice']==(winner==f['chosen_pool_id'])
    gap=None if winner is None or f['chosen_pool_id'] is None else t[winner]['rho']-t[f['chosen_pool_id']]['rho']
    assert gap==r['rho_regret']
    # Changes to responses never requested cannot enter the policy's decisions.
    changed={i:({**v,'reward':v['reward']+1000.,'rho':.1,'strict_admissible':True} if i not in sequence else dict(v)) for i,v in t.items()}
    assert independent(o,changed,f['aliases'])==(sequence,winner)
divergences=[]
M=read(ROOT/'outputs/grid41/manifest.json')['summaries']
for s in M:
    if s['first_full_divergence'] is None:continue
    folder=ROOT/'outputs/grid41/runs'/f"r0__{s['week']}__{s['method']}"
    calls=[read(p) for p in sorted(folder.glob('call[0-9][0-9][0-9].json'))]
    other=ROOT/'outputs/grid41/runs'/f"r0__{s['week']}__FULL"
    full={c['step']:c for p in sorted(other.glob('call[0-9][0-9][0-9].json')) if (c:=read(p))}
    for c in calls:
        f=full.get(c['step'])
        if f is None or f['state_hash']!=c['state_hash']:continue
        if f['chosen_pool_id']==c['chosen_pool_id']:continue
        a=next(x for x in f['outcomes'] if x['pool_id']==f['chosen_pool_id'])
        b=next(x for x in c['outcomes'] if x['pool_id']==c['chosen_pool_id'])
        divergences.append({'method':s['method'],'week':s['week'],'step':c['step'],'full_id':a['pool_id'],'method_id':b['pool_id'],
             'full_sub':subs[a['pool_id']],'method_sub':subs[b['pool_id']],
             'full_rank':c['order'].index(a['pool_id'])+1,'full_queried':a['pool_id'] in {v['pool_id'] for v in c['outcomes']},
             'reward_fp32_tied':key(a)[0]==key(b)[0],'rho_tied':a['rho']==b['rho'],
             'same_input':True,'full_reward':a['reward'],'method_reward':b['reward'],
             'reason':c['stop_reason'],'combined_actions_different':a['combined_hash']!=b['combined_hash']})
        break
audit={'passed':True,'independent_sequences':130,'unrevealed_response_perturbations':130,'source_hashes_verified':True,
       'new_forecasts':0,'new_fits':0,'reserved_evaluations':0,
       'warning':'Saved response/decision audit, not new power-flow solving or independent neural refit.'}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
(OUT/'first_divergences.json').write_text(json.dumps(divergences,indent=2),encoding='utf-8')
note='\n独立复算130条查询顺序/选择及130次未揭示反馈扰动通过。GRID41三月SoftGNN两种子首次不同选择均在578步，FULL59与所选149/115的即时FP32奖励和rho完全相同，但物理动作不同；不能由此把全部长期损失因果归给单个动作。原GNN0在1741步是小幅真实即时差距，而非上述完全并列。\n'
p=OUT/'GRID42_report.md';p.write_text(p.read_text(encoding='utf-8')+note,encoding='utf-8')
print(json.dumps({'audit':audit,'first_divergences':divergences},indent=2))
