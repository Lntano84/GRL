"""Replay old complete development feedback; no environment, inference or new solve."""
import csv,hashlib,json
from pathlib import Path
from safe_completion import safe_complete,hand_checks,key
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid46'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
assert not (OUT/'review.json').exists() and hand_checks()==4
subs=read(ROOT/'outputs/grid42b/library.json')['subids'];rankers={'GNN0','GNN1','PPO','SOFTGNN0','SOFTGNN1'}
sources=[r for r in read(ROOT/'outputs/grid33/review.json')['rows']+read(ROOT/'outputs/grid40b/offline_review.json')['rows'] if r['ranker'] in rankers]
assert len(sources)==130;rows=[];hashes={}
for r in sources:
    p=ROOT/'outputs/grid26/runs'/f"r0__{r['week']}__FULL"/r['call'];hashes[p.relative_to(ROOT).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    full=read(p);truth={x['pool_id']:x for x in full['outcomes']};aliases={int(k):int(v) for k,v in full['aliases'].items()};sequence=[]
    def observe(i,phase):
        assert i not in sequence;sequence.append(i);return {**truth[i],'phase':phase}
    z=safe_complete(r['order'],observe,'PATIENCE64',subs,aliases)
    assert r['queries']<=z['public_queries']<=len(full['representatives'])
    valid=[x for x in z['outcomes'] if x['strict_admissible']]
    assert z['chosen_pool_id']==(max(valid,key=key)['pool_id'] if valid else None)
    rows.append({'week':r['week'],'call':r['call'],'step':full['step'],'ranker':r['ranker'],
       'old_queries':r['queries'],'queries':z['public_queries'],'added':z['author_completion_queries'],
       'old_choice':r['choice'],'choice':z['chosen_pool_id'],'full_choice':full['chosen_pool_id'],
       'same_full_choice':z['chosen_pool_id']==full['chosen_pool_id'],'sequence':sequence,'stop_reason':z['stop_reason']})
summary=[]
for name in sorted(rankers):
    g=[r for r in rows if r['ranker']==name]
    summary.append({'ranker':name,'old_queries':sum(r['old_queries'] for r in g),'queries':sum(r['queries'] for r in g),
       'same_full_choice':sum(r['same_full_choice'] for r in g),
       'critical1733':next(r for r in g if r['week']=='2035-03-05_0' and r['step']==1733)['same_full_choice']})
(OUT/'review.json').write_text(json.dumps({'passed':True,'hand_checks':4,'summaries':summary,'rows':rows,'source_hashes':hashes,'new_fits':0,'new_forecasts':0},indent=2),encoding='utf-8')
with (OUT/'per_state.csv').open('w',encoding='utf-8',newline='') as f:
    fields=[k for k in rows[0] if k!='sequence'];w=csv.DictWriter(f,fields);w.writeheader();w.writerows({k:r[k] for k in fields} for r in rows)
print(json.dumps(summary,indent=2))
