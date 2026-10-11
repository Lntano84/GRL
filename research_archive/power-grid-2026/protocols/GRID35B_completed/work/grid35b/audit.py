"""Independent BFS levels and separate saved-feedback stopping reconstruction."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid35b'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
R=read(OUT/'review.json');assert R['passed_engineering']
for p,h in {**read(OUT/'code_freeze.json'),**R['input_hashes']}.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
def score(item):return struct.unpack('f',struct.pack('f',item['reward']))[0],-item['pool_id']
def stopped(order,truth,aliases):
    seen=set();best=None
    def pay(ids):
        nonlocal best
        before=None if best is None else score(best)
        for i in ids:
            assert i not in seen;seen.add(i)
            if truth[i]['strict_admissible'] and (best is None or score(truth[i])>score(best)):best=truth[i]
        return best is not None and (before is None or score(best)>before)
    pay(order[:128]);fallback=len(order)>128 and (best is None or best['rho']>=.9)
    if fallback:
        stale=0
        for start in range(128,len(order),64):
            stale=0 if pay(order[start:start+64]) else stale+1
            if best is not None and best['rho']<.9:break
            if best is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    if best is not None and fallback and len(seen)<len(order):
        station=subs[best['pool_id']];target={int(r) for i,r in aliases.items() if subs[int(i)]==station}
        pay([i for i in order if i in target and i not in seen])
    return len(seen),None if best is None else best['pool_id']
for row in R['rows']:
    fullfolder=ROOT/'outputs/grid26/runs'/f"r0__{row['week']}__FULL";ownfolder=fullfolder.parent/f"r0__{row['week']}__GNN0_128_GUARD"
    own=read(ownfolder/row['call']);aliases={int(i):int(r) for i,r in own['aliases'].items()}
    with np.load(ownfolder/(Path(row['call']).stem+'_inputs.npz')) as z:
        n=len(z['x'])//2;edge=z['edges'].copy();rho=z['edge_attr'][:edge.shape[1]//2,0].tolist()
    pairs=[(int(edge[0,j])%n,int(edge[1,j])%n) for j in range(len(rho))]
    roots={s for j in sorted(range(len(rho)),key=lambda j:(-rho[j],j))[:3] for s in pairs[j]}
    adj=[set() for _ in range(n)];pressure=[0.]*n
    for (u,v),p in zip(pairs,rho):adj[u].add(v);adj[v].add(u);pressure[u]=max(pressure[u],p);pressure[v]=max(pressure[v],p)
    distances=[n+1]*n;front=roots;known=set();level=0
    while front:
        for s in front:distances[s]=level
        known.update(front);front={t for s in front for t in adj[s]}-known;level+=1
    raw=sorted(own['eligible'],key=lambda i:(distances[subs[i]],-pressure[subs[i]],i));dedup=list(dict.fromkeys(aliases[i] for i in raw))
    assert dedup==row['order']
    truth={x['pool_id']:x for x in read(fullfolder/row['call'])['outcomes']}
    assert stopped(dedup,truth,aliases)==(row['queries'],row['choice'])
A={'passed':True,'independent_public_order_checks':26,'independent_stopping_checks':26,'new_fits':0,'new_forecasts':0}
(OUT/'audit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
s=R['summary'];lines=['# GRID35B 原有非学习排序部分','','从公开输入独立重建26项BFS顺序和停止决定，无新拟合/AC预测。', '',
    f"PHYSICS_TAIL：查询{s['queries']}、同FULL选择{s['same_full_choice']}/26、最大rho差{s['max_rho_regret']:.7f}、1733关键动作找回{s['critical_choice_match']}。",'',R['warning']]
(OUT/'GRID35B_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED AND AUDITED:existing public tail26states,no livequality result.\n',encoding='utf-8')
print(json.dumps(A))
