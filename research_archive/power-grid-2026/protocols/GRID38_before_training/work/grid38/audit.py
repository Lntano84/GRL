"""Saved target arithmetic and independent semantic-stop reconstruction."""
import hashlib,json,math,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid38'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def f32(v):return struct.unpack('f',struct.pack('f',v))[0]
D=read(OUT/'design.json');R=read(OUT/'review.json')
for p,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
old=read(ROOT/'outputs/grid21/design.json');targets=read(OUT/'targets.json');assert len(targets)==167
positive=0
for row in targets:
    c=read(ROOT/row['source']);valid=[r for r in c['outcomes'] if r['strict_admissible']]
    # Canonical classes, not raw aliases, define both original and new targets.
    meta=read(ROOT/'outputs/grid21/inputs/train'/f"{row['id']}.json")
    import numpy as np
    with np.load(ROOT/'outputs/grid21/inputs/train'/f"{row['id']}.npz") as z:reps=set(z['representatives'].tolist())
    valid=[(i,r) for i,r in zip(c['pool_ids'],c['outcomes']) if i in reps and r['strict_admissible']]
    expected=[0.]*len(row['target'])
    if valid:
        best=min(r['rho'] for i,r in valid);values={i:math.exp(-(r['rho']-best)/old['temperature']) for i,r in valid};total=math.fsum(values.values())
        mx=max(f32(r['reward']) for i,r in valid);optimal={i for i,r in valid if f32(r['reward'])==mx};assert optimal==set(row['reward_optimal'])
        for i,r in valid:expected[i]=(1-D['mixture'])*values[i]/total+(D['mixture']/len(optimal) if i in optimal else 0.)
        assert abs(math.fsum(row['target'])-1)<1e-6;positive+=1
    assert max(abs(a-b) for a,b in zip(expected,row['target']))<1e-7
    assert hashlib.sha256((ROOT/'outputs/grid21/inputs/train'/f"{row['id']}.npz").read_bytes()).hexdigest()==row['original_input_sha256']
subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
def key(row):return f32(row['reward']),-row['pool_id']
def independent(order,truth,aliases):
    seen=set();best=None
    def acquire(ids):
        nonlocal best
        previous=None if best is None else key(best)
        for i in ids:
            assert i not in seen;seen.add(i);r=truth[i]
            if r['strict_admissible'] and (best is None or key(r)>key(best)):best=r
        return best is not None and (previous is None or key(best)>previous)
    acquire(order[:128]);fallback=len(order)>128 and (best is None or best['rho']>=.9)
    if fallback:
        stale=0
        for offset in range(128,len(order),64):
            stale=0 if acquire(order[offset:offset+64]) else stale+1
            if best is not None and best['rho']<.9:break
            if best is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    if best is not None and fallback and len(seen)<len(order):
        sub=subs[best['pool_id']];targets={int(rep) for i,rep in aliases.items() if subs[int(i)]==sub}
        acquire([i for i in order if i in targets and i not in seen])
    return len(seen),None if best is None else best['pool_id']
for row in R['rows']:
    folder=ROOT/'outputs/grid26/runs'/f"r0__{row['week']}__FULL";full=read(folder/row['call']);truth={r['pool_id']:r for r in full['outcomes']}
    own=read(folder.parent/f"r0__{row['week']}__GNN0_128_GUARD"/row['call'])
    assert independent(row['order'],truth,own['aliases'])==(row['queries'],row['choice'])
A={'passed':True,'target_rows_checked':167,'positive_rows':positive,'independent_stop_checks':len(R['rows']),'new_refits':0,'new_forecasts':0,'independent_forward':False,'all_original_sources_unchanged':True}
(OUT/'audit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
lines=['# GRID38 单一训练目标修订','','原soft-rho目标保留一半，另一半均匀分配给实际FP32奖励最优的规范动作类；相同原初始化、数据顺序、优化器和64轮，不改变网络或部署规则。', '',
 '| 模型 | 原查询 | 新查询 | 查询减少 | 原/新同FULL选择 | 最大rho差 | 找回关键动作 |', '|---|---:|---:|---:|---:|---:|---|']
for r in R['summaries']:lines.append(f"| {r['ranker']} | {r['old_queries']} | {r['queries']} | {100*r['query_change']:.3f}% | {r['old_same_full_choice']}/{r['same_full_choice']} | {r['max_rho_regret']:.7f} | {r['known_critical_choice_match']} |")
lines+=['',R['warning'],'','这是一个配置假设检验，目标修改本身不能构成算法新意。闭环质量和完整控制时间尚未测，不能从离线查询量宣布提速。']
(OUT/'GRID38_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FOUR MODELS AND OFFLINE SCREEN AUDITED; original models intact; read all seeds before live trial.\n',encoding='utf-8')
print(json.dumps(A))
