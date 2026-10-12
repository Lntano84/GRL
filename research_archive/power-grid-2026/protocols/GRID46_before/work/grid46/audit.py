"""Independent old-feedback sequence reconstruction and hidden-feedback perturbation."""
import hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid46'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def key(r):return struct.unpack('f',struct.pack('f',r['reward']))[0],-r['pool_id']
subs=read(ROOT/'outputs/grid42b/library.json')['subids']
orders={(r['week'],r['call'],r['ranker']):r['order'] for r in read(ROOT/'outputs/grid33/review.json')['rows']+read(ROOT/'outputs/grid40b/offline_review.json')['rows']}
def independent(order,truth,aliases):
    seen=set();sequence=[];best=None
    def acquire(ids):
        nonlocal best
        before=None if best is None else key(best)
        for i in ids:
            assert i not in seen;seen.add(i);sequence.append(i);r=truth[i]
            if r['strict_admissible'] and (best is None or key(r)>key(best)):best=r
        return best is not None and (before is None or key(best)>before)
    acquire(order[:128]);fallback=len(order)>128 and (best is None or best['rho']>=.9)
    if fallback:
        stale=0
        for at in range(128,len(order),64):
            stale=0 if acquire(order[at:at+64]) else stale+1
            if best is not None and best['rho']<.9:break
            if best is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    if fallback and best is not None and len(seen)<len(order):
        sub=subs[best['pool_id']];targets={int(rep) for i,rep in aliases.items() if subs[int(i)]==sub}
        acquire([i for i in order if i not in seen and i in targets])
    if best is not None and best['rho']>=.9 and len(seen)<len(order):acquire([i for i in order if i not in seen])
    return sequence,None if best is None else best['pool_id']
R=read(OUT/'review.json');assert R['passed'] and len(R['rows'])==130
for p,h in R['source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
for r in R['rows']:
    f=read(ROOT/'outputs/grid26/runs'/f"r0__{r['week']}__FULL"/r['call']);t={v['pool_id']:v for v in f['outcomes']};o=orders[r['week'],r['call'],r['ranker']]
    sequence,choice=independent(o,t,f['aliases']);assert sequence==r['sequence'] and len(sequence)==r['queries'] and choice==r['choice']
    changed={i:({**v,'rho':.1,'reward':1000.,'strict_admissible':True} if i not in sequence else dict(v)) for i,v in t.items()}
    assert independent(o,changed,f['aliases'])==(sequence,choice)
A={'passed':True,'independent_sequences':130,'unrevealed_feedback_checks':130,'all_critical1733':all(s['critical1733'] for s in R['summaries']),
   'warning':'Cached old development responses only;not a closed-loop quality guarantee,new physics run,or new safety certificate.'}
(OUT/'audit.json').write_text(json.dumps(A,indent=2),encoding='utf-8')
report=['# GRID46 既有作者阈值的共同回补诊断','', '| 排序器 | 原查询 | 保护后查询 | 同FULL选择 | 找回开发1733 |','|---|---:|---:|---:|---:|']
for s in R['summaries']:report.append(f"| {s['ranker']} | {s['old_queries']} | {s['queries']} | {s['same_full_choice']}/26 | {s['critical1733']} |")
report+=['','130条独立序列复算和130次未观测反馈扰动通过。新增训练/物理预测均0；原模型与停止规则保留。',A['warning'],
  '该回补重新采用作者原0.9缓冲条件，仅是保守的质量—时间端点，不包装成新阈值、安全理论或图独有贡献。下一步只在旧开发周测试实际速度和质量；确认八家族不参与训练。']
(OUT/'GRID46_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8');print(json.dumps(A))
