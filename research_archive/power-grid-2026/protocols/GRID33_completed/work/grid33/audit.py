"""Independent saved-order replay, no learned inference/refit or simulator invocation."""
import hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid33'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
R=read(OUT/'review.json');D=read(OUT/'design.json');assert R['passed']
for file,h in R['public_input_hashes'].items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
for file,h in read(OUT/'code_freeze.json').items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
def score(row):return struct.unpack('f',struct.pack('f',row['reward']))[0],-row['pool_id']
def independent(order,truth,aliases):
    seen=set();best=None
    def acquire(ids):
        nonlocal best
        previous=None if best is None else score(best)
        for i in ids:
            assert i not in seen;seen.add(i);item=truth[i]
            if item['strict_admissible'] and (best is None or score(item)>score(best)):best=item
        return best is not None and (previous is None or score(best)>previous)
    acquire(order[:128]);fallback=len(order)>128 and (best is None or best['rho']>=.9)
    if fallback:
        stale=0
        for offset in range(128,len(order),64):
            stale=0 if acquire(order[offset:offset+64]) else stale+1
            if best is not None and best['rho']<.9:break
            if best is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    if best is not None and fallback and len(seen)<len(order):
        sub=subs[best['pool_id']]
        targets={int(rep) for i,rep in aliases.items() if subs[int(i)]==sub}
        acquire([i for i in order if i in targets and i not in seen])
    return len(seen),None if best is None else best['pool_id']
for row in R['rows']:
    folder=ROOT/'outputs/grid26/runs'/f"r0__{row['week']}__FULL"
    truth={x['pool_id']:x for x in read(folder/row['call'])['outcomes']}
    own=read(folder.parent/f"r0__{row['week']}__GNN0_128_GUARD"/row['call'])
    assert independent(row['order'],truth,own['aliases'])==(row['queries'],row['choice'])
for summary in R['summaries']:
    group=[r for r in R['rows'] if r['ranker']==summary['ranker']]
    assert len(group)==26 and sum(r['queries'] for r in group)==summary['queries']
    assert sum(r['same_full_choice'] for r in group)==summary['same_full_choice']
    assert summary['known_critical_choice_match']
audit={'passed':True,'independent_stop_reconstructions':len(R['rows']),'historical_inference_order_checks_main':52,
       'refits':0,'new_forecasts':0,'independent_forward':False,
       'warning':'Saved orders and legal dispatch-only stopping independently replayed. Does not independently refit/re-infer networks or establish closed-loop quality.'}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['# GRID33 现成排序器同状态诊断','','26个已暴露状态，共130个排序/停止比较。合法输入重新推理的GNN0/1顺序与历史52次逐项一致；独立实现从保存顺序重算130项查询和选择，无新AC仿真/拟合。', '',
       '| 排序器 | 查询 | 同FULL选择 | 最大rho差 | 找回1733关键动作 |', '|---|---:|---:|---:|---|']
for s in R['summaries']:lines.append(f"| {s['ranker']} | {s['queries']} | {s['same_full_choice']}/26 | {s['max_rho_regret']:.7f} | {s['known_critical_choice_match']} |")
lines += ['', '全部现成排序器通过已知关键动作门槛，因此需公平闭环比较，不能把MLP省略。GNN0相对MLP0/PPO查询差仅为有限开发线索，不是图结构必要性、长期成本或封存泛化证据。', '',R['warning']]
(OUT/'GRID33_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(audit))
