"""Independent stdlib stopping reconstruction and arithmetic for both cheap controls."""
import hashlib,json,struct
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid35'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
R=read(OUT/'review.json');assert R['passed_engineering']
for file,h in {**R['input_hashes'],**read(OUT/'code_freeze.json')}.items():assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==h,file
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
        for start in range(128,len(order),64):
            stale=0 if acquire(order[start:start+64]) else stale+1
            if best is not None and best['rho']<.9:break
            if best is not None and len(seen)>=256 and stale>=2 and len(seen)<len(order):break
    if best is not None and fallback and len(seen)<len(order):
        sub=subs[best['pool_id']];target={int(r) for i,r in aliases.items() if subs[int(i)]==sub}
        acquire([i for i in order if i in target and i not in seen])
    return len(seen),None if best is None else best['pool_id']
for row in R['rows']:
    folder=ROOT/'outputs/grid26/runs'/f"r0__{row['week']}__FULL"
    truth={r['pool_id']:r for r in read(folder/row['call'])['outcomes']}
    own=read(folder.parent/f"r0__{row['week']}__GNN0_128_GUARD"/row['call'])
    assert independent(row['order'],truth,own['aliases'])==(row['queries'],row['choice'])
for s in R['summaries']:
    group=[r for r in R['rows'] if r['rule']==s['rule']]
    assert len(group)==26 and sum(r['queries'] for r in group)==s['queries']
    assert sum(r['same_full_choice'] for r in group)==s['same_full_choice']
audit={'passed':True,'independent_stop_reconstructions':52,'hidden_feedback_checks_main':52,'new_fits':0,'new_forecasts':0}
(OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
lines=['# GRID35 两个非学习顺序对照','','所有方法共享既有停止规则，以下只为26个已暴露状态上的选择/查询诊断。', '',
    '| 顺序 | 查询 | 同FULL选择 | 最大rho差 | 找回1733关键动作 |','|---|---:|---:|---:|---|']
for s in R['summaries']:lines.append(f"| {s['rule']} | {s['queries']} | {s['same_full_choice']}/26 | {s['max_rho_regret']:.7f} | {s['critical_choice_match']} |")
for s in R['existing_rankers']:lines.append(f"| {s['ranker']} (GRID33) | {s['queries']} | {s['same_full_choice']}/26 | {s['max_rho_regret']:.7f} | {s['known_critical_choice_match']} |")
lines+=['',R['warning'],'','失败只阻断当前顺序/停止组合；通过只允许闭环资格检查，不是证明廉价规则或GNN的整体速度。']
(OUT/'GRID35_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('COMPLETED:52cheaporderdiagnostics auditted;closedloopnotstarted.\n',encoding='utf-8')
print(json.dumps(audit))
