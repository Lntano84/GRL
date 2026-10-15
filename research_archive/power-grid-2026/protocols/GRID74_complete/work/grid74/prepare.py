"""One probe batch adds actual evidence before conditional verification."""
from pathlib import Path
import datetime as dt,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid74';sys.path.insert(0,str(ROOT/'work/grid72'));import feature_gate as G
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir()
    assert not read(ROOT/'outputs/grid73/training_finished.json')['structural_qualified']
    D=read(ROOT/'outputs/grid72/design.json')
    D.update(stage='GRID74_PROBE_THEN_CONDITIONAL_VERIFY',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        change='One fixed64-candidate probe after initial safe128. If probe changes incumbent, defer to exact GRID70. Otherwise predict residual need using prefix192 feedback and the same two fixed forests. Retain GRID72/73, no parameter sweep.',
        label='Same teacher final-action-change target from GRID72. Fit only stable-probe training cases. Improved-probe cases always defer; no classifier needed for them.',
        calibration='Zero omitted teacher changes; threshold=min residual in-support positive portable probability. Outside-training-support always defer. Same10% total safe-calibration query saving screen; stopped cost192, not128. No closed-loop quality guarantee.',
        qualification='Zero omitted teacher-action changes and>=10% total safe-state calibration query saving. Structural/plain feature contrast reported; passing only enables separately frozen pilot.',
        probe_batch=64,read_evaluation_results=False)
    save(OUT/'design.json',D)
    old_items={r['id']:r for r in read(ROOT/'outputs/grid72/input_receipts.json')['items']}
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];X=[];Y=[];T=[];stable=[];items=[];checks=0
    for rec in read(ROOT/'outputs/grid64/records.json'):
        if rec['id'] not in old_items:continue
        item=old_items[rec['id']];rows=rec['rows'];order=[r['pool_id'] for r in rows]
        prefix=rows[:min(192,len(rows))];before=G.M.best_row(rows[:128]);after=G.M.best_row(prefix)
        assert before is not None and after is not None
        raw=read(ROOT/rec['source']);hashes={r['combined_action_hash']:r['pool_id'] for r in rows}
        aliases={int(i):hashes[r['combined_action_hash']] for i,r in zip(raw['pool_ids'],raw['outcomes']) if r['combined_action_hash'] in hashes}
        x=G.features(prefix,order,subs,aliases,rec['rho_limit']);changed=before['pool_id']!=after['pool_id']
        x[5]=G.M.quality(after)-G.M.quality(before);x[6]=0. if changed else 1/16.
        altered=[dict(r,reward=-999.,rho=999.,strict_admissible=False) for r in rows[len(prefix):]]
        z=G.features((prefix+altered)[:len(prefix)],order,subs,aliases,rec['rho_limit']);z[5:7]=x[5:7]
        assert np.array_equal(x,z);checks+=1
        X.append(x);Y.append(item['changed_action']);T.append(item['split']=='train');stable.append(not changed)
        items.append(dict(item,probe_queries=len(prefix),probe_changes_incumbent=changed))
    np.savez_compressed(OUT/'inputs.npz',X=np.stack(X),y=np.asarray(Y,dtype=np.int64),train=np.asarray(T),stable=np.asarray(stable))
    save(OUT/'input_receipts.json',dict(items=items,source='outputs/grid72/input_receipts.json',source_sha256=sha(ROOT/'outputs/grid72/input_receipts.json'),legal_suffix_checks=checks))
    save(OUT/'preflight.json',dict(passed=True,rows=len(items),train=sum(T),calibration=len(items)-sum(T),stable_train=sum(t and s for t,s in zip(T,stable)),
        residual_train_positive=sum(y for y,t,s in zip(Y,T,stable) if t and s),residual_calibration_positive=sum(y for y,t,s in zip(Y,T,stable) if not t and s),legal_suffix_checks=checks,new_queries=0))
    frozen=read(ROOT/'outputs/grid73/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+list(OUT.glob('*')):
        if p.is_file():frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen);(OUT/'RUN_STATE.md').write_text('FROZEN:probe-then-verify,2 fixed fits; no new physical query.\n',encoding='utf-8')
    print(json.dumps(read(OUT/'preflight.json')))
if __name__=='__main__':main()
