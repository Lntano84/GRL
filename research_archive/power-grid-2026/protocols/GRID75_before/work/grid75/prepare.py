from pathlib import Path
import datetime as dt,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid75';sys.path.insert(0,str(Path(__file__).parent));import feature_gate as G
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir()
    assert not read(ROOT/'outputs/grid74/training_finished.json')['structural_qualified']
    D=read(ROOT/'outputs/grid72/design.json')
    D.update(stage='GRID75_PUBLIC_GRID_CONTEXT_GATE',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        change='Final fixed-input qualification: add real public grid operating context (global38/local) from audited saved inputs. Same RF parameters, initial128 target, split and calibration. No additional model search after this result.',
        feature_names=G.NAMES,plain_features=24,struct_features=38,support_indices=G.SUPPORT,
        context='Public graph x/edges/global arrays from GRID21 cached input, before any queried suffix. No prior, targets, instance/date identifiers or hidden action outcomes are feature inputs.',
        models=['plain','struct'],caps=dict(fits=2,wall_s=600,new_physical_queries=0))
    save(OUT/'design.json',D)
    old_items={r['id']:r for r in read(ROOT/'outputs/grid72/input_receipts.json')['items']}
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids'];X=[];Y=[];T=[];items=[];checks=0
    for rec in read(ROOT/'outputs/grid64/records.json'):
        if rec['id'] not in old_items:continue
        item=old_items[rec['id']];rows=rec['rows'];order=[r['pool_id'] for r in rows];raw=read(ROOT/rec['source'])
        hashes={r['combined_action_hash']:r['pool_id'] for r in rows}
        aliases={int(i):hashes[r['combined_action_hash']] for i,r in zip(raw['pool_ids'],raw['outcomes']) if r['combined_action_hash'] in hashes}
        source=ROOT/'outputs/grid21/inputs/train'/f"{rec['id']}.npz"
        with np.load(source) as z:context={k:z[k] for k in ['x','edges','glob']}
        x=G.features(rows[:128],order,subs,aliases,rec['rho_limit'],context)
        changed=[dict(r,reward=-999.,rho=999.,strict_admissible=False) for r in rows[128:]]
        assert np.array_equal(x,G.features((rows[:128]+changed)[:128],order,subs,aliases,rec['rho_limit'],context));checks+=1
        X.append(x);Y.append(item['changed_action']);T.append(item['split']=='train')
        items.append(dict(item,context_source=source.relative_to(ROOT).as_posix(),context_sha256=sha(source)))
    np.savez_compressed(OUT/'inputs.npz',X=np.stack(X),y=np.asarray(Y,dtype=np.int64),train=np.asarray(T))
    save(OUT/'input_receipts.json',dict(items=items,source='outputs/grid72/input_receipts.json',legal_suffix_checks=checks))
    save(OUT/'preflight.json',dict(passed=True,rows=len(items),train=sum(T),calibration=len(items)-sum(T),
        feature_count=38,legal_suffix_checks=checks,no_new_physical_queries=True))
    frozen=read(ROOT/'outputs/grid74/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+list(OUT.glob('*')):
        if p.is_file():frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen);(OUT/'RUN_STATE.md').write_text('FROZEN:public grid-context qualification,2 fixed forests.\n',encoding='utf-8')
    print(json.dumps(read(OUT/'preflight.json')))
if __name__=='__main__':main()
