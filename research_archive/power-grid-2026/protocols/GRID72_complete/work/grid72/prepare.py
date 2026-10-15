"""Freeze two small fits and derive targets from fully paid training traces."""
from pathlib import Path
import datetime as dt,hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid72';sys.path.insert(0,str(Path(__file__).parent))
import feature_gate as G
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def main():
    assert not OUT.exists();OUT.mkdir()
    sp=importlib.util.spec_from_file_location('grid72_guard',ROOT/'work/grid70/policies.py');P=importlib.util.module_from_spec(sp);sp.loader.exec_module(P)
    records=read(ROOT/'outputs/grid64/records.json');subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    excluded={w[:10] for w in read(ROOT/'outputs/grid70/design.json')['weeks']}
    assert not excluded&{r['family'] for r in records}
    design=dict(stage='GRID72_CONDITIONAL_VERIFICATION_QUALIFICATION',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        authorization='User: start the conference-comparison contribution direction.',
        models=['plain','struct'],feature_names=G.NAMES,plain_features=12,struct_features=20,
        forest=dict(n_estimators=128,max_depth=4,min_samples_leaf=4,class_weight='balanced',random_state=0,n_jobs=1),
        label='GRID70 guard final chosen action differs from the initial128 safe incumbent. Training target for teacher imitation only; not a long-term cost loss label.',
        calibration='Preserve all calibration cases that change teacher action: threshold=min portable probability among in-support positive cases. Threshold0 if no in-support positives. Defer on out-of-training support. Never choose threshold using evaluation dates.',
        support_indices=G.SUPPORT,split='Unchanged GRID64:14 train families129 states,4 calibration families22 states; all13 evaluation families excluded.',
        qualification='Require no omitted changed-action calibration case and >=10% total query saving vs uniform guard on safe calibration states. Plain model is fixed feature ablation, not an alternate selected winner. Structural incremental value reported even if absent. Passing enables a separately frozen closed-loop pilot, not a learned quality guarantee.',
        next_baseline='One literature DQN family; same20 legal features and RETURN/GLOBAL64/DELEGATE_GUARD mode actions. Delegate available at initial128; matches our complete action option. Complete source/action/utility mapping must be frozen before DQN fits. No full A*/RRT* reproduction claim.',
        caps=dict(fits=2,wall_s=600,new_physical_queries=0),read_evaluation_results=False)
    save(OUT/'design.json',design)
    items=[];X=[];Y=[];perturb_checks=0
    for rec in records:
        rows=rec['rows'];order=[r['pool_id'] for r in rows];assert order==sorted(order)
        prefix=rows[:128];best=G.M.best_row(prefix)
        if best is None or best['rho']>=.9:continue
        raw=read(ROOT/rec['source']);hashes={r['combined_action_hash']:r['pool_id'] for r in rows}
        aliases={int(i):hashes[r['combined_action_hash']] for i,r in zip(raw['pool_ids'],raw['outcomes']) if r['combined_action_hash'] in hashes}
        truth={r['pool_id']:r for r in rows}
        teacher=P.run(order,lambda i,p:{**truth[i],'phase':p},'FRAME_GUARD',subs,aliases,limit=rec['rho_limit'])
        x=G.features(prefix,order,subs,aliases,rec['rho_limit']);y=int(teacher['chosen_pool_id']!=best['pool_id'])
        altered=[{**r,'reward':-999.,'rho':999.,'strict_admissible':False} for r in rows[128:]]
        assert np.array_equal(x,G.features((prefix+altered)[:128],order,subs,aliases,rec['rho_limit']));perturb_checks+=1
        X.append(x);Y.append(y);items.append(dict(id=rec['id'],family=rec['family'],split=rec['split'],changed_action=y,
            guard_queries=teacher['public_queries'],initial_queries=len(prefix),prefix_action=best['pool_id'],teacher_action=teacher['chosen_pool_id'],
            source=rec['source'],source_sha256=sha(ROOT/rec['source'])))
    np.savez_compressed(OUT/'inputs.npz',X=np.stack(X),y=np.asarray(Y,dtype=np.int64),train=np.asarray([r['split']=='train' for r in items]))
    save(OUT/'input_receipts.json',dict(items=items,source_records_sha256=sha(ROOT/'outputs/grid64/records.json'),
        evaluation_families_excluded=sorted(excluded),suffix_checks=perturb_checks,new_queries=0))
    save(OUT/'preflight.json',dict(passed=True,rows=len(items),train=sum(r['split']=='train' for r in items),calibration=sum(r['split']=='calibration' for r in items),
        train_positive=sum(r['changed_action'] for r in items if r['split']=='train'),calibration_positive=sum(r['changed_action'] for r in items if r['split']=='calibration'),
        legal_suffix_checks=perturb_checks,no_evaluation_family_overlap=True,new_queries=0))
    frozen=read(ROOT/'outputs/grid70/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+list(OUT.glob('*')):
        if p.is_file():frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    save(OUT/'code_freeze.json',frozen)
    (OUT/'RUN_STATE.md').write_text('FROZEN:2 small conditional-verification fits; no physical replay.\n',encoding='utf-8')
    print(json.dumps(read(OUT/'preflight.json')))
if __name__=='__main__':main()
