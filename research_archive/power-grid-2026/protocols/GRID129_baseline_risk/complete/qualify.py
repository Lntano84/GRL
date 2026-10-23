"""Independently align new baseline ledger with original GRID101 inputs/labels."""
from pathlib import Path
import gzip,hashlib,json
import numpy as np

R=Path('D:/Codex/GRL研究');O=R/'outputs/grid129_baseline_risk'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    paths=[Path(__file__),R/'outputs/grid101_gain_learning/data.npz',R/'outputs/grid101_gain_learning/rows.json',
           R/'outputs/grid101_gain_learning/design.json',R/'outputs/grid121_date_transfer/design.json',
           O/'train_label_ledger.jsonl.gz',O/'train_public_inputs.npz']
    before={p.relative_to(R).as_posix():sha(p) for p in paths}
    rows=json.loads(paths[2].read_text(encoding='utf8'));index={r['sample_id']:i for i,r in enumerate(rows)}
    with np.load(paths[1]) as z:oldx=z['X'];oldgain=z['gain']
    with np.load(O/'train_public_inputs.npz') as z:newx=z['X'];ids=z['sample_ids'];q=z['known_queries'];keys=z.files
    assert set(keys)=={'X','known_queries','sample_ids'}
    with gzip.open(O/'train_label_ledger.jsonl.gz','rt',encoding='utf8') as f:ledger=[json.loads(line) for line in f]
    assert len(ids)==len(ledger)==2378
    err=0.
    for i,(sid,l) in enumerate(zip(ids,ledger)):
        j=index[str(sid)];assert str(sid)==l['sample_id'] and rows[j]['partition']=='train'
        e=float(np.max(abs(newx[i,:37]-oldx[j])));err=max(err,e);assert np.allclose(newx[i,:37],oldx[j],rtol=1e-6,atol=1e-6)
        assert abs(l['full_baseline_gain']-float(oldgain[j]))<1e-12
        assert q[i]==int(round(float(np.expm1(oldx[j,3]))))==l['known_saved_queries_if_skip']
    folds=json.loads(paths[4].read_text(encoding='utf8'))['folds'][0]
    fit=set(folds['fit_dates']);cal=set(folds['held_dates']);assert len(fit)==12 and len(cal)==4 and not fit&cal
    assert {l['date'] for l in ledger}==fit|cal
    for part,dates in [('fit',fit),('calibration',cal)]:
        rr=[l for l in ledger if l['date'] in dates]
        assert any(l['full_baseline_gain']>0 for l in rr) and any(l['full_baseline_gain']==0 for l in rr)
    assert before=={p.relative_to(R).as_posix():sha(p) for p in paths}
    result=dict(passed=True,all_2378_full_gain_labels_match_grid101=True,public_first37_match_grid101=True,
        maximum_raw_feature_difference=err,public_file_has_no_private_labels=True,known_queries_match_public_feature=True,
        fit_dates=sorted(fit),calibration_dates=sorted(cal),date_roles_disjoint=True,
        fit_prefixes=sum(l['date'] in fit for l in ledger),calibration_prefixes=sum(l['date'] in cal for l in ledger),
        fits=0,model_forwards=0,source_unchanged=True)
    (O/'cross_version_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    (O/'cross_version_source_hashes.json').write_text(json.dumps(before,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
