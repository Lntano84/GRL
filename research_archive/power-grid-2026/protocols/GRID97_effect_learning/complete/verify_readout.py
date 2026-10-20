"""Arithmetic-only second check, no fitting/prediction/physical simulation."""
import json,hashlib,math
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid97_effect_learning'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    results=read(OUT/'results.json');pred=read(OUT/'predictions.json');checked=0
    for r in results:
        for field,selector in [('primary_nonpilot',lambda p:not p['pilot_known']),('all56_or_transfer',lambda p:True),('known_pilots',lambda p:p['pilot_known'])]:
            rows=[p for p in pred if p['scheme']==r['scheme'] and p['model']==r['model'] and selector(p)];m=r[field]
            assert len(rows)==m['n']
            if not rows:continue
            a=[p['p_same_path'] for p in rows if p['y']==1];b=[p['p_same_path'] for p in rows if p['y']==0]
            if a and b:
                auc=sum((x>y)+.5*(x==y) for x in a for y in b)/(len(a)*len(b));assert math.isclose(auc,m['auc'],abs_tol=1e-12)
                ba=.5*(sum(x>=.5 for x in a)/len(a)+sum(x<.5 for x in b)/len(b));assert math.isclose(ba,m['balanced_accuracy'],abs_tol=1e-12)
            assert math.isclose(sum((p['p_same_path']-p['y'])**2 for p in rows)/len(rows),m['brier'],abs_tol=1e-12)
            for t in [.5,.95]:
                selected=[p for p in rows if p['p_same_path']>=t];v=m['thresholds'][str(t)]
                assert v['proposed_omissions']==len(selected) and v['changed_path_misclassified']==sum(p['y']==0 for p in selected)
            checked+=1
    assert Counter(p['model'] for p in pred)=={n:79 for n in ['LOGISTIC','FOREST64','TABPFN_V2','ALWAYS_RETAIN','ALWAYS_OMIT','TRAIN_PRIOR']}
    for f,h in read(OUT/'freeze.json')['sha256'].items():assert sha(ROOT/f)==h,f
    clean=ROOT/'outputs/frame_label_cleanup_20261020'
    for f,h in read(clean/'artifact_manifest.json').items():assert sha(clean/f)==h,f
    assert read(OUT/'review.json')['passed']
    report=(OUT/'GRID97_report.md').read_text(encoding='utf8')
    assert all(s in report for s in ['0.500','0.498','0.381','未部署','9例','0.95','67.103'])
    receipt=dict(passed=True,arithmetic_metric_groups=checked,prediction_rows=len(pred),freeze_items=len(read(OUT/'freeze.json')['sha256']),old_runtime_sklearn='1.7.2 (verified separately after evaluation)',new_runtime_sklearn='1.6.1',new_physical_replays=0,extra_fits=0,extra_predictions=0,
        notes='Pairwise AUC, binary class recalls, Brier and threshold counts recomputed without sklearn. Tiny single-class date diagnostics are not two-class BA evidence. Original files unchanged.')
    (OUT/'readout_verification.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
