from pathlib import Path
import datetime as dt,hashlib,json
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid73'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists();OUT.mkdir()
    assert not read(ROOT/'outputs/grid72/training_finished.json')['structural_qualified']
    D=dict(stage='GRID73_LOCAL_STRUCTURAL_GATE',frozen_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        change='Replace global RF score threshold with one depth3 locally calibrated decision tree. Same labels, inputs, split, support and no evaluation records. Preserve failed GRID72.',
        models=['plain','struct'],plain_features=12,struct_features=20,
        tree=dict(max_depth=3,min_samples_leaf=4,class_weight='balanced',random_state=0),
        leaf_rule='Allow early return only in a leaf with zero train positives, >=4 training cases from>=3 families, >=1 calibration case and zero calibration positives. Defer outside previous training-support envelope. No threshold sweep.',
        qualification='Same screen: zero omitted changed-action calibration cases and >=10% total safe-state query saving. Plain feature ablation reported, not alternate selected winner.',
        sources=['outputs/grid72/inputs.npz','outputs/grid72/input_receipts.json','outputs/grid72/design.json'],
        caps=dict(fits=2,wall_s=600,new_physical_queries=0),target='Teacher action change only, not long-term cost/survival risk.')
    (OUT/'design.json').write_text(json.dumps(D,ensure_ascii=False,indent=2),encoding='utf-8')
    frozen=read(ROOT/'outputs/grid72/code_freeze.json')
    for p,h in frozen.items():assert sha(ROOT/p)==h,p
    for p in list(Path(__file__).parent.glob('*.py'))+[OUT/'design.json']:
        frozen[p.relative_to(ROOT).as_posix()]=sha(p)
    (OUT/'code_freeze.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    (OUT/'RUN_STATE.md').write_text('FROZEN:two local gates; no physical simulation.\n',encoding='utf-8')
    print(json.dumps({'frozen':len(frozen),'fits':2,'physical_queries':0}))
if __name__=='__main__':main()
