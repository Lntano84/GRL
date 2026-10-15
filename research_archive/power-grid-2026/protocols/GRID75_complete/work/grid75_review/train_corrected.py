"""Keep frozen sources, repair only portable comparison, deterministic refit."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(Path(__file__).parent))
from runtime_corrected import predict,decide
hi=np.float32(1.000000238418579);lo=np.float32(1.0000001192092896);mid=(float(hi)+float(lo))/2
m=dict(trees=[dict(left=[1,-1,-1],right=[2,-1,-1],feature=[0,-1,-1],threshold=[mid,0.,0.],positive=[.5,0.,1.])],support=[],threshold=.5)
assert bool(hi<=mid) and not bool(float(hi)<=mid)
assert predict(m,[hi])==1. and predict(m,[lo])==0.
source=ROOT/'work/grid75/train.py';s=source.read_text(encoding='utf-8')
marker='def main():';assert s.count(marker)==1
patch='from runtime_corrected import predict as corrected_predict,decide as corrected_decide\nG.predict=corrected_predict;G.decide=corrected_decide\n'
s=s.replace(marker,patch+marker)
receipt=dict(original_train_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),correction='Explicit float32 value promoted to double before comparison with double tree threshold; frozen feature/fit sources unchanged.',
    hand_check='Adjacent float32 split with upper rounded midpoint; naive numpy scalar comparison disagrees with sklearn double threshold semantics.',
    original_failure='Portable inference parity assertion before any model artifact persisted. One interrupted plain fit, deterministic refit under same data/seed/parameters.',new_physical_queries=0)
(ROOT/'outputs/grid75/portable_correction.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
exec(compile(s,str(source),'exec'),{'__file__':str(source),'__name__':'__main__'})
