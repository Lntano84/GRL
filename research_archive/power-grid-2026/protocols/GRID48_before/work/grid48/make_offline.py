"""Derive a fixed cached comparison;only scorer forward changes from GRID38."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
s=(ROOT/'work/grid38/offline.py').read_text(encoding='utf-8')
changes={"OUT=ROOT/'outputs/grid38'":"OUT=ROOT/'outputs/grid48'",
    'from prior import ResidualScorer':'from free_residual import FreeResidualScorer',
    "assert len(read(OUT/'training_finished.json')['fits'])==4":"assert len(read(OUT/'training_finished.json')['fits'])==2",
    "model=ResidualScorer(kind,":"model=FreeResidualScorer(kind,",
    'assert np.isfinite(residual).all() and np.max(np.abs(residual))<=2.+1e-6':'assert np.isfinite(residual).all()',
    "'new_fits':4":"'new_fits':2",'All four new models':'Both fixed new models',
    'The original and revised MLP':'The original and revised GNN'}
changes.pop('The original and revised MLP')
for a,b in changes.items():assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
# The local scorer module is visible even after prepending the original modules.
s=s.replace("sys.path[:0]=[str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]","sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT/'work/grid21'),str(ROOT/'work/grid20')]")
assert not (HERE/'offline.py').exists();(HERE/'offline.py').write_text(s,encoding='utf-8')
print('Derived original-fast-policy comparison,2GNNseeds,26old states;no guard/model portfolio changes.')
