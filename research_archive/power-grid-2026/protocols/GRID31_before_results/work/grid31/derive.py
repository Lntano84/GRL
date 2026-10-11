"""Create explicit three-arm measured pilot from the retained GRID29C driver."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'work/grid31'
source=(ROOT/'work/grid29c/run.py').read_text(encoding='utf-8')
changes=[
    ('OUT=ROOT/\'outputs/grid29c\'','OUT=ROOT/\'outputs/grid31\''),
    ("str(ROOT/'work/grid29'),str(ROOT/'work/grid08')","str(ROOT/'work/grid30'),str(ROOT/'work/grid29'),str(ROOT/'work/grid08')"),
    ('from cache_dc import run_dc_cached','from cache_dc import run_dc_cached\n    from cache_safe import compute_safe_cached'),
    ("setattr(OptimModule,name,wrapper(name,run_dc_cached if name=='run_dc' and item['cache_dc'] else fn))",
     "chosen=run_dc_cached if name=='run_dc' and item['cache_dc'] else compute_safe_cached if name=='compute_optimum_safe' and item['cache_safe'] else fn\n        setattr(OptimModule,name,wrapper(name,chosen))"),
    ("'cache_counts':[dict(getattr(m,'_grid29_dc_counts',{})) for m in optimizers],",
     "'cache_counts':[dict(getattr(m,'_grid29_dc_counts',{})) for m in optimizers],\n            'safe_cache_counts':[dict(getattr(m,'_grid30_safe_counts',{})) for m in optimizers],"),
    ("'planned':4","'planned':6"),
    (';4balanced original/cache component profiles.',';6balanced original/DC/DC+safe component profiles.')]
for old,new in changes:
    expected=2 if old=="'planned':4" else 1
    assert source.count(old)==expected,(old,source.count(old));source=source.replace(old,new)
source=source.replace('Identical component instrumentation in original/shared-cache closed loops.',
    'GRID31 three measured arms; derived from retained GRID29C. Original/DC/DC+safe.')
ast.parse(source)
assert not (OUT/'run.py').exists();(OUT/'run.py').write_text(source,encoding='utf-8')
print('Created explicit GRID31 driver; no scientific run.')
