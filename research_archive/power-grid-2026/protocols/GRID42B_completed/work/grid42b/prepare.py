"""Preserve the failed import-only probe, then avoid its generic module collision."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
folder=ROOT/'work/grid42b'; folder.mkdir(parents=True,exist_ok=True)
source=(ROOT/'work/grid42/policy.py').read_text(encoding='utf-8')
assert not (folder/'tied_policy.py').exists()
(folder/'tied_policy.py').write_text(source,encoding='utf-8')
review=(ROOT/'work/grid42/review.py').read_text(encoding='utf-8')
changes={"OUT=ROOT/'outputs/grid42'":"OUT=ROOT/'outputs/grid42b'",
         "str(ROOT/'work/grid42')":"str(ROOT/'work/grid42b')",
         'from policy import tied_complete, hand_checks, key':'from tied_policy import tied_complete, hand_checks, key',
         "ROOT/'work/grid42/policy.py'":"ROOT/'work/grid42b/tied_policy.py'",
         "'stage':'GRID42_OBSERVED_TIE_COMPLETION'":"'stage':'GRID42B_OBSERVED_TIE_COMPLETION'"}
for a,b in changes.items(): assert review.count(a)==1,(a,review.count(a)); review=review.replace(a,b)
(folder/'review.py').write_text(review,encoding='utf-8')
(ROOT/'outputs/grid42/engineering_failure.json').write_text(json.dumps({
    'status':'IMPORT_ONLY_FAILURE','error':'ImportError: GRID28 generic policy import resolved GRID42 policy rather than GRID27 staged module.',
    'cached_comparisons_completed':0,'new_fits':0,'new_forecasts':0,
    'repair':'Use distinct tied_policy module name in GRID42B; old source and frozen design untouched.'},indent=2),encoding='utf-8')
print('Prepared distinct module name; original failed attempt preserved.')
