"""Freeze a single existing-threshold guard diagnosis, development data only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid46'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
assert read(ROOT/'outputs/grid45/review.json')['passed_independent_scalar_audit']
OUT.mkdir(exist_ok=False)
D={'stage':'GRID46_COMMON_AUTHOR_THRESHOLD_DIAGNOSTIC','cause':'GRID44 April23 first physical divergence: bothGNN andPPO stop above author .9 buffer with true unqueried reward improvement. Do not train on confirmation data.',
   'change':'After unchanged GRID28 semantic completion,if observed incumbent still rho>=original author .9 and candidates remain,query every remaining canonical candidate. All rankers receive same rule;no new threshold,tie rule,model or optimizer. Historical strict-author fallback already exists:do not call restoration a new safety method.',
   'scope':'130cached decisions=26old FULL states x5existing rankers. Old development feedback only;no new forecasts or refits,no GRID44 feedback used as labels.',
   'screen':'Proceed to bounded development closed loop only if old critical1733 recovered for all5rankers,queries never exceed FULL,unrevealed feedback invariance passes. Does not promise longitudinal quality or whole speed.',
   'models':'OriginalGNN0/1,authorPPO,GRID40BSoftGNN0/1 unchanged;GRID43tiecandidate not used.',
   'reserved_status':'GRID44eightfamilies now exposed confirmation but allvariants still excluded from training/tuning;later reuse cannot be independent.'}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),HERE/'safe_completion.py',HERE/'review.py',HERE/'audit.py',ROOT/'work/grid28/policy.py',ROOT/'work/grid27/policy.py',ROOT/'outputs/grid42b/library.json',ROOT/'outputs/grid33/review.json',ROOT/'outputs/grid40b/offline_review.json']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print(json.dumps({'frozen':True,'cached_decisions':130,'new_fits':0,'new_forecasts':0}))
