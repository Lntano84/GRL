"""Freeze two nonlearning order controls before observing their diagnostic outcomes."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid35'
assert json.loads((ROOT/'outputs/grid34/review.json').read_text())['passed']
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
D={'stage':'GRID35_NONLEARNING_ORDER_SCREEN','weeks':json.loads((ROOT/'outputs/grid33/design.json').read_text())['weeks'],
   'rules':['STATIC_ID','LOCAL_STRESS'],
   'static_rule':'Eligible representative pool IDs ascending; no learned weights or outcomes.',
   'stress_rule':'Max current incident line rho on the two public busbars of the candidate original substation, descending; pool ID ascending ties. All candidates at a substation share its legal current-load score; no action forecast/sensitivity information is read.',
   'public_subids_source':'outputs/grid28/offline.json public_subids: static public candidate-to-substation mapping, not learned weights.',
   'stopping':'Unchanged GRID28 local_complete PATIENCE64,initial128,batch64,min256,patience2,safe.9,semanticcompletion of incumbent substation including aliases.',
   'source':'Same26 exposed FULL-state feedback and legal public inputs as GRID33. FULL feedback is consulted only by dispatcher for chosen queries; not an online strategy result.',
   'gate':'Known March1733 FULL action must be recovered for unchanged closed-loop eligibility. A passed screen does not prove closed-loop cost; a failure rejects only this order/stopping configuration.',
   'comparisons':'Report both controls and all5existing GRID33 rankers, never choose a better cheap rule to hide the other. No declaration of GNN architecture advantage from a small screen.',
   'caps':{'wall_s':120,'new_forecasts':0,'new_fits':0,'reserved_evaluations':0}}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[Path(__file__).resolve(),ROOT/'work/grid35/offline.py',ROOT/'work/grid35/audit.py',ROOT/'work/grid28/policy.py',ROOT/'work/grid27/policy.py']
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:52public-order screens,no refit/newforecasts/finaldates.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'screens':52}))
