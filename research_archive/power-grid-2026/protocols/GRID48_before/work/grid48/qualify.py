"""Training-only check of bounded prior-residual representation; no test labels.

Odds can change by at most exp(4) between two actions when corrections are in
[-2,2]. This can limit optimal-action probability, but with909actions DOES NOT
make a candidate's inclusion in a128shortlist mathematically impossible.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid48'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
OUT.mkdir(exist_ok=True);assert not (OUT/'qualification.json').exists()
D=read(ROOT/'outputs/grid21/design.json');ids=read(ROOT/'outputs/grid21/training_finished.json')['training_input_ids']
reserved={r['date'] for r in read(ROOT/'outputs/grid_final_holdout/reserved_date_families.json')['date_families']}
assert all(i[:10] not in reserved for i in ids)
rows=[];hashes={}
for identifier in ids:
    p=ROOT/'outputs/grid21/inputs/train'/f'{identifier}.npz';meta=read(p.with_suffix('.json'));assert meta['week'] in D['training_weeks']
    hashes[p.relative_to(ROOT).as_posix()]=sha(p)
    with np.load(p) as z:q=z['target'].astype(float);prior=z['prior'].astype(float);mask=z['mask'].astype(bool)
    if not q.any():continue
    peak=mask & (q==q.max());assert peak.any() and not np.any(q[~mask])
    pbest=prior[peak].max();forced=int(np.sum(mask & ~peak & (prior>pbest+4)))
    probability=np.zeros_like(prior);probability[mask]=np.exp(prior[mask]-prior[mask].max());probability/=probability.sum()
    peakprior=probability[peak].sum();cap=math.exp(4)*peakprior/(math.exp(4)*peakprior+(1-peakprior))
    # Direct corner evaluation independently checks the analytic mass ceiling.
    corner=prior+np.where(peak,2.,-2.);cornerprob=np.zeros_like(prior);cornerprob[mask]=np.exp(corner[mask]-corner[mask].max());cornerprob/=cornerprob.sum()
    assert abs(cornerprob[peak].sum()-cap)<1e-12
    rows.append({'id':identifier,'teacher_peak_candidates':int(peak.sum()),'prior_peak_best_rank':int(np.sum(mask & (prior>pbest)))+1,
       'forced_ahead_of_best_peak':forced,'cannot_rank_peak_first':forced>0,'cannot_include_peak_in128':forced>=128,
       'target_peak_mass':float(q[peak].sum()),'best_possible_peak_mass_with_bound':float(cap),
       'target_peak_mass_exceeds_bound_cap':bool(float(q[peak].sum())>cap+1e-6)})
assert len(rows)==148 and len(ids)==167
assert not any(r['cannot_include_peak_in128'] for r in rows)
R={'passed_mathematical_checks':True,'source':'Only167immutable GRID21training inputs,148positive targets;no development or GRID44 feedback.',
   'rows':rows,'n_positive':148,'peak_cannot_be_ranked_first':sum(r['cannot_rank_peak_first'] for r in rows),
   'peak_mass_demand_exceeds_bound_cap':sum(r['target_peak_mass_exceeds_bound_cap'] for r in rows),
   'peak_in128_forbidden':0,'source_hashes':hashes,'new_fits':0,'new_forecasts':0,
   'warning':'Representation limit on peak probability/first rank,not proof of shortlist failure,closed-loop loss,or root cause of a particular held-out state. A larger function class may overfit weak training data.'}
(OUT/'qualification.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('TRAINING-ONLY QUALIFICATION COMPLETE;no model fit or development inference yet.\n',encoding='utf-8')
print(json.dumps({k:R[k] for k in ['passed_mathematical_checks','n_positive','peak_cannot_be_ranked_first','peak_mass_demand_exceeds_bound_cap','peak_in128_forbidden','warning']}))
