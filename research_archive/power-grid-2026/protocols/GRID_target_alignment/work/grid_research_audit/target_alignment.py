"""Read training-only cached labels: soft-rho targets versus actual FP32 selection."""
import csv,hashlib,json,math,os,struct
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid_target_audit';OUT.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
D=read(ROOT/'outputs/grid21/design.json');records=[];artifacts=[]
for meta_file in sorted((ROOT/'outputs/grid21/inputs/train').glob('*.json')):
    meta=read(meta_file);assert meta['week'] in D['training_weeks']
    source=ROOT/meta['source'];assert hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_sha256']
    c=read(source);scores=dict(zip(c['pool_ids'],c['outcomes']))
    with np.load(meta_file.with_suffix('.npz')) as z:
        target=z['target'].copy();mask=z['mask'].copy();prior=z['prior'].copy();reps=z['representatives'].tolist()
    assert reps==list(np.flatnonzero(mask))
    valid=[i for i in reps if scores[i]['strict_admissible']]
    expected=np.zeros_like(target)
    r={'id':meta_file.stem,'week':meta['week'],'step':meta['step'],'representatives':len(reps),'valid':len(valid)}
    if valid:
        best=min(scores[i]['rho'] for i in valid)
        expected[valid]=np.exp(-np.array([scores[i]['rho']-best for i in valid])/D['temperature']);expected/=expected.sum(dtype=np.float64)
        mx=max(f32(scores[i]['reward']) for i in valid);bestset=[i for i in valid if f32(scores[i]['reward'])==mx];teacher=min(bestset)
        order=sorted(reps,key=lambda i:(-float(prior[i]),i));rank=order.index(teacher)+1
        probs=[float(target[i]) for i in valid];entropy=-math.fsum(p*math.log(p) for p in probs if p>0)
        min_rho_set=[i for i in valid if scores[i]['rho']==best]
        immutable_ahead=sum(float(prior[j])-float(prior[teacher])>=4 for j in reps if j!=teacher)
        r.update(teacher=teacher,reward_optimal_count=len(bestset),teacher_rho=scores[teacher]['rho'],best_rho=best,
            reward_teacher_also_rho_best=teacher in min_rho_set,
            teacher_prior_rank=rank,teacher_in_prior128=rank<=128,
            soft_mass_teacher=float(target[teacher]),soft_mass_reward_optimal=math.fsum(float(target[i]) for i in bestset),
            effective_soft_candidates=math.exp(entropy),
            expected_soft_rho_regret=math.fsum(float(target[i])*(scores[i]['rho']-best) for i in valid),
            soft_mass_rho_gap_over_001=math.fsum(float(target[i]) for i in valid if scores[i]['rho']-best>.001),
            target_peak_matches_actual_reward=all(f32(scores[i]['reward'])==mx for i in valid if target[i]==target.max()),
            unavoidable_ahead_from_residual_bound=immutable_ahead,
            independent_residual_can_promote_into128=immutable_ahead<128)
    assert np.max(np.abs(expected-target))<1e-7
    records.append(r);artifacts.append({'path':meta['source'],'sha256':meta['source_sha256']})
active=[r for r in records if r['valid']]
def average(k):return math.fsum(r[k] for r in active)/len(active)
R={'rows':len(records),'valid_rows':len(active),'source_hashes':artifacts,'temperature':D['temperature'],
   'teacher_is_rho_best':sum(r['reward_teacher_also_rho_best'] for r in active),
   'target_peak_matches_reward':sum(r['target_peak_matches_actual_reward'] for r in active),
   'prior128_contains_teacher':sum(r['teacher_in_prior128'] for r in active),
   'all_can_promote_into128_under_independent_residuals':all(r['independent_residual_can_promote_into128'] for r in active),
   'mean_teacher_mass':average('soft_mass_teacher'),'mean_reward_optimal_mass':average('soft_mass_reward_optimal'),
   'median_reward_optimal_mass':float(np.median([r['soft_mass_reward_optimal'] for r in active])),
   'mean_effective_candidates':average('effective_soft_candidates'),
   'mean_expected_soft_rho_regret':average('expected_soft_rho_regret'),
   'mean_mass_gap_over_001':average('soft_mass_rho_gap_over_001'),
   'new_fits':0,'new_forecasts':0,'development_labels_opened':0,'final_dates_opened':0,
   'interpretation':'Valid peaks/reward alignment are verified, not a code defect. Distributed soft mass and bounded prior residual are approximations; neither proves downstream harm nor hard-label superiority. Independent residual promotion possibility is only a mathematical capacity bound, not realizability by the shared network.'}
(OUT/'audit.json').write_text(json.dumps(R,indent=2),encoding='utf-8')
keys=sorted({k for r in records for k in r})
with (OUT/'per_training_state.csv').open('w',encoding='utf-8',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(records)
lines=['# Training-only objective alignment audit','',f"Rows {R['rows']}; positive-label states {R['valid_rows']}; temperature {R['temperature']}.",'',
    f"Teacher is rho minimum: {R['teacher_is_rho_best']}/{len(active)}; target peaks are reward optimal: {R['target_peak_matches_reward']}/{len(active)}.",
    f"Prior128 teacher coverage: {R['prior128_contains_teacher']}/{len(active)}. Mean teacher mass {R['mean_teacher_mass']:.6f}; optimal-set mass {R['mean_reward_optimal_mass']:.6f}; effective candidates {R['mean_effective_candidates']:.2f}.",
    f"Mean target expected rho gap {R['mean_expected_soft_rho_regret']:.6f}; mass beyond .001 gap {R['mean_mass_gap_over_001']:.4f}.",'',R['interpretation'],
    '', 'Saved source checks are training-only. No fitting, rollout or heldout outcome access. A sharpening trial, if chosen, is a model configuration hypothesis and not new methodological contribution.']
(OUT/'TARGET_ALIGNMENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in R.items() if k!='source_hashes'},indent=2))
