"""Freeze one free-residual fit using existing training data only."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid48';HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
Q=read(OUT/'qualification.json');assert Q['passed_mathematical_checks'] and Q['peak_mass_demand_exceeds_bound_cap']>0
assert Q['peak_in128_forbidden']==0 and not (OUT/'design.json').exists()
assert read(ROOT/'outputs/grid47/review.json')['passed_saved_data']
old=read(ROOT/'outputs/grid21/design.json')
D={'stage':'GRID48_ONE_FREE_PRIOR_RESIDUAL','change':'Only final scoring correction changes2*tanh(raw/2)to raw. Same original graph/plan/base decoder and106449trainable parameters,exact original initial tensors,zerohead,publicPPO rank prior,target,KLregularizer,optimizer,shuffle,batch8,64epochs,2seeds. No new modules,new teacher queries,RL,cutoff or guard.',
   'motivation':'Training-only analytic odds ceiling exp(4):72/148teacherpeaks cannot reach first rank;70/148desired peak masses exceed optimistic box bound. This does NOT forbid128shortlist inclusion;no claim it caused held-out failure. The existing prior KL0.1 remains to retain strong PPO behavior without a hard representational cap.',
   'training_weeks':old['training_weeks'],'development_weeks':old['development_weeks'],'kinds':['gnn'],'seeds':[0,1],
   'epochs':old['epochs'],'batch_size':old['batch_size'],'lr':old['lr'],'weight_decay':old['weight_decay'],'prior_KL_weight':old['prior_KL_weight'],
   'evaluation':'After both fixed final checkpoints saved,compare original/new on26oldFULL states with unchanged fast GRID28 PATIENCE64+local completion. Neither strict GRID47guard nor GRID43tiecompletion adopted. Finite/mask/fullfixedinput checks and knowncritical1733 required before any bounded physicalpilot. Keep bothseeds,never selectcheckpoint on development. Offline gain not deployed quality or speed.',
   'scope':'167unchanged originaltrain inputs,148positive;no GRID44family/variant used in training,tuning,or refit. All confirmation outputs retained unchanged. Later confirmation must use new frozen datefamilies. This change alone is not a novel algorithmic contribution.',
   'caps':{'training_s':600,'fits':2,'new_forecasts':0,'GRID44_evaluations':0}}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=list(HERE.glob('*.py'))+[ROOT/'work/grid21/prior.py',ROOT/'work/grid20/components.py',ROOT/'outputs/grid21/training_finished.json',OUT/'qualification.json']
files += [ROOT/p for p in Q['source_hashes']]
for seed in [0,1]:files += [ROOT/f'outputs/grid21/models/gnn_seed{seed}/{name}.pt' for name in ['initial','final']]
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN2FITS:only remove residual tanh bound;originalfast controller retained,no GRID44reuse. No fit started.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'fits':2,'old_target_unchanged':True,'no_new_forecasts':True}))
