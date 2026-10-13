"""One prior-only revision; identical original initialization/data/optimization."""
import hashlib,json,py_compile,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/grid58'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(ROOT/'outputs/grid57_prior_review/review.json')['passed']
assert not OUT.exists();OUT.mkdir()
D=read(ROOT/'outputs/grid50/design.json')
D.update(stage='GRID58_ONE_FREQUENCY_PRIOR_REVISION',
    change='Only the rank prior changes: public teacher-frequency order/STATIC_ID replaces original PPO+physical-tail order. Same soft-compressed residual architecture/scale2,zero-head original initialization,167oldtraining states/148targets,optimizer,shuffle,64epochs,two seeds,K128/stopping/guard.',
    scope='One bounded prior revision motivated by audited strong ablation. No new teacher,model menu,constant/epoch selection,held-date labels or alternative frequency transform.',
    evaluation='After two fixed fits, replay only26existing olddevelopment FULL-feedback states. Both known1733choices must match before any separately frozen olddevelopment closedloop; otherwise preserve models and stop this revision. Cached diagnostics are not deployment scores.',
    cap_note='Training600s/twofits;no new simulation. No using GRID52/55/57evaluation feedback to fit or select a checkpoint.')
assert set(w[:10] for w in D['training_weeks']).isdisjoint(w[:10] for w in read(ROOT/'outputs/grid55/design.json')['weeks'])
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
shutil.copy2(ROOT/'work/grid50/compressed_residual.py',HERE/'compressed_residual.py')
source=(ROOT/'work/grid50/train.py').read_text(encoding='utf-8').replace("'outputs/grid50'","'outputs/grid58'")
needle='from compressed_residual import CompressedResidualScorer';assert source.count(needle)==1
source=source.replace(needle,needle+'\nfrom frequency_prior import frequency_logs')
needle="prior=z['prior'].copy()";assert source.count(needle)==1
source=source.replace(needle,"prior=frequency_logs(z['representatives'],len(mask))")
assert not (HERE/'train.py').exists();(HERE/'train.py').write_text(source,encoding='utf-8')
freeze=read(ROOT/'outputs/grid50/code_freeze.json')
for p,h in freeze.items():assert sha(ROOT/p)==h,p
for p in list(HERE.glob('*.py'))+[OUT/'design.json',ROOT/'outputs/grid57_prior_review/review.json',ROOT/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz']:
    freeze[p.relative_to(ROOT).as_posix()]=sha(p)
    if p.suffix=='.py':py_compile.compile(str(p),doraise=True)
(OUT/'code_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8')
import importlib.util
spec=importlib.util.spec_from_file_location('grid58_freq',HERE/'frequency_prior.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
for reps in [[8,4,1],[902,51],[]]:
    logs=m.frequency_logs(reps,909);assert sorted(reps,key=lambda i:(-float(logs[i]),i))==sorted(reps)
(OUT/'preflight.json').write_text(json.dumps({'passed':True,'zero_residual_is_exact_static_order':True,'checks':3,'new_fits_planned':2,'new_simulations':0,'final8date_family_training_overlap':0},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN:two fixed frequency-prior GNN fits;no new teacher or rollout.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'fits':2,'only_prior_changed':True}))
