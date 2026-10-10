"""Preserved V2: conservative fixed-policy prior and continued PPO training."""
from pathlib import Path
import json,hashlib,math
ROOT=Path(__file__).resolve().parents[2];WORK=ROOT/'work/grid08';OUT=ROOT/'outputs/grid08'
target=WORK/'pilot_v2.py';assert not target.exists(),'Refuse to overwrite V2'
review=json.loads((OUT/'v1/training_review.json').read_text())
assert len(review['records'])==2 and all(r['complete_schedule'] for r in review['records'])
prior=.95
shifts={r['kind']:math.log(prior/(1-prior))-math.log(r['final_restore_probability_quantiles'][2]/(1-r['final_restore_probability_quantiles'][2])) for r in review['records']}
source=(WORK/'pilot_v1.py').read_text(encoding='utf-8')
source=source.replace("default='v1'","default='v2'",1)
source=source.replace("OUT/'v1/design.json'","OUT/'v2/design.json'",1)
old="model=ActorCritic(args.kind,2*env.n_sub,max_edges=2*env.n_line);optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])"
new="""model=ActorCritic(args.kind,2*env.n_sub,max_edges=2*env.n_line)
        saved=torch.load(OUT/'v1/train'/args.kind/'final.pt',weights_only=False)
        model.load_state_dict(saved['state'])
        with torch.no_grad():model.policy.actor.bias[1]+=D['warm_start']['bias_shifts'][args.kind]
        optimizer=torch.optim.Adam(model.parameters(),lr=CFG['learning_rate'])
        write_json(RUN/'warm_start.json',D['warm_start'])"""
assert source.count(old)==1
source=source.replace(old,new,1)
target.write_text(source,encoding='utf-8')
(OUT/'v2').mkdir(exist_ok=True)
D=json.loads((OUT/'v1/design.json').read_text());D['version']='v2'
D['training_epochs']=4;D['ppo']['learning_rate']=.00015
D['warm_start']={'source':'V1 final checkpoints','restore_prior_median':prior,'bias_shifts':shifts,
    'basis':'Training-only actor probability medians. Bias shift preserves input-dependent differences. Independent critic and all remaining learned weights are unchanged; Adam is reset.',
    'input_hashes':{kind:hashlib.sha256((OUT/'v1/train'/kind/'final.pt').read_bytes()).hexdigest() for kind in shifts}}
D['revision']={'changes':['Start near the qualified fixed restoration policy instead of a random policy.',
    'Four additional passes over the same four training weeks; unchanged order.',
    'Halve the learning rate for continued training; unchanged PPO clipping and entropy.'],
    'unchanged':['Legal information','Original controller and candidate actions','Safety screens','Reward and terminal penalty','Evaluation weeks'],
    'scope':'Development adaptation after V0/V1 diagnostics. No final-test use; all earlier versions retained. No claim these variants identify independent causal effects.'}
D['caps']['physical_steps']=220000
D['caps']['scope']='Cumulative V0 + V1 + V2; planned full-horizon maximum including qualification and failed attempts is 202,343 physical steps. Per-process wall limit remains enforced; verify actual bytes and counts at delivery.'
(OUT/'v2/design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=[target,WORK/'common.py',WORK/'networks_v1.py',OUT/'v2/design.json',OUT/'v1/training_review.json']
(OUT/'v2/code_freeze.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
print(json.dumps({'prior':prior,'bias_shifts':shifts,'generated_runner':str(target)}))
