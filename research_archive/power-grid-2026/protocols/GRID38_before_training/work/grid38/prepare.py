"""One target-alignment revision, preserving the original and matched MLP ablation."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid38'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
old=read(ROOT/'outputs/grid21/design.json');a=read(ROOT/'outputs/grid_target_audit/audit.json')
assert a['valid_rows']==148 and a['target_peak_matches_reward']==148
OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'design.json').exists()
D={'stage':'GRID38_ONE_TARGET_ALIGNMENT_REVISION','training_weeks':old['training_weeks'],'development_weeks':old['development_weeks'],
   'change':'Only supervised target changes: q_new=.5*q_original+.5*Uniform(argmax FP32 reward among canonical strictly admissible candidates). Retain all exact reward ties; do not choose a label using arbitrary pool ID. Invalid candidates receive zero target mass.',
   'hypothesis':'Broad soft target may dilute accurate-best-action ranking for a simulator-screened search. Concentrating half the target on reward-optimal classes may improve early recovery of the exhaustive choice, while retaining half the original multi-action soft supervision. This is a testable model configuration, not novel learning theory or evidence that soft targets are wrong.',
   'unchanged':'Existing167 cached training states/148 usable, legal feature arrays and public prior, same GNN/MLP architecture, exact original initial weights, residual bound2, KLweight0.1,64epochs,batch8,Adam and shuffle seeds. No new teacher queries,RL,cutoff/stopping/guard changes.',
   'seeds':[0,1],'kinds':['gnn','mlp'],'epochs':old['epochs'],'batch_size':old['batch_size'],'lr':old['lr'],'weight_decay':old['weight_decay'],'prior_KL_weight':old['prior_KL_weight'],'mixture':.5,
   'evaluation':'After all4 fits and final checkpoints are saved, re-infer the same26 exposed FULL-state cached feedback, compare original/new GNN and original/new MLP under unchanged semantic stop. Keep both seeds and all results. Known1733 safety regression blocks unchanged dispatch; no selecting the best seed/checkpoint. Actual closed-loop quality/time must then be measured, offline ranking is not a deployment claim.',
   'scope':'Development iteration only; two dev dates overlap oldtraining; author PPO original training identity unknown. Final8 date families remain sealed. No sweeps or additional cheap-rule/algorithm portfolio expansion.',
   'references':[{'url':'https://arxiv.org/html/2503.15190v2','use':'Soft multi-action imitation and simulation screening are existing approaches; paper cautions single hard-label imitation. Keep multiple candidates rather than assuming hard labels superior.'},
      {'url':'https://arxiv.org/html/2407.19865v2','use':'Imitation plus small simulation budgets has already been studied; our target revision alone is not a paper contribution.'},
      {'url':'https://arxiv.org/html/2604.01830v1','use':'GNN risk prior, TopK and reweighted RL policies exist; do not claim GNN+RL/prior composition itself as new.'}],
   'caps':{'training_s':1200,'fits':4,'new_forecasts':0,'reserved_evaluations':0},
   'execution':'Run training only after the GRID37 serial timing comparison has finished; avoid training workload concurrent with benchmark workers.'}
(OUT/'design.json').write_text(json.dumps(D,indent=2),encoding='utf-8')
files=list((ROOT/'work/grid38').glob('*.py'))+[ROOT/p for p in read(ROOT/'outputs/grid21/code_freeze.json')]
for p in sorted((ROOT/'outputs/grid21/inputs/train').glob('*')):
    if p.is_file():files.append(p)
for kind in D['kinds']:
    for seed in D['seeds']:files += [ROOT/f'outputs/grid21/models/{kind}_seed{seed}/{name}.pt' for name in ['initial','final']]
files += [ROOT/r['path'] for r in a['source_hashes']]
files.append(ROOT/'outputs/grid21/training_finished.json')
(OUT/'code_freeze.json').write_text(json.dumps({p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2),encoding='utf-8')
(OUT/'RUN_STATE.md').write_text('FROZEN; wait for GRID37 before4matched tiny fits. No final dates or RL.\n',encoding='utf-8')
print(json.dumps({'frozen':True,'fits':4,'target_only_revision':True}))
