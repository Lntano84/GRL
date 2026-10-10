"""Training-only diagnostics. Never open validation/test trajectories or labels."""
from common import *
import argparse,gzip

ap=argparse.ArgumentParser();ap.add_argument('--version',default='v0');args=ap.parse_args()
if args.version in ['v1','v2']:
    from networks_v1 import ActorCritic
records=[]
for kind in ['gnn','mlp']:
    folder=OUT/args.version/'train'/kind
    checkpoint=folder/'final.pt'
    if not (folder/'finished.json').exists():
        manifest=folder/'manifest.json'
        if not manifest.exists():continue
        count=len(json.loads(manifest.read_text()))
        if not count:continue
        # Manifest is atomically persisted after checkpoint writing; do not
        # open a newer checkpoint while the live runner is still writing it.
        n_weeks=len(json.loads((OUT/'design.json').read_text())['training_weeks'])
        checkpoint=folder/f'epoch{(count-1)//n_weeks}_week{(count-1)%n_weeks}.pt'
    saved=torch.load(checkpoint,weights_only=False)
    model=ActorCritic(kind,saved['n_nodes'],max_edges=saved['max_edges']);model.load_state_dict(saved['state']);model.eval()
    summaries=[];predictions=[];label_agreement=[];teacher_labels=[];choices=[]
    for path in sorted(folder.glob('e*_*')):
        if not (path/'summary.json').exists():continue
        summaries.append(json.loads((path/'summary.json').read_text()))
        with gzip.open(path/'steps.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                row=json.loads(line)
                if row['proposal']['offered']:choices.append(row['choice'])
        file=path/'teacher.npz'
        if not file.exists():continue
        with np.load(file) as z:
            for i,truth in enumerate(z['actions']):
                with torch.no_grad():
                    logits,_=model(*tensor_features((z['x'][i],z[f'e{i}'],z['g'][i])))
                    p=float(logits.softmax(-1)[0,1]);prediction=int(logits.argmax(-1).item())
                predictions.append(p);teacher_labels.append(int(truth));label_agreement.append(prediction==int(truth))
    updates=[]
    if (folder/'updates.jsonl').exists():
        updates=[json.loads(l) for l in (folder/'updates.jsonl').read_text().splitlines()]
    records.append({'kind':kind,'checkpoint':str(checkpoint.relative_to(ROOT)),
        'complete_schedule':(folder/'finished.json').exists(),'completed_episodes':len(summaries),
        'training_episodes':summaries,'offered_samples':len(predictions),
        'training_selected_fraction':float(np.mean(choices)) if choices else None,
        'final_restore_probability_quantiles':np.quantile(predictions,[0,.1,.5,.9,1]).tolist() if predictions else [],
        'final_deterministic_restore_fraction':float(np.mean(np.asarray(predictions)>.5)) if predictions else None,
        'immediate_teacher_restore_fraction':float(np.mean(teacher_labels)) if teacher_labels else None,
        'immediate_teacher_agreement':float(np.mean(label_agreement)) if label_agreement else None,
        'rollout_offered_fraction':sum(u['offered'] for u in updates)/sum(u['samples'] for u in updates) if updates else None,
        'updates':len(updates),'optimizer_wall_s':sum(u['wall_s'] for u in updates)})
report={'scope':'Training-only behavior diagnostics; immediate-cost teacher is a legal heuristic, not an optimal long-horizon label. Samples from repeated episodes are correlated.',
    'records':records}
write_json(OUT/args.version/'training_review.json',report)
print(json.dumps(report,default=json_default,indent=2))
