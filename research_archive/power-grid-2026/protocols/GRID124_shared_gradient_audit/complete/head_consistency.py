"""Descriptive heads-at-first-divergence check; classifier threshold never gates search."""
import diagnose as d
import gzip
np=d.np

def main():
    assert not (d.O/'head_consistency.json').exists();d.verify()
    d.save(d.O/'posthoc_head_plan.json',dict(posthoc=True,script_sha256=d.sha(d.W/'head_consistency.py'),rule='All changed saved trajectories, own auxiliary target, natural zero logit/0.5 probability diagnostic only. No threshold sweep or action substitution.',purpose='Separate auxiliary event mistakes from observed-profitable states recognized by event head but stopped by net head. These are realized outcomes,not proof of logically contradictory conditional expectations.'))
    Q=np.load(d.S/'predictions.npz');metas={p:d.read(d.S/f'{p}_metadata.json') for p in ['train','validation']};ix={p:{(m['sample_id'],m['position']):i for i,m in enumerate(mm)} for p,mm in metas.items()};folds=d.read(d.S/'design.json')['folds'];targets={p:np.load(d.S/f'{p}_data.npz') for p in metas};groups={};records=[]
    with gzip.open(d.O/'first_divergences.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=d.json.loads(line)
            if r['group']=='same_actions':continue
            split='full' if r['domain']=='validation' else next('f'+str(ff['fold']) for ff in folds if r['date'] in ff['held_dates']);part='validation' if split=='full' else 'train';i=ix[part][r['sample_id'],r['first_divergence_position']];q=Q[f'{split}__{r["arm"]}_seed{r["seed"]}__{part}'][i];label=bool(targets[part]['gain_event'][i] if r['arm']=='AUX_GAIN' else targets[part]['net'][i]>0);correct=bool((q[0]>0)==label);key='|'.join([r['domain'],r['arm'],r['group']])
            if key not in groups:groups[key]=dict(records=0,aux_correct=0,aux_positive=0,mass=0.,correct_mass=0.,probability_mass_sum=0.)
            g=groups[key];g['records']+=1;g['aux_correct']+=int(correct);g['aux_positive']+=int(q[0]>0);g['mass']+=r['date_case_seed_weight'];g['correct_mass']+=r['date_case_seed_weight']*correct;g['probability_mass_sum']+=r['date_case_seed_weight']*float(1/(1+np.exp(-float(q[0]))));records.append(dict(domain=r['domain'],arm=r['arm'],seed=r['seed'],sample_id=r['sample_id'],position=r['first_divergence_position'],group=r['group'],aux_logit=float(q[0]),predicted_net=float(q[1]),own_aux_label=label,aux_correct=correct))
    d.save(d.O/'head_consistency.json',dict(groups=groups,records=records,no_action_rule_changed=True,new_predictions=0))
    print('GRID124_HEAD_DIAGNOSIS',len(records),flush=True)

if __name__=='__main__':main()
