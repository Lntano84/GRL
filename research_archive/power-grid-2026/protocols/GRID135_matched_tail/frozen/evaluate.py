"""After all12fits/cutoffs frozen, build public validation contexts and audit decisions."""
from common135 import *
import collections,gzip,copy
from sklearn.metrics import roc_auc_score
def summary(rows):
    per={}
    for date in sorted({r['date'] for r in rows}):
        rr=[r for r in rows if r['date']==date];G=math.fsum(r['full_gain'] for r in rr);Q=sum(r['full_queries'] for r in rr);qp=sum(r['queries'] for r in rr)/Q;loss=math.fsum(r['gain_lost'] for r in rr)/G if G else 0.
        per[date]=dict(gain_loss=loss,gain_capture=1-loss,query_fraction=qp,query_saving=1-qp,utility=float(np.mean([r['net_utility'] for r in rr])),action_disagreements=sum(r['action_disagreement'] for r in rr))
    return dict(per_date=per,macro={k:float(np.mean([v[k] for v in per.values()])) for k in ['gain_loss','gain_capture','query_fraction','query_saving','utility']},
        total_action_disagreements=sum(r['action_disagreement'] for r in rows),expensive_disagreements=sum(r['expensive'] and r['action_disagreement'] for r in rows),expensive_skipped=sum(r['expensive'] and r['queries']==0 for r in rows))
def main():
    verify();configure();assert not (O/'evaluation_started.json').exists();assert read(O/'state.json')['stage']=='all12models_and_cutoffs_frozen_before_validation'
    for n,h in read(O/'evaluation_freeze.json').items():assert sha(O/n)==h
    save('evaluation_started.json',dict(all12weights_cutoffs_frozen=True))
    D=read(O/'design.json');F=read(O/'fits.json');cases=[c for c in pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes()) if c['partition']=='validation'];assert len(cases)==590
    cm={c['sample_id']:c for c in cases};exp=np.array([int(c['queries'].sum())>64 for c in cases]);exp_cases=[c for c,e in zip(cases,exp) if e]
    idx={c['sample_id']:i for i,c in enumerate(exp_cases)};fullq={c['sample_id']:int(c['queries'].sum()) for c in exp_cases}
    norm=dict(np.load(O/'normalization.npz'));th=read(O/'thresholds.json');old={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for l in f:
            r=json.loads(l)
            if r['sample_id'] in idx and r['position']==0:old[r['sample_id']]=r['public_input']
    sys.path.insert(0,str(R/'work/grid111_action_feedback'))
    from public import Decoder,best
    dec=Decoder();bycall=collections.defaultdict(list)
    for m in read(O/'validation_source_metadata.json'):
        if m['sample_id'] in idx:bycall[m['source_call']].append(m)
    raw={};publicchecks=0
    for path,ms in bycall.items():
        decoded=dec.decode(path);call=decoded[0];truth={r['pool_id']:r for r in call['outcomes']};aliases=decoded[-1]
        for m in ms:
            sid=m['sample_id'];n=m['visible_count'];visible=[truth[i] for i in call['order'][:n]];station=dec.subs[best(visible)['pool_id']];target={v for k,v in aliases.items() if dec.subs[k]==station};pending=[i for i in call['order'][n:] if i in target]
            assert len(pending)==fullq[sid];c=dec.context(decoded,visible,pending,station);prev=old[sid]
            assert all(c[k]==prev[k] for k in ['seen_ids','seen_plans','seen_feedback','incumbent_id','incumbent_reward','incumbent_plan'])
            assert c['next_ids'][:64]==prev['next_ids'] and c['next_plans'][:64]==prev['next_plans'];raw[sid]=c;publicchecks+=1
    assert len(raw)==len(exp_cases)
    public={extent:[encoded(raw[c['sample_id']],c['states'][0],norm,extent,D['fee'],D['gain_scale']) for c in exp_cases] for extent in ['FIRST','FULL']}
    for extent,rr in public.items():
        with (O/f'validation_{extent}.pkl').open('wb') as f:pickle.dump(rr,f,protocol=4)
    save('validation_metadata.json',[dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group']) for c in exp_cases])
    y=np.array([float(c['gains'].sum())>0 for c in exp_cases]);np.savez_compressed(O/'validation_targets.npz',positive=y)
    with np.load(R/'outputs/grid130_calibrated_gate/validation_scores.npz') as z:oldscore=z['score'];oldids=z['sample_ids']
    assert all(str(sid)==c['sample_id'] for sid,c in zip(oldids,cases));cheap_th=read(R/'outputs/grid132_conditional_gate/fits.json')['thresholds']['GROUP']['cheap']
    base=read(R/'outputs/grid133_nonlinear_gate/validation_results.json');results=dict(base);trajectories=[];logits={};diagnostics={};practical={};traincal=np.load(O/'train_calibration_logits.npz')
    yy={part:np.load(O/f'{part}_targets.npz')['positive'] for part in ['fit','calibration']};trainpublic={part:{e:pickle.loads((O/f'{part}_{e}.pkl').read_bytes()) for e in ['FIRST','FULL']} for part in ['fit','calibration']}
    auditerror=0.;numchecks=0;P=read(O/'preflight.json')
    for rec in F['records']:
        name=rec['name'];arm=rec['arm'];seed=rec['seed'];extent=arm.split('_')[0];mode=arm.split('_')[1];path=O/'models'/f'{name}.pt';assert sha(path)==rec['checkpoint_sha256']
        cp=torch.load(path,weights_only=False,map_location='cpu');assert cp['updates']==901 and all(int(v['step'])==901 for v in cp['optimizer']['state'].values());assert cp['initial_hash']==P['initial_hashes'][str(seed)]
        net=net_for(arm,seed);net.load_state_dict(cp['state_dict']);assert digest(net)==rec['final_hash'];va=predict(net,public[extent]);logits[name]=va;score=oldscore.copy();score[exp]=no_gain(va)
        newrows=[]
        for i,c in enumerate(cases):
            G=math.fsum(float(v) for v in c['gains']);Q=int(c['queries'].sum());cut=th[name] if exp[i] else cheap_th;skip=score[i]>cut;p=0 if skip else len(c['gains']);gain=0. if skip else G;paid=0 if skip else Q
            r=dict(sample_id=c['sample_id'],date=c['date'],call=c['physical_state_group'],method=name,expensive=bool(exp[i]),batches=p,gain=gain,gain_lost=G-gain,queries=paid,full_gain=G,full_queries=Q,chosen_id=c['chosen'][p],baseline_id=c['chosen'][-1],action_disagreement=c['chosen'][p]!=c['chosen'][-1],net_utility=(gain-D['fee']*paid)/D['gain_scale'],no_gain_score=float(score[i]),cutoff=float(cut))
            newrows.append(r)
        trajectories+=newrows;v=summary(newrows);results[name]=v
        g=base['GROUP'];dominates=all(v['per_date'][d]['gain_loss']<=g['per_date'][d]['gain_loss']+1e-12 and v['per_date'][d]['query_fraction']<=g['per_date'][d]['query_fraction']+1e-12 for d in v['per_date']) and any(v['per_date'][d]['query_fraction']<g['per_date'][d]['query_fraction']-1e-12 for d in v['per_date'])
        practical[name]=dict(expensive_zero_disagreement=v['expensive_disagreements']==0,perdate_joint_improvement_vs_group=bool(dominates),conditional_screen=bool(v['expensive_disagreements']==0 and dominates),legacy_zero_total_disagreements=v['total_action_disagreements']==0)
        diagnostics[name]={}
        weights={k:v.numpy().astype(float) for k,v in cp['state_dict'].items()};assert sum(v.size for v in weights.values())==5938
        for part,rr,truth,saved in [('fit',trainpublic['fit'][extent],yy['fit'],traincal[name+'__fit']),('calibration',trainpublic['calibration'][extent],yy['calibration'],traincal[name+'__calibration']),('validation',public[extent],y,va)]:
            ss=no_gain(saved);auc=float(roc_auc_score(truth,1-ss));diagnostics[name][part]=dict(prefixes=len(rr),auc=auc,no_gain_score_min=float(ss.min()),no_gain_score_max=float(ss.max()))
            selected=sorted(set([0,len(rr)-1]+list(range(0,len(rr),73))+list(np.flatnonzero(truth))))
            for i in selected:
                q=numpy_forward(weights,rr[i],mode);err=float(abs(q-saved[i]).max());auditerror=max(auditerror,err);assert np.allclose(q,saved[i],atol=3e-5,rtol=3e-5),(name,part,i,err);numchecks+=1
        # Predeclared binding diagnostic, not a new policy/readout or selection criterion.
        shifted=[]
        for row in public[extent]:
            v={k:a.copy() for k,a in row.items()};v['context_feedback']=np.roll(v['context_feedback'],1,axis=0);shifted.append(v)
        shiftedscore=no_gain(predict(net,shifted));original=no_gain(va);diagnostics[name]['binding_shift']=dict(max_abs_probability_change=float(abs(shiftedscore-original).max()),calibrated_decision_changes=int(((shiftedscore>th[name])!=(original>th[name])).sum()),positive_case_decision_changes=int((((shiftedscore>th[name])!=(original>th[name]))&y).sum()))
        exposure=np.load(O/f'exposure_{name}.npz');assert exposure['draws'].tolist()==P['sampling_streams'][str(seed)]['counts'];assert len(exposure['losses'])==901
    # Entire repeated-seed families are reported; no best seed selection.
    families={}
    for arm in ARMS:
        vv=[results[f'{arm}_seed{s}'] for s in range(3)];families[arm]=dict(macro={k:float(np.mean([v['macro'][k] for v in vv])) for k in vv[0]['macro']},total_disagreements_by_seed=[v['total_action_disagreements'] for v in vv],expensive_disagreements_by_seed=[v['expensive_disagreements'] for v in vv],all_seeds_conditional_screen=all(practical[f'{arm}_seed{s}']['conditional_screen'] for s in range(3)))
    contrasts={}
    for label,a,b in [('range_primary','FULL_JOINT','FIRST_JOINT'),('association_primary','FULL_JOINT','FULL_FACTOR'),('range_factor','FULL_FACTOR','FIRST_FACTOR'),('association_first','FIRST_JOINT','FIRST_FACTOR')]:
        paired=[results[f'{a}_seed{s}']['macro']['query_saving']-results[f'{b}_seed{s}']['macro']['query_saving'] for s in range(3)]
        dates={d:float(np.mean([results[f'{a}_seed{s}']['per_date'][d]['query_saving']-results[f'{b}_seed{s}']['per_date'][d]['query_saving'] for s in range(3)])) for d in results[f'{a}_seed0']['per_date']}
        protected=all(results[f'{z}_seed{s}']['expensive_disagreements']==0 for z in [a,b] for s in range(3));signal=protected and all(v>1e-12 for v in paired) and sum(v>1e-12 for v in dates.values())>=3
        contrasts[label]=dict(a=a,b=b,query_saving_deltas_by_seed=paired,perdate_seed_mean_deltas=dates,both_families_expensive_zero_disagreement=protected,mechanism_signal=bool(signal))
    interaction=[contrasts['association_primary']['query_saving_deltas_by_seed'][i]-contrasts['association_first']['query_saving_deltas_by_seed'][i] for i in range(3)]
    # Recompute every saved action,query and tail gain; cheap branch exactly reproduces original GLOBAL.
    oldrows={r['sample_id']:r for r in read(R/'outputs/grid132_conditional_gate/validation_trajectories.json') if r['method']=='GLOBAL'}
    for r in trajectories:
        c=cm[r['sample_id']];p=r['batches'];assert r['chosen_id']==c['chosen'][p] and r['queries']==sum(int(v) for v in c['queries'][:p]);assert abs(r['gain_lost']-math.fsum(float(v) for v in c['gains'][p:]))<1e-12
        assert (r['no_gain_score']>r['cutoff'])==(p==0)
        if not r['expensive']:assert all(r[k]==oldrows[r['sample_id']][k] for k in ['queries','chosen_id','action_disagreement'])
    save('validation_results.json',results);save('validation_trajectories.json',trajectories);save('families.json',families);save('diagnostics.json',diagnostics)
    np.savez_compressed(O/'validation_logits.npz',**logits)
    decision=dict(contrasts=contrasts,factorial_interaction_by_seed=interaction,practical_by_model=practical,adopted=False,ready_to_scale=False,
        reason='Exposed local development;new models not automatically deployed even if conditional screen passes. Existing7cheap discrepancies kept separate.')
    save('decision.json',decision);verify()
    review=dict(passed=True,fits=12,all_final901_optimizer_states_checked=True,matched_initializations_and_draws=True,parameters_each5938=True,
        all_models_and_cutoffs_frozen_before_new_validation=True,validation_expensive_public_contexts=publicchecks,validation_source_calls=len(bycall),
        independent_numpy_forwards=numchecks,max_abs_forward_error=auditerror,all_saved_action_query_gain_checks=len(trajectories),cheap_branch_exactly_replayed=True,
        final_tests_absent=True,new_physical_calls=0,new_training_data=0,all_seeds_adverse_dates_retained=True,source_hashes_unchanged=True)
    save('review.json',review);save('state.json',dict(stage='audited_awaiting_publication',fits=12));print(json.dumps(dict(families=families,contrasts=contrasts,review=review),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
