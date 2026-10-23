"""Freeze design and qualify controlled input/architecture before twelve fixed fits."""
from common135 import *
import gzip,collections,datetime,copy,shutil
def main():
    assert not O.exists();O.mkdir(parents=True);configure();sealed()
    assert read(R/'outputs/grid134_full_tail_inputs/additional_review.json')['passed']
    with np.load(R/'outputs/grid129_baseline_risk/train_public_inputs.npz') as z:X=z['X'];ids=z['sample_ids'].tolist();Q=z['known_queries']
    with np.load(R/'outputs/grid130_calibrated_gate/train_calibration_scores.npz') as z:fit=z['fitmask'];cal=z['calmask']
    with gzip.open(R/'outputs/grid129_baseline_risk/train_label_ledger.jsonl.gz','rt',encoding='utf8') as f:labels=[json.loads(l) for l in f]
    assert all(i==r['sample_id'] for i,r in zip(ids,labels));ii=np.flatnonzero(fit&(Q>64));cc=np.flatnonzero(cal&(Q>64))
    old=read(R/'outputs/grid132_conditional_gate/training_started.json');assert old['fit_ids']==[ids[i] for i in ii] and old['calibration_ids']==[ids[i] for i in cc]
    weights=np.array(old['weights']);prob=weights/weights.sum();task=read(R/'outputs/grid102_search_rl/design.json')
    # Keep historical expected sample exposure rather than repeat7250 updates on a much smaller population.
    updates=math.ceil(7250*len(ii)/3582);assert updates==901
    design=dict(id='GRID135_MATCHED_FULL_TAIL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='qualified_before_training',
        authorization='User explicitly asks continue; prior autonomous research authorization and training announcement before start.',
        arms=ARMS,seeds=[0,1,2],new_fits=12,fit_order=[[arm,s] for s in range(3) for arm in ARMS],
        target='Binary complete locked-station completion gain>0. One gate, then original full completion or skip. Supervised classification,not RL.',
        scope='Same445 expensive fit/193 calibration cases from old12/4dates; original4exposed validation only after all weights/thresholds frozen.',
        features='Same37 summary information for all; normalized on445 fit cases only; last4 historical global slots zero. Original68 action and8 feedback descriptor. No date/id/unbought outcome as predictor.',
        extent='FIRST only next64 public plans; FULL all pending public plans (max237 in source train/cal). Same already-paid context,incumbent,known FULL remaining count/cost for all four arms.',
        association='JOINT keeps each action with its paid feedback;FACTOR averages feedback embeddings before joining action. Same5938 named parameters;scope changes inputs,not tensor count.',
        loss='Two-logit cross entropy,unweighted per minibatch;sampling probabilities equal original GRID132 fit weights normalized. No class reweighting.',
        normalization='Weighted centered variance from445fit cases,zero/nearzero std<1e-12=>1;clip[-20,20] inherited convention;no calibration fit.',
        budget=dict(updates=updates,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,max_grad_norm=10.,
            sampler_seed_offset=20261022,wall_cap_s=1800,checkpoint='final901 only,no checkpoint/seed selection',
            rationale='ceil(7250*445/3582)=901,approximately old expected draws per state;new task and sampling population differ,not a compute-matched comparison to old trials.'),
        fee=task['cost_per_query'],gain_scale=task['gain_scale'],
        calibration='For each model max s=P(zero gain) among positive-gain or different-action calibration witnesses;strict greater SKIP,ties fallback. Cheap branch reuses GRID132 cheap cutoff and GRID130 original scores.',
        model_comparisons='Primary range contrast FULL_JOINT-FIRST_JOINT;primary association contrast FULL_JOINT-FULL_FACTOR;FIRST pairing contrast and factorial interaction also reported.',
        mechanism_screen='For a contrast:both families no expensive action disagreements;all3paired macro query-saving deltas>0,and>=3/4date mean deltas>0. Report every seed/date and ties. Not significance.',
        practical_screen='For every seed,learned expensive branch has zero disagreements and eachdate gain loss/queryfraction no worse thanGROUP,with strict lower queries on at least1date. No automatic adoption/scale.',
        legacy_screen='Old zero total action mismatch remains reported separately,but fixed cheap branch already has7 known differences;not used as a purported achievable primary screen for this expensive-only intervention.',
        interpretation='All datasets exposed development;quality means local prediction-reward increments,not whole-week cost or AC safety. Same updates/labels/params not same training wall time.',
        prohibition=['morethan12fits','refit/loss/fee/threshold/seedsearch','new scenarios or labels','physical replay','final8test','paper/figure','automatic scale'])
    save('design.json',design)
    mean=np.sum(X[ii,:37].astype(float)*prob[:,None],0);var=np.sum((X[ii,:37]-mean)**2*prob[:,None],0);scale=np.sqrt(var);scale[scale<1e-12]=1
    np.savez_compressed(O/'normalization.npz',mean=mean,scale=scale);norm=dict(mean=mean,scale=scale)
    raw={}
    with gzip.open(R/'outputs/grid134_full_tail_inputs/train_calibration_public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for l in f:
            r=json.loads(l);raw[r['sample_id']]=r['public_input']
    for part,jj in [('fit',ii),('calibration',cc)]:
        save(f'{part}_metadata.json',[labels[i] for i in jj])
        np.savez_compressed(O/f'{part}_targets.npz',positive=np.array([labels[i]['full_baseline_gain']>0 for i in jj]),
            witness=np.array([labels[i]['full_baseline_gain']>0 or labels[i]['action_changed_if_skip'] for i in jj]),
            **(dict(probability=prob,weights=weights) if part=='fit' else {}))
        for extent in ['FIRST','FULL']:
            rr=[encoded(raw[ids[i]],X[i],norm,extent,design['fee'],design['gain_scale']) for i in jj]
            with (O/f'{part}_{extent}.pkl').open('wb') as f:pickle.dump(rr,f,protocol=4)
        first=pickle.loads((O/f'{part}_FIRST.pkl').read_bytes());full=pickle.loads((O/f'{part}_FULL.pkl').read_bytes())
        for a,b in zip(first,full):
            assert np.array_equal(a['next_action'],b['next_action'][:64]) and len(a['next_action'])==64
            assert all(np.array_equal(a[k],b[k]) for k in a if k!='next_action')
    initial={};streams={};checks={};effects={};rng=np.random.default_rng(91)
    synth=[dict(context_action=rng.normal(size=(n,68)).astype('f'),context_feedback=rng.normal(size=(n,8)).astype('f'),next_action=rng.normal(size=(q,68)).astype('f'),anchor_action=rng.normal(size=68).astype('f'),global41=rng.normal(size=41).astype('f'),known_query_cost=np.ones(1,'f')) for n,q in [(7,70),(3,66)]]
    inp=collate(synth)
    for s in range(3):
        nets=[net_for(a,s) for a in ARMS];assert len({digest(n) for n in nets})==1;assert all(sum(p.numel() for p in n.parameters())==5938 for n in nets);initial[str(s)]=digest(nets[0])
        rg=np.random.default_rng(20261022+s);h=hashlib.sha256();counts=np.zeros(445,np.int64)
        for _ in range(updates):
            jj=rg.choice(445,size=32,replace=True,p=prob);h.update(jj.astype('<i8').tobytes());np.add.at(counts,jj,1)
        streams[str(s)]=dict(sha256=h.hexdigest(),counts=counts.tolist(),draws=int(counts.sum()))
    for mode in ['JOINT','FACTOR']:
        net=net_for('FULL_'+mode,0);before=digest(net);q=net(**inp)
        def check(name,x):
            delta=float((net(**x)-q).abs().max());assert delta<1e-6,(mode,name,delta);checks[mode+'_'+name]=delta
        x={k:v.clone() for k,v in inp.items()}
        for k in ['context_action','context_feedback','context_mask']:x[k]=x[k].flip(1)
        check('paired_context_permutation',x)
        x={k:v.clone() for k,v in inp.items()}
        for k in ['next_action','next_mask']:x[k]=x[k].flip(1)
        check('all_candidate_permutation',x)
        x={k:v.clone() for k,v in inp.items()}
        for k,mask in [('context_action','context_mask'),('context_feedback','context_mask'),('next_action','next_mask')]:x[k][~x[mask]]=float('nan')
        check('poisoned_padding',x)
        x={k:v.clone() for k,v in inp.items()};x['context_feedback'][0,:7]=x['context_feedback'][0,:7].flip(0)
        delta=float((net(**x)-q).abs().max());effects[mode]=delta
        assert delta<1e-6 if mode=='FACTOR' else delta>1e-8
        torch.nn.functional.cross_entropy(q,torch.tensor([0,1])).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters());assert digest(net)==before;net.zero_grad(set_to_none=True)
    save('preflight.json',dict(passed=True,parameters_each=5938,fit_cases=445,calibration_cases=193,initial_hashes=initial,sampling_streams=streams,
        invariance_errors=checks,pairing_counterexample=effects,synthetic_backward_only=True,optimizer_steps=0,full_vs_first_only_candidate_extent_differs=True,
        known_full_cost_equal=True,labels_separate_from_public_tensors=True,final_tests_absent=True))
    # Freeze sources for future validation construction without evaluating their outcomes.
    with np.load(R/'outputs/grid130_calibrated_gate/validation_scores.npz') as z:valid_ids=set(z['sample_ids'].tolist())
    meta=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples'];valid_meta=[m for m in meta if m['sample_id'] in valid_ids]
    assert len(valid_meta)==590;save('validation_source_metadata.json',valid_meta)
    paths=[R/'outputs/grid134_full_tail_inputs/train_calibration_public_inputs.jsonl.gz',R/'outputs/grid134_full_tail_inputs/additional_review.json',R/'outputs/grid134_full_tail_inputs/dataset_receipt.json',R/'outputs/grid129_baseline_risk/train_public_inputs.npz',R/'outputs/grid129_baseline_risk/train_label_ledger.jsonl.gz',R/'outputs/grid130_calibrated_gate/train_calibration_scores.npz',R/'outputs/grid130_calibrated_gate/validation_scores.npz',R/'outputs/grid132_conditional_gate/fits.json',R/'outputs/grid132_conditional_gate/validation_results.json',R/'outputs/grid132_conditional_gate/validation_trajectories.json',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/design.json',R/'outputs/grid03/scoring_smoke/agent_episode/dict_action_space.json',R/'outputs/grid03/scoring_smoke/agent_episode/dict_observation_space.json',R/'outputs/grid28/offline.json',R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz',R/'outputs/grid98_feature_enrichment/template_metadata.json']
    for name in sorted({m['source_call'] for m in valid_meta}):
        p=R/name;paths += [p,p.with_name(p.stem+'_state.npz'),p.parent/'metadata.json']
    paths += [R/'outputs/grid133_nonlinear_gate/validation_results.json']
    paths+=list(O.glob('*.json'))+list(O.glob('*.npz'))+list(O.glob('*.pkl'))
    save('source_hashes.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    code=list(W.glob('*.py'))+[R/'work/grid113_relational_learning/model.py',R/'work/grid113_relational_learning/common.py',R/'work/grid112_relational_plan/schema.py',R/'work/grid111_action_feedback/public.py',R/'work/grid111_action_feedback/common111.py',R/'work/grid107_direct_value/common107.py']
    save('code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code});save('state.json',dict(stage='qualified_awaiting_push',fits=0));verify()
    d=A/'frozen';d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),O/'design.json',O/'preflight.json',O/'source_hashes.json',O/'code_freeze.json',O/'normalization.npz']:shutil.copy2(p,d/p.name)
    (d/'archive_manifest.json').write_text(json.dumps({p.name:sha(p) for p in d.iterdir()},indent=2)+'\n',encoding='utf8')
    print('GRID135_QUALIFIED: 12fits planned,901updates each,5938params,zero optimizer steps',flush=True)
if __name__=='__main__':main()
