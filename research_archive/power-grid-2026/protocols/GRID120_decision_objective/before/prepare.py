"""Freeze two classification objectives and matched initialization/exposure."""
import datetime
from common120 import *

def main():
    assert not O.exists(),'Preserve historical protocols';O.mkdir(parents=True);configure();verify113();sealed()
    assert read(R/'outputs/grid119_fixed_base_learning/review.json')['passed']
    old=read(R/'outputs/grid107_direct_value/design.json');x,y,meta=data('train');p=np.load(S/'train_targets.npz')['sampling_probability']
    Z=float(p@abs(y).astype(float));assert Z>0
    # Exact utility regret reduction and the conditional logistic optimum, not a new label.
    decisions=np.array([0,1],dtype=int)[:,None];oracle=np.maximum(y,0)
    regret=oracle-decisions*y
    ref=np.abs(y)*((decisions!=(y>0)).astype(int));assert np.array_equal(regret,ref)
    pos=float(p@np.maximum(y,0));neg=float(p@np.maximum(-y,0));prior=float(p@(y>0))
    analytical=dict(weight_normalizer=Z,positive_weight_mass=pos/Z,negative_weight_mass=neg/Z,
        original_positive_probability=prior,expected_net=float(p@y),
        unweighted_margin_derivative_at_zero=.5-prior,
        utility_weighted_margin_derivative_at_zero=-.5*float(p@y)/Z,
        interpretation='Descriptive train-distribution intercept only; no guarantee of conditional fit or validation benefit.')
    # Stable logistic, analytic derivative and autograd agree on synthetic outputs; no optimizer.
    q=torch.tensor([[.2,-.1],[-.3,.7],[1.,1.]],requires_grad=True);yy=torch.tensor([.6,-.2,0.])
    for arm in ARMS:
        q.grad=None;ll=loss_per_state(q,yy,arm,Z);ll.mean().backward()
        margin=(q[:,1]-q[:,0]).detach().numpy().astype(float);t=(yy.numpy()>0).astype(float);weights=abs(yy.numpy())/Z if arm=='UTILITY_BCE' else np.ones(3)
        independent=(np.logaddexp(0.,margin)-t*margin)*weights
        assert np.allclose(ll.detach().numpy(),independent,atol=1e-6,rtol=1e-6)
        g=(1/(1+np.exp(-margin))-t)*weights/3
        assert np.allclose(q.grad.numpy(),np.column_stack([-g,g]),atol=1e-6)
    match=read(R/'outputs/grid107_direct_value/preflight.json');streams=read(S/'preflight.json')['sampling_streams'];initial={}
    for seed in range(3):
        m=build(seed);assert digest(m)==match['initial_q_hashes'][str(seed)] and sum(a.numel() for a in m.parameters())==6978
        a=build(seed);assert digest(a)==digest(m);initial[str(seed)]=digest(m)
        rng=np.random.default_rng(20261022+seed);h=hashlib.sha256();counts=np.zeros(len(y),np.int64)
        for _ in range(7250):
            ids=rng.choice(len(y),32,replace=True,p=p);h.update(ids.astype('<i8').tobytes());np.add.at(counts,ids,1)
        assert h.hexdigest()==streams[str(seed)]['index_sha256']
        assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))};norm=np.load(R/'outputs/grid102_search_rl/normalization.npz');checked={}
    for part in ['train','validation']:
        xx,yy,mm=data(part)
        for i,row in enumerate(mm):
            c,j=cm[row['sample_id'],row['position']];assert c['partition']==part
            assert np.float32((c['gains'][j]-old['cost_per_query']*c['queries'][j])/old['gain_scale'])==yy[i]
            assert np.array_equal(xx[i],np.clip((c['states'][j]-norm['mean'])/norm['scale'],-20,20).astype('f'))
        checked[part]=len(yy)
    d=dict(id='GRID120_DECISION_OBJECTIVE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User authorizes further learning experiments and more time; execute one bounded objective comparison before adding architectures or data.',
        question='Does learning cost-weighted stop/continue decisions improve complete-search net utility beyond sign classification and original Huber value regression?',
        arms=ARMS,reuse='Original GRID107 Huber seeds0/1/2 and historical gamma0DQN; no repeated fits.',
        architecture='Original41->64ReLU->64ReLU->2,6978parameters, exact original random initial tensors per seed.',
        labels='t=1[y>0], where y is original immediate normalized net continuation. Zero ties STOP.',
        objectives=dict(SIGN_BCE='BCEWithLogits(q1-q0,t),mean over32states.',
            UTILITY_BCE='abs(y)/Z * BCEWithLogits(q1-q0,t),mean over32states. Z=original-training-probability weighted E|y|,fixed once; not per-batch normalization.',
            normalizer=Z,weight_clipping='None. Existing norm10 gradient clipping retained, clipping counted.',
            output='Two logits, not calibrated action values. Continue iff q1>q0; no threshold tuning.'),
        proof='For any binary action a, max(y,0)-a*y=|y|*1[a !=1[y>0]]. For fixed x, weighted-logistic p*=E[y+|x]/E[|y||x] when denominator>0; p*>.5 iff E[y|x]>0. Population/unrestricted-function statement, not finite-data or closed-loop guarantee.',
        loss_scope='Changes decision target/objective deliberately; no price/physical reward/dates or inputs changed. Compare new two-output classification to old two-output regression by identical updates/states/initialization, not identical scalar supervision or gradient scale.',
        train=dict(seeds=[0,1,2],updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=10.,cpu_threads=1,
            fit_order=[(a,s) for s in range(3) for a in ARMS],new_fits=6,total_update_cap=43500,total_wall_cap_s=1800,
            endpoint='Final only; all six fits before any validation prediction; all seeds retained.',
            sampling='Exact original7250x32 stream each seed, same original probability; no oversampling or validation-selected weights.'),
        split=dict(train_scenarios=32,validation_scenarios=8,train_dates=16,validation_dates=4,train_prefixes=2378,validation_prefixes=590,train_states=3582,validation_states=856),
        cost_per_query=old['cost_per_query'],gain_scale=old['gain_scale'],
        gates=dict(primary='Four-date macro complete-sequence net utility.',
            objective_signal='UTILITY family mean>SIGN; all3paired seed differences>0 and at least3/4date differences>0.',
            useful_candidate='Objective signal AND UTILITY family mean above original Huber,DQN0 andBATCH1. No automatic physical replay/scaling.',
            diagnostics=['gain capture and query fraction','positive misses/unprofitable continues','physical-call-balanced utility','training vs validation utility and cost-sensitive regret','gradient clipping and all seed/date results']),
        literature=dict(paper='Zadrozny, Langford and Abe, Cost-Sensitive Learning by Cost-Proportionate Example Weighting, ICDM2003.',url='https://hunch.net/~jl/projects/reductions/costing/finalICDM2003.pdf',
            used='Example-dependent misclassification costs and transparent weighting; not claiming our exact BCE implementation or wholeweek guarantee is in that paper.'),
        prior_evidence='GRID108 MSE/longer training failures retained; GRID114 Huber-vs-mean mismatch; GRID119 fixed corrections not above original. This tests decision objective without more network/input.',
        limits='Exposed4dates/samegrid/FULLteacherpath; correlatedprefixes, immediate gain not wholeweek value. No environment, forecast, finaltest, installation, fee/seed/threshold search.',
        roadmap='After this bounded screen, assess whether objectives, teacher-path transfer or sequential credit needs work; do not run unplanned algorithms based on best validation outcome.')
    save(O/'design.json',d);save(O/'training_distribution.json',analytical)
    save(O/'preflight.json',dict(passed=True,original_initial_hashes=initial,sampling_streams=streams,states_checked=checked,
        exact_regret_identity=True,logistic_derivatives_checked=True,train_only_weight_normalizer=True,optimizer_instances=0,optimizer_steps=0,new_fits=0,validation_predictions=0,new_physical_calls=0,final_test_materialized=False))
    code=list(W.glob('*.py'))+[R/f'work/{n}' for n in ['grid115_matched_module_plan/matched_model.py','grid113_relational_learning/model.py','grid113_relational_learning/common.py','grid107_direct_value/common107.py','grid112_relational_plan/schema.py']]
    save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code})
    files=[O/'design.json',O/'preflight.json',O/'training_distribution.json']+[S/f'{part}_{n}' for part in ['train','validation'] for n in ['public.pkl','metadata.json','targets.npz']]
    files+=[R/f'outputs/grid107_direct_value/{n}' for n in ['design.json','preflight.json','fits.json','state_predictions.jsonl.gz','validation_results.json','validation_trajectories.json']]
    files+=[R/f'outputs/grid107_direct_value/training_exposure_seed{s}.npz' for s in range(3)]
    files+=[R/f'outputs/{n}' for n in ['grid102_search_rl/cases.pkl','grid102_search_rl/normalization.npz','grid100_data_foundation/design.json','grid108_loss_budget/decision.json','grid114_fit_diagnosis/decision.json','grid119_fixed_base_learning/decision.json']]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});save(O/'state.json',dict(stage='qualified_before_push',new_fits=0))
    print('GRID120_QUALIFIED',json.dumps(analytical),flush=True)
if __name__=='__main__':main()
