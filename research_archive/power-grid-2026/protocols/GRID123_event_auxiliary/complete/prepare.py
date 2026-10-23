"""Freeze new auxiliary labels, fixed task weights, and paired existing controls."""
from common123 import *

def main():
    assert not O.exists();O.mkdir(parents=True);configure();verify120();folds.verify();sealed();assert read(R/'outputs/grid122_false_stop_audit/review.json')['passed']
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cm={(c['sample_id'],j):(c,j) for c in cases for j in range(len(c['states']))};source=read(OLD/'design.json');checks={}
    for part in ['train','validation']:
        x,y,meta=data(part);g=[]
        for i,m in enumerate(meta):
            c,j=cm[m['sample_id'],m['position']];assert c['partition']==part;assert np.float32((c['gains'][j]-source['cost_per_query']*c['queries'][j])/source['gain_scale'])==y[i];g.append(c['gains'][j])
        g=np.asarray(g);event=(g>0).astype('f');prob=np.load(S/'train_targets.npz')['sampling_probability'] if part=='train' else np.full(len(y),1/len(y))
        np.savez_compressed(O/f'{part}_data.npz',X=x,net=y,gain_event=event,probability=prob);save(O/f'{part}_metadata.json',meta)
        checks[part]=dict(states=len(y),gain_events=int(event.sum()),net_positives=int((y>0).sum()),gain_event_calls=len({m['physical_state_group'] for i,m in enumerate(meta) if event[i]}),net_positive_calls=len({m['physical_state_group'] for i,m in enumerate(meta) if y[i]>0}))
    assert checks['train']['gain_events']==438 and checks['train']['net_positives']==211
    init=read(OLD/'preflight.json')['initial_q_hashes']
    for s in range(3):assert digest(build(s))==init[str(s)] and sum(p.numel() for p in build(s).parameters())==6978
    # Main gradient unchanged from the original two-head averaged regression.
    q=torch.tensor([[.2,-.1],[-.3,.7],[.1,.0]],requires_grad=True);yy=torch.tensor([.6,-.2,0.]);ev=torch.tensor([1.,1.,0.])
    original=torch.nn.functional.smooth_l1_loss(q,torch.stack([torch.zeros_like(yy),yy],1),beta=1.)
    original_grad=torch.autograd.grad(original,q)[0]
    for arm in ARMS:
        main,aux=components(q,yy,ev,arm);grad=torch.autograd.grad(main+aux,q)[0]
        assert torch.equal(original_grad[:,1],grad[:,1]);assert torch.isfinite(grad).all()
        expected=.5*(torch.sigmoid(q[:,0])-(ev if arm=='AUX_GAIN' else (yy>0).float()))/len(yy)
        assert torch.allclose(grad[:,0],expected,atol=1e-7)
    d=dict(id='GRID123_EVENT_AUXILIARY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User: 可以, approving proposed net-regression main task with improvement-event auxiliary supervision and same-capacity controls.',
        question='Does raw improvement-event auxiliary supervision reduce costly false stops while retaining net-value regression, beyond a net-sign auxiliary task?',
        architecture='Original41->64ReLU->64ReLU->2,6978parameters, exact original random tensors each seed. No new module/parameters/input.',
        heads=dict(main='q[:,1] predicts original normalized immediate net; continue iff q[:,1]>0,tieSTOP.',aux='q[:,0] is auxiliary logit; never used in action rule.'),
        groups=dict(NET_ZERO='Reuse original trained Huber models but read main net against exact STOP0; no refits.',AUX_NETSIGN='0.5*Huber(main,net)+0.5*BCE(aux,1[net>0]).',AUX_GAIN='0.5*Huber(main,net)+0.5*BCE(aux,1[raw gain>0]).',
            historical='Also show original deployed readout q1>q0,HUBER/DQN/BATCH1 separately; changing readout alone is measured, not hidden.'),
        matching='Both new arms share architecture/init/main-loss coefficient0.5/sample stream/optimizer/update budget; only auxiliary label differs. Compared with original Huber,replace zero-target stop-head loss withBCE. Equal half-task weights fixed before fit,not optimality or equal gradient-scale claim.',
        no_tuning='No class weights,aux coefficient sweep,scheduling,threshold/price/date/seed search; preserve all adverse seeds. Record component gradients at scheduled steps1/7250,not extra updates.',
        training=dict(seeds=[0,1,2],updates=7250,batch=32,lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,clip_grad_norm=10.,wall_cap_s=1800,
            fit_order=[(split,s,a) for split in ['full','f0','f1','f2','f3'] for s in range(3) for a in ARMS],new_fits=30,reused_backbones=15,endpoint='All30 fixed fits before any original validation or held-date evaluation.'),
        folds=read(CV/'design.json')['folds'],split='Original32train/8validation scenarios,16/4dates. Four fixed original-training date folds inheritedGRID121; each fold12fit/4held,groupskepttogether.',
        sampling='Full exact GRID107 stream; fold exact GRID121 stream and fold-local normalization. No warmstart across fits or pretrained checkpoints.',
        cost_per_query=source['cost_per_query'],gain_scale=source['gain_scale'],
        historical_task_dependency='AsGRID121: price/scale historically defined on16dates,heldfold results conditionalonfixedtask,not fully nested independent confirmation.',
        criteria=dict(primary='Complete-search four-validation-date macro net utility AND16date OOF utility.',
            incremental='AUX_GAIN>NET_ZERO family mean,all3pairedseed differences>0,>=3/4validationdates and>=12/16OOFdates positive.',
            event_specific='Same paired direction conditions for AUX_GAIN>AUX_NETSIGN; no attributing generic auxiliary benefit to raw-gain label.',
            practical='Also compare original deployed Huber,DQN andBATCH1. No physical/scaling/test even if all exploratory conditions pass.'),
        limits='Already exposed same-grid teacher-path dates/correlatedprefixes; instant gains not wholeweek control cost. No simulation/newdata/testmaterialization/install,paper or figure edit.')
    save(O/'design.json',d)
    save(O/'preflight.json',dict(passed=True,targets=checks,original_initial_hashes=init,main_output_gradient_matches_original=True,aux_logit_gradient_checked=True,parameters=6978,validation_model_predictions=0,optimizer_instances=0,new_fits=0,new_physical_calls=0,final_test_materialized=False))
    code=list(W.glob('*.py'))+[R/f'work/{n}' for n in ['grid120_decision_objective/common120.py','grid121_date_transfer/common121.py','grid115_matched_module_plan/matched_model.py','grid113_relational_learning/common.py','grid113_relational_learning/model.py','grid107_direct_value/common107.py','grid112_relational_plan/schema.py']]
    save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in code})
    files=[O/n for n in ['design.json','preflight.json','train_data.npz','validation_data.npz','train_metadata.json','validation_metadata.json']]
    files += [OLD/n for n in ['design.json','preflight.json','fits.json','state_predictions.jsonl.gz','validation_trajectories.json','validation_results.json']]+[OLD/f'models/direct_seed{s}_final.zip' for s in range(3)]
    files += [CV/n for n in ['design.json','metadata.json','train_arrays.npz','train_cases.pkl','predictions.npz','fits.json','preflight.json','oof_results.json','oof_trajectories.json']]
    files += [CV/f'fold{f}_normalization.npz' for f in range(4)]
    files += [CV/f'fold{f}_seed{s}_expected_exposure.npz' for f in range(4) for s in range(3)]
    files += [OLD/f'training_exposure_seed{s}.npz' for s in range(3)]
    files += [R/f'outputs/{n}' for n in ['grid102_search_rl/cases.pkl','grid100_data_foundation/design.json','grid122_false_stop_audit/label_factorization.json','grid122_false_stop_audit/review.json']]
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files});save(O/'state.json',dict(stage='qualified_before_push',new_fits=0));print('GRID123_QUALIFIED',json.dumps(checks),flush=True)
if __name__=='__main__':main()
