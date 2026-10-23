"""Match concrete protocol and verify random initialization only. No backward/optimizer/fit."""
from pathlib import Path
import sys,json,hashlib,datetime,inspect,collections,os
R=Path('D:/Codex/GRL研究');W=Path(__file__).parent;O=R/'outputs/grid115_matched_module_plan';REPO=R/'work/GRL-results-integration'
sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from common import read,save,sha,configure,collate,digest,verify as verify113,sealed
from matched_model import OriginalBackbone,MatchedModule,BRANCH_SEED_OFFSET
import torch,numpy as np
def main():
    assert not O.exists(),'Preserve prior plans';O.mkdir(parents=True);configure();verify113();sealed()
    assert read(R/'outputs/grid114_fit_diagnosis/review.json')['passed']
    paths=[R/f'outputs/grid107_direct_value/{n}' for n in ['design.json','preflight.json','train.npz','training_rows.json','fits.json','validation_results.json','validation_trajectories.json']]
    paths += [R/f'outputs/grid113_relational_learning/{n}' for n in ['code_freeze.json','data_freeze.json','design.json','preflight.json','review.json']]
    paths += [R/f'outputs/grid114_fit_diagnosis/{n}' for n in ['GRID114_report.md','decision.json','review.json']]
    paths += [R/f'work/grid113_relational_learning/{n}' for n in ['model.py','common.py']]+[R/'work/grid112_relational_plan/schema.py']
    paths += list(W.glob('*.py'));save(O/'input_manifest.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    old=read(R/'outputs/grid107_direct_value/design.json');oldp=read(R/'outputs/grid107_direct_value/preflight.json');newp=read(R/'outputs/grid113_relational_learning/preflight.json')
    sampler={};z=np.load(R/'outputs/grid107_direct_value/train.npz');prob=z['sampling_probability']
    for seed in range(3):
        rng=np.random.default_rng(20261022+seed);counts=np.zeros(3582,np.int64);h=hashlib.sha256()
        for _ in range(7250):
            ix=rng.choice(3582,32,replace=True,p=prob);np.add.at(counts,ix,1);h.update(ix.astype('<i8').tobytes())
        assert h.hexdigest()==newp['sampling_streams'][str(seed)]['index_sha256']
        assert np.array_equal(counts,np.load(R/f'outputs/grid107_direct_value/training_exposure_seed{seed}.npz')['state_draws'])
        sampler[str(seed)]=dict(index_sha256=h.hexdigest(),state_draws=int(counts.sum()),scalar_targets=int(2*counts.sum()))
    # Synthetic data only; no inference on training/validation examples or labels.
    rng=np.random.default_rng(115);synthetic=[]
    for n,m in [(7,5),(3,2)]:
        synthetic.append(dict(context_action=rng.normal(size=(n,68)).astype('f'),context_feedback=rng.normal(size=(n,8)).astype('f'),
            next_action=rng.normal(size=(m,68)).astype('f'),anchor_action=rng.normal(size=68).astype('f'),global41=rng.normal(size=41).astype('f'),known_query_cost=np.array([.02],'f')))
    inp=collate(synthetic);init={}
    for seed in range(3):
        torch.manual_seed(seed);base=OriginalBackbone();joint=MatchedModule(seed,'JOINT');factor=MatchedModule(seed,'FACTOR')
        assert digest(base)==digest(joint.base)==digest(factor.base)==oldp['initial_q_hashes'][str(seed)]
        assert digest(joint)==digest(factor)
        assert sum(p.numel() for p in base.parameters())==6978
        assert sum(p.numel() for p in joint.delta.parameters())==5938
        assert sum(p.numel() for p in joint.parameters())==12916
        assert all(p.requires_grad for p in joint.parameters())
        assert all(torch.equal(p,dict(factor.named_parameters())[k]) for k,p in joint.named_parameters())
        before=digest(joint)
        with torch.no_grad():
            qb=base(inp['global41']);qj=joint(**inp);qf=factor(**inp)
            assert torch.equal(qb,qj) and torch.equal(qb,qf)
            assert torch.count_nonzero(joint.delta(**inp))==0
            # Zero output is deliberate; hidden branch features are not all zero.
            hidden={}
            handle=joint.delta.head[-1].register_forward_pre_hook(lambda module,args:hidden.update(value=args[0].detach().clone()))
            joint(**inp);handle.remove();assert torch.count_nonzero(hidden['value'])>0
        assert before==digest(joint)
        init[str(seed)]=dict(original_base_hash=digest(base),matched_full_initial_hash=digest(joint),
            synthetic_initial_prediction_exact=True,zero_correction=True,hidden_correction_features_not_all_zero=True)
    plan=dict(id='GRID115_PRESERVED_BACKBONE_MODULE_PLAN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='matched_proposal_no_training',
        authorization='User: 好的，匹配一下吧. Match and show concrete plan; no training authorization inferred.',
        question='With the original41->64->64->2 backbone retained, does paired action-feedback correction improve complete-search value beyond equal-capacity factorized correction and the original backbone?',
        groups=[dict(name='BASE',model='Original41->64->64->2,6978parameters',source='Reuse GRID107 DIRECT seeds0/1/2; no repeated fits.'),
            dict(name='BASE_JOINT',model='Original6978 plus GRID113 JOINT correction5938 =12916trainable parameters.'),
            dict(name='BASE_FACTOR',model='Original6978 plus GRID113 FACTOR correction5938 =12916trainable parameters.')],
        formula='q_total(global41,public_sets)=q_base(global41)+delta(public_sets). Return CONTINUE iff q_total[1]>q_total[0]; tieSTOP.',
        initialization=dict(base='Same random initialization as GRID107, byte-exact hashes checked all3seeds; NO pretrained or continued checkpoint.',
            branch_seed_offset=BRANCH_SEED_OFFSET,branch='JOINT/FACTOR identical named tensors per seed. Final16->2 correction weights and biases zero; other layers default original random initialization.',
            preservation='Retain original architecture and exact initial output, not a guarantee of preserving decisions after training. All base and correction weights train jointly from scratch.',
            zero_head_gradient='At first optimizer step upstream correction gradients are initially zero; final correction layer can learn. Verify upstream flow after the first planned update without extra research updates.'),
        unchanged=dict(train_scenarios=32,validation_scenarios=8,train_dates=16,validation_dates=4,train_states=3582,validation_states=856,
            train_prefixes=2378,validation_prefixes=590,seeds=[0,1,2],normalization='Original frozen41; full public geometry/paid feedback schema fromGRID112/113 without truncation.',
            updates=7250,states_per_batch=32,scalar_targets_per_update=64,optimizer='Adam lr0.0003 betas0.9/0.999 eps1e-8 weight_decay0; clip_grad_norm10.',
            loss='Original smooth_l1 beta1 mean over32x2. No auxiliary/classification/contrastive loss, class weighting or fee/threshold change.',
            cost_per_query=old['cost_per_query'],gain_scale=old['gain_scale'],target='Original immediate normalized net continuation, STOP0.',
            sampling='Original state index stream per seed: equal date/equal startingprefix/uniform reachableposition. Same per-state counts and both scalar targets.',
            train_mode='From scratch, joint optimization of base+correction, single CPU thread. No frozen pretrained baseline, continuation epochs or validation checkpoint selection.'),
        future_fits=dict(count=6,order=[dict(seed=s,arm=a) for s in range(3) for a in ['BASE_JOINT','BASE_FACTOR']],
            endpoint='Final7250 only; all3seeds retained; validation predictions only after all6fits finish.',maximum_total_training_wall_s=1800,
            timeout='Preserve complete and partial fits; do not extend automatically or repeat completed fits.'),
        matching_boundaries=dict(primary='BASE_JOINT versus BASE_FACTOR is parameter/init/optimizer/stream/input-information/budget matched; only feedback association before/after pooling differs.',
            base_reference='BASE is smaller6978 vs12916. VersusBASE tests added-module practical value, not a parameter-matched causal proof. No dummy unused parameters.',
            resource='Equal optimization steps and scalar-target exposure do not imply equal FLOPs/training wall time/inference latency. Keep all actual time receipts; no speed claim.',
            history='GRID107 BASE reused only for offline decisions/targets. Never mix historical online timing into new efficiency claims.'),
        evaluation=dict(primary='Complete saved local-search validation trajectory four-date macro normalized net utility, all seeds.',
            mechanism_signal='JOINT family mean>FACTOR, all3paired seed differences>0,>=3/4 date differences>0 after seed averaging; report actual magnitudes and all adverse results.',
            incremental='Report JOINT/FACTOR against paired old BASE and family mean. If JOINT exceedsFACTOR but notBASE, do not claim useful addition to old model.',
            progression='Requires mechanism signal AND JOINT family mean aboveBASE, BATCH1 and historical gamma0DQN family; no automatic scaling/replay even if passed.',
            secondary=['physical-call-balanced utility','gain capture','query fraction','positive misses and unprofitable continues','train/validation weighted loss and sign-regret','per-date/per-seed and single/multi-batch results'],
            diagnostics='Three preexposed108/552/RF-path552 always shown. Same fixed label-independent cyclic-pair sensitivity used after fits only; not an acceptance criterion or deployed policy.',
            interpretation='Four exposed dates, correlated prefixes, same grid, FULL teacher-path immediate value. No whole-week quality, statistical significance, latency or cross-grid claim.'),
        before_future_training=['User reviews this concrete matched plan.',
            'Freeze actual trainer,original-data hashes and approved model code; push before fitting.',
            'Recheck exact base/full init hashes, stream hashes, all target formulas and public-target boundary.',
            'Synthetic backward only without optimizer before fit; first actual step must count within7250. Check upstream residual gradient flow after it, no extra warmup training.',
            'Audit both-arm code paths/masks/permutation/bus invariance inherited fromGRID112/113; keep initial zero residual expectation distinct from trained association sensitivity.'],
        no_execution=['No optimizer instantiated','No backward or optimizer step','No trained/validation prediction','No new fit,physical observation,forecast,scenario,final-test access,loss/network variant search'],
        loss_choice='Hold Huber fixed to isolate architecture; GRID108 adverse MSE/long-training evidence preserved. This phase does not claim to fix loss-decision mismatch.')
    save(O/'plan.json',plan)
    save(O/'matching_check.json',dict(passed=True,actual_parameters=dict(BASE=6978,BASE_JOINT=12916,BASE_FACTOR=12916,correction_each=5938),
        original_base_initial_hashes_matched=True,init=init,original_sampling_streams=sampler,synthetic_forward_only=True,
        backward_calls=0,optimizer_instances=0,optimizer_steps=0,new_fits=0,train_validation_model_inference=0,new_physical_calls=0,
        targets_fee_scale_unchanged=True,original_frozen_sources_unchanged=True,final_test_unmaterialized=True))
    assert all(sha(R/n)==h for n,h in read(O/'input_manifest.json').items());verify113();sealed()
    print('MATCHED:BASE6978,JOINT/FACTOR12916;exactoldbaseinit+zeroinitialcorrection+oldstreams;NO_TRAINING',flush=True)
if __name__=='__main__':main()
