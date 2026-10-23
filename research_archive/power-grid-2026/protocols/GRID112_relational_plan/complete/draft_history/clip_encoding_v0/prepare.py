"""Specify the candidate architecture and compare plan; validate schema, never train."""
from pathlib import Path
import collections,datetime,gzip,hashlib,json,pickle,shutil
import numpy as np
from schema import MAX_DEVICES,FEEDBACK,canonical,pack
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid112_relational_plan';W=Path(__file__).parent;REPO=R/'work/GRL-results-integration'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item())+'\n',encoding='utf8')
def linear(a,b,bias=True):return a*b+(b if bias else 0)
def relation(plan):
    p=np.array(plan);i,j=np.triu_indices(len(p),1);return (p[i]>0)&(p[j]>0)&(p[i]==p[j])

def main():
    assert not (O/'schema_check.json').exists()
    assert read(R/'outputs/grid111_action_feedback/complete.json')['passed']
    paths=[R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid102_search_rl/normalization.npz',
        R/'outputs/grid102_search_rl/design.json',R/'outputs/grid107_direct_value/train.npz',R/'outputs/grid107_direct_value/design.json',
        R/'outputs/grid107_direct_value/training_rows.json',R/'outputs/grid98_feature_enrichment/template_metadata.json',R/'outputs/grid111_action_feedback/review.json',
        R/'outputs/grid100_data_foundation/design.json']+list(W.glob('*.py'))
    sources={p.relative_to(R).as_posix():sha(p) for p in paths};save('input_manifest.json',sources)
    static=read(R/'outputs/grid98_feature_enrichment/template_metadata.json');assert max(static['sub_info'])==MAX_DEVICES
    parts=dict(action_encoder=linear(68,16)+linear(16,16),feedback_encoder=linear(8,16)+linear(16,16),
        pair_value_encoder=linear(32,16)+linear(16,16),query_key_projections=linear(16,8,False)*2,
        common_encoder=linear(58,16)+linear(16,16),candidate_decoder=linear(48,16)+linear(16,16),head=linear(48,16)+linear(16,2))
    assert sum(parts.values())==5938
    save('parameter_count.json',dict(parts=parts,total_each_arm=sum(parts.values()),old_41_mlp=6978,neural_network_instantiated=False))
    old=read(R/'outputs/grid107_direct_value/design.json');D=read(R/'outputs/grid102_search_rl/design.json')
    plan=dict(id='GRID112_RELATIONAL_INPUT_PROPOSAL',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='proposal_for_user_review_no_training',
        question='Does keeping each public action connection pattern paired with its already-paid feedback improve estimation of next-batch net gain beyond independent feedback aggregation?',
        source=dict(train_states=3582,train_prefixes=2378,train_dates=16,train_scenarios=32,validation_states=856,validation_prefixes=590,validation_dates=4,validation_scenarios=8,final_test_scenarios_unmaterialized=8),
        context='All already-paid same-station feedback, including flagged invalid/terminal rows. No truncation, reward-based topK or prototype selection.',
        action_descriptor=dict(width=68,canonical_partition17='First active terminal anchors canonical bus0, other bus1; preserves complete two-bus partition.',
            relative_partition17='Canonical bits minus canonical incumbent bits.',active17='Connected status flag, distinguishes disconnected from canonical0.',present17='Padding/device-presence mask.',
            limitations='Public local terminal order kept. Invariant to bus1/2 swap and context-row permutation, not to arbitrary terminal permutations. Same118-station grid only; max17 derives from public static grid, not labels. No cross-grid transfer claim.'),
        feedback_descriptor=dict(width=8,fields=FEEDBACK,reward='(float32 reward-current incumbent float32 reward)/old gain_scale, clip[-20,20]. Nonfinite reward=>0 with missing flag.',
            prohibited='No unbought outcomes, future reward, after_id, full-week cost, date or IDs as scalar predictors.'),
        shared_inputs='Frozen normalized41 summary, encoded incumbent partition, known next-batch query cost(old price/scale); latter is an existing-information deterministic transform.',
        network=dict(hidden=16,key_query=8,parameters_each_arm=5938,hidden_activation='ReLU after every hidden linear, including both encoder layers; head final2 linear.',
            action='A:68->16->16 shared for context/query/anchor.',feedback='F:8->16->16.',
            joint='v_i=V([A(action_i),F(feedback_i)]), V:32->16->16.',
            factor='m_F=masked_mean_i F(feedback_i); v_i=V([A(action_i),m_F]). Same weights/shapes, but feedback pooled before associating with actions.',
            attention='q_j=Q(A(next_j));k_i=K(A(context_i)), each16->8 without bias; alpha_ji=masked_softmax(q_j dot k_i/sqrt8); r_j=sum alpha_ji v_i.',
            common='G:[global41,anchor_A16,known_cost1](58)->16->16.',
            candidate='H:[next_A16,r_j16,G16](48)->16->16.',
            output='[masked_mean_j H_j16,masked_max_j H_j16,G16](48)->16->2 => q_stop,q_continue.',
            masks='Exclude padded context logits and next rows before mean/max. No empty context or next set enters gate.',
            interpretation='Attention averages learned latent vectors, not raw rewards. Linear value head can exceed observed-reward-derived values; no calibrated uncertainty or optimal extrapolation guarantee.'),
        primary_comparison=dict(arms=['JOINT','FACTOR'],new_fits_if_later_authorized=6,seeds=[0,1,2],
            matching='Exact same named parameter tensors/initial seed/Adam/state index stream/targets/update budget per seed. Only F aggregation location differs. This isolates joint-vs-factorized aggregation in this architecture, not all ways to use action-feedback information.',
            controls='BATCH1,STOP,FULL_LOCAL reuse existing deterministic teacher-path traces. Old DIRECT and gamma0DQN all-seed families remain historical anchors; architecture/parameter count differs so do not call their comparison a pure representation ablation.',
            no_shuffle='Use explicit factorized aggregation, not one chosen corruption permutation or a recoverable deterministic reward permutation.'),
        target=dict(stop=0,continue_formula='(max(r_best,max new strictly admissible float32 rewards)-r_best - cost_per_query*next_queries)/gain_scale',
            cost_per_query=D['cost_per_query'],gain_scale=D['gain_scale'],scope='Original immediate teacher-path search value, not whole-week value or future-grid RL.'),
        proposed_training=dict(mode='supervised direct two-action value regression, not a new RL training run',updates=7250,states_per_batch=32,scalar_targets_per_update=64,
            optimizer='Adam lr0.0003 default original betas/epsilon, max_grad_norm10, CPU one thread.',loss='smooth_l1 beta1 mean over32x2 scalar targets, unchanged.',
            sampling='Exactly old train.npz probability; equal date, equal initial prefix within date, uniform reachable position. Match stored GRID107 index stream per seed.',
            endpoint='Final7250 only. No extra loss/budget/network/fee/seed variants, validation checkpoint selection or oversampling.',
            logging='Train-only fixed periodic loss checkpoints; validation only after all planned fits. Retain all3seeds.',
            maximum_training_wall_s=1800,timeout='Stop and preserve partial weights/error; no automatic cap extension or rerun complete fits.'),
        evaluation=dict(primary='Four-date macro COMPLETE saved search-trajectory net utility, all3seeds; not pooled one-step MSE.',
            secondary=['physical-call-balanced utility','gain capture','paid query fraction','positive misses and unprofitable continues','train/validation value errors','per-date/per-seed and single/multi-batch results'],
            diagnostic='108,552,RF-path552 always shown, never training oversampling or acceptance based only on552.',
            mechanism_signal='JOINT family mean utility > FACTOR, all3 paired seed utility deltas>0, at least3/4 date deltas positive after seed averaging. Report mixed outcomes if not met; no significance claim from3seeds/4exposed dates.',
            progression='Even if mechanism signal, competitive candidate additionally needs family mean utility above BATCH1 and BOTH historical DIRECT/gamma0DQN families. No automatic scale-up or physical rollout.',
            uncertainty='No fitted predictive distribution this first round. Seeds describe optimization variability, not calibrated outcome uncertainty.'),
        pretraining_requirements=['User reviews concrete plan before training starts','Implement both forwards and verify actual5938 params, shared init and original sampling stream',
            'Context joint-row permutation invariance, next-order invariance, canonical-bus invariance, padding invariance',
            'FACTOR separately permuted feedback invariant; JOINT responds to a constructed correspondence counterexample',
            'No gradient/inference path from future labels to inputs; saved43xx metadata/shape source checks',
            'Single backward/optimizer engineering smoke check only after training-stage authorization; never count it as a research fit or silently reuse weights'],
        executed_this_stage='Input schema arithmetic and parameter formula only. No neural model/optimizer instantiated, no forward, backward, fit, new physical call or test labels.')
    save('plan.json',plan)
    # No target/future-target file is opened; original states only supply the paid-feedback41 values.
    C=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());states={(c['sample_id'],j):x for c in C for j,x in enumerate(c['states'])}
    norm=np.load(R/'outputs/grid102_search_rl/normalization.npz');check=collections.Counter();maxseen=0;maxnext=0;clip=0
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);c=r['public_input'];g=np.clip((states[r['sample_id'],r['position']]-norm['mean'])/norm['scale'],-20,20).astype(np.float32)
            out=pack(c,g,D['cost_per_query'],D['gain_scale']);assert out['context_action'].shape==(len(c['seen_ids']),68) and out['context_feedback'].shape==(len(c['seen_ids']),8)
            assert out['next_action'].shape==(len(c['next_ids']),68);maxseen=max(maxseen,len(c['seen_ids']));maxnext=max(maxnext,len(c['next_ids']))
            for p in c['seen_plans']+c['next_plans']:
                bits,active,mask=canonical(p);restored=np.where(active[:len(p)]>0,bits[:len(p)]+1,-1)
                assert np.array_equal(relation(restored),relation(p));check['lossless_partition_roundtrips']+=1
                swapped=[3-v if v>0 else v for v in p]
                assert all(np.array_equal(a,b) for a,b in zip(canonical(p),canonical(swapped)));check['bus_label_checks']+=1
            assert all(np.array_equal(v,pack({**c,'seen_ids':[100000+i for i in range(len(c['seen_ids']))],
                'next_ids':[200000+i for i in range(len(c['next_ids']))],'incumbent_id':999999},g,D['cost_per_query'],D['gain_scale'])[k]) for k,v in out.items());check['id_relabel_invariance']+=1
            try:pack(dict(c,next_rewards=[99.]),g,D['cost_per_query'],D['gain_scale'])
            except AssertionError:check['future_input_rejections']+=1
            else:raise AssertionError('Forbidden future input accepted')
            clip+=int(np.sum(np.abs(out['context_feedback'][:,0])==20));check['states']+=1
    assert check['states']==4438 and maxseen==267 and maxnext==64
    base=read(R/'outputs/grid100_data_foundation/design.json');tests=[r for r in base['scenario_plan'] if r['partition']=='test'];assert len(tests)==8
    assert all(not (R/base['environment']/'chronics'/r['scenario']).exists() and not (R/'outputs/grid100_data_foundation/datasets'/r['scenario']).exists() for r in tests)
    assert all(sha(R/p)==h for p,h in sources.items())
    save('schema_check.json',dict(passed=True,checks=dict(check),maximum_context_rows=maxseen,maximum_next_rows=maxnext,
        clipped_reward_gap_tokens=clip,geometry_encoding_lossless_for_existing_two_bus_partitions=True,
        model_instantiations=0,forward_passes=0,optimizer_steps=0,new_fits=0,new_physical_calls=0,final_test_ungenerated=True,
        unexecuted='Actual neural forward/padding/permutation/gradient checks remain pretraining engineering requirements.'))
    print(read(O/'schema_check.json'),flush=True)
if __name__=='__main__':main()
