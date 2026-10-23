"""Read-only GRID113 endpoint diagnosis. No optimizer, fit, environment or policy deployment."""
from pathlib import Path
import sys,os,json,gzip,pickle,hashlib,datetime,collections
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid113_relational_learning'))
from common import read,save,sha,configure,collate,digest,verify as verify113,sealed
from model import RelationalValue
import numpy as np,torch
W=Path(__file__).parent;O=R/'outputs/grid114_fit_diagnosis';S=R/'outputs/grid113_relational_learning'
def huber(e):return np.where(abs(e)<1,.5*e**2,abs(e)-.5)
def stats(v):
    v=np.asarray(v);return dict(n=len(v),min=float(v.min()),median=float(np.median(v)),max=float(v.max()),mean=float(v.mean())) if len(v) else dict(n=0)
def risk(q,y,p):
    t=np.column_stack([np.zeros_like(y),y]);err=q-t;h=huber(err).mean(1);pos=y>0;miss=pos&(q[:,1]<=q[:,0]);neg=~pos&(q[:,1]>q[:,0])
    g=.5*np.clip(q[:,1]-y,-1,1)*p
    return dict(weighted_huber=float(p@h),positive_huber_contribution=float(p[pos]@h[pos]),negative_huber_contribution=float(p[~pos]@h[~pos]),
        stop_output_contribution=float(p@huber(q[:,0])*.5),continue_output_contribution=float(p@huber(q[:,1]-y)*.5),
        positive_misses=int(miss.sum()),negative_continues=int(neg.sum()),missed_positive_sampling_mass=float(p[miss].sum()),
        misses_qcontinue_nonpositive=int((miss&(q[:,1]<=0)).sum()),misses_qcontinue_positive_but_stop_larger=int((miss&(q[:,1]>0)).sum()),
        positive_continue_error=stats(q[pos,1]-y[pos]),positive_advantage=stats((q[:,1]-q[:,0])[pos]),
        continue_bias_gradient=dict(positive=float(g[pos].sum()),negative=float(g[~pos].sum()),total=float(g.sum())),
        missed_target=stats(y[miss]))
def old_predictions(meta):
    lookup={(r['sample_id'],r['position']):i for i,r in enumerate(meta)};names=[f'DIRECT{s}' for s in range(3)]+[f'DQN0_{s}' for s in range(3)]
    ans={n:np.full((len(meta),2),np.nan,np.float32) for n in names}
    with gzip.open(R/'outputs/grid107_direct_value/state_predictions.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);k=(r['sample_id'],r['position'])
            if r['method'] in ans and k in lookup:ans[r['method']][lookup[k]]=[r['q0'],r['q1']]
    assert all(np.isfinite(a).all() for a in ans.values());return ans
def detailed_forward(m,b):
    ca=b['context_action'];cf=b['context_feedback'];na=b['next_action'];cm=b['context_mask'][...,None];nm=b['next_mask'][...,None]
    a=m.A(ca);f=m.F(cf);q=m.A(na);anchor=m.A(b['anchor_action'])
    if m.arm=='FACTOR':f=((f*cm).sum(1,keepdim=True)/cm.sum(1,keepdim=True)).expand_as(a)
    v=m.V(torch.cat([a,f],-1));logit=m.Q(q)@m.K(a).transpose(1,2)/np.sqrt(8)
    alpha=logit.masked_fill(~b['context_mask'][:,None,:],-torch.inf).softmax(-1);context=alpha@v
    g=m.G(torch.cat([b['global41'],anchor,b['known_query_cost']],-1));h=m.H(torch.cat([q,context,g[:,None,:].expand(-1,len(na[0]),-1)],-1))
    mean=(h*nm).sum(1)/nm.sum(1);maxi=h.masked_fill(~nm,-torch.inf).max(1).values;z=torch.cat([mean,maxi,g],-1);score=m.head(z)
    ent=-(alpha*alpha.clamp_min(1e-30).log()).sum(-1)
    sizes=b['context_mask'].sum(1);den=sizes.clamp_min(2).float().log()
    normalized=(ent*b['next_mask']).sum(1)/b['next_mask'].sum(1)/den
    normalized=torch.where(sizes>1,normalized,torch.zeros_like(normalized))
    return score,dict(attention_normalized_entropy=normalized,head_input=z,action=a,feedback=m.F(cf),candidate=h)
def gradient_vectors(m,rows,y,p):
    # Exact endpoint gradient of the ORIGINAL sampling-weighted two-action Huber risk.
    # Separate positive/negative contributions. No optimizer object or update.
    ans={};params=list(m.parameters())
    for label,mask in [('positive',y>0),('negative',y<=0)]:
        m.zero_grad(set_to_none=True);ids=np.flatnonzero(mask)
        for j in range(0,len(ids),32):
            ix=ids[j:j+32];q=m(**collate([rows[i] for i in ix]));tar=torch.from_numpy(np.column_stack([np.zeros(len(ix),np.float32),y[ix]]))
            loss=torch.nn.functional.smooth_l1_loss(q,tar,beta=1.,reduction='none').mean(1)
            (loss*torch.as_tensor(p[ix],dtype=torch.float32)).sum().backward()
        ans[label]=np.concatenate([x.grad.detach().numpy().ravel().astype(np.float64) if x.grad is not None else np.zeros(x.numel()) for x in params])
    a=ans['positive'];b=ans['negative'];groups={};offset=0
    for name,x in m.named_parameters():
        key=name.split('.')[0];n=x.numel();aa=a[offset:offset+n];bb=b[offset:offset+n];g=groups.setdefault(key,dict(positive_sq=0.,negative_sq=0.,dot=0.,n=0))
        g['positive_sq']+=float(aa@aa);g['negative_sq']+=float(bb@bb);g['dot']+=float(aa@bb);g['n']+=n;offset+=n
    for g in groups.values():
        g['positive_norm']=np.sqrt(g.pop('positive_sq'));g['negative_norm']=np.sqrt(g.pop('negative_sq'));g['cosine']=g['dot']/max(g['positive_norm']*g['negative_norm'],1e-30)
    m.zero_grad(set_to_none=True)
    return dict(positive_norm=float(np.linalg.norm(a)),negative_norm=float(np.linalg.norm(b)),total_norm=float(np.linalg.norm(a+b)),
        cosine=float(a@b/max(np.linalg.norm(a)*np.linalg.norm(b),1e-30)),groups=groups)
def main():
    assert not O.exists(),'Never overwrite prior diagnosis';O.mkdir(parents=True);configure();verify113();sealed()
    assert read(S/'review.json')['passed'];assert read(S/'fits.json')['new_fits']==6
    plan=dict(id='GRID114_FROZEN_ENDPOINT_FIT_DIAGNOSIS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        authorization='User: 继续查. Analyze why positive training decisions still fail; no new fits.',
        scope=['all3582train+856validation saved outputs','original sampling/loss and exposure accounting','all6fixed checkpoint endpoint gradients and latent activation/attention diagnostics',
               'complete train-positive list and common failures versus old DIRECT','whole-search payoff attribution already completed GRID113'],
        controls='All arms/seeds and full train/validation partitions; no selection of checkpoint/threshold/features, no changed decisions deployed.',
        limits='Endpoint gradients and representations are descriptive, not training-trajectory/causal or generalization proof. Labels are used only after fixed forward for audit/loss.',
        zero_new_fits=True,optimizer_instances=0,physical_calls=0,final_test_materialized=False)
    save(O/'design.json',plan)
    paths=[S/n for n in ['train_public.pkl','train_targets.npz','train_metadata.json','validation_public.pkl','validation_targets.npz','validation_metadata.json',
        'state_predictions.npz','fits.json','design.json','review.json','train_stdout.log','decision.json']]
    paths+=list((S/'models').glob('*.pt'))+list(S.glob('exposure_*.npz'))
    paths += [R/'outputs/grid107_direct_value/state_predictions.jsonl.gz',R/'outputs/grid108_loss_budget/GRID108_report.md',R/'outputs/grid109_input_coverage/GRID109_report.md',
        R/'work/grid113_relational_learning/model.py',R/'work/grid113_relational_learning/common.py',W/'analyze.py']
    save(O/'input_manifest.json',{p.relative_to(R).as_posix():sha(p) for p in paths})
    meta={s:read(S/f'{s}_metadata.json') for s in ['train','validation']};rows={s:pickle.loads((S/f'{s}_public.pkl').read_bytes()) for s in meta}
    y={s:np.load(S/f'{s}_targets.npz')['target'].astype(np.float64) for s in meta};p=np.load(S/'train_targets.npz')['sampling_probability'];Q=np.load(S/'state_predictions.npz')
    old={s:old_predictions(meta[s]) for s in meta};metrics={};exposure={};latent={};grad={};failrows=[]
    pos=y['train']>0;zero_gradient=.5*(-np.clip(y['train'],-1,1))*p
    zero=dict(positive_states=int(pos.sum()),negative_states=int((~pos).sum()),positive_sampling_mass=float(p[pos].sum()),
        positive_targets_above_huber_unit=int((y['train']>1).sum()),positive_target_sampling_mass_above_unit=float(p[y['train']>1].sum()),
        weighted_target_mean=float(p@y['train']),unweighted_target_summary=stats(y['train']),positive_target_summary=stats(y['train'][pos]),
        at_zero_continue_bias_gradient_positive=float(zero_gradient[pos].sum()),at_zero_continue_bias_gradient_negative=float(zero_gradient[~pos].sum()),
        at_zero_continue_bias_gradient_total=float(zero_gradient.sum()),
        note='Derivative w.r.t a shared continue-output intercept. Gradient descent moves opposite this sign; not a claim that an input-dependent optimum must share its sign.')
    for n in ['DIRECT0','DIRECT1','DIRECT2','DQN0_0','DQN0_1','DQN0_2']:
        metrics[n]={s:risk(old[s][n],y[s],p if s=='train' else np.ones(len(y[s]))/len(y[s])) for s in meta}
    for rec in read(S/'fits.json')['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";metrics[name]={s:risk(Q[name+'__'+s],y[s],p if s=='train' else np.ones(len(y[s]))/len(y[s])) for s in meta}
        z=np.load(S/f'exposure_{name}.npz');cnt=z['state_draws'];loss=z['minibatch_losses'];assert len(loss)==7250
        exposure[name]=dict(positive_draws=int(cnt[pos].sum()),positive_counts=stats(cnt[pos]),negative_draws=int(cnt[~pos].sum()),positive_never_seen=int((cnt[pos]==0).sum()),
            mean_minibatch_loss_first500=float(loss[:500].mean()),mean_minibatch_loss_updates6251_6750=float(loss[-1000:-500].mean()),mean_minibatch_loss_last500=float(loss[-500:].mean()))
        torch.manual_seed(rec['seed']);m=RelationalValue(rec['arm']);cp=torch.load(S/'models'/f'{name}.pt',map_location='cpu',weights_only=False);m.load_state_dict(cp['state_dict']);m.eval();before=digest(m)
        positive_grad=gradient_vectors(m,rows['train'],y['train'].astype(np.float32),p);grad[name]=positive_grad
        lr={};alive={k:np.zeros(16,bool) for k in ['action','feedback','candidate']};headrows=[];entropy=[];embedding_collisions=0;checked=0
        with torch.no_grad():
            for j in range(0,len(rows['train']),32):
                rr=rows['train'][j:j+32];b=collate(rr);q,d=detailed_forward(m,b)
                assert np.allclose(q.numpy(),Q[name+'__train'][j:j+32],atol=2e-6,rtol=2e-6)
                entropy.extend(d['attention_normalized_entropy'].tolist());headrows.append(d['head_input'].numpy())
                for key,mask in [('action','context_mask'),('feedback','context_mask'),('candidate','next_mask')]:alive[key]|=(d[key][b[mask]]>0).any(0).numpy()
                for k,r in enumerate(rr):
                    raw=r['next_action'];emb=m.A(torch.from_numpy(raw)).numpy()
                    a=len(np.unique(raw,axis=0));e=len(np.unique(emb,axis=0));embedding_collisions+=int(e<a);checked+=1
        head=np.concatenate(headrows)
        lr=dict(attention_entropy_all_train=stats(entropy),attention_entropy_positive_train=stats(np.asarray(entropy)[pos]),
            final_hidden_units_ever_positive={k:int(v.sum()) for k,v in alive.items()},
            exact_action_encoder_next_set_collisions=embedding_collisions,states_checked_for_collisions=checked,
            head_input_exact_unique_rows=len(np.unique(head,axis=0)),head_input_total_rows=len(head),
            warning='Entropy is normalized by log(context size), singleton set=0. Near-uniform weights do not imply paired value aggregation is unused. No causal claim.')
        latent[name]=lr;assert before==digest(m)==rec['final_hash']
        for s in meta:
            q=Q[name+'__'+s];miss=(y[s]>0)&(q[:,1]<=q[:,0]);oi=old[s][f"DIRECT{rec['seed']}"];omiss=(y[s]>0)&(oi[:,1]<=oi[:,0])
            for i in np.flatnonzero(miss):failrows.append(dict(meta[s][i],method=name,target_normalized_net=float(y[s][i]),q_stop=float(q[i,0]),q_continue=float(q[i,1]),
                q_continue_nonpositive=bool(q[i,1]<=0),old_direct_also_missed=bool(omiss[i]),sampling_probability=float(p[i]) if s=='train' else None,
                actual_state_draws=int(cnt[i]) if s=='train' else None))
        print(name,'READONLY_GRADIENT_AND_REPRESENTATION_DONE',flush=True)
    consensus={}
    for s in meta:
        j=np.stack([(Q[f'JOINT_seed{k}__{s}'][:,1]<=Q[f'JOINT_seed{k}__{s}'][:,0])&(y[s]>0) for k in range(3)])
        od=np.stack([(old[s][f'DIRECT{k}'][:,1]>old[s][f'DIRECT{k}'][:,0]) for k in range(3)])
        mask=j.all(0)&od.all(0);idx=np.flatnonzero(mask)
        consensus[s]=dict(joint_all3_miss=int(j.all(0).sum()),joint_all3_miss_direct_all3_correct=len(idx),physical_calls=len({meta[s][i]['physical_state_group'] for i in idx}),
            dates=sorted({meta[s][i]['date'] for i in idx}),rows=[dict(meta[s][i],target=float(y[s][i])) for i in idx])
    save(O/'zero_prediction_loss_geometry.json',zero);save(O/'endpoint_metrics.json',metrics);save(O/'exposure_and_loss_windows.json',exposure)
    save(O/'gradient_decomposition.json',grad);save(O/'representation_diagnostics.json',latent);save(O/'all_positive_misses.json',failrows);save(O/'consensus_failures.json',consensus)
    verify113();sealed();assert all(sha(R/n)==h for n,h in read(O/'input_manifest.json').items())
    save(O/'analysis_receipt.json',dict(passed=True,models=6,training_states=3582,validation_states=856,zero_fits=True,zero_optimizer_instances=True,
        endpoint_gradient_only=True,all_weight_hashes_unchanged=True,all_source_hashes_unchanged=True,physical_calls=0,final_test_materialized=False))
    print(json.dumps(zero,indent=2),flush=True)
if __name__=='__main__':main()
