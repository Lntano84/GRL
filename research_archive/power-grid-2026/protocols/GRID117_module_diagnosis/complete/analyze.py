"""Fixed-checkpoint module diagnosis; no optimizer, backward, fit or physical forecast."""
from pathlib import Path
import sys,collections,datetime,gzip
R=Path('D:/Codex/GRL研究');sys.path.insert(0,str(R/'work/grid116_matched_module'))
from common116 import *
from common116 import verify as verify116
import torch,numpy as np
W=Path(__file__).parent;O=R/'outputs/grid117_module_diagnosis';S116=R/'outputs/grid116_matched_module';S113=R/'outputs/grid113_relational_learning'
def stat(x):
    x=np.asarray(x,dtype=float);return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),min=float(x.min()),max=float(x.max())) if len(x) else dict(n=0)
def instruments(net,b):
    m=net.delta;cm=b['context_mask'][...,None];nm=b['next_mask'][...,None]
    a=m.A(b['context_action']);f=m.F(b['context_feedback']);q=m.A(b['next_action']);anchor=m.A(b['anchor_action']);fm=(f*cm).sum(1,keepdim=True)/cm.sum(1,keepdim=True)
    joint_input=torch.cat([a,f],-1);factor_input=torch.cat([a,fm.expand_as(a)],-1)
    j1=m.V[0](joint_input);f1=m.V[0](factor_input);j2=m.V[2](j1.relu());f2=m.V[2](f1.relu());vj=j2.relu();vf=f2.relu()
    logits=(m.Q(q)@m.K(a).transpose(1,2))/np.sqrt(8);alpha=logits.masked_fill(~b['context_mask'][:,None,:],-torch.inf).softmax(-1)
    uniform=b['context_mask'][:,None,:].float()/b['context_mask'].sum(1)[:,None,None]
    actual_v=vj if net.arm=='JOINT' else vf
    g=m.G(torch.cat([b['global41'],anchor,b['known_query_cost']],-1));base=net.base(b['global41']);out={}
    for name,att,v in [('actual',alpha,actual_v),('factorized_same_weights',alpha,vf),('uniform_attention',uniform,actual_v),('both',uniform,vf)]:
        context=att@v
        if context.shape[1]==1:context=context.expand(-1,q.shape[1],-1)
        hh=m.H(torch.cat([q,context,g[:,None,:].expand(-1,q.shape[1],-1)],-1))
        hmean=(hh*nm).sum(1)/nm.sum(1);hmax=hh.masked_fill(~nm,-torch.inf).max(1).values
        delta=m.head(torch.cat([hmean,hmax,g],-1));out['delta_'+name]=delta;out[name]=base+delta
    out['base']=base
    entropy=-(alpha*alpha.clamp_min(1e-30).log()).sum(-1);n=b['context_mask'].sum(1);den=n.clamp_min(2).float().log()
    out['entropy']=(entropy*b['next_mask']).sum(1)/b['next_mask'].sum(1)/den
    out['tv_uniform']=((alpha-uniform).abs().sum(-1)*b['next_mask']).sum(1)/b['next_mask'].sum(1)/2
    # Under a common ReLU pattern across the observed paired and factorized rows, V is affine on these rows.
    masks=[j1>0,f1>0,j2>0,f2>0]
    constant1=(((masks[0]==masks[0][:,:1])&(masks[1]==masks[0][:,:1]))|~cm).all((1,2))
    constant2=(((masks[2]==masks[2][:,:1])&(masks[3]==masks[2][:,:1]))|~cm).all((1,2))
    out['affine_compatible']=constant1&constant2
    mean_j=(vj*cm).sum(1)/cm.sum(1);mean_f=(vf*cm).sum(1)/cm.sum(1);out['uniform_value_gap']=(mean_j-mean_f).norm(dim=1)
    assert torch.allclose(mean_j[out['affine_compatible']],mean_f[out['affine_compatible']],atol=3e-5,rtol=3e-5)
    return out
def main():
    assert not O.exists(),'Preserve prior diagnosis';O.mkdir(parents=True);configure();verify116();sealed();assert read(S116/'review.json')['passed']
    design=dict(id='GRID117_FIXED_MODULE_DIAGNOSIS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User: 再查一下原因吧. Read-only analysis, no new training.',
        questions=['Where does extra paid continuation occur, grouped by physical calls rather than repeated prefixes?',
            'At each first actual common-prefix divergence, which additive component accounts for the sign?',
            'Does paired V encoding behave affinely and does attention approximate uniform averaging?',
            'At identical frozen weights, do factorizing feedback, making attention uniform, or both change original decisions?'],
        fixed_diagnostics=['actual','factorized_same_weights','uniform_attention','both'],
        scope='All six checkpoints, full3582train/856validation. Fixed forward diagnostics only, no variant selection or counterfactual-policy utility ranking.',
        attribution_limit='Base and correction coadapted jointly. Components and inference interventions are arithmetic explanations, not independently trained ablations or causal training attribution.',
        zero_fits=True,zero_backward=True,zero_optimizer_instances=True,new_physical_calls=0,final_test_materialized=False)
    save(O/'design.json',design)
    sources=[S116/n for n in ['design.json','fits.json','review.json','decision.json','state_predictions.npz','train_trajectories.json','validation_trajectories.json','validation_results.json','payoff_accounting.json','binding_sensitivity.json']]
    sources+=list((S116/'models').glob('*.pt'))
    sources+=[S113/f'{p}_{n}' for p in ['train','validation'] for n in ['public.pkl','metadata.json','targets.npz']]
    sources += [R/'outputs/grid102_search_rl/cases.pkl',R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz',R/'outputs/grid107_direct_value/state_predictions.jsonl.gz',R/'outputs/grid107_direct_value/design.json',W/'analyze.py']
    save(O/'input_manifest.json',{p.relative_to(R).as_posix():sha(p) for p in sources})
    rawmeta={}
    with gzip.open(R/'outputs/grid111_action_feedback/public_inputs.jsonl.gz','rt',encoding='utf8') as f:
        for line in f:
            r=json.loads(line);rawmeta[r['sample_id'],r['position']]={k:r[k] for k in ['step','source_call']}
    data={p:pickle.loads((S113/f'{p}_public.pkl').read_bytes()) for p in ['train','validation']};meta={p:read(S113/f'{p}_metadata.json') for p in data};target={p:np.load(S113/f'{p}_targets.npz')['target'] for p in data}
    lookup={p:{(r['sample_id'],r['position']):i for i,r in enumerate(meta[p])} for p in data};pred=np.load(S116/'state_predictions.npz');arrays={};summary={};models={}
    for rec in read(S116/'fits.json')['records']:
        name=f"{rec['arm']}_seed{rec['seed']}";net=build(rec['seed'],rec['arm']);cp=torch.load(S116/'models'/f'{name}.pt',map_location='cpu',weights_only=False);net.load_state_dict(cp['state_dict']);before=digest(net);net.eval();models[name]=net;summary[name]={}
        for part in data:
            pieces=collections.defaultdict(list)
            with torch.no_grad():
                for j in range(0,len(data[part]),32):
                    out=instruments(net,collate(data[part][j:j+32]))
                    for k,v in out.items():pieces[k].append(v.numpy())
            values={k:np.concatenate(v) for k,v in pieces.items()};q=values['actual'];assert np.allclose(q,pred[name+'__'+part],atol=3e-6,rtol=3e-6)
            assert np.allclose(values['base']+values['delta_actual'],q,atol=1e-7,rtol=1e-7)
            for k,v in values.items():arrays[name+'__'+part+'__'+k]=v
            positive=target[part]>0;eligible=np.array([len(r['context_action'])>1 for r in data[part]]);adv=q[:,1]-q[:,0];baseadv=values['base'][:,1]-values['base'][:,0];corr=values['delta_actual'][:,1]-values['delta_actual'][:,0]
            entry=dict(attention_entropy_eligible=stat(values['entropy'][eligible]),attention_tv_uniform_eligible=stat(values['tv_uniform'][eligible]),
                affine_compatible_states=int(values['affine_compatible'].sum()),total_states=len(q),affine_uniform_value_gap=stat(values['uniform_value_gap'][values['affine_compatible']]),
                paired_uniform_value_gap=stat(values['uniform_value_gap']),base_advantage=stat(baseadv),correction_advantage=stat(corr),
                base_correction_correlation=float(np.corrcoef(baseadv,corr)[0,1]),
                delta_flips_base_stop_to_continue=int(((baseadv<=0)&(adv>0)).sum()),delta_flips_base_continue_to_stop=int(((baseadv>0)&(adv<=0)).sum()),variants={})
            for variant in ['factorized_same_weights','uniform_attention','both']:
                qv=values[variant];change=(qv[:,1]-qv[:,0])-adv;flip=(qv[:,1]>qv[:,0])!=(adv>0)
                entry['variants'][variant]=dict(abs_advantage_change=stat(abs(change)),positive_abs_advantage_change=stat(abs(change[positive])),
                    decision_sign_changes=int(flip.sum()),positive_sign_changes=int((flip&positive).sum()))
                if variant=='factorized_same_weights' and rec['arm']=='BASE_FACTOR':assert np.allclose(qv,q,atol=3e-6,rtol=3e-6)
            summary[name][part]=entry
        assert before==digest(net)==rec['final_hash'];print(name+' COMPONENTS_CHECKED',flush=True)
    np.savez_compressed(O/'component_forward_outputs.npz',**arrays);save(O/'component_summary.json',summary)
    cases=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());bycase={c['sample_id']:c for c in cases};trajectories=read(S116/'validation_trajectories.json');traj={(r['method'],r['sample_id']):r for r in trajectories};dates=sorted({c['date'] for c in cases if c['partition']=='validation'});ndate=collections.Counter(c['date'] for c in cases if c['partition']=='validation')
    fee=read(S116/'design.json')['unchanged']['cost_per_query'];scale=read(S116/'design.json')['unchanged']['gain_scale'];divergences=[];contributions=collections.defaultdict(lambda:dict(net=0.,gain=0.,fee=0.,prefixes=set(),seeds=set()))
    for s in range(3):
        jn=f'BASE_JOINT_seed{s}';fn=f'BASE_FACTOR_seed{s}'
        for c in cases:
            if c['partition']!='validation':continue
            j=traj[jn,c['sample_id']];f=traj[fn,c['sample_id']];w=1/(len(dates)*ndate[c['date']]*3)
            delta=j['net_utility']-f['net_utility'];gain=(j['gain']-f['gain'])/scale;charge=-fee*(j['queries']-f['queries'])/scale;assert abs(delta-gain-charge)<1e-12
            cc=contributions[c['physical_state_group']];cc['net']+=w*delta;cc['gain']+=w*gain;cc['fee']+=w*charge
            if delta!=0:cc['prefixes'].add(c['sample_id']);cc['seeds'].add(s)
            p=0
            while p<len(c['states']):
                ix=lookup['validation'][c['sample_id'],p];qj=pred[jn+'__validation'][ix];qf=pred[fn+'__validation'][ix];aj=bool(qj[1]>qj[0]);af=bool(qf[1]>qf[0])
                if aj!=af:
                    vals={}
                    for label,n in [('joint',jn),('factor',fn)]:
                        base=arrays[n+'__validation__base'][ix];corr=arrays[n+'__validation__delta_actual'][ix];total=arrays[n+'__validation__actual'][ix]
                        vals[label]=dict(base_margin=float(base[1]-base[0]),correction_margin=float(corr[1]-corr[0]),total_margin=float(total[1]-total[0]))
                    assert abs((vals['joint']['total_margin']-vals['factor']['total_margin'])-((vals['joint']['base_margin']-vals['factor']['base_margin'])+(vals['joint']['correction_margin']-vals['factor']['correction_margin'])))<3e-6
                    row=dict(meta['validation'][ix],**rawmeta[c['sample_id'],p],seed=s,first_divergence_position=p,joint_continues=aj,factor_continues=af,
                        next_batch_net=float((c['gains'][p]-fee*c['queries'][p])/scale),extra_queries_joint_minus_factor=j['queries']-f['queries'],
                        net_joint_minus_factor=delta,weighted_family_contribution=w*delta,components=vals)
                    if aj and not af:
                        row['joint_continue_requires_correction']=vals['joint']['base_margin']<=0
                        row['factor_stop_requires_correction']=vals['factor']['base_margin']>0
                    divergences.append(row);break
                if not aj:break
                p+=1
    calls=[dict(physical_state_group=k,net_contribution=v['net'],gain_contribution=v['gain'],fee_contribution=v['fee'],changed_prefixes=len(v['prefixes']),seeds=sorted(v['seeds'])) for k,v in contributions.items()]
    calls.sort(key=lambda r:r['net_contribution']);expected=read(S116/'decision.json')['family_mean_utility'];assert abs(sum(r['net_contribution'] for r in calls)-(expected['JOINT']-expected['FACTOR']))<1e-12
    overview={}
    for s in range(3):
        rr=[r for r in divergences if r['seed']==s];extra=[r for r in rr if r['joint_continues']];bad=[r for r in extra if r['next_batch_net']<=0]
        overview[str(s)]=dict(first_divergences=len(rr),different_physical_calls=len({r['physical_state_group'] for r in rr}),joint_continues_factor_stops=len(extra),
            those_next_batch_unprofitable=len(bad),unprofitable_distinct_calls=len({r['physical_state_group'] for r in bad}),
            unprofitable_joint_continue_requires_correction=sum(r['joint_continue_requires_correction'] for r in bad),
            unprofitable_joint_base_already_continues=sum(not r['joint_continue_requires_correction'] for r in bad),
            unprofitable_factor_stop_requires_correction=sum(r['factor_stop_requires_correction'] for r in bad))
    save(O/'first_divergences.json',divergences);save(O/'divergence_overview.json',overview);save(O/'physical_call_contributions.json',calls)
    verify116();sealed();assert all(sha(R/n)==h for n,h in read(O/'input_manifest.json').items())
    save(O/'analysis_receipt.json',dict(passed=True,checkpoints=6,train_states=3582,validation_states=856,complete_first_common_prefix_divergences=len(divergences),
        all_weight_hashes_unchanged=True,all_source_hashes_unchanged=True,new_fits=0,backward_calls=0,optimizer_instances=0,new_physical_calls=0,final_test_materialized=False))
    print(json.dumps(overview,indent=2),flush=True)
if __name__=='__main__':main()
