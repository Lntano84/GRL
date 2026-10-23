"""Frozen-endpoint diagnosis only: no optimizer construction, step, or new fit."""
from pathlib import Path
import sys, json, hashlib, datetime, shutil, collections
R=Path('D:/Codex/GRL研究')
sys.path.insert(0,str(R/'work/grid123_event_auxiliary'))
import common123 as src
import numpy as np
import torch
W=Path(__file__).parent; O=R/'outputs/grid124_shared_gradient_audit'; S=src.O
REPO=R/'work/GRL-results-integration'; A=REPO/'research_archive/power-grid-2026/protocols/GRID124_shared_gradient_audit'
read=src.read; save=src.save; sha=src.sha
ARMS=['NET_ZERO','AUX_NETSIGN','AUX_GAIN']

def verify():
    for fn in ['code_freeze.json','data_freeze.json']:
        for name,h in read(O/fn).items():assert sha(R/name)==h,name
    src.verify(); src.sealed()

def docs(text,next_steps):
    for n in ['RESEARCH_STATE.md','DECISIONS.md','EXPERIMENT_LOG.md']:
        with (REPO/'docs'/n).open('a',encoding='utf8') as f:f.write('\n\n'+text+'\n')
    (REPO/'docs/NEXT_STEPS.md').write_text(next_steps,encoding='utf8')

def archive(phase):
    d=A/phase;d.mkdir(parents=True,exist_ok=False)
    for p in [*W.glob('*.py'),*O.glob('*')]:
        if p.is_file():shutil.copy2(p,d/p.name)
    save(d/'archive_manifest.json',{p.name:sha(p) for p in d.iterdir() if p.is_file()})

def prepare():
    assert not O.exists();O.mkdir(parents=True);src.verify();src.sealed()
    assert read(S/'complete.json')['passed'] and read(S/'review.json')['passed']
    d=dict(id='GRID124_SHARED_GRADIENT_AUDIT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='User: 继续查. Diagnose existing auxiliary-learning results; no retraining.',
      purpose=['Does auxiliary output learn its own label?','Is main regression already worse on fitting dates?','Are shared gradients large, opposed, or both?'],
      scope='All30 GRID123 auxiliary endpoints and their original initializations;15 cached regression controls, all original full/fold splits and3seeds.',
      gradients='Exact expectation under original fitting-side state sampling probabilities, at initial and final tensors in float64 diagnostic copies. No held/validation gradient; no optimization. Save shared6848 main/aux gradients. Independent NumPy manual backprop checks all60 pairs.',
      adaptive_metric='Final diagonal 1/(sqrt(saved Adam v/(1-beta2**7250))+eps), on shared parameters only. Report gmain dot D*gaux and gmain dot D*(gmain+gaux). D fixed, first moments omitted: NOT an actual Adam update or retraining counterfactual.',
      metrics='Use saved float32 predictions; fitting-side original sampling weights, held conditional original weights, validation date/case/state-balanced weights. Main Huber/sign-regret and auxiliary BCE/Brier/AUC; fit-prevalence constant BCE reference. No threshold/class-weight search. Event-vs-net label mismatch is descriptive, not inherent logical contradiction.',
      curves='Ten consecutive nonoverlapping725-update blocks of all saved scheduled minibatch main/aux losses. No endpoint or window selection.',
      interpretation='Gradient size alone is not causal evidence; opposed gradients at endpoints do not prove persistent conflict. Fit/held state metrics differ from complete-search utility; correlate prefixes/date groups are not independent samples. No pvalues or causal claims.',
      prohibited=['optimizer instances/steps','fit','loss-weight search','model/threshold change','environment prediction','new data','final8tests','physical replay','paper/figures'])
    save(O/'design.json',d)
    files=[S/n for n in ['design.json','review.json','complete.json','fits.json','predictions.npz','train_data.npz','validation_data.npz','train_metadata.json','validation_metadata.json','decision.json','full_train_results.json','validation_results.json','oof_results.json','fit_side_results.json','summary_analysis.json','code_freeze.json','data_freeze.json']]
    files+=list((S/'models').glob('*.pt'))+list(S.glob('exposure_*.npz'))+list(S.glob('gradient_components_*.json'))
    files+=[src.CV/n for n in ['design.json','train_arrays.npz','metadata.json']]+list(src.CV.glob('fold*_normalization.npz'))
    files+=[R/'outputs/grid102_search_rl/cases.pkl',O/'design.json']
    save(O/'data_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in files})
    save(O/'code_freeze.json',{p.relative_to(R).as_posix():sha(p) for p in [*W.glob('*.py'),src.W/'common123.py',R/'work/grid121_date_transfer/common121.py']})
    save(O/'preflight.json',dict(passed=True,source_review_passed=True,checkpoints=30,new_fits=0,optimizer_instances=0,new_physical_calls=0,final_test_materialized=False))
    archive('before')
    docs('## 2026-10-23 — GRID124 frozen-endpoint gradient diagnosis frozen\n\nUser continues diagnosis,not retraining. All30auxiliary endpoints plus exact initial tensors,15cachedbasecontrols. Exact fit-only expectation gradients raw and fixedsaved-Adam-vmetric; independent NumPybackprop; mainfit/heldlosses,auxBCE/Brier/AUC,ten725-update savedcurveblocks. No optimizer/weightsearch/thresholdchange/physical/test. Endpointgradients not persistent or Adamcounterfactual causality.',
      '# Next steps — GRID124训练任务干扰核查\n\n1. 核对辅助分类能否学到标签，以及收益回归在训练侧是否已经受损。\n2. 对全部30终点计算训练分布期望梯度，并独立核验方向与尺度；不更新权重。\n3. 汇报能确认的原因与尚不能确认的部分，不根据本轮诊断自动调系数或重训。\n')
    print('GRID124_FROZEN',flush=True)

def grad_pair(net,x,y,event,prob,arm):
    params=list(net.parameters());shared=params[:4]
    q=net(torch.from_numpy(x.astype(float)));yy=torch.from_numpy(y.astype(float));p=torch.from_numpy(prob.astype(float));lab=torch.from_numpy((event if arm=='AUX_GAIN' else y>0).astype(float))
    main=.5*(p*torch.nn.functional.smooth_l1_loss(q[:,1],yy,reduction='none',beta=1.)).sum()
    aux=.5*(p*torch.nn.functional.binary_cross_entropy_with_logits(q[:,0],lab,reduction='none')).sum()
    gm=torch.autograd.grad(main,shared,retain_graph=True);ga=torch.autograd.grad(aux,shared)
    return np.concatenate([v.detach().numpy().ravel() for v in gm]),np.concatenate([v.detach().numpy().ravel() for v in ga])

def geometry(m,a,D=None):
    dot=float(m@a);nm=float(np.linalg.norm(m));na=float(np.linalg.norm(a));out=dict(main_norm=nm,aux_norm=na,aux_over_main=na/nm if nm else None,cosine=dot/(nm*na) if nm*na else None,dot=dot,combined_main_directional_dot=float(m@(m+a)))
    if D is not None:
        mm=float(m@(D*m));aa=float(a@(D*a));ma=float(m@(D*a));out.update(adam_v_metric_main_dot=mm,adam_v_metric_aux_dot=aa,adam_v_metric_cross=ma,adam_v_metric_cosine=ma/np.sqrt(mm*aa) if mm*aa else None,adam_v_metric_combined_main_dot=mm+ma)
    return out

def auc(score,label,p):
    pos=p[label].sum();neg=p[~label].sum()
    if not pos or not neg:return None
    ids=np.argsort(score,kind='stable');s=score[ids];l=label[ids];w=p[ids];starts=np.r_[0,np.flatnonzero(np.diff(s))+1];pw=np.add.reduceat(w*l,starts);nw=np.add.reduceat(w*~l,starts)
    return float(pw@(np.cumsum(nw)-.5*nw)/(pos*neg))

def metrics(q,y,event,p,arm,prevalence):
    p=p/p.sum();err=q[:,1].astype(float)-y;hub=np.where(abs(err)<1,.5*err**2,abs(err)-.5);go=q[:,1]>0
    out=dict(states=len(y),main_huber=float(p@hub),sign_regret=float(p@(np.maximum(y,0)-go*y)),false_stop_mass=float(p@((y>0)&~go)),unprofitable_continue_mass=float(p@((y<=0)&go)))
    if arm!='NET_ZERO':
        t=(event>0) if arm=='AUX_GAIN' else y>0;z=q[:,0].astype(float);pred=1/(1+np.exp(-z));base=np.clip(prevalence,1e-15,1-1e-15)
        out.update(aux_prevalence=float(p@t),aux_auc=auc(z,t,p),aux_bce=float(p@(np.logaddexp(0,z)-t*z)),constant_fit_prevalence_bce=float(p@(-t*np.log(base)-(~t)*np.log1p(-base))),aux_brier=float(p@((pred-t)**2)),constant_fit_prevalence_brier=float(p@((base-t)**2)))
    return out

def run():
    assert not (O/'analysis.json').exists();verify();src.configure();Q=np.load(S/'predictions.npz');records=[];vectors={};metric={};curves={}
    cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());counts=collections.Counter(c['date'] for c in cases if c['partition']=='validation');bycase={c['sample_id']:c for c in cases}
    meta=read(S/'validation_metadata.json');pv=np.array([1/len(counts)/counts[m['date']]/len(bycase[m['sample_id']]['states']) for m in meta]);assert abs(pv.sum()-1)<1e-12
    ptrain=np.load(S/'train_data.npz')['probability']
    for split in ['full','f0','f1','f2','f3']:
        x,y,event,ids,p=src.getdata(split);assert abs(p.sum()-1)<1e-12;held=np.setdiff1d(np.arange(len(y)),ids);ph=ptrain[held]/ptrain[held].sum() if len(held) else None
        for arm in ARMS:
            for seed in range(3):
                name=f'{split}__{arm}_seed{seed}';q=Q[name+'__train'];prev=float(p@(event[ids] if arm=='AUX_GAIN' else y[ids]>0));rr=dict(fit=metrics(q[ids],y[ids],event[ids],p,arm,prev))
                if split=='full':
                    _,vy,ve,_,_=src.getdata('full','validation');rr['validation']=metrics(Q[name+'__validation'],vy,ve,pv,arm,prev)
                else:rr['held']=metrics(q[held],y[held],event[held],ph,arm,prev)
                metric[name]=rr
                if arm=='NET_ZERO':continue
                cp=torch.load(S/'models'/f'{name}.pt',weights_only=False,map_location='cpu');net=src.build(seed).double()
                for stage in ['initial','final']:
                    if stage=='final':net.load_state_dict(cp['state_dict'])
                    m,a=grad_pair(net,x[ids],y[ids],event[ids],p,arm);assert len(m)==len(a)==6848
                    D=None
                    if stage=='final':
                        state=cp['optimizer']['state'];pg=cp['optimizer']['param_groups'][0];ss=[state[k] for k in pg['params'][:4]];assert all(int(v['step'])==7250 for v in ss)
                        v=np.concatenate([u['exp_avg_sq'].numpy().astype(float).ravel() for u in ss]);D=1/(np.sqrt(v/(1-pg['betas'][1]**7250))+pg['eps']);assert np.isfinite(D).all()
                    key=name+'__'+stage;vectors[key+'__main']=m;vectors[key+'__aux']=a
                    if D is not None:vectors[key+'__D']=D
                    records.append(dict(name=name,split=split,arm=arm,seed=seed,stage=stage,fit_states=len(ids),**geometry(m,a,D)))
                e=np.load(S/f'exposure_{name}.npz');loss=e['half_main_aux_losses'];assert loss.shape==(7250,2);curves[name]=[dict(first_update=725*k+1,last_update=725*(k+1),half_main_mean=float(loss[725*k:725*(k+1),0].mean()),half_aux_mean=float(loss[725*k:725*(k+1),1].mean())) for k in range(10)]
    np.savez_compressed(O/'gradient_vectors.npz',**vectors)
    save(O/'analysis.json',dict(gradient_records=records,state_metrics=metric,training_curve_blocks=curves,new_fits=0,optimizer_instances=0,weight_updates=0,new_physical_calls=0,final_test_materialized=False))
    verify();print('GRID124_ANALYZED',len(records),len(metric),flush=True)

def manual(net,x,y,event,p,arm):
    w=[net[f'q_net.{k}.weight'].numpy().astype(float) for k in [0,2,4]];b=[net[f'q_net.{k}.bias'].numpy().astype(float) for k in [0,2,4]]
    a0=x.astype(float);z1=a0@w[0].T+b[0];a1=np.maximum(z1,0);z2=a1@w[1].T+b[1];a2=np.maximum(z2,0);q=a2@w[2].T+b[2];lab=event if arm=='AUX_GAIN' else y>0
    dm=np.zeros_like(q);dm[:,1]=.5*p*np.clip(q[:,1]-y,-1,1);da=np.zeros_like(q);da[:,0]=.5*p*(1/(1+np.exp(-q[:,0]))-lab)
    def back(d):
        d2=(d@w[2])*(z2>0);g2=d2.T@a1;b2=d2.sum(0);d1=(d2@w[1])*(z1>0);g1=d1.T@a0;b1=d1.sum(0)
        return np.concatenate([g1.ravel(),b1,g2.ravel(),b2])
    return back(dm),back(da)

def audit():
    assert not (O/'review.json').exists();verify();src.configure();d=read(O/'analysis.json');G=np.load(O/'gradient_vectors.npz');maximum=0.;count=0
    meta=read(S/'train_metadata.json');folds=read(S/'design.json')['folds']
    for r in d['gradient_records']:
        x,y,e,ids,p=src.getdata(r['split']);net=src.build(r['seed'])
        if r['stage']=='final':net.load_state_dict(torch.load(S/'models'/f"{r['name']}.pt",weights_only=False,map_location='cpu')['state_dict'])
        sd=net.state_dict();m,a=manual(sd,x[ids],y[ids],e[ids],p,r['arm']);key=r['name']+'__'+r['stage'];sm,sa=G[key+'__main'],G[key+'__aux'];maximum=max(maximum,float(abs(m-sm).max()),float(abs(a-sa).max()))
        assert np.allclose(m,sm,rtol=1e-9,atol=1e-11) and np.allclose(a,sa,rtol=1e-9,atol=1e-11),(key,maximum)
        D=G[key+'__D'] if r['stage']=='final' else None
        for k,v in geometry(m,a,D).items():
            if v is None:assert r[k] is None
            else:assert abs(v-r[k])<=1e-8*max(1,abs(v)),(key,k)
        if r['split']!='full':assert all(meta[i]['date'] not in folds[int(r['split'][1:])]['held_dates'] for i in ids)
        count+=1
    # Independent pairwise weighted AUC on all30 validation/held score vectors.
    Q=np.load(S/'predictions.npz');max_auc=0.
    for name,rr in d['state_metrics'].items():
        split,armseed=name.split('__');arm=armseed.rsplit('_seed',1)[0]
        if arm=='NET_ZERO':continue
        part='validation' if split=='full' else 'train';_,y,e,ids,p=src.getdata(split,part)
        if split=='full':
            meta=read(S/'validation_metadata.json');cases=src.pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cc=collections.Counter(c['date'] for c in cases if c['partition']=='validation');by={c['sample_id']:c for c in cases};ids=np.arange(len(y));p=np.array([1/len(cc)/cc[m['date']]/len(by[m['sample_id']]['states']) for m in meta]);tag='validation'
        else:ids=np.setdiff1d(np.arange(len(y)),ids);p=np.load(S/'train_data.npz')['probability'][ids];p/=p.sum();tag='held'
        q=Q[name+'__'+part][ids];label=(e[ids]>0) if arm=='AUX_GAIN' else y[ids]>0;s=q[:,0];pp=p[label];nn=p[~label];pairs=(s[label,None]>s[None,~label])+.5*(s[label,None]==s[None,~label]);av=float(pp@pairs@nn/(pp.sum()*nn.sum()));max_auc=max(max_auc,abs(av-rr[tag]['aux_auc']));assert abs(av-rr[tag]['aux_auc'])<1e-10
    assert d['new_fits']==d['optimizer_instances']==d['weight_updates']==0;verify();src.sealed()
    save(O/'review.json',dict(passed=True,manual_numpy_gradient_pairs=count,max_gradient_abs_error=maximum,pairwise_aux_auc_checks=30,max_auc_error=max_auc,fit_only_gradient=True,source_hashes_unchanged=True,new_fits=0,optimizer_instances=0,weight_updates=0,new_physical_calls=0,final_test_materialized=False))
    print('GRID124_AUDIT_PASSED',count,maximum,max_auc,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','run','audit']);a=p.parse_args();{'prepare':prepare,'run':run,'audit':audit}[a.phase]()
