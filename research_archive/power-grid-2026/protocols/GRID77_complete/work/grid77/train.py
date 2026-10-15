"""Six fixed stronger conference-adaptation fits, common features/modes."""
from pathlib import Path
import copy,hashlib,json,math,os,sys,time
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
import torch.nn.functional as F
from metacontrol import QModel,prefixes,utility,mask_q,replay,DIM
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid77'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def main():
    assert not (OUT/'training_finished.json').exists();D=read(OUT/'design.json')
    assert read(OUT/'learning_preflight.json')['passed']
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    records=read(OUT/'records.json');subs=read(OUT/'data_receipts.json')['subids']
    for r in records:
        r['subids']=subs;r['aliases']={int(k):int(v) for k,v in r['aliases'].items()}
        r['context']={k:np.asarray(v,dtype=np.int64 if k=='edges' else np.float32) for k,v in r['context'].items()}
        r['prefixes']=prefixes(r)
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='calibration']
    begun=time.perf_counter();results=[]
    for costindex,cost in enumerate(D['cost_levels']):
      beta=math.log1p(cost)
      for seed in D['seeds']:
        folder=OUT/'models'/f'cost{costindex}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=QModel();target=copy.deepcopy(model);torch.save(model.state_dict(),folder/'initial.pt')
        opt=torch.optim.Adam(model.parameters(),lr=D['lr']);cap=D['replay_capacity']
        S=np.empty((cap,DIM),np.float32);SN=np.empty_like(S);A=np.empty(cap,np.int64);R=np.empty(cap,np.float32);T=np.empty(cap,np.float32)
        size=steps=updates=0;lossvalue=0.;curve=[];start=time.perf_counter()
        for episode in range(D['episodes']):
          assert time.perf_counter()-begun<D['caps']['training_wall_s']
          rec=train[int(rng.integers(len(train)))];ps=rec['prefixes'];j=0
          while j<len(ps)-1:
            p=ps[j];state=p['state'];epsilon=1. if steps<D['min_explore_steps'] else max(D['epsilon_final'],1-(1-D['epsilon_final'])*(steps-D['min_explore_steps'])/D['epsilon_anneal_steps'])
            allowed=[1]+([0] if state[2]>=.5 else [])+([2] if j==0 else [])
            if rng.random()<epsilon:action=int(rng.choice(allowed))
            else:
              with torch.inference_mode():q=mask_q(model(torch.from_numpy(state)),torch.from_numpy(state)).numpy()
              prob=np.exp((q-q.max())/.001);prob/=prob.sum();action=int(rng.choice(3,p=prob))
            assert action in allowed
            if action==0:nextstate=state;reward=0.;terminal=True
            elif action==2:
              assert j==0;nextstate=state
              reward=utility(rec['guard_quality'],rec['guard_queries']/len(rec['rows']),beta)-utility(p['quality'],p['end']/len(rec['rows']),beta);terminal=True
            else:
              nxt=ps[j+1];nextstate=nxt['state'];reward=utility(nxt['quality'],nxt['end']/len(rec['rows']),beta)-utility(p['quality'],p['end']/len(rec['rows']),beta)
              j+=1;terminal=j==len(ps)-1
            ix=steps%cap;S[ix]=state;SN[ix]=nextstate;A[ix]=action;R[ix]=reward;T[ix]=terminal;steps+=1;size=min(size+1,cap)
            if steps>=D['min_explore_steps'] and size>=D['batch_size']:
              ix=rng.integers(size,size=D['batch_size']);s=torch.from_numpy(S[ix]);sn=torch.from_numpy(SN[ix]);a=torch.from_numpy(A[ix]);rr=torch.from_numpy(R[ix]);terminal_tensor=torch.from_numpy(T[ix])
              with torch.no_grad():
                prob=torch.softmax(mask_q(model(sn),sn)/.001,dim=1);v=(prob*target(sn)).sum(1)
                y=rr+D['gamma']*(1-terminal_tensor)*v;desired=model(s).clone();desired[torch.arange(len(ix)),a]=y
              loss=F.mse_loss(model(s),desired);assert torch.isfinite(loss);opt.zero_grad();loss.backward();opt.step();updates+=1;lossvalue=float(loss.detach())
              with torch.no_grad():
                for tp,mp in zip(target.parameters(),model.parameters()):tp.mul_(D['target_retention']).add_(mp,alpha=1-D['target_retention'])
            if terminal:break
          if (episode+1)%3000==0:
            row=dict(episode=episode+1,steps=steps,updates=updates,latest_td_mse=lossvalue,wall_s=time.perf_counter()-start);curve.append(row);save(folder/'curve.json',curve)
            print(json.dumps(dict(costindex=costindex,seed=seed,**row)),flush=True)
        assert updates>0 and all(torch.isfinite(p).all() for p in model.parameters());torch.save(model.state_dict(),folder/'final.pt')
        per=[]
        for rec in val:
            got=replay(model,rec);per.append(dict(id=rec['id'],family=rec['family'],queries=got['end'],regret=max(0.,rec['best_quality']-got['quality']),
                guard_queries=rec['guard_queries'],guard_regret=max(0.,rec['best_quality']-rec['guard_quality']),mode=got['mode']))
        mean=lambda k:float(np.mean([r[k] for r in per]))
        summary=dict(costindex=costindex,cost_level=cost,beta=beta,seed=seed,steps=steps,updates=updates,wall_s=time.perf_counter()-start,
            calibration_queries=mean('queries'),calibration_mean_regret=mean('regret'),calibration_max_regret=max(r['regret'] for r in per),
            guard_mean_regret=mean('guard_regret'),guard_max_regret=max(r['guard_regret'] for r in per),rows=per,
            final_sha256=sha(folder/'final.pt'),initial_sha256=sha(folder/'initial.pt'))
        save(folder/'summary.json',summary);results.append(summary);del S,SN,A,R,T,opt,target,model
    options=[]
    for index,cost in enumerate(D['cost_levels']):
        own=[r for r in results if r['costindex']==index]
        options.append(dict(costindex=index,cost_level=cost,feasible=all(r['calibration_mean_regret']<=r['guard_mean_regret']+1e-7 and r['calibration_max_regret']<=r['guard_max_regret']+1e-7 for r in own),
            mean_queries=float(np.mean([r['calibration_queries'] for r in own])),mean_regret=float(np.mean([r['calibration_mean_regret'] for r in own])),max_regret=max(r['calibration_max_regret'] for r in own)))
    feasible=[r for r in options if r['feasible']]
    chosen=min(feasible,key=lambda r:(r['mean_queries'],r['mean_regret'],r['costindex'])) if feasible else min(options,key=lambda r:(r['mean_regret'],r['max_regret'],r['mean_queries'],r['costindex']))
    save(OUT/'selected.json',dict(chosen,calibration_quality_matched=bool(feasible)))
    save(OUT/'training_finished.json',dict(passed_engineering=True,fits=6,wall_s=time.perf_counter()-begun,selected=chosen,calibration_quality_matched=bool(feasible),options=options,results=results,new_teacher_queries=0))
    (OUT/'RUN_STATE.md').write_text('SIX FITS COMPLETE:calibration quality matched '+str(bool(feasible))+'; physical evaluation not started.\n',encoding='utf-8')
    print(json.dumps(dict(fits=6,selected=chosen,calibration_quality_matched=bool(feasible),wall_s=time.perf_counter()-begun)),flush=True)
if __name__=='__main__':main()
