"""Six fixed source-style Double DQN fits; held-family calibration, no evaluation data."""
from pathlib import Path
import os,time,json,hashlib,math,copy,sys
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import torch
import torch.nn.functional as F
from metacontrol import QModel,prefixes,utility,mask_q,replay
torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid64'
read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(t,p)
def main():
    assert not (OUT/'training_finished.json').exists()
    D=read(OUT/'design.json');records=read(OUT/'records.json')
    for p,h in read(OUT/'code_freeze.json').items():assert sha(ROOT/p)==h,p
    train=[r for r in records if r['split']=='train'];val=[r for r in records if r['split']=='calibration']
    for r in train+val:r['prefixes']=prefixes(r['rows'],r['rho_limit'])
    begun=time.perf_counter();results=[]
    for costindex,cost in enumerate(D['cost_levels']):
      beta=math.log1p(cost)
      for seed in D['seeds']:
        folder=OUT/'models'/f'cost{costindex}_seed{seed}';folder.mkdir(parents=True,exist_ok=False)
        torch.manual_seed(seed);rng=np.random.default_rng(seed);model=QModel();target=copy.deepcopy(model)
        torch.save(model.state_dict(),folder/'initial.pt');opt=torch.optim.Adam(model.parameters(),lr=D['lr'])
        cap=D['replay_capacity'];dim=12
        S=np.empty((cap,dim),np.float32);SN=np.empty_like(S);A=np.empty(cap,np.int64);R=np.empty(cap,np.float32);T=np.empty(cap,np.float32)
        size=steps=updates=0;lossvalue=0.;curve=[];start=time.perf_counter()
        for episode in range(D['episodes']):
          assert time.perf_counter()-begun<D['caps']['training_wall_s']
          # Sampling states from already-paid training performance traces; no simulator calls.
          rec=train[int(rng.integers(len(train)))];ps=rec['prefixes'];j=0;episode_return=0.
          while j<len(ps)-1:
            p=ps[j];state=p['state'];epsilon=1. if steps<D['min_explore_steps'] else max(D['epsilon_final'],1-(1-D['epsilon_final'])*(steps-D['min_explore_steps'])/D['epsilon_anneal_steps'])
            if state[2]<.5:action=1
            elif rng.random()<epsilon:action=int(rng.integers(2))
            else:
              with torch.inference_mode():q=mask_q(model(torch.from_numpy(state)),torch.from_numpy(state)).numpy()
              prob=np.exp((q-q.max())/.001);prob/=prob.sum();action=int(rng.choice(2,p=prob))
            if action==0:nextstate=state;reward=0.;terminal=True
            else:
              nextp=ps[j+1];nextstate=nextp['state'];reward=utility(nextp['quality'],nextp['end']/len(rec['rows']),beta)-utility(p['quality'],p['end']/len(rec['rows']),beta)
              j+=1;terminal=j==len(ps)-1
            index=steps%cap;S[index]=state;SN[index]=nextstate;A[index]=action;R[index]=reward;T[index]=terminal
            steps+=1;size=min(size+1,cap);episode_return+=reward
            if steps>=D['min_explore_steps'] and size>=D['batch_size']:
              ix=rng.integers(size,size=D['batch_size']);s=torch.from_numpy(S[ix]);sn=torch.from_numpy(SN[ix])
              a=torch.from_numpy(A[ix]);rr=torch.from_numpy(R[ix]);terminal_tensor=torch.from_numpy(T[ix])
              with torch.no_grad():
                probabilities=torch.softmax(mask_q(model(sn),sn)/.001,dim=1)
                vnext=(probabilities*target(sn)).sum(1);y=rr+D['gamma']*(1-terminal_tensor)*vnext
                desired=model(s).clone();desired[torch.arange(len(ix)),a]=y
              prediction=model(s);loss=F.mse_loss(prediction,desired);assert torch.isfinite(loss)
              opt.zero_grad();loss.backward();opt.step();updates+=1;lossvalue=float(loss.detach())
              with torch.no_grad():
                for tp,mp in zip(target.parameters(),model.parameters()):tp.mul_(D['target_retention']).add_(mp,alpha=1-D['target_retention'])
            if terminal:break
          if (episode+1)%1000==0:
            row={'episode':episode+1,'steps':steps,'updates':updates,'latest_td_mse':lossvalue,'wall_s':time.perf_counter()-start}
            curve.append(row);save(folder/'curve.json',curve);print(json.dumps({'costindex':costindex,'seed':seed,**row}),flush=True)
        assert updates>0
        assert all(torch.isfinite(p).all() for p in model.parameters())
        torch.save(model.state_dict(),folder/'final.pt')
        per=[]
        for rec in val:
            got=replay(model,rec['rows'],rec['rho_limit']);per.append({'id':rec['id'],'family':rec['family'],'queries':got['end'],'regret':max(0.,rec['best_quality']-got['quality']),'frame_queries':rec['frame_queries'],'frame_regret':max(0.,rec['best_quality']-rec['frame_quality'])})
        mean=lambda k:float(np.mean([p[k] for p in per]))
        summary={'costindex':costindex,'cost_level':cost,'beta':beta,'seed':seed,'episodes':D['episodes'],'steps':steps,'updates':updates,'wall_s':time.perf_counter()-start,'calibration_queries':mean('queries'),'calibration_mean_regret':mean('regret'),'calibration_max_regret':max(p['regret'] for p in per),'frame_mean_regret':mean('frame_regret'),'frame_max_regret':max(p['frame_regret'] for p in per),'final_sha256':sha(folder/'final.pt'),'initial_sha256':sha(folder/'initial.pt'),'rows':per}
        save(folder/'summary.json',summary);results.append(summary)
        del S,SN,A,R,T,opt,target,model
    options=[]
    for index,cost in enumerate(D['cost_levels']):
      own=[r for r in results if r['costindex']==index]
      feasible=all(r['calibration_mean_regret']<=r['frame_mean_regret']+1e-7 and r['calibration_max_regret']<=r['frame_max_regret']+1e-7 for r in own)
      options.append({'costindex':index,'cost_level':cost,'feasible':feasible,'mean_queries':sum(r['calibration_queries'] for r in own)/2,'mean_regret':sum(r['calibration_mean_regret'] for r in own)/2,'max_regret':max(r['calibration_max_regret'] for r in own)})
    feasible=[r for r in options if r['feasible']]
    selected=min(feasible,key=lambda r:(r['mean_queries'],r['mean_regret'],r['costindex'])) if feasible else min(options,key=lambda r:(r['mean_regret'],r['max_regret'],r['mean_queries'],r['costindex']))
    result={'passed_finite_training':True,'fits':results,'options':options,'selected':selected,'calibration_quality_matched':bool(feasible),'training_wall_s':time.perf_counter()-begun,'new_teacher_queries':0,'evaluation_data_accessed':False,'new_training_role':'Published literature baseline stopping policy only; no GNN or new physical control RL.'}
    save(OUT/'training_finished.json',result);(OUT/'RUN_STATE.md').write_text('SIX FITS COMPLETE: calibration selected; learning/interface qualification pending.\n',encoding='utf-8');print(json.dumps({'finished':True,'selected':selected,'wall_s':result['training_wall_s']}),flush=True)
if __name__=='__main__':main()
