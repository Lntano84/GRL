"""Public-only learned completion runtime. Never constructs or queries an environment."""
from pathlib import Path
import importlib.util,json,time
import numpy as np
R=Path('D:/Codex/GRL研究')
def load(name,path):
    sp=importlib.util.spec_from_file_location(name,R/path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
D=load('g183_model_api','work/grid174_power_helper/run.py')
G=load('g183_global','work/grid91_mechanism/policies.py')
S=load('g183_schema','work/grid112_relational_plan/schema.py')
E=load('g183_electrical','work/grid180_consensus_runtime/runtime_features.py')
# The original source imports ROOT from a live-controller module. Bind that
# constant directly, preserving the exact feature implementation and its hash.
source=R/'work/grid102_search_rl/observations.py'
text=source.read_text(encoding='utf8');assert text.count('from common import ROOT')==1
F={'ROOT':R,'__name__':'g183_visible_projection'}
exec(compile(text.replace('from common import ROOT',''),str(source),'exec'),F)
observe=F['observe'];best=F['best'];reward=F['reward']
FEEDBACK=['rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
class InputUnavailable(ValueError):pass

class PublicState:
    """Current physical observation, static device layout, and prepared commands only."""
    def __init__(self,ob,base,actions,static):
        self.ob=ob;self.base=base;self.actions=actions;self.static=static
        self.bounds=np.r_[0,np.cumsum(static.sub_info)];self.plans={};self.unknown={};self.power=None
    def plan(self,i):
        if i not in self.plans:
            comb=self.actions[i]+self.base;ob=self.ob;static=self.static
            plan=ob.topo_vect.copy();cmd=comb.set_bus;switch=comb.change_bus
            if np.any(switch&(cmd!=0)):raise InputUnavailable('overlapping_set_and_switch_bus')
            plan[cmd!=0]=cmd[cmd!=0];live=switch&(plan>0);plan[live]=3-plan[live]
            op=static.line_or_pos_topo_vect;ep=static.line_ex_pos_topo_vect
            off=(comb.line_set_status<0)|(comb.line_change_status&ob.line_status)
            on=(comb.line_set_status>0)|(comb.line_change_status&~ob.line_status)
            plan[op[off]]=-1;plan[ep[off]]=-1
            self.unknown[i]=set(np.r_[op[on&(plan[op]<=0)],ep[on&(plan[ep]<=0)]].tolist());self.plans[i]=plan
        return self.plans[i]
    def context(self,visible,next_ids,station,targets):
        inc=best(visible)
        if inc is None or inc['pool_id'] not in targets:raise InputUnavailable('incumbent_not_in_locked_station')
        local=[r for r in visible if r['pool_id'] in targets];a,b=self.bounds[station:station+2];positions=set(range(a,b))
        ids=[r['pool_id'] for r in local]+list(next_ids)
        for i in ids:
            self.plan(i)
            if self.unknown[i]&positions:raise InputUnavailable('reconnection_bus_not_known')
            if not set(self.plans[i][a:b].tolist())<={-1,1,2}:raise InputUnavailable('invalid_intended_bus')
        if not 0<b-a<=17:raise InputUnavailable('station_exceeds_frozen_device_padding')
        return dict(seen_ids=[r['pool_id'] for r in local],seen_plans=[self.plans[r['pool_id']][a:b].tolist() for r in local],
            seen_feedback=[{k:r[k] for k in FEEDBACK} for r in local],next_ids=list(next_ids),next_plans=[self.plans[i][a:b].tolist() for i in next_ids],
            incumbent_id=inc['pool_id'],incumbent_reward=float(np.float32(inc['reward'])),incumbent_plan=self.plans[inc['pool_id']][a:b].tolist())
    def electrical(self,c,station):
        ob=self.ob;static=self.static
        if self.power is None:
            self.power=np.zeros(len(ob.topo_vect),float)
            for pos,value,sign in [('gen_pos_topo_vect','gen_p',1),('load_pos_topo_vect','load_p',-1),('line_or_pos_topo_vect','p_or',-1),('line_ex_pos_topo_vect','p_ex',-1),('storage_pos_topo_vect','storage_power',-1)]:
                self.power[getattr(static,pos)]=sign*np.asarray(getattr(ob,value),float)
            self.power[ob.topo_vect<=0]=0
        a,b=self.bounds[station:station+2]
        return E.power8(dict(terminal_power=self.power[a:b],rho=ob.rho,line_status=ob.line_status,
            incident_lines=(static.line_or_to_subid==station)|(static.line_ex_to_subid==station),anchor_plan=c['incumbent_plan'],next_plans=c['next_plans']))

class FrozenModels:
    def __init__(self,seed):
        import torch
        assert seed in [0,1,2];self.seed=seed;self.nets={};self.old=D.G.old_api()
        folder=R/'outputs/grid182_deployment_bundle';manifest=D.read(folder/'deployment_manifest.json')
        for arm in ['STATE4','POWER8','FACTOR']:
            p=folder/'models'/f'{arm}_seed{seed}.pt';assert D.sha(p)==manifest['models'][p.name]
            net,_=D.G.build(seed,'ZERO_MAIN') if arm=='FACTOR' else D.build(seed)
            net.load_state_dict(torch.load(p,map_location='cpu',weights_only=False)['state_dict']);net.eval();self.nets[arm]=net
        assert D.sha(folder/'normalization.npz')==manifest['normalization_sha256']
        with np.load(folder/'normalization.npz') as z:self.norm={k:z[k].copy() for k in z.files}
        task=D.read(R/'outputs/grid102_search_rl/design.json');self.fee=task['cost_per_query'];self.scale=task['gain_scale']
    def inputs(self,raw,power,next_queries):
        g=np.clip((raw-self.norm['mean41'])/self.norm['scale41'],-20,20).astype('f')
        p=np.clip((power-self.norm['mean8'])/self.norm['scale8'],-20,20).astype('f')
        x8=np.r_[g,np.float32(self.fee*next_queries/self.scale),p].astype('f');x4=x8.copy();x4[46:]=0
        return g,x4,x8
    def small(self,arm,x):
        import torch
        with torch.inference_mode():return self.nets[arm](torch.from_numpy(x[None,:]))[0].numpy().copy()
    def factor(self,public):
        import torch
        with torch.inference_mode():return self.nets['FACTOR'](**self.old.collate([public]))[0].numpy().copy()
    def decide(self,mode,raw,c,power):
        assert mode in ['FACTOR','CONSENSUS'];scores={};delegated=mode=='FACTOR'
        g=np.clip((raw-self.norm['mean41'])/self.norm['scale41'],-20,20).astype('f')
        if mode=='CONSENSUS':
            g,x4,x8=self.inputs(raw,power,len(c['next_ids']))
            scores['STATE4']=self.small('STATE4',x4);scores['POWER8']=self.small('POWER8',x8)
            votes=[bool(scores[k][1]>scores[k][0]) for k in ['STATE4','POWER8']];delegated=votes[0]!=votes[1]
            if not delegated:return int(votes[0]),dict(delegated=False,scores={k:v.tolist() for k,v in scores.items()})
        scores['FACTOR']=self.factor(S.pack(c,g,self.fee,self.scale))
        return int(scores['FACTOR'][1]>scores['FACTOR'][0]),dict(delegated=delegated,scores={k:v.tolist() for k,v in scores.items()})

class CompletionPolicy:
    def __init__(self,method,models=None):
        assert method in ['FRAME_GUARD','LOCAL_BATCH1','FACTOR','CONSENSUS'];self.method=method;self.models=models
        assert (models is not None)==(method in ['FACTOR','CONSENSUS'])
    def complete(self,result,order,reveal,subids,aliases,limit,provider):
        rows=result['outcomes'];inc=best(rows);initial=None if inc is None else reward(inc)
        pending=result['completion_candidate_ids'].copy();budget=len(pending);purchased=[];decisions=[];batch=0;feature_s=inference_s=0.;fallback=None
        station=result['local_substation'];targets=set() if not budget else {int(rep) for alias,rep in aliases.items() if int(subids[int(alias)])==station}
        while pending:
            ids=pending[:64];details={};raw=power=None
            if self.method=='LOCAL_BATCH1':action=int(batch==0)
            elif fallback:action=1;details['engineering_fallback']=fallback
            else:
                t=time.perf_counter()
                try:
                    raw=observe(rows,order,aliases,subids,limit,targets,pending,budget,batch)
                    c=provider.context(rows,ids,station,targets)
                    if self.method=='CONSENSUS':power=provider.electrical(c,station)
                except InputUnavailable as e:
                    fallback=str(e);action=1;details['engineering_fallback']=fallback
                feature_s+=time.perf_counter()-t
                if fallback is None:
                    t=time.perf_counter();action,details=self.models.decide(self.method,raw,c,power);inference_s+=time.perf_counter()-t
            decisions.append(dict(batch=batch,next_ids=ids,action=action,paid_queries=len(rows),remaining=len(pending),
                raw_features=None if raw is None else raw.tolist(),physical8=None if power is None else power.tolist(),**details))
            if action==0:break
            before=G.key(inc)
            for i in ids:
                assert i not in {r['pool_id'] for r in rows};r=reveal(i,'local_completion');assert r['pool_id']==i
                rows.append(r);purchased.append(i)
                if r['strict_admissible'] and G.key(r)>G.key(inc):inc=r
            result['events'].append(dict(queries=len(rows),phase='local_completion',chosen=inc['pool_id'],improved=G.key(inc)>before))
            pending=pending[len(ids):];batch+=1
        result.update(chosen_pool_id=None if inc is None else inc['pool_id'],public_queries=len(rows),local_queries=len(purchased),extra_selected_ids=purchased,
            stop_reason=result.get('stop_reason','')+('_learned_stop' if pending else '_local_exhausted') if budget else result.get('stop_reason',''),
            learned_decisions=decisions,local_feature_s=feature_s,local_inference_s=inference_s,learning_method=self.method,engineering_fallback=fallback,
            local_reward_gain=0. if initial is None else reward(inc)-initial)
        return result
    def run(self,order,reveal,subids,aliases,limit=None,provider=None,context=None):
        if self.method=='FRAME_GUARD':
            r=G.run(order,reveal,'FRAME_GUARD',subids,aliases,limit=limit,context=context)
            r.update(learned_decisions=[],local_feature_s=0.,local_inference_s=0.,learning_method=self.method,engineering_fallback=None);return r
        r=G.run(order,reveal,'NO_LOCAL',subids,aliases,limit=limit,context=context)
        return self.complete(r,order,reveal,subids,aliases,limit,provider)
