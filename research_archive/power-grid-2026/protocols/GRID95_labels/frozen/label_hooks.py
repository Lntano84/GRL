"""Read-only state witnesses and one predeclared completion omission."""
from pathlib import Path
import hashlib,importlib.util,json,pickle,random
import numpy as np

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/grid95_labels'
ITEM=None;CASES={};CURRENT=None;OMISSIONS=0


def sha_bytes(v):return hashlib.sha256(v).hexdigest()
def digest(x):return sha_bytes(np.ascontiguousarray(x).tobytes())
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def packed(value):
    if isinstance(value,np.ndarray):return dict(shape=list(value.shape),dtype=value.dtype.str,sha256=digest(value))
    if isinstance(value,np.generic):return packed(value.item())
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):return value if np.isfinite(value) else str(value)
    if isinstance(value,(list,tuple)):
        if all(isinstance(v,(str,bool,int,float,np.generic,type(None))) for v in value):return [packed(v) for v in value]
    if hasattr(value,'get_state') and type(value).__module__.startswith('numpy'):
        return dict(rng_sha256=sha_bytes(pickle.dumps(value.get_state(),protocol=5)))
    if type(value).__module__.startswith('cvxpy') and type(value).__name__ in ('Parameter','Variable'):
        return dict(cvxtype=type(value).__name__,value=packed(value.value))
    if type(value).__module__.startswith('grid2op.Action'):
        from common import action_vector
        return dict(action_sha256=digest(action_vector(value)))
    return NotImplemented


def flat_state(obj):
    data={};unserialized=[]
    for name,v in sorted(vars(obj).items()):
        if name.endswith('_counts') or name in ['solve_status','_timer_owner']:continue
        pv=packed(v)
        if pv is not NotImplemented:data[name]=pv
        elif name in ('_grid29_dc_programs','_grid30_safe_programs'):
            entries=[]
            for key,entry in v.items():
                problem=entry[0]
                entries.append(dict(key_sha256=sha_bytes(pickle.dumps(key,protocol=5)),
                    parameter_values=[packed(p.value) for p in problem.parameters()],
                    variable_values=[packed(p.value) for p in problem.variables()]))
            data[name]=entries
        else:unserialized.append(dict(attribute=name,type=type(v).__module__+'.'+type(v).__name__))
    return dict(values=data,unserialized=unserialized)


def memory(ctl,env):
    import torch
    objects=dict(controller=ctl,agent=ctl.base,base_optimizer=ctl.base.optim,restore_optimizer=ctl.safe,
        reconnect=ctl.base.reconnect,recover_topology=ctl.base.recover_topo,
        topology_search=ctl.base.topo_n1_unsafe)
    data={k:flat_state(v) for k,v in objects.items()}
    if hasattr(env,'_oppSpace'):
        data['opponent_space']=flat_state(env._oppSpace)
        if hasattr(env._oppSpace,'opponent'):data['opponent']=flat_state(env._oppSpace.opponent)
    data['environment_rng']=packed(env.space_prng)
    global_rngs=dict(python=sha_bytes(pickle.dumps(random.getstate(),protocol=5)),
        numpy=sha_bytes(pickle.dumps(np.random.get_state(),protocol=5)),torch=digest(torch.get_rng_state().numpy()))
    # Unsupported engine/solver objects are inventoried, not claimed serialized.
    raw=json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    return dict(sha256=sha_bytes(raw),inventory=data,global_rngs_record_only=global_rngs,
        scope='Reviewed numeric controller/optimizer parameters, program values and explicitly seeded object RNG witnesses. Unused global RNGs recorded separately, not claimed identical. No opaque engine or solver clone. Full-prefix replay is the reconstruction method; solver uses warm_start=False.')


def feedback_signature(rows):
    keys=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible','phase']
    return [{k:r[k] for k in keys} for r in rows]


def install(Search,item):
    global ITEM,CASES
    ITEM=item;D=read(OUT/'design.json');CASES={c['id']:c for c in D['cases']}
    spec=importlib.util.spec_from_file_location('g95_adapter',ROOT/'work/grid91_mechanism/adapter.py')
    adapter=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter);adapter.install(Search,item)
    original_choose=Search.choose;original_run=adapter.P.run

    def run(order,observe,policy,subids,aliases,*args,**kwargs):
        global CURRENT
        if CURRENT is None:return original_run(order,observe,policy,subids,aliases,*args,**kwargs)
        prior=[]
        def wrapped(i,phase):
            if phase=='local_completion' and CURRENT['after_global'] is None:
                CURRENT['after_global']=memory(CURRENT['ctl'],CURRENT['env'])
                CURRENT['global_feedback']=feedback_signature(prior)
            row=observe(i,phase)
            if phase!='local_completion':prior.append(row)
            return row
        chosen_policy='NO_LOCAL' if CURRENT['skip'] else policy
        result=original_run(order,wrapped,chosen_policy,subids,aliases,*args,**kwargs)
        if CURRENT['after_global'] is None:
            CURRENT['after_global']=memory(CURRENT['ctl'],CURRENT['env'])
            CURRENT['global_feedback']=feedback_signature(prior)
        return result
    adapter.P.run=run

    def choose(self,obs,base,reward,done=False,**kwargs):
        global CURRENT,OMISSIONS
        step=int(obs.current_step)+1
        watch=next((CASES[i] for i in item['watch_cases'] if CASES[i]['step']==step),None)
        if watch is None:return original_choose(self,obs,base,reward,done,**kwargs)
        from common import action_vector
        assert CURRENT is None
        assert digest(obs.to_vect())==watch['state_hash'],'Target state differs from frozen original; stop before intervention'
        assert digest(action_vector(base))==watch['base_action_hash'],'Target public base action differs'
        skip=item.get('omit_case')==watch['id']
        CURRENT=dict(ctl=self.ctl,env=self.env,skip=skip,after_global=None,global_feedback=None)
        before=memory(self.ctl,self.env)
        try:
            chosen,log,features,prior=original_choose(self,obs,base,reward,done,**kwargs)
            expected=read(ROOT/watch['source'])
            expected_feedback=[r for r in expected['outcomes'] if r['phase']!='local_completion']
            assert CURRENT['global_feedback']==feedback_signature(expected_feedback)
            assert log['completion_budget']==watch['completion_budget']
            assert log['chosen_pool_id']==(watch['before_choice'] if skip else watch['after_choice'])
            if skip:assert log['local_queries']==0;OMISSIONS+=1
            snapshot=dict(case_id=watch['id'],step=step,skip=skip,state_hash=watch['state_hash'],
                base_action_hash=watch['base_action_hash'],before_search=before,after_global=CURRENT['after_global'],
                global_feedback=CURRENT['global_feedback'],chosen_pool_id=log['chosen_pool_id'],
                public_queries=log['public_queries'],local_queries=log['local_queries'],completion_budget=log['completion_budget'])
            path=OUT/'runs'/f"r0__{item['week']}__{item['method']}"/f"{watch['id']}_witness.json"
            assert not path.exists();save(path,snapshot)
            log['label_intervention']=dict(case_id=watch['id'],omitted_once=skip,continuation_policy='FRAME_GUARD')
            return chosen,log,features,prior
        finally:CURRENT=None
    Search.choose=choose


def step_signature(env,obs,info):
    fields={}
    for k in ['opponent_attack_line','opponent_attack_sub','opponent_attack_duration']:
        v=info.get(k);pv=packed(v)
        fields[k]=pv if pv is not NotImplemented else str(type(v))
    rho=np.asarray(obs.rho);valid=bool(np.isfinite(rho).all() and float(np.max(rho))>0)
    return dict(opponent_realization=fields,load_p_sha256=digest(obs.load_p),timestamp=str(obs.get_time_stamp()),
        max_rho=float(np.max(rho)) if valid else None,
        overloaded_lines=int(np.sum(rho>1)) if valid else None,
        summed_overload_excess=float(np.maximum(rho-1,0).sum()) if valid else None)


def assert_intervention_count(item):
    expected=int(item.get('omit_case') is not None)
    assert OMISSIONS==expected,(OMISSIONS,expected)
