"""Omit completion exactly once, then restore unchanged FRAME; reuse GRID95 witnesses."""
import copy,gzip,json,re
from common106 import ROOT,OUT,read,save,digest,loadmod
H=loadmod('grid106_old_witness','work/grid95_labels/label_hooks.py')
CURRENT=None;COUNT=0;ITEM=None;D=None;OLDROWS=None
EXCLUDED=['base_optimizer','restore_optimizer','reconnect','recover_topology','topology_search']
def normalized_memory(m):
    data=copy.deepcopy(m['inventory'])
    for k in EXCLUDED:data[k]['values'].pop('space_prng',None)
    return data
def equal_memory(a,b):return normalized_memory(a)==normalized_memory(b)
def install(Search,item):
    global ITEM,D,OLDROWS
    ITEM=item;D=read(OUT/'design.json')
    with gzip.open(ROOT/D['reference']/'steps.jsonl.gz','rt',encoding='utf8') as f:OLDROWS=[json.loads(x) for x in f]
    adapter=loadmod('grid106_adapter104','work/grid104_learning_closedloop/adapter.py')
    adapter.install(Search,{**item,'method':'FRAME_GUARD'})
    original_choose=Search.choose;original_run=adapter.P.G.run
    def run(order,reveal,policy,subids,aliases,*args,**kwargs):
        if CURRENT is None:return original_run(order,reveal,policy,subids,aliases,*args,**kwargs)
        prior=[]
        def wrapped(i,phase):
            if phase=='local_completion' and CURRENT['after_global'] is None:
                CURRENT['after_global']=H.memory(CURRENT['ctl'],CURRENT['env'])
                CURRENT['global_feedback']=H.feedback_signature(prior)
            r=reveal(i,phase)
            if phase!='local_completion':prior.append(r)
            return r
        result=original_run(order,wrapped,'NO_LOCAL' if CURRENT['skip'] else policy,subids,aliases,*args,**kwargs)
        if CURRENT['after_global'] is None:
            CURRENT['after_global']=H.memory(CURRENT['ctl'],CURRENT['env'])
            CURRENT['global_feedback']=H.feedback_signature(prior)
        return result
    adapter.P.G.run=run
    def choose(self,obs,base,reward,done=False,**kwargs):
        global CURRENT,COUNT
        step=int(obs.current_step)+1
        cases=[c for c in D['cases'] if c['id'] in item['watch_cases'] and c['step']==step]
        if not cases:return original_choose(self,obs,base,reward,done,**kwargs)
        assert len(cases)==1 and CURRENT is None
        c=cases[0];skip=item['omit_case']==c['id']
        from common import action_vector
        assert digest(obs.to_vect())==c['state_hash'] and digest(action_vector(base))==c['base_action_hash']
        CURRENT=dict(ctl=self.ctl,env=self.env,skip=skip,after_global=None,global_feedback=None)
        before=H.memory(self.ctl,self.env)
        try:
            selected,log,features,prior=original_choose(self,obs,base,reward,done,**kwargs)
            ref=read(ROOT/c['source'])
            assert CURRENT['global_feedback']==H.feedback_signature([r for r in ref['outcomes'] if r['phase']!='local_completion'])
            assert log['completion_budget']==c['completion_budget']
            assert log['chosen_pool_id']==(c['before_choice'] if skip else c['after_choice'])
            assert log['local_queries']==(0 if skip else c['completion_budget'])
            witness=dict(case_id=c['id'],step=step,skip=skip,state_hash=c['state_hash'],base_action_hash=c['base_action_hash'],
                before_search=before,after_global=CURRENT['after_global'],global_feedback=CURRENT['global_feedback'],
                chosen_pool_id=log['chosen_pool_id'],public_queries=log['public_queries'],local_queries=log['local_queries'])
            if skip:
                q=OUT/'runs'/f"r0__{item['week']}__FRAME_QUAL"/f"{c['id']}_witness.json"
                known=read(q)
                assert equal_memory(known['before_search'],before),'Pre-search numeric/used RNG witness mismatch'
                assert equal_memory(known['after_global'],witness['after_global']),'After-global witness mismatch'
                COUNT+=1
            folder=OUT/'runs'/f"r0__{item['week']}__{item['method']}"
            assert not (folder/f"{c['id']}_witness.json").exists()
            save(folder/f"{c['id']}_witness.json",witness)
            log['single_intervention']=dict(case_id=c['id'],omitted_once=skip,continuation='Unchanged FRAME_GUARD')
            return selected,log,features,prior
        finally:CURRENT=None
    Search.choose=choose
def on_step(row,env,obs,info):
    row['label_audit']=H.step_signature(env,obs,info)
    target=next((c['step'] for c in D['cases'] if c['id']==ITEM['omit_case']),None)
    if target is None or row['step']<target:
        ref=OLDROWS[row['step']-1]
        for k in ['step','before_hash','action_hash','after_hash','raw_cost','done','complete','simulations']:
            assert row[k]==ref[k],(row['step'],k,'Historical prefix not reproduced')
    elif row['step']==target:assert row['before_hash']==OLDROWS[target-1]['before_hash']
def finished(item):assert COUNT==int(item['omit_case'] is not None),(COUNT,item['omit_case'])
