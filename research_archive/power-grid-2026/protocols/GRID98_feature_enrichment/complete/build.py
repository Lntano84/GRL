"""Decode saved vectors, redact completion feedback, add fixed features. No fits."""
from pathlib import Path
import json,hashlib,sys,importlib.util,time,copy
import numpy as np
from feature_core import compute,FEATURES
ROOT=Path(__file__).resolve().parents[2];W=Path(__file__).parent;OUT=ROOT/'outputs/grid98_feature_enrichment'
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,d):
    p=OUT/name;p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def check_freeze():
    f=read(OUT/'freeze.json')
    for n,h in f['sha256'].items():assert sha(ROOT/n)==h,n
    for n,h in read(ROOT/'outputs/grid97_effect_learning/freeze.json')['sha256'].items():assert sha(ROOT/n)==h,n
def load_module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
OLD=load_module('grid98_oldfeatures',ROOT/'work/grid94_predictability/data.py')
FIELDS=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
def redact(call,public):
    # No access to after_choice, costs, outcomes of omitted branches or case labels.
    aliases={int(k):v for k,v in call['aliases'].items()}
    x,no=OLD.features(call['order'],call['outcomes'],aliases,public['subids'],call['rho_limit'])
    rows=[{k:r[k] for k in FIELDS} for r in no['outcomes']];best=no['chosen_pool_id'];station=public['subids'][best]
    targets=sorted({int(v) for k,v in aliases.items() if public['subids'][int(k)]==station})
    context=dict(order=call['order'],aliases={str(k):v for k,v in aliases.items()},subids=public['subids'],visible=rows,best_id=best,target_ids=targets,
        target_plans={str(i):public['plans'][i] for i in targets},current_station_topology=public['topology'],current_rho=public['rho'],current_line_status=public['line_status'],incident_lines=public['incident_lines'],rho_limit=call['rho_limit'])
    return context,x

def main():
    check_freeze();assert not (OUT/'features.json').exists(),'preserve existing extraction'
    sys.path.insert(0,str(ROOT/'work/grid08'));import common
    from grid2op.Environment import BaseEnv
    from grid2op.Observation import BaseObservation
    env=common.make_env();obs0=env.get_obs()
    def forbidden(*a,**k):raise RuntimeError('GRID98 forbids physical step/reset/simulate')
    original_step,original_reset,original_simulate=BaseEnv.step,BaseEnv.reset,BaseObservation.simulate
    BaseEnv.step=BaseEnv.reset=BaseObservation.simulate=forbidden
    initial_queries=int(env.nb_highres_called);assert initial_queries==0
    subs=read(ROOT/'outputs/grid28/offline.json')['public_subids']
    with np.load(ROOT/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz') as z:vectors=z['action_space'].copy()
    library=[]
    for i,v in enumerate(vectors):
        a=env.action_space();a.from_vect(v);assert int(a.as_dict()['set_bus_vect']['modif_subs_id'][0])==subs[i];library.append(a)
    libhash=[common.digest(common.action_vector(a)) for a in library]
    cases=read(ROOT/'outputs/grid96_value_learning/design.json')['cases'];old={r['case_id']:r for r in read(ROOT/'outputs/grid97_effect_learning/dataset.json')}
    features=[];provenance=[];times=[];checks={'source_cases':0,'original15_exact':0,'masked_alias_reconstructions':0,'visible_hash_checks':0,'hidden_deleted':0,'hidden_poisoned':0,'bus_swap_checks':0,'forbidden_input_rejections':0}
    try:
        for case in cases:
            t=time.perf_counter();p=ROOT/case['source'];call=read(p);meta=read(p.parent/'metadata.json');assert meta['library_hashes']==libhash
            with np.load(p.with_name(p.stem+'_state.npz')) as z:
                obs=obs0.copy();obs.from_vect(z['observation']);base=env.action_space();base.from_vect(z['base_action'])
            assert common.digest(obs.to_vect())==case['state_hash']==call['state_hash']
            assert common.digest(common.action_vector(base))==case['base_action_hash']
            eligible=[i for i,s in enumerate(subs) if obs.time_before_cooldown_sub[s]==0];assert eligible==call['eligible']
            intent=(base.line_set_status>0)|base.line_change_status|(base.set_bus[env.line_or_pos_topo_vect]>0)|(base.set_bus[env.line_ex_pos_topo_vect]>0)
            disconnected=(~obs.line_status)&(~intent);pos=np.concatenate([env.line_or_pos_topo_vect[disconnected],env.line_ex_pos_topo_vect[disconnected]])
            actions={};aliases={};first={};reps=[];masked=[]
            for i in eligible:
                a=library[i].copy();b=a.set_bus.copy()
                if np.any(b[pos]!=0):b[pos]=0;a.set_bus=b;masked.append(i)
                v=common.action_vector(a+base);h=common.digest(v)
                if h not in first:first[h]=i;reps.append(i);actions[i]=a
                aliases[i]=first[h]
            assert reps==call['representatives'] and aliases=={int(k):v for k,v in call['aliases'].items()} and masked==call['masked_ids']
            checks['masked_alias_reconstructions']+=1
            # Obtain the anchor only from public prefix feedback, not saved post-completion IDs.
            oldx,no=OLD.features(call['order'],call['outcomes'],aliases,subs,call['rho_limit'])
            assert oldx==old[case['id']]['features']==case['features'];checks['original15_exact']+=1
            station=subs[no['chosen_pool_id']];start=int(np.sum(env.sub_info[:station]));positions=np.arange(start,start+int(env.sub_info[station]));assert np.all(env.topo_vect_to_sub[positions]==station) if hasattr(env,'topo_vect_to_sub') else True
            plans={}
            for i in reps:
                combined=actions[i]+base
                plan=obs.topo_vect.copy();commands=combined.set_bus;plan[commands!=0]=commands[commands!=0]
                switch=combined.change_bus
                assert not np.any(switch&(commands!=0)),'ambiguous bus commands'
                live_switch=switch&(plan>0);plan[live_switch]=3-plan[live_switch]
                force_off=(combined.line_set_status<0)|(combined.line_change_status&obs.line_status)
                plan[env.line_or_pos_topo_vect[force_off]]=-1;plan[env.line_ex_pos_topo_vect[force_off]]=-1
                plans[i]=plan[positions].astype(int).tolist()
            public=dict(subids=subs,plans=plans,topology=obs.topo_vect[positions].astype(int).tolist(),rho=obs.rho.astype(float).tolist(),line_status=obs.line_status.astype(bool).tolist(),incident_lines=((env.line_or_to_subid==station)|(env.line_ex_to_subid==station)).tolist())
            context,x0=redact(call,public);t0=time.perf_counter();new=compute(context);feature_s=time.perf_counter()-t0
            assert x0==oldx
            seen={r['pool_id'] for r in context['visible']};hidden=set(call['order'])-seen
            assert [r['pool_id'] for r in context['visible']]==[r['pool_id'] for r in call['outcomes'] if r['phase']!='local_completion']
            for r in context['visible']:
                original=next(z for z in call['outcomes'] if z['pool_id']==r['pool_id']);assert original['combined_hash']==common.digest(common.action_vector(actions[r['pool_id']]+base));checks['visible_hash_checks']+=1
            stripped=copy.deepcopy(call);stripped['outcomes']=[r for r in call['outcomes'] if r['pool_id'] in seen]
            stripped_context,xx=redact(stripped,public);assert stripped_context==context and compute(stripped_context)==new and xx==oldx;checks['hidden_deleted']+=1
            poisoned=copy.deepcopy(call)
            for r in poisoned['outcomes']:
                if r['pool_id'] not in seen:r.update(reward=9999.,rho=.001,strict_admissible=True,done=False,illegal=False,ambiguous=False,exceptions=[])
            pc,xx=redact(poisoned,public);assert pc==context and compute(pc)==new and xx==oldx;checks['hidden_poisoned']+=1
            swapped=copy.deepcopy(context)
            swapped['target_plans']={k:[3-v if v>0 else v for v in values] for k,values in context['target_plans'].items()}
            swapped['current_station_topology']=[3-v if v>0 else v for v in context['current_station_topology']]
            assert compute(swapped)==new;checks['bus_swap_checks']+=1
            try:compute(dict(context,after_choice=case['after_choice']))
            except AssertionError:checks['forbidden_input_rejections']+=1
            else:raise AssertionError('future input unexpectedly accepted')
            save('contexts/'+case['id']+'.json',context)
            features.append(dict(case_id=case['id'],original15=oldx,added20=new,all35=dict(oldx,**new)))
            provenance.append(dict(case_id=case['id'],source=case['source'],source_sha256=sha(p),state_sha256=sha(p.with_name(p.stem+'_state.npz')),state_hash=case['state_hash'],context_sha256=sha(OUT/'contexts'/(case['id']+'.json')),station=station,visible_ids=[r['pool_id'] for r in context['visible']],remaining_ids=[i for i in call['order'] if i in context['target_ids'] and i not in seen],post_completion_feedback_rows_read_by_extractor=0))
            times.append(dict(case_id=case['id'],pure_feature_seconds=feature_s,offline_reconstruction_and_checks_seconds=time.perf_counter()-t))
            checks['source_cases']+=1;print(case['id'],'features and leakage checks passed',flush=True)
        assert int(env.nb_highres_called)==initial_queries
        template=dict(n_sub=env.n_sub,n_line=env.n_line,dim_topo=env.dim_topo,sub_info=env.sub_info.tolist(),line_or_to_subid=env.line_or_to_subid.tolist(),line_ex_to_subid=env.line_ex_to_subid.tolist(),line_or_pos_topo_vect=env.line_or_pos_topo_vect.tolist(),line_ex_pos_topo_vect=env.line_ex_pos_topo_vect.tolist(),library_hashes=libhash,template_forecast_counter_before=initial_queries,template_forecast_counter_after=int(env.nb_highres_called),step_calls=0,reset_calls=0,simulate_calls=0)
        save('template_metadata.json',template);save('features.json',features);save('provenance.json',provenance);save('extraction_timing.json',times);save('build_checks.json',dict(passed=True,checks=checks,new_forecasts=0,new_replays=0,new_fits=0,new_predictions=0))
        save('labels_and_groups_unchanged.json',[{k:v for k,v in r.items() if k!='features'} for r in old.values()])
        save('original_folds_unchanged.json',read(ROOT/'outputs/grid97_effect_learning/design.json')['folds']);check_freeze()
    finally:
        BaseEnv.step,BaseEnv.reset,BaseObservation.simulate=original_step,original_reset,original_simulate;env.close()
if __name__=='__main__':main()
