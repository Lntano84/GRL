"""Decode recorded vectors with serialized spaces, without an environment or solve."""
from pathlib import Path
import collections,copy,datetime,gzip,hashlib,json,pickle,time,warnings
import numpy as np
from structure import compute,signature,NAMES
R=Path('D:/Codex/GRL研究');O=R/'outputs/grid110_structure_audit';W=Path(__file__).parent
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(v):return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
def av(a):return np.concatenate([a._get_array_from_attr_name(n).astype(np.float32) for n in type(a).attr_list_vect])
def save(n,x):(O/n).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,default=lambda a:a.item())+'\n',encoding='utf8')
FIELDS=['pool_id','rho','reward','done','illegal','ambiguous','exceptions','strict_admissible']
def main():
    assert not (O/'extraction_receipt.json').exists()
    meta=read(R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json')['samples']
    paths=[R/m['source_call'] for m in meta];paths=list(dict.fromkeys(paths))
    schema=R/'outputs/grid03/scoring_smoke/agent_episode'
    inputs=paths+[p.with_name(p.stem+'_state.npz') for p in paths]+[p.parent/'metadata.json' for p in paths]
    inputs+=[schema/'dict_observation_space.json',schema/'dict_action_space.json',R/'outputs/grid98_feature_enrichment/template_metadata.json',
        R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz',R/'outputs/grid28/offline.json',R/'outputs/grid102_search_rl/cases.pkl',
        R/'outputs/grid102_search_rl/normalization.npz',R/'outputs/grid100_data_foundation/bulk_dataset_manifest.json',
        R/'outputs/grid100_data_foundation/design.json',R/'outputs/grid100_data_foundation/training_input_contract.json',
        R/'outputs/grid109_input_coverage/positive_flip_details.json',R/'outputs/grid109_input_coverage/critical_coverage.json',
        R/'outputs/grid109_input_coverage/validation_coverage.json',R/'outputs/grid99_feature_learning/GRID99_report.md',
        R/'work/grid98_feature_enrichment/feature_core.py',R/'work/grid98_feature_enrichment/build.py']+list(W.glob('*.py'))
    frozen={p.relative_to(R).as_posix():sha(p) for p in dict.fromkeys(inputs)};save('input_manifest.json',frozen)
    save('plan.json',dict(id='GRID110_STRUCTURE_FEASIBILITY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Saved-data reconstruction and descriptive distinguishability only; no learner change, fit, inference or physical simulation.',
        historical='GRID98 already computed related full-completion descriptors on56 exposed branch cases; GRID99 did not improve trajectory-equivalence classification. Reuse arithmetic ideas, no new novelty or assumed improvement.',
        descriptors=NAMES,next_batch='Original next at most64, not whole remaining set; station stays locked as GRID102.',
        samples='ALL4438 train/validation reachable states, no state/date or descriptor selection by labels. Three March live probes separately.',
        reconstruction='Serialized observation/action spaces only, no environment construction. Verify vector roundtrip, grid mapping, public library, original masking/aliases and recorded combined-command hashes.',
        unknown='Any ambiguous reconnection without known terminal buses is explicitly unsupported, never guessed.',
        isolation='Only previously bought feedback chooses current incumbent. Structures use public commands/current observation. Delete/poison unbought feedback, bus-label swap and forbidden-key rejection checks.',
        comparison='Original41 RMS, structural11 RMS, all52 RMS; train-only date-balanced standardization of11, same clipped[-20,20] rule. Descriptive nearest1/5/20 distinct train physical calls, no fitted predictor, threshold or feature selection.',
        prohibitions=['no model training/inference','no new physical observations/forecasts/rollouts','no final-test contents','no feature selection by validation outcomes','no claim descriptive separation proves learned benefit']))
    warnings.filterwarnings('ignore',message='Numba cannot be loaded.*')
    import grid2op
    from grid2op.Environment import BaseEnv
    from grid2op.Observation import ObservationSpace,BaseObservation
    from grid2op.Action import ActionSpace
    calls=collections.Counter()
    def forbidden(*a,**kw):calls['forbidden_calls']+=1;raise RuntimeError('GRID110 forbids environment/physical methods')
    grid2op.make=BaseEnv.__init__=BaseEnv.step=BaseEnv.reset=BaseObservation.simulate=forbidden
    os=ObservationSpace.from_dict(read(schema/'dict_observation_space.json'));space=ActionSpace.from_dict(read(schema/'dict_action_space.json'))
    assert os.size()==4460 and space.size()==1605
    template=read(R/'outputs/grid98_feature_enrichment/template_metadata.json')
    for name in ['sub_info','line_or_to_subid','line_ex_to_subid','line_or_pos_topo_vect','line_ex_pos_topo_vect']:
        assert np.array_equal(getattr(space,name),template[name])
    save('decoded_schema.json',dict(observation_fields=list(os.attr_list_vect),observation_shapes=os.shape.tolist(),action_fields=list(space.attr_list_vect),action_shapes=space.shape.tolist(),
        topology='Public static endpoints/substation mapping equals GRID98 recorded template.',environment_constructed=False))
    with np.load(R/'work/grid02/ljn_nn/assets/action_N1_unsafe.npz') as z:library=[space.from_vect(v) for v in z['action_space']]
    subs=read(R/'outputs/grid28/offline.json')['public_subids'];libhash=[digest(av(a)) for a in library]
    assert libhash==template['library_hashes']
    for i,a in enumerate(library):assert int(a.as_dict()['set_bus_vect']['modif_subs_id'][0])==subs[i]
    C=pickle.loads((R/'outputs/grid102_search_rl/cases.pkl').read_bytes());cs={c['sample_id']:c for c in C};bycall=collections.defaultdict(list)
    for m in meta:bycall[m['source_call']].append(m)
    orp=space.line_or_pos_topo_vect;exp=space.line_ex_pos_topo_vect
    subpos=np.r_[0,np.cumsum(space.sub_info)];checks=collections.Counter();rows=[];new=[];provenance=[];unsupported=[]
    start=time.perf_counter()
    for no,(path,ms) in enumerate(bycall.items()):
        p=R/path;call=read(p);truth={r['pool_id']:r for r in call['outcomes']};md=read(p.parent/'metadata.json');assert md['library_hashes']==libhash
        assert sha(p)==ms[0]['source_sha256']
        with np.load(p.with_name(p.stem+'_state.npz')) as z:
            obs=os.from_vect(z['observation']);base=space.from_vect(z['base_action'])
            assert np.array_equal(obs.to_vect(),z['observation']) and np.array_equal(av(base),z['base_action'])
            assert digest(obs.to_vect())==call['state_hash'];checks['vector_roundtrips']+=2
        eligible=[i for i,s in enumerate(subs) if obs.time_before_cooldown_sub[s]==0];assert eligible==call['eligible']
        intent=(base.line_set_status>0)|base.line_change_status|(base.set_bus[orp]>0)|(base.set_bus[exp]>0)
        disconnected=(~obs.line_status)&(~intent);positions=np.r_[orp[disconnected],exp[disconnected]]
        reps=[];aliases={};actions={};first={};masked=[];plans={};unknown={}
        for i in eligible:
            a=library[i].copy();bus=a.set_bus.copy()
            if np.any(bus[positions]!=0):bus[positions]=0;a.set_bus=bus;masked.append(i)
            combined=a+base;key=av(combined).tobytes()
            if key not in first:
                first[key]=i;reps.append(i);actions[i]=combined
                assert digest(av(combined))==truth[i]['combined_hash'];checks['combined_action_hashes']+=1
                plan=obs.topo_vect.copy();commands=combined.set_bus;plan[commands!=0]=commands[commands!=0];switch=combined.change_bus
                assert not np.any(switch&(commands!=0));live_switch=switch&(plan>0);plan[live_switch]=3-plan[live_switch]
                force_off=(combined.line_set_status<0)|(combined.line_change_status&obs.line_status)
                force_on=(combined.line_set_status>0)|(combined.line_change_status&~obs.line_status)
                plan[orp[force_off]]=-1;plan[exp[force_off]]=-1
                unknown[i]=set(np.r_[orp[force_on&(plan[orp]<=0)],exp[force_on&(plan[exp]<=0)]].tolist());plans[i]=plan
            aliases[i]=first[key]
        assert reps==call['representatives'] and aliases=={int(k):v for k,v in call['aliases'].items()} and masked==call['masked_ids'];checks['public_preparations']+=1
        for m in ms:
            c=cs[m['sample_id']];n=m['visible_count'];visible=[{k:truth[i][k] for k in FIELDS} for i in call['order'][:n]]
            best=lambda vs:max((v for v in vs if v['strict_admissible']),key=lambda v:(float(np.float32(v['reward'])),-v['pool_id']))
            anchor=best(visible)['pool_id'];station=subs[anchor];target={int(v) for k,v in aliases.items() if subs[k]==station};remaining=[i for i in call['order'][n:] if i in target]
            pos=np.arange(subpos[station],subpos[station+1]);original=remaining.copy()
            for j in range(len(c['states'])):
                incumbent=best(visible)['pool_id'];assert incumbent==c['chosen'][j]
                nxt=remaining[:64];assert len(nxt)==int(c['queries'][j])
                seen=[v['pool_id'] for v in visible if v['pool_id'] in target]
                involved=set(seen)|set(nxt)
                if any(unknown[i]&set(pos) for i in involved):
                    unsupported.append(dict(sample_id=m['sample_id'],position=j,reason='reconnection terminal bus not reconstructible from public command/current topology'))
                    raise RuntimeError('Unsupported topology explicitly retained; no fabricated features')
                context=dict(plans={i:plans[i][pos].astype(int).tolist() for i in involved},incumbent_id=incumbent,seen_station_ids=seen,next_ids=nxt,
                    current_station_topology=obs.topo_vect[pos].astype(int).tolist(),rho=obs.rho.astype(float).tolist(),line_status=obs.line_status.tolist(),
                    incident_lines=((space.line_or_to_subid==station)|(space.line_ex_to_subid==station)).tolist())
                x=compute(context)
                # Suffix feedback cannot enter the feature API; changing/deleting it leaves visible records and context unchanged.
                visibleids={v['pool_id'] for v in visible}
                stripped={i:r for i,r in truth.items() if i in visibleids}
                poisoned={i:({**r,'reward':9999.,'rho':.001,'strict_admissible':True} if i not in visibleids else r) for i,r in truth.items()}
                ids=[v['pool_id'] for v in visible]
                assert [{k:stripped[i][k] for k in FIELDS} for i in ids]==visible
                assert [{k:poisoned[i][k] for k in FIELDS} for i in ids]==visible
                assert np.array_equal(x,compute(copy.deepcopy(context)));checks['suffix_isolation_states']+=1
                swapped=copy.deepcopy(context);swapped['plans']={i:[3-v if v>0 else v for v in t] for i,t in context['plans'].items()}
                swapped['current_station_topology']=[3-v if v>0 else v for v in context['current_station_topology']]
                assert np.array_equal(x,compute(swapped));checks['bus_label_invariance']+=1
                try:compute(dict(context,future_gain=999.))
                except AssertionError:checks['forbidden_key_rejections']+=1
                else:raise AssertionError('Forbidden input accepted')
                rows.append(dict(sample_id=m['sample_id'],position=j,partition=c['partition'],date=c['date'],scenario=c['scenario'],physical_state_group=c['physical_state_group'],step=m['step'],source_call=path,
                    state_hash=call['state_hash'],station_for_audit_only=station,incumbent_id_for_audit_only=incumbent,next_ids_for_audit_only=nxt,seen_station_ids_for_audit_only=seen))
                new.append(x)
                visible.extend({k:truth[i][k] for k in FIELDS} for i in nxt);remaining=remaining[len(nxt):]
            assert not remaining
        provenance.append(dict(source_call=path,roundtrip=True,aliases=True,combined_hashes=len(reps),unknown_reconnection_candidates=sum(bool(v) for v in unknown.values())))
        if no%40==0:print('decoded',no+1,'/',len(bycall),'calls;',len(rows),'reachable states',flush=True)
    assert len(rows)==4438 and len(provenance)==278 and not unsupported and calls['forbidden_calls']==0
    Z=np.stack(new);np.savez_compressed(O/'diagnostic_descriptors.npz',X=Z);save('rows.json',rows);save('provenance.json',provenance)
    save('extraction_receipt.json',dict(passed=True,checks=dict(checks),physical_calls=len(provenance),reachable_states=len(rows),descriptor_count=11,
        unsupported_states=unsupported,environment_constructions=0,new_forecasts=0,new_physical_steps=0,new_fits=0,new_predictions=0,learner_inputs_changed=False,
        physical_methods_forbidden=True,elapsed_s=time.perf_counter()-start))
    assert all(sha(R/p)==h for p,h in frozen.items())
    print('EXTRACTED',len(rows),'no environment/simulation/model',flush=True)
if __name__=='__main__':main()
